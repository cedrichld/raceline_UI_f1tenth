#!/usr/bin/env python3
"""
Raceline Studio — Web-based raceline management for F1Tenth racing.

Wraps the existing raceline_optimizer.py as a backend, serves a modern UI
for managing up to 4 racelines (primary, pass-left, pass-right, fallback),
generating offset variants, tuning velocity profiles, and exporting CSVs.

Usage:
    pip install flask
    cd <this directory>
    python3 app.py
    → opens http://localhost:5050
"""
import os, sys, json, glob, yaml, hashlib, datetime, threading, time, webbrowser
import numpy as np
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file

import rclpy
from rclpy.node import Node
from raceline_msgs.srv import UpdateRaceline

# ═══════════════════════════════════════════════════════════════════════════
# WORKSPACE PATHS — edit these to match your setup
# ═══════════════════════════════════════════════════════════════════════════
PURE_PURSUIT_WAYPOINTS = "~/roboracer_ws/src/labs/lab5/pure_pursuit/waypoints"
MPPI_WAYPOINTS         = "~/roboracer_ws/src/cbf_mppi_f1tenth/mppi_bringup/waypoints"

MAPS_DIRS = ["~/roboracer_ws/src/f1tenth_gym_ros/maps"]

# Map stem (no extension) that gets floated to the top of the dropdown
DEFAULT_MAP = "racetrack_levine_cleaned"

# MPPI export endpoint writes here (under the MPPI waypoints root)
MPPI_EXPORT_SUBDIR = "lev_testing"
# ═══════════════════════════════════════════════════════════════════════════

sys.path.insert(0, str(Path(__file__).resolve().parent))
import raceline_optimizer as ropt

app = Flask(__name__)

WP_ROOTS = {
    "pure_pursuit": Path(os.path.expanduser(PURE_PURSUIT_WAYPOINTS)),
    "mppi":         Path(os.path.expanduser(MPPI_WAYPOINTS)),
}
MPPI_WP_DIR = WP_ROOTS["mppi"] / MPPI_EXPORT_SUBDIR
MAPS_DIRS = [Path(os.path.expanduser(d)) for d in MAPS_DIRS]


# ── ROS2 node with service clients ────────────────────────────────────────
class StudioNode(Node):
    def __init__(self):
        super().__init__('raceline_studio')
        self.pp_client   = self.create_client(UpdateRaceline, '/pure_pursuit/update_raceline')
        self.mppi_client = self.create_client(UpdateRaceline, '/mppi/update_raceline')

STUDIO_NODE: "StudioNode | None" = None


def _resolve(rel_path):
    """Map a sidebar path like 'pure_pursuit/race/foo.csv' to its absolute file."""
    parts = rel_path.split("/", 1)
    if len(parts) < 2 or parts[0] not in WP_ROOTS:
        raise ValueError(f"Path must start with one of {list(WP_ROOTS)}: {rel_path!r}")
    return WP_ROOTS[parts[0]] / parts[1]

# ── Helpers ─────────────────────────────────────────────────────────────────
def _find_maps():
    """Return list of {name, yaml_path, img_path} for all available track maps.
    Handles both PNG and PGM image formats (PGM converted on-the-fly via /api/map-image)."""
    maps = []
    for d in MAPS_DIRS:
        if not d.exists():
            continue
        for y in sorted(d.glob("*.yaml")):
            meta = yaml.safe_load(y.read_text())
            if "image" not in meta:
                continue
            img = d / meta["image"]
            if not img.exists():
                # Try alternate extensions (pgm ↔ png)
                for ext in ['.pgm', '.png', '.bmp']:
                    alt = img.with_suffix(ext)
                    if alt.exists():
                        img = alt
                        break
            if not img.exists():
                continue
            maps.append({"name": f"{d.name}/{y.stem}", "yaml": str(y), "image": str(img),
                         "resolution": meta.get("resolution", 0.05),
                         "origin": meta.get("origin", [0, 0, 0])})
    maps.sort(key=lambda m: (not m["name"].endswith("/" + DEFAULT_MAP), m["name"]))
    return maps

def _find_csvs():
    """List CSVs from all waypoint roots, prefixed with their root name
    (e.g. 'pure_pursuit/race/foo.csv', 'mppi/mppi/sim/bar.csv')."""
    csvs = []
    for prefix, root in WP_ROOTS.items():
        if not root.exists():
            continue
        for f in sorted(root.rglob("*.csv")):
            csvs.append(f"{prefix}/{f.relative_to(root)}")
    return sorted(csvs)

def _load_csv(rel_path):
    """Load waypoints CSV → list of [x, y, yaw, speed_ratio].

    'mppi/...' paths are Levine 9-col (';' delimited, 3 header lines): extract
    [x_m, y_m, psi_rad, vx_mps] and rescale vx to a 0-1 ratio using the file's
    own min/max so the colormap spans red→cyan regardless of absolute speed.
    Everything else uses the standard pure-pursuit 4-col loader."""
    full = _resolve(rel_path)
    if rel_path.startswith("mppi/"):
        wps9 = np.loadtxt(str(full), delimiter=';', skiprows=3)
        xy_yaw_v = wps9[:, [1, 2, 3, 5]]   # x_m, y_m, psi_rad, vx_mps
        vx = xy_yaw_v[:, 3]
        vmin, vmax = float(vx.min()), float(vx.max())
        span = max(vmax - vmin, 1e-6)
        xy_yaw_v[:, 3] = np.clip((vx - vmin) / span, 0.0, 1.0)
        return xy_yaw_v.tolist()
    return ropt.load_waypoints(str(full)).tolist()

def _save_csv(rel_path, data):
    """Save [[x,y,yaw,speed], ...] to CSV."""
    full = _resolve(rel_path)
    full.parent.mkdir(parents=True, exist_ok=True)
    wps = np.array(data)
    ropt.save_waypoints(wps, str(full))


# ── MPPI export (Levine 9-column format) ───────────────────────────────────
def _signed_curvature(xy):
    """Signed Menger curvature on a closed loop. Sign convention matches
    cubic_spline._calc_kappa_from_xy: positive = CCW (left turn) in ENU."""
    n = len(xy)
    k = np.zeros(n)
    for i in range(n):
        p0 = xy[(i - 1) % n]
        p1 = xy[i]
        p2 = xy[(i + 1) % n]
        d1 = p1 - p0
        d2 = p2 - p1
        cross = d1[0] * d2[1] - d1[1] * d2[0]   # signed
        a = np.linalg.norm(d1)
        b = np.linalg.norm(d2)
        c = np.linalg.norm(p2 - p0)
        denom = a * b * c
        if denom > 1e-10:
            k[i] = 2.0 * cross / denom
    return k


def _to_mppi_rows(wps, v_min, v_max, w_tr_const=1.0):
    """Convert [x, y, yaw, speed_ratio] (closed loop, N rows) into the
    Levine 9-column array: s_m, x_m, y_m, psi_rad, kappa_radpm, vx_mps,
    ax_mps2, w_tr_right_m, w_tr_left_m. All purely geometric/kinematic."""
    wps = np.asarray(wps, dtype=float)
    n = len(wps)
    xy = wps[:, :2]
    psi = wps[:, 2]
    ratio = np.clip(wps[:, 3], 0.0, 1.0)

    # Segment lengths (closed loop)
    diffs = np.roll(xy, -1, axis=0) - xy
    ds = np.linalg.norm(diffs, axis=1)

    # s_m: cumulative arc length, starting at 0
    s = np.concatenate([[0.0], np.cumsum(ds[:-1])])

    # kappa: signed Menger
    kappa = _signed_curvature(xy)

    # vx: invert profile_velocity normalization (raceline_optimizer.py:224)
    vx = v_min + ratio * (v_max - v_min)

    # ax: forward-difference of v² / (2·ds), wrap last entry from segment 0
    ds_safe = np.maximum(ds, 1e-6)
    ax = (np.roll(vx, -1) ** 2 - vx ** 2) / (2.0 * ds_safe)

    w_tr_right = np.full(n, float(w_tr_const))
    w_tr_left  = np.full(n, float(w_tr_const))

    rows = np.column_stack([s, xy[:, 0], xy[:, 1], psi, kappa, vx, ax, w_tr_right, w_tr_left])
    return rows


def _save_mppi_csv(out_path, rows):
    """Write 3-line header + ';'-delimited body, %.18e precision (matches
    levine.csv exactly so Track.from_numpy parses it identically)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
        "# " + hashlib.md5(rows.tobytes()).hexdigest() + "\n"
        "# s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2; w_tr_right_m; w_tr_left_m\n"
    )
    with open(out_path, "w") as f:
        f.write(header)
        np.savetxt(f, rows, delimiter=";", fmt="%.18e")

def _offset_raceline(base_wps, offset_m, map_yaml=None, safety_margin=0.15):
    """Generate a raceline offset perpendicular to the base line.
    offset_m > 0 → left (positive y in ROS), < 0 → right.
    If map_yaml provided, clamps points to stay within track walls (with safety_margin).
    Returns np array of [x, y, yaw, speed]."""
    xy = base_wps[:, :2]
    n = len(xy)

    # Tangent vectors via central differences (wrapped)
    tangents = np.zeros_like(xy)
    for i in range(n):
        tangents[i] = xy[(i + 1) % n] - xy[(i - 1) % n]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    tangents /= norms

    # Normal = rotate tangent 90° CCW
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])

    # Compute per-point feasible offset using distance transform
    if map_yaml:
        try:
            from scipy.ndimage import distance_transform_edt
            img, resolution, origin = ropt.load_map(map_yaml)
            free = (img > 250).astype(np.uint8)
            dt = distance_transform_edt(free) * resolution

            # For each point, find max offset that keeps it safety_margin away from walls
            feasible = np.full(n, abs(offset_m))
            for i in range(n):
                # Sample along the normal from base to full offset, find where dt drops below margin
                for frac in np.linspace(1.0, 0.0, 30):
                    trial = xy[i] + frac * offset_m * normals[i]
                    r, c = ropt.world_to_pixel(trial[0], trial[1], resolution, origin, img.shape)
                    r, c = int(round(r)), int(round(c))
                    if 0 <= r < dt.shape[0] and 0 <= c < dt.shape[1] and dt[r, c] >= safety_margin:
                        feasible[i] = abs(frac * offset_m)
                        break
                else:
                    feasible[i] = 0.0

            # Smooth the feasible offsets to avoid jagged transitions
            from scipy.ndimage import uniform_filter1d
            feasible = uniform_filter1d(feasible, size=max(3, n // 15), mode='wrap')

            sign = 1.0 if offset_m > 0 else -1.0
            capped = np.minimum(np.full(n, abs(offset_m)), feasible)
            new_xy = xy + sign * capped[:, None] * normals
        except Exception as e:
            print(f"[offset] Wall clamp failed: {e}")
            new_xy = xy + offset_m * normals
    else:
        new_xy = xy + offset_m * normals

    # Final spline smooth
    try:
        from scipy.interpolate import splprep, splev
        tck, _ = splprep([new_xy[:, 0], new_xy[:, 1]], s=0.5, per=True, k=3)
        u = np.linspace(0, 1, n, endpoint=False)
        sx, sy = splev(u, tck)
        new_xy = np.column_stack([sx, sy])
    except Exception:
        pass

    # Recompute yaw
    dx = np.diff(new_xy[:, 0], append=new_xy[0, 0])
    dy = np.diff(new_xy[:, 1], append=new_xy[0, 1])
    yaw = np.arctan2(dy, dx)

    # Keep base speed profile
    result = np.column_stack([new_xy, yaw, base_wps[:, 3]])
    return result


# ── API Routes ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/maps")
def api_maps():
    return jsonify(_find_maps())

@app.route("/api/map-image")
def api_map_image():
    path = request.args.get("path", "")
    if not os.path.isfile(path):
        return "not found", 404
    # PGM files aren't browser-native — convert to PNG on the fly
    if path.lower().endswith('.pgm'):
        from PIL import Image
        import io
        img = Image.open(path)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    return send_file(path, mimetype="image/png")

@app.route("/api/csvs")
def api_csvs():
    return jsonify(_find_csvs())

@app.route("/api/csv")
def api_csv_load():
    rel = request.args.get("path", "")
    try:
        return jsonify(_load_csv(rel))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/csv", methods=["POST"])
def api_csv_save():
    body = request.json
    rel = body.get("path", "")
    data = body.get("data", [])
    try:
        _save_csv(rel, data)
        return jsonify({"ok": True, "path": rel})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/offset", methods=["POST"])
def api_offset():
    body = request.json
    base = np.array(body["base"])  # [[x,y,yaw,speed], ...]
    offset_m = float(body.get("offset", 0.3))
    map_yaml = body.get("map_yaml")
    safety_margin = float(body.get("safety_margin", 0.15))
    result = _offset_raceline(base, offset_m, map_yaml=map_yaml, safety_margin=safety_margin)
    return jsonify(result.tolist())

@app.route("/api/profile", methods=["POST"])
def api_profile():
    body = request.json
    wps = np.array(body["waypoints"])
    params = body.get("params", {})
    profiled = ropt.profile_velocity(
        wps,
        v_max=params.get("v_max", 5.0),
        v_min=params.get("v_min", 2.0),
        a_max=params.get("a_max", 5.0),
        a_brake=params.get("a_brake", 8.0),
        mu=params.get("mu", 0.34),
    )
    return jsonify(profiled.tolist())

@app.route("/api/estimate-laptime", methods=["POST"])
def api_estimate_laptime():
    body = request.json
    wps = np.array(body["waypoints"])
    v_min = body.get("v_min")
    v_max = body.get("v_max")
    try:
        t = ropt.estimate_lap_time(wps, v_min=v_min, v_max=v_max)
        return jsonify({"lap_time": round(t, 3)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/extract-centerline", methods=["POST"])
def api_extract_centerline():
    body = request.json
    yaml_path = body["map_yaml"]
    num_points = body.get("num_points", 150)
    margin = body.get("margin", 0.20)
    try:
        img, resolution, origin = ropt.load_map(yaml_path)
        result = ropt.extract_centerline(img, resolution, origin,
                                         num_points=num_points, margin=margin)
        # extract_centerline returns (waypoints, track_widths) tuple
        wps = result[0] if isinstance(result, tuple) else result
        return jsonify(wps.tolist())
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400

@app.route("/api/reverse", methods=["POST"])
def api_reverse():
    """Reverse waypoint order and flip yaw by pi."""
    body = request.json
    wps = np.array(body["waypoints"])
    wps = wps[::-1].copy()
    # Flip yaw by pi
    wps[:, 2] = np.arctan2(np.sin(wps[:, 2] + np.pi), np.cos(wps[:, 2] + np.pi))
    return jsonify(wps.tolist())

@app.route("/api/resample", methods=["POST"])
def api_resample():
    """Resample waypoints with curvature-biased point distribution."""
    body = request.json
    wps = np.array(body["waypoints"])
    num_points = body.get("num_points", len(wps))
    turn_bias = body.get("turn_bias", 2.0)
    try:
        from scipy.interpolate import splprep, splev
        xy = wps[:, :2]
        # Tight spline fit — preserve original trajectory shape, just redistribute points
        # Try interpolating (s=0) first, fall back to small s if it fails
        try:
            tck, _ = splprep([xy[:, 0], xy[:, 1]], s=0, per=True, k=3)
        except Exception:
            tck, _ = splprep([xy[:, 0], xy[:, 1]], s=0.001, per=True, k=3)
        # Evaluate at many uniform points to get clean curvature estimate
        u_dense = np.linspace(0, 1, max(300, len(xy) * 4), endpoint=False)
        x_d, y_d = splev(u_dense, tck)
        xy_dense = np.column_stack([x_d, y_d])
        # Compute biased u values from the dense curve
        u_new = ropt.curvature_biased_resample(xy_dense, num_points, turn_bias=turn_bias)
        x_s, y_s = splev(u_new, tck)
        dx_s, dy_s = splev(u_new, tck, der=1)
        yaw_s = np.arctan2(dy_s, dx_s)
        # Interpolate speed from original
        orig_u = np.linspace(0, 1, len(wps), endpoint=False)
        speed_s = np.interp(u_new, orig_u, wps[:, 3])
        result = np.column_stack([x_s, y_s, yaw_s, speed_s])
        return jsonify(result.tolist())
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/api/inflation", methods=["POST"])
def api_inflation():
    """Return inflation zone points for visualization (no optimization, just the map inflation)."""
    body = request.json
    yaml_path = body["map_yaml"]
    car_width = body.get("car_width", 0.30)
    try:
        img, resolution, origin = ropt.load_map(yaml_path)
        _, _, newly_blocked = ropt.inflate_map(img, resolution, car_width)
        pts = []
        if newly_blocked.any():
            rows, cols = np.where(newly_blocked)
            # Subsample for performance
            step = max(1, len(rows) // 3000)
            for r, c in zip(rows[::step], cols[::step]):
                wx, wy = ropt.pixel_to_world(r, c, resolution, origin, img.shape)
                pts.append([float(wx), float(wy)])
        return jsonify(pts)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    body = request.json
    wps = np.array(body["waypoints"])
    yaml_path = body["map_yaml"]
    margin = body.get("margin", 0.20)
    car_width = body.get("car_width", 0.30)
    num_points = body.get("num_points", 150)
    mu = body.get("mu", 0.34)
    try:
        img, resolution, origin = ropt.load_map(yaml_path)
        # Get track widths for optimization
        img_inflated, free_inflated, newly_blocked = ropt.inflate_map(img, resolution, car_width)
        # Compute track widths at each waypoint
        from scipy.ndimage import distance_transform_edt
        dt = distance_transform_edt(free_inflated) * resolution
        xy = wps[:, :2]
        track_widths = np.zeros(len(xy))
        for i, (x, y) in enumerate(xy):
            r, c = ropt.world_to_pixel(x, y, resolution, origin, img.shape)
            r, c = int(round(r)), int(round(c))
            if 0 <= r < dt.shape[0] and 0 <= c < dt.shape[1]:
                track_widths[i] = dt[r, c]
            else:
                track_widths[i] = margin
        # Run optimization
        turn_bias = body.get("turn_bias", 2.0)
        opt_wps = ropt.optimize_raceline(wps, track_widths, margin=margin,
                                         num_points=num_points, turn_bias=turn_bias)
        # Profile velocity
        v_max = body.get("v_max", 5.0)
        a_max = body.get("a_max", 5.0)
        a_brake = body.get("a_brake", 8.0)
        v_min = body.get("v_min", 0.5)
        opt_wps = ropt.profile_velocity(opt_wps, mu=mu, v_max=v_max, v_min=v_min, a_max=a_max, a_brake=a_brake)
        # Return inflation zone for visualization (sparse — every 4th blocked pixel)
        inflation_pts = []
        if newly_blocked.any():
            rows, cols = np.where(newly_blocked)
            for r, c in zip(rows[::4], cols[::4]):
                wx, wy = ropt.pixel_to_world(r, c, resolution, origin, img.shape)
                inflation_pts.append([float(wx), float(wy)])
        return jsonify({"waypoints": opt_wps.tolist(), "inflation": inflation_pts})
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400


@app.route("/api/export-mppi", methods=["POST"])
def api_export_mppi():
    """Export the active raceline in Levine 9-column format for the MPPI controller.

    Body: {waypoints: [[x,y,yaw,speed_ratio], ...], v_min, v_max, name, w_tr}
    Writes to MPPI_WP_DIR/<name>.csv and verifies it parses with the MPPI loader.
    """
    body = request.json
    wps = np.array(body["waypoints"])
    v_min = float(body.get("v_min", 1.5))
    v_max = float(body.get("v_max", 7.6))
    name = (body.get("name") or "raceline").strip()
    if not name.endswith(".csv"):
        name += ".csv"
    w_tr = float(body.get("w_tr", 1.0))

    try:
        rows = _to_mppi_rows(wps, v_min, v_max, w_tr_const=w_tr)
        out_path = MPPI_WP_DIR / name
        _save_mppi_csv(out_path, rows)

        # Sanity-check: re-load with the exact loader settings the MPPI uses
        check = np.loadtxt(str(out_path), delimiter=";", skiprows=3)
        if check.shape != rows.shape or check.shape[1] != 9:
            return jsonify({"error": f"Round-trip shape mismatch: got {check.shape}, expected {rows.shape}"}), 500

        return jsonify({
            "ok": True,
            "path": str(out_path),
            "rows": int(rows.shape[0]),
            "vx_min": float(rows[:, 5].min()),
            "vx_max": float(rows[:, 5].max()),
            "kappa_min": float(rows[:, 4].min()),
            "kappa_max": float(rows[:, 4].max()),
            "s_total": float(rows[-1, 0] + np.linalg.norm(rows[0, 1:3] - rows[-1, 1:3])),
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 400


@app.route("/api/push", methods=["POST"])
def api_push():
    """Send the active raceline directly to a running node via ROS2 service.

    Body: {target: 'pp'|'mppi', waypoints: [[x,y,yaw,ratio], ...], v_min, v_max}
    PP target sends 4-col; MPPI target encodes vx using v_min/v_max and sends 9-col.
    """
    if STUDIO_NODE is None:
        return jsonify({"error": "ROS node not initialized"}), 500
    body = request.json
    target = body.get("target", "")
    wps = np.array(body["waypoints"])
    if target == "mppi":
        v_min = float(body.get("v_min", 1.5))
        v_max = float(body.get("v_max", 7.6))
        rows = _to_mppi_rows(wps, v_min, v_max)
        client, fmt = STUDIO_NODE.mppi_client, "mppi"
    elif target == "pp":
        rows = wps
        client, fmt = STUDIO_NODE.pp_client, "pure_pursuit"
    else:
        return jsonify({"error": f"Unknown target {target!r}"}), 400

    if not client.wait_for_service(timeout_sec=1.0):
        return jsonify({"error": f"{target} service not available"}), 503

    req = UpdateRaceline.Request()
    r, c = rows.shape
    req.data   = rows.flatten().astype(float).tolist()
    req.rows   = int(r)
    req.cols   = int(c)
    req.format = fmt

    future = client.call_async(req)
    start = time.time()
    while not future.done() and time.time() - start < 3.0:
        time.sleep(0.02)
    if not future.done():
        return jsonify({"error": "timeout"}), 504
    resp = future.result()
    return jsonify({"success": resp.success, "message": resp.message,
                    "rows": int(r), "format": fmt})


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    global STUDIO_NODE
    rclpy.init()
    STUDIO_NODE = StudioNode()
    threading.Thread(target=rclpy.spin, args=(STUDIO_NODE,), daemon=True).start()
    try:
        port = 5050
        url = f"http://localhost:{port}"
        print(f"\n  Raceline Studio → {url}\n")
        threading.Timer(0.1, lambda: webbrowser.open(url)).start()
        app.run(host="0.0.0.0", port=port, use_reloader=False)
    finally:
        STUDIO_NODE.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
