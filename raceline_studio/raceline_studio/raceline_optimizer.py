#!/usr/bin/env python3
"""
Raceline Optimizer — Trajectory planning tool for F1Tenth racing.

Subcommands:
    extract   MAP.yaml              Extract centerline from SLAM map
    optimize  WAYPOINTS.csv MAP.yaml  Minimum-curvature raceline optimization
    profile   WAYPOINTS.csv         Add optimal velocity profile to waypoints
    edit      WAYPOINTS.csv [MAP.yaml]  Interactive raceline editor
    show      WAYPOINTS.csv [MAP.yaml]  Visualize raceline with speed coloring

All outputs use CSV format: x, y, yaw, speed
Compatible with pure_pursuit_node waypoint loading.

Usage examples:
    python3 raceline_optimizer.py extract map.yaml -o center.csv --show
    python3 raceline_optimizer.py profile waypoints.csv -o profiled.csv --vmax 5.0 --show
    python3 raceline_optimizer.py edit waypoints.csv map.yaml -o edited.csv
"""

import argparse
import os
import sys
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from scipy import ndimage
from scipy.optimize import minimize
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

# ---------------------------------------------------------------------------
# Defaults — tune these via CLI flags
# ---------------------------------------------------------------------------
DEFAULTS = {
    'num_points': 200,
    'v_max': 5.0,         # m/s — max straight-line speed
    'a_max': 5.0,         # m/s^2 — max longitudinal acceleration
    'a_brake': 8.0,       # m/s^2 — max braking deceleration
    'mu': 0.34,            # friction coefficient (rubber on smooth floor)
    'g': 9.81,            # gravity
    'v_min': 1.0,         # m/s — minimum speed (never fully stop)
    'margin': 0.20,       # meters — safety margin from track walls
    'wheelbase': 0.3302,  # F1Tenth wheelbase in meters
    'car_width': 0.30,    # meters — car body width for wall inflation
}

# ===================================================================
#  UTILITY FUNCTIONS
# ===================================================================

def load_map(yaml_path):
    """Load occupancy grid map from YAML + PGM files.

    Returns: img (np.array uint8), resolution (float), origin (list[3])
    """
    config = {}
    with open(yaml_path) as f:
        for line in f:
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, val = line.split(':', 1)
                config[key.strip()] = val.strip()

    pgm_path = os.path.join(os.path.dirname(yaml_path), config['image'])
    resolution = float(config['resolution'])
    origin_str = config['origin'].strip('[]')
    origin = [float(x) for x in origin_str.split(',')]

    img = np.array(Image.open(pgm_path))
    return img, resolution, origin


def pixel_to_world(row, col, resolution, origin, img_shape):
    """Convert pixel (row, col) to world (x, y) meters."""
    x = col * resolution + origin[0]
    y = (img_shape[0] - 1 - row) * resolution + origin[1]
    return x, y


def world_to_pixel(x, y, resolution, origin, img_shape):
    """Convert world (x, y) to pixel (row, col)."""
    col = (x - origin[0]) / resolution
    row = (img_shape[0] - 1) - (y - origin[1]) / resolution
    return int(round(row)), int(round(col))


def inflate_map(img, resolution, car_width):
    """Inflate obstacles by half car width. Returns inflated image + free mask."""
    import cv2
    free = (img >= 250).astype(np.uint8)
    inflate_px = int(np.ceil((car_width / 2.0) / resolution))
    if inflate_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (2 * inflate_px + 1, 2 * inflate_px + 1))
        free_inflated = cv2.erode(free, kernel, iterations=1)
    else:
        free_inflated = free
    # Build an inflated image (for display): obstacles where free was eroded
    img_inflated = img.copy()
    newly_blocked = (free == 1) & (free_inflated == 0)
    img_inflated[newly_blocked] = 128  # mark inflated zone as gray
    return img_inflated, free_inflated.astype(bool), newly_blocked


def load_waypoints(path):
    """Load waypoints CSV: x, y, yaw, speed."""
    return np.loadtxt(path, delimiter=',')


def save_waypoints(waypoints, path):
    """Save waypoints CSV: x, y, yaw, speed."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    np.savetxt(path, waypoints, fmt='%.6f', delimiter=', ')
    print(f"Saved {len(waypoints)} waypoints to {path}")


def compute_curvature(xy):
    """Compute Menger curvature at each point of a closed loop.

    Args:
        xy: (N, 2) array of [x, y] coordinates
    Returns:
        curvature: (N,) array of unsigned curvature values
    """
    N = len(xy)
    curvature = np.zeros(N)
    for i in range(N):
        p0 = xy[(i - 1) % N]
        p1 = xy[i]
        p2 = xy[(i + 1) % N]
        d1 = p1 - p0
        d2 = p2 - p1
        cross = abs(d1[0] * d2[1] - d1[1] * d2[0])
        a = np.linalg.norm(d1)
        b = np.linalg.norm(d2)
        c = np.linalg.norm(p2 - p0)
        denom = a * b * c
        if denom > 1e-10:
            curvature[i] = 2.0 * cross / denom
    return curvature


def estimate_lap_time(waypoints, v_min=None, v_max=None):
    """Estimate lap time from waypoints with speed ratios."""
    v_min = v_min or DEFAULTS['v_min']
    v_max = v_max or DEFAULTS['v_max']
    xy = waypoints[:, :2]
    ratios = waypoints[:, 3]
    # Convert ratios to absolute speeds for estimation
    speeds = v_min + ratios * (v_max - v_min)
    diffs = np.roll(xy, -1, axis=0) - xy
    ds = np.linalg.norm(diffs, axis=1)
    avg_speeds = (speeds + np.roll(speeds, -1)) / 2.0
    avg_speeds = np.maximum(avg_speeds, 0.1)
    times = ds / avg_speeds
    return np.sum(times)


# ===================================================================
#  VELOCITY PROFILING
# ===================================================================

def profile_velocity(waypoints, v_max=None, a_max=None, a_brake=None,
                     mu=None, v_min=None):
    """Add optimal velocity profile to waypoints using forward-backward integration.

    This is the single biggest performance gain available. Works on ANY waypoints.

    Args:
        waypoints: (N, 4) array [x, y, yaw, speed]
    Returns:
        (N, 4) array with speed column filled by optimal profile
    """
    v_max = v_max or DEFAULTS['v_max']
    a_max = a_max or DEFAULTS['a_max']
    a_brake = a_brake or DEFAULTS['a_brake']
    mu = mu or DEFAULTS['mu']
    v_min = v_min or DEFAULTS['v_min']
    g = DEFAULTS['g']

    N = len(waypoints)
    xy = waypoints[:, :2]

    # Segment lengths
    diffs = np.roll(xy, -1, axis=0) - xy
    ds = np.linalg.norm(diffs, axis=1)

    # Curvature at each point — smoothed to avoid noise from closely-spaced waypoints
    # (turn-biased resampling creates dense clusters where Menger curvature is unreliable)
    curvature = compute_curvature(xy)
    from scipy.ndimage import uniform_filter1d
    curvature = uniform_filter1d(curvature, size=max(3, N // 10), mode='wrap')

    # Max cornering speed: v = sqrt(mu * g / kappa)
    v_cornering = np.full(N, v_max)
    nonzero = curvature > 1e-6
    v_cornering[nonzero] = np.minimum(
        np.sqrt(mu * g / curvature[nonzero]), v_max
    )
    v_cornering = np.clip(v_cornering, v_min, v_max)

    # Forward pass (acceleration-limited) — 2 laps for closed loop propagation
    v_forward = v_cornering.copy()
    for lap in range(2):
        for i in range(N):
            idx = i
            prev_idx = (i - 1) % N
            v_accel = np.sqrt(max(v_forward[prev_idx] ** 2 + 2.0 * a_max * ds[prev_idx], 0))
            v_forward[idx] = min(v_forward[idx], v_accel)

    # Backward pass (braking-limited) — 2 laps
    v_profile = v_forward.copy()
    for lap in range(2):
        for i in range(N - 1, -1, -1):
            idx = i
            next_idx = (i + 1) % N
            v_brake = np.sqrt(max(v_profile[next_idx] ** 2 + 2.0 * a_brake * ds[idx], 0))
            v_profile[idx] = min(v_profile[idx], v_brake)

    # Normalize to 0–1 ratio: 0 = v_min (tightest corner), 1 = v_max (straight)
    speed_ratio = (v_profile - v_min) / (v_max - v_min) if v_max > v_min else np.ones(N)
    speed_ratio = np.clip(speed_ratio, 0.0, 1.0)

    result = waypoints.copy()
    result[:, 3] = speed_ratio
    return result


# ===================================================================
#  MAP / CENTERLINE EXTRACTION
# ===================================================================

def extract_centerline_contour(free_space, dt):
    """Extract track centerline using iterative erosion + contour tracing.

    Strategy: erode the free space until it becomes a thin loop, then
    extract the contour. This is fast (uses cv2 compiled operations) and
    produces a clean, ordered closed path — exactly what we need for racing.

    Args:
        free_space: binary mask of free space
        dt: distance transform of free_space

    Returns:
        points_rc: (N, 2) array of (row, col) pixel coordinates along centerline
    """
    import cv2

    mask = (free_space.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Iteratively erode until the shape is about to break
    # A track loop has a hole; when erosion removes the hole, we've gone too far
    best_contour = None
    prev_mask = mask.copy()

    max_erosions = int(dt.max())
    print(f"  Max DT: {dt.max():.0f} px, iterating erosion...")

    for i in range(max_erosions):
        eroded = cv2.erode(prev_mask, kernel, iterations=1)

        # Check if eroded region still has a hole (is still a loop)
        contours, hierarchy = cv2.findContours(
            eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if hierarchy is None or len(contours) < 2:
            # No more inner contours — the loop collapsed
            print(f"  Loop collapsed at erosion {i}. Using previous step.")
            break

        # Check if any contour has a child (inner boundary = hole)
        has_hole = False
        for ci in range(len(contours)):
            if hierarchy[0][ci][2] >= 0:  # has child
                has_hole = True
                # The child contour is the inner boundary of the eroded shape
                child_idx = hierarchy[0][ci][2]
                best_contour = contours[child_idx]
                break

        if not has_hole:
            print(f"  No hole at erosion {i}. Using previous step.")
            break

        prev_mask = eroded

    if best_contour is None or len(best_contour) < 20:
        # Fallback: use the contour at 50% of max DT
        print("  Fallback: thresholding DT at 50% of max")
        threshold = dt.max() * 0.5
        thin = (dt > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            thin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            best_contour = max(contours, key=cv2.contourArea)

    if best_contour is None or len(best_contour) < 20:
        raise ValueError("Could not extract centerline from map.")

    # Convert cv2 contour (N, 1, 2) → (N, 2) as (row, col)
    pts = best_contour.squeeze()  # (N, 2) as (col, row) in cv2 convention
    points_rc = np.column_stack([pts[:, 1], pts[:, 0]])  # → (row, col)

    print(f"  Extracted contour: {len(points_rc)} points")
    return points_rc


def extract_centerline(img, resolution, origin, num_points=None, margin=None):
    """Extract track centerline from occupancy grid map.

    Args:
        img: grayscale occupancy grid (uint8)
        resolution: meters per pixel
        origin: [x, y, theta] of map origin

    Returns:
        waypoints: (N, 4) array [x, y, yaw, 0]
        track_widths: (N,) distance from centerline to nearest wall in meters
    """
    num_points = num_points or DEFAULTS['num_points']

    # Binary free-space mask
    free = img >= 250  # 254 and 255 are free in ROS map_server convention

    # Keep largest connected component — do NOT fill holes (the track is a ring)
    labeled, num_features = ndimage.label(free)
    if num_features == 0:
        raise ValueError("No free space found in map. Check threshold.")
    component_sizes = ndimage.sum(free, labeled, range(1, num_features + 1))
    largest_label = np.argmax(component_sizes) + 1
    free = (labeled == largest_label)

    # Distance transform — distance to nearest wall for every free pixel
    dt = ndimage.distance_transform_edt(free)

    # Extract centerline using iterative erosion + contour tracing
    print("Extracting centerline...")
    points_px = extract_centerline_contour(free, dt)

    # Convert to world coordinates
    points_world = np.array([
        pixel_to_world(r, c, resolution, origin, img.shape)
        for r, c in points_px
    ])

    # Track width at each point (in meters)
    raw_widths = np.array([dt[r, c] * resolution for r, c in points_px])

    # Force CCW orientation in world coordinates (positive signed area via shoelace)
    # cv2.findContours + y-axis flip in pixel_to_world usually yields CW in world space.
    signed_area = 0.5 * np.sum(
        points_world[:, 0] * np.roll(points_world[:, 1], -1)
        - np.roll(points_world[:, 0], -1) * points_world[:, 1]
    )
    if signed_area < 0:
        points_world = points_world[::-1]
        raw_widths = raw_widths[::-1]
        print("  Reversed contour order → CCW")

    # Smooth and resample with B-spline
    # Use periodic spline for closed loop
    try:
        smoothing = len(points_world) * 0.005
        tck, u = splprep([points_world[:, 0], points_world[:, 1]],
                         s=smoothing, per=True)
    except Exception:
        # Fallback: less smoothing
        tck, u = splprep([points_world[:, 0], points_world[:, 1]],
                         s=0, per=True)

    u_new = np.linspace(0, 1, num_points, endpoint=False)
    x_new, y_new = splev(u_new, tck)

    # Compute yaw from spline derivatives
    dx, dy = splev(u_new, tck, der=1)
    yaw_new = np.arctan2(dy, dx)

    # Interpolate track widths to resampled points
    # Find nearest original point for each new point
    orig_tree = cKDTree(points_world)
    new_points = np.column_stack([x_new, y_new])
    _, nearest = orig_tree.query(new_points)
    nearest = np.clip(nearest, 0, len(raw_widths) - 1)
    track_widths = raw_widths[nearest]

    waypoints = np.column_stack([x_new, y_new, yaw_new, np.zeros(num_points)])
    return waypoints, track_widths


# ===================================================================
#  RACELINE OPTIMIZATION (MINIMUM CURVATURE)
# ===================================================================

def curvature_biased_resample(xy, num_points, turn_bias=2.0):
    """Resample a closed loop with denser points in high-curvature regions.

    Args:
        xy: (N, 2) array of [x, y] positions
        num_points: target number of output points
        turn_bias: how much to concentrate in turns (1.0 = uniform, 8.0 = extreme bias)
    Returns:
        u_new: (num_points,) array of spline parameters in [0, 1)
    """
    N = len(xy)
    curv = compute_curvature(xy)
    # Smooth curvature to avoid single-point spikes
    from scipy.ndimage import uniform_filter1d
    curv = uniform_filter1d(curv, size=max(3, N // 20), mode='wrap')
    curv_norm = curv / (np.max(curv) + 1e-10)
    # Weight: raise to a power for stronger effect at high bias
    weights = 1.0 + (turn_bias - 1.0) * curv_norm
    # Compute arc-length parameterization
    diffs = np.diff(xy, axis=0, append=xy[:1])
    seg_lengths = np.linalg.norm(diffs, axis=1)
    seg_lengths = np.maximum(seg_lengths, 1e-10)
    # Weighted arc length: segments in high-curvature get inflated length → more samples
    weighted_lengths = seg_lengths * weights
    cum = np.cumsum(weighted_lengths)
    cum = np.concatenate([[0], cum])
    cum = cum / cum[-1]  # normalize to [0, 1]
    # Source u values (one per original point + wrap)
    u_src = np.linspace(0, 1, N + 1, endpoint=True)
    # Target: uniform in weighted arc-length space
    u_targets = np.linspace(0, 1, num_points, endpoint=False)
    u_new = np.interp(u_targets, cum, u_src)
    # Clamp to valid spline range
    u_new = np.clip(u_new, 0, 1 - 1e-10)
    return u_new


def optimize_raceline(waypoints, track_widths, margin=None, num_points=None, turn_bias=2.0):
    """Optimize raceline for minimum curvature within track bounds.

    Shifts each waypoint laterally to minimize sum of squared curvatures
    while staying within the track.

    Args:
        waypoints: (N, 4) centerline waypoints [x, y, yaw, speed]
        track_widths: (N,) distance from centerline to wall
        margin: safety margin from walls in meters
        turn_bias: point density bias toward turns (1.0=uniform, 3.0=heavy)

    Returns:
        optimized: (N, 4) waypoints with optimized x, y positions
    """
    margin = margin or DEFAULTS['margin']
    num_points = num_points or DEFAULTS['num_points']

    N = len(waypoints)
    xy = waypoints[:, :2]

    # Compute normal vectors (perpendicular to path tangent)
    tangents = np.roll(xy, -1, axis=0) - xy
    norms = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    lengths = np.linalg.norm(norms, axis=1, keepdims=True)
    lengths[lengths < 1e-6] = 1.0
    norms /= lengths

    # Available lateral displacement
    max_disp = np.maximum(track_widths - margin, 0.0)

    def raceline_from_alpha(alpha):
        return xy + (alpha[:, np.newaxis] * max_disp[:, np.newaxis]) * norms

    def objective(alpha):
        pts = raceline_from_alpha(alpha)
        p0 = np.roll(pts, 1, axis=0)
        p1 = pts
        p2 = np.roll(pts, -1, axis=0)
        d1 = p1 - p0
        d2 = p2 - p1
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        norm_d1 = np.linalg.norm(d1, axis=1)
        norm_d2 = np.linalg.norm(d2, axis=1)
        norm_d02 = np.linalg.norm(p2 - p0, axis=1)
        denom = norm_d1 * norm_d2 * norm_d02
        denom = np.maximum(denom, 1e-10)
        curvature = cross / denom
        return np.sum(curvature ** 2)

    print(f"Optimizing raceline ({N} points)...")
    alpha0 = np.zeros(N)
    bounds = [(-1.0, 1.0)] * N

    result = minimize(
        objective,
        alpha0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 200, 'ftol': 1e-8, 'disp': True}
    )

    if not result.success:
        print(f"Warning: optimization did not fully converge: {result.message}")

    optimized_xy = raceline_from_alpha(result.x)

    # Smooth the result with curvature-biased resampling
    try:
        tck, u = splprep([optimized_xy[:, 0], optimized_xy[:, 1]],
                         s=0.01, per=True)
        # Bias point density toward turns
        u_new = curvature_biased_resample(optimized_xy, num_points, turn_bias=turn_bias)
        x_s, y_s = splev(u_new, tck)
        dx_s, dy_s = splev(u_new, tck, der=1)
        yaw_s = np.arctan2(dy_s, dx_s)
        return np.column_stack([x_s, y_s, yaw_s, np.zeros(num_points)])
    except Exception:
        # If spline fails, just use the raw optimized points
        dx = np.roll(optimized_xy[:, 0], -1) - optimized_xy[:, 0]
        dy = np.roll(optimized_xy[:, 1], -1) - optimized_xy[:, 1]
        yaw = np.arctan2(dy, dx)
        return np.column_stack([optimized_xy[:, 0], optimized_xy[:, 1],
                                yaw, np.zeros(N)])


# ===================================================================
#  VISUALIZATION
# ===================================================================

def get_map_extent(img, resolution, origin):
    """Get [xmin, xmax, ymin, ymax] for imshow extent."""
    xmin = origin[0]
    ymin = origin[1]
    xmax = xmin + img.shape[1] * resolution
    ymax = ymin + img.shape[0] * resolution
    return [xmin, xmax, ymin, ymax]


def plot_raceline_on_map(ax, waypoints, img=None, resolution=None, origin=None):
    """Plot raceline colored by speed ratio, optionally with map background."""
    if img is not None:
        extent = get_map_extent(img, resolution, origin)
        ax.imshow(img, cmap='gray', origin='upper', extent=extent, alpha=0.5)

    xy = waypoints[:, :2]
    ratios = waypoints[:, 3]

    # Create colored line segments
    closed_xy = np.vstack([xy, xy[0:1]])
    closed_ratios = np.append(ratios, ratios[0])

    points = closed_xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_ratios = (closed_ratios[:-1] + closed_ratios[1:]) / 2

    norm = plt.Normalize(0, 1.0)
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linewidth=2.5)
    lc.set_array(seg_ratios)
    ax.add_collection(lc)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Speed ratio (0=slow, 1=fast)', shrink=0.8)

    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')


def show_raceline(waypoints, img=None, resolution=None, origin=None,
                  title="Raceline"):
    """Full visualization: map + raceline, curvature plot, speed plot."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Raceline on map
    plot_raceline_on_map(axes[0], waypoints, img, resolution, origin)
    lap_time = estimate_lap_time(waypoints)
    axes[0].set_title(f"{title}\nEst. lap time: {lap_time:.2f}s")

    # Curvature plot
    curvature = compute_curvature(waypoints[:, :2])
    xy = waypoints[:, :2]
    diffs = np.roll(xy, -1, axis=0) - xy
    ds = np.linalg.norm(diffs, axis=1)
    distance = np.cumsum(ds)
    distance = np.insert(distance[:-1], 0, 0)

    axes[1].plot(distance, curvature, 'b-', linewidth=0.8)
    axes[1].fill_between(distance, curvature, alpha=0.3)
    axes[1].set_xlabel('Distance along track (m)')
    axes[1].set_ylabel('Curvature (1/m)')
    axes[1].set_title('Curvature Profile')
    axes[1].grid(True, alpha=0.3)

    # Speed ratio plot
    ratios = waypoints[:, 3]
    axes[2].plot(distance, ratios, 'g-', linewidth=1.5)
    axes[2].fill_between(distance, ratios, alpha=0.3, color='green')
    axes[2].set_xlabel('Distance along track (m)')
    axes[2].set_ylabel('Speed ratio (0=slow, 1=fast)')
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title(f'Speed Ratio (avg: {ratios.mean():.2f})')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ===================================================================
#  INTERACTIVE EDITOR
# ===================================================================

class RacelineEditor:
    """Interactive matplotlib-based raceline editor.

    Controls:
        Left-click + drag: move nearest waypoint
        Shift + left-click: insert new waypoint
        Right-click: delete nearest waypoint
        'p': re-run velocity profiling
        'o': re-run optimization (requires map data)
        's': save to file
        'q': quit
    """

    PICK_RADIUS = 0.3  # meters — how close click must be to grab a point

    def __init__(self, waypoints, output_path, img=None, resolution=None,
                 origin=None, track_widths=None):
        self.waypoints = waypoints.copy()
        self.output_path = output_path
        self.img = img
        self.resolution = resolution
        self.origin = origin
        self.track_widths = track_widths
        self.dragging_idx = None
        self.modified = False

        self.fig, self.ax = plt.subplots(figsize=(14, 10))

        # Disable matplotlib default keybindings that conflict with ours
        plt.rcParams['keymap.save'] = []
        plt.rcParams['keymap.quit'] = []
        plt.rcParams['keymap.quit_all'] = []
        plt.rcParams['keymap.pan'] = []           # frees 'p'

        # Event connections
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._redraw()

    def _redraw(self):
        self.ax.cla()

        # Map background
        if self.img is not None:
            extent = get_map_extent(self.img, self.resolution, self.origin)
            self.ax.imshow(self.img, cmap='gray', origin='upper',
                          extent=extent, alpha=0.5)

        xy = self.waypoints[:, :2]
        ratios = self.waypoints[:, 3]

        # Colored line
        closed_xy = np.vstack([xy, xy[0:1]])
        closed_ratios = np.append(ratios, ratios[0])
        points = closed_xy.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_ratios = (closed_ratios[:-1] + closed_ratios[1:]) / 2

        norm = plt.Normalize(0, 1.0)
        lc = LineCollection(segments, cmap='RdYlGn', norm=norm, linewidth=2.5)
        lc.set_array(seg_ratios)
        self.ax.add_collection(lc)

        # Waypoint markers (small dots)
        self.ax.scatter(xy[:, 0], xy[:, 1], c=ratios, cmap='RdYlGn',
                       s=8, zorder=5, norm=norm, edgecolors='none')

        # Every 10th point: larger marker with index
        step = max(1, len(xy) // 20)
        for i in range(0, len(xy), step):
            self.ax.annotate(str(i), (xy[i, 0], xy[i, 1]),
                           fontsize=6, color='blue', alpha=0.7)

        mod_str = " [MODIFIED]" if self.modified else ""
        if self.dragging_idx is not None:
            lap_str = f"Est. lap: --s"
        else:
            self._cached_lap_time = estimate_lap_time(self.waypoints)
            lap_str = f"Est. lap: {self._cached_lap_time:.2f}s"
        self.ax.set_title(
            f"Raceline Editor — {len(self.waypoints)} pts — "
            f"{lap_str}{mod_str}\n"
            f"[drag=move] [shift+click=insert] [right-click=delete] "
            f"[p=profile] [o=optimize] [s=save] [q=quit]"
        )
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

        self.fig.canvas.draw_idle()

    def _nearest_idx(self, x, y):
        """Find index of nearest waypoint to (x, y)."""
        dists = np.sqrt((self.waypoints[:, 0] - x) ** 2 +
                        (self.waypoints[:, 1] - y) ** 2)
        idx = np.argmin(dists)
        return idx, dists[idx]

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        idx, dist = self._nearest_idx(event.xdata, event.ydata)

        if event.button == 3:  # Right-click: delete
            if len(self.waypoints) > 10:
                self.waypoints = np.delete(self.waypoints, idx, axis=0)
                if self.track_widths is not None and len(self.track_widths) > idx:
                    self.track_widths = np.delete(self.track_widths, idx)
                self.modified = True
                self._redraw()
            return

        if event.button == 1:
            if hasattr(event, 'key') and event.key == 'shift':
                # Shift+click: insert point
                self._insert_point(event.xdata, event.ydata)
                return

            if dist < self.PICK_RADIUS:
                self.dragging_idx = idx

    def _on_release(self, event):
        if self.dragging_idx is not None:
            self.modified = True
            # Recompute yaw for moved point and neighbors
            N = len(self.waypoints)
            for offset in [-1, 0, 1]:
                i = (self.dragging_idx + offset) % N
                j = (i + 1) % N
                dx = self.waypoints[j, 0] - self.waypoints[i, 0]
                dy = self.waypoints[j, 1] - self.waypoints[i, 1]
                self.waypoints[i, 2] = np.arctan2(dy, dx)
            self.dragging_idx = None
            self._redraw()

    def _on_motion(self, event):
        if self.dragging_idx is not None and event.inaxes == self.ax:
            self.waypoints[self.dragging_idx, 0] = event.xdata
            self.waypoints[self.dragging_idx, 1] = event.ydata
            # Live update (lightweight)
            self._redraw()

    def _insert_point(self, x, y):
        """Insert a new waypoint at (x, y) between the two nearest neighbors."""
        idx, _ = self._nearest_idx(x, y)
        N = len(self.waypoints)
        # Insert after the nearest point
        next_idx = (idx + 1) % N
        mid_yaw = np.arctan2(y - self.waypoints[idx, 1],
                             x - self.waypoints[idx, 0])
        new_pt = np.array([[x, y, mid_yaw, self.waypoints[idx, 3]]])
        self.waypoints = np.insert(self.waypoints, idx + 1, new_pt, axis=0)
        if self.track_widths is not None:
            tw = self.track_widths[idx] if idx < len(self.track_widths) else 0.5
            self.track_widths = np.insert(self.track_widths, idx + 1, tw)
        self.modified = True
        self._redraw()

    def _on_key(self, event):
        if event.key == 'p':
            print("Re-profiling velocity...")
            self.waypoints = profile_velocity(self.waypoints)
            self.modified = True
            self._redraw()
            print("Done.")
        elif event.key == 'o':
            if self.track_widths is not None and self.img is not None:
                print("Re-optimizing raceline...")
                self.waypoints = optimize_raceline(
                    self.waypoints, self.track_widths)
                self.waypoints = profile_velocity(self.waypoints)
                self.modified = True
                self._redraw()
                print("Done.")
            else:
                print("Cannot optimize without map data. Load a map YAML.")
        elif event.key == 's':
            save_waypoints(self.waypoints, self.output_path)
            self.modified = False
            self._redraw()
        elif event.key == 'q':
            if self.modified:
                print("Warning: unsaved changes. Press 's' to save or 'q' again.")
                self.modified = False  # next 'q' will close
            else:
                plt.close(self.fig)

    def run(self):
        """Show the editor window. Returns waypoints when closed."""
        plt.show()
        return self.waypoints


# ===================================================================
#  CLI SUBCOMMANDS
# ===================================================================

def cmd_extract(args):
    """Extract centerline from SLAM map."""
    img, resolution, origin = load_map(args.map)
    print(f"Map loaded: {img.shape[1]}x{img.shape[0]} pixels, "
          f"resolution={resolution} m/px")

    waypoints, track_widths = extract_centerline(
        img, resolution, origin,
        num_points=args.num_points,
        margin=args.margin
    )
    print(f"Extracted {len(waypoints)} centerline points")

    if args.reverse:
        waypoints = waypoints[::-1]
        track_widths = track_widths[::-1]
        # Recompute yaw for reversed direction
        for i in range(len(waypoints)):
            dx = waypoints[(i+1) % len(waypoints), 0] - waypoints[i, 0]
            dy = waypoints[(i+1) % len(waypoints), 1] - waypoints[i, 1]
            waypoints[i, 2] = np.arctan2(dy, dx)
        print("Reversed waypoint order")

    # Save track widths alongside waypoints (as a separate .npy file for optimize)
    output = args.output or os.path.splitext(args.map)[0] + '_centerline.csv'
    save_waypoints(waypoints, output)

    tw_path = os.path.splitext(output)[0] + '_trackwidths.npy'
    np.save(tw_path, track_widths)
    print(f"Track widths saved to {tw_path}")

    if args.show:
        fig = show_raceline(waypoints, img, resolution, origin,
                           title="Extracted Centerline")
        plt.show()


def cmd_optimize(args):
    """Optimize raceline for minimum curvature."""
    waypoints = load_waypoints(args.waypoints)
    img, resolution, origin = load_map(args.map)
    car_width = args.car_width

    # Inflate map by half car width
    img_inflated, free_inflated, inflated_zone = inflate_map(
        img, resolution, car_width)
    print(f"Inflated obstacles by {car_width/2:.3f}m (half car width {car_width}m)")

    # Compute track widths from inflated free space
    dt = ndimage.distance_transform_edt(free_inflated)

    # Try to use pre-computed track widths, else estimate from inflated map
    tw_path = os.path.splitext(args.waypoints)[0] + '_trackwidths.npy'
    if os.path.exists(tw_path) and car_width < 0.01:
        track_widths = np.load(tw_path)
        print(f"Loaded track widths from {tw_path}")
    else:
        print("Computing track widths from inflated map...")
        track_widths = np.array([
            dt[world_to_pixel(wp[0], wp[1], resolution, origin, img.shape)] * resolution
            for wp in waypoints
        ])

    num_pts = args.num_points if args.num_points else len(waypoints)
    optimized = optimize_raceline(
        waypoints, track_widths,
        margin=args.margin,
        num_points=num_pts
    )

    # Add velocity profile (ratios 0-1)
    optimized = profile_velocity(
        optimized,
        a_max=args.amax, a_brake=args.abrake,
        mu=args.mu
    )

    if args.reverse:
        optimized = optimized[::-1]
        for i in range(len(optimized)):
            dx = optimized[(i+1) % len(optimized), 0] - optimized[i, 0]
            dy = optimized[(i+1) % len(optimized), 1] - optimized[i, 1]
            optimized[i, 2] = np.arctan2(dy, dx)
        # Re-profile since direction affects braking zones
        optimized = profile_velocity(
            optimized, a_max=args.amax, a_brake=args.abrake, mu=args.mu)
        print("Reversed waypoint order and re-profiled")

    output = args.output or os.path.splitext(args.waypoints)[0] + '_optimized.csv'
    save_waypoints(optimized, output)

    # Save track widths for the optimized line
    tw_opt = np.array([
        dt[world_to_pixel(wp[0], wp[1], resolution, origin, img.shape)] * resolution
        for wp in optimized
    ])
    tw_opt_path = os.path.splitext(output)[0] + '_trackwidths.npy'
    np.save(tw_opt_path, tw_opt)

    if args.show:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        # Panel 1: Inflation visualization (zoomed to track)
        extent = get_map_extent(img, resolution, origin)
        axes[0].imshow(img, cmap='gray', origin='upper', extent=extent, alpha=0.4)
        # Overlay inflated zone in red
        inflate_rgba = np.zeros((*inflated_zone.shape, 4))
        inflate_rgba[inflated_zone] = [1, 0.2, 0.2, 0.6]  # red with alpha
        axes[0].imshow(inflate_rgba, origin='upper', extent=extent)
        # Show raceline on top
        xy = optimized[:, :2]
        axes[0].plot(xy[:, 0], xy[:, 1], 'b-', linewidth=1.5, label='Raceline')
        axes[0].plot(waypoints[:, 0], waypoints[:, 1], 'g--', linewidth=1,
                    alpha=0.5, label='Centerline')
        axes[0].legend(fontsize=8)
        axes[0].set_title(f'Inflation: {car_width/2:.2f}m + margin {args.margin:.2f}m')
        axes[0].set_aspect('equal')
        # Zoom to trajectory
        pad = 3
        axes[0].set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
        axes[0].set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)

        # Panel 2 & 3: speed-colored raceline + profiles
        plot_raceline_on_map(axes[1], optimized, img, resolution, origin)
        lap_time = estimate_lap_time(optimized)
        axes[1].set_title(f"Optimized Raceline\nEst. lap: {lap_time:.2f}s")

        # Speed ratio profile
        diffs = np.roll(xy, -1, axis=0) - xy
        ds = np.linalg.norm(diffs, axis=1)
        distance = np.insert(np.cumsum(ds)[:-1], 0, 0)
        ratios = optimized[:, 3]
        axes[2].plot(distance, ratios, 'g-', linewidth=1.5)
        axes[2].fill_between(distance, ratios, alpha=0.3, color='green')
        axes[2].set_xlabel('Distance (m)')
        axes[2].set_ylabel('Speed ratio (0=slow, 1=fast)')
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].set_title(f'Speed Ratio (avg: {ratios.mean():.2f})')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def cmd_profile(args):
    """Add velocity profile to waypoints."""
    waypoints = load_waypoints(args.waypoints)
    print(f"Loaded {len(waypoints)} waypoints")

    if args.reverse:
        waypoints = waypoints[::-1]
        for i in range(len(waypoints)):
            dx = waypoints[(i+1) % len(waypoints), 0] - waypoints[i, 0]
            dy = waypoints[(i+1) % len(waypoints), 1] - waypoints[i, 1]
            waypoints[i, 2] = np.arctan2(dy, dx)
        print("Reversed waypoint order")

    profiled = profile_velocity(
        waypoints,
        a_max=args.amax, a_brake=args.abrake,
        mu=args.mu
    )

    ratios = profiled[:, 3]
    print(f"Speed ratio: min={ratios.min():.2f}, max={ratios.max():.2f}, "
          f"avg={ratios.mean():.2f} (0=slow, 1=fast)")
    print(f"Estimated lap time: {estimate_lap_time(profiled):.2f}s")

    output = args.output or os.path.splitext(args.waypoints)[0] + '_profiled.csv'
    save_waypoints(profiled, output)

    if args.show:
        fig = show_raceline(profiled, title="Velocity-Profiled Raceline")
        plt.show()


def cmd_edit(args):
    """Interactive raceline editor."""
    waypoints = load_waypoints(args.waypoints)
    print(f"Loaded {len(waypoints)} waypoints")

    img = resolution = origin = track_widths = None
    if args.map:
        img, resolution, origin = load_map(args.map)
        print(f"Map loaded: {img.shape[1]}x{img.shape[0]}")

        # Try to load track widths
        tw_path = os.path.splitext(args.waypoints)[0] + '_trackwidths.npy'
        if os.path.exists(tw_path):
            track_widths = np.load(tw_path)

    output = args.output or os.path.splitext(args.waypoints)[0] + '_edited.csv'

    editor = RacelineEditor(waypoints, output, img, resolution, origin,
                           track_widths)
    result = editor.run()

    if editor.modified:
        save_waypoints(result, output)


def cmd_show(args):
    """Visualize raceline."""
    waypoints = load_waypoints(args.waypoints)
    print(f"Loaded {len(waypoints)} waypoints")

    img = resolution = origin = None
    if args.map:
        img, resolution, origin = load_map(args.map)

    ratios = waypoints[:, 3]
    if ratios.max() < 0.01:
        print("Note: speed column is all zeros. Run 'profile' first for "
              "speed coloring.")

    print(f"Speed ratio: min={ratios.min():.2f}, max={ratios.max():.2f}, "
          f"avg={ratios.mean():.2f} (0=slow, 1=fast)")
    if ratios.max() > 0.01:
        print(f"Estimated lap time: {estimate_lap_time(waypoints):.2f}s")

    fig = show_raceline(waypoints, img, resolution, origin, title="Raceline")
    plt.show()


# ===================================================================
#  MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Raceline Optimizer — Trajectory planning for F1Tenth',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='command', help='Subcommand')

    # --- extract ---
    p_ext = subparsers.add_parser('extract',
        help='Extract centerline from SLAM map')
    p_ext.add_argument('map', help='Map YAML file path')
    p_ext.add_argument('-o', '--output', help='Output CSV path')
    p_ext.add_argument('-n', '--num-points', type=int,
                       default=DEFAULTS['num_points'],
                       help=f"Number of waypoints (default: {DEFAULTS['num_points']})")
    p_ext.add_argument('--margin', type=float, default=DEFAULTS['margin'],
                       help=f"Wall margin in meters (default: {DEFAULTS['margin']})")
    p_ext.add_argument('--reverse', action='store_true',
                       help='Reverse waypoint order (e.g. CW→CCW)')
    p_ext.add_argument('--show', action='store_true', help='Show plot')

    # --- optimize ---
    p_opt = subparsers.add_parser('optimize',
        help='Minimum-curvature raceline optimization')
    p_opt.add_argument('waypoints', help='Input waypoints CSV')
    p_opt.add_argument('map', help='Map YAML file path')
    p_opt.add_argument('-o', '--output', help='Output CSV path')
    p_opt.add_argument('-n', '--num-points', type=int, default=None,
                       help='Output points (default: same as input)')
    p_opt.add_argument('--margin', type=float, default=DEFAULTS['margin'])
    p_opt.add_argument('--car-width', type=float, default=DEFAULTS['car_width'],
                       help=f"Car width for wall inflation (default: {DEFAULTS['car_width']}m)")
    p_opt.add_argument('--mu', type=float, default=DEFAULTS['mu'],
                       help=f"Friction coeff — tire grip (default: {DEFAULTS['mu']})")
    p_opt.add_argument('--amax', type=float, default=DEFAULTS['a_max'],
                       help=f"Physics model max accel m/s^2 (default: {DEFAULTS['a_max']})")
    p_opt.add_argument('--abrake', type=float, default=DEFAULTS['a_brake'],
                       help=f"Physics model max brake m/s^2 (default: {DEFAULTS['a_brake']})")
    p_opt.add_argument('--reverse', action='store_true',
                       help='Reverse waypoint order (e.g. CW→CCW)')
    p_opt.add_argument('--show', action='store_true', help='Show plot')

    # --- profile ---
    p_prof = subparsers.add_parser('profile',
        help='Add optimal velocity profile (speed ratios 0-1) to waypoints')
    p_prof.add_argument('waypoints', help='Input waypoints CSV')
    p_prof.add_argument('-o', '--output', help='Output CSV path')
    p_prof.add_argument('--mu', type=float, default=DEFAULTS['mu'],
                        help=f"Friction coeff — tire grip (default: {DEFAULTS['mu']})")
    p_prof.add_argument('--amax', type=float, default=DEFAULTS['a_max'],
                        help=f"Physics model max accel m/s^2 (default: {DEFAULTS['a_max']})")
    p_prof.add_argument('--abrake', type=float, default=DEFAULTS['a_brake'],
                        help=f"Physics model max brake m/s^2 (default: {DEFAULTS['a_brake']})")
    p_prof.add_argument('--reverse', action='store_true',
                        help='Reverse waypoint order (e.g. CW→CCW)')
    p_prof.add_argument('--show', action='store_true', help='Show plot')

    # --- edit ---
    p_edit = subparsers.add_parser('edit',
        help='Interactive raceline editor')
    p_edit.add_argument('waypoints', help='Input waypoints CSV')
    p_edit.add_argument('map', nargs='?', default=None,
                        help='Optional map YAML for background')
    p_edit.add_argument('-o', '--output', help='Output CSV path')

    # --- show ---
    p_show = subparsers.add_parser('show',
        help='Visualize raceline with speed coloring')
    p_show.add_argument('waypoints', help='Waypoints CSV')
    p_show.add_argument('map', nargs='?', default=None,
                        help='Optional map YAML for background')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        'extract': cmd_extract,
        'optimize': cmd_optimize,
        'profile': cmd_profile,
        'edit': cmd_edit,
        'show': cmd_show,
    }
    commands[args.command](args)


if __name__ == '__main__':
    main()
