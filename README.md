# raceline_UI_f1tenth

Web-based raceline editor for F1TENTH. Loads waypoint CSVs from a Pure Pursuit
folder and an MPPI folder, lets you edit/optimize/profile racelines, and saves
back in either format (4-col Pure Pursuit or 9-col Levine MPPI).

![Raceline Studio](raceline_studio.png)

## Setup

```bash
pip install -r requirements.txt
```

Then open `app.py` and set the two workspace paths at the top:

```python
PURE_PURSUIT_WAYPOINTS = "~/ros2_ws/.../pure_pursuit/waypoints"
MPPI_WAYPOINTS         = "~/ros2_ws/.../mppi_bringup/waypoints"
MAPS_DIRS              = [...]
```

## Run

```bash
python3 app.py
```

Open `http://localhost:5050`.

## Layout

- `app.py` — Flask backend, all path/save/load endpoints
- `raceline_optimizer.py` — backend for centerline extraction, min-curvature
  optimization, velocity profiling. Also runs as a CLI:
  ```bash
  python3 raceline_optimizer.py extract map.yaml -o center.csv --show
  python3 raceline_optimizer.py optimize center.csv map.yaml -o race.csv
  python3 raceline_optimizer.py profile race.csv -o final.csv --vmax 5.0
  python3 raceline_optimizer.py edit final.csv map.yaml
  ```
- `templates/index.html` — single-page UI

## Save buttons

Two sections in the right sidebar — Pure Pursuit and MPPI. Each has
**Overwrite** (writes back to the active slot's path) and **Save As New**
(writes to `pure_pursuit/race/<name>.csv` or `mppi/sim/<name>.csv`). MPPI
saves use the current v_min / v_max sliders to encode absolute speeds in the
Levine 9-column format.
