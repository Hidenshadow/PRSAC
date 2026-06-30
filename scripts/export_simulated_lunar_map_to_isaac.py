#!/usr/bin/env python
"""Export a simulated lunar planning map as IsaacLab visualization assets.

This creates a non-destructive asset bundle:

* ``moon_planning_flat.npy``: height field compatible with the existing
  IsaacLab custom waypoint script.
* ``global_planner_waypoints_clean.csv``: global-frame waypoints accepted by
  ``--waypoint-format global``.
* ``obstacle_mask.npy`` and ``metadata.json`` for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_simulated_lunar_clean_map import (
    astar,
    make_cost,
    make_lunar_dem,
    make_obstacle_mask,
    smooth_path,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exports" / "isaac_visualization" / "sim_lunar_clean"
DEFAULT_ISAAC_CUSTOM = Path(r"<DESKTOP>\ISAAC\IsaacLab\scripts\custom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--map-size-m", type=float, default=50.0)
    parser.add_argument("--height-scale-m", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--isaac-custom-dir", type=Path, default=DEFAULT_ISAAC_CUSTOM)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_waypoints(
    path: Path,
    route_rc: np.ndarray,
    map_size_m: float,
    grid_size: int,
    waypoint_stride: int = 6,
) -> None:
    route = np.asarray(route_rc, dtype=np.float32)
    if len(route) == 0:
        raise ValueError("empty route")
    keep = list(range(0, len(route), max(int(waypoint_stride), 1)))
    if keep[-1] != len(route) - 1:
        keep.append(len(route) - 1)
    # Waypoints are written in the existing global planner frame:
    # origin top-left, +x right, +y down, units meters.
    scale = float(map_size_m) / float(max(int(grid_size) - 1, 1))

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "x_global_m", "y_global_m", "row_global", "col_global"],
        )
        writer.writeheader()
        for out_idx, route_idx in enumerate(keep):
            row = float(route[route_idx, 0])
            col = float(route[route_idx, 1])
            writer.writerow(
                {
                    "index": out_idx,
                    "x_global_m": f"{col * scale:.6f}",
                    "y_global_m": f"{row * scale:.6f}",
                    "row_global": f"{row:.3f}",
                    "col_global": f"{col:.3f}",
                }
            )


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_size = int(args.grid_size)
    shape = (grid_size, grid_size)
    rng = np.random.default_rng(int(args.seed))
    obstacle_mask = make_obstacle_mask(shape, rng)
    start = (max(2, int(round(0.092 * (grid_size - 1)))), max(2, int(round(0.100 * (grid_size - 1)))))
    goal = (min(grid_size - 3, int(round(0.883 * (grid_size - 1)))), min(grid_size - 3, int(round(0.866 * (grid_size - 1)))))
    obstacle_mask[start[0] - 3 : start[0] + 4, start[1] - 3 : start[1] + 4] = False
    obstacle_mask[goal[0] - 4 : goal[0] + 5, goal[1] - 4 : goal[1] + 5] = False

    height_norm = make_lunar_dem(shape, obstacle_mask, rng).astype(np.float32)
    height_m = (height_norm - float(height_norm.mean())) * float(args.height_scale_m)
    cost = make_cost(height_norm, obstacle_mask)
    route = smooth_path(astar(cost, obstacle_mask, start, goal))

    terrain_path = output_dir / "moon_planning_flat.npy"
    obstacle_path = output_dir / "obstacle_mask.npy"
    waypoint_path = output_dir / "global_planner_waypoints_clean.csv"
    metadata_path = output_dir / "metadata.json"
    run_cmd_path = output_dir / "run_in_isaac.ps1"

    np.save(terrain_path, height_m.astype(np.float32))
    np.save(obstacle_path, obstacle_mask.astype(np.bool_))
    write_waypoints(waypoint_path, route, map_size_m=float(args.map_size_m), grid_size=grid_size)

    isaac_custom_dir = args.isaac_custom_dir
    isaac_script = isaac_custom_dir / "create_moon_planning_robot_waypoint.py"
    metadata = {
        "terrain_path": str(terrain_path),
        "obstacle_mask_path": str(obstacle_path),
        "waypoint_path": str(waypoint_path),
        "isaac_custom_dir": str(isaac_custom_dir),
        "isaac_script": str(isaac_script),
        "grid_size": grid_size,
        "map_size_m": float(args.map_size_m),
        "height_scale_m": float(args.height_scale_m),
        "seed": int(args.seed),
        "start_rc": list(start),
        "goal_rc": list(goal),
        "num_waypoints": sum(1 for _ in waypoint_path.open("r", encoding="utf-8")) - 1,
        "note": "To run with the existing Isaac script, copy moon_planning_flat.npy into the Isaac custom directory or add a terrain-npy argument there.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    command = (
        "$isaac = \"C:\\Users\\anonymous\\Desktop\\ISAAC\\IsaacLab\"\n"
        "$custom = Join-Path $isaac \"scripts\\custom\"\n"
        "# Non-destructive preview command. Copy the terrain file only after backing up the existing one:\n"
        f"# Copy-Item -LiteralPath \"{terrain_path}\" -Destination (Join-Path $custom \"moon_planning_flat.npy\")\n"
        "Set-Location $isaac\n"
        ".\\isaaclab.bat -p scripts\\custom\\create_moon_planning_robot_waypoint.py "
        f"--waypoint-file \"{waypoint_path}\" --waypoint-format global "
        f"--map-size-m {float(args.map_size_m):.6g} --map-resolution 0.03 "
        "--spawn-at-first-waypoint 1 --enable-simple-avoid 1\n"
    )
    run_cmd_path.write_text(command, encoding="utf-8")

    print(terrain_path)
    print(waypoint_path)
    print(obstacle_path)
    print(metadata_path)
    print(run_cmd_path)


if __name__ == "__main__":
    main()

