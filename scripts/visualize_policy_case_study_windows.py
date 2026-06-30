#!/usr/bin/env python
"""Create a Windows-safe 3D policy case-study visualization.

This deliberately avoids Isaac/RTX rendering. It renders the exported policy
case-study terrain, path, and a simplified rover body with Matplotlib so the
case can be inspected on Windows before running the full Isaac evaluation on
Linux.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import struct
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_DIR = PROJECT_ROOT / "exports" / "isaac_policy_case_study" / "level2_medium_valt_sac_seed0_task0"
DEFAULT_ROVER_STL = (
    Path(r"<DESKTOP>\ISAAC\IsaacLab\scripts\custom\continuum_description")
    / "visual"
    / "chassis_link.STL"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rover-stl", type=Path, default=DEFAULT_ROVER_STL)
    parser.add_argument("--terrain-stride", type=int, default=1)
    parser.add_argument("--max-rover-triangles", type=int, default=2600)
    parser.add_argument("--make-gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif-frames", type=int, default=72)
    parser.add_argument("--gif-dpi", type=int, default=110)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_waypoints(path: Path) -> np.ndarray:
    points: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            points.append((float(row["x_global_m"]), float(row["y_global_m"])))
    if not points:
        raise ValueError(f"no waypoints in {path}")
    return np.asarray(points, dtype=np.float32)


def height_at_xy(height: np.ndarray, x: float, y: float, map_size_m: float) -> float:
    rows, cols = height.shape
    col = int(round(float(x) / map_size_m * float(cols - 1)))
    row = int(round(float(y) / map_size_m * float(rows - 1)))
    row = int(np.clip(row, 0, rows - 1))
    col = int(np.clip(col, 0, cols - 1))
    return float(height[row, col])


def load_binary_stl(path: Path, max_triangles: int) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"invalid STL: {path}")
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + tri_count * 50
    if expected > len(data):
        raise ValueError(f"STL appears truncated: {path}")
    triangles = np.empty((tri_count, 3, 3), dtype=np.float32)
    offset = 84
    for index in range(tri_count):
        values = struct.unpack_from("<12f", data, offset)
        triangles[index] = np.asarray(values[3:12], dtype=np.float32).reshape(3, 3)
        offset += 50
    if tri_count > int(max_triangles):
        keep = np.linspace(0, tri_count - 1, int(max_triangles)).astype(np.int64)
        triangles = triangles[keep]
    return triangles


def normalized_rover_mesh(path: Path, max_triangles: int) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        triangles = load_binary_stl(path, max_triangles=max_triangles)
    except Exception:
        return None
    vertices = triangles.reshape(-1, 3)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    scale = float(np.max(vertices.max(axis=0) - vertices.min(axis=0)))
    if scale <= 1e-8:
        return None
    triangles = (triangles - center) / scale
    # SolidWorks export axes do not need to be physically exact here; this is a
    # visual rover body marker placed on the policy route.
    triangles = triangles[..., [0, 1, 2]]
    return triangles.astype(np.float32)


def add_rover(
    ax,
    rover_mesh: np.ndarray | None,
    position_xyz: tuple[float, float, float],
    yaw: float,
    size_m: float,
) -> None:
    x, y, z = position_xyz
    if rover_mesh is None:
        body = np.array(
            [
                [-0.45, -0.25, 0.00],
                [0.45, -0.25, 0.00],
                [0.45, 0.25, 0.00],
                [-0.45, 0.25, 0.00],
                [-0.45, -0.25, 0.22],
                [0.45, -0.25, 0.22],
                [0.45, 0.25, 0.22],
                [-0.45, 0.25, 0.22],
            ],
            dtype=np.float32,
        )
        faces = [
            [body[i] for i in (0, 1, 2, 3)],
            [body[i] for i in (4, 5, 6, 7)],
            [body[i] for i in (0, 1, 5, 4)],
            [body[i] for i in (1, 2, 6, 5)],
            [body[i] for i in (2, 3, 7, 6)],
            [body[i] for i in (3, 0, 4, 7)],
        ]
        mesh = np.asarray(faces, dtype=np.float32)
    else:
        mesh = rover_mesh.copy()

    rot = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    mesh = mesh @ rot.T
    mesh = mesh * float(size_m)
    mesh[..., 0] += float(x)
    mesh[..., 1] += float(y)
    mesh[..., 2] += float(z) + 0.16
    collection = Poly3DCollection(mesh, facecolor="#d8d8d8", edgecolor="#555555", linewidths=0.08, alpha=0.92)
    ax.add_collection3d(collection)


def add_path_tube(ax, path_xyz: np.ndarray) -> None:
    segments = np.stack([path_xyz[:-1], path_xyz[1:]], axis=1)
    outline = Line3DCollection(segments, colors="white", linewidths=5.5, alpha=0.85)
    line = Line3DCollection(segments, colors="#1266ff", linewidths=3.0, alpha=1.0)
    ax.add_collection3d(outline)
    ax.add_collection3d(line)


def make_figure(
    height: np.ndarray,
    cost: np.ndarray,
    obstacle: np.ndarray,
    waypoints_xy: np.ndarray,
    metadata: dict,
    rover_mesh: np.ndarray | None,
    azim: float,
    elev: float = 58.0,
) -> plt.Figure:
    map_size_m = float(metadata.get("map_size_m", 20.0))
    rows, cols = height.shape
    stride = max(1, int(metadata.get("_terrain_stride", 1)))
    x = np.linspace(0.0, map_size_m, cols, dtype=np.float32)[::stride]
    y = np.linspace(0.0, map_size_m, rows, dtype=np.float32)[::stride]
    xx, yy = np.meshgrid(x, y)
    zz = height[::stride, ::stride]
    cc = cost[::stride, ::stride]
    oo = obstacle[::stride, ::stride]

    fig = plt.figure(figsize=(8.0, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.get_cmap("viridis")(np.clip(cc, 0.0, 1.0))
    colors[oo] = np.array([0.02, 0.02, 0.02, 1.0])
    ax.plot_surface(xx, yy, zz, facecolors=colors, linewidth=0.0, antialiased=False, shade=False, alpha=0.98)

    path_z = np.array([height_at_xy(height, x0, y0, map_size_m) + 0.11 for x0, y0 in waypoints_xy], dtype=np.float32)
    path_xyz = np.column_stack([waypoints_xy[:, 0], waypoints_xy[:, 1], path_z])
    add_path_tube(ax, path_xyz)

    start = path_xyz[0]
    goal = path_xyz[-1]
    ax.scatter([start[0]], [start[1]], [start[2] + 0.15], s=95, c="#2ca02c", edgecolors="white", linewidths=1.2)
    ax.scatter([goal[0]], [goal[1]], [goal[2] + 0.15], s=95, c="#d62728", edgecolors="white", linewidths=1.2)

    rover_index = min(max(len(path_xyz) // 4, 1), len(path_xyz) - 1)
    rover_pos = path_xyz[rover_index]
    direction = path_xyz[min(rover_index + 1, len(path_xyz) - 1)] - path_xyz[max(rover_index - 1, 0)]
    yaw = float(np.arctan2(direction[1], direction[0]))
    add_rover(ax, rover_mesh, (float(rover_pos[0]), float(rover_pos[1]), float(rover_pos[2])), yaw, size_m=1.15)

    ax.set_xlim(0.0, map_size_m)
    ax.set_ylim(0.0, map_size_m)
    z_pad = max(0.3, 0.15 * float(np.nanmax(height) - np.nanmin(height) + 1e-6))
    ax.set_zlim(float(np.nanmin(height)) - z_pad, float(np.nanmax(height)) + 1.2)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 0.22))
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def main() -> None:
    args = parse_args()
    case_dir = resolve(args.case_dir)
    output_dir = resolve(args.output_dir) if args.output_dir else case_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["_terrain_stride"] = int(args.terrain_stride)
    height = np.load(case_dir / "terrain_heightfield.npy").astype(np.float32)
    cost = np.load(case_dir / "composite_cost.npy").astype(np.float32)
    obstacle = np.load(case_dir / "obstacle_mask.npy").astype(bool)
    waypoints = load_waypoints(case_dir / "policy_waypoints.csv")
    rover_mesh = normalized_rover_mesh(Path(args.rover_stl), max_triangles=int(args.max_rover_triangles))

    fig = make_figure(height, cost, obstacle, waypoints, metadata, rover_mesh, azim=-45.0, elev=58.0)
    png_path = output_dir / "windows_policy_case_study_3d.png"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    if args.make_gif:
        fig = make_figure(height, cost, obstacle, waypoints, metadata, rover_mesh, azim=-45.0, elev=58.0)
        ax = fig.axes[0]

        def update(frame: int):
            ax.view_init(elev=58.0, azim=-70.0 + 360.0 * frame / max(int(args.gif_frames), 1))
            return []

        gif_path = output_dir / "windows_policy_case_study_orbit.gif"
        animation = FuncAnimation(fig, update, frames=int(args.gif_frames), interval=70, blit=False)
        animation.save(gif_path, writer=PillowWriter(fps=14), dpi=int(args.gif_dpi))
        plt.close(fig)
        print(gif_path)

    print(png_path)


if __name__ == "__main__":
    main()

