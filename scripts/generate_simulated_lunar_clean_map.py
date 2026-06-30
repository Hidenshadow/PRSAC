#!/usr/bin/env python
"""Generate a clean simulated lunar planning map with clear obstacles.

The output is intended for a paper front figure: no text, labels, numbers,
legends, colorbars, captions, or watermarks.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter, zoom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "frontpage_simulated_lunar_clean_map"


@dataclass(frozen=True)
class ObstacleSpec:
    row: float
    col: float
    radius_row: float
    radius_col: float
    angle_deg: float


OBSTACLES = (
    ObstacleSpec(50, 63, 12, 17, -18),
    ObstacleSpec(74, 103, 18, 14, 30),
    ObstacleSpec(103, 86, 13, 19, -40),
    ObstacleSpec(115, 126, 20, 15, 18),
    ObstacleSpec(143, 156, 17, 22, -28),
    ObstacleSpec(163, 99, 15, 12, 12),
    ObstacleSpec(184, 185, 18, 14, 34),
    ObstacleSpec(199, 137, 12, 18, -32),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--grid-size", type=int, default=240)
    parser.add_argument("--render-scale", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--dpi", type=int, default=520)
    return parser.parse_args()


def unit(values: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.nanpercentile(arr, low_pct))
    hi = float(np.nanpercentile(arr, high_pct))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def ellipse_field(
    shape: tuple[int, int],
    spec: ObstacleSpec,
) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float32)
    y = rows - float(spec.row)
    x = cols - float(spec.col)
    angle = np.deg2rad(float(spec.angle_deg))
    ca = float(np.cos(angle))
    sa = float(np.sin(angle))
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    return (xr / float(spec.radius_col)) ** 2 + (yr / float(spec.radius_row)) ** 2


def make_obstacle_mask(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    noise = gaussian_filter(rng.normal(size=shape).astype(np.float32), sigma=2.4)
    noise = unit(noise, 2.0, 98.0) - 0.5
    mask = np.zeros(shape, dtype=bool)
    for spec in OBSTACLES:
        field = ellipse_field(shape, spec)
        irregular = field + 0.18 * noise
        mask |= irregular <= 1.0

    for center_row, center_col in ((38, 171), (73, 178), (91, 37), (130, 205), (207, 55), (221, 113)):
        rr, cc = np.indices(shape, dtype=np.float32)
        radius = float(rng.uniform(4.0, 7.0))
        mask |= (rr - center_row) ** 2 + (cc - center_col) ** 2 <= radius**2
    return mask


def crater(height: np.ndarray, row: float, col: float, radius: float, depth: float, rim: float) -> None:
    rows, cols = np.indices(height.shape, dtype=np.float32)
    r = np.sqrt((rows - float(row)) ** 2 + (cols - float(col)) ** 2) / float(radius)
    bowl = -float(depth) * np.exp(-(r**2) * 1.5)
    ring = float(rim) * np.exp(-((r - 1.0) ** 2) / 0.035)
    height += bowl + ring


def make_lunar_dem(shape: tuple[int, int], obstacle_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = np.zeros(shape, dtype=np.float32)
    for sigma, weight in ((42, 1.15), (20, 0.85), (8, 0.46), (3, 0.20)):
        base += weight * gaussian_filter(rng.normal(size=shape).astype(np.float32), sigma=sigma)

    height = unit(base, 1.0, 99.0) * 2.0 - 1.0
    for row, col, radius, depth, rim in (
        (56, 151, 22, 0.42, 0.25),
        (95, 55, 18, 0.36, 0.23),
        (127, 187, 30, 0.34, 0.20),
        (178, 75, 24, 0.40, 0.25),
        (205, 162, 20, 0.32, 0.20),
    ):
        crater(height, row, col, radius, depth, rim)

    obstacle_ridge = gaussian_filter(obstacle_mask.astype(np.float32), sigma=1.2)
    height += 0.34 * obstacle_ridge
    height += 0.08 * gaussian_filter(rng.normal(size=shape).astype(np.float32), sigma=0.65)
    return unit(height, 0.5, 99.5)


def hillshade(height: np.ndarray, vertical_exaggeration: float = 20.0) -> np.ndarray:
    z = unit(height) * float(vertical_exaggeration)
    gy, gx = np.gradient(z)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(28.0)
    shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    return np.clip((shaded + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def lunar_rgb(height: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shade = hillshade(height)
    relief = unit(np.abs(height - gaussian_filter(height, sigma=2.0)), 1.0, 99.0)
    fine = gaussian_filter(rng.normal(size=height.shape).astype(np.float32), sigma=0.55)
    luminance = 0.10 + 0.82 * unit(0.84 * shade + 0.18 * height + 0.12 * relief, 0.4, 99.6)
    luminance += 0.025 * (unit(fine, 1.0, 99.0) - 0.5)
    rgb = np.stack([1.04 * luminance, 1.02 * luminance, 0.96 * luminance], axis=-1)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def make_cost(height: np.ndarray, obstacle_mask: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(height)
    slope = unit(np.sqrt(gx * gx + gy * gy), 2.0, 98.0)
    rough = unit(np.abs(height - gaussian_filter(height, sigma=2.0)), 1.0, 99.0)
    cost = 1.0 + 3.1 * slope + 1.7 * rough
    obstacle_buffer = gaussian_filter(obstacle_mask.astype(np.float32), sigma=3.0)
    cost += 4.6 * obstacle_buffer
    return cost.astype(np.float32)


def astar(
    cost: np.ndarray,
    obstacle: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> np.ndarray:
    rows, cols = cost.shape
    moves = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, np.sqrt(2.0)),
        (-1, 1, np.sqrt(2.0)),
        (1, -1, np.sqrt(2.0)),
        (1, 1, np.sqrt(2.0)),
    ]
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    min_step = max(float(np.nanmin(cost)), 0.01)

    def heuristic(cell: tuple[int, int]) -> float:
        return min_step * float(np.hypot(goal[0] - cell[0], goal[1] - cell[1]))

    frontier: list[tuple[float, float, tuple[int, int]]] = [(heuristic(start), 0.0, start)]
    best = {start: 0.0}
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    visited: set[tuple[int, int]] = set()
    while frontier:
        _, current_cost, cell = heapq.heappop(frontier)
        if cell in visited:
            continue
        visited.add(cell)
        if cell == goal:
            path = [cell]
            while path[-1] in parents:
                path.append(parents[path[-1]])
            path.reverse()
            return np.asarray(path, dtype=np.float32)
        row, col = cell
        for dr, dc, distance in moves:
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or obstacle[nr, nc]:
                continue
            next_cost = current_cost + distance * 0.5 * (float(cost[row, col]) + float(cost[nr, nc]))
            neighbor = (nr, nc)
            if next_cost < best.get(neighbor, float("inf")):
                best[neighbor] = next_cost
                parents[neighbor] = cell
                heapq.heappush(frontier, (next_cost + heuristic(neighbor), next_cost, neighbor))
    raise RuntimeError("A* failed to find a clean-map path")


def smooth_path(path: np.ndarray, samples: int = 380) -> np.ndarray:
    if len(path) < 4:
        return path.astype(np.float32)
    points = path.astype(np.float64)
    tck, _ = splprep([points[:, 1], points[:, 0]], s=max(len(points) * 0.22, 2.0), k=3)
    u = np.linspace(0.0, 1.0, samples)
    cols, rows = splev(u, tck)
    return np.column_stack([rows, cols]).astype(np.float32)


def draw_obstacles(ax: plt.Axes, obstacle_mask: np.ndarray) -> None:
    dark = np.ma.array(obstacle_mask.astype(np.float32), mask=~obstacle_mask)
    cmap = LinearSegmentedColormap.from_list("obstacle", ["#0a0a0a", "#000000"])
    ax.imshow(dark, cmap=cmap, vmin=0.0, vmax=1.0, alpha=0.82, interpolation="nearest", zorder=8)
    ax.contour(obstacle_mask.astype(np.float32), levels=[0.5], colors="#f2f2ec", linewidths=0.95, alpha=0.78, zorder=9)


def draw_path(ax: plt.Axes, path: np.ndarray) -> None:
    line = ax.plot(path[:, 1], path[:, 0], color="#1161ff", linewidth=5.6, solid_capstyle="round", zorder=18)[0]
    line.set_path_effects([pe.Stroke(linewidth=8.7, foreground="white", alpha=0.96), pe.Normal()])


def draw_markers(ax: plt.Axes, start: tuple[int, int], goal: tuple[int, int]) -> None:
    ax.scatter([start[1]], [start[0]], s=180, c="#25d84a", edgecolors="#053c12", linewidths=1.9, zorder=22)
    ax.scatter([goal[1]], [goal[0]], s=180, c="#ff2424", edgecolors="#690000", linewidths=1.9, zorder=22)


def save_map(
    rgb: np.ndarray,
    obstacle_mask: np.ndarray,
    path: np.ndarray | None,
    start: tuple[int, int],
    goal: tuple[int, int],
    output_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 5.6), dpi=dpi)
    ax.imshow(rgb, interpolation="bilinear")
    draw_obstacles(ax, obstacle_mask)
    if path is not None:
        draw_path(ax, path)
        draw_markers(ax, start, goal)
    ax.set_xlim(-0.8, rgb.shape[1] - 0.2)
    ax.set_ylim(rgb.shape[0] - 0.2, -0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_size = int(args.grid_size)
    shape = (grid_size, grid_size)
    rng = np.random.default_rng(int(args.seed))
    obstacle_mask = make_obstacle_mask(shape, rng)
    start = (22, 24)
    goal = (211, 207)
    obstacle_mask[start[0] - 3 : start[0] + 4, start[1] - 3 : start[1] + 4] = False
    obstacle_mask[goal[0] - 4 : goal[0] + 5, goal[1] - 4 : goal[1] + 5] = False

    height = make_lunar_dem(shape, obstacle_mask, rng)
    cost = make_cost(height, obstacle_mask)
    path = smooth_path(astar(cost, obstacle_mask, start, goal))

    scale = max(int(args.render_scale), 1)
    height_hr = zoom(height, zoom=scale, order=3, mode="nearest")
    obstacle_hr = zoom(obstacle_mask.astype(np.float32), zoom=scale, order=0, mode="nearest") > 0.5
    path_hr = path * float(scale)
    start_hr = (int(start[0] * scale), int(start[1] * scale))
    goal_hr = (int(goal[0] * scale), int(goal[1] * scale))
    rgb = lunar_rgb(height_hr, rng)

    main_path = output_dir / "simulated_clean_lunar_map_with_path.png"
    no_path = output_dir / "simulated_clean_lunar_map_obstacles_only.png"
    metadata = output_dir / "simulated_clean_lunar_map_metadata.json"
    save_map(rgb, obstacle_hr, path_hr, start_hr, goal_hr, main_path, dpi=int(args.dpi))
    save_map(rgb, obstacle_hr, None, start_hr, goal_hr, no_path, dpi=int(args.dpi))
    metadata.write_text(
        json.dumps(
            {
                "grid_size": grid_size,
                "render_scale": scale,
                "render_shape": list(rgb.shape[:2]),
                "seed": int(args.seed),
                "start": list(start),
                "goal": list(goal),
                "obstacle_cells": int(obstacle_mask.sum()),
                "path_cells": int(len(path)),
                "with_path": str(main_path),
                "obstacles_only": str(no_path),
                "style": "simulated clean lunar surface, explicit obstacle mask, no text",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(main_path)
    print(no_path)
    print(metadata)


if __name__ == "__main__":
    main()
