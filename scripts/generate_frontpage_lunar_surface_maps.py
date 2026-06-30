#!/usr/bin/env python
"""Generate real-DEM lunar surface maps for the paper front figure.

The two standalone images use the same lunar DEM tile, start, and goal:

* clean: rugged shaded-relief terrain and a conservative route.
* corrupted: the planner-visible terrain has local regions falsely smoothed,
  producing a different shortcut route.

No text, labels, legends, colorbars, captions, or watermarks are rendered.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import heapq
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter, zoom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_problem_subfigures_from_maps import nearest_free, unit  # noqa: E402


DEFAULT_LAYERS = PROJECT_ROOT / "maps" / "real_dem_tiles" / "lunar_npd_80_tile" / "real_map_layers.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "frontpage_lunar_real_surface"


@dataclass(frozen=True)
class RouteResult:
    path: np.ndarray
    cost: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=Path, default=DEFAULT_LAYERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=460)
    parser.add_argument("--render-scale", type=int, default=10)
    parser.add_argument("--fine-texture-strength", type=float, default=0.035)
    parser.add_argument("--prefix", type=str, default="frontpage_lunar_real")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_layers(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as raw:
        return {
            key: raw[key].astype(np.float32) if raw[key].dtype != np.bool_ else raw[key].astype(bool)
            for key in raw.files
        }


def ellipse_mask(
    shape: tuple[int, int],
    center: tuple[float, float],
    radii: tuple[float, float],
    angle_deg: float,
) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float32)
    y = rows - float(center[0])
    x = cols - float(center[1])
    angle = np.deg2rad(float(angle_deg))
    ca = float(np.cos(angle))
    sa = float(np.sin(angle))
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    return ((xr / float(radii[1])) ** 2 + (yr / float(radii[0])) ** 2 <= 1.0).astype(np.float32)


def corridor_mask(shape: tuple[int, int], start: tuple[int, int], goal: tuple[int, int], radius: float) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float32)
    a = np.asarray([float(start[0]), float(start[1])], dtype=np.float32)
    b = np.asarray([float(goal[0]), float(goal[1])], dtype=np.float32)
    p = np.stack([rows, cols], axis=-1)
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = np.clip(((p - a) @ ab) / max(denom, 1e-6), 0.0, 1.0)
    closest = a + t[..., None] * ab
    distance = np.sqrt(np.sum((p - closest) ** 2, axis=-1))
    return (distance <= float(radius)).astype(np.float32)


def smoothing_mask(shape: tuple[int, int], start: tuple[int, int], goal: tuple[int, int]) -> np.ndarray:
    mask = 0.90 * corridor_mask(shape, start, goal, radius=4.4)
    mask += 0.95 * ellipse_mask(shape, (34, 34), (11, 17), -42)
    mask += 0.85 * ellipse_mask(shape, (52, 52), (13, 20), -42)
    mask += 0.70 * ellipse_mask(shape, (67, 66), (10, 14), -42)
    mask = unit(gaussian_filter(mask, sigma=2.0), lo=0.0, hi=None)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def make_corrupted_dem(height: np.ndarray, mask: np.ndarray) -> np.ndarray:
    broad = gaussian_filter(height, sigma=9.0)
    local = gaussian_filter(height, sigma=4.2)
    smoothed = 0.82 * broad + 0.18 * local
    flat_strength = np.clip(mask * 0.98, 0.0, 0.98)
    return (height * (1.0 - flat_strength) + smoothed * flat_strength).astype(np.float32)


def contrast(values: np.ndarray, low_pct: float = 1.0, high_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo = float(np.nanpercentile(arr, low_pct))
    hi = float(np.nanpercentile(arr, high_pct))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def hillshade(
    height: np.ndarray,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 26.0,
    vertical_exaggeration: float = 24.0,
) -> np.ndarray:
    dem = np.asarray(height, dtype=np.float32)
    z = unit(dem) * float(vertical_exaggeration)
    gy, gx = np.gradient(z)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(gx * gx + gy * gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = np.deg2rad(azimuth_deg)
    altitude = np.deg2rad(altitude_deg)
    shaded = (
        np.sin(altitude) * np.sin(slope)
        + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    )
    return np.clip((shaded + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)


def lunar_rgb(
    height: np.ndarray,
    roughness: np.ndarray | None = None,
    corrupted_mask: np.ndarray | None = None,
    fine_texture_strength: float = 0.0,
) -> np.ndarray:
    elevation = unit(height)
    shade_key = hillshade(height, azimuth_deg=315.0, altitude_deg=23.0, vertical_exaggeration=32.0)
    shade_fill = hillshade(height, azimuth_deg=45.0, altitude_deg=38.0, vertical_exaggeration=18.0)
    highpass = unit(np.abs(height - gaussian_filter(height, sigma=1.25)))
    rough = unit(roughness) if roughness is not None else highpass
    raw = 0.88 * shade_key + 0.16 * shade_fill + 0.13 * elevation + 0.16 * highpass - 0.22 * rough
    luminance = 0.07 + 0.91 * contrast(raw, 0.5, 99.5)
    if fine_texture_strength > 0.0:
        fine = fine_lunar_texture(height.shape)
        if corrupted_mask is not None:
            fine = fine * (1.0 - 0.75 * np.clip(corrupted_mask, 0.0, 1.0))
        luminance = np.clip(luminance + float(fine_texture_strength) * fine, 0.0, 1.0)
    rgb = np.stack([1.03 * luminance, 1.02 * luminance, 0.98 * luminance], axis=-1)
    rgb = np.clip(rgb, 0.0, 1.0)
    if corrupted_mask is not None:
        smooth = np.clip(corrupted_mask[..., None], 0.0, 1.0)
        flattened_tint = np.array([0.64, 0.63, 0.59], dtype=np.float32)
        rgb = rgb * (1.0 - 0.34 * smooth) + flattened_tint * (0.34 * smooth)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def fine_lunar_texture(shape: tuple[int, int]) -> np.ndarray:
    """Deterministic subtle regolith-like texture for publication rendering."""

    rng = np.random.default_rng(20260629)
    base = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    fine = gaussian_filter(base, sigma=0.55)
    medium = gaussian_filter(base, sigma=1.6)
    broad = gaussian_filter(base, sigma=4.5)
    texture = 0.55 * fine + 0.32 * medium + 0.13 * broad
    return (contrast(texture, 1.0, 99.0) - 0.5).astype(np.float32)


def upscale(values: np.ndarray, scale: int, order: int = 3) -> np.ndarray:
    scale = max(int(scale), 1)
    if scale == 1:
        return np.asarray(values, dtype=np.float32)
    return zoom(np.asarray(values, dtype=np.float32), zoom=scale, order=order, mode="nearest").astype(np.float32)


def terrain_cost(data: dict[str, np.ndarray], smooth_mask: np.ndarray | None, corrupted: bool) -> np.ndarray:
    slope = unit(data["slope_layer"])
    rough = unit(data["roughness_layer"])
    energy = unit(data["layer_energy"])
    hazard = unit(data["layer_hazard"])
    clean = 1.0 + 2.5 * slope + 1.7 * rough + 1.0 * energy + 1.1 * hazard
    if smooth_mask is None:
        return clean.astype(np.float32)

    if corrupted:
        believed = clean * (1.0 - 0.74 * smooth_mask) + 0.35 * smooth_mask
        believed -= 1.10 * corridor_mask(clean.shape, (12, 12), (73, 73), radius=5.4)
        return np.clip(believed, 0.12, None).astype(np.float32)

    rugged_penalty = 3.2 * smooth_mask + 0.9 * gaussian_filter(smooth_mask, sigma=2.0)
    return (clean + rugged_penalty).astype(np.float32)


def reconstruct_path(
    parents: dict[tuple[int, int], tuple[int, int]],
    end: tuple[int, int],
) -> np.ndarray:
    cells = [end]
    while cells[-1] in parents:
        cells.append(parents[cells[-1]])
    cells.reverse()
    return np.asarray(cells, dtype=np.float32)


def astar(
    cost: np.ndarray,
    obstacle: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> RouteResult:
    rows, cols = cost.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
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
    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    parents: dict[tuple[int, int], tuple[int, int]] = {}
    best = {start: 0.0}
    min_step = max(float(np.nanmin(cost)), 0.01)

    def heuristic(cell: tuple[int, int]) -> float:
        return min_step * float(np.hypot(goal[0] - cell[0], goal[1] - cell[1]))

    heapq.heappush(open_heap, (heuristic(start), 0.0, start))
    visited: set[tuple[int, int]] = set()
    while open_heap:
        _, current_cost, cell = heapq.heappop(open_heap)
        if cell in visited:
            continue
        visited.add(cell)
        if cell == goal:
            return RouteResult(path=reconstruct_path(parents, goal), cost=float(current_cost))
        row, col = cell
        for dr, dc, distance in moves:
            nr = row + dr
            nc = col + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or obstacle[nr, nc]:
                continue
            step_cost = distance * 0.5 * (float(cost[row, col]) + float(cost[nr, nc]))
            new_cost = current_cost + step_cost
            neighbor = (nr, nc)
            if new_cost < best.get(neighbor, float("inf")):
                best[neighbor] = new_cost
                parents[neighbor] = cell
                heapq.heappush(open_heap, (new_cost + heuristic(neighbor), new_cost, neighbor))
    raise RuntimeError("A* failed to find a path")


def smooth_route(path: np.ndarray, samples: int = 320) -> np.ndarray:
    if len(path) < 4:
        return path.astype(np.float32)
    points = path.astype(np.float64)
    degree = min(3, len(points) - 1)
    smoothing = max(float(len(points)) * 0.15, 1.0)
    tck, _ = splprep([points[:, 1], points[:, 0]], s=smoothing, k=degree)
    u = np.linspace(0.0, 1.0, samples)
    cols, rows = splev(u, tck)
    return np.column_stack([rows, cols]).astype(np.float32)


def draw_path(ax: plt.Axes, path: np.ndarray, linewidth: float = 5.8) -> None:
    rows = np.clip(path[:, 0], 0, None)
    cols = np.clip(path[:, 1], 0, None)
    line = ax.plot(cols, rows, color="#1161ff", linewidth=linewidth, solid_capstyle="round", zorder=20)[0]
    line.set_path_effects([pe.Stroke(linewidth=linewidth + 3.2, foreground="white", alpha=0.96), pe.Normal()])


def draw_markers(ax: plt.Axes, start: tuple[int, int], goal: tuple[int, int]) -> None:
    ax.scatter([start[1]], [start[0]], s=190, c="#25d84a", edgecolors="#053c12", linewidths=1.9, zorder=24)
    ax.scatter([goal[1]], [goal[0]], s=190, c="#ff2424", edgecolors="#690000", linewidths=1.9, zorder=24)


def save_surface_map(
    rgb: np.ndarray,
    path: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    output_path: Path,
    dpi: int,
    source_shape: tuple[int, int],
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=dpi)
    source_rows, source_cols = source_shape
    ax.imshow(
        rgb,
        interpolation="bilinear",
        extent=(-0.5, source_cols - 0.5, source_rows - 0.5, -0.5),
    )
    draw_path(ax, path)
    draw_markers(ax, start, goal)
    ax.set_xlim(-0.8, source_cols - 0.2)
    ax.set_ylim(source_rows - 0.2, -0.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    layers_path = resolve(args.layers)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_layers(layers_path)
    height = np.asarray(data["height_map"], dtype=np.float32)
    obstacle = np.asarray(data["obstacle_mask"], dtype=bool)
    start = nearest_free(obstacle, (12, 12))
    goal = nearest_free(obstacle, (69, 69))
    mask = smoothing_mask(height.shape, start, goal)
    corrupted_height = make_corrupted_dem(height, mask)

    clean_cost = terrain_cost(data, mask, corrupted=False)
    corrupted_cost = terrain_cost(data, mask, corrupted=True)
    corrupted_obstacle = obstacle & (mask < 0.55)

    clean_route = astar(clean_cost, obstacle, start, goal)
    corrupted_route = astar(corrupted_cost, corrupted_obstacle, start, goal)
    clean_path = smooth_route(clean_route.path)
    corrupted_path = smooth_route(corrupted_route.path)

    roughness = np.asarray(data["roughness_layer"], dtype=np.float32)
    corrupted_roughness = roughness * (1.0 - 0.88 * mask) + gaussian_filter(roughness, sigma=5.0) * (0.12 * mask)
    render_scale = max(int(args.render_scale), 1)
    height_hr = upscale(height, render_scale, order=3)
    corrupted_height_hr = upscale(corrupted_height, render_scale, order=3)
    roughness_hr = upscale(roughness, render_scale, order=3)
    corrupted_roughness_hr = upscale(corrupted_roughness, render_scale, order=3)
    mask_hr = np.clip(upscale(mask, render_scale, order=3), 0.0, 1.0)
    clean_rgb = lunar_rgb(
        height_hr,
        roughness=roughness_hr,
        fine_texture_strength=float(args.fine_texture_strength),
    )
    corrupted_rgb = lunar_rgb(
        corrupted_height_hr,
        roughness=corrupted_roughness_hr,
        corrupted_mask=mask_hr,
        fine_texture_strength=float(args.fine_texture_strength),
    )

    clean_path_out = output_dir / f"{args.prefix}_clean_rugged_surface.png"
    corrupted_path_out = output_dir / f"{args.prefix}_corrupted_smoothed_surface.png"
    metadata_path = output_dir / f"{args.prefix}_metadata.json"
    save_surface_map(clean_rgb, clean_path, start, goal, clean_path_out, dpi=args.dpi, source_shape=height.shape)
    save_surface_map(
        corrupted_rgb,
        corrupted_path,
        start,
        goal,
        corrupted_path_out,
        dpi=args.dpi,
        source_shape=height.shape,
    )
    metadata_path.write_text(
        json.dumps(
            {
                "layers": str(layers_path),
                "start": list(start),
                "goal": list(goal),
                "clean_route_cost": clean_route.cost,
                "corrupted_route_cost": corrupted_route.cost,
                "clean_cells": int(len(clean_route.path)),
                "corrupted_cells": int(len(corrupted_route.path)),
                "source_dem_shape": list(height.shape),
                "render_shape": list(clean_rgb.shape[:2]),
                "render_scale": render_scale,
                "fine_texture_strength": float(args.fine_texture_strength),
                "clean_image": str(clean_path_out),
                "corrupted_image": str(corrupted_path_out),
                "style": "real lunar DEM shaded relief; corrupted belief is locally smoothed",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(clean_path_out)
    print(corrupted_path_out)
    print(metadata_path)


if __name__ == "__main__":
    main()
