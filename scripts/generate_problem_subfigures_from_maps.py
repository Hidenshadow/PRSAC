#!/usr/bin/env python
"""Generate standalone problem-definition subfigures from local terrain maps.

The outputs intentionally contain no text, labels, numbers, legends, equations,
colorbars, captions, or watermarks so they can be assembled and annotated later.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Ellipse, Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYERS = (
    PROJECT_ROOT
    / "maps"
    / "real_dem_tiles"
    / "marsdteed_ridge_pgda_500_level3_tile"
    / "real_map_layers.npz"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "problem_subfigures_from_maps"

OBJECTIVE_KEYS = (
    "layer_distance",
    "layer_energy",
    "layer_hazard",
    "layer_communication",
    "layer_illumination",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=Path, default=DEFAULT_LAYERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def unit(values: np.ndarray, lo: float | None = None, hi: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    low = float(np.nanpercentile(arr[finite], 1.0)) if lo is None else float(lo)
    high = float(np.nanpercentile(arr[finite], 99.0)) if hi is None else float(hi)
    if high <= low:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr - low) / (high - low)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def nearest_free(mask: np.ndarray, target: tuple[int, int]) -> tuple[int, int]:
    obstacle = np.asarray(mask, dtype=bool)
    row, col = int(target[0]), int(target[1])
    row = int(np.clip(row, 0, obstacle.shape[0] - 1))
    col = int(np.clip(col, 0, obstacle.shape[1] - 1))
    if not obstacle[row, col]:
        return row, col
    free = np.argwhere(~obstacle)
    distances = np.sum((free - np.array([[row, col]])) ** 2, axis=1)
    best = free[int(np.argmin(distances))]
    return int(best[0]), int(best[1])


def smooth_path(points: list[tuple[float, float]], samples: int = 260) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    degree = min(3, len(points) - 1)
    tck, _ = splprep([pts[:, 1], pts[:, 0]], s=1.0, k=degree)
    u = np.linspace(0.0, 1.0, samples)
    cols, rows = splev(u, tck)
    return np.column_stack([np.asarray(rows), np.asarray(cols)]).astype(np.float32)


def path_mask(shape: tuple[int, int], path: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    for row, col in path:
        rr = int(np.clip(round(float(row)), 0, shape[0] - 1))
        cc = int(np.clip(round(float(col)), 0, shape[1] - 1))
        mask[rr, cc] = 1.0
    return unit(gaussian_filter(mask, sigma=sigma), lo=0.0, hi=None)


def gaussian_blob(shape: tuple[int, int], center: tuple[float, float], sigma: float) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float32)
    return np.exp(-(((rows - center[0]) ** 2 + (cols - center[1]) ** 2) / (2.0 * sigma**2))).astype(np.float32)


def make_cost_fields(data: dict[str, np.ndarray], clean_path: np.ndarray, shortcut_path: np.ndarray) -> dict[str, np.ndarray]:
    shape = data["height_norm"].shape
    energy = np.asarray(data["layer_energy"], dtype=np.float32)
    hazard = np.asarray(data["layer_hazard"], dtype=np.float32)
    communication = np.asarray(data["layer_communication"], dtype=np.float32)
    illumination = np.asarray(data["layer_illumination"], dtype=np.float32)
    slope = np.asarray(data["slope_layer"], dtype=np.float32)
    roughness = np.asarray(data["roughness_layer"], dtype=np.float32)
    obstacle = np.asarray(data["obstacle_mask"], dtype=bool)

    terrain = unit(
        0.25 * energy
        + 0.38 * hazard
        + 0.13 * communication
        + 0.10 * illumination
        + 0.10 * slope
        + 0.04 * roughness
    )
    terrain = unit(0.72 * terrain + 0.28 * gaussian_filter(terrain, sigma=1.3))

    diagonal_danger = np.zeros(shape, dtype=np.float32)
    for center, sigma in (((30, 30), 8.5), ((50, 50), 10.0), ((68, 67), 8.0)):
        diagonal_danger += gaussian_blob(shape, center, sigma)
    diagonal_danger = unit(diagonal_danger)

    clean_corridor = path_mask(shape, clean_path, sigma=4.0)
    shortcut_corridor = path_mask(shape, shortcut_path, sigma=3.4)

    true_cost = unit(0.70 * terrain + 0.55 * diagonal_danger - 0.30 * clean_corridor)
    true_cost = np.clip(true_cost + obstacle.astype(np.float32) * 0.12, 0.0, 1.0)

    clean_belief = unit(0.88 * true_cost + 0.12 * gaussian_filter(terrain, sigma=2.0) - 0.20 * clean_corridor)
    corrupted_belief = unit(
        0.70 * terrain
        + 0.18 * diagonal_danger
        - 0.48 * shortcut_corridor
        + 0.14 * clean_corridor
    )
    recovered_belief = unit(0.86 * corrupted_belief + 0.10 * diagonal_danger)

    for field in (true_cost, clean_belief, corrupted_belief, recovered_belief):
        field[obstacle] = np.clip(np.maximum(field[obstacle], 0.72), 0.0, 1.0)

    return {
        "true": true_cost.astype(np.float32),
        "clean_belief": clean_belief.astype(np.float32),
        "corrupted_belief": corrupted_belief.astype(np.float32),
        "recovered_belief": recovered_belief.astype(np.float32),
        "diagonal_danger": diagonal_danger.astype(np.float32),
    }


def cost_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "terrain_cost",
        [
            "#123d84",
            "#1567b1",
            "#1fa187",
            "#86c65a",
            "#ffe082",
            "#f28e2b",
            "#d7301f",
            "#7f0000",
        ],
        N=256,
    )


def terrain_texture(data: dict[str, np.ndarray]) -> np.ndarray:
    height = unit(data["height_norm"])
    gy, gx = np.gradient(gaussian_filter(height, sigma=1.0))
    shade = 0.55 + 0.40 * unit(-0.85 * gx - 0.45 * gy)
    return np.clip(shade, 0.0, 1.0)


def feature_circles(data: dict[str, np.ndarray], count: int = 44) -> list[tuple[float, float, float, float]]:
    obstacle = np.asarray(data["obstacle_mask"], dtype=bool)
    rough = unit(data["roughness_layer"])
    slope = unit(data["slope_layer"])
    score = 0.60 * rough + 0.40 * slope + 0.35 * obstacle.astype(np.float32)
    flat = score.reshape(-1)
    order = np.argsort(flat)[::-1]
    rng = np.random.default_rng(17)
    circles: list[tuple[float, float, float, float]] = []
    occupied: list[tuple[int, int]] = []
    shape = score.shape
    for index in order:
        row = int(index // shape[1])
        col = int(index % shape[1])
        if row < 4 or col < 4 or row > shape[0] - 5 or col > shape[1] - 5:
            continue
        if any((row - r) ** 2 + (col - c) ** 2 < 55 for r, c in occupied):
            continue
        radius = float(rng.uniform(0.9, 2.4) * (1.0 + 0.8 * flat[index]))
        alpha = float(rng.uniform(0.18, 0.34))
        circles.append((float(row), float(col), radius, alpha))
        occupied.append((row, col))
        if len(circles) >= count:
            break
    return circles


def draw_map(
    values: np.ndarray,
    data: dict[str, np.ndarray],
    path: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    output_path: Path,
    circles: list[tuple[float, float, float, float]],
    dashed_blobs: bool = False,
    dpi: int = 240,
) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 5.0), dpi=dpi)
    ax.imshow(values, cmap=cost_cmap(), vmin=0.0, vmax=1.0, interpolation="bilinear")
    ax.imshow(terrain_texture(data), cmap="gray", alpha=0.16, vmin=0.0, vmax=1.0, interpolation="bilinear")

    obstacle = np.asarray(data["obstacle_mask"], dtype=bool)
    if obstacle.any():
        overlay = np.ma.array(obstacle.astype(np.float32), mask=~obstacle)
        ax.imshow(overlay, cmap=LinearSegmentedColormap.from_list("obstacle", ["#2d2d2d", "#101010"]), alpha=0.52)

    for row, col, radius, alpha in circles:
        ax.add_patch(
            Circle(
                (col, row),
                radius,
                facecolor=(0.06, 0.06, 0.06, alpha),
                edgecolor=(1.0, 1.0, 1.0, 0.16),
                linewidth=0.35,
                zorder=4,
            )
        )

    if dashed_blobs:
        for row, col, width, height, angle in ((31, 31, 24, 16, -42), (50, 50, 28, 18, -40), (68, 67, 23, 15, -38)):
            ax.add_patch(
                Ellipse(
                    (col, row),
                    width,
                    height,
                    angle=angle,
                    fill=False,
                    edgecolor=(0.0, 0.0, 0.0, 0.78),
                    linewidth=2.0,
                    linestyle=(0, (5, 4)),
                    zorder=7,
                )
            )

    draw_path(ax, path)
    draw_markers(ax, start, goal)
    ax.set_xlim(-0.5, values.shape[1] - 0.5)
    ax.set_ylim(values.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def draw_path(ax: plt.Axes, path: np.ndarray, linewidth: float = 5.0) -> None:
    cols = path[:, 1]
    rows = path[:, 0]
    line = ax.plot(cols, rows, color="#1358ff", linewidth=linewidth, solid_capstyle="round", zorder=8)[0]
    line.set_path_effects([pe.Stroke(linewidth=linewidth + 3.0, foreground="white", alpha=0.95), pe.Normal()])
    ax.scatter(cols[::28], rows[::28], s=18, c="#1358ff", edgecolors="white", linewidths=0.65, zorder=9)


def draw_markers(ax: plt.Axes, start: tuple[int, int], goal: tuple[int, int]) -> None:
    ax.scatter(
        [start[1]],
        [start[0]],
        s=150,
        c="#2ecc40",
        edgecolors="#084b13",
        linewidths=1.6,
        zorder=10,
    )
    ax.scatter(
        [goal[1]],
        [goal[0]],
        s=150,
        c="#ff2b2b",
        edgecolors="#6e0000",
        linewidths=1.6,
        zorder=10,
    )


def draw_multilayer(
    data: dict[str, np.ndarray],
    start: tuple[int, int],
    goal: tuple[int, int],
    path: np.ndarray,
    output_path: Path,
    dpi: int,
) -> None:
    height = unit(data["height_norm"])
    obstacle = np.asarray(data["obstacle_mask"], dtype=bool)
    row_slice = slice(0, height.shape[0], 2)
    col_slice = slice(0, height.shape[1], 2)
    h_small = height[row_slice, col_slice]
    rows, cols = np.indices(h_small.shape, dtype=np.float32)
    x = cols * 2.0
    y = rows * 2.0
    z = 4.5 * h_small

    fig = plt.figure(figsize=(6.0, 7.0), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    base_colors = plt.get_cmap("gray")(0.22 + 0.58 * h_small)
    obs_small = obstacle[row_slice, col_slice]
    base_colors[obs_small, :3] = 0.04
    base_colors[..., 3] = 1.0
    ax.plot_surface(x, y, z, facecolors=base_colors, rstride=1, cstride=1, linewidth=0.0, antialiased=False, shade=False)

    layer_values = [
        unit(data["layer_distance"]),
        unit(data["layer_energy"]),
        unit(data["layer_hazard"]),
        unit(data["layer_communication"]),
        unit(data["layer_illumination"]),
        unit(
            0.25 * data["uncertainty_distance"]
            + 0.25 * data["uncertainty_energy"]
            + 0.25 * data["uncertainty_hazard"]
            + 0.125 * data["uncertainty_communication"]
            + 0.125 * data["uncertainty_illumination"]
        ),
        obstacle.astype(np.float32),
    ]
    z0 = 12.0
    for idx, layer in enumerate(layer_values):
        small = layer[row_slice, col_slice]
        layer_z = np.full_like(h_small, z0 + idx * 5.1, dtype=np.float32)
        colors = cost_cmap()(small)
        if idx == len(layer_values) - 1:
            colors = plt.get_cmap("gray_r")(small)
        colors[..., 3] = 0.46
        ax.plot_surface(
            x,
            y,
            layer_z,
            facecolors=colors,
            rstride=1,
            cstride=1,
            linewidth=0.15,
            edgecolor=(1.0, 1.0, 1.0, 0.12),
            antialiased=True,
            shade=False,
        )

    rows_path = path[:, 0]
    cols_path = path[:, 1]
    path_z = 4.5 * height[np.clip(rows_path.astype(int), 0, height.shape[0] - 1), np.clip(cols_path.astype(int), 0, height.shape[1] - 1)] + 0.9
    line = ax.plot(cols_path, rows_path, path_z, color="#1358ff", linewidth=5.0, zorder=30)[0]
    line.set_path_effects([pe.Stroke(linewidth=8.0, foreground="white", alpha=0.95), pe.Normal()])
    ax.scatter([start[1]], [start[0]], [path_z[0] + 0.6], s=72, color="#2ecc40", edgecolor="#084b13", linewidth=1.0, depthshade=False)
    ax.scatter([goal[1]], [goal[0]], [path_z[-1] + 0.6], s=72, color="#ff2b2b", edgecolor="#6e0000", linewidth=1.0, depthshade=False)

    draw_rover_3d(ax, start, height)
    ax.view_init(elev=29, azim=-63)
    ax.set_xlim(0, height.shape[1])
    ax.set_ylim(height.shape[0], 0)
    ax.set_zlim(-1, z0 + len(layer_values) * 5.1 + 3)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def draw_rover_3d(ax, start: tuple[int, int], height: np.ndarray) -> None:
    row = float(start[0] + 8)
    col = float(start[1] + 7)
    z = float(4.5 * height[int(np.clip(row, 0, height.shape[0] - 1)), int(np.clip(col, 0, height.shape[1] - 1))] + 0.8)
    ax.scatter([col], [row], [z], s=120, marker="s", color="#7f7f7f", edgecolor="#202020", linewidth=0.7, depthshade=False)
    for dc, dr in ((-2.4, -1.6), (2.4, -1.6), (-2.4, 1.6), (2.4, 1.6)):
        ax.scatter([col + dc], [row + dr], [z - 0.35], s=42, marker="o", color="#2b2b2b", edgecolor="#101010", linewidth=0.5, depthshade=False)
    ax.plot([col, col + 2.0], [row, row - 3.2], [z + 0.5, z + 2.2], color="#555555", linewidth=1.2)
    ax.scatter([col + 2.0], [row - 3.2], [z + 2.2], s=32, color="#9e9e9e", edgecolor="#303030", linewidth=0.4, depthshade=False)


def main() -> None:
    args = parse_args()
    layers_path = resolve(args.layers)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(layers_path) as raw:
        data = {key: raw[key].astype(np.float32) if raw[key].dtype != np.bool_ else raw[key].astype(bool) for key in raw.files}

    obstacle = np.asarray(data["obstacle_mask"], dtype=bool)
    start = nearest_free(obstacle, (12, 12))
    goal = nearest_free(obstacle, (87, 87))

    clean_path = smooth_path(
        [
            start,
            (17, 30),
            (31, 43),
            (48, 38),
            (61, 59),
            (73, 72),
            goal,
        ]
    )
    shortcut_path = smooth_path(
        [
            start,
            (27, 25),
            (41, 41),
            (56, 56),
            (72, 70),
            goal,
        ]
    )
    recovered_path = smooth_path(
        [
            start,
            (18, 27),
            (33, 45),
            (49, 42),
            (64, 63),
            (77, 75),
            goal,
        ]
    )

    fields = make_cost_fields(data, clean_path, shortcut_path)
    circles = feature_circles(data)

    draw_multilayer(data, start, goal, clean_path, output_dir / "subfigure_01_multilayer_map_optimization.png", args.dpi)
    draw_map(fields["clean_belief"], data, clean_path, start, goal, output_dir / "subfigure_02_clean_belief_path.png", circles, dpi=args.dpi)
    draw_map(fields["corrupted_belief"], data, shortcut_path, start, goal, output_dir / "subfigure_03_corrupted_belief_shortcut_path.png", circles, dpi=args.dpi)
    draw_map(fields["true"], data, clean_path, start, goal, output_dir / "subfigure_04_true_eval_clean_path.png", circles, dpi=args.dpi)
    draw_map(fields["true"], data, shortcut_path, start, goal, output_dir / "subfigure_05_true_eval_corrupted_shortcut_path.png", circles, dashed_blobs=True, dpi=args.dpi)
    draw_map(fields["recovered_belief"], data, recovered_path, start, goal, output_dir / "subfigure_06_recovered_path_on_corrupted_belief.png", circles, dpi=args.dpi)
    draw_map(fields["true"], data, recovered_path, start, goal, output_dir / "subfigure_07_true_eval_recovered_path.png", circles, dpi=args.dpi)

    print(output_dir)
    for path in sorted(output_dir.glob("subfigure_*.png")):
        print(path)


if __name__ == "__main__":
    main()
