#!/usr/bin/env python
"""Generate clean and exaggerated corrupted lunar belief maps for the front figure.

Outputs are standalone images with no text, labels, legends, equations,
colorbars, captions, or watermarks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_problem_subfigures_from_maps import (  # noqa: E402
    cost_cmap,
    gaussian_blob,
    make_cost_fields,
    nearest_free,
    path_mask,
    smooth_path,
    terrain_texture,
    unit,
)


DEFAULT_LAYERS = PROJECT_ROOT / "maps" / "real_dem_tiles" / "lunar_npd_80_tile" / "real_map_layers.npz"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "frontpage_lunar_clean_corrupted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=Path, default=DEFAULT_LAYERS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=420)
    parser.add_argument("--prefix", type=str, default="frontpage_lunar")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_layers(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as raw:
        return {
            key: raw[key].astype(np.float32) if raw[key].dtype != np.bool_ else raw[key].astype(bool)
            for key in raw.files
        }


def corruption_artifacts(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = np.indices(shape, dtype=np.float32)
    band = np.exp(-((cols - rows - 2.0) ** 2) / (2.0 * 4.0**2)).astype(np.float32)
    blobs = (
        1.15 * gaussian_blob(shape, (30, 31), 8.0)
        + 1.25 * gaussian_blob(shape, (49, 50), 10.5)
        + 1.05 * gaussian_blob(shape, (66, 65), 7.5)
        + 0.72 * gaussian_blob(shape, (23, 55), 8.5)
        + 0.58 * gaussian_blob(shape, (59, 28), 7.5)
    )
    texture = gaussian_filter(np.sin(rows * 0.41 + cols * 0.23) + np.cos(rows * 0.17 - cols * 0.36), sigma=1.2)
    return unit(0.55 * band + 0.70 * blobs + 0.10 * unit(texture))


def make_frontpage_fields(
    data: dict[str, np.ndarray],
    clean_path: np.ndarray,
    shortcut_path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fields = make_cost_fields(data, clean_path, shortcut_path)
    obstacle = np.asarray(data["obstacle_mask"], dtype=bool)
    shape = fields["clean_belief"].shape
    shortcut = path_mask(shape, shortcut_path, sigma=4.3)
    artifacts = corruption_artifacts(shape)

    clean = unit(0.88 * fields["clean_belief"] + 0.12 * gaussian_filter(fields["true"], sigma=1.0))
    corrupted = unit(
        0.56 * fields["corrupted_belief"]
        + 0.66 * artifacts
        + 0.35 * fields["diagonal_danger"]
        - 0.88 * shortcut
    )
    corrupted = np.clip(corrupted - 0.30 * shortcut + 0.12 * artifacts, 0.0, 1.0)

    clean[obstacle] = np.maximum(clean[obstacle], 0.78)
    corrupted[obstacle] = np.maximum(corrupted[obstacle], 0.80)
    return clean.astype(np.float32), corrupted.astype(np.float32), artifacts.astype(np.float32)


def draw_path(ax: plt.Axes, path: np.ndarray, linewidth: float = 5.5) -> None:
    cols = path[:, 1]
    rows = path[:, 0]
    line = ax.plot(cols, rows, color="#1057ff", linewidth=linewidth, solid_capstyle="round", zorder=20)[0]
    line.set_path_effects([pe.Stroke(linewidth=linewidth + 3.4, foreground="white", alpha=0.96), pe.Normal()])
    ax.scatter(cols[::30], rows[::30], s=18, c="#1057ff", edgecolors="white", linewidths=0.55, zorder=21)


def draw_markers(ax: plt.Axes, start: tuple[int, int], goal: tuple[int, int]) -> None:
    ax.scatter(
        [start[1]],
        [start[0]],
        s=170,
        c="#27d746",
        edgecolors="#063d12",
        linewidths=1.7,
        zorder=24,
    )
    ax.scatter(
        [goal[1]],
        [goal[0]],
        s=170,
        c="#ff2525",
        edgecolors="#670000",
        linewidths=1.7,
        zorder=24,
    )


def draw_lunar_features(ax: plt.Axes, data: dict[str, np.ndarray], strong: bool) -> None:
    obstacle = np.asarray(data["obstacle_mask"], dtype=bool)
    if obstacle.any():
        overlay = np.ma.array(obstacle.astype(np.float32), mask=~obstacle)
        obstacle_cmap = LinearSegmentedColormap.from_list("front_obstacle", ["#242424", "#050505"])
        ax.imshow(overlay, cmap=obstacle_cmap, alpha=0.48 if not strong else 0.56, interpolation="nearest", zorder=6)


def draw_corruption_overlay(ax: plt.Axes, artifacts: np.ndarray, shape: tuple[int, int]) -> None:
    hot = np.ma.array(artifacts, mask=artifacts < 0.34)
    cmap = LinearSegmentedColormap.from_list(
        "attack_glow",
        [
            (1.0, 0.92, 0.10, 0.00),
            (1.0, 0.42, 0.00, 0.16),
            (0.95, 0.02, 0.02, 0.26),
            (0.35, 0.00, 0.00, 0.32),
        ],
        N=256,
    )
    ax.imshow(hot, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="bilinear", zorder=12)

    rng = np.random.default_rng(11)
    for _ in range(18):
        row = float(rng.integers(6, shape[0] - 7))
        col = float(rng.integers(6, shape[1] - 7))
        if artifacts[int(row), int(col)] < 0.38:
            continue
        size = float(rng.uniform(1.4, 3.1))
        ax.add_patch(
            Rectangle(
                (col - size / 2.0, row - size / 2.0),
                size,
                size,
                facecolor=(0.13, 0.00, 0.00, 0.35),
                edgecolor=(1.0, 0.75, 0.35, 0.18),
                linewidth=0.25,
                zorder=17,
            )
        )


def save_map(
    values: np.ndarray,
    data: dict[str, np.ndarray],
    path: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    output_path: Path,
    artifacts: np.ndarray | None,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 5.4), dpi=dpi)
    ax.imshow(values, cmap=cost_cmap(), vmin=0.0, vmax=1.0, interpolation="bilinear", zorder=1)
    ax.imshow(terrain_texture(data), cmap="gray", alpha=0.21, vmin=0.0, vmax=1.0, interpolation="bilinear", zorder=4)
    draw_lunar_features(ax, data, strong=artifacts is not None)
    if artifacts is not None:
        draw_corruption_overlay(ax, artifacts, values.shape)
    draw_path(ax, path)
    draw_markers(ax, start, goal)
    ax.set_xlim(-0.8, values.shape[1] - 0.2)
    ax.set_ylim(values.shape[0] - 0.2, -0.8)
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
    obstacle = np.asarray(data["obstacle_mask"], dtype=bool)
    start = nearest_free(obstacle, (12, 12))
    goal = nearest_free(obstacle, (73, 73))

    clean_path = smooth_path(
        [
            start,
            (15, 26),
            (28, 39),
            (42, 36),
            (53, 49),
            (60, 64),
            goal,
        ],
        samples=280,
    )
    shortcut_path = smooth_path(
        [
            start,
            (23, 23),
            (37, 37),
            (52, 52),
            (64, 64),
            goal,
        ],
        samples=280,
    )
    clean, corrupted, artifacts = make_frontpage_fields(data, clean_path, shortcut_path)
    clean_path_out = output_dir / f"{args.prefix}_clean_lunar_map.png"
    corrupted_path_out = output_dir / f"{args.prefix}_corrupted_lunar_map_exaggerated.png"
    metadata_path = output_dir / f"{args.prefix}_metadata.json"
    save_map(clean, data, clean_path, start, goal, clean_path_out, artifacts=None, dpi=args.dpi)
    save_map(corrupted, data, shortcut_path, start, goal, corrupted_path_out, artifacts=artifacts, dpi=args.dpi)
    metadata_path.write_text(
        json.dumps(
            {
                "layers": str(layers_path),
                "start": list(start),
                "goal": list(goal),
                "clean_image": str(clean_path_out),
                "corrupted_image": str(corrupted_path_out),
                "style": "standalone no-text frontpage lunar belief maps",
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
