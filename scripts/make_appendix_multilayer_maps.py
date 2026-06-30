#!/usr/bin/env python
"""Create an appendix figure showing representative multi-layer maps."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maps.map_generator import generate_costmap  # noqa: E402


OUTPUT = Path(r"<USER_HOME>\Downloads\appendix_multilayer_maps.png")
OUTPUT_JPG = Path(r"<USER_HOME>\Downloads\appendix_multilayer_maps.jpg")


def cost_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "planetary_cost",
        ["#1f3c88", "#2378b7", "#36b779", "#f2df5b", "#f08a24", "#c0392b"],
        N=256,
    )


def normalize(values: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo = float(np.nanpercentile(arr[finite], lo_pct))
    hi = float(np.nanpercentile(arr[finite], hi_pct))
    if hi - lo <= 1e-8:
        out = np.zeros_like(arr, dtype=np.float32)
    else:
        out = (arr - lo) / (hi - lo)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def hillshade(height: np.ndarray) -> np.ndarray:
    height_norm = normalize(height)
    dy, dx = np.gradient(height_norm)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(38.0)
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    shaded = (
        np.sin(altitude) * np.sin(slope)
        + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    )
    return normalize(shaded, 1.0, 99.0)


def surface_rgb(height: np.ndarray, palette: str) -> np.ndarray:
    shade = hillshade(height)
    height_norm = normalize(height)
    if palette == "mars":
        low = np.array([0.38, 0.15, 0.08], dtype=np.float32)
        mid = np.array([0.72, 0.33, 0.16], dtype=np.float32)
        high = np.array([0.95, 0.64, 0.36], dtype=np.float32)
        base = (1.0 - height_norm[..., None]) * low + height_norm[..., None] * high
        base = 0.65 * base + 0.35 * mid
        rgb = base * (0.52 + 0.72 * shade[..., None])
    elif palette == "lunar":
        base = np.array([0.55, 0.55, 0.53], dtype=np.float32)
        rgb = base[None, None, :] * (0.45 + 0.78 * shade[..., None])
    else:
        base = np.array([0.50, 0.50, 0.48], dtype=np.float32)
        rgb = base[None, None, :] * (0.48 + 0.74 * shade[..., None])
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def load_npz_layers(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path)
    uncertainty = np.mean(
        [
            np.asarray(raw[f"uncertainty_{name}"], dtype=np.float32)
            for name in ("energy", "hazard", "communication", "illumination")
        ],
        axis=0,
    )
    height = normalize(np.asarray(raw["height_norm"], dtype=np.float32))
    return {
        "terrain": height,
        "energy": np.asarray(raw["layer_energy"], dtype=np.float32),
        "hazard": np.asarray(raw["layer_hazard"], dtype=np.float32),
        "communication": np.asarray(raw["layer_communication"], dtype=np.float32),
        "illumination": np.asarray(raw["layer_illumination"], dtype=np.float32),
        "uncertainty": np.asarray(uncertainty, dtype=np.float32),
        "obstacle": np.asarray(raw["obstacle_mask"], dtype=bool).astype(np.float32),
    }


def synthetic_layers() -> dict[str, np.ndarray]:
    costmap = generate_costmap(
        map_size=60,
        rng=np.random.default_rng(909),
        scenario="uncertain_hazard_corridor",
        min_start_goal_distance_ratio=0.68,
    )
    uncertainty = np.mean(
        [
            np.asarray(costmap.uncertainty_layers[name], dtype=np.float32)
            for name in ("energy", "hazard", "communication", "illumination")
        ],
        axis=0,
    )
    height = normalize(np.asarray(costmap.height_map, dtype=np.float32))
    return {
        "terrain": height,
        "energy": np.asarray(costmap.layers["energy"], dtype=np.float32),
        "hazard": np.asarray(costmap.layers["hazard"], dtype=np.float32),
        "communication": np.asarray(costmap.layers["communication"], dtype=np.float32),
        "illumination": np.asarray(costmap.layers["illumination"], dtype=np.float32),
        "uncertainty": np.asarray(uncertainty, dtype=np.float32),
        "obstacle": np.asarray(costmap.obstacle_mask, dtype=bool).astype(np.float32),
    }


def draw() -> None:
    rows = [
        (
            "Level 1\nSynthetic",
            {**synthetic_layers(), "surface_palette": "synthetic"},
        ),
        (
            "Level 2\nLunar DEM",
            {
                **load_npz_layers(PROJECT_ROOT / "maps" / "real_dem_tiles" / "lunar_npd_80_tile" / "real_map_layers.npz"),
                "surface_palette": "lunar",
            },
        ),
        (
            "Level 3\nMars DTM",
            {
                **load_npz_layers(
                    PROJECT_ROOT
                    / "maps"
                    / "real_dem_tiles"
                    / "marsdteed_ridge_pgda_500_level3_tile"
                    / "real_map_layers.npz"
                ),
                "surface_palette": "mars",
            },
        ),
    ]
    columns = [
        ("surface", "Surface"),
        ("terrain", "DEM"),
        ("energy", "Energy"),
        ("hazard", "Hazard"),
        ("communication", "Comm."),
        ("illumination", "Illum."),
        ("uncertainty", "Uncertainty"),
        ("obstacle", "Obstacle"),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.linewidth": 0.55,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=len(columns),
        figsize=(12.1, 4.9),
        constrained_layout=True,
    )
    cmap = cost_cmap()
    for r, (row_label, data) in enumerate(rows):
        for c, (key, col_label) in enumerate(columns):
            ax = axes[r, c]
            if key == "surface":
                ax.imshow(
                    surface_rgb(np.asarray(data["terrain"], dtype=np.float32), str(data.get("surface_palette", "lunar"))),
                    origin="upper",
                    interpolation="nearest",
                )
            elif key == "terrain":
                field = data[key]
                ax.imshow(normalize(field), cmap="gray", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
            elif key == "obstacle":
                field = data[key]
                ax.imshow(field, cmap="gray_r", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
            else:
                field = data[key]
                ax.imshow(normalize(field), cmap=cmap, vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
            if r == 0:
                ax.set_title(col_label, fontsize=8.5, pad=3)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=8.5, rotation=0, labelpad=31, va="center", ha="right")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("0.35")
                spine.set_linewidth(0.45)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=450)
    fig.savefig(OUTPUT_JPG, dpi=450, pil_kwargs={"quality": 95})
    plt.close(fig)
    print(OUTPUT)
    print(OUTPUT_JPG)


if __name__ == "__main__":
    draw()

