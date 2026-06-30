#!/usr/bin/env python3
"""Draw a publication-style overview of benchmark environments.

The figure shows one representative clean/corrupted belief pair for each of
the three environment levels. The full benchmark still contains Easy, Medium,
and Hard variants within each level; this figure is intended to communicate
the environment families and the corrupted-belief setting without overcrowding
the page.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "levels" / "ppo_difficulty"
OUT_DIR = ROOT / "figures"
PNG_PATH = OUT_DIR / "environment_levels_overview.png"
PDF_PATH = OUT_DIR / "environment_levels_overview.pdf"


LEVEL_ROWS = [
    {
        "level": "Level 1",
        "family": "Synthetic controls",
        "config": "level1_hard.json",
        "variants": "40/60/80 grid variants",
        "cmap": ["#182133", "#3e4b62", "#8b96a6", "#eef0ef"],
        "hazard": "#E69F00",
        "accent": "#4D6C8F",
        "corridor_width": 0.050,
        "line_phase": 0.00,
    },
    {
        "level": "Level 2",
        "family": "Lunar / VIPER DEM",
        "config": "level2_hard.json",
        "variants": "40/80/80 grid variants",
        "cmap": ["#181a1d", "#4b4f55", "#9ea3a5", "#f1eee7"],
        "hazard": "#D55E00",
        "accent": "#737A83",
        "corridor_width": 0.040,
        "line_phase": 0.65,
    },
    {
        "level": "Level 3",
        "family": "Mars DTEED DEM",
        "config": "level3_hard.json",
        "variants": "40/100/100 grid variants",
        "cmap": ["#24130d", "#60301f", "#a85b34", "#d59661", "#f1d2a5"],
        "hazard": "#C94F2A",
        "accent": "#B5653D",
        "corridor_width": 0.046,
        "line_phase": 1.15,
    },
]

COLORS = {
    "ink": "#1f2933",
    "muted": "#607080",
    "edge": "#cbd5df",
    "panel_bg": "#f8fafc",
    "corrupt": "#00A9C8",
    "obstacle": "#111827",
    "start": "#FFD166",
    "goal": "#EF476F",
    "nominal": "#009E73",
    "corrupted_route": "#0072B2",
    "recovered": "#00A9C8",
}


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def make_cmap(name: str, colors: list[str]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, colors, N=256)


def normalize(arr: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.nanpercentile(arr[finite], [lower, upper])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def smooth_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    scale: int,
    blur_radius: float,
) -> np.ndarray:
    h, w = shape
    low = rng.random((max(4, h // scale), max(4, w // scale)))
    img = Image.fromarray((low * 255).astype(np.uint8), mode="L")
    img = img.resize((w, h), Image.Resampling.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.asarray(img, dtype=np.float32) / 255.0


def hillshade(height: np.ndarray) -> np.ndarray:
    z = normalize(height)
    gy, gx = np.gradient(z)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy) * 3.0)
    aspect = np.arctan2(-gx, gy)
    altitude = np.deg2rad(45.0)
    azimuth = np.deg2rad(315.0)
    shade = (
        np.sin(altitude) * np.sin(slope)
        + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    )
    return normalize(shade, 0.5, 99.5)


def gaussian_blob(
    x: np.ndarray,
    y: np.ndarray,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    theta: float = 0.0,
) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    xr = (x - cx) * ct + (y - cy) * st
    yr = -(x - cx) * st + (y - cy) * ct
    return np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))


def make_synthetic_environment(seed: int = 1111, size: int = 96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    x = xx / (size - 1)
    y = yy / (size - 1)

    terrain = (
        0.50 * smooth_noise(rng, (size, size), scale=8, blur_radius=4.0)
        + 0.31 * smooth_noise(rng, (size, size), scale=18, blur_radius=8.0)
        + 0.19 * smooth_noise(rng, (size, size), scale=42, blur_radius=17.0)
    )
    terrain += 0.045 * np.sin(4.0 * np.pi * (x + 0.18 * y))
    terrain += 0.040 * np.sin(6.4 * np.pi * (y - 0.12 * x) + 0.6)
    terrain = normalize(terrain)

    upper_mass = gaussian_blob(x, y, 0.46, 0.73, 0.27, 0.17, theta=-0.20)
    lower_mass = gaussian_blob(x, y, 0.56, 0.26, 0.30, 0.18, theta=0.18)
    obstacle = ((upper_mass > 0.56) | (lower_mass > 0.56)).astype(np.float32)

    hazard = 0.62 * gaussian_blob(x, y, 0.51, 0.50, 0.13, 0.16, theta=0.0)
    hazard += 0.28 * upper_mass + 0.28 * lower_mass
    hazard += 0.20 * gaussian_blob(x, y, 0.25, 0.55, 0.10, 0.25, theta=-0.30)
    hazard += 0.18 * gaussian_blob(x, y, 0.78, 0.47, 0.10, 0.23, theta=0.35)
    hazard += 0.10 * smooth_noise(rng, (size, size), scale=20, blur_radius=9.0)
    hazard = normalize(hazard, 2.0, 99.5)

    base = normalize(0.72 * terrain + 0.28 * hillshade(terrain), 0.5, 99.5)
    return base, hazard, obstacle


def load_config(filename: str) -> dict:
    with (CONFIG_DIR / filename).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_environment(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    config = load_config(str(row["config"]))
    if "map_source" not in config:
        base, hazard, obstacle = make_synthetic_environment(
            seed=int(config.get("fixed_map_seed", 1111)),
            size=96,
        )
        return base, hazard, obstacle, config

    data = np.load(ROOT / str(config["map_source"]), allow_pickle=True)
    height = data["height_norm"] if "height_norm" in data else normalize(data["height_map"])
    shade = hillshade(height)
    base = normalize(0.68 * normalize(height) + 0.32 * shade, 0.5, 99.5)

    if "layer_hazard" in data:
        hazard = normalize(data["layer_hazard"], 1.0, 99.5)
    elif "slope_layer" in data:
        hazard = normalize(data["slope_layer"], 1.0, 99.5)
    else:
        hazard = normalize(height, 1.0, 99.5)

    obstacle = np.asarray(data["obstacle_mask"], dtype=np.float32) if "obstacle_mask" in data else np.zeros_like(base)
    return base, hazard, obstacle, config


def make_corruption_mask(shape: tuple[int, int], width: float, phase: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = xx / (w - 1)
    y = yy / (h - 1)

    yline = 0.18 + 0.67 * x + 0.055 * np.sin(2.15 * np.pi * x + phase)
    dist = y - yline
    corridor = np.exp(-0.5 * (dist / width) ** 2)
    central_gate = gaussian_blob(x, y, 0.52, 0.51, 0.38, 0.34, theta=-0.15)
    start_goal_gate = np.clip((x + y - 0.18) / 0.18, 0, 1) * np.clip((1.78 - x - y) / 0.22, 0, 1)
    mask = corridor * np.clip(central_gate * 1.35, 0.0, 1.0) * start_goal_gate
    return normalize(mask, 0.0, 99.8)


def corrupted_hazard(hazard: np.ndarray, mask: np.ndarray, phase: float) -> np.ndarray:
    h, w = hazard.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = xx / (w - 1)
    y = yy / (h - 1)
    false_penalty = 0.18 * gaussian_blob(x, y, 0.25, 0.72, 0.13, 0.10, theta=0.20)
    false_penalty += 0.15 * gaussian_blob(x, y, 0.76, 0.30, 0.12, 0.12, theta=-0.30)
    false_penalty += 0.08 * gaussian_blob(x, y, 0.62, 0.77, 0.18, 0.08, theta=phase)
    visible = hazard * (1.0 - 0.82 * mask) + false_penalty * (1.0 - 0.35 * mask)
    return np.clip(visible, 0.0, 1.0)


def rgba_overlay(color: str, alpha: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*alpha.shape, 4), dtype=np.float32)
    rgba[..., :3] = mpl.colors.to_rgb(color)
    rgba[..., 3] = np.clip(alpha, 0.0, 1.0)
    return rgba


def risk_alpha(hazard: np.ndarray, base_threshold: float = 0.54) -> np.ndarray:
    hazard = normalize(hazard, 0.5, 99.5)
    threshold = max(base_threshold, float(np.nanquantile(hazard, 0.72)))
    alpha = np.clip((hazard - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    if np.mean(alpha > 0.04) < 0.025:
        threshold = float(np.nanquantile(hazard, 0.78))
        alpha = np.clip((hazard - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    return np.clip(alpha**0.72 * 0.56, 0.0, 0.58)


def resize_layer(arr: np.ndarray, factor: int, *, nearest: bool = False) -> np.ndarray:
    if factor <= 1:
        return arr
    arr = np.asarray(arr, dtype=np.float32)
    img = Image.fromarray(np.clip(arr, 0.0, 1.0))
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    img = img.resize((arr.shape[1] * factor, arr.shape[0] * factor), resample)
    return np.asarray(img, dtype=np.float32)


def upscale_layers(
    base: np.ndarray,
    hazard: np.ndarray,
    obstacle: np.ndarray,
    factor: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_hi = normalize(resize_layer(base, factor), 0.2, 99.8)
    hazard_hi = normalize(resize_layer(hazard, factor), 0.2, 99.8)
    obstacle_hi = resize_layer(obstacle, factor, nearest=True)
    return base_hi, hazard_hi, obstacle_hi


def catmull_rom(points: list[tuple[float, float]], samples_per_segment: int = 36) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return pts
    padded = np.vstack([pts[0], pts, pts[-1]])
    curve: list[np.ndarray] = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False, dtype=np.float32)
        t2 = t * t
        t3 = t2 * t
        segment = 0.5 * (
            (2.0 * p1)
            + (-p0 + p2) * t[:, None]
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2[:, None]
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3[:, None]
        )
        curve.append(segment)
    curve.append(pts[-1][None, :])
    return np.vstack(curve)


def normalized_to_pixels(points: list[tuple[float, float]], shape: tuple[int, int]) -> list[tuple[float, float]]:
    h, w = shape
    return [(x * (w - 1), y * (h - 1)) for x, y in points]


def route_points(kind: str, row_index: int) -> list[tuple[float, float]]:
    if kind == "corrupted":
        return [
            (0.14, 0.14),
            (0.25, 0.25),
            (0.39, 0.38),
            (0.54, 0.52),
            (0.69, 0.67),
            (0.86, 0.83),
        ]
    if row_index == 0:
        return [
            (0.14, 0.14),
            (0.24, 0.31),
            (0.34, 0.49),
            (0.37, 0.61),
            (0.50, 0.72),
            (0.68, 0.81),
            (0.86, 0.83),
        ]
    if row_index == 1:
        return [
            (0.14, 0.14),
            (0.27, 0.18),
            (0.45, 0.23),
            (0.63, 0.36),
            (0.74, 0.56),
            (0.86, 0.83),
        ]
    return [
        (0.14, 0.14),
        (0.26, 0.17),
        (0.43, 0.25),
        (0.61, 0.40),
        (0.74, 0.60),
        (0.86, 0.83),
    ]


def draw_route(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    shape: tuple[int, int],
    color: str,
    *,
    linewidth: float = 2.6,
    alpha: float = 0.98,
    zorder: int = 14,
) -> None:
    pixel_points = normalized_to_pixels(points, shape)
    curve = catmull_rom(pixel_points, samples_per_segment=44)
    x, y = curve[:, 0], curve[:, 1]
    ax.plot(x, y, color="white", lw=linewidth + 3.2, alpha=0.88, solid_capstyle="round", zorder=zorder)
    ax.plot(x, y, color=color, lw=linewidth, alpha=alpha, solid_capstyle="round", zorder=zorder + 1)

    start_idx = max(0, len(curve) - 10)
    arrow_bg = FancyArrowPatch(
        tuple(curve[start_idx]),
        tuple(curve[-1]),
        arrowstyle="-|>",
        mutation_scale=13.0,
        lw=linewidth + 3.4,
        color="white",
        alpha=0.88,
        zorder=zorder + 2,
        shrinkA=0,
        shrinkB=4,
    )
    arrow_fg = FancyArrowPatch(
        tuple(curve[start_idx]),
        tuple(curve[-1]),
        arrowstyle="-|>",
        mutation_scale=11.0,
        lw=linewidth,
        color=color,
        alpha=alpha,
        zorder=zorder + 3,
        shrinkA=0,
        shrinkB=4,
    )
    ax.add_patch(arrow_bg)
    ax.add_patch(arrow_fg)


def draw_illustrative_paths(ax: plt.Axes, shape: tuple[int, int], *, row_index: int, corrupted: bool) -> None:
    draw_route(
        ax,
        route_points("nominal", row_index),
        shape,
        COLORS["nominal"],
        linewidth=2.6,
        alpha=0.96,
        zorder=13,
    )
    if corrupted:
        draw_route(
            ax,
            route_points("corrupted", row_index),
            shape,
            COLORS["corrupted_route"],
            linewidth=2.8,
            alpha=0.98,
            zorder=17,
        )


def draw_task_markers(ax: plt.Axes, shape: tuple[int, int]) -> None:
    h, w = shape
    start = (0.14 * (w - 1), 0.14 * (h - 1))
    goal = (0.86 * (w - 1), 0.83 * (h - 1))
    ax.scatter(
        [start[0], goal[0]],
        [start[1], goal[1]],
        s=[120, 180],
        c=["white", "white"],
        marker="o",
        edgecolors="white",
        linewidths=0.0,
        alpha=0.84,
        zorder=24,
    )
    ax.scatter(
        [start[0]],
        [start[1]],
        s=70,
        c=[COLORS["start"]],
        marker="o",
        edgecolors=COLORS["ink"],
        linewidths=1.3,
        zorder=25,
    )
    ax.scatter(
        [goal[0]],
        [goal[1]],
        s=110,
        c=[COLORS["goal"]],
        marker="*",
        edgecolors="white",
        linewidths=1.0,
        zorder=25,
    )


def draw_panel(
    ax: plt.Axes,
    *,
    base: np.ndarray,
    hazard: np.ndarray,
    obstacle: np.ndarray,
    row: dict,
    row_index: int,
    corrupted: bool,
) -> None:
    cmap = make_cmap(f"{row['level']}_{'corrupt' if corrupted else 'clean'}", list(row["cmap"]))
    mask = make_corruption_mask(base.shape, float(row["corridor_width"]), float(row["line_phase"]))
    visible_hazard = corrupted_hazard(hazard, mask, float(row["line_phase"])) if corrupted else hazard

    ax.imshow(base, cmap=cmap, vmin=0, vmax=1, origin="lower", interpolation="lanczos")
    ax.imshow(
        rgba_overlay(str(row["hazard"]), risk_alpha(visible_hazard)),
        origin="lower",
        interpolation="lanczos",
    )

    if corrupted:
        outer_glow = np.clip(mask**0.45 * 0.22, 0.0, 0.25)
        inner_glow = np.clip(mask**1.25 * 0.48, 0.0, 0.52)
        ax.imshow(
            rgba_overlay("#B7F5FF", outer_glow),
            origin="lower",
            interpolation="lanczos",
        )
        ax.imshow(
            rgba_overlay(COLORS["corrupt"], inner_glow),
            origin="lower",
            interpolation="lanczos",
        )

    obstacle_alpha = np.clip(obstacle, 0.0, 1.0) * 0.78
    if np.any(obstacle_alpha > 0):
        ax.imshow(
            rgba_overlay(COLORS["obstacle"], obstacle_alpha),
            origin="lower",
            interpolation="nearest",
        )

    draw_illustrative_paths(ax, base.shape, row_index=row_index, corrupted=corrupted)
    draw_task_markers(ax, base.shape)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(COLORS["edge"])
        spine.set_linewidth(0.85)


def add_row_label(fig: plt.Figure, y: float, row: dict, config: dict, shape: tuple[int, int]) -> None:
    fig.add_artist(
        Rectangle(
            (0.066, y - 0.092),
            0.008,
            0.178,
            transform=fig.transFigure,
            facecolor=str(row["accent"]),
            edgecolor="none",
            alpha=0.95,
        )
    )
    grid = int(config.get("map_size", shape[0]))
    distance = float(config.get("min_distance_ratio", 0.0))
    rep = f"shown: {grid} x {grid}"
    distance_text = rf"$d_{{min}}\geq {distance:.2f}D$" if distance > 0 else ""

    fig.text(0.082, y + 0.052, str(row["level"]), ha="left", va="center", fontsize=12.2, fontweight="bold", color=COLORS["ink"])
    fig.text(0.082, y + 0.024, str(row["family"]), ha="left", va="center", fontsize=9.0, color=COLORS["ink"])
    fig.text(0.082, y - 0.003, rep, ha="left", va="center", fontsize=7.7, color=COLORS["muted"])
    fig.text(0.082, y - 0.029, distance_text, ha="left", va="center", fontsize=7.7, color=COLORS["muted"])
    fig.text(0.082, y - 0.056, str(row["variants"]), ha="left", va="center", fontsize=7.7, color=COLORS["muted"])


def main() -> None:
    setup_matplotlib()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.6, 8.65), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        left=0.295,
        right=0.985,
        bottom=0.118,
        top=0.885,
        wspace=0.030,
        hspace=0.102,
    )

    fig.text(
        0.295,
        0.940,
        "Clean and Corrupted Terrain Beliefs",
        ha="left",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.295,
        0.912,
        "Representative maps from the three benchmark environment levels.",
        ha="left",
        va="center",
        fontsize=8.8,
        color=COLORS["muted"],
    )

    header_y = 0.886
    fig.text(0.452, header_y, "Clean terrain belief", ha="center", va="bottom", fontsize=10.0, fontweight="bold", color=COLORS["ink"])
    fig.text(0.805, header_y, "Corrupted planner-visible belief", ha="center", va="bottom", fontsize=10.0, fontweight="bold", color=COLORS["ink"])

    for idx, row in enumerate(LEVEL_ROWS):
        base, hazard, obstacle, config = load_environment(row)
        original_shape = base.shape
        base, hazard, obstacle = upscale_layers(base, hazard, obstacle, factor=4)
        clean_ax = fig.add_subplot(gs[idx, 0])
        corrupt_ax = fig.add_subplot(gs[idx, 1])
        draw_panel(clean_ax, base=base, hazard=hazard, obstacle=obstacle, row=row, row_index=idx, corrupted=False)
        draw_panel(corrupt_ax, base=base, hazard=hazard, obstacle=obstacle, row=row, row_index=idx, corrupted=True)

        pos = clean_ax.get_position()
        center_y = pos.y0 + pos.height / 2.0
        add_row_label(fig, center_y, row, config, original_shape)

    legend_handles = [
        Patch(facecolor="#b8b8b8", edgecolor=COLORS["edge"], label="terrain relief"),
        Patch(facecolor=LEVEL_ROWS[1]["hazard"], edgecolor="none", alpha=0.56, label="high-risk terrain layer"),
        Patch(facecolor=COLORS["corrupt"], edgecolor="none", alpha=0.46, label="corrupted low-cost corridor"),
        Patch(facecolor=COLORS["obstacle"], edgecolor="none", alpha=0.78, label="obstacle"),
        Patch(facecolor=COLORS["nominal"], edgecolor="none", alpha=0.90, label="nominal / recovered path"),
        Patch(facecolor=COLORS["corrupted_route"], edgecolor="none", alpha=0.90, label="corrupted path"),
    ]
    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.640, 0.042),
        ncol=3,
        frameon=True,
        fontsize=7.45,
        handlelength=1.15,
        columnspacing=0.92,
        borderpad=0.50,
        handletextpad=0.38,
    )
    leg.get_frame().set_facecolor((1, 1, 1, 0.94))
    leg.get_frame().set_edgecolor(COLORS["edge"])
    leg.get_frame().set_linewidth(0.8)

    fig.savefig(PNG_PATH, dpi=600)
    fig.savefig(PDF_PATH)
    plt.close(fig)

    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    main()
