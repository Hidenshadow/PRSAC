#!/usr/bin/env python3
"""Draw a publication-style route recovery overview figure.

The figure illustrates three global route outcomes on the same terrain:
nominal planning on a clean belief, corrupted planning through a misleading
high-cost corridor, and recovery through RL + MFA*.
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Polygon
import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
PNG_PATH = OUT_DIR / "route_recovery_overview.png"
PDF_PATH = OUT_DIR / "route_recovery_overview.pdf"

BACKGROUND_CANDIDATES = [
    ROOT / "assets" / "terrain_background.png",
    ROOT / "assets" / "rover_terrain_background.png",
    ROOT / "figures" / "terrain_background.png",
    Path("/mnt/data/exploring_the_barren_landscape_of_mars.png"),
    Path("/mnt/data/rover_path_comparison_on_rugged_terrain.png"),
]

NOMINAL_COLOR = "#2ca02c"
CORRUPTED_COLOR = "#1f77ff"
RECOVERED_COLOR = "#00d5e8"
HIGH_COST_COLOR = "#e66a2c"


def _blurred_noise(
    rng: np.random.Generator,
    shape: tuple[int, int],
    scale: int,
    blur_radius: float,
) -> np.ndarray:
    """Generate smooth value noise using PIL resize and Gaussian blur."""
    h, w = shape
    low_h = max(4, h // scale)
    low_w = max(4, w // scale)
    low = rng.random((low_h, low_w))
    img = Image.fromarray((low * 255).astype(np.uint8), mode="L")
    img = img.resize((w, h), Image.Resampling.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def make_procedural_terrain(width: int = 2200, height: int = 1238) -> np.ndarray:
    """Create a grayscale lunar/Mars-like shaded terrain background."""
    rng = np.random.default_rng(7)

    n1 = _blurred_noise(rng, (height, width), scale=10, blur_radius=7)
    n2 = _blurred_noise(rng, (height, width), scale=24, blur_radius=14)
    n3 = _blurred_noise(rng, (height, width), scale=54, blur_radius=26)
    n4 = _blurred_noise(rng, (height, width), scale=110, blur_radius=42)
    height_field = 0.42 * n1 + 0.30 * n2 + 0.20 * n3 + 0.08 * n4

    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / (width - 1)
    y = yy / (height - 1)

    # Add broad ridges and crater-like depressions to make the scene read as
    # planetary terrain without relying on a photographic source.
    ridge = 0.055 * np.sin(2.4 * math.pi * (x + 0.17 * y))
    ridge += 0.035 * np.sin(5.1 * math.pi * (x - 0.33 * y) + 0.8)
    height_field += ridge

    craters = [
        (0.19, 0.68, 0.105, -0.070),
        (0.73, 0.29, 0.085, -0.060),
        (0.78, 0.71, 0.125, -0.050),
        (0.37, 0.23, 0.075, -0.050),
    ]
    for cx, cy, radius, depth in craters:
        d = np.sqrt((x - cx) ** 2 + ((y - cy) * 1.15) ** 2)
        bowl = np.exp(-(d / radius) ** 2)
        rim = np.exp(-((d - radius * 0.88) / (radius * 0.16)) ** 2)
        height_field += depth * bowl + 0.045 * rim

    height_field = (height_field - height_field.min()) / (
        height_field.max() - height_field.min()
    )

    # Hillshade gives the raster a more physical relief appearance.
    gy, gx = np.gradient(height_field)
    azimuth = math.radians(315)
    altitude = math.radians(38)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(gx * gx + gy * gy) * 4.2)
    aspect = np.arctan2(-gx, gy)
    shade = (
        np.sin(altitude) * np.sin(slope)
        + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
    )
    shade = (shade - shade.min()) / (shade.max() - shade.min())

    terrain = 0.62 * height_field + 0.38 * shade
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    terrain = 0.17 + 0.70 * terrain

    rgb = np.dstack([terrain, terrain, terrain])
    return rgb


def load_background() -> np.ndarray:
    """Load a clean terrain image if available, otherwise create one."""
    for path in BACKGROUND_CANDIDATES:
        if path.exists():
            img = Image.open(path).convert("RGB")
            w, h = img.size
            target_ratio = 16.0 / 9.0
            ratio = w / h
            if abs(ratio - target_ratio) > 0.02:
                if ratio > target_ratio:
                    new_w = int(h * target_ratio)
                    left = (w - new_w) // 2
                    img = img.crop((left, 0, left + new_w, h))
                else:
                    new_h = int(w / target_ratio)
                    top = max(0, (h - new_h) // 2)
                    img = img.crop((0, top, w, top + new_h))
            img = img.resize((2200, 1238), Image.Resampling.LANCZOS)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            # Convert to restrained grayscale so route colors carry the message.
            gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
            gray = 0.20 + 0.67 * ((gray - gray.min()) / (gray.ptp() + 1e-8))
            return np.dstack([gray, gray, gray])
    return make_procedural_terrain()


def high_cost_overlay(width: int = 1400, height: int = 788) -> np.ndarray:
    """Create a semi-transparent irregular rotated oval for true high cost."""
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx / (width - 1)
    y = yy / (height - 1)

    cx, cy = 0.52, 0.50
    theta = math.radians(-12)
    xr = (x - cx) * math.cos(theta) - (y - cy) * math.sin(theta)
    yr = (x - cx) * math.sin(theta) + (y - cy) * math.cos(theta)

    a, b = 0.19, 0.275
    angle = np.arctan2(yr / b, xr / a)
    irregular = (
        1.0
        + 0.11 * np.sin(3.0 * angle + 0.6)
        + 0.06 * np.sin(7.0 * angle - 0.9)
        + 0.035 * np.cos(11.0 * angle)
    )
    r = np.sqrt((xr / (a * irregular)) ** 2 + (yr / (b * irregular)) ** 2)
    core = np.clip(1.0 - r, 0.0, 1.0)
    edge = np.clip((1.08 - r) / 0.16, 0.0, 1.0)

    rgba = np.zeros((height, width, 4), dtype=np.float32)
    color = mpl.colors.to_rgb(HIGH_COST_COLOR)
    rgba[..., :3] = color
    rgba[..., 3] = 0.36 * edge + 0.10 * core
    return rgba


def catmull_rom(points: list[tuple[float, float]], samples_per_segment: int = 42) -> np.ndarray:
    """Return a smooth Catmull-Rom spline through normalized control points."""
    pts = np.asarray(points, dtype=np.float64)
    padded = np.vstack([pts[0], pts, pts[-1]])
    curve = []
    for i in range(1, len(padded) - 2):
        p0, p1, p2, p3 = padded[i - 1], padded[i], padded[i + 1], padded[i + 2]
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
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


def draw_route(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    color: str,
    zorder: int,
    glow: bool = False,
) -> np.ndarray:
    route = catmull_rom(points)
    x, y = route[:, 0], route[:, 1]

    if glow:
        ax.plot(
            x,
            y,
            color=color,
            linewidth=12.0,
            alpha=0.22,
            solid_capstyle="round",
            zorder=zorder - 1,
        )

    ax.plot(
        x,
        y,
        color="white",
        linewidth=9.2,
        alpha=0.88,
        solid_capstyle="round",
        zorder=zorder,
    )
    ax.plot(
        x,
        y,
        color=color,
        linewidth=4.8,
        alpha=0.98,
        solid_capstyle="round",
        zorder=zorder + 1,
    )

    start_idx = max(0, len(route) - 18)
    arrow_start = route[start_idx]
    arrow_end = route[-1]
    ax.add_patch(
        FancyArrowPatch(
            arrow_start,
            arrow_end,
            arrowstyle="-|>",
            mutation_scale=26,
            linewidth=10.0,
            color="white",
            alpha=0.88,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder + 2,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            arrow_start,
            arrow_end,
            arrowstyle="-|>",
            mutation_scale=21,
            linewidth=4.9,
            color=color,
            alpha=0.98,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder + 3,
        )
    )
    return route


def draw_rover_marker(ax: plt.Axes, xy: tuple[float, float]) -> None:
    x, y = xy
    body = Polygon(
        [
            (x - 0.020, y - 0.007),
            (x + 0.020, y - 0.007),
            (x + 0.015, y + 0.011),
            (x - 0.015, y + 0.011),
        ],
        closed=True,
        facecolor="#f4f4f4",
        edgecolor="#222222",
        linewidth=1.2,
        zorder=30,
    )
    ax.add_patch(body)
    for dx in (-0.015, 0.015):
        ax.add_patch(
            Circle(
                (x + dx, y - 0.012),
                0.0075,
                facecolor="#222222",
                edgecolor="white",
                linewidth=0.7,
                zorder=31,
            )
        )


def draw_goal_marker(ax: plt.Axes, xy: tuple[float, float]) -> None:
    x, y = xy
    ax.add_patch(
        Circle(
            (x, y),
            0.018,
            facecolor="white",
            edgecolor="#222222",
            linewidth=1.3,
            alpha=0.97,
            zorder=30,
        )
    )
    ax.add_patch(
        Circle(
            (x, y),
            0.0085,
            facecolor="#222222",
            edgecolor="none",
            alpha=0.92,
            zorder=31,
        )
    )


def draw_figure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.0,
        }
    )

    background = load_background()

    nominal = [
        (0.12, 0.12),
        (0.20, 0.26),
        (0.27, 0.43),
        (0.34, 0.61),
        (0.48, 0.77),
        (0.66, 0.86),
        (0.86, 0.83),
    ]
    corrupted = [
        (0.12, 0.12),
        (0.23, 0.22),
        (0.36, 0.34),
        (0.48, 0.48),
        (0.57, 0.60),
        (0.69, 0.73),
        (0.86, 0.83),
    ]
    recovered = [
        (0.12, 0.12),
        (0.26, 0.13),
        (0.43, 0.17),
        (0.60, 0.27),
        (0.70, 0.45),
        (0.73, 0.63),
        (0.86, 0.83),
    ]

    fig, ax = plt.subplots(figsize=(10.8, 6.075), dpi=300)
    ax.imshow(
        background, extent=[0, 1, 0, 1], origin="lower", aspect="auto", zorder=0
    )
    ax.imshow(
        high_cost_overlay(),
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="auto",
        zorder=2,
    )

    # A subtle contour strengthens the sense of a real high-cost area while
    # keeping annotations minimal.
    theta = np.linspace(0, 2 * math.pi, 260)
    cx, cy = 0.52, 0.50
    rot = math.radians(-12)
    a, b = 0.195, 0.282
    jitter = 1.0 + 0.04 * np.sin(3 * theta + 0.6) + 0.025 * np.sin(7 * theta - 0.9)
    xr = a * jitter * np.cos(theta)
    yr = b * jitter * np.sin(theta)
    contour_x = cx + xr * math.cos(rot) - yr * math.sin(rot)
    contour_y = cy + xr * math.sin(rot) + yr * math.cos(rot)
    ax.plot(
        contour_x,
        contour_y,
        color=HIGH_COST_COLOR,
        linewidth=1.5,
        alpha=0.48,
        zorder=3,
    )

    draw_route(ax, nominal, NOMINAL_COLOR, zorder=9)
    draw_route(ax, corrupted, CORRUPTED_COLOR, zorder=12)
    draw_route(ax, recovered, RECOVERED_COLOR, zorder=15, glow=True)

    draw_rover_marker(ax, nominal[0])
    draw_goal_marker(ax, nominal[-1])

    handles = [
        Line2D([0], [0], color=NOMINAL_COLOR, lw=4.8, label="Nominal route"),
        Line2D([0], [0], color=CORRUPTED_COLOR, lw=4.8, label="Corrupted route"),
        Line2D([0], [0], color=RECOVERED_COLOR, lw=4.8, label="Recovered route (RL + MFA*)"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.982, 0.045),
        frameon=True,
        fancybox=False,
        framealpha=0.82,
        facecolor="white",
        edgecolor="#d0d0d0",
        fontsize=9.6,
        handlelength=2.2,
        borderpad=0.75,
        labelspacing=0.55,
    )
    for line in legend.get_lines():
        line.set_solid_capstyle("round")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    draw_figure()
    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
