#!/usr/bin/env python3
"""Draw a publication-quality training and recovery protocol overview.

This figure summarizes the current clean-training, attack-shock, and LDAC-SAC
recovery workflow used in the route recovery experiments.
"""

from __future__ import annotations

from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Polygon
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"
PNG_PATH = OUT_DIR / "training_flow_overview.png"
PDF_PATH = OUT_DIR / "training_flow_overview.pdf"


COLORS = {
    "ink": "#1f2933",
    "muted": "#5f6b7a",
    "line": "#aeb8c2",
    "panel_edge": "#d5dbe3",
    "panel_bg": "#f8fafc",
    "clean": "#009E73",
    "clean_light": "#e6f5ef",
    "attack": "#D55E00",
    "attack_light": "#fff1e8",
    "policy": "#0072B2",
    "policy_light": "#eaf3fb",
    "recovery": "#00A9C8",
    "recovery_light": "#e8f8fb",
    "planner": "#6f5fb8",
    "planner_light": "#f1effb",
    "eval": "#7a7f87",
    "eval_light": "#f1f3f5",
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


def add_round_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    *,
    facecolor: str = "white",
    edgecolor: str = COLORS["panel_edge"],
    textcolor: str = COLORS["ink"],
    linewidth: float = 1.15,
    fontsize: float = 9.0,
    weight: str = "normal",
    radius: float = 0.018,
    zorder: int = 3,
    align: str = "center",
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.010,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2.0,
        y + h / 2.0,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        color=textcolor,
        fontweight=weight,
        linespacing=1.18,
        zorder=zorder + 1,
    )
    return patch


def add_panel(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    color: str,
    light: str,
    index: str,
) -> None:
    panel = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.020",
        linewidth=1.1,
        edgecolor=COLORS["panel_edge"],
        facecolor=COLORS["panel_bg"],
        zorder=1,
    )
    ax.add_patch(panel)
    header = FancyBboxPatch(
        (x + 0.014, y + h - 0.105),
        w - 0.028,
        0.083,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        linewidth=0.0,
        facecolor=light,
        zorder=2,
    )
    ax.add_patch(header)
    ax.add_patch(Circle((x + 0.048, y + h - 0.063), 0.018, facecolor=color, edgecolor="white", lw=1.2, zorder=4))
    ax.text(
        x + 0.048,
        y + h - 0.063,
        index,
        ha="center",
        va="center",
        fontsize=9.0,
        color="white",
        fontweight="bold",
        zorder=5,
    )
    ax.text(
        x + 0.078,
        y + h - 0.049,
        title,
        ha="left",
        va="center",
        fontsize=11.5,
        color=COLORS["ink"],
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        x + 0.078,
        y + h - 0.080,
        subtitle,
        ha="left",
        va="center",
        fontsize=7.8,
        color=COLORS["muted"],
        zorder=4,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    lw: float = 1.6,
    rad: float = 0.0,
    style: str = "-|>",
    mutation_scale: float = 12.0,
    alpha: float = 1.0,
    zorder: int = 5,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=4,
        shrinkB=4,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_mini_map(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    mode: str,
) -> None:
    add_round_box(
        ax,
        (x, y),
        (w, h),
        "",
        facecolor="white",
        edgecolor="#cbd5df",
        linewidth=1.0,
        radius=0.012,
        zorder=3,
    )
    rng_angles = [0.22, 0.45, 0.66, 0.82]
    for index, frac in enumerate(rng_angles):
        yy = y + h * frac
        color = "#dfe4ea" if index % 2 == 0 else "#edf0f3"
        ax.plot(
            [x + 0.018, x + w - 0.018],
            [yy, yy + 0.016 * math.sin(index + 1)],
            color=color,
            lw=1.0,
            zorder=4,
        )
    cost_color = COLORS["attack"] if mode != "clean" else "#a0a8b2"
    cost_alpha = 0.27 if mode != "clean" else 0.12
    theta = [2 * math.pi * i / 90 for i in range(90)]
    poly = [
        (
            x + w * (0.54 + 0.18 * math.cos(t) + 0.025 * math.sin(3 * t)),
            y + h * (0.52 + 0.27 * math.sin(t) + 0.018 * math.cos(4 * t)),
        )
        for t in theta
    ]
    ax.add_patch(Polygon(poly, closed=True, facecolor=cost_color, edgecolor=cost_color, alpha=cost_alpha, lw=0.9, zorder=5))
    start = (x + 0.14 * w, y + 0.18 * h)
    goal = (x + 0.86 * w, y + 0.80 * h)
    ax.add_patch(Circle(start, 0.0065, facecolor="#222222", edgecolor="white", lw=0.5, zorder=8))
    ax.add_patch(Circle(goal, 0.0072, facecolor="white", edgecolor="#222222", lw=0.8, zorder=8))
    if mode == "clean":
        pts = [
            start,
            (x + 0.27 * w, y + 0.42 * h),
            (x + 0.44 * w, y + 0.72 * h),
            (x + 0.66 * w, y + 0.86 * h),
            goal,
        ]
        c = COLORS["clean"]
    elif mode == "attack":
        pts = [
            start,
            (x + 0.31 * w, y + 0.33 * h),
            (x + 0.51 * w, y + 0.52 * h),
            (x + 0.69 * w, y + 0.68 * h),
            goal,
        ]
        c = COLORS["policy"]
    else:
        pts = [
            start,
            (x + 0.33 * w, y + 0.19 * h),
            (x + 0.58 * w, y + 0.30 * h),
            (x + 0.72 * w, y + 0.55 * h),
            goal,
        ]
        c = COLORS["recovery"]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="white", lw=3.8, solid_capstyle="round", zorder=7)
    ax.plot(xs, ys, color=c, lw=2.3, solid_capstyle="round", zorder=8)


def draw_protocol_figure() -> None:
    setup_matplotlib()
    fig, ax = plt.subplots(figsize=(13.6, 7.65), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.040,
        0.955,
        "Training and Recovery Protocol",
        ha="left",
        va="top",
        fontsize=15.2,
        color=COLORS["ink"],
        fontweight="bold",
    )
    ax.text(
        0.040,
        0.918,
        "RL adapts planner preferences; the classical MFA* planner remains the route generator.",
        ha="left",
        va="top",
        fontsize=9.8,
        color=COLORS["muted"],
    )

    px = [0.040, 0.355, 0.670]
    py = 0.105
    pw = 0.285
    ph = 0.765
    add_panel(ax, px[0], py, pw, ph, "Clean map training", "learn nominal preference controller", COLORS["clean"], COLORS["clean_light"], "1")
    add_panel(ax, px[1], py, pw, ph, "Corruption shock", "freeze checkpoint; measure shock", COLORS["attack"], COLORS["attack_light"], "2")
    add_panel(ax, px[2], py, pw, ph, "LDAC recovery", "recover under corrupted planner-visible belief", COLORS["recovery"], COLORS["recovery_light"], "3")

    # Stage 1 boxes.
    b1 = add_round_box(
        ax,
        (px[0] + 0.030, py + 0.570),
        (0.100, 0.085),
        "Clean belief\n+ mission state",
        facecolor=COLORS["clean_light"],
        edgecolor=COLORS["clean"],
        fontsize=8.2,
    )
    b2 = add_round_box(
        ax,
        (px[0] + 0.157, py + 0.570),
        (0.098, 0.085),
        "RL policy\nSAC/PPO",
        facecolor=COLORS["policy_light"],
        edgecolor=COLORS["policy"],
        fontsize=8.5,
        weight="bold",
    )
    b3 = add_round_box(
        ax,
        (px[0] + 0.030, py + 0.430),
        (0.100, 0.080),
        "Preference action\nweights + lambda",
        facecolor="white",
        edgecolor=COLORS["line"],
        fontsize=7.8,
    )
    b4 = add_round_box(
        ax,
        (px[0] + 0.157, py + 0.430),
        (0.098, 0.080),
        "MFA* global\nplanner",
        facecolor=COLORS["planner_light"],
        edgecolor=COLORS["planner"],
        fontsize=8.3,
        weight="bold",
    )
    draw_mini_map(ax, px[0] + 0.048, py + 0.210, 0.190, 0.150, mode="clean")
    b5 = add_round_box(
        ax,
        (px[0] + 0.057, py + 0.110),
        (0.172, 0.062),
        "True cost reward\nrelative to heuristic",
        facecolor="white",
        edgecolor=COLORS["line"],
        fontsize=7.9,
    )
    b6 = add_round_box(
        ax,
        (px[0] + 0.062, py + 0.015),
        (0.162, 0.060),
        "Save clean checkpoint",
        facecolor=COLORS["clean_light"],
        edgecolor=COLORS["clean"],
        fontsize=8.1,
        weight="bold",
    )
    arrow(ax, (px[0] + 0.130, py + 0.612), (px[0] + 0.157, py + 0.612), color=COLORS["clean"])
    arrow(ax, (px[0] + 0.206, py + 0.570), (px[0] + 0.080, py + 0.510), color=COLORS["policy"], rad=0.12)
    arrow(ax, (px[0] + 0.130, py + 0.470), (px[0] + 0.157, py + 0.470), color=COLORS["planner"])
    arrow(ax, (px[0] + 0.206, py + 0.430), (px[0] + 0.145, py + 0.365), color=COLORS["planner"])
    arrow(ax, (px[0] + 0.144, py + 0.210), (px[0] + 0.143, py + 0.172), color=COLORS["line"])
    arrow(
        ax,
        (px[0] + 0.230, py + 0.142),
        (px[0] + 0.228, py + 0.570),
        color=COLORS["line"],
        rad=-0.48,
        alpha=0.70,
        zorder=2,
    )
    arrow(ax, (px[0] + 0.143, py + 0.110), (px[0] + 0.143, py + 0.075), color=COLORS["clean"])

    # Stage 2 boxes.
    c1 = add_round_box(
        ax,
        (px[1] + 0.045, py + 0.575),
        (0.195, 0.078),
        "Clean checkpoint is frozen",
        facecolor=COLORS["clean_light"],
        edgecolor=COLORS["clean"],
        fontsize=8.5,
        weight="bold",
    )
    c2 = add_round_box(
        ax,
        (px[1] + 0.045, py + 0.460),
        (0.195, 0.080),
        "Corrupt planner-visible belief\ntrue terrain remains hidden",
        facecolor=COLORS["attack_light"],
        edgecolor=COLORS["attack"],
        fontsize=8.0,
    )
    draw_mini_map(ax, px[1] + 0.048, py + 0.245, 0.190, 0.150, mode="attack")
    c3 = add_round_box(
        ax,
        (px[1] + 0.057, py + 0.132),
        (0.172, 0.066),
        "Shock evaluation\nclean vs corrupted",
        facecolor=COLORS["eval_light"],
        edgecolor=COLORS["eval"],
        fontsize=8.1,
        weight="bold",
    )
    c4 = add_round_box(
        ax,
        (px[1] + 0.057, py + 0.035),
        (0.172, 0.058),
        "Attack drop\npath-level mismatch",
        facecolor="white",
        edgecolor=COLORS["line"],
        fontsize=7.9,
    )
    arrow(ax, (px[1] + 0.142, py + 0.575), (px[1] + 0.142, py + 0.540), color=COLORS["line"])
    arrow(ax, (px[1] + 0.142, py + 0.460), (px[1] + 0.142, py + 0.395), color=COLORS["attack"])
    arrow(ax, (px[1] + 0.142, py + 0.245), (px[1] + 0.142, py + 0.198), color=COLORS["line"])
    arrow(ax, (px[1] + 0.142, py + 0.132), (px[1] + 0.142, py + 0.093), color=COLORS["line"])

    # Stage 3 boxes.
    r1 = add_round_box(
        ax,
        (px[2] + 0.040, py + 0.585),
        (0.205, 0.070),
        "Initialize recovery policy\nfrom clean checkpoint",
        facecolor=COLORS["clean_light"],
        edgecolor=COLORS["clean"],
        fontsize=8.0,
        weight="bold",
    )
    r2 = add_round_box(
        ax,
        (px[2] + 0.030, py + 0.482),
        (0.105, 0.072),
        "Local attack\nmixture",
        facecolor=COLORS["attack_light"],
        edgecolor=COLORS["attack"],
        fontsize=8.0,
    )
    r3 = add_round_box(
        ax,
        (px[2] + 0.157, py + 0.482),
        (0.100, 0.072),
        "Recovery\npolicy",
        facecolor=COLORS["policy_light"],
        edgecolor=COLORS["policy"],
        fontsize=8.2,
        weight="bold",
    )
    r4 = add_round_box(
        ax,
        (px[2] + 0.030, py + 0.378),
        (0.105, 0.072),
        "Preference action\nweights + lambda",
        facecolor="white",
        edgecolor=COLORS["line"],
        fontsize=7.6,
    )
    r5 = add_round_box(
        ax,
        (px[2] + 0.157, py + 0.378),
        (0.100, 0.072),
        "MFA* planner\nunder attack",
        facecolor=COLORS["planner_light"],
        edgecolor=COLORS["planner"],
        fontsize=7.9,
        weight="bold",
    )
    draw_mini_map(ax, px[2] + 0.048, py + 0.205, 0.190, 0.135, mode="recovery")
    r6 = add_round_box(
        ax,
        (px[2] + 0.030, py + 0.105),
        (0.227, 0.068),
        "LDAC-SAC update\nQ-gated anchor + advantage margin",
        facecolor=COLORS["recovery_light"],
        edgecolor=COLORS["recovery"],
        fontsize=7.8,
        weight="bold",
    )
    r7 = add_round_box(
        ax,
        (px[2] + 0.030, py + 0.020),
        (0.105, 0.056),
        "Update attack\nprobabilities",
        facecolor=COLORS["attack_light"],
        edgecolor=COLORS["attack"],
        fontsize=7.5,
    )
    r8 = add_round_box(
        ax,
        (px[2] + 0.157, py + 0.020),
        (0.100, 0.056),
        "Recovery\ncheckpoints",
        facecolor=COLORS["eval_light"],
        edgecolor=COLORS["eval"],
        fontsize=7.7,
        weight="bold",
    )
    arrow(ax, (px[2] + 0.142, py + 0.585), (px[2] + 0.142, py + 0.554), color=COLORS["clean"])
    arrow(ax, (px[2] + 0.135, py + 0.518), (px[2] + 0.157, py + 0.518), color=COLORS["attack"])
    arrow(ax, (px[2] + 0.207, py + 0.482), (px[2] + 0.082, py + 0.450), color=COLORS["policy"], rad=0.15)
    arrow(ax, (px[2] + 0.135, py + 0.414), (px[2] + 0.157, py + 0.414), color=COLORS["planner"])
    arrow(ax, (px[2] + 0.207, py + 0.378), (px[2] + 0.145, py + 0.340), color=COLORS["planner"])
    arrow(ax, (px[2] + 0.143, py + 0.205), (px[2] + 0.143, py + 0.173), color=COLORS["line"])
    arrow(
        ax,
        (px[2] + 0.210, py + 0.173),
        (px[2] + 0.210, py + 0.482),
        color=COLORS["recovery"],
        rad=-0.34,
        alpha=0.80,
        zorder=2,
    )
    arrow(ax, (px[2] + 0.082, py + 0.105), (px[2] + 0.082, py + 0.076), color=COLORS["attack"])
    arrow(ax, (px[2] + 0.207, py + 0.105), (px[2] + 0.207, py + 0.076), color=COLORS["eval"])
    arrow(
        ax,
        (px[2] + 0.082, py + 0.076),
        (px[2] + 0.082, py + 0.482),
        color=COLORS["attack"],
        rad=0.36,
        alpha=0.72,
        zorder=2,
    )

    # Cross-stage arrows.
    arrow(ax, (px[0] + pw + 0.006, py + 0.617), (px[1] - 0.006, py + 0.617), color=COLORS["line"], lw=1.9)
    arrow(ax, (px[1] + pw + 0.006, py + 0.617), (px[2] - 0.006, py + 0.617), color=COLORS["line"], lw=1.9)

    # Compact global evaluation strip.
    strip_y = 0.020
    ax.plot([0.045, 0.955], [strip_y + 0.055, strip_y + 0.055], color="#e2e7ed", lw=1.0, zorder=0)
    add_round_box(
        ax,
        (0.238, 0.020),
        (0.170, 0.052),
        "Fixed benchmark attack",
        facecolor="white",
        edgecolor=COLORS["line"],
        fontsize=7.7,
    )
    add_round_box(
        ax,
        (0.418, 0.020),
        (0.170, 0.052),
        "Held-out corruptions",
        facecolor="white",
        edgecolor=COLORS["line"],
        fontsize=7.7,
    )
    add_round_box(
        ax,
        (0.598, 0.020),
        (0.170, 0.052),
        "Recovery performance index",
        facecolor="white",
        edgecolor=COLORS["line"],
        fontsize=7.5,
    )

    # Small legend for visual encodings.
    legend_handles = [
        Line2D([0], [0], color=COLORS["clean"], lw=4, label="clean belief"),
        Line2D([0], [0], color=COLORS["attack"], lw=4, label="corrupted belief"),
        Line2D([0], [0], color=COLORS["recovery"], lw=4, label="recovery update"),
        Line2D([0], [0], color=COLORS["planner"], lw=4, label="classical planner"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.963, 0.956),
        frameon=False,
        fontsize=8.0,
        handlelength=1.8,
        ncol=2,
        columnspacing=1.0,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(PDF_PATH, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    draw_protocol_figure()
    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
