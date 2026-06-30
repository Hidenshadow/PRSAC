#!/usr/bin/env python
"""Export standalone three-stage learning-curve subfigures for PPT assembly."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import plot_all_learning_protocol_curves as all_learning
import plot_all_learning_three_stage_training as three_stage
import plot_ldac_protocol_curves as protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "sac_modified"
    / "analysis"
    / "all_learning_protocol_20260629"
    / "three_stage_subfigures"
)

METHOD_STYLES = {
    "ppo": {"label": "PPO", "color": "#0072B2", "linewidth": 2.4},
    "sac": {"label": "SAC", "color": "#009E73", "linewidth": 2.4},
    "cdr_sac": {"label": "CDR-SAC", "color": "#E69F00", "linewidth": 2.4},
    "stackelberg_sac": {"label": "Stackelberg-SAC", "color": "#CC79A7", "linewidth": 2.4},
    "valt_sac": {"label": "VALT-SAC", "color": "#56B4E9", "linewidth": 2.4},
    "ldac_sac": {"label": "PR-SAC", "color": "#D55E00", "linewidth": 3.4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=protocol.DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios", nargs="+", default=list(protocol.DEFAULT_SCENARIOS))
    parser.add_argument("--baseline-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--y-limits", nargs=2, type=float, default=[70.0, 106.0])
    parser.add_argument("--band", choices=("sem", "std"), default="sem")
    parser.add_argument("--dpi", type=int, default=900)
    parser.add_argument("--width", type=float, default=5.8)
    parser.add_argument("--height", type=float, default=3.55)
    parser.add_argument("--no-title", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def style_matplotlib() -> None:
    protocol.setup_matplotlib()
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 11,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_method_solid(
    ax: plt.Axes,
    method: str,
    item: dict[str, object],
    nominal_steps: int,
    recovery_steps: int,
    band: str,
) -> bool:
    style = METHOD_STYLES[method]
    color = style["color"]
    linewidth = float(style["linewidth"])
    train = item["train"]
    recovery = item["recovery"]
    plotted = False

    if isinstance(train, pd.DataFrame) and not train.empty:
        x_train = three_stage.phase_x(train["global_step"], nominal_steps, 0.0, three_stage.CLEAN_END)
        y_train = train["performance_smooth"].to_numpy(dtype=float)
        y_center = train["performance_mean"].to_numpy(dtype=float)
        ax.plot(x_train, y_train, color=color, linestyle="-", linewidth=max(1.8, linewidth - 0.55), alpha=0.82)
        three_stage.fill_band(ax, x_train, y_center, three_stage.band_from(train, band), color, alpha=0.075)
        plotted = True

    if isinstance(recovery, pd.DataFrame) and not recovery.empty:
        x_recovery = three_stage.phase_x(
            recovery["recovery_step"],
            recovery_steps,
            three_stage.RECOVERY_START,
            1.0,
        )
        y_recovery = recovery["performance_mean"].to_numpy(dtype=float)
        ax.plot(x_recovery, y_recovery, color=color, linestyle="-", linewidth=linewidth, alpha=0.98)
        three_stage.fill_band(ax, x_recovery, y_recovery, three_stage.band_from(recovery, band), color, alpha=0.11)

        if isinstance(train, pd.DataFrame) and not train.empty:
            clean_y = three_stage.finite(train.iloc[-1]["performance_smooth"])
            shock_y = three_stage.finite(recovery.iloc[0]["performance_mean"])
            three_stage.draw_drop_transition(ax, clean_y, shock_y, color, linewidth, alpha=0.90)
        plotted = True

    return plotted


def export_scenario(
    scenario: str,
    data: dict[tuple[str, str], dict[str, object]],
    output_dir: Path,
    y_limits: tuple[float, float],
    band: str,
    dpi: int,
    figsize: tuple[float, float],
    show_title: bool,
) -> None:
    available = [data[(method, scenario)] for method in all_learning.METHODS if data[(method, scenario)]["seed_dirs"]]
    nominal_steps, recovery_steps = (50_000, 20_480)
    if available:
        nominal_steps = int(available[0]["nominal_steps"])
        recovery_steps = int(available[0]["recovery_steps"])

    fig, ax = plt.subplots(figsize=figsize)
    three_stage.decorate_three_stage_axis(
        ax,
        y_limits,
        show_ylabel=True,
        show_xlabel=True,
        nominal_steps=nominal_steps,
        recovery_steps=recovery_steps,
    )

    plotted = False
    for method in all_learning.METHODS:
        plotted = plot_method_solid(ax, method, data[(method, scenario)], nominal_steps, recovery_steps, band) or plotted

    if show_title:
        ax.set_title(protocol.scenario_title(scenario), pad=8)
    if not plotted:
        ax.text(0.5, 0.5, "No completed result", transform=ax.transAxes, ha="center", va="center", color="0.45")

    fig.tight_layout(pad=0.6)
    stem = f"three_stage_{scenario}"
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if suffix == ".png":
            kwargs["dpi"] = dpi
        fig.savefig(output_dir / f"{stem}{suffix}", **kwargs)
    plt.close(fig)


def export_legend_bar(output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(16.0, 1.05))
    ax.set_axis_off()
    handles = [
        Patch(
            facecolor=style["color"],
            edgecolor=style["color"],
        )
        for style in METHOD_STYLES.values()
    ]
    labels = [style["label"] for style in METHOD_STYLES.values()]
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(labels),
        frameon=False,
        handlelength=3.2,
        handleheight=1.05,
        columnspacing=1.45,
        handletextpad=0.55,
        fontsize=18,
    )
    fig.tight_layout(pad=0.05)
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.02, "transparent": True}
        if suffix == ".png":
            kwargs["dpi"] = dpi
        fig.savefig(output_dir / f"learning_algorithm_color_bar{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    style_matplotlib()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_root = resolve(args.baseline_root)
    scenarios = list(args.scenarios)
    y_limits = (float(args.y_limits[0]), float(args.y_limits[1]))

    data = all_learning.collect_data(baseline_root, scenarios, args.baseline_seeds, args.smooth_window)
    for scenario in scenarios:
        export_scenario(
            scenario,
            data,
            output_dir,
            y_limits,
            args.band,
            args.dpi,
            (float(args.width), float(args.height)),
            show_title=not args.no_title,
        )
    export_legend_bar(output_dir, args.dpi)

    print(f"Saved {len(scenarios)} scenario subfigures to: {output_dir}")
    print(f"Saved color bar: {output_dir / 'learning_algorithm_color_bar.png'}")


if __name__ == "__main__":
    main()
