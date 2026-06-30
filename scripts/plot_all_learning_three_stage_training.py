#!/usr/bin/env python
"""Plot three-stage clean-training, corruption-introduction, and recovery curves."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_all_learning_protocol_curves as all_learning
import plot_ldac_protocol_curves as protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "sac_modified" / "analysis" / "all_learning_protocol_20260629"

CLEAN_END = 0.25
MAP_END = 0.32
RECOVERY_START = MAP_END


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=protocol.DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios", nargs="+", default=list(protocol.DEFAULT_SCENARIOS))
    parser.add_argument("--baseline-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--y-limits", nargs=2, type=float, default=[70.0, 106.0])
    parser.add_argument("--filename-prefix", type=str, default="fig_all_learning_three_stage_training")
    parser.add_argument("--band", choices=("sem", "std"), default="sem")
    parser.add_argument("--dpi", type=int, default=900)
    parser.add_argument("--title", type=str, default="Clean training, map corruption, and recovery")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def phase_x(values: np.ndarray | pd.Series, duration: int, left: float, right: float) -> np.ndarray:
    duration = max(1.0, float(duration))
    return left + (right - left) * np.asarray(values, dtype=float) / duration


def finite(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def band_from(frame: pd.DataFrame, band: str) -> np.ndarray:
    std = np.nan_to_num(frame["performance_std"].to_numpy(dtype=float), nan=0.0)
    if band == "std":
        return std
    counts = frame.get("num_samples", pd.Series(np.ones(len(frame))))
    count_values = np.asarray(counts, dtype=float)
    count_values = np.clip(count_values, 1.0, None)
    return std / np.sqrt(count_values)


def fill_band(ax: plt.Axes, x: np.ndarray, y: np.ndarray, delta: np.ndarray, color: str, alpha: float = 0.12) -> None:
    if len(x) < 2:
        return
    ax.fill_between(x, y - delta, y + delta, color=color, alpha=alpha, linewidth=0)


def drop_transition_xy(y0: float, y1: float, n: int = 36) -> tuple[np.ndarray, np.ndarray]:
    """Draw the map-corruption shock as a fast nonlinear transition."""
    t = np.linspace(0.0, 1.0, n)
    x = CLEAN_END + (MAP_END - CLEAN_END) * t
    drop = 1.0 - np.exp(-4.5 * t)
    drop = drop / max(drop[-1], 1e-12)
    y = y0 + (y1 - y0) * drop
    return x, y


def draw_drop_transition(
    ax: plt.Axes,
    y0: float,
    y1: float,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    if not (math.isfinite(y0) and math.isfinite(y1)):
        return
    x_drop, y_drop = drop_transition_xy(y0, y1)
    ax.plot(
        x_drop,
        y_drop,
        color=color,
        linestyle="-",
        linewidth=max(1.25, linewidth - 0.35),
        alpha=alpha,
        solid_capstyle="round",
    )


def plot_three_stage_method(
    ax: plt.Axes,
    method: str,
    item: dict[str, object],
    nominal_steps: int,
    recovery_steps: int,
    band: str,
    label: str | None = None,
    alpha: float = 0.98,
) -> bool:
    spec = all_learning.METHODS[method]
    color = str(spec["color"])
    linestyle = spec["linestyle"]
    linewidth = float(spec["linewidth"])
    train = item["train"]
    recovery = item["recovery"]
    plotted = False

    if isinstance(train, pd.DataFrame) and not train.empty:
        x_train = phase_x(train["global_step"], nominal_steps, 0.0, CLEAN_END)
        y_train = train["performance_smooth"].to_numpy(dtype=float)
        y_band_center = train["performance_mean"].to_numpy(dtype=float)
        ax.plot(
            x_train,
            y_train,
            color=color,
            linestyle=linestyle,
            linewidth=max(1.0, linewidth - 0.45),
            alpha=0.74 * alpha,
        )
        fill_band(ax, x_train, y_band_center, band_from(train, band), color, alpha=0.08)
        plotted = True

    if isinstance(recovery, pd.DataFrame) and not recovery.empty:
        x_recovery = phase_x(recovery["recovery_step"], recovery_steps, RECOVERY_START, 1.0)
        y_recovery = recovery["performance_mean"].to_numpy(dtype=float)
        ax.plot(
            x_recovery,
            y_recovery,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label if label is not None else str(spec["label"]),
        )
        fill_band(ax, x_recovery, y_recovery, band_from(recovery, band), color, alpha=0.12)

        if isinstance(train, pd.DataFrame) and not train.empty:
            clean_y = finite(train.iloc[-1]["performance_smooth"])
            shock_y = finite(recovery.iloc[0]["performance_mean"])
            draw_drop_transition(ax, clean_y, shock_y, color, linewidth, alpha=0.90)
        plotted = True
    return plotted


def decorate_three_stage_axis(
    ax: plt.Axes,
    y_limits: tuple[float, float],
    show_ylabel: bool,
    show_xlabel: bool,
    nominal_steps: int,
    recovery_steps: int,
) -> None:
    y_min, y_max = y_limits
    ax.axvspan(0.0, CLEAN_END, color="#edf6fb", alpha=0.96, zorder=0)
    ax.axvspan(CLEAN_END, MAP_END, color="#fdebea", alpha=0.96, zorder=0)
    ax.axvspan(MAP_END, 1.0, color="#fff5e6", alpha=0.96, zorder=0)
    ax.axhline(100.0, color="0.45", linewidth=0.8, linestyle="-", alpha=0.75)
    ax.axvline(CLEAN_END, color="0.35", linewidth=1.0, linestyle="--", alpha=0.82)
    ax.axvline(MAP_END, color="0.35", linewidth=1.0, linestyle="--", alpha=0.82)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, axis="y", color="0.86", linewidth=0.75)
    ax.grid(False, axis="x")
    ax.set_xticks([0.0, CLEAN_END, MAP_END, 1.0])
    ax.set_xticklabels(["0", f"{nominal_steps // 1000}k", "drop", f"+{recovery_steps // 1000}k"])
    ax.text(CLEAN_END / 2.0, 0.965, "Clean training", transform=ax.transAxes, ha="center", va="top", fontsize=8.2)
    ax.text((CLEAN_END + MAP_END) / 2.0, 0.965, "Map", transform=ax.transAxes, ha="center", va="top", fontsize=8.2)
    ax.text((MAP_END + 1.0) / 2.0, 0.965, "Recovery", transform=ax.transAxes, ha="center", va="top", fontsize=8.2)
    if show_ylabel:
        ax.set_ylabel("Performance index")
    if show_xlabel:
        ax.set_xlabel("Protocol phase")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_grid(
    scenarios: list[str],
    data: dict[tuple[str, str], dict[str, object]],
    output_path: Path,
    y_limits: tuple[float, float],
    title: str,
    band: str,
    dpi: int,
) -> None:
    protocol.setup_matplotlib()
    ncols = 3
    nrows = int(math.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 3.55 * nrows), sharey=True, squeeze=False)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(scenarios):
            ax.axis("off")
            continue
        scenario = scenarios[idx]
        available = [data[(method, scenario)] for method in all_learning.METHODS if data[(method, scenario)]["seed_dirs"]]
        nominal_steps, recovery_steps = (50_000, 20_480)
        if available:
            nominal_steps = int(available[0]["nominal_steps"])
            recovery_steps = int(available[0]["recovery_steps"])

        decorate_three_stage_axis(
            ax,
            y_limits,
            show_ylabel=idx % ncols == 0,
            show_xlabel=idx // ncols == nrows - 1,
            nominal_steps=nominal_steps,
            recovery_steps=recovery_steps,
        )
        plotted_any = False
        for method in all_learning.METHODS:
            plotted_any = plot_three_stage_method(
                ax,
                method,
                data[(method, scenario)],
                nominal_steps,
                recovery_steps,
                band,
            ) or plotted_any
        ax.set_title(protocol.scenario_title(scenario), fontsize=12, pad=8)
        if not plotted_any:
            ax.text(0.5, 0.5, "No completed result", transform=ax.transAxes, ha="center", va="center", color="0.45")

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.004),
            handlelength=2.8,
        )
    fig.text(0.995, 0.006, f"Lines: mean; bands: $\\pm${band.upper()}", ha="right", va="bottom", fontsize=8.0, color="0.35")
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0.0, 0.055, 1.0, 0.962])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)


def aggregate_method_curve(
    method: str,
    scenarios: list[str],
    data: dict[tuple[str, str], dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        item = data[(method, scenario)]
        if not item["seed_dirs"]:
            continue
        nominal_steps = int(item["nominal_steps"])
        recovery_steps = int(item["recovery_steps"])
        train = item["train"]
        recovery = item["recovery"]
        if isinstance(train, pd.DataFrame) and not train.empty:
            x = phase_x(train["global_step"], nominal_steps, 0.0, CLEAN_END)
            y = train["performance_smooth"].to_numpy(dtype=float)
            for x_value, y_value in zip(x, y):
                rows.append({"method": method, "scenario": scenario, "stage": "clean", "x": float(x_value), "y": float(y_value)})
        if isinstance(train, pd.DataFrame) and not train.empty and isinstance(recovery, pd.DataFrame) and not recovery.empty:
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "stage": "drop_start",
                    "x": CLEAN_END,
                    "y": finite(train.iloc[-1]["performance_smooth"]),
                }
            )
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "stage": "drop_end",
                    "x": RECOVERY_START,
                    "y": finite(recovery.iloc[0]["performance_mean"]),
                }
            )
        if isinstance(recovery, pd.DataFrame) and not recovery.empty:
            x = phase_x(recovery["recovery_step"], recovery_steps, RECOVERY_START, 1.0)
            y = recovery["performance_mean"].to_numpy(dtype=float)
            for x_value, y_value in zip(x, y):
                rows.append({"method": method, "scenario": scenario, "stage": "recovery", "x": float(x_value), "y": float(y_value)})

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["x_round"] = frame["x"].round(6)
    summary = (
        frame.groupby(["method", "stage", "x_round"])["y"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"x_round": "x", "mean": "performance_mean", "std": "performance_std", "count": "num_scenarios"})
        .sort_values(["stage", "x"])
    )
    summary["performance_std"] = summary["performance_std"].fillna(0.0)
    summary["performance_sem"] = summary["performance_std"] / np.sqrt(summary["num_scenarios"].clip(lower=1))
    return summary


def aggregate_curves(
    scenarios: list[str],
    data: dict[tuple[str, str], dict[str, object]],
) -> pd.DataFrame:
    frames = [aggregate_method_curve(method, scenarios, data) for method in all_learning.METHODS]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_aggregate(
    aggregate: pd.DataFrame,
    output_path: Path,
    y_limits: tuple[float, float],
    title: str,
    band: str,
    dpi: int,
) -> None:
    protocol.setup_matplotlib()
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    decorate_three_stage_axis(ax, y_limits, True, False, 50_000, 20_480)

    for method, spec in all_learning.METHODS.items():
        subset = aggregate[aggregate["method"] == method]
        if subset.empty:
            continue
        color = str(spec["color"])
        linestyle = spec["linestyle"]
        linewidth = float(spec["linewidth"])
        label = str(spec["label"])

        for stage in ("clean", "recovery"):
            stage_rows = subset[subset["stage"] == stage].sort_values("x")
            if stage_rows.empty:
                continue
            x = stage_rows["x"].to_numpy(dtype=float)
            y = stage_rows["performance_mean"].to_numpy(dtype=float)
            delta_col = "performance_sem" if band == "sem" else "performance_std"
            delta = stage_rows[delta_col].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=color,
                linestyle=linestyle,
                linewidth=max(1.2, linewidth - 0.25 if stage == "clean" else linewidth),
                alpha=0.78 if stage == "clean" else 0.98,
                label=label if stage == "recovery" else None,
            )
            fill_band(ax, x, y, delta, color, alpha=0.10 if stage == "clean" else 0.13)

        start = subset[subset["stage"] == "drop_start"]
        end = subset[subset["stage"] == "drop_end"]
        if not start.empty and not end.empty:
            draw_drop_transition(
                ax,
                float(start["performance_mean"].iloc[0]),
                float(end["performance_mean"].iloc[0]),
                color,
                linewidth,
                alpha=0.92,
            )

    ax.set_title(title, fontsize=12, pad=8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.025),
            ncol=3,
            frameon=False,
            handlelength=2.9,
        )
    ax.text(
        0.995,
        0.02,
        f"Scenario-level mean; shaded bands: $\\pm${band.upper()} across scenarios",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
        color="0.35",
    )
    fig.tight_layout(rect=[0.0, 0.18, 1.0, 1.0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    fig.savefig(output_path.with_suffix(".pdf"))
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_root = resolve(args.baseline_root)
    output_dir = resolve(args.output_dir)
    scenarios = list(args.scenarios)
    y_limits = (float(args.y_limits[0]), float(args.y_limits[1]))

    data = all_learning.collect_data(baseline_root, scenarios, args.baseline_seeds, args.smooth_window)

    aggregate = aggregate_curves(scenarios, data)
    aggregate_path = output_dir / f"{args.filename_prefix}_aggregate_curve.csv"
    aggregate.to_csv(aggregate_path, index=False)

    aggregate_png = output_dir / f"{args.filename_prefix}_aggregate.png"
    plot_aggregate(aggregate, aggregate_png, y_limits, args.title, args.band, args.dpi)

    grid_png = output_dir / f"{args.filename_prefix}_grid.png"
    plot_grid(scenarios, data, grid_png, y_limits, args.title, args.band, args.dpi)

    scenario_summary = pd.DataFrame(all_learning.scenario_summary_rows(data, scenarios))
    scenario_summary_path = output_dir / f"{args.filename_prefix}_scenario_summary.csv"
    scenario_summary.to_csv(scenario_summary_path, index=False)

    aggregate_summary = all_learning.aggregate_summary(scenario_summary)
    aggregate_summary_path = output_dir / f"{args.filename_prefix}_method_summary.csv"
    aggregate_summary.to_csv(aggregate_summary_path, index=False)

    print(f"Saved aggregate figure: {aggregate_png}")
    print(f"Saved aggregate PDF: {aggregate_png.with_suffix('.pdf')}")
    print(f"Saved aggregate SVG: {aggregate_png.with_suffix('.svg')}")
    print(f"Saved grid figure: {grid_png}")
    print(f"Saved grid PDF: {grid_png.with_suffix('.pdf')}")
    print(f"Saved grid SVG: {grid_png.with_suffix('.svg')}")
    print(f"Saved aggregate curve data: {aggregate_path}")
    print(f"Saved scenario summary: {scenario_summary_path}")
    print(f"Saved method summary: {aggregate_summary_path}")


if __name__ == "__main__":
    main()
