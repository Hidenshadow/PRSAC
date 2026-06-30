#!/usr/bin/env python
"""Plot recovery-focused algorithm differences across all nine scenarios."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import plot_ldac_protocol_curves as protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "recovery_algorithm_differences"
LEVELS = ("level1", "level2", "level3")
DIFFICULTIES = ("easy", "medium", "hard")
SCENARIOS = tuple(f"{level}_{difficulty}" for difficulty in DIFFICULTIES for level in LEVELS)
ALGORITHM_ORDER = ("ppo", "sac", "cdr_sac", "ldac_sac")
BASELINE_ALGORITHMS = ("ppo", "sac", "cdr_sac")

ALGORITHM_LABELS = {
    "ppo": "PPO baseline",
    "sac": "SAC baseline",
    "cdr_sac": "CDR-SAC",
    "ldac_sac": "LDAC-SAC",
}

ALGORITHM_COLORS = {
    "ppo": protocol.OKABE_ITO["blue"],
    "sac": protocol.OKABE_ITO["bluish_green"],
    "cdr_sac": protocol.OKABE_ITO["orange"],
    "ldac_sac": protocol.OKABE_ITO["vermillion"],
}

ALGORITHM_LINESTYLES = {
    "ppo": "-",
    "sac": "-",
    "cdr_sac": "--",
    "ldac_sac": "-",
}

DIFFICULTY_COLORS = {
    "easy": "#0072B2",
    "medium": "#009E73",
    "hard": "#D55E00",
}

LEVEL_MARKERS = {
    "level1": "o",
    "level2": "s",
    "level3": "^",
}

SCATTER_LABEL_OFFSETS = {
    "level2_hard": (-0.28, 0.10),
    "level3_medium": (0.16, 0.24),
    "level3_hard": (-0.34, -0.20),
    "level1_medium": (0.14, -0.34),
}


@dataclass(frozen=True)
class MetricRow:
    scenario: str
    level: str
    difficulty: str
    algorithm: str
    num_seed_dirs: int
    shock_index: float
    final_recovery_index: float
    best_recovery_index: float
    recovery_auc_index: float
    recovery_regret_index: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=protocol.DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--y-limits", nargs=2, type=float, default=[72.0, 100.5])
    parser.add_argument("--prefix", type=str, default="fig_recovery_algorithm_difference")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def scenario_parts(scenario: str) -> tuple[str, str]:
    level, difficulty = scenario.split("_", 1)
    return level, difficulty


def scenario_short_label(scenario: str) -> str:
    level, difficulty = scenario_parts(scenario)
    return f"L{level[-1]}{difficulty[0].upper()}"


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.25",
            "axes.labelcolor": "0.15",
            "xtick.color": "0.15",
            "ytick.color": "0.15",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, png_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=600)
    fig.savefig(png_path.with_suffix(".pdf"))
    fig.savefig(png_path.with_suffix(".jpg"), dpi=600)
    print(f"Saved: {png_path}")
    print(f"Saved: {png_path.with_suffix('.pdf')}")
    print(f"Saved: {png_path.with_suffix('.jpg')}")


def curve_auc(curve: pd.DataFrame) -> float:
    if curve.empty:
        return math.nan
    grouped = (
        curve[["recovery_step", "performance_mean"]]
        .dropna()
        .groupby("recovery_step", as_index=False)["performance_mean"]
        .mean()
        .sort_values("recovery_step")
    )
    if grouped.empty:
        return math.nan
    x = grouped["recovery_step"].to_numpy(dtype=float)
    y = grouped["performance_mean"].to_numpy(dtype=float)
    if len(x) == 1 or float(x.max()) == float(x.min()):
        return float(y[0])
    return float(np.trapezoid(y, x) / (x.max() - x.min()))


def collect_metrics(data: dict[tuple[str, str], dict[str, object]], scenarios: list[str]) -> pd.DataFrame:
    rows: list[MetricRow] = []
    for scenario in scenarios:
        level, difficulty = scenario_parts(scenario)
        for algorithm in ALGORITHM_ORDER:
            item = data[(algorithm, scenario)]
            recovery = item["recovery"]
            if isinstance(recovery, pd.DataFrame) and not recovery.empty:
                shock = float(recovery.iloc[0]["performance_mean"])
                final = float(recovery.iloc[-1]["performance_mean"])
                best = float(recovery["performance_mean"].max())
                auc = curve_auc(recovery)
                regret = 100.0 - auc if math.isfinite(auc) else math.nan
            else:
                shock = final = best = auc = regret = math.nan
            rows.append(
                MetricRow(
                    scenario=scenario,
                    level=level,
                    difficulty=difficulty,
                    algorithm=algorithm,
                    num_seed_dirs=len(item["seed_dirs"]),
                    shock_index=shock,
                    final_recovery_index=final,
                    best_recovery_index=best,
                    recovery_auc_index=auc,
                    recovery_regret_index=regret,
                )
            )
    metrics = pd.DataFrame([row.__dict__ for row in rows])
    auc_pivot = metrics.pivot(index="scenario", columns="algorithm", values="recovery_auc_index")
    final_pivot = metrics.pivot(index="scenario", columns="algorithm", values="final_recovery_index")
    metrics["auc_minus_sac"] = metrics.apply(
        lambda row: row["recovery_auc_index"] - auc_pivot.loc[row["scenario"], "sac"]
        if row["algorithm"] == "ldac_sac"
        else math.nan,
        axis=1,
    )
    metrics["auc_minus_best_baseline"] = metrics.apply(
        lambda row: row["recovery_auc_index"] - max(
            auc_pivot.loc[row["scenario"], algorithm]
            for algorithm in BASELINE_ALGORITHMS
            if algorithm in auc_pivot.columns
        )
        if row["algorithm"] == "ldac_sac"
        else math.nan,
        axis=1,
    )
    metrics["final_minus_sac"] = metrics.apply(
        lambda row: row["final_recovery_index"] - final_pivot.loc[row["scenario"], "sac"]
        if row["algorithm"] == "ldac_sac"
        else math.nan,
        axis=1,
    )
    metrics["final_minus_best_baseline"] = metrics.apply(
        lambda row: row["final_recovery_index"] - max(
            final_pivot.loc[row["scenario"], algorithm]
            for algorithm in BASELINE_ALGORITHMS
            if algorithm in final_pivot.columns
        )
        if row["algorithm"] == "ldac_sac"
        else math.nan,
        axis=1,
    )
    return metrics


def plot_recovery_only_curves(
    data: dict[tuple[str, str], dict[str, object]],
    scenarios: list[str],
    output_path: Path,
    y_limits: tuple[float, float],
) -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(
        len(DIFFICULTIES),
        len(LEVELS),
        figsize=(13.6, 9.2),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    y_min, y_max = y_limits

    for row_idx, difficulty in enumerate(DIFFICULTIES):
        for col_idx, level in enumerate(LEVELS):
            scenario = f"{level}_{difficulty}"
            ax = axes[row_idx, col_idx]
            ax.axhline(100.0, color="0.45", linewidth=0.8, alpha=0.75)
            ax.axvline(0.0, color="0.35", linewidth=1.0, linestyle="--", alpha=0.8)
            for algorithm in ALGORITHM_ORDER:
                recovery = data[(algorithm, scenario)]["recovery"]
                if not isinstance(recovery, pd.DataFrame) or recovery.empty:
                    continue
                curve = recovery.sort_values("recovery_step")
                x = curve["recovery_step"].to_numpy(dtype=float) / 1000.0
                y = curve["performance_mean"].to_numpy(dtype=float)
                std = curve["performance_std"].to_numpy(dtype=float)
                count = curve["num_samples"].to_numpy(dtype=float)
                sem = np.divide(std, np.sqrt(np.maximum(count, 1.0)), out=np.zeros_like(std), where=np.isfinite(std))
                color = ALGORITHM_COLORS[algorithm]
                linewidth = 2.9 if algorithm == "ldac_sac" else 2.2 if algorithm == "cdr_sac" else 1.8
                alpha = 0.98 if algorithm == "ldac_sac" else 0.90 if algorithm == "cdr_sac" else 0.82
                zorder = 5 if algorithm == "ldac_sac" else 4 if algorithm == "cdr_sac" else 3
                ax.plot(
                    x,
                    y,
                    color=color,
                    linestyle=ALGORITHM_LINESTYLES[algorithm],
                    linewidth=linewidth,
                    alpha=alpha,
                    label=ALGORITHM_LABELS[algorithm],
                    zorder=zorder,
                )
                ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.11 if algorithm == "ldac_sac" else 0.08, linewidth=0)

            if row_idx == 0:
                ax.set_title(protocol.LEVEL_LABELS[level])
            if col_idx == 0:
                ax.set_ylabel(f"{protocol.DIFFICULTY_LABELS[difficulty]}\nPerformance index")
            if row_idx == len(DIFFICULTIES) - 1:
                ax.set_xlabel("Recovery step after attack (k)")
            ax.set_xlim(0.0, 20.5)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks([0, 5, 10, 15, 20])
            ax.grid(True, axis="y", color="0.87", linewidth=0.75)
            ax.grid(True, axis="x", color="0.92", linewidth=0.55)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.012), handlelength=3.0)
    fig.suptitle("Recovery after map-corruption attack", fontsize=14, y=0.992)
    fig.text(0.5, 0.047, "Lines show mean across completed seeds and evaluation domains; bands show SEM.", ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=[0.018, 0.072, 1.0, 0.958])
    save_figure(fig, output_path)
    plt.close(fig)


def plot_auc_advantage_heatmap(metrics: pd.DataFrame, output_path: Path) -> None:
    setup_matplotlib()
    ldac = metrics[metrics["algorithm"] == "ldac_sac"].set_index("scenario")
    matrix = np.array(
        [
            [float(ldac.loc[f"{level}_{difficulty}", "auc_minus_best_baseline"]) for level in LEVELS]
            for difficulty in DIFFICULTIES
        ]
    )
    max_abs = max(1.0, float(np.nanmax(np.abs(matrix))) + 0.15)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    image = ax.imshow(matrix, cmap="RdYlGn", norm=norm)
    ax.set_xticks(np.arange(len(LEVELS)))
    ax.set_xticklabels([protocol.LEVEL_LABELS[level] for level in LEVELS])
    ax.set_yticks(np.arange(len(DIFFICULTIES)))
    ax.set_yticklabels([protocol.DIFFICULTY_LABELS[difficulty] for difficulty in DIFFICULTIES])
    ax.set_xlabel("Terrain level")
    ax.set_ylabel("Scenario difficulty")
    ax.set_title("LDAC-SAC recovery AUC advantage over best baseline", pad=12)

    for row_idx, difficulty in enumerate(DIFFICULTIES):
        for col_idx, level in enumerate(LEVELS):
            value = matrix[row_idx, col_idx]
            scenario = f"{level}_{difficulty}"
            best_baseline = (
                metrics[(metrics["scenario"] == scenario) & (metrics["algorithm"].isin(BASELINE_ALGORITHMS))]
                .sort_values("recovery_auc_index", ascending=False)
                .iloc[0]["algorithm"]
            )
            text_color = "white" if abs(value) > 1.45 else "0.08"
            best_label = ALGORITHM_LABELS.get(str(best_baseline), str(best_baseline))
            ax.text(col_idx, row_idx, f"{value:+.2f}\nvs {best_label}", ha="center", va="center", fontsize=9, color=text_color)

    ax.set_xticks(np.arange(-0.5, len(LEVELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DIFFICULTIES), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUC advantage (performance-index points)")
    fig.text(0.5, 0.03, "Recovery AUC is the average performance index over 0-20k recovery steps.", ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=[0.0, 0.055, 1.0, 1.0])
    save_figure(fig, output_path)
    plt.close(fig)


def plot_paired_auc_scatter(metrics: pd.DataFrame, output_path: Path) -> None:
    setup_matplotlib()
    pivot = metrics.pivot(index="scenario", columns="algorithm", values="recovery_auc_index")
    ldac = pivot["ldac_sac"]
    best_baseline = pivot[[algorithm for algorithm in BASELINE_ALGORITHMS if algorithm in pivot.columns]].max(axis=1)

    lo = min(float(best_baseline.min()), float(ldac.min())) - 1.0
    hi = max(float(best_baseline.max()), float(ldac.max())) + 1.0
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    ax.plot([lo, hi], [lo, hi], color="0.35", linestyle="--", linewidth=1.1, label="Equal performance")

    for scenario in SCENARIOS:
        level, difficulty = scenario_parts(scenario)
        x = float(best_baseline.loc[scenario])
        y = float(ldac.loc[scenario])
        marker = LEVEL_MARKERS[level]
        color = DIFFICULTY_COLORS[difficulty]
        ax.scatter(x, y, s=92, marker=marker, color=color, edgecolor="white", linewidth=0.9, zorder=3)
        dx, dy = SCATTER_LABEL_OFFSETS.get(scenario, (0.08, 0.08))
        ax.text(
            x + dx,
            y + dy,
            scenario_short_label(scenario),
            fontsize=8.8,
            color="0.18",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Best non-LDAC recovery AUC")
    ax.set_ylabel("LDAC-SAC recovery AUC")
    ax.set_title("Paired recovery AUC across nine scenarios", pad=10)
    ax.grid(True, color="0.88", linewidth=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    difficulty_handles = [
        ax.scatter([], [], s=70, marker="o", color=DIFFICULTY_COLORS[difficulty], edgecolor="white", linewidth=0.8, label=protocol.DIFFICULTY_LABELS[difficulty])
        for difficulty in DIFFICULTIES
    ]
    level_handles = [
        ax.scatter([], [], s=70, marker=LEVEL_MARKERS[level], color="0.55", edgecolor="white", linewidth=0.8, label=protocol.LEVEL_LABELS[level])
        for level in LEVELS
    ]
    leg1 = ax.legend(handles=difficulty_handles, loc="lower right", frameon=False, title="Difficulty")
    ax.add_artist(leg1)
    ax.legend(handles=level_handles, loc="upper left", frameon=False, title="Level")
    fig.text(0.5, 0.025, "Points above the dashed line indicate higher recovery AUC than the strongest non-LDAC method in that scenario.", ha="center", fontsize=9, color="0.35")
    fig.tight_layout(rect=[0.0, 0.055, 1.0, 1.0])
    save_figure(fig, output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    baseline_root = resolve(args.baseline_root)
    scenarios = list(SCENARIOS)
    data = protocol.collect_data(baseline_root, scenarios, args.baseline_seeds, args.smooth_window)
    metrics = collect_metrics(data, scenarios)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{args.prefix}_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    plot_recovery_only_curves(
        data,
        scenarios,
        output_dir / f"{args.prefix}_recovery_only_curves.png",
        y_limits=(float(args.y_limits[0]), float(args.y_limits[1])),
    )
    plot_auc_advantage_heatmap(metrics, output_dir / f"{args.prefix}_auc_advantage_heatmap.png")
    plot_paired_auc_scatter(metrics, output_dir / f"{args.prefix}_paired_auc_scatter.png")


if __name__ == "__main__":
    main()
