#!/usr/bin/env python
"""Plot ACBR-PPO easy recovery runs against PPO/SAC baseline protocol curves."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from plot_ppo_difficulty_performance_story import (  # noqa: E402
    experiment_dir,
    normalized_shock_recovery_curve,
    normalized_training_curve,
    performance_summary_row,
)


LEVELS = [
    ("level1", "Level 1 synthetic"),
    ("level2", "Level 2 lunar"),
    ("level3", "Level 3 Mars"),
]
DIFFICULTY = ("easy", "Easy", 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=PROJECT_ROOT / "runs" / "rl_baselines")
    parser.add_argument(
        "--acbr-root",
        type=Path,
        default=PROJECT_ROOT / "runs" / "acbr_recovery_easy_1seed_matrix",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "acbr_recovery_easy_1seed_matrix" / "comparison",
    )
    parser.add_argument("--smooth-window", type=int, default=3)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def acbr_seed_dir(acbr_root: Path, level: str) -> Path:
    return acbr_root / f"{level}_easy_shock_recovery_1seed" / "seed0"


def normalized_single_seed_training_curve(seed_dir: Path, smooth_window: int) -> pd.DataFrame:
    csv_path = seed_dir / "nominal_training_eval.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(csv_path)
    required = {"global_step", "mean_scalar_cost"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame = frame[["global_step", "mean_scalar_cost"]].dropna().sort_values("global_step")
    if frame.empty:
        return pd.DataFrame()
    final_cost = float(frame.iloc[-1]["mean_scalar_cost"])
    if not math.isfinite(final_cost) or final_cost <= 0.0:
        return pd.DataFrame()
    frame = frame.copy()
    frame["performance_mean"] = 100.0 * final_cost / frame["mean_scalar_cost"].clip(lower=1e-12)
    frame["performance_std"] = 0.0
    frame["num_seeds"] = 1
    frame["performance_smooth"] = (
        frame["performance_mean"].rolling(window=max(1, smooth_window), center=True, min_periods=1).mean()
    )
    return frame[["global_step", "performance_mean", "performance_std", "num_seeds", "performance_smooth"]]


def normalized_single_seed_recovery_curve(seed_dir: Path) -> pd.DataFrame:
    csv_path = seed_dir / "shock_recovery_curve.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    curve = pd.read_csv(csv_path)
    required = {"eval_domain", "phase", "attack_type", "recovery_step", "mean_attacked_scalar_cost"}
    if not required.issubset(curve.columns):
        return pd.DataFrame()

    clean = curve[(curve["phase"] == "shock") & (curve["attack_type"] == "none")][
        ["eval_domain", "mean_attacked_scalar_cost"]
    ].rename(columns={"mean_attacked_scalar_cost": "clean_nominal_cost"})
    attacked = curve[
        (curve["attack_type"] == "environment") & (curve["phase"].isin(["shock", "recovery"]))
    ][["eval_domain", "phase", "recovery_step", "mean_attacked_scalar_cost"]]
    if clean.empty or attacked.empty:
        return pd.DataFrame()

    merged = attacked.merge(clean, on="eval_domain", how="inner")
    merged["performance_index"] = (
        100.0
        * merged["clean_nominal_cost"].astype(float)
        / merged["mean_attacked_scalar_cost"].astype(float).clip(lower=1e-12)
    )
    grouped = (
        merged.groupby("recovery_step")["performance_index"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "performance_mean", "std": "performance_std", "count": "num_eval_domains"})
        .sort_values("recovery_step")
    )
    return grouped


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.22",
            "axes.labelcolor": "0.15",
            "xtick.color": "0.15",
            "ytick.color": "0.15",
            "font.size": 10,
        }
    )


def plot_protocol(baseline_root: Path, acbr_root: Path, output_dir: Path, smooth_window: int) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_matplotlib()

    fig, axes = plt.subplots(1, len(LEVELS), figsize=(15.2, 4.4), sharex=True)
    axes = np.asarray(axes).reshape(1, len(LEVELS))[0]
    summary_rows: list[dict[str, object]] = []
    difficulty, difficulty_label, map_size = DIFFICULTY

    specs = [
        ("ppo", "PPO", baseline_root, 5, "#2f6b9a", "-"),
        ("sac", "SAC", baseline_root, 5, "#2a7f62", "--"),
        ("acbr_ppo", "ACBR-PPO", acbr_root, 1, "#7a3db8", "-."),
    ]
    attack_color = "#c23b22"

    for ax, (level, level_label) in zip(axes, LEVELS):
        panel_values: list[pd.Series] = []
        plotted_any = False
        nominal_end_for_axis = 50_000
        max_recovery_for_axis = 20_480

        for algorithm, label, root, seed_count, color, linestyle in specs:
            if algorithm == "acbr_ppo":
                run_dir = acbr_seed_dir(root, level)
                train = normalized_single_seed_training_curve(run_dir, smooth_window)
                recovery = normalized_single_seed_recovery_curve(run_dir)
            else:
                run_dir = experiment_dir(root, algorithm, level, difficulty, seed_count)
                train = normalized_training_curve(run_dir, smooth_window)
                recovery = normalized_shock_recovery_curve(run_dir)
            if train.empty or recovery.empty:
                continue

            plotted_any = True
            summary_rows.append(performance_summary_row(algorithm, level, difficulty, train, recovery))
            nominal_end = int(train["global_step"].max())
            nominal_end_for_axis = max(nominal_end_for_axis, nominal_end)
            recovery = recovery.copy()
            recovery["timeline_step"] = nominal_end + recovery["recovery_step"].astype(int)
            max_recovery_for_axis = int(max(max_recovery_for_axis, recovery["recovery_step"].max()))

            ax.plot(
                train["global_step"],
                train["performance_smooth"],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                alpha=0.62,
                label=f"{label} clean training",
            )
            shock_y = float(recovery.iloc[0]["performance_mean"])
            ax.plot([nominal_end, nominal_end], [100.0, shock_y], color=attack_color, linewidth=1.5, alpha=0.72)
            ax.scatter([nominal_end], [shock_y], color=attack_color, s=28, zorder=4)
            ax.plot(
                recovery["timeline_step"],
                recovery["performance_mean"],
                color=color,
                linestyle=linestyle,
                linewidth=2.4,
                marker="o",
                markersize=3.2,
                label=f"{label} recovery",
            )
            panel_values.extend([train["performance_smooth"], recovery["performance_mean"]])

        if not plotted_any:
            ax.text(0.5, 0.5, "missing completed data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{level_label}\n{difficulty_label} ({map_size}x{map_size})")
            continue

        ax.axhline(100.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.85)
        ax.axvline(nominal_end_for_axis, color=attack_color, linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(f"{level_label}\n{difficulty_label} ({map_size}x{map_size})")
        ax.grid(alpha=0.24)
        ax.set_xlim(0, nominal_end_for_axis + max_recovery_for_axis)
        ax.set_xticks([0, nominal_end_for_axis // 2, nominal_end_for_axis, nominal_end_for_axis + 20_000])
        ax.set_xticklabels(["0", "25k", "50k", "70k"])
        ax.set_xlabel("protocol step")

        y_values = pd.concat([*panel_values, pd.Series([100.0])], ignore_index=True)
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        span = max(25.0, y_max - y_min)
        center = 0.5 * (y_min + y_max)
        ax.set_ylim(min(y_min - 0.10 * span, center - 0.55 * span), max(y_max + 0.10 * span, center + 0.55 * span))

    axes[0].set_ylabel("performance index\n(clean nominal = 100)")
    handles, labels = [], []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("PPO/SAC Baselines vs ACBR-PPO: Easy Maps", y=1.13, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / "fig_acbr_easy_vs_rl_baselines_protocol_performance.png"
    pdf_path = output_dir / "fig_acbr_easy_vs_rl_baselines_protocol_performance.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "acbr_easy_vs_rl_baselines_performance_story.csv", index=False)
    return summary


def main() -> int:
    args = parse_args()
    baseline_root = resolve(args.baseline_root)
    acbr_root = resolve(args.acbr_root)
    output_dir = resolve(args.output_dir)
    summary = plot_protocol(baseline_root, acbr_root, output_dir, args.smooth_window)
    print(f"Saved figure: {output_dir / 'fig_acbr_easy_vs_rl_baselines_protocol_performance.png'}")
    print(f"Saved PDF: {output_dir / 'fig_acbr_easy_vs_rl_baselines_protocol_performance.pdf'}")
    print(f"Saved summary: {output_dir / 'acbr_easy_vs_rl_baselines_performance_story.csv'}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
