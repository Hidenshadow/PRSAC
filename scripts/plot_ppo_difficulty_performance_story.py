#!/usr/bin/env python
"""Build PPO/SAC clean-train -> attack-drop -> recovery benchmark figures."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "rl_baselines"
DEFAULT_SEED_COUNT = 5
DEFAULT_ALGORITHMS = ("ppo", "sac")

LEVELS = [
    ("level1", "Level 1 synthetic"),
    ("level2", "Level 2 lunar"),
    ("level3", "Level 3 Mars"),
]
DIFFICULTIES = [
    ("easy", "Easy", 40),
    ("medium", "Medium", 60),
    ("hard", "Hard", 80),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed-count", type=int, default=DEFAULT_SEED_COUNT)
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--smooth-window", type=int, default=3)
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    if match is None:
        return -1
    return int(match.group(1))


def experiment_dir(root: Path, algorithm: str, level: str, difficulty: str, seed_count: int) -> Path:
    nested = root / algorithm / f"{level}_{difficulty}_shock_recovery_{seed_count}seeds"
    if nested.exists():
        return nested
    return root / f"{level}_{difficulty}_shock_recovery_{seed_count}seeds"


def normalized_training_curve(experiment: Path, smooth_window: int) -> pd.DataFrame:
    """Return training performance normalized so each seed ends at 100."""

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(experiment.glob("seed*/nominal_training_eval.csv")):
        frame = pd.read_csv(csv_path)
        required = {"global_step", "mean_scalar_cost"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[["global_step", "mean_scalar_cost"]].dropna().sort_values("global_step")
        if frame.empty:
            continue
        final_cost = float(frame.iloc[-1]["mean_scalar_cost"])
        if not math.isfinite(final_cost) or final_cost <= 0:
            continue
        frame = frame.copy()
        frame["seed"] = seed_from_path(csv_path)
        frame["performance_index"] = 100.0 * final_cost / frame["mean_scalar_cost"].clip(lower=1e-12)
        frames.append(frame[["seed", "global_step", "performance_index"]])

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby("global_step")["performance_index"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "performance_mean", "std": "performance_std", "count": "num_seeds"})
    )
    grouped["performance_smooth"] = (
        grouped["performance_mean"].rolling(window=max(1, smooth_window), center=True, min_periods=1).mean()
    )
    return grouped


def normalized_shock_recovery_curve(experiment: Path) -> pd.DataFrame:
    """Return attack/recovery performance normalized to clean nominal = 100."""

    csv_path = experiment / "shock_recovery_curve_aggregate.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    curve = pd.read_csv(csv_path)
    required = {
        "eval_domain",
        "phase",
        "attack_type",
        "recovery_step",
        "mean_attacked_scalar_cost_mean",
    }
    if not required.issubset(curve.columns):
        return pd.DataFrame()

    clean = curve[(curve["phase"] == "shock") & (curve["attack_type"] == "none")][
        ["eval_domain", "mean_attacked_scalar_cost_mean"]
    ].rename(columns={"mean_attacked_scalar_cost_mean": "clean_nominal_cost"})
    attacked = curve[
        (curve["attack_type"] == "environment")
        & (curve["phase"].isin(["shock", "recovery"]))
    ][["eval_domain", "phase", "recovery_step", "mean_attacked_scalar_cost_mean"]]
    if clean.empty or attacked.empty:
        return pd.DataFrame()

    merged = attacked.merge(clean, on="eval_domain", how="inner")
    merged["performance_index"] = (
        100.0
        * merged["clean_nominal_cost"].astype(float)
        / merged["mean_attacked_scalar_cost_mean"].astype(float).clip(lower=1e-12)
    )
    grouped = (
        merged.groupby("recovery_step")["performance_index"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "performance_mean", "std": "performance_std", "count": "num_eval_domains"})
        .sort_values("recovery_step")
    )
    return grouped


def performance_summary_row(
    algorithm: str,
    level: str,
    difficulty: str,
    train: pd.DataFrame,
    recovery: pd.DataFrame,
) -> dict[str, object]:
    train_start = float(train.iloc[0]["performance_mean"]) if not train.empty else float("nan")
    train_end = float(train.iloc[-1]["performance_mean"]) if not train.empty else float("nan")
    shock = float(recovery.iloc[0]["performance_mean"]) if not recovery.empty else float("nan")
    final = float(recovery.iloc[-1]["performance_mean"]) if not recovery.empty else float("nan")
    best = float(recovery["performance_mean"].max()) if not recovery.empty else float("nan")
    attack_drop = 100.0 - shock
    final_gain = final - shock
    best_gain = best - shock
    closure = 100.0 * final_gain / attack_drop if attack_drop >= 5.0 else float("nan")
    best_closure = 100.0 * best_gain / attack_drop if attack_drop >= 5.0 else float("nan")
    return {
        "algorithm": algorithm,
        "level": level,
        "difficulty": difficulty,
        "train_start_index": train_start,
        "train_end_index": train_end,
        "attack_shock_index": shock,
        "final_recovery_index": final,
        "best_recovery_index": best,
        "attack_drop_index_points": attack_drop,
        "final_recovered_index_points": final_gain,
        "best_recovered_index_points": best_gain,
        "final_recovery_closure_pct": closure,
        "best_recovery_closure_pct": best_closure,
    }


def fmt_pct(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.1f}%"


def write_markdown(summary: pd.DataFrame, output_dir: Path) -> None:
    rows = [
        "# RL Baseline Shock-Recovery Summary",
        "",
        "This summary uses a performance index where 100 is the clean nominal checkpoint before attack. Higher is better.",
        "The main benchmark requirement is a clear attack drop; weak recovery is acceptable because it motivates a stronger recovery algorithm.",
        "",
        "Main story figure: `fig_rl_baseline_protocol_performance.png`.",
        "",
        "| Algorithm | Level | Difficulty | Train start | Attack shock | Final recovery | Best recovery | Best closure | Interpretation |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in summary.iterrows():
        attack_drop = float(row["attack_drop_index_points"])
        best_closure = float(row["best_recovery_closure_pct"])
        final_gain = float(row["final_recovered_index_points"])
        if attack_drop < 5.0:
            interpretation = "weak attack; not a main benchmark case"
        elif attack_drop < 10.0:
            interpretation = "visible attack; useful secondary case"
        elif final_gain < 1.0:
            interpretation = "clear attack; recovery remains weak"
        elif best_closure >= 50.0:
            interpretation = "clear attack; recovers strongly"
        elif best_closure >= 20.0:
            interpretation = "clear attack; partially recovers"
        else:
            interpretation = "clear attack; recovery remains limited"
        rows.append(
            "| {algorithm} | {level} | {difficulty} | {train_start:.1f} | {shock:.1f} | {final:.1f} | {best:.1f} | {closure} | {interp} |".format(
                algorithm=str(row["algorithm"]),
                level=row["level"],
                difficulty=row["difficulty"],
                train_start=float(row["train_start_index"]),
                shock=float(row["attack_shock_index"]),
                final=float(row["final_recovery_index"]),
                best=float(row["best_recovery_index"]),
                closure=fmt_pct(best_closure),
                interp=interpretation,
            )
        )
    rows.extend(
        [
            "",
            "Reading the plot:",
            "",
            "- algorithm-colored training curve: clean training normalized to end at the pre-attack nominal checkpoint;",
            "- red marker/segment: environment attack shock at the same checkpoint;",
            "- green/orange/purple continuation: fine-tuning under attack.",
        ]
    )
    (output_dir / "rl_baseline_summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_protocol(
    root: Path,
    output_dir: Path,
    seed_count: int,
    algorithms: list[str],
    smooth_window: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
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

    fig, axes = plt.subplots(len(LEVELS), len(DIFFICULTIES), figsize=(14.8, 10.8), sharex=True)
    axes = np.asarray(axes).reshape(len(LEVELS), len(DIFFICULTIES))
    summary_rows: list[dict[str, object]] = []

    attack_color = "#c23b22"
    algorithm_colors = {
        "ppo": "#2f6b9a",
        "sac": "#2a7f62",
    }
    algorithm_linestyles = {
        "ppo": "-",
        "sac": "--",
    }

    for row_idx, (level, level_label) in enumerate(LEVELS):
        for col_idx, (difficulty, difficulty_label, map_size) in enumerate(DIFFICULTIES):
            ax = axes[row_idx, col_idx]
            panel_values: list[pd.Series] = []
            plotted_any = False
            nominal_end_for_axis = 50_000
            max_recovery_for_axis = 20_480

            for algorithm in algorithms:
                algorithm = str(algorithm).lower()
                experiment = experiment_dir(root, algorithm, level, difficulty, seed_count)
                train = normalized_training_curve(experiment, smooth_window)
                recovery = normalized_shock_recovery_curve(experiment)
                if train.empty or recovery.empty:
                    continue

                plotted_any = True
                summary_rows.append(performance_summary_row(algorithm, level, difficulty, train, recovery))
                nominal_end = int(train["global_step"].max())
                nominal_end_for_axis = nominal_end
                recovery = recovery.copy()
                recovery["timeline_step"] = nominal_end + recovery["recovery_step"].astype(int)
                max_recovery_for_axis = int(max(max_recovery_for_axis, recovery["recovery_step"].max()))
                color = algorithm_colors.get(algorithm, "0.25")
                linestyle = algorithm_linestyles.get(algorithm, "-")

                ax.plot(
                    train["global_step"],
                    train["performance_smooth"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.0,
                    alpha=0.75,
                    label=f"{algorithm.upper()} clean training",
                )
                shock_y = float(recovery.iloc[0]["performance_mean"])
                ax.plot([nominal_end, nominal_end], [100.0, shock_y], color=attack_color, linewidth=1.6, alpha=0.75)
                ax.scatter([nominal_end], [shock_y], color=attack_color, s=28, zorder=4)

                ax.plot(
                    recovery["timeline_step"],
                    recovery["performance_mean"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.4,
                    marker="o",
                    markersize=3.2,
                    label=f"{algorithm.upper()} recovery",
                )
                panel_values.extend(
                    [
                        train["performance_smooth"],
                        recovery["performance_mean"],
                    ]
                )

            if not plotted_any:
                ax.text(
                    0.5,
                    0.5,
                    "missing completed data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
                ax.set_title(f"{level_label}\n{difficulty_label} ({map_size}x{map_size})")
                continue

            ax.axhline(100.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.85)
            ax.axvline(nominal_end_for_axis, color=attack_color, linestyle="--", linewidth=1.0, alpha=0.8)
            ax.set_title(f"{level_label}\n{difficulty_label} ({map_size}x{map_size})")
            ax.grid(alpha=0.24)
            ax.set_xlim(0, nominal_end_for_axis + max_recovery_for_axis)
            ax.set_xticks([0, nominal_end_for_axis // 2, nominal_end_for_axis, nominal_end_for_axis + 20_000])
            ax.set_xticklabels(["0", "25k", "50k", "70k"])

            y_values = pd.concat(
                [*panel_values, pd.Series([100.0])],
                ignore_index=True,
            )
            y_min = float(y_values.min())
            y_max = float(y_values.max())
            span = max(25.0, y_max - y_min)
            center = 0.5 * (y_min + y_max)
            lower = min(y_min - 0.10 * span, center - 0.55 * span)
            upper = max(y_max + 0.10 * span, center + 0.55 * span)
            ax.set_ylim(lower, upper)

            if col_idx == 0:
                ax.set_ylabel("performance index\n(clean nominal = 100)")
            if row_idx == len(LEVELS) - 1:
                ax.set_xlabel("protocol step")

    handles, labels = [], []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("RL Baselines: Clean Training, Environment Attack Shock, and Recovery", y=1.075, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / "fig_rl_baseline_protocol_performance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "rl_baseline_performance_story.csv", index=False)
    write_markdown(summary, output_dir)
    return summary


def main() -> int:
    args = parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    if args.output_dir is None:
        output_dir = root / f"paper_story_{args.seed_count}seeds"
    else:
        output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    summary = plot_protocol(root, output_dir, args.seed_count, args.algorithms, args.smooth_window)
    print(f"Saved figure: {output_dir / 'fig_rl_baseline_protocol_performance.png'}")
    print(f"Saved summary: {output_dir / 'rl_baseline_performance_story.csv'}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
