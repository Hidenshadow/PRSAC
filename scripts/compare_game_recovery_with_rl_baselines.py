#!/usr/bin/env python
"""Compare 1-seed game recovery results with PPO/SAC baseline protocol data."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    DIFFICULTIES,
    LEVELS,
    experiment_dir,
    normalized_shock_recovery_curve,
    normalized_training_curve,
    performance_summary_row,
)


DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "runs" / "rl_baselines"
DEFAULT_GAME_ROOT = PROJECT_ROOT / "runs" / "game_recovery_protocol_analysis"
DEFAULT_OUTPUT_DIR = DEFAULT_GAME_ROOT / "compare_with_rl_baselines"
DEFAULT_BASELINE_SUMMARY = DEFAULT_BASELINE_ROOT / "paper_story_5seeds" / "rl_baseline_performance_story.csv"
DEFAULT_GAME_SUMMARY = DEFAULT_GAME_ROOT / "paper_story_1seed" / "game_recovery_performance_story.csv"

SUMMARY_METRICS = [
    "attack_shock_index",
    "final_recovery_index",
    "best_recovery_index",
    "attack_drop_index_points",
    "final_recovered_index_points",
    "best_recovered_index_points",
    "final_recovery_closure_pct",
    "best_recovery_closure_pct",
]


@dataclass(frozen=True)
class AlgorithmSpec:
    algorithm: str
    label: str
    root: Path
    seed_count: int
    color: str
    linestyle: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--game-root", type=Path, default=DEFAULT_GAME_ROOT)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--game-summary", type=Path, default=DEFAULT_GAME_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-window", type=int, default=3)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_summary(path: Path, seed_count: int, source_label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame.insert(1, "seed_count", seed_count)
    frame.insert(2, "source", source_label)
    return frame


def build_combined_summary(baseline_summary: Path, game_summary: Path) -> pd.DataFrame:
    baseline = load_summary(baseline_summary, 5, "rl_baselines/paper_story_5seeds")
    game = load_summary(game_summary, 1, "game_recovery_protocol_analysis/paper_story_1seed")
    combined = pd.concat([baseline, game], ignore_index=True)
    level_order = {level: idx for idx, (level, _) in enumerate(LEVELS)}
    difficulty_order = {difficulty: idx for idx, (difficulty, _, _) in enumerate(DIFFICULTIES)}
    combined["_level_order"] = combined["level"].map(level_order)
    combined["_difficulty_order"] = combined["difficulty"].map(difficulty_order)
    combined["_algorithm_order"] = combined["algorithm"].map({"ppo": 0, "sac": 1, "game_ppo": 2}).fillna(99)
    combined = combined.sort_values(["_level_order", "_difficulty_order", "_algorithm_order"]).drop(
        columns=["_level_order", "_difficulty_order", "_algorithm_order"]
    )
    return combined


def build_delta_table(combined: pd.DataFrame) -> pd.DataFrame:
    game = combined[combined["algorithm"] == "game_ppo"].set_index(["level", "difficulty"])
    rows: list[dict[str, object]] = []
    for baseline_algorithm in ["ppo", "sac"]:
        baseline = combined[combined["algorithm"] == baseline_algorithm].set_index(["level", "difficulty"])
        for key in game.index.intersection(baseline.index):
            game_row = game.loc[key]
            baseline_row = baseline.loc[key]
            row: dict[str, object] = {
                "level": key[0],
                "difficulty": key[1],
                "baseline_algorithm": baseline_algorithm,
                "game_algorithm": "game_ppo",
                "baseline_seed_count": int(baseline_row["seed_count"]),
                "game_seed_count": int(game_row["seed_count"]),
            }
            for metric in SUMMARY_METRICS:
                game_value = float(game_row[metric]) if pd.notna(game_row[metric]) else math.nan
                baseline_value = float(baseline_row[metric]) if pd.notna(baseline_row[metric]) else math.nan
                row[f"baseline_{metric}"] = baseline_value
                row[f"game_{metric}"] = game_value
                row[f"delta_{metric}"] = game_value - baseline_value
            rows.append(row)
    return pd.DataFrame(rows)


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


def plot_combined_protocol(
    baseline_root: Path,
    game_root: Path,
    output_dir: Path,
    smooth_window: int,
) -> pd.DataFrame:
    specs = [
        AlgorithmSpec("ppo", "PPO (5 seeds)", baseline_root, 5, "#2f6b9a", "-"),
        AlgorithmSpec("sac", "SAC (5 seeds)", baseline_root, 5, "#2a7f62", "--"),
        AlgorithmSpec("game_ppo", "Game-PPO (1 seed)", game_root, 1, "#7a3db8", "-."),
    ]

    setup_matplotlib()
    fig, axes = plt.subplots(len(LEVELS), len(DIFFICULTIES), figsize=(15.2, 11.0), sharex=True)
    axes = np.asarray(axes).reshape(len(LEVELS), len(DIFFICULTIES))
    summary_rows: list[dict[str, object]] = []
    attack_color = "#c23b22"

    for row_idx, (level, level_label) in enumerate(LEVELS):
        for col_idx, (difficulty, difficulty_label, map_size) in enumerate(DIFFICULTIES):
            ax = axes[row_idx, col_idx]
            panel_values: list[pd.Series] = []
            plotted_any = False
            nominal_end_for_axis = 50_000
            max_recovery_for_axis = 20_480

            for spec in specs:
                experiment = experiment_dir(spec.root, spec.algorithm, level, difficulty, spec.seed_count)
                train = normalized_training_curve(experiment, smooth_window)
                recovery = normalized_shock_recovery_curve(experiment)
                if train.empty or recovery.empty:
                    continue

                plotted_any = True
                summary = performance_summary_row(spec.algorithm, level, difficulty, train, recovery)
                summary["seed_count"] = spec.seed_count
                summary_rows.append(summary)

                nominal_end = int(train["global_step"].max())
                nominal_end_for_axis = max(nominal_end_for_axis, nominal_end)
                recovery = recovery.copy()
                recovery["timeline_step"] = nominal_end + recovery["recovery_step"].astype(int)
                max_recovery_for_axis = int(max(max_recovery_for_axis, recovery["recovery_step"].max()))

                ax.plot(
                    train["global_step"],
                    train["performance_smooth"],
                    color=spec.color,
                    linestyle=spec.linestyle,
                    linewidth=1.8,
                    alpha=0.58,
                )
                shock_y = float(recovery.iloc[0]["performance_mean"])
                ax.plot([nominal_end, nominal_end], [100.0, shock_y], color=attack_color, linewidth=1.4, alpha=0.55)
                ax.scatter([nominal_end], [shock_y], color=attack_color, s=24, zorder=4)
                ax.plot(
                    recovery["timeline_step"],
                    recovery["performance_mean"],
                    color=spec.color,
                    linestyle=spec.linestyle,
                    linewidth=2.35,
                    marker="o",
                    markersize=3.0,
                    label=spec.label,
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

            y_values = pd.concat([*panel_values, pd.Series([100.0])], ignore_index=True)
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
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("PPO/SAC Baselines vs Game-PPO: Clean Training, Attack Shock, and Recovery", y=1.075, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    path = output_dir / "fig_game_vs_rl_baselines_protocol_performance.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def plot_final_recovery_bars(combined: pd.DataFrame, output_dir: Path) -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(len(LEVELS), len(DIFFICULTIES), figsize=(14.2, 10.2), sharey=True)
    axes = np.asarray(axes).reshape(len(LEVELS), len(DIFFICULTIES))
    colors = {"ppo": "#2f6b9a", "sac": "#2a7f62", "game_ppo": "#7a3db8"}
    labels = {"ppo": "PPO\n5 seeds", "sac": "SAC\n5 seeds", "game_ppo": "Game-PPO\n1 seed"}
    algorithms = ["ppo", "sac", "game_ppo"]

    for row_idx, (level, level_label) in enumerate(LEVELS):
        for col_idx, (difficulty, difficulty_label, map_size) in enumerate(DIFFICULTIES):
            ax = axes[row_idx, col_idx]
            panel = combined[(combined["level"] == level) & (combined["difficulty"] == difficulty)].set_index("algorithm")
            x = np.arange(len(algorithms))
            finals = [float(panel.loc[alg, "final_recovery_index"]) if alg in panel.index else math.nan for alg in algorithms]
            bests = [float(panel.loc[alg, "best_recovery_index"]) if alg in panel.index else math.nan for alg in algorithms]
            shocks = [float(panel.loc[alg, "attack_shock_index"]) if alg in panel.index else math.nan for alg in algorithms]

            ax.bar(x, finals, color=[colors[alg] for alg in algorithms], alpha=0.78, width=0.62)
            ax.scatter(x, bests, color="black", s=26, marker="D", label="best recovery" if row_idx == 0 and col_idx == 0 else None)
            ax.scatter(x, shocks, color="#c23b22", s=28, marker="x", label="attack shock" if row_idx == 0 and col_idx == 0 else None)
            ax.axhline(100.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([labels[alg] for alg in algorithms], fontsize=8)
            ax.set_title(f"{level_label}\n{difficulty_label} ({map_size}x{map_size})")
            ax.grid(axis="y", alpha=0.24)
            ax.set_ylim(70, 103)
            if col_idx == 0:
                ax.set_ylabel("performance index")

    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Final Recovery Index Comparison", y=1.055, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / "fig_game_vs_rl_baselines_final_recovery_index.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.2f}"


def count_wins(delta: pd.DataFrame, baseline_algorithm: str, metric: str) -> tuple[int, int]:
    subset = delta[delta["baseline_algorithm"] == baseline_algorithm].copy()
    values = pd.to_numeric(subset[f"delta_{metric}"], errors="coerce").dropna()
    return int((values > 0).sum()), int(len(values))


def write_markdown(combined: pd.DataFrame, delta: pd.DataFrame, output_dir: Path) -> None:
    ppo_final_wins, ppo_final_total = count_wins(delta, "ppo", "final_recovery_index")
    sac_final_wins, sac_final_total = count_wins(delta, "sac", "final_recovery_index")
    ppo_best_wins, ppo_best_total = count_wins(delta, "ppo", "best_recovery_index")
    sac_best_wins, sac_best_total = count_wins(delta, "sac", "best_recovery_index")

    rows = [
        "# Game-PPO vs PPO/SAC Baseline Comparison",
        "",
        "This comparison combines the data behind `fig_rl_baseline_protocol_performance.png` with the new Game-PPO recovery runs.",
        "",
        "Important: PPO/SAC baseline rows are 5-seed aggregates, while Game-PPO rows are 1-seed results. Treat this as a screening comparison, not a final statistical claim.",
        "",
        f"- Final recovery index: Game-PPO beats PPO in {ppo_final_wins}/{ppo_final_total} cases.",
        f"- Final recovery index: Game-PPO beats SAC in {sac_final_wins}/{sac_final_total} cases.",
        f"- Best recovery index: Game-PPO beats PPO in {ppo_best_wins}/{ppo_best_total} cases.",
        f"- Best recovery index: Game-PPO beats SAC in {sac_best_wins}/{sac_best_total} cases.",
        "",
        "Generated files:",
        "",
        "- `combined_performance_story.csv`: PPO/SAC/Game-PPO metrics in one table.",
        "- `game_vs_rl_baseline_deltas.csv`: Game-PPO minus PPO/SAC deltas.",
        "- `fig_game_vs_rl_baselines_protocol_performance.png`: full protocol overlay.",
        "- `fig_game_vs_rl_baselines_final_recovery_index.png`: compact final/best/shock index comparison.",
        "",
        "| Level | Difficulty | PPO final | SAC final | Game-PPO final | Game-PPO - PPO | Game-PPO - SAC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for level, _ in LEVELS:
        for difficulty, _, _ in DIFFICULTIES:
            panel = combined[(combined["level"] == level) & (combined["difficulty"] == difficulty)].set_index("algorithm")
            ppo_final = float(panel.loc["ppo", "final_recovery_index"]) if "ppo" in panel.index else math.nan
            sac_final = float(panel.loc["sac", "final_recovery_index"]) if "sac" in panel.index else math.nan
            game_final = float(panel.loc["game_ppo", "final_recovery_index"]) if "game_ppo" in panel.index else math.nan
            rows.append(
                f"| {level} | {difficulty} | {fmt_float(ppo_final)} | {fmt_float(sac_final)} | "
                f"{fmt_float(game_final)} | {fmt_float(game_final - ppo_final)} | {fmt_float(game_final - sac_final)} |"
            )

    (output_dir / "comparison_summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    baseline_root = resolve(args.baseline_root)
    game_root = resolve(args.game_root)
    baseline_summary = resolve(args.baseline_summary)
    game_summary = resolve(args.game_summary)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = build_combined_summary(baseline_summary, game_summary)
    combined.to_csv(output_dir / "combined_performance_story.csv", index=False)

    delta = build_delta_table(combined)
    delta.to_csv(output_dir / "game_vs_rl_baseline_deltas.csv", index=False)

    protocol_summary = plot_combined_protocol(baseline_root, game_root, output_dir, args.smooth_window)
    protocol_summary.to_csv(output_dir / "combined_performance_story_from_curves.csv", index=False)
    plot_final_recovery_bars(combined, output_dir)
    write_markdown(combined, delta, output_dir)

    print(f"Saved combined summary: {output_dir / 'combined_performance_story.csv'}")
    print(f"Saved deltas: {output_dir / 'game_vs_rl_baseline_deltas.csv'}")
    print(f"Saved protocol figure: {output_dir / 'fig_game_vs_rl_baselines_protocol_performance.png'}")
    print(f"Saved index figure: {output_dir / 'fig_game_vs_rl_baselines_final_recovery_index.png'}")
    print(f"Saved markdown summary: {output_dir / 'comparison_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
