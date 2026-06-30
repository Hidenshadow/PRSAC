#!/usr/bin/env python
"""Compare completed adaptive-bandit recovery runs with PPO/SAC/Game-PPO baselines."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
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
DEFAULT_ADAPTIVE_SOURCE_ROOTS = (
    PROJECT_ROOT / "runs" / "adaptive_bandit_game_recovery_matrix",
    PROJECT_ROOT / "runs" / "adaptive_bandit_game_recovery_extra_seed1",
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "adaptive_bandit_protocol_analysis" / "compare_completed_with_rl_baselines"
DEFAULT_BASELINE_SUMMARY = DEFAULT_BASELINE_ROOT / "paper_story_5seeds" / "rl_baseline_performance_story.csv"
DEFAULT_GAME_SUMMARY = DEFAULT_GAME_ROOT / "paper_story_1seed" / "game_recovery_performance_story.csv"

ALGORITHM_ORDER = {
    "ppo": 0,
    "sac": 1,
    "game_ppo": 2,
    "adaptive_bandit_ppo": 3,
    "ap_cvar_ppo": 4,
}
COLORS = {
    "ppo": "#2f6b9a",
    "sac": "#2a7f62",
    "game_ppo": "#7a3db8",
    "adaptive_bandit_ppo": "#c06b2c",
    "ap_cvar_ppo": "#b23a48",
}
LINESTYLES = {
    "ppo": "-",
    "sac": "--",
    "game_ppo": "-.",
    "adaptive_bandit_ppo": ":",
    "ap_cvar_ppo": ":",
}
LABELS = {
    "ppo": "PPO (5 seeds)",
    "sac": "SAC (5 seeds)",
    "game_ppo": "Game-PPO (1 seed)",
    "adaptive_bandit_ppo": "Adaptive-Bandit PPO (completed)",
    "ap_cvar_ppo": "AP-CVaR PPO (completed)",
}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--game-root", type=Path, default=DEFAULT_GAME_ROOT)
    parser.add_argument("--adaptive-source-roots", nargs="*", type=Path, default=list(DEFAULT_ADAPTIVE_SOURCE_ROOTS))
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--game-summary", type=Path, default=DEFAULT_GAME_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--algorithm-name", type=str, default="adaptive_bandit_ppo")
    parser.add_argument("--algorithm-label", type=str, default=None)
    parser.add_argument("--algorithm-tick-label", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    if match is None:
        return -1
    return int(match.group(1))


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


def discover_adaptive_seed_dirs(source_roots: list[Path]) -> dict[tuple[str, str], list[Path]]:
    discovered: dict[tuple[str, str], dict[int, Path]] = {}
    for root in source_roots:
        if not root.exists():
            continue
        for level, _ in LEVELS:
            for difficulty, _, _ in DIFFICULTIES:
                experiment_dir_name = f"{level}_{difficulty}_shock_recovery_1seed"
                for seed_dir in sorted((root / experiment_dir_name).glob("seed*")):
                    if not seed_dir.is_dir():
                        continue
                    if not (seed_dir / "shock_recovery_curve.csv").exists():
                        continue
                    if not (seed_dir / "nominal_training_eval.csv").exists():
                        continue
                    discovered.setdefault((level, difficulty), {})[seed_from_path(seed_dir)] = seed_dir
    return {key: [items[seed] for seed in sorted(items)] for key, items in discovered.items()}


def normalized_training_curve_from_seed_dirs(seed_dirs: list[Path], smooth_window: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed_dir in seed_dirs:
        csv_path = seed_dir / "nominal_training_eval.csv"
        frame = pd.read_csv(csv_path)
        required = {"global_step", "mean_scalar_cost"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[["global_step", "mean_scalar_cost"]].dropna().sort_values("global_step")
        if frame.empty:
            continue
        final_cost = float(frame.iloc[-1]["mean_scalar_cost"])
        if not math.isfinite(final_cost) or final_cost <= 0.0:
            continue
        frame = frame.copy()
        frame["seed"] = seed_from_path(seed_dir)
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


def normalized_shock_recovery_curve_from_seed_dirs(seed_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed_dir in seed_dirs:
        curve = pd.read_csv(seed_dir / "shock_recovery_curve.csv")
        required = {"eval_domain", "phase", "attack_type", "recovery_step", "mean_attacked_scalar_cost"}
        if not required.issubset(curve.columns):
            continue
        clean = curve[(curve["phase"] == "shock") & (curve["attack_type"] == "none")][
            ["eval_domain", "mean_attacked_scalar_cost"]
        ].rename(columns={"mean_attacked_scalar_cost": "clean_nominal_cost"})
        attacked = curve[
            (curve["attack_type"] == "environment")
            & (curve["phase"].isin(["shock", "recovery"]))
        ][["eval_domain", "phase", "recovery_step", "mean_attacked_scalar_cost"]]
        if clean.empty or attacked.empty:
            continue
        merged = attacked.merge(clean, on="eval_domain", how="inner")
        merged["seed"] = seed_from_path(seed_dir)
        merged["performance_index"] = (
            100.0
            * merged["clean_nominal_cost"].astype(float)
            / merged["mean_attacked_scalar_cost"].astype(float).clip(lower=1e-12)
        )
        frames.append(merged[["seed", "eval_domain", "phase", "recovery_step", "performance_index"]])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby("recovery_step")["performance_index"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "performance_mean", "std": "performance_std", "count": "num_eval_points"})
        .sort_values("recovery_step")
    )
    return grouped


def load_summary(path: Path, seed_count: int, source_label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.insert(1, "seed_count", seed_count)
    frame.insert(2, "source", source_label)
    return frame


def adaptive_summary(
    discovered: dict[tuple[str, str], list[Path]],
    smooth_window: int,
    algorithm_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for level, _ in LEVELS:
        for difficulty, _, _ in DIFFICULTIES:
            seed_dirs = discovered.get((level, difficulty), [])
            if not seed_dirs:
                continue
            train = normalized_training_curve_from_seed_dirs(seed_dirs, smooth_window)
            recovery = normalized_shock_recovery_curve_from_seed_dirs(seed_dirs)
            if train.empty or recovery.empty:
                continue
            row = performance_summary_row(algorithm_name, level, difficulty, train, recovery)
            row["seed_count"] = len(seed_dirs)
            row["source"] = ";".join(str(path) for path in seed_dirs)
            rows.append(row)
    return pd.DataFrame(rows)


def sort_summary(frame: pd.DataFrame) -> pd.DataFrame:
    level_order = {level: idx for idx, (level, _) in enumerate(LEVELS)}
    difficulty_order = {difficulty: idx for idx, (difficulty, _, _) in enumerate(DIFFICULTIES)}
    frame = frame.copy()
    frame["_level_order"] = frame["level"].map(level_order)
    frame["_difficulty_order"] = frame["difficulty"].map(difficulty_order)
    frame["_algorithm_order"] = frame["algorithm"].map(ALGORITHM_ORDER).fillna(99)
    return frame.sort_values(["_level_order", "_difficulty_order", "_algorithm_order"]).drop(
        columns=["_level_order", "_difficulty_order", "_algorithm_order"]
    )


def build_combined_summary(
    baseline_summary: Path,
    game_summary: Path,
    adaptive: pd.DataFrame,
) -> pd.DataFrame:
    baseline = load_summary(baseline_summary, 5, "rl_baselines/paper_story_5seeds")
    game = load_summary(game_summary, 1, "game_recovery_protocol_analysis/paper_story_1seed")
    if adaptive.empty:
        combined = pd.concat([baseline, game], ignore_index=True)
    else:
        combined = pd.concat([baseline, game, adaptive], ignore_index=True)
    return sort_summary(combined)


def standard_curve(
    root: Path,
    algorithm: str,
    level: str,
    difficulty: str,
    seed_count: int,
    smooth_window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment = experiment_dir(root, algorithm, level, difficulty, seed_count)
    return normalized_training_curve(experiment, smooth_window), normalized_shock_recovery_curve(experiment)


def plot_protocol(
    baseline_root: Path,
    game_root: Path,
    discovered: dict[tuple[str, str], list[Path]],
    output_dir: Path,
    smooth_window: int,
    algorithm_name: str,
    output_prefix: str,
) -> pd.DataFrame:
    setup_matplotlib()
    fig, axes = plt.subplots(len(LEVELS), len(DIFFICULTIES), figsize=(15.8, 11.2), sharex=True)
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

            standard_specs = [
                ("ppo", baseline_root, 5),
                ("sac", baseline_root, 5),
                ("game_ppo", game_root, 1),
            ]
            curve_specs: list[tuple[str, pd.DataFrame, pd.DataFrame, int]] = []
            for algorithm, root, seed_count in standard_specs:
                train, recovery = standard_curve(root, algorithm, level, difficulty, seed_count, smooth_window)
                curve_specs.append((algorithm, train, recovery, seed_count))

            adaptive_dirs = discovered.get((level, difficulty), [])
            if adaptive_dirs:
                curve_specs.append(
                    (
                        algorithm_name,
                        normalized_training_curve_from_seed_dirs(adaptive_dirs, smooth_window),
                        normalized_shock_recovery_curve_from_seed_dirs(adaptive_dirs),
                        len(adaptive_dirs),
                    )
                )

            for algorithm, train, recovery, seed_count in curve_specs:
                if train.empty or recovery.empty:
                    continue
                plotted_any = True
                summary = performance_summary_row(algorithm, level, difficulty, train, recovery)
                summary["seed_count"] = seed_count
                summary_rows.append(summary)

                nominal_end = int(train["global_step"].max())
                nominal_end_for_axis = max(nominal_end_for_axis, nominal_end)
                recovery = recovery.copy()
                recovery["timeline_step"] = nominal_end + recovery["recovery_step"].astype(int)
                max_recovery_for_axis = int(max(max_recovery_for_axis, recovery["recovery_step"].max()))

                color = COLORS[algorithm]
                linestyle = LINESTYLES[algorithm]
                ax.plot(
                    train["global_step"],
                    train["performance_smooth"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.75,
                    alpha=0.55 if algorithm != algorithm_name else 0.72,
                )
                shock_y = float(recovery.iloc[0]["performance_mean"])
                ax.plot([nominal_end, nominal_end], [100.0, shock_y], color=attack_color, linewidth=1.25, alpha=0.45)
                ax.scatter([nominal_end], [shock_y], color=attack_color, s=22, zorder=4)
                ax.plot(
                    recovery["timeline_step"],
                    recovery["performance_mean"],
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.25 if algorithm != algorithm_name else 2.75,
                    marker="o",
                    markersize=3.0,
                    label=LABELS[algorithm],
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
            ax.set_ylim(min(y_min - 0.10 * span, center - 0.55 * span), max(y_max + 0.10 * span, center + 0.55 * span))
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
    fig.suptitle(f"PPO/SAC/Game-PPO vs {LABELS[algorithm_name]}: Completed Runs", y=1.075, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / f"fig_{output_prefix}_protocol_performance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return sort_summary(pd.DataFrame(summary_rows))


def plot_final_recovery_bars(
    combined: pd.DataFrame,
    output_dir: Path,
    algorithm_name: str,
    algorithm_tick_label: str,
    output_prefix: str,
) -> None:
    setup_matplotlib()
    fig, axes = plt.subplots(len(LEVELS), len(DIFFICULTIES), figsize=(15.2, 10.4), sharey=True)
    axes = np.asarray(axes).reshape(len(LEVELS), len(DIFFICULTIES))
    algorithms = ["ppo", "sac", "game_ppo", algorithm_name]
    tick_labels = ["PPO\n5 seeds", "SAC\n5 seeds", "Game-PPO\n1 seed", algorithm_tick_label]

    for row_idx, (level, level_label) in enumerate(LEVELS):
        for col_idx, (difficulty, difficulty_label, map_size) in enumerate(DIFFICULTIES):
            ax = axes[row_idx, col_idx]
            panel = combined[(combined["level"] == level) & (combined["difficulty"] == difficulty)].set_index("algorithm")
            x = np.arange(len(algorithms))
            finals = [float(panel.loc[alg, "final_recovery_index"]) if alg in panel.index else math.nan for alg in algorithms]
            bests = [float(panel.loc[alg, "best_recovery_index"]) if alg in panel.index else math.nan for alg in algorithms]
            shocks = [float(panel.loc[alg, "attack_shock_index"]) if alg in panel.index else math.nan for alg in algorithms]

            ax.bar(x, finals, color=[COLORS[alg] for alg in algorithms], alpha=0.78, width=0.62)
            ax.scatter(x, bests, color="black", s=24, marker="D", label="best recovery" if row_idx == 0 and col_idx == 0 else None)
            ax.scatter(x, shocks, color="#c23b22", s=28, marker="x", label="attack shock" if row_idx == 0 and col_idx == 0 else None)
            if algorithm_name not in panel.index:
                ax.text(x[-1], 72.0, "pending", rotation=90, ha="center", va="bottom", fontsize=7, color="0.35")
            ax.axhline(100.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, fontsize=7)
            ax.set_title(f"{level_label}\n{difficulty_label} ({map_size}x{map_size})")
            ax.grid(axis="y", alpha=0.24)
            ax.set_ylim(70, 103)
            if col_idx == 0:
                ax.set_ylabel("performance index")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle(f"Final Recovery Index: Completed {LABELS[algorithm_name]} Runs Added", y=1.055, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_dir / f"fig_{output_prefix}_final_recovery_index.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_adaptive_delta_table(combined: pd.DataFrame, algorithm_name: str) -> pd.DataFrame:
    adaptive = combined[combined["algorithm"] == algorithm_name].set_index(["level", "difficulty"])
    rows: list[dict[str, object]] = []
    for baseline_algorithm in ["ppo", "sac", "game_ppo"]:
        baseline = combined[combined["algorithm"] == baseline_algorithm].set_index(["level", "difficulty"])
        for key in adaptive.index.intersection(baseline.index):
            adaptive_row = adaptive.loc[key]
            baseline_row = baseline.loc[key]
            row: dict[str, object] = {
                "level": key[0],
                "difficulty": key[1],
                "baseline_algorithm": baseline_algorithm,
                "adaptive_seed_count": int(adaptive_row["seed_count"]),
            }
            for metric in SUMMARY_METRICS:
                adaptive_value = float(adaptive_row[metric]) if pd.notna(adaptive_row[metric]) else math.nan
                baseline_value = float(baseline_row[metric]) if pd.notna(baseline_row[metric]) else math.nan
                row[f"baseline_{metric}"] = baseline_value
                row[f"adaptive_{metric}"] = adaptive_value
                row[f"delta_{metric}"] = adaptive_value - baseline_value
            rows.append(row)
    return pd.DataFrame(rows)


def count_wins(delta: pd.DataFrame, baseline_algorithm: str, metric: str) -> tuple[int, int]:
    subset = delta[delta["baseline_algorithm"] == baseline_algorithm]
    values = pd.to_numeric(subset[f"delta_{metric}"], errors="coerce").dropna()
    return int((values > 0).sum()), int(len(values))


def fmt_float(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.2f}"


def write_markdown(
    combined: pd.DataFrame,
    adaptive: pd.DataFrame,
    delta: pd.DataFrame,
    output_dir: Path,
    algorithm_name: str,
    output_prefix: str,
) -> None:
    adaptive_cases = int(len(adaptive))
    ppo_wins, ppo_total = count_wins(delta, "ppo", "final_recovery_index")
    sac_wins, sac_total = count_wins(delta, "sac", "final_recovery_index")
    game_wins, game_total = count_wins(delta, "game_ppo", "final_recovery_index")
    algorithm_label = LABELS[algorithm_name]
    rows = [
        f"# {algorithm_label} Completed Comparison",
        "",
        f"This comparison adds completed {algorithm_label} runs.",
        "",
        "Important: PPO/SAC are 5-seed baseline aggregates; Game-PPO is 1 seed.",
        "",
        f"- Completed adaptive panels included: {adaptive_cases}.",
        f"- Final recovery index: {algorithm_label} beats PPO in {ppo_wins}/{ppo_total} completed comparable panels.",
        f"- Final recovery index: {algorithm_label} beats SAC in {sac_wins}/{sac_total} completed comparable panels.",
        f"- Final recovery index: {algorithm_label} beats Game-PPO in {game_wins}/{game_total} completed comparable panels.",
        "",
        "Generated files:",
        "",
        f"- `fig_{output_prefix}_protocol_performance.png`",
        f"- `fig_{output_prefix}_final_recovery_index.png`",
        "- `combined_completed_performance_story.csv`",
        f"- `{algorithm_name}_completed_performance_story.csv`",
        f"- `{output_prefix}_deltas.csv`",
        "",
        "| Level | Difficulty | New algo seeds | PPO final | SAC final | Game-PPO final | New algo final | New algo - PPO | New algo - Game-PPO |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, adaptive_row in adaptive.iterrows():
        level = str(adaptive_row["level"])
        difficulty = str(adaptive_row["difficulty"])
        panel = combined[(combined["level"] == level) & (combined["difficulty"] == difficulty)].set_index("algorithm")
        ppo_final = float(panel.loc["ppo", "final_recovery_index"]) if "ppo" in panel.index else math.nan
        sac_final = float(panel.loc["sac", "final_recovery_index"]) if "sac" in panel.index else math.nan
        game_final = float(panel.loc["game_ppo", "final_recovery_index"]) if "game_ppo" in panel.index else math.nan
        adaptive_final = float(adaptive_row["final_recovery_index"])
        rows.append(
            f"| {level} | {difficulty} | {int(adaptive_row['seed_count'])} | "
            f"{fmt_float(ppo_final)} | {fmt_float(sac_final)} | {fmt_float(game_final)} | "
            f"{fmt_float(adaptive_final)} | {fmt_float(adaptive_final - ppo_final)} | "
            f"{fmt_float(adaptive_final - game_final)} |"
        )
    (output_dir / "comparison_summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    algorithm_name = str(args.algorithm_name)
    output_prefix = str(args.output_prefix or f"{algorithm_name}_vs_rl_baselines")
    if algorithm_name not in ALGORITHM_ORDER:
        ALGORITHM_ORDER[algorithm_name] = max(ALGORITHM_ORDER.values(), default=3) + 1
    if args.algorithm_label:
        LABELS[algorithm_name] = str(args.algorithm_label)
    elif algorithm_name not in LABELS:
        LABELS[algorithm_name] = algorithm_name
    if algorithm_name not in COLORS:
        COLORS[algorithm_name] = "#b23a48"
    if algorithm_name not in LINESTYLES:
        LINESTYLES[algorithm_name] = ":"
    algorithm_tick_label = str(args.algorithm_tick_label or LABELS[algorithm_name].replace(" ", "\n"))

    baseline_root = resolve(args.baseline_root)
    game_root = resolve(args.game_root)
    adaptive_source_roots = [resolve(path) for path in args.adaptive_source_roots]
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = discover_adaptive_seed_dirs(adaptive_source_roots)
    adaptive = adaptive_summary(discovered, args.smooth_window, algorithm_name)
    adaptive_summary_path = output_dir / f"{algorithm_name}_completed_performance_story.csv"
    adaptive.to_csv(adaptive_summary_path, index=False)

    combined = build_combined_summary(resolve(args.baseline_summary), resolve(args.game_summary), adaptive)
    combined.to_csv(output_dir / "combined_completed_performance_story.csv", index=False)

    delta = build_adaptive_delta_table(combined, algorithm_name)
    delta_path = output_dir / f"{output_prefix}_deltas.csv"
    delta.to_csv(delta_path, index=False)

    protocol_summary = plot_protocol(
        baseline_root,
        game_root,
        discovered,
        output_dir,
        args.smooth_window,
        algorithm_name,
        output_prefix,
    )
    protocol_summary.to_csv(output_dir / "combined_completed_performance_story_from_curves.csv", index=False)
    plot_final_recovery_bars(combined, output_dir, algorithm_name, algorithm_tick_label, output_prefix)
    write_markdown(combined, adaptive, delta, output_dir, algorithm_name, output_prefix)

    print(f"Completed adaptive panels: {len(adaptive)}")
    print(f"Saved protocol figure: {output_dir / f'fig_{output_prefix}_protocol_performance.png'}")
    print(f"Saved index figure: {output_dir / f'fig_{output_prefix}_final_recovery_index.png'}")
    print(f"Saved combined summary: {output_dir / 'combined_completed_performance_story.csv'}")
    print(f"Saved adaptive summary: {adaptive_summary_path}")
    print(f"Saved deltas: {delta_path}")
    print(f"Saved markdown summary: {output_dir / 'comparison_summary.md'}")
    if not adaptive.empty:
        print(adaptive[["algorithm", "level", "difficulty", "seed_count", "final_recovery_index", "best_recovery_index"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
