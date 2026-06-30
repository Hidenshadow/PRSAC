#!/usr/bin/env python
"""Fixed-format clean-train -> attack -> recovery protocol plot.

This script intentionally standardizes the paper-style protocol figure:

- x-axis: protocol step
- y-axis: performance index, where clean nominal at attack time is 100
- clean training curve before the attack
- red attack shock at the nominal checkpoint
- recovery curve after the attack

It reads seed-level outputs directly, so it does not require a separate
aggregation step.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "rl_baselines"
DEFAULT_ALGORITHMS = ("ppo", "sac")
DEFAULT_SCENARIOS = (
    "level1_easy",
    "level1_medium",
    "level1_hard",
    "level2_easy",
    "level2_medium",
    "level2_hard",
    "level3_easy",
    "level3_medium",
    "level3_hard",
)

LEVEL_LABELS = {
    "level1": "Level 1 synthetic",
    "level2": "Level 2 lunar",
    "level3": "Level 3 Mars",
}
DIFFICULTY_LABELS = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}
DEFAULT_MAP_SIZES = {
    "easy": 40,
    "medium": 60,
    "hard": 80,
}

ALGORITHM_STYLES = {
    "ppo": {
        "label": "PPO",
        "clean_color": "#2f6b9a",
        "recovery_color": "#2f6b9a",
        "linestyle": "-",
    },
    "sac": {
        "label": "SAC",
        "clean_color": "#2a7f62",
        "recovery_color": "#2a7f62",
        "linestyle": "--",
    },
    "game_sac": {
        "label": "Game-aware SAC",
        "clean_color": "#7a3db8",
        "recovery_color": "#7a3db8",
        "linestyle": "-.",
    },
    "game_ppo": {
        "label": "Game-PPO",
        "clean_color": "#7a3db8",
        "recovery_color": "#7a3db8",
        "linestyle": "-.",
    },
}


@dataclass(frozen=True)
class ExperimentData:
    algorithm: str
    scenario: str
    experiment_dir: Path
    train: pd.DataFrame
    recovery: pd.DataFrame
    nominal_timesteps: int
    recovery_timesteps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Run root containing algorithm directories.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <root>/comparison_5seed_complete_scenarios.",
    )
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all"],
        help="Scenario names, e.g. level2_easy. Use 'all' for the standard 3x3 order.",
    )
    parser.add_argument("--seed-count", type=int, default=5, help="Experiment dir hint and complete-seed threshold.")
    parser.add_argument(
        "--require-complete-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require --seed-count completed seed directories before plotting an algorithm/scenario.",
    )
    parser.add_argument(
        "--blank-incomplete-scenarios",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Leave a scenario panel blank if any requested algorithm is missing or incomplete.",
    )
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--filename-prefix", type=str, default="fig_protocol_performance_index_9scenes_complete5seeds")
    parser.add_argument(
        "--y-limits",
        type=float,
        nargs=2,
        default=[70.0, 102.0],
        metavar=("YMIN", "YMAX"),
        help="Shared y-axis limits for all non-empty panels.",
    )
    parser.add_argument(
        "--include-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show missing panels instead of dropping incomplete scenarios.",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=0,
        help="Panel columns. Default: 1 for one scenario, otherwise up to 3.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def scenario_parts(scenario: str) -> tuple[str, str]:
    parts = scenario.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"scenario must look like level2_easy, got {scenario!r}")
    return parts[0], parts[1]


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else -1


def experiment_candidates(root: Path, algorithm: str, scenario: str, seed_count: int | None) -> Iterable[Path]:
    suffixes: list[str]
    if seed_count is None:
        suffixes = [f"{scenario}_shock_recovery_*seed*"]
    else:
        suffixes = [
            f"{scenario}_shock_recovery_{seed_count}seed",
            f"{scenario}_shock_recovery_{seed_count}seeds",
        ]

    bases = [
        root / algorithm,
        root / algorithm.lower(),
        root,
    ]
    seen: set[Path] = set()
    for base in bases:
        for suffix in suffixes:
            matches = sorted(base.glob(suffix)) if "*" in suffix else [base / suffix]
            for match in matches:
                if match in seen:
                    continue
                seen.add(match)
                yield match


def find_experiment(root: Path, algorithm: str, scenario: str, seed_count: int | None) -> Path | None:
    for candidate in experiment_candidates(root, algorithm, scenario, seed_count):
        if candidate.exists() and any(candidate.glob("seed*/nominal_training_eval.csv")):
            return candidate
    return None


def is_completed_seed_dir(seed_dir: Path) -> bool:
    return (
        (seed_dir / "nominal_training_eval.csv").exists()
        and (seed_dir / "shock_recovery_summary.csv").exists()
        and (seed_dir / "shock_recovery_curve.csv").exists()
        and (seed_dir / "checkpoints" / "checkpoint_nominal.pt").exists()
    )


def has_complete_seed_count(experiment: Path, seed_count: int | None) -> bool:
    if seed_count is None:
        return True
    completed = [seed_dir for seed_dir in sorted(experiment.glob("seed*")) if is_completed_seed_dir(seed_dir)]
    return len(completed) >= seed_count


def read_run_config(seed_dir: Path) -> dict:
    path = seed_dir / "run_config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def protocol_timesteps(experiment: Path) -> tuple[int, int]:
    for seed_dir in sorted(experiment.glob("seed*")):
        config = read_run_config(seed_dir)
        command_args = config.get("command_args", {})
        nominal = command_args.get("nominal_timesteps")
        recovery = command_args.get("recovery_timesteps")
        if nominal and recovery:
            return int(nominal), int(recovery)
    return 50_000, 20_480


def map_size_from_config(experiment: Path, difficulty: str) -> int:
    for seed_dir in sorted(experiment.glob("seed*")):
        config = read_run_config(seed_dir)
        level_config = config.get("command_args", {}).get("level_config")
        if not level_config:
            continue
        path = Path(level_config)
        path = path if path.is_absolute() else PROJECT_ROOT / path
        if not path.exists():
            continue
        try:
            level_data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        description = str(level_data.get("description", ""))
        match = re.search(r"(\d+)x\1", description)
        if match:
            return int(match.group(1))
    return DEFAULT_MAP_SIZES.get(difficulty, 40)


def normalized_training_curve(experiment: Path, smooth_window: int) -> pd.DataFrame:
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
        .sort_values("global_step")
    )
    grouped["performance_smooth"] = grouped["performance_mean"].rolling(
        window=max(1, smooth_window),
        center=True,
        min_periods=1,
    ).mean()
    return grouped


def normalized_recovery_curve(experiment: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(experiment.glob("seed*/shock_recovery_curve.csv")):
        curve = pd.read_csv(csv_path)
        required = {
            "eval_domain",
            "phase",
            "attack_type",
            "recovery_step",
            "mean_attacked_scalar_cost",
        }
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
        merged["seed"] = seed_from_path(csv_path)
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
        .rename(columns={"mean": "performance_mean", "std": "performance_std", "count": "num_samples"})
        .sort_values("recovery_step")
    )
    return grouped


def load_experiment(
    root: Path,
    algorithm: str,
    scenario: str,
    seed_count: int | None,
    smooth_window: int,
    require_complete_seeds: bool = False,
) -> ExperimentData | None:
    experiment = find_experiment(root, algorithm, scenario, seed_count)
    if experiment is None:
        return None
    if require_complete_seeds and not has_complete_seed_count(experiment, seed_count):
        return None
    train = normalized_training_curve(experiment, smooth_window)
    recovery = normalized_recovery_curve(experiment)
    if train.empty or recovery.empty:
        return None
    nominal_timesteps, recovery_timesteps = protocol_timesteps(experiment)
    return ExperimentData(
        algorithm=algorithm,
        scenario=scenario,
        experiment_dir=experiment,
        train=train,
        recovery=recovery,
        nominal_timesteps=nominal_timesteps,
        recovery_timesteps=recovery_timesteps,
    )


def algorithm_style(algorithm: str) -> dict[str, str]:
    key = algorithm.lower()
    if key in ALGORITHM_STYLES:
        return ALGORITHM_STYLES[key]
    return {
        "label": algorithm.upper(),
        "clean_color": "#b0b0b0",
        "recovery_color": "#4d4d4d",
        "linestyle": "-",
    }


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


def summary_row(data: ExperimentData) -> dict[str, object]:
    recovery = data.recovery
    shock = float(recovery.iloc[0]["performance_mean"])
    final = float(recovery.iloc[-1]["performance_mean"])
    best = float(recovery["performance_mean"].max())
    attack_drop = 100.0 - shock
    final_gain = final - shock
    best_gain = best - shock
    return {
        "algorithm": data.algorithm,
        "scenario": data.scenario,
        "experiment_dir": str(data.experiment_dir),
        "train_start_index": float(data.train.iloc[0]["performance_mean"]),
        "train_end_index": float(data.train.iloc[-1]["performance_mean"]),
        "attack_shock_index": shock,
        "final_recovery_index": final,
        "best_recovery_index": best,
        "attack_drop_index_points": attack_drop,
        "final_recovered_index_points": final_gain,
        "best_recovered_index_points": best_gain,
        "final_recovery_closure_pct": 100.0 * final_gain / attack_drop if attack_drop >= 5.0 else math.nan,
        "best_recovery_closure_pct": 100.0 * best_gain / attack_drop if attack_drop >= 5.0 else math.nan,
    }


def plot_panel(
    ax: plt.Axes,
    scenario: str,
    data_by_algorithm: dict[str, ExperimentData],
    y_limits: tuple[float, float] | None,
) -> list[pd.Series]:
    level, difficulty = scenario_parts(scenario)
    panel_values: list[pd.Series] = []
    nominal_end = 50_000
    recovery_end = 20_480

    if data_by_algorithm:
        first = next(iter(data_by_algorithm.values()))
        nominal_end = first.nominal_timesteps
        recovery_end = first.recovery_timesteps
        map_size = map_size_from_config(first.experiment_dir, difficulty)
    else:
        map_size = DEFAULT_MAP_SIZES.get(difficulty, 40)

    for algorithm, data in data_by_algorithm.items():
        style = algorithm_style(algorithm)
        train_x = data.train["global_step"].to_numpy(dtype=float).copy()
        train_y = data.train["performance_smooth"].to_numpy(dtype=float).copy()
        if train_x.size == 0:
            continue
        if train_x[-1] < nominal_end:
            train_x = np.append(train_x, nominal_end)
            train_y = np.append(train_y, 100.0)
        else:
            train_x[-1] = nominal_end
            train_y[-1] = 100.0

        recovery = data.recovery.copy()
        recovery["timeline_step"] = nominal_end + recovery["recovery_step"].astype(int)
        recovery_end = max(recovery_end, int(recovery["recovery_step"].max()))

        ax.plot(
            train_x,
            train_y,
            color=style["clean_color"],
            linestyle=style["linestyle"],
            linewidth=2.7,
            alpha=1.0,
            label=f"{style['label']} clean training",
        )
        shock_y = float(recovery.iloc[0]["performance_mean"])
        ax.plot([nominal_end, nominal_end], [100.0, shock_y], color="#c23b22", linewidth=1.6, alpha=0.75)
        ax.scatter([nominal_end], [shock_y], color="#c23b22", s=30, zorder=4)
        ax.plot(
            recovery["timeline_step"],
            recovery["performance_mean"],
            color=style["recovery_color"],
            linestyle=style["linestyle"],
            linewidth=2.4,
            marker="o",
            markersize=3.1,
            label=f"{style['label']} recovery",
        )
        panel_values.extend([pd.Series(train_y), recovery["performance_mean"]])

    ax.axhline(100.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.axvline(nominal_end, color="#c23b22", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.grid(alpha=0.24)
    ax.set_xlim(0, nominal_end + recovery_end)
    ax.set_xticks([0, nominal_end // 2, nominal_end, nominal_end + 20_000])
    ax.set_xticklabels(["0", "25k", "50k", "70k"])
    ax.set_title(
        f"{LEVEL_LABELS.get(level, level)}\n{DIFFICULTY_LABELS.get(difficulty, difficulty.title())} ({map_size}x{map_size})"
    )

    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif panel_values:
        y_values = pd.concat([*panel_values, pd.Series([100.0])], ignore_index=True)
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        span = max(12.0, y_max - y_min)
        center = 0.5 * (y_min + y_max)
        ax.set_ylim(
            min(y_min - 0.10 * span, center - 0.55 * span),
            max(y_max + 0.10 * span, center + 0.55 * span),
        )
    return panel_values


def hide_unused_axes(axes: np.ndarray, start: int) -> None:
    for ax in axes.ravel()[start:]:
        ax.axis("off")


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    output_dir = resolve(args.output_dir) if args.output_dir else root / "comparison_5seed_complete_scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = list(DEFAULT_SCENARIOS) if args.scenarios == ["all"] else args.scenarios

    loaded: dict[str, dict[str, ExperimentData]] = {}
    summary_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for scenario in scenarios:
        loaded[scenario] = {}
        for algorithm in args.algorithms:
            data = load_experiment(
                root,
                algorithm,
                scenario,
                args.seed_count,
                args.smooth_window,
                args.require_complete_seeds,
            )
            if data is None:
                missing.append(f"{algorithm}/{scenario}")
                continue
            loaded[scenario][algorithm] = data
            summary_rows.append(summary_row(data))

        if args.blank_incomplete_scenarios and len(loaded[scenario]) != len(args.algorithms):
            missing_algorithms = set(args.algorithms) - set(loaded[scenario])
            for algorithm in sorted(loaded[scenario]):
                missing.append(f"{algorithm}/{scenario} blanked because {sorted(missing_algorithms)} incomplete")
            loaded[scenario] = {}
            summary_rows = [row for row in summary_rows if row["scenario"] != scenario]

    plotted_scenarios = [
        scenario for scenario in scenarios if loaded.get(scenario) or args.include_missing
    ]
    if not plotted_scenarios:
        details = "\n".join(f"- {item}" for item in missing[:20])
        raise SystemExit(f"No completed scenario data found under {root}\n{details}")

    setup_matplotlib()
    n_panels = len(plotted_scenarios)
    if args.ncols > 0:
        ncols = args.ncols
    else:
        ncols = 1 if n_panels == 1 else min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig_width = 7.2 * ncols
    fig_height = 4.5 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False, sharex=False)

    for idx, scenario in enumerate(plotted_scenarios):
        ax = axes.ravel()[idx]
        if not loaded.get(scenario):
            level, difficulty = scenario_parts(scenario)
            ax.text(0.5, 0.5, "missing complete 5 seeds", transform=ax.transAxes, ha="center", va="center", color="0.35")
            ax.set_title(f"{LEVEL_LABELS.get(level, level)}\n{DIFFICULTY_LABELS.get(difficulty, difficulty.title())}")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue
        y_limits = tuple(float(value) for value in args.y_limits) if args.y_limits else None
        plot_panel(ax, scenario, loaded[scenario], y_limits)
        if idx % ncols == 0:
            ax.set_ylabel("performance index\n(clean nominal = 100)")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("protocol step")

    hide_unused_axes(axes, len(plotted_scenarios))

    handles: list[object] = []
    labels: list[str] = []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        legend_cols = min(len(args.algorithms), len(labels))
        fig.legend(handles, labels, loc="upper center", ncol=legend_cols, frameon=False, bbox_to_anchor=(0.5, 1.02))

    title = args.title or "PPO/SAC baseline protocol, complete 5-seed scenarios only"
    title_y = 1.08 if handles else 1.02
    fig.suptitle(title, y=title_y, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / f"{args.filename_prefix}.png"
    pdf_path = output_dir / f"{args.filename_prefix}.pdf"
    summary_path = output_dir / "protocol_performance_index_summary.csv"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved summary: {summary_path}")
    if missing:
        print("Missing/incomplete runs skipped:")
        for item in missing:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
