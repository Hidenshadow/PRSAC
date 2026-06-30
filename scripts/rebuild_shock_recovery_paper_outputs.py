#!/usr/bin/env python
"""Rebuild paper-facing PPO shock-recovery summaries from curve CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_shock_recovery_experiment import plot_outputs, write_output_guide  # noqa: E402
from utils.paper_metrics import numeric_summary_columns, summarize_shock_recovery_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild paper summaries from completed shock-recovery curves.")
    parser.add_argument("root", type=Path, nargs="?", default=PROJECT_ROOT / "runs" / "ppo_difficulty")
    parser.add_argument("--expected-seeds", type=int, default=5)
    parser.add_argument("--include-partial-aggregates", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def infer_level_difficulty(name: str) -> tuple[str, str]:
    match = re.search(r"(level\d+)_(easy|medium|hard)_shock_recovery_(\d+)seeds", name)
    if match is None:
        return "", ""
    return match.group(1), match.group(2)


def infer_algorithm(experiment_dir: Path, root: Path) -> str:
    try:
        relative = experiment_dir.relative_to(root)
    except ValueError:
        relative = experiment_dir
    if len(relative.parts) >= 2 and relative.parts[0] in {"ppo", "sac"}:
        return relative.parts[0]
    return "ppo"


def seed_from_dir(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if match is None:
        raise ValueError(f"could not infer seed from {path}")
    return int(match.group(1))


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in frame.columns
    ]
    return frame


def rebuild_seed(seed_dir: Path, write_plots: bool) -> pd.DataFrame | None:
    curve_path = seed_dir / "shock_recovery_curve.csv"
    if not curve_path.exists():
        return None
    curve = pd.read_csv(curve_path)
    summary = summarize_shock_recovery_frame(curve)
    summary.to_csv(seed_dir / "shock_recovery_summary.csv", index=False)
    if write_plots:
        plot_outputs(curve, seed_dir)
        write_output_guide(seed_dir)
    return summary


def aggregate_root(experiment_dir: Path, expected_seeds: int, include_partial: bool) -> None:
    curve_count = len(list(experiment_dir.glob("seed*/shock_recovery_curve.csv")))
    if curve_count == 0:
        return
    if curve_count < expected_seeds and not include_partial:
        return
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "aggregate_shock_recovery_3seeds.py"), str(experiment_dir)],
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def main() -> int:
    args = parse_args()
    root = args.root if args.root.is_absolute() else PROJECT_ROOT / args.root
    write_plots = not args.skip_plots

    completed_rows: list[pd.DataFrame] = []
    for experiment_dir in sorted(root.glob("**/*_shock_recovery_*seeds")):
        level, difficulty = infer_level_difficulty(experiment_dir.name)
        if not level or not difficulty:
            continue
        algorithm = infer_algorithm(experiment_dir, root)
        curve_count = len(list(experiment_dir.glob("seed*/shock_recovery_curve.csv")))
        if curve_count < args.expected_seeds and not args.include_partial_aggregates:
            print(
                f"Skipping {experiment_dir}: found {curve_count} completed seeds, "
                f"expected {args.expected_seeds}",
                flush=True,
            )
            continue
        for seed_dir in sorted(experiment_dir.glob("seed*")):
            summary = rebuild_seed(seed_dir, write_plots)
            if summary is None:
                continue
            summary = summary.copy()
            summary.insert(0, "training_seed", seed_from_dir(seed_dir))
            summary.insert(0, "difficulty", difficulty)
            summary.insert(0, "level", level)
            summary.insert(0, "algorithm", algorithm)
            summary.insert(0, "experiment", experiment_dir.name)
            completed_rows.append(summary)
        aggregate_root(experiment_dir, args.expected_seeds, args.include_partial_aggregates)

    if not completed_rows:
        raise SystemExit(f"No completed shock_recovery_curve.csv files found under {root}")

    combined = pd.concat(completed_rows, ignore_index=True)
    combined_path = root / "paper_summary_all_completed_seeds.csv"
    combined.to_csv(combined_path, index=False)

    numeric_cols = numeric_summary_columns(combined)
    aggregate = (
        combined.groupby(["algorithm", "experiment", "level", "difficulty", "eval_domain"])[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate = flatten_columns(aggregate)
    seed_counts = (
        combined.groupby(["algorithm", "experiment", "level", "difficulty", "eval_domain"])["training_seed"]
        .nunique()
        .reset_index(name="num_training_seeds")
    )
    aggregate = aggregate.merge(seed_counts, on=["algorithm", "experiment", "level", "difficulty", "eval_domain"], how="left")
    aggregate_path = root / "paper_summary_aggregate_completed_seeds.csv"
    aggregate.to_csv(aggregate_path, index=False)

    print(f"Saved paper seed summary: {combined_path}")
    print(f"Saved paper aggregate summary: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
