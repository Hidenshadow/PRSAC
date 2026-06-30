#!/usr/bin/env python
"""Aggregate PPO shock-recovery runs across seeds."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.paper_metrics import numeric_summary_columns, summarize_shock_recovery_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate seed*/ PPO shock-recovery outputs.")
    parser.add_argument("output_root", type=Path)
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    if match is None:
        raise ValueError(f"could not infer seed from path: {path}")
    return int(match.group(1))


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in frame.columns
    ]
    return frame


def main() -> int:
    args = parse_args()
    output_root = args.output_root

    curve_frames = []
    for path in sorted(output_root.glob("seed*/shock_recovery_curve.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "training_seed", seed_from_path(path))
        curve_frames.append(frame)
    if not curve_frames:
        raise SystemExit(f"No shock_recovery_curve.csv files found under {output_root}")

    combined_curve = pd.concat(curve_frames, ignore_index=True)
    combined_curve_path = output_root / "shock_recovery_curve_all_seeds.csv"
    combined_curve.to_csv(combined_curve_path, index=False)

    curve_metric_cols = [
        "mean_attacked_scalar_cost",
        "std_attacked_scalar_cost",
        "absolute_degradation",
        "relative_degradation",
        "success_rate",
        "mean_path_length",
        "mean_attacked_cell_exposure_ratio",
        "mean_hazard_exposure",
        "mean_belief_hazard_exposure",
        "mean_uncertainty_exposure",
        "mean_belief_uncertainty_exposure",
        "mean_belief_scalar_cost",
        "mean_lambda_uncertainty",
        "mean_map_mismatch_penalty",
        "mean_map_mismatch_abs_error",
        "mean_path_confidence",
        "true_belief_mismatch_rate",
        "mean_mismatched_cells",
        "mean_belief_abs_error",
        "mean_true_minus_belief_error",
        "mean_selected_confidence",
    ]
    existing_curve_metrics = [col for col in curve_metric_cols if col in combined_curve.columns]
    curve_aggregate = (
        combined_curve.groupby(
            ["eval_domain", "phase", "attack_type", "recovery_step", "checkpoint_role"]
        )[existing_curve_metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    curve_aggregate = flatten_columns(curve_aggregate)
    curve_aggregate_path = output_root / "shock_recovery_curve_aggregate.csv"
    curve_aggregate.to_csv(curve_aggregate_path, index=False)

    summary_frames = []
    for path in sorted(output_root.glob("seed*/shock_recovery_curve.csv")):
        frame = pd.read_csv(path)
        summary = summarize_shock_recovery_frame(frame)
        summary.to_csv(path.parent / "shock_recovery_summary.csv", index=False)
        summary.insert(0, "training_seed", seed_from_path(path))
        summary_frames.append(summary)
    if summary_frames:
        combined_summary = pd.concat(summary_frames, ignore_index=True)
        combined_summary_path = output_root / "shock_recovery_summary_all_seeds.csv"
        combined_summary.to_csv(combined_summary_path, index=False)

        existing_summary_metrics = numeric_summary_columns(combined_summary)
        summary_aggregate = (
            combined_summary.groupby(["eval_domain"])[existing_summary_metrics]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary_aggregate = flatten_columns(summary_aggregate)
        seed_counts = (
            combined_summary.groupby("eval_domain")["training_seed"]
            .nunique()
            .reset_index(name="num_training_seeds")
        )
        summary_aggregate = summary_aggregate.merge(seed_counts, on="eval_domain", how="left")
        summary_aggregate_path = output_root / "shock_recovery_summary_aggregate.csv"
        summary_aggregate.to_csv(summary_aggregate_path, index=False)
        print(f"Saved combined summary: {combined_summary_path}")
        print(f"Saved summary aggregate: {summary_aggregate_path}")

    print(f"Saved combined curve: {combined_curve_path}")
    print(f"Saved curve aggregate: {curve_aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
