#!/usr/bin/env python3
"""Summarize LDAC tuning pilot results against the source baseline runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "sac_modified" / "ldac_tuning_pilot_20260624"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def pi(clean_cost: float, recovery_cost: float) -> float:
    if not math.isfinite(clean_cost) or not math.isfinite(recovery_cost) or recovery_cost <= 0.0:
        return math.nan
    return 100.0 * clean_cost / recovery_cost


def summary_points(summary_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    rows = []
    for row in frame.itertuples(index=False):
        clean = float(row.clean_nominal_cost)
        rows.append(
            {
                "eval_domain": str(row.eval_domain),
                "final_pi": pi(clean, float(row.final_recovery_cost)),
                "best_pi": pi(clean, float(row.best_recovery_cost)),
                "best_recovery_step": int(row.best_recovery_step),
                "attack_degradation_pct": float(row.attack_degradation_pct),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = resolve(args.root)
    jobs_path = root / "tuning_jobs.csv"
    if not jobs_path.exists():
        raise FileNotFoundError(jobs_path)
    jobs = pd.read_csv(jobs_path)

    rows = []
    for job in jobs.itertuples(index=False):
        candidate_summary = Path(job.seed_dir) / "shock_recovery_summary.csv"
        baseline_summary = Path(job.source_seed_dir) / "shock_recovery_summary.csv"
        if not candidate_summary.exists():
            rows.append(
                {
                    "candidate": job.candidate,
                    "scenario": job.scenario,
                    "seed": int(job.seed),
                    "target_step": int(job.target_step),
                    "status": "missing",
                }
            )
            continue
        candidate = summary_points(candidate_summary)
        baseline = summary_points(baseline_summary).rename(
            columns={
                "final_pi": "baseline_final_pi",
                "best_pi": "baseline_best_pi",
                "best_recovery_step": "baseline_best_recovery_step",
            }
        )
        merged = candidate.merge(baseline, on="eval_domain", how="left")
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "candidate": job.candidate,
                    "scenario": job.scenario,
                    "seed": int(job.seed),
                    "target_step": int(job.target_step),
                    "eval_domain": row.eval_domain,
                    "status": "complete",
                    "final_pi": float(row.final_pi),
                    "best_pi": float(row.best_pi),
                    "best_recovery_step": int(row.best_recovery_step),
                    "baseline_final_pi": float(row.baseline_final_pi),
                    "baseline_best_pi": float(row.baseline_best_pi),
                    "baseline_best_recovery_step": int(row.baseline_best_recovery_step),
                    "delta_final_pi": float(row.final_pi) - float(row.baseline_final_pi),
                    "delta_best_pi": float(row.best_pi) - float(row.baseline_best_pi),
                    "attack_degradation_pct": float(row.attack_degradation_pct_x),
                }
            )

    detail = pd.DataFrame(rows)
    detail_path = root / "tuning_summary_detail.csv"
    detail.to_csv(detail_path, index=False)

    complete = detail[detail["status"] == "complete"].copy()
    if complete.empty:
        print(f"No complete tuning runs found. Detail: {detail_path}")
        return

    aggregate = (
        complete.groupby(["candidate", "scenario", "eval_domain"], dropna=False)
        .agg(
            num_points=("final_pi", "count"),
            final_pi_mean=("final_pi", "mean"),
            best_pi_mean=("best_pi", "mean"),
            baseline_final_pi_mean=("baseline_final_pi", "mean"),
            baseline_best_pi_mean=("baseline_best_pi", "mean"),
            delta_final_pi_mean=("delta_final_pi", "mean"),
            delta_best_pi_mean=("delta_best_pi", "mean"),
        )
        .reset_index()
        .sort_values(["candidate", "scenario", "eval_domain"])
    )
    aggregate_path = root / "tuning_summary_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    heldout = complete[complete["eval_domain"].astype(str).str.contains("heldout", case=False, na=False)]
    leaderboard = (
        heldout.groupby("candidate")
        .agg(
            complete_points=("final_pi", "count"),
            heldout_final_pi=("final_pi", "mean"),
            heldout_best_pi=("best_pi", "mean"),
            heldout_delta_final_pi=("delta_final_pi", "mean"),
            heldout_delta_best_pi=("delta_best_pi", "mean"),
        )
        .reset_index()
        .sort_values(["heldout_delta_best_pi", "heldout_delta_final_pi"], ascending=False)
    )
    leaderboard_path = root / "tuning_leaderboard_heldout.csv"
    leaderboard.to_csv(leaderboard_path, index=False)

    print(f"Saved detail: {detail_path}")
    print(f"Saved aggregate: {aggregate_path}")
    print(f"Saved heldout leaderboard: {leaderboard_path}")
    print(leaderboard.to_string(index=False))

    manifest_path = root / "tuning_summary_manifest.json"
    manifest = {
        "detail_csv": str(detail_path),
        "aggregate_csv": str(aggregate_path),
        "heldout_leaderboard_csv": str(leaderboard_path),
        "num_complete_rows": int(len(complete)),
        "num_jobs": int(len(jobs)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
