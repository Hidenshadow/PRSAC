#!/usr/bin/env python
"""Evaluate a CleanRL checkpoint on the same task sampler as a source run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_attack_recovery_finetune import evaluate_checkpoint
from run_shock_recovery_experiment import PROJECT_ROOT, plot_outputs, summarize_recovery
from utils.recovery_runner_helpers import (
    add_protocol_columns,
    build_eval_episodes,
    disabled_attack,
    load_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--role", type=str, default="reevaluated_checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run_dir = args.source_run_dir if args.source_run_dir.is_absolute() else PROJECT_ROOT / args.source_run_dir
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else PROJECT_ROOT / args.checkpoint
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    level_config, base_args, env_attack, nominal_checkpoint = load_source(source_run_dir)
    map_size, episodes_by_domain = build_eval_episodes(
        source_run_dir,
        level_config,
        base_args,
        int(args.seed),
        int(args.eval_episodes),
    )
    rows = []
    rows.extend(
        add_protocol_columns(
            evaluate_checkpoint(
                nominal_checkpoint,
                0,
                episodes_by_domain,
                map_size,
                env_attack,
                disabled_attack(),
                int(args.seed),
            ),
            phase="shock",
            recovery_step=0,
            checkpoint_role="nominal",
        )
    )
    rows.extend(
        row
        for row in add_protocol_columns(
            evaluate_checkpoint(
                checkpoint_path,
                1,
                episodes_by_domain,
                map_size,
                env_attack,
                disabled_attack(),
                int(args.seed),
            ),
            phase="recovery",
            recovery_step=1,
            checkpoint_role=str(args.role),
        )
        if row["attack_type"] == "environment"
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "shock_recovery_curve.csv", index=False)
    summarize_recovery(frame, output_dir)
    plot_outputs(frame, output_dir)
    print(f"Saved same-task checkpoint evaluation to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
