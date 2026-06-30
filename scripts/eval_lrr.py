"""Evaluate an LRR residual checkpoint without local search."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.lrr_policy import LRRPolicy, load_lrr_residual
from algorithms.local_repair import action_range_fraction
from scripts.train_lrr import append_csv, evaluate_lrr_policy, resolve_runtime_config
from utils.cleanrl_policy import load_cleanrl_agent
from utils.recovery_runner_helpers import build_eval_episodes, infer_seed, load_source, source_num_eval_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Local Residual Repair.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--lrr-checkpoint", type=Path, required=True)
    parser.add_argument("--clean-checkpoint", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="Evaluation episodes per domain. Use 0 to match the source PPO run.",
    )
    parser.add_argument("--reward-cost-key", type=str, default="scalar_cost")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run_dir = args.source_run_dir if args.source_run_dir.is_absolute() else PROJECT_ROOT / args.source_run_dir
    seed = int(args.seed if args.seed is not None else infer_seed(source_run_dir))
    level_config, base_args, env_attack, source_checkpoint = load_source(source_run_dir)
    clean_checkpoint_path = args.clean_checkpoint or source_checkpoint
    runtime = resolve_runtime_config(level_config, base_args, args.reward_cost_key)

    clean_policy, clean_checkpoint = load_cleanrl_agent(clean_checkpoint_path, device=args.device)
    obs_dim = int(clean_checkpoint["obs_dim"])
    action_dim = int(clean_checkpoint["action_dim"])
    action_low = np.zeros(action_dim, dtype=np.float32)
    action_high = np.ones(action_dim, dtype=np.float32)
    try:
        raw_checkpoint = torch.load(args.lrr_checkpoint, map_location=args.device, weights_only=False)
    except TypeError:
        raw_checkpoint = torch.load(args.lrr_checkpoint, map_location=args.device)
    delta_max = np.asarray(
        raw_checkpoint.get("delta_max", action_range_fraction(0.10, action_low, action_high, action_dim)),
        dtype=np.float32,
    )
    policy = LRRPolicy(
        clean_policy,
        obs_dim,
        action_dim,
        action_low,
        action_high,
        delta_max,
        use_repair_gate=bool(raw_checkpoint.get("use_repair_gate", False)),
    )
    checkpoint = load_lrr_residual(args.lrr_checkpoint, policy)

    eval_episodes = int(args.eval_episodes)
    if eval_episodes <= 0:
        eval_episodes = source_num_eval_episodes(source_run_dir)
    map_size, domains = build_eval_episodes(source_run_dir, level_config, base_args, seed, int(eval_episodes))
    rows = evaluate_lrr_policy(
        policy,
        clean_policy,
        domains,
        map_size,
        runtime,
        env_attack,
        seed,
        int(checkpoint.get("metadata", {}).get("iteration", 0)),
        args.lrr_checkpoint,
    )
    output_csv = args.output_csv or args.lrr_checkpoint.parent.parent / "lrr_eval.csv"
    append_csv(output_csv, rows)
    env_pi = [float(row["performance_index"]) for row in rows if row["attack_type"] == "environment"]
    print(f"Saved LRR eval to {output_csv}; environment PI={float(np.nanmean(env_pi)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
