"""Evaluate a BVR verifier checkpoint without true-cost candidate labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.bvr import action_range_fraction, load_bvr_checkpoint
from scripts.train_bvr import evaluate_bvr_policy
from scripts.train_lrr import append_csv, resolve_runtime_config
from utils.cleanrl_policy import load_cleanrl_agent
from utils.recovery_runner_helpers import build_eval_episodes, infer_seed, load_source, source_num_eval_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Belief-Route Verifier Recovery.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--bvr-checkpoint", type=Path, required=True)
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
    parser.add_argument("--selection-margin", type=float, default=None)
    parser.add_argument("--belief-safety-penalty", type=float, default=None)
    parser.add_argument("--belief-cost-margin", type=float, default=None)
    parser.add_argument("--belief-constraint-margin", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_run_dir = args.source_run_dir if args.source_run_dir.is_absolute() else PROJECT_ROOT / args.source_run_dir
    seed = int(args.seed if args.seed is not None else infer_seed(source_run_dir))
    level_config, base_args, env_attack, source_checkpoint = load_source(source_run_dir)
    verifier, metadata = load_bvr_checkpoint(args.bvr_checkpoint, device=args.device)
    clean_checkpoint_path = args.clean_checkpoint or metadata.get("clean_policy_checkpoint") or source_checkpoint
    runtime = dict(metadata.get("runtime") or resolve_runtime_config(level_config, base_args, args.reward_cost_key))

    clean_policy, clean_checkpoint = load_cleanrl_agent(clean_checkpoint_path, device=args.device)
    action_dim = int(clean_checkpoint["action_dim"])
    action_low = np.asarray(metadata.get("action_low", np.zeros(action_dim, dtype=np.float32)), dtype=np.float32)
    action_high = np.asarray(metadata.get("action_high", np.ones(action_dim, dtype=np.float32)), dtype=np.float32)
    if action_low.size != action_dim:
        action_low = np.zeros(action_dim, dtype=np.float32)
    if action_high.size != action_dim:
        action_high = np.ones(action_dim, dtype=np.float32)
    search_radius = np.asarray(
        metadata.get("search_radius", action_range_fraction(0.08, action_low, action_high, action_dim)),
        dtype=np.float32,
    )
    if search_radius.size != action_dim:
        search_radius = action_range_fraction(0.08, action_low, action_high, action_dim)
    pairwise_candidate_mode = str(metadata.get("pairwise_candidate_mode", "adjacent"))
    selection_margin = float(
        args.selection_margin if args.selection_margin is not None else metadata.get("selection_margin", 0.0)
    )
    belief_safety_penalty = float(
        args.belief_safety_penalty
        if args.belief_safety_penalty is not None
        else metadata.get("belief_safety_penalty", 1.0)
    )
    belief_cost_margin = float(
        args.belief_cost_margin if args.belief_cost_margin is not None else metadata.get("belief_cost_margin", 0.02)
    )
    belief_constraint_margin = float(
        args.belief_constraint_margin
        if args.belief_constraint_margin is not None
        else metadata.get("belief_constraint_margin", 0.02)
    )

    eval_episodes = int(args.eval_episodes)
    if eval_episodes <= 0:
        eval_episodes = source_num_eval_episodes(source_run_dir)
    map_size, domains = build_eval_episodes(source_run_dir, level_config, base_args, seed, int(eval_episodes))
    rows = evaluate_bvr_policy(
        verifier,
        clean_policy,
        domains,
        map_size,
        runtime,
        env_attack,
        seed,
        int(metadata.get("iteration", 0)),
        action_low,
        action_high,
        search_radius,
        pairwise_candidate_mode,
        selection_margin,
        belief_safety_penalty,
        belief_cost_margin,
        belief_constraint_margin,
        args.bvr_checkpoint,
    )
    output_csv = args.output_csv or args.bvr_checkpoint.parent.parent / "bvr_eval.csv"
    append_csv(output_csv, rows)
    env_pi = [float(row["performance_index"]) for row in rows if row["attack_type"] == "environment"]
    print(f"Saved BVR eval to {output_csv}; environment PI={float(np.nanmean(env_pi)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
