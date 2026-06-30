"""Train Belief-Route Verifier Recovery from an existing clean PPO source run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.bvr import (
    BVRCandidateBatch,
    BVRTrainer,
    BVRTrainerConfig,
    RouteVerifier,
    action_range_fraction,
    extract_route_verifier_features,
    generate_bvr_candidates,
    plan_belief_route_action,
    route_verifier_feature_names,
    save_bvr_checkpoint,
    select_bvr_action_from_features,
)
from algorithms.local_repair import EpisodePlannerContext, evaluate_planner_action, performance_index
from envs.attack_wrappers import apply_environment_attack_to_episode, attack_enabled
from run_attack_recovery_finetune import config_value
from scripts.train_lrr import (
    action_result,
    append_csv,
    append_jsonl,
    clean_nominal_cost,
    clean_policy_action,
    flatten_domains,
    resolve_runtime_config,
)
from utils.cleanrl_policy import load_cleanrl_agent
from utils.metrics import compute_observation
from utils.recovery_runner_helpers import (
    build_eval_episodes,
    infer_seed,
    json_safe,
    load_source,
    save_json,
    source_num_eval_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Belief-Route Verifier Recovery.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--clean-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--rollout-episodes-per-iteration", type=int, default=64)
    parser.add_argument("--max-candidate-sets-per-iteration", type=int, default=128)
    parser.add_argument("--verifier-batch-size", type=int, default=32)
    parser.add_argument("--verifier-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-temperature", type=float, default=1.5)
    parser.add_argument("--set-weight-beta", type=float, default=2.0)
    parser.add_argument("--set-weight-max", type=float, default=5.0)
    parser.add_argument("--advantage-loss-weight", type=float, default=1.0)
    parser.add_argument("--advantage-beta", type=float, default=2.0)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--benefit-loss-weight", type=float, default=2.0)
    parser.add_argument("--benefit-epsilon", type=float, default=0.25)
    parser.add_argument("--benefit-positive-weight", type=float, default=3.0)
    parser.add_argument("--max-replay-sets", type=int, default=2048)
    parser.add_argument("--search-radius-fraction", type=float, default=0.08)
    parser.add_argument(
        "--selection-margin",
        type=float,
        default=5e-4,
        help="Use the clean PPO anchor unless predicted candidate advantage exceeds this verifier-score margin.",
    )
    parser.add_argument("--belief-safety-penalty", type=float, default=1.0)
    parser.add_argument("--belief-cost-margin", type=float, default=0.02)
    parser.add_argument("--belief-constraint-margin", type=float, default=0.02)
    parser.add_argument(
        "--pairwise-candidate-mode",
        choices=("none", "adjacent", "all"),
        default="adjacent",
        help="Small coupled planner-weight candidates for verifier reranking.",
    )
    parser.add_argument("--improvement-epsilon", type=float, default=0.25)
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="Evaluation episodes per domain. Use 0 to match the source PPO run.",
    )
    parser.add_argument("--reward-cost-key", type=str, default="scalar_cost")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--activation", choices=("tanh", "relu"), default="tanh")
    parser.add_argument("--final-init-std", type=float, default=0.0)
    parser.add_argument("--failure-cost", type=float, default=1e6)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def config_text(config: dict[str, Any], key: str, default: Any) -> Any:
    return config_value(config, key, config.get(key.replace("-", "_"), default))


def attack_value(env_attack: dict[str, Any], base_args: dict[str, Any], key: str, default: Any) -> Any:
    return env_attack.get(key, config_text(base_args, key.replace("_", "-"), default))


def make_context(
    episode: Any,
    runtime: dict[str, Any],
    env_attack: dict[str, Any],
    base_args: dict[str, Any],
    *,
    failure_cost: float,
) -> EpisodePlannerContext:
    context = EpisodePlannerContext(
        episode,
        action_mode=runtime["action_mode"],
        action_gain=float(runtime["action_gain"]),
        max_uncertainty_lambda=float(runtime["max_uncertainty_lambda"]),
        allow_diagonal=True,
        attack_budget_fraction=float(attack_value(env_attack, base_args, "attack_budget_fraction", 0.18)),
        attack_strength=float(attack_value(env_attack, base_args, "attack_strength", 1.0)),
        attacker_temperature=float(attack_value(env_attack, base_args, "attacker_temperature", 0.5)),
        attacker_response=str(attack_value(env_attack, base_args, "attacker_response", "zscore_topk")),
        attacker_top_fraction=float(attack_value(env_attack, base_args, "attacker_top_fraction", 0.15)),
        attacker_sharpness=float(attack_value(env_attack, base_args, "attacker_sharpness", 3.0)),
        reward_cost_key=str(runtime["reward_cost_key"]),
    )
    context.failure_cost = float(failure_cost)
    return context


def apply_attack_if_needed(episode: Any, env_attack: dict[str, Any], seed: int) -> Any:
    if attack_enabled(env_attack):
        return apply_environment_attack_to_episode(episode, env_attack, np.random.default_rng(seed))
    return episode


def default_output_dir(source_run_dir: Path, seed: int) -> Path:
    experiment_name = source_run_dir.parent.name
    return PROJECT_ROOT / "runs" / "bvr" / f"{experiment_name}_bvr" / f"seed{seed}"


def build_candidate_batch(
    clean_policy: torch.nn.Module,
    episode: Any,
    attacked_episode: Any,
    map_size: int,
    runtime: dict[str, Any],
    env_attack: dict[str, Any],
    base_args: dict[str, Any],
    device: str | torch.device,
    action_low: np.ndarray,
    action_high: np.ndarray,
    search_radius: np.ndarray,
    pairwise_candidate_mode: str,
    failure_cost: float,
) -> tuple[np.ndarray, np.ndarray, BVRCandidateBatch]:
    obs = compute_observation(
        attacked_episode,
        map_size,
        observation_mode=str(runtime["observation_mode"]),
        max_uncertainty_lambda=float(runtime["max_uncertainty_lambda"]),
    )
    clean_action = clean_policy_action(clean_policy, obs, device)
    candidates = generate_bvr_candidates(
        clean_action,
        action_low,
        action_high,
        search_radius,
        pairwise_candidate_mode=pairwise_candidate_mode,
    )
    nominal = clean_nominal_cost(clean_policy, episode, map_size, runtime, device)
    context = make_context(attacked_episode, runtime, env_attack, base_args, failure_cost=float(failure_cost))
    features: list[np.ndarray] = []
    scores: list[float] = []
    raw_results: list[dict[str, Any]] = []
    for candidate in candidates:
        score, result = evaluate_planner_action(
            context,
            candidate.action,
            nominal,
            reward_cost_key=str(runtime["reward_cost_key"]),
            failure_cost=float(failure_cost),
        )
        result = dict(result)
        result["mission_priority"] = np.asarray(attacked_episode.mission_priority, dtype=np.float32)
        result["candidate_type"] = candidate.candidate_type
        features.append(
            extract_route_verifier_features(
                obs,
                clean_action,
                candidate.action,
                result,
                map_size=map_size,
                max_uncertainty_lambda=float(runtime["max_uncertainty_lambda"]),
            )
        )
        scores.append(float(score))
        raw_results.append(result)
    return (
        obs.astype(np.float32),
        clean_action.astype(np.float32),
        BVRCandidateBatch(
            candidates=candidates,
            features=np.stack(features, axis=0).astype(np.float32),
            true_scores=np.asarray(scores, dtype=np.float32),
            raw_results=raw_results,
        ),
    )


def build_eval_candidate_features(
    episode: Any,
    observation: np.ndarray,
    clean_action: np.ndarray,
    candidates: list[Any],
    map_size: int,
    runtime: dict[str, Any],
    env_attack: dict[str, Any] | None,
) -> np.ndarray:
    features: list[np.ndarray] = []
    for candidate in candidates:
        result = plan_belief_route_action(episode, candidate.action, runtime, env_attack)
        features.append(
            extract_route_verifier_features(
                observation,
                clean_action,
                candidate.action,
                result,
                map_size=map_size,
                max_uncertainty_lambda=float(runtime["max_uncertainty_lambda"]),
            )
        )
    return np.stack(features, axis=0).astype(np.float32)


def evaluate_bvr_policy(
    verifier: RouteVerifier,
    clean_policy: torch.nn.Module,
    domains: dict[str, tuple[int, list[Any]]],
    map_size: int,
    runtime: dict[str, Any],
    env_attack: dict[str, Any],
    seed: int,
    iteration: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    search_radius: np.ndarray,
    pairwise_candidate_mode: str,
    selection_margin: float = 0.0,
    belief_safety_penalty: float = 1.0,
    belief_cost_margin: float = 0.02,
    belief_constraint_margin: float = 0.02,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    device = next(verifier.parameters()).device
    for eval_domain, (map_seed, episodes) in domains.items():
        for attack_type in ("none", "environment"):
            nominal_costs: list[float] = []
            attacked_costs: list[float] = []
            pis: list[float] = []
            successes: list[float] = []
            lambda_values: list[float] = []
            selected_indices: list[float] = []
            score_spreads: list[float] = []
            clean_selected_count = 0
            for episode_index, episode in enumerate(episodes):
                eval_episode = episode
                active_attack = None
                if attack_type == "environment":
                    eval_episode = apply_attack_if_needed(
                        episode,
                        env_attack,
                        seed + 700_000 + iteration * 10_000 + episode_index,
                    )
                    active_attack = env_attack
                nominal = clean_nominal_cost(clean_policy, episode, map_size, runtime, device)
                obs = compute_observation(
                    eval_episode,
                    map_size,
                    observation_mode=str(runtime["observation_mode"]),
                    max_uncertainty_lambda=float(runtime["max_uncertainty_lambda"]),
                )
                clean_action = clean_policy_action(clean_policy, obs, device)
                candidates = generate_bvr_candidates(
                    clean_action,
                    action_low,
                    action_high,
                    search_radius,
                    pairwise_candidate_mode=pairwise_candidate_mode,
                )
                features = build_eval_candidate_features(
                    eval_episode,
                    obs,
                    clean_action,
                    candidates,
                    map_size,
                    runtime,
                    active_attack,
                )
                selection = select_bvr_action_from_features(
                    verifier,
                    candidates,
                    features,
                    clean_action,
                    selection_margin=float(selection_margin),
                    belief_safety_penalty=float(belief_safety_penalty),
                    belief_cost_margin=float(belief_cost_margin),
                    belief_constraint_margin=float(belief_constraint_margin),
                )
                result, attacked_cost, lambda_uncertainty = action_result(
                    eval_episode,
                    selection.action,
                    runtime,
                    active_attack,
                )
                nominal_costs.append(float(nominal))
                attacked_costs.append(float(attacked_cost))
                pis.append(performance_index(nominal, attacked_cost))
                successes.append(1.0 if bool(result.get("success", False)) else 0.0)
                lambda_values.append(float(lambda_uncertainty))
                selected_indices.append(float(selection.selected_index))
                score_spreads.append(float(np.max(selection.verifier_scores) - np.min(selection.verifier_scores)))
                if selection.selected_index == 0:
                    clean_selected_count += 1
            rows.append(
                {
                    "global_step": int(iteration),
                    "iteration": int(iteration),
                    "eval_domain": eval_domain,
                    "map_pool_seed": int(map_seed),
                    "attack_type": attack_type,
                    "mean_nominal_scalar_cost": float(np.nanmean(nominal_costs)),
                    "mean_attacked_scalar_cost": float(np.nanmean(attacked_costs)),
                    "performance_index": float(np.nanmean(pis)),
                    "success_rate": float(np.nanmean(successes)),
                    "mean_lambda_uncertainty": float(np.nanmean(lambda_values)),
                    "mean_selected_candidate_index": float(np.nanmean(selected_indices)),
                    "clean_anchor_selection_rate": float(clean_selected_count / max(len(episodes), 1)),
                    "selection_margin": float(selection_margin),
                    "belief_safety_penalty": float(belief_safety_penalty),
                    "belief_cost_margin": float(belief_cost_margin),
                    "belief_constraint_margin": float(belief_constraint_margin),
                    "mean_verifier_score_spread": float(np.nanmean(score_spreads)),
                    "candidate_count": int(2 * len(action_low) + 1)
                    if str(pairwise_candidate_mode) == "none"
                    else int(len(generate_bvr_candidates(np.full_like(action_low, 0.5), action_low, action_high, search_radius, pairwise_candidate_mode))),
                    "checkpoint_path": str(checkpoint_path or ""),
                }
            )
    return rows


def candidate_debug_row(
    iteration: int,
    domain: str,
    episode_index: int,
    clean_action: np.ndarray,
    batch: BVRCandidateBatch,
    verifier: RouteVerifier,
    selection_margin: float = 0.0,
    belief_safety_penalty: float = 1.0,
    belief_cost_margin: float = 0.02,
    belief_constraint_margin: float = 0.02,
) -> dict[str, Any]:
    finite_scores = np.where(np.isfinite(batch.true_scores), batch.true_scores, -1e6)
    best_index = int(np.argmax(finite_scores))
    ungated_selection = select_bvr_action_from_features(
        verifier,
        batch.candidates,
        batch.features,
        clean_action,
        selection_margin=float(selection_margin),
        belief_safety_penalty=0.0,
    )
    selection = select_bvr_action_from_features(
        verifier,
        batch.candidates,
        batch.features,
        clean_action,
        selection_margin=float(selection_margin),
        belief_safety_penalty=float(belief_safety_penalty),
        belief_cost_margin=float(belief_cost_margin),
        belief_constraint_margin=float(belief_constraint_margin),
    )
    clean_score = float(finite_scores[0]) if finite_scores.size else float("nan")
    best_score = float(finite_scores[best_index]) if finite_scores.size else float("nan")
    selected_score = float(finite_scores[selection.selected_index]) if finite_scores.size else float("nan")
    return {
        "iteration": int(iteration),
        "state_id": f"{domain}:{episode_index}",
        "episode_id": int(episode_index),
        "eval_domain": domain,
        "candidate_count": int(len(batch.candidates)),
        "clean_score": clean_score,
        "best_score": best_score,
        "best_improvement_over_clean": float(best_score - clean_score),
        "best_candidate_type": batch.candidates[best_index].candidate_type,
        "selected_score": selected_score,
        "selected_regret": float(best_score - selected_score),
        "selected_candidate_type": selection.selected_candidate_type,
        "selected_index": int(selection.selected_index),
        "ungated_selected_candidate_type": ungated_selection.selected_candidate_type,
        "ungated_selected_index": int(ungated_selection.selected_index),
        "selection_margin": float(selection_margin),
        "belief_safety_penalty": float(belief_safety_penalty),
        "belief_cost_margin": float(belief_cost_margin),
        "belief_constraint_margin": float(belief_constraint_margin),
        "margin_fallback_to_clean": bool(ungated_selection.selected_index != 0 and selection.selected_index == 0),
        "belief_safety_changed_selection": bool(ungated_selection.selected_index != selection.selected_index),
        "clean_action": clean_action.tolist(),
        "best_action": np.asarray(batch.candidates[best_index].action, dtype=np.float32).tolist(),
        "selected_action": np.asarray(selection.action, dtype=np.float32).tolist(),
        "verifier_scores": selection.verifier_scores.tolist(),
        "true_scores": batch.true_scores.tolist(),
    }


def main() -> int:
    args = parse_args()
    source_run_dir = args.source_run_dir if args.source_run_dir.is_absolute() else PROJECT_ROOT / args.source_run_dir
    seed = int(args.seed if args.seed is not None else infer_seed(source_run_dir))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    level_config, base_args, env_attack, source_checkpoint = load_source(source_run_dir)
    clean_checkpoint_path = args.clean_checkpoint or source_checkpoint
    output_dir = args.output_dir or default_output_dir(source_run_dir, seed)
    runtime = resolve_runtime_config(level_config, base_args, args.reward_cost_key)

    if args.dry_run:
        print(f"Would train BVR from {clean_checkpoint_path} -> {output_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    clean_policy, clean_checkpoint = load_cleanrl_agent(clean_checkpoint_path, device=args.device)
    obs_dim = int(clean_checkpoint["obs_dim"])
    action_dim = int(clean_checkpoint["action_dim"])
    action_low = np.zeros(action_dim, dtype=np.float32)
    action_high = np.ones(action_dim, dtype=np.float32)
    search_radius = action_range_fraction(args.search_radius_fraction, action_low, action_high, action_dim)
    feature_names = route_verifier_feature_names(obs_dim, action_dim)
    verifier = RouteVerifier(
        len(feature_names),
        hidden_sizes=(int(args.hidden_size), int(args.hidden_size)),
        activation=str(args.activation),
        final_init_std=float(args.final_init_std),
    ).to(args.device)
    trainer = BVRTrainer(
        verifier,
        BVRTrainerConfig(
            batch_size=int(args.verifier_batch_size),
            epochs=int(args.verifier_epochs),
            learning_rate=float(args.learning_rate),
            target_temperature=float(args.target_temperature),
            set_weight_beta=float(args.set_weight_beta),
            set_weight_max=float(args.set_weight_max),
            max_replay_sets=int(args.max_replay_sets),
            advantage_loss_weight=float(args.advantage_loss_weight),
            advantage_beta=float(args.advantage_beta),
            advantage_clip=float(args.advantage_clip),
            benefit_loss_weight=float(args.benefit_loss_weight),
            benefit_epsilon=float(args.benefit_epsilon),
            benefit_positive_weight=float(args.benefit_positive_weight),
        ),
    )
    eval_episodes = int(args.eval_episodes)
    if eval_episodes <= 0:
        eval_episodes = source_num_eval_episodes(source_run_dir)
    map_size, domains = build_eval_episodes(source_run_dir, level_config, base_args, seed, int(eval_episodes))
    flat_episodes = flatten_domains(domains)
    if not flat_episodes:
        raise RuntimeError("no episodes available for BVR training")

    save_json(
        output_dir / "bvr_run_config.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": "bvr",
            "source_run_dir": str(source_run_dir),
            "clean_policy_checkpoint": str(clean_checkpoint_path),
            "level_config": level_config,
            "base_config_args": base_args,
            "environment_attack": env_attack,
            "runtime": runtime,
            "args": vars(args),
            "resolved_eval_episodes": int(eval_episodes),
            "search_radius": search_radius.tolist(),
            "feature_names": feature_names,
            "pairwise_candidate_mode": str(args.pairwise_candidate_mode),
            "selection_margin": float(args.selection_margin),
            "belief_safety_penalty": float(args.belief_safety_penalty),
            "belief_cost_margin": float(args.belief_cost_margin),
            "belief_constraint_margin": float(args.belief_constraint_margin),
        },
    )

    replay_features: list[np.ndarray] = []
    replay_scores: list[np.ndarray] = []
    best_recovery_index = float("-inf")
    best_iteration = 0
    best_checkpoint_path = checkpoints_dir / "best_bvr_verifier.pt"
    initial_checkpoint = checkpoints_dir / "bvr_verifier_iter_00000.pt"
    checkpoint_metadata = {
        "clean_policy_checkpoint": str(clean_checkpoint_path),
        "iteration": 0,
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "action_low": action_low.tolist(),
        "action_high": action_high.tolist(),
        "search_radius": search_radius.tolist(),
        "feature_names": feature_names,
        "pairwise_candidate_mode": str(args.pairwise_candidate_mode),
        "selection_margin": float(args.selection_margin),
        "belief_safety_penalty": float(args.belief_safety_penalty),
        "belief_cost_margin": float(args.belief_cost_margin),
        "belief_constraint_margin": float(args.belief_constraint_margin),
        "benefit_loss_weight": float(args.benefit_loss_weight),
        "benefit_epsilon": float(args.benefit_epsilon),
        "benefit_positive_weight": float(args.benefit_positive_weight),
        "runtime": runtime,
    }
    save_bvr_checkpoint(initial_checkpoint, verifier, checkpoint_metadata)
    eval_rows = evaluate_bvr_policy(
        verifier,
        clean_policy,
        domains,
        map_size,
        runtime,
        env_attack,
        seed,
        0,
        action_low,
        action_high,
        search_radius,
        str(args.pairwise_candidate_mode),
        float(args.selection_margin),
        float(args.belief_safety_penalty),
        float(args.belief_cost_margin),
        float(args.belief_constraint_margin),
        initial_checkpoint,
    )
    append_csv(output_dir / "bvr_recovery_curve.csv", eval_rows)
    env_pi = [float(row["performance_index"]) for row in eval_rows if row["attack_type"] == "environment"]
    if env_pi:
        best_recovery_index = float(np.nanmean(env_pi))
        shutil.copy2(initial_checkpoint, best_checkpoint_path)

    for iteration in range(1, int(args.num_iterations) + 1):
        rng = random.Random(seed + 2000 * iteration)
        sample_count = min(
            int(args.rollout_episodes_per_iteration),
            int(args.max_candidate_sets_per_iteration),
            len(flat_episodes),
        )
        selected = rng.sample(flat_episodes, sample_count) if sample_count < len(flat_episodes) else list(flat_episodes)

        debug_rows: list[dict[str, Any]] = []
        best_improvements: list[float] = []
        selected_regrets: list[float] = []
        positive_sets = 0
        candidate_count = 0
        for local_index, (domain, episode_index, episode) in enumerate(selected):
            attacked_episode = apply_attack_if_needed(
                episode,
                env_attack,
                seed + 400_000 + iteration * 10_000 + local_index,
            )
            _obs, clean_action, batch = build_candidate_batch(
                clean_policy,
                episode,
                attacked_episode,
                map_size,
                runtime,
                env_attack,
                base_args,
                args.device,
                action_low,
                action_high,
                search_radius,
                str(args.pairwise_candidate_mode),
                float(args.failure_cost),
            )
            replay_features.append(batch.features)
            replay_scores.append(batch.true_scores)
            candidate_count = len(batch.candidates)
            row = candidate_debug_row(
                iteration,
                domain,
                episode_index,
                clean_action,
                batch,
                verifier,
                float(args.selection_margin),
                float(args.belief_safety_penalty),
                float(args.belief_cost_margin),
                float(args.belief_constraint_margin),
            )
            debug_rows.append(row)
            best_improvement = float(row["best_improvement_over_clean"])
            best_improvements.append(best_improvement)
            selected_regrets.append(float(row["selected_regret"]))
            if best_improvement > float(args.improvement_epsilon):
                positive_sets += 1

        if len(replay_features) > int(args.max_replay_sets):
            replay_features = replay_features[-int(args.max_replay_sets) :]
            replay_scores = replay_scores[-int(args.max_replay_sets) :]
        train_metrics = trainer.update(replay_features, replay_scores)
        checkpoint_path = checkpoints_dir / f"bvr_verifier_iter_{iteration:05d}.pt"
        checkpoint_metadata = {
            **checkpoint_metadata,
            "iteration": int(iteration),
            "best_recovery_index": float(best_recovery_index),
        }
        save_bvr_checkpoint(checkpoint_path, verifier, checkpoint_metadata)
        eval_rows = evaluate_bvr_policy(
            verifier,
            clean_policy,
            domains,
            map_size,
            runtime,
            env_attack,
            seed,
            iteration,
            action_low,
            action_high,
            search_radius,
            str(args.pairwise_candidate_mode),
            float(args.selection_margin),
            float(args.belief_safety_penalty),
            float(args.belief_cost_margin),
            float(args.belief_constraint_margin),
            checkpoint_path,
        )
        append_csv(output_dir / "bvr_recovery_curve.csv", eval_rows)
        append_jsonl(output_dir / "bvr_candidate_sets.jsonl", debug_rows)

        env_pi = [float(row["performance_index"]) for row in eval_rows if row["attack_type"] == "environment"]
        current_recovery_index = float(np.nanmean(env_pi)) if env_pi else float("nan")
        if np.isfinite(current_recovery_index) and current_recovery_index > best_recovery_index:
            best_recovery_index = current_recovery_index
            best_iteration = int(iteration)
            shutil.copy2(checkpoint_path, best_checkpoint_path)
        metrics_row = {
            "iteration": int(iteration),
            "global_step": int(iteration),
            "bvr/loss": train_metrics["bvr/loss"],
            "bvr/advantage_loss": train_metrics["bvr/advantage_loss"],
            "bvr/benefit_loss": train_metrics["bvr/benefit_loss"],
            "bvr/top1_accuracy": train_metrics["bvr/top1_accuracy"],
            "bvr/positive_benefit_label_rate": train_metrics["bvr/positive_benefit_label_rate"],
            "bvr/predicted_benefit_rate": train_metrics["bvr/predicted_benefit_rate"],
            "bvr/mean_selected_regret": train_metrics["bvr/mean_selected_regret"],
            "bvr/mean_score_spread": train_metrics["bvr/mean_score_spread"],
            "bvr/num_candidate_sets": int(train_metrics["bvr/num_candidate_sets"]),
            "bvr/num_states_evaluated": int(len(selected)),
            "bvr/positive_candidate_sets": int(positive_sets),
            "bvr/positive_candidate_rate": float(positive_sets / max(len(selected), 1)),
            "bvr/mean_best_improvement_over_clean": float(np.nanmean(best_improvements)) if best_improvements else 0.0,
            "bvr/max_best_improvement_over_clean": float(np.nanmax(best_improvements)) if best_improvements else 0.0,
            "bvr/mean_preupdate_selected_regret": float(np.nanmean(selected_regrets)) if selected_regrets else 0.0,
            "bvr/candidate_count": int(candidate_count),
            "bvr/search_radius": float(np.mean(search_radius)),
            "bvr/selection_margin": float(args.selection_margin),
            "bvr/belief_safety_penalty": float(args.belief_safety_penalty),
            "bvr/belief_cost_margin": float(args.belief_cost_margin),
            "bvr/belief_constraint_margin": float(args.belief_constraint_margin),
            "bvr/advantage_loss_weight": float(args.advantage_loss_weight),
            "bvr/advantage_beta": float(args.advantage_beta),
            "bvr/benefit_loss_weight": float(args.benefit_loss_weight),
            "bvr/benefit_epsilon": float(args.benefit_epsilon),
            "bvr/benefit_positive_weight": float(args.benefit_positive_weight),
            "bvr/pairwise_candidate_mode": str(args.pairwise_candidate_mode),
            "performance_index": current_recovery_index,
            "final_recovery_index": current_recovery_index,
            "best_recovery_index": best_recovery_index,
            "best_iteration": int(best_iteration),
            "best_checkpoint_path": str(best_checkpoint_path),
            "checkpoint_path": str(checkpoint_path),
        }
        append_csv(output_dir / "bvr_metrics.csv", [metrics_row])
        print(
            " ".join(
                [
                    f"iter={iteration}",
                    f"pi={current_recovery_index:.3f}",
                    f"positive={positive_sets}/{len(selected)}",
                    f"top1={train_metrics['bvr/top1_accuracy']:.3f}",
                    f"benefit={train_metrics['bvr/positive_benefit_label_rate']:.3f}",
                    f"regret={train_metrics['bvr/mean_selected_regret']:.3f}",
                    f"loss={train_metrics['bvr/loss']:.6f}",
                ]
            ),
            flush=True,
        )

    print(f"Saved BVR outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
