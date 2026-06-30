"""Train Local Residual Repair from an existing clean PPO source run.

This script does not fine-tune PPO. It freezes the nominal checkpoint, queries
local planner repairs only during label generation, and trains the bounded LRR
residual by weighted regression.
"""

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

from algorithms.lrr_policy import LRRPolicy, save_lrr_checkpoint
from algorithms.lrr_trainer import LRRTrainer, LRRTrainerConfig
from algorithms.local_repair import (
    EpisodePlannerContext,
    LocalRepairConfig,
    RepairLabel,
    action_range_fraction,
    evaluate_local_repair,
    evaluate_planner_action,
    performance_index,
)
from envs.attack_wrappers import apply_environment_attack_to_episode, attack_enabled
from run_attack_recovery_finetune import config_value
from utils.cleanrl_policy import load_cleanrl_agent
from utils.metrics import (
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    compute_observation,
    plan_with_weights,
)
from utils.recovery_runner_helpers import (
    build_eval_episodes,
    infer_seed,
    json_safe,
    load_source,
    save_json,
    source_num_eval_episodes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Local Residual Repair.")
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--clean-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--rollout-episodes-per-iteration", type=int, default=20)
    parser.add_argument("--max-repair-states-per-iteration", type=int, default=256)
    parser.add_argument("--repair-batch-size", type=int, default=64)
    parser.add_argument("--repair-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--delta-max-fraction", type=float, default=0.10)
    parser.add_argument("--search-radius-fraction", type=float, default=0.05)
    parser.add_argument(
        "--pairwise-candidate-mode",
        choices=("none", "adjacent", "all"),
        default="none",
        help="Optional two-dimensional local repair candidates; adjacent is a compact default for coupled planner weights.",
    )
    parser.add_argument("--improvement-epsilon", type=float, default=0.25)
    parser.add_argument(
        "--repair-target-mode",
        choices=("best", "soft", "surface", "response_surface"),
        default="best",
        help=(
            "Repair label target: best imitates the single best candidate; "
            "soft averages improved candidates; surface fits a local response surface."
        ),
    )
    parser.add_argument(
        "--soft-target-temperature",
        type=float,
        default=2.0,
        help="Performance-index temperature for soft repair targets.",
    )
    parser.add_argument(
        "--repair-target-blend",
        type=float,
        default=1.0,
        help="Blend from current residual toward the locally best residual target; 1.0 keeps the original LRR label.",
    )
    parser.add_argument(
        "--surface-ridge",
        type=float,
        default=1e-3,
        help="Ridge coefficient for response-surface repair targets.",
    )
    parser.add_argument(
        "--surface-max-step-fraction",
        type=float,
        default=1.0,
        help="Maximum response-surface step as a fraction of local search radius.",
    )
    parser.add_argument(
        "--surface-disable-diagonal",
        action="store_true",
        help="Disable full-dimensional response-surface proposal; keeps axis/pair/existing proposals only.",
    )
    parser.add_argument(
        "--target-residual-norm-clip",
        type=float,
        default=0.0,
        help="Optional L2 clip for supervised target residuals in normalized action space; <=0 disables it.",
    )
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--w-max", type=float, default=5.0)
    parser.add_argument("--lambda-zero", type=float, default=1e-3)
    parser.add_argument(
        "--use-repair-gate",
        action="store_true",
        help="Enable Selective LRR: learn when to apply the residual repair.",
    )
    parser.add_argument(
        "--lambda-gate",
        type=float,
        default=0.1,
        help="Weight for the repair-gate binary loss.",
    )
    parser.add_argument(
        "--gate-initial-logit",
        type=float,
        default=2.0,
        help="Initial repair-gate logit used when --use-repair-gate is enabled.",
    )
    parser.add_argument(
        "--positive-gate-weight",
        type=float,
        default=1.0,
        help="Gate loss weight for states with positive local repair labels.",
    )
    parser.add_argument(
        "--negative-gate-weight",
        type=float,
        default=1.0,
        help="Gate loss weight for evaluated states without useful local repairs.",
    )
    parser.add_argument(
        "--negative-gate-max-improvement",
        type=float,
        default=0.0,
        help="Only states with local best improvement <= this value are used as gate-negative labels.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="Evaluation episodes per domain. Use 0 to match the source PPO run.",
    )
    parser.add_argument("--reward-cost-key", type=str, default="scalar_cost")
    parser.add_argument("--residual-only", action="store_true", help="Ablation: train only residual shrinkage.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def config_text(config: dict[str, Any], key: str, default: Any) -> Any:
    return config_value(config, key, config.get(key.replace("-", "_"), default))


def resolve_runtime_config(
    level_config: dict[str, Any],
    base_args: dict[str, Any],
    reward_cost_key_override: str,
) -> dict[str, Any]:
    return {
        "observation_mode": str(config_text(base_args, "observation-mode", level_config.get("observation_mode", "terrain"))),
        "action_mode": str(config_text(base_args, "action-mode", level_config.get("action_mode", "preference_delta"))),
        "action_gain": float(config_text(base_args, "action-gain", level_config.get("action_gain", 2.0))),
        "max_uncertainty_lambda": float(
            config_text(base_args, "max-uncertainty-lambda", level_config.get("max_uncertainty_lambda", 1.0))
        ),
        "reward_cost_key": str(reward_cost_key_override or level_config.get("reward_cost_key", "scalar_cost")),
    }


def attack_value(env_attack: dict[str, Any], base_args: dict[str, Any], key: str, default: Any) -> Any:
    return env_attack.get(key, config_text(base_args, key.replace("_", "-"), default))


def make_context(
    episode: Any,
    runtime: dict[str, Any],
    env_attack: dict[str, Any],
    base_args: dict[str, Any],
) -> EpisodePlannerContext:
    return EpisodePlannerContext(
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


def tensor_action(policy: LRRPolicy, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    action, clean_action, residual, _gate = tensor_action_details(policy, obs)
    return action, clean_action, residual


def tensor_action_details(policy: LRRPolicy, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=policy.device).unsqueeze(0)
    with torch.no_grad():
        output = policy(obs_tensor)
    return (
        output.action.squeeze(0).detach().cpu().numpy().astype(np.float32),
        output.clean_action.squeeze(0).detach().cpu().numpy().astype(np.float32),
        output.residual.squeeze(0).detach().cpu().numpy().astype(np.float32),
        output.repair_gate.squeeze(0).detach().cpu().numpy().astype(np.float32),
    )


def clean_policy_action(clean_policy: torch.nn.Module, obs: np.ndarray, device: str | torch.device) -> np.ndarray:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        action = clean_policy.get_deterministic_action(obs_tensor)
    return action.squeeze(0).detach().cpu().numpy().astype(np.float32)


def action_result(
    episode: Any,
    action: np.ndarray,
    runtime: dict[str, Any],
    env_attack: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float, float]:
    weights = action_to_planning_weights(
        episode,
        action,
        action_mode=str(runtime["action_mode"]),
        action_gain=float(runtime["action_gain"]),
    )
    lambda_uncertainty = action_to_uncertainty_lambda(
        action,
        max_uncertainty_lambda=float(runtime["max_uncertainty_lambda"]),
    )
    active_attack = env_attack if attack_enabled(env_attack) else {}
    result = plan_with_weights(
        episode,
        weights,
        lambda_uncertainty=lambda_uncertainty,
        allow_diagonal=True,
        attacker_temperature=float(active_attack.get("attacker_temperature", 0.5)),
        attacker_response=str(active_attack.get("attacker_response", "zscore_topk")),
        attacker_top_fraction=float(active_attack.get("attacker_top_fraction", 0.15)),
        attacker_sharpness=float(active_attack.get("attacker_sharpness", 3.0)),
        attack_strength=float(active_attack.get("attack_strength", 1.0)),
    )
    cost = float(result.get(str(runtime["reward_cost_key"]), result.get("attacked_scalar_cost", result["scalar_cost"])))
    return result, float(cost), float(lambda_uncertainty)


def clean_nominal_cost(
    clean_policy: torch.nn.Module,
    episode: Any,
    map_size: int,
    runtime: dict[str, Any],
    device: str | torch.device,
) -> float:
    obs = compute_observation(
        episode,
        map_size,
        observation_mode=str(runtime["observation_mode"]),
        max_uncertainty_lambda=float(runtime["max_uncertainty_lambda"]),
    )
    action = clean_policy_action(clean_policy, obs, device)
    result, _cost, _lambda = action_result(episode, action, runtime, None)
    return float(result.get("scalar_cost", _cost))


def flatten_domains(domains: dict[str, tuple[int, list[Any]]]) -> list[tuple[str, int, Any]]:
    items: list[tuple[str, int, Any]] = []
    for domain, (_map_seed, episodes) in domains.items():
        for index, episode in enumerate(episodes):
            items.append((domain, index, episode))
    return items


def apply_attack_if_needed(episode: Any, env_attack: dict[str, Any], seed: int) -> Any:
    if attack_enabled(env_attack):
        return apply_environment_attack_to_episode(episode, env_attack, np.random.default_rng(seed))
    return episode


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = list(rows[0].keys())
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), sort_keys=True) + "\n")


def evaluate_lrr_policy(
    policy: LRRPolicy,
    clean_policy: torch.nn.Module,
    domains: dict[str, tuple[int, list[Any]]],
    map_size: int,
    runtime: dict[str, Any],
    env_attack: dict[str, Any],
    seed: int,
    iteration: int,
    checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_domain, (map_seed, episodes) in domains.items():
        for attack_type in ("none", "environment"):
            nominal_costs: list[float] = []
            attacked_costs: list[float] = []
            pis: list[float] = []
            successes: list[float] = []
            lambda_values: list[float] = []
            residual_norms: list[float] = []
            gate_values: list[float] = []
            for episode_index, episode in enumerate(episodes):
                eval_episode = episode
                active_attack = None
                if attack_type == "environment":
                    eval_episode = apply_attack_if_needed(
                        episode,
                        env_attack,
                        seed + 500_000 + iteration * 10_000 + episode_index,
                    )
                    active_attack = env_attack
                nominal = clean_nominal_cost(clean_policy, episode, map_size, runtime, policy.device)
                obs = compute_observation(
                    eval_episode,
                    map_size,
                    observation_mode=str(runtime["observation_mode"]),
                    max_uncertainty_lambda=float(runtime["max_uncertainty_lambda"]),
                )
                action, _clean_action, residual, repair_gate = tensor_action_details(policy, obs)
                result, attacked_cost, lambda_uncertainty = action_result(eval_episode, action, runtime, active_attack)
                nominal_costs.append(nominal)
                attacked_costs.append(float(attacked_cost))
                pis.append(performance_index(nominal, attacked_cost))
                successes.append(1.0 if bool(result.get("success", False)) else 0.0)
                lambda_values.append(float(lambda_uncertainty))
                residual_norms.append(float(np.linalg.norm(residual)))
                gate_values.append(float(np.mean(repair_gate)))
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
                    "mean_residual_norm": float(np.nanmean(residual_norms)),
                    "mean_repair_gate": float(np.nanmean(gate_values)),
                    "checkpoint_path": str(checkpoint_path or ""),
                }
            )
    return rows


def debug_row(
    iteration: int,
    domain: str,
    episode_index: int,
    clean_action: np.ndarray,
    current_action: np.ndarray,
    current_residual: np.ndarray,
    label: RepairLabel | None,
    scores: np.ndarray,
    candidates: list[Any],
) -> dict[str, Any]:
    best_index = int(np.nanargmax(scores)) if len(scores) else -1
    current_score = float(scores[0]) if len(scores) else float("nan")
    best_score = float(scores[best_index]) if best_index >= 0 else float("nan")
    best_action = candidates[best_index].action if best_index >= 0 else np.full_like(current_action, np.nan)
    return {
        "iteration": int(iteration),
        "state_id": f"{domain}:{episode_index}",
        "episode_id": int(episode_index),
        "eval_domain": domain,
        "current_score": current_score,
        "best_score": best_score,
        "improvement": float(best_score - current_score) if np.isfinite(current_score) else float("nan"),
        "current_action": current_action.tolist(),
        "clean_action": clean_action.tolist(),
        "best_action": np.asarray(best_action, dtype=np.float32).tolist(),
        "current_residual": current_residual.tolist(),
        "target_residual": label.target_residual.tolist() if label is not None else [],
        "chosen_candidate_type": candidates[best_index].candidate_type if best_index >= 0 else "",
        "target_candidate_type": label.chosen_candidate_type if label is not None else "",
        "positive_label": label is not None,
        "repair_weight": float(label.weight) if label is not None else 0.0,
    }


def default_output_dir(source_run_dir: Path, seed: int) -> Path:
    experiment_name = source_run_dir.parent.name
    return PROJECT_ROOT / "runs" / "lrr" / f"{experiment_name}_lrr" / f"seed{seed}"


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
        print(f"Would train LRR from {clean_checkpoint_path} -> {output_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    clean_policy, clean_checkpoint = load_cleanrl_agent(clean_checkpoint_path, device=args.device)
    obs_dim = int(clean_checkpoint["obs_dim"])
    action_dim = int(clean_checkpoint["action_dim"])
    action_low = np.zeros(action_dim, dtype=np.float32)
    action_high = np.ones(action_dim, dtype=np.float32)
    delta_max = action_range_fraction(args.delta_max_fraction, action_low, action_high, action_dim)
    search_radius = action_range_fraction(args.search_radius_fraction, action_low, action_high, action_dim)

    policy = LRRPolicy(
        clean_policy,
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        delta_max=delta_max,
        use_repair_gate=bool(args.use_repair_gate),
        gate_initial_logit=float(args.gate_initial_logit),
    )
    trainer = LRRTrainer(
        policy,
        LRRTrainerConfig(
            repair_batch_size=int(args.repair_batch_size),
            repair_epochs=int(args.repair_epochs),
            learning_rate=float(args.learning_rate),
            lambda_zero=float(args.lambda_zero),
            lambda_gate=float(args.lambda_gate),
            positive_gate_weight=float(args.positive_gate_weight),
            negative_gate_weight=float(args.negative_gate_weight),
        ),
    )
    local_config = LocalRepairConfig(
        improvement_epsilon=float(args.improvement_epsilon),
        beta=float(args.beta),
        w_max=float(args.w_max),
        target_mode=str(args.repair_target_mode),
        soft_target_temperature=float(args.soft_target_temperature),
        surface_ridge=float(args.surface_ridge),
        surface_max_step_fraction=float(args.surface_max_step_fraction),
        surface_allow_diagonal=not bool(args.surface_disable_diagonal),
        pairwise_candidate_mode=str(args.pairwise_candidate_mode),
        target_blend=float(args.repair_target_blend),
        target_residual_norm_clip=float(args.target_residual_norm_clip),
        reward_cost_key=str(runtime["reward_cost_key"]),
    )

    eval_episodes = int(args.eval_episodes)
    if eval_episodes <= 0:
        eval_episodes = source_num_eval_episodes(source_run_dir)
    map_size, domains = build_eval_episodes(source_run_dir, level_config, base_args, seed, int(eval_episodes))
    flat_episodes = flatten_domains(domains)
    if not flat_episodes:
        raise RuntimeError("no episodes available for LRR training")

    save_json(
        output_dir / "lrr_run_config.json",
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": "lrr",
            "variant": "selective_lrr" if args.use_repair_gate else "lrr",
            "source_run_dir": str(source_run_dir),
            "clean_policy_checkpoint": str(clean_checkpoint_path),
            "level_config": level_config,
            "base_config_args": base_args,
            "environment_attack": env_attack,
            "runtime": runtime,
            "args": vars(args),
            "resolved_eval_episodes": int(eval_episodes),
            "delta_max": delta_max.tolist(),
            "search_radius": search_radius.tolist(),
            "pairwise_candidate_mode": str(args.pairwise_candidate_mode),
            "repair_target_mode": str(args.repair_target_mode),
            "soft_target_temperature": float(args.soft_target_temperature),
            "surface_ridge": float(args.surface_ridge),
            "surface_max_step_fraction": float(args.surface_max_step_fraction),
            "surface_allow_diagonal": not bool(args.surface_disable_diagonal),
            "repair_target_blend": float(args.repair_target_blend),
            "target_residual_norm_clip": float(args.target_residual_norm_clip),
            "use_repair_gate": bool(args.use_repair_gate),
            "lambda_gate": float(args.lambda_gate),
            "gate_initial_logit": float(args.gate_initial_logit),
            "positive_gate_weight": float(args.positive_gate_weight),
            "negative_gate_weight": float(args.negative_gate_weight),
            "negative_gate_max_improvement": float(args.negative_gate_max_improvement),
        },
    )

    best_recovery_index = float("-inf")
    best_iteration = 0
    best_checkpoint_path = checkpoints_dir / "best_lrr_residual.pt"
    initial_checkpoint = checkpoints_dir / "lrr_residual_iter_00000.pt"
    save_lrr_checkpoint(
        initial_checkpoint,
        policy,
        {
            "clean_policy_checkpoint": str(clean_checkpoint_path),
            "iteration": 0,
            "delta_max": delta_max.tolist(),
            "search_radius": search_radius.tolist(),
            "use_repair_gate": bool(args.use_repair_gate),
        },
    )
    eval_rows = evaluate_lrr_policy(
        policy,
        clean_policy,
        domains,
        map_size,
        runtime,
        env_attack,
        seed,
        0,
        initial_checkpoint,
    )
    append_csv(output_dir / "lrr_recovery_curve.csv", eval_rows)
    env_pi = [float(row["performance_index"]) for row in eval_rows if row["attack_type"] == "environment"]
    if env_pi:
        best_recovery_index = float(np.nanmean(env_pi))
        shutil.copy2(initial_checkpoint, best_checkpoint_path)

    for iteration in range(1, int(args.num_iterations) + 1):
        rng = random.Random(seed + 1000 * iteration)
        sample_count = min(
            int(args.rollout_episodes_per_iteration),
            int(args.max_repair_states_per_iteration),
            len(flat_episodes),
        )
        selected = rng.sample(flat_episodes, sample_count) if sample_count < len(flat_episodes) else list(flat_episodes)

        repair_observations: list[np.ndarray] = []
        positive_labels: list[RepairLabel] = []
        zero_observations: list[np.ndarray] = []
        negative_gate_observations: list[np.ndarray] = []
        improvements: list[float] = []
        residual_norms: list[float] = []
        repair_gate_values: list[float] = []
        clean_action_norms: list[float] = []
        lrr_action_norms: list[float] = []
        debug_rows: list[dict[str, Any]] = []
        candidate_count = 0

        for local_index, (domain, episode_index, episode) in enumerate(selected):
            attacked_episode = apply_attack_if_needed(
                episode,
                env_attack,
                seed + 300_000 + iteration * 10_000 + local_index,
            )
            obs = compute_observation(
                attacked_episode,
                map_size,
                observation_mode=str(runtime["observation_mode"]),
                max_uncertainty_lambda=float(runtime["max_uncertainty_lambda"]),
            )
            current_action, clean_action, current_residual, repair_gate = tensor_action_details(policy, obs)
            zero_observations.append(obs.astype(np.float32))
            residual_norms.append(float(np.linalg.norm(current_residual)))
            repair_gate_values.append(float(np.mean(repair_gate)))
            clean_action_norms.append(float(np.linalg.norm(clean_action)))
            lrr_action_norms.append(float(np.linalg.norm(current_action)))

            if args.residual_only:
                continue

            nominal = clean_nominal_cost(clean_policy, episode, map_size, runtime, policy.device)
            context = make_context(attacked_episode, runtime, env_attack, base_args)
            repair = evaluate_local_repair(
                context,
                nominal,
                current_action,
                clean_action,
                action_low,
                action_high,
                search_radius,
                delta_max,
                config=local_config,
                evaluator=evaluate_planner_action,
            )
            candidate_count = len(repair.candidates)
            best_score = float(np.nanmax(repair.scores)) if len(repair.scores) else float("nan")
            current_score = float(repair.scores[0]) if len(repair.scores) else float("nan")
            local_improvement = float(best_score - current_score)
            improvements.append(local_improvement)
            debug_rows.append(
                debug_row(
                    iteration,
                    domain,
                    episode_index,
                    clean_action,
                    current_action,
                    current_residual,
                    repair.label,
                    repair.scores,
                    repair.candidates,
                )
            )
            if repair.label is not None:
                repair_observations.append(obs.astype(np.float32))
                positive_labels.append(repair.label)
            elif np.isfinite(local_improvement) and local_improvement <= float(args.negative_gate_max_improvement):
                negative_gate_observations.append(obs.astype(np.float32))

        if repair_observations:
            repair_obs_array = np.stack(repair_observations, axis=0)
        else:
            repair_obs_array = np.zeros((0, obs_dim), dtype=np.float32)
        if zero_observations:
            zero_obs_array = np.stack(zero_observations, axis=0)
        else:
            zero_obs_array = np.zeros((0, obs_dim), dtype=np.float32)
        if negative_gate_observations:
            negative_gate_obs_array = np.stack(negative_gate_observations, axis=0)
        else:
            negative_gate_obs_array = np.zeros((0, obs_dim), dtype=np.float32)

        train_metrics = trainer.update(
            repair_obs_array,
            positive_labels,
            zero_obs_array,
            negative_gate_obs_array,
        )
        checkpoint_path = checkpoints_dir / f"lrr_residual_iter_{iteration:05d}.pt"
        save_lrr_checkpoint(
            checkpoint_path,
            policy,
            {
                "clean_policy_checkpoint": str(clean_checkpoint_path),
                "iteration": iteration,
                "delta_max": delta_max.tolist(),
                "search_radius": search_radius.tolist(),
                "use_repair_gate": bool(args.use_repair_gate),
                "runtime": runtime,
            },
        )
        eval_rows = evaluate_lrr_policy(
            policy,
            clean_policy,
            domains,
            map_size,
            runtime,
            env_attack,
            seed,
            iteration,
            checkpoint_path,
        )
        append_csv(output_dir / "lrr_recovery_curve.csv", eval_rows)
        append_jsonl(output_dir / "lrr_repair_labels.jsonl", debug_rows)

        env_pi = [float(row["performance_index"]) for row in eval_rows if row["attack_type"] == "environment"]
        current_recovery_index = float(np.nanmean(env_pi)) if env_pi else float("nan")
        if np.isfinite(current_recovery_index) and current_recovery_index > best_recovery_index:
            best_recovery_index = current_recovery_index
            best_iteration = int(iteration)
            shutil.copy2(checkpoint_path, best_checkpoint_path)
        mean_improvement = float(np.nanmean(improvements)) if improvements else 0.0
        max_improvement = float(np.nanmax(improvements)) if improvements else 0.0
        target_residual_norms = [float(np.linalg.norm(label.target_residual)) for label in positive_labels]
        metrics_row = {
            "iteration": int(iteration),
            "global_step": int(iteration),
            "lrr/loss_total": train_metrics["lrr/loss_total"],
            "lrr/loss_repair": train_metrics["lrr/loss_repair"],
            "lrr/loss_zero": train_metrics["lrr/loss_zero"],
            "lrr/loss_gate": train_metrics["lrr/loss_gate"],
            "lrr/num_states_evaluated": int(0 if args.residual_only else len(selected)),
            "lrr/num_positive_repairs": int(len(positive_labels)),
            "lrr/num_negative_gate_states": int(len(negative_gate_observations)),
            "lrr/positive_repair_rate": float(len(positive_labels) / max(len(selected), 1)) if not args.residual_only else 0.0,
            "lrr/mean_improvement": mean_improvement,
            "lrr/max_improvement": max_improvement,
            "lrr/mean_repair_weight": train_metrics["lrr/mean_repair_weight"],
            "lrr/mean_target_residual_norm": float(np.nanmean(target_residual_norms)) if target_residual_norms else 0.0,
            "lrr/max_target_residual_norm": float(np.nanmax(target_residual_norms)) if target_residual_norms else 0.0,
            "lrr/mean_residual_norm": float(np.nanmean(residual_norms)) if residual_norms else 0.0,
            "lrr/max_residual_norm": float(np.nanmax(residual_norms)) if residual_norms else 0.0,
            "lrr/mean_repair_gate": float(np.nanmean(repair_gate_values)) if repair_gate_values else 1.0,
            "lrr/min_repair_gate": float(np.nanmin(repair_gate_values)) if repair_gate_values else 1.0,
            "lrr/max_repair_gate": float(np.nanmax(repair_gate_values)) if repair_gate_values else 1.0,
            "lrr/mean_clean_action_norm": float(np.nanmean(clean_action_norms)) if clean_action_norms else 0.0,
            "lrr/mean_lrr_action_norm": float(np.nanmean(lrr_action_norms)) if lrr_action_norms else 0.0,
            "lrr/candidate_count": int(candidate_count),
            "lrr/search_radius": float(np.mean(search_radius)),
            "lrr/delta_max": float(np.mean(delta_max)),
            "lrr/pairwise_candidate_mode": str(args.pairwise_candidate_mode),
            "lrr/repair_target_mode": str(args.repair_target_mode),
            "lrr/soft_target_temperature": float(args.soft_target_temperature),
            "lrr/surface_ridge": float(args.surface_ridge),
            "lrr/surface_max_step_fraction": float(args.surface_max_step_fraction),
            "lrr/surface_allow_diagonal": not bool(args.surface_disable_diagonal),
            "lrr/use_repair_gate": bool(args.use_repair_gate),
            "lrr/lambda_gate": float(args.lambda_gate),
            "lrr/negative_gate_max_improvement": float(args.negative_gate_max_improvement),
            "performance_index": current_recovery_index,
            "final_recovery_index": current_recovery_index,
            "best_recovery_index": best_recovery_index,
            "best_iteration": int(best_iteration),
            "best_checkpoint_path": str(best_checkpoint_path),
            "checkpoint_path": str(checkpoint_path),
        }
        append_csv(output_dir / "lrr_metrics.csv", [metrics_row])
        print(
            " ".join(
                [
                    f"iter={iteration}",
                    f"pi={current_recovery_index:.3f}",
                    f"positive={len(positive_labels)}/{len(selected)}",
                    f"gate={metrics_row['lrr/mean_repair_gate']:.3f}",
                    f"mean_improvement={mean_improvement:.3f}",
                    f"loss={train_metrics['lrr/loss_total']:.6f}",
                ]
            ),
            flush=True,
        )

    print(f"Saved LRR outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
