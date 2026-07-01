"""Shared recovery-protocol helpers for the paper experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from envs.attack_wrappers import (
    apply_environment_attack_to_episode,
    apply_observation_attack,
    attack_enabled,
    load_attack_config,
)
from utils.evaluation_policy import (
    load_model,
    predict_action,
    resolve_action_config,
    resolve_observation_mode,
)
from utils.cleanrl_policy import predict_acbr_candidate_scores
from utils.metrics import (
    DEFAULT_MAP_SEED_POOL_SIZE,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    candidate_planner_configs,
    compute_observation,
    make_curriculum_planning_episode,
    plan_with_weights,
)
from utils.planner_regret import (
    CandidateActionConfig,
    generate_candidate_actions,
    merge_candidate_action_sets,
    planning_config_to_action,
)
from utils.planner_residual_belief import (
    PlannerResidualFeatureBuilder,
    PlannerResidualFeatureConfig,
    neutral_probe_action,
)


def acbr_context_array(features: np.ndarray | list[float] | None, context_dim: int) -> np.ndarray:
    dim = max(int(context_dim), 1)
    if features is None:
        return np.zeros(dim, dtype=np.float32)
    array = np.asarray(features, dtype=np.float32).reshape(-1)
    if array.size >= dim:
        return array[:dim].astype(np.float32)
    out = np.zeros(dim, dtype=np.float32)
    out[: array.size] = array
    return out


def acbr_structured_candidates(
    episode: Any,
    action_dim: int,
    action_mode: str,
    action_gain: float,
    max_uncertainty_lambda: float,
    count: int,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if int(count) <= 0:
        return np.zeros((0, int(action_dim)), dtype=np.float32), []
    configs = candidate_planner_configs(episode, max_uncertainty_lambda=max_uncertainty_lambda)
    preferred = [
        "safe_rover",
        "mission_safe_blend",
        "hazard_only_uncertainty_high",
        "energy_only_uncertainty_high",
        "illumination_only_uncertainty_high",
        "communication_only_uncertainty_high",
        "mission_hazard_blend_uncertainty_high",
        "emergency_uncertainty_rule",
        "distance_only",
        "distance_only_uncertainty_low",
    ]
    names = [name for name in preferred if name in configs]
    names.extend([name for name in configs if name not in names])
    actions: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    for name in names[: int(count)]:
        config = configs[name]
        action = planning_config_to_action(
            episode,
            np.asarray(config["weights"], dtype=np.float32),
            float(config["lambda_uncertainty"]),
            action_dim=int(action_dim),
            action_mode=action_mode,
            action_gain=action_gain,
            max_uncertainty_lambda=max_uncertainty_lambda,
        )
        clipped = np.clip(np.asarray(action, dtype=np.float32), low, high)
        if not any(np.allclose(clipped, item, atol=1e-6, rtol=0.0) for item in actions):
            actions.append(clipped)
            meta.append({"kind": f"structured:{name}"})
    if not actions:
        return np.zeros((0, int(action_dim)), dtype=np.float32), []
    return np.stack(actions, axis=0).astype(np.float32), meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, default="configs/ppo_lunar_map_pool_relative_reward.json")
    parser.add_argument("--checkpoint", type=str, default="runs/robustness/seed0/nominal_ppo/checkpoint.pt")
    parser.add_argument(
        "--attack-config",
        type=str,
        default="runs/robustness/attack_sweep/recommended_attack_config.json",
    )
    parser.add_argument("--output-dir", type=str, default="runs/robustness/recovery_env_attack")
    parser.add_argument("--finetune-timesteps", type=int, default=30000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--num-eval-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--in-domain-seed", type=int, default=909)
    parser.add_argument("--heldout-seed", type=int, default=1919)
    parser.add_argument("--map-pool-size", type=int, default=None)
    parser.add_argument("--observation-attack-config", type=str, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output directory before running. Refuses paths outside the project root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_base_args(path_text: str) -> dict[str, Any]:
    data = json.loads(Path(path_text).read_text(encoding="utf-8"))
    return dict(data.get("args", data))


def config_value(config: dict[str, Any], hyphen_key: str, default: Any) -> Any:
    return config.get(hyphen_key, config.get(hyphen_key.replace("-", "_"), default))


def load_environment_attack(path_text: str) -> dict[str, Any]:
    data = load_attack_config(path_text)
    if "environment_attack" in data:
        data = dict(data["environment_attack"])
    data.setdefault("enabled", True)
    data.setdefault("type", "env_zscore_topk")
    data.setdefault("attacker_response", "zscore_topk")
    data.setdefault("attacker_temperature", 0.3)
    data.setdefault("attacker_top_fraction", 0.35)
    data.setdefault("attacker_sharpness", 5.0)
    data.setdefault("attack_strength", 1.0)
    data.setdefault("apply_during_training", True)
    data.setdefault("reward_uses_attacked_cost", True)
    return data


def training_script_for_algo(algo: str) -> str:
    normalized = str(algo).lower()
    if normalized == "ppo":
        return "train_cleanrl_ppo.py"
    if normalized == "sac":
        return "train_cleanrl_sac.py"
    raise ValueError(f"unsupported training algorithm: {algo}")


def run_name_for_algo(algo: str, seed: int) -> str:
    normalized = str(algo).lower()
    if normalized == "ppo":
        return f"cleanrl_ppo_costmap_seed{seed}"
    if normalized == "sac":
        return f"cleanrl_sac_costmap_seed{seed}"
    raise ValueError(f"unsupported training algorithm: {algo}")


def build_train_command(
    python_exe: str,
    base_args: dict[str, Any],
    init_checkpoint: Path,
    env_attack: dict[str, Any],
    observation_attack: dict[str, Any] | None,
    output_dir: Path,
    chunk_index: int,
    chunk_timesteps: int,
    seed: int,
    algo: str = "ppo",
    extra_train_args: dict[str, Any] | None = None,
) -> tuple[list[str], Path]:
    log_dir = output_dir / "train_chunks" / f"chunk_{chunk_index:04d}"
    train_args = dict(base_args)
    train_args["log-dir"] = str(log_dir)
    train_args["init-checkpoint"] = str(init_checkpoint)
    train_args["total-timesteps"] = int(chunk_timesteps)
    train_args["seed"] = int(seed)
    if attack_enabled(env_attack):
        train_args["environment-attack-config"] = json.dumps(env_attack)
    if attack_enabled(observation_attack):
        train_args["observation-attack-config"] = json.dumps(observation_attack)
    if attack_enabled(env_attack) and str(env_attack.get("type", "env_zscore_topk")) == "env_zscore_topk":
        train_args["reward-cost-key"] = "soft_attacked_scalar_cost"
        train_args["attacker-response"] = env_attack.get("attacker_response", "zscore_topk")
        train_args["attacker-temperature"] = env_attack.get("attacker_temperature", 0.3)
        train_args["attacker-top-fraction"] = env_attack.get("attacker_top_fraction", 0.35)
        train_args["attacker-sharpness"] = env_attack.get("attacker_sharpness", 5.0)
        train_args["attack-strength"] = env_attack.get("attack_strength", 1.0)
    else:
        # Environment attacks that mutate map information, including
        # env_belief_mismatch, train against scalar_cost. For belief mismatch,
        # scalar_cost is evaluated on the true map after planning on the belief
        # map.
        train_args["reward-cost-key"] = "scalar_cost"
    train_args["early-stop-patience"] = 0
    train_args.pop("environment_attack", None)
    train_args.pop("observation_attack", None)
    train_args.pop("finetune-timesteps", None)
    if extra_train_args:
        train_args.update(extra_train_args)

    command = [python_exe, training_script_for_algo(algo)]
    for key, value in train_args.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command, log_dir / run_name_for_algo(algo, seed) / "final_model.pt"


def checkpoint_step(path: Path) -> int:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    return int(checkpoint.get("global_step", 0))


def require_existing_file(path_text: str | None, label: str) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def clean_output_dir(output_dir: Path, dry_run: bool) -> None:
    if not output_dir.exists():
        return
    project_root = Path(__file__).resolve().parent.resolve()
    resolved_output = output_dir.resolve()
    try:
        resolved_output.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"refusing to clean output outside project root: {resolved_output}") from exc
    if dry_run:
        print(f"Would remove output directory: {resolved_output}")
        return
    shutil.rmtree(resolved_output)
    print(f"Removed output directory: {resolved_output}")


def write_run_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    base_args: dict[str, Any],
    env_attack: dict[str, Any],
    observation_attack: dict[str, Any],
    seed: int,
    map_size: int,
    scenario: str,
    map_pool_size: int,
) -> None:
    num_envs = int(config_value(base_args, "num-envs", 1))
    num_steps = int(config_value(base_args, "num-steps", 1))
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command_args": vars(args),
        "resolved": {
            "seed": int(seed),
            "map_size": int(map_size),
            "scenario": scenario,
            "map_pool_size": int(map_pool_size),
            "batch_size": int(num_envs * num_steps),
        },
        "base_config_args": base_args,
        "environment_attack": env_attack,
        "observation_attack": observation_attack,
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def generate_episodes(
    num_episodes: int,
    seed: int,
    map_size: int,
    scenario: str,
    fixed_map_seed: int,
    map_pool_size: int,
    min_start_goal_distance_ratio: float = 0.55,
) -> list[Any]:
    rng = np.random.default_rng(seed)
    map_cache: dict[Any, Any] = {}
    return [
        make_curriculum_planning_episode(
            map_size=map_size,
            rng=rng,
            allow_diagonal=True,
            scenario=scenario,
            min_start_goal_distance_ratio=min_start_goal_distance_ratio,
            map_sampling_mode="map_seed_pool",
            fixed_map_seed=fixed_map_seed,
            map_seed_pool_size=map_pool_size,
            map_cache=map_cache,
        )
        for _ in tqdm(range(num_episodes), desc=f"episodes seed{fixed_map_seed}", leave=False)
    ]


def eval_attack_cost(result: dict[str, Any], env_attack: dict[str, Any] | None) -> float:
    if not attack_enabled(env_attack):
        return float(result.get("scalar_cost", np.nan))
    if str(env_attack.get("type", "env_zscore_topk")) == "env_zscore_topk":
        return float(result.get("soft_attacked_scalar_cost", result.get("attacked_scalar_cost", np.nan)))
    return float(result.get("scalar_cost", np.nan))


def policy_plan_config(
    model_type: str,
    model: Any,
    model_config: dict[str, Any],
    episode: Any,
    map_size: int,
    observation_attack: dict[str, Any] | None,
    environment_attack: dict[str, Any] | None,
    obs_seed: int,
) -> tuple[np.ndarray, float]:
    observation_mode = resolve_observation_mode("auto", model_config)
    action_mode, action_gain, max_lambda = resolve_action_config("auto", None, None, model_config)
    config = model_config.get("config") if isinstance(model_config.get("config"), dict) else model_config
    obs = compute_observation(
        episode,
        map_size,
        observation_mode=observation_mode,
        max_uncertainty_lambda=max_lambda,
    )
    if attack_enabled(observation_attack):
        obs_space = spaces.Box(low=0.0, high=1.0, shape=obs.shape, dtype=np.float32)
        obs = apply_observation_attack(obs, observation_attack, np.random.default_rng(obs_seed), obs_space)
    residual_features = None
    if str(model_config.get("policy_class", "")) == "cleanrl_residual_belief_actor_critic":
        action_dim = int(getattr(model, "action_dim", model_config.get("action_dim", 6)))
        probe_action = neutral_probe_action(action_dim, 0.5)
        probe_weights = action_to_planning_weights(
            episode,
            probe_action,
            action_mode=action_mode,
            action_gain=action_gain,
        )
        probe_lambda = action_to_uncertainty_lambda(probe_action, max_uncertainty_lambda=max_lambda)
        probe_result = plan_with_weights(
            episode,
            probe_weights,
            lambda_uncertainty=probe_lambda,
            allow_diagonal=True,
            attacker_temperature=float(environment_attack.get("attacker_temperature", 0.5))
            if attack_enabled(environment_attack)
            else 0.5,
            attacker_response=str(environment_attack.get("attacker_response", "zscore_topk"))
            if attack_enabled(environment_attack)
            else "zscore_topk",
            attacker_top_fraction=float(environment_attack.get("attacker_top_fraction", 0.15))
            if attack_enabled(environment_attack)
            else 0.15,
            attacker_sharpness=float(environment_attack.get("attacker_sharpness", 3.0))
            if attack_enabled(environment_attack)
            else 3.0,
            attack_strength=float(environment_attack.get("attack_strength", 1.0))
            if attack_enabled(environment_attack)
            else 1.0,
        )
        builder = PlannerResidualFeatureBuilder(
            PlannerResidualFeatureConfig(
                action_dim=action_dim,
                normalize_features=bool(config.get("prb_normalize_features", True)),
                feature_clip=float(config.get("prb_feature_clip", 5.0)),
                use_component_costs=bool(config.get("prb_use_component_costs", True)),
                use_scalar_cost=bool(config.get("prb_use_scalar_cost", True)),
            )
        )
        reward_cost_key = str(config.get("reward_cost_key", "attacked_scalar_cost"))
        residual_features = builder.build_from_info(probe_result, probe_action, reward_cost_key).features
    action = predict_action(model_type, model, obs, residual_features=residual_features)
    if bool(config.get("acbr_enabled", False)) and hasattr(model, "predict_acbr_costs"):
        action_dim = int(getattr(model, "action_dim", model_config.get("action_dim", len(action))))
        low = np.zeros(action_dim, dtype=np.float32)
        high = np.ones(action_dim, dtype=np.float32)
        candidate_config = CandidateActionConfig(
            num_candidates=int(config.get("pr_num_candidates", 24)),
            local_sigma=float(config.get("pr_local_sigma", 0.12)),
            num_random_candidates=int(config.get("pr_num_random_candidates", 6)),
            include_policy_action=True,
            include_nominal_action=True,
            include_zero_delta=True,
            include_axis_perturbations=True,
            risk_local_sigma=float(config.get("pr_risk_local_sigma", 0.20)),
            include_risk_axis_perturbations=True,
            include_risk_block_perturbations=True,
            zero_action_value=0.5 if action_mode == "preference_delta" else 0.0,
        )
        candidates, candidate_meta = generate_candidate_actions(
            action,
            action,
            low,
            high,
            candidate_config,
            np.random.default_rng(obs_seed + 700_000),
        )
        structured_candidates, structured_meta = acbr_structured_candidates(
            episode,
            action_dim,
            action_mode,
            float(action_gain),
            float(max_lambda),
            int(config.get("pr_num_structured_candidates", 10)),
            low,
            high,
        )
        if structured_candidates.size > 0:
            candidates, candidate_meta = merge_candidate_action_sets(
                candidates,
                candidate_meta,
                structured_candidates,
                structured_meta,
                low,
                high,
                dedup_tol=float(candidate_config.dedup_tol),
            )
        context = acbr_context_array(residual_features, int(config.get("acbr_context_dim", 16)))
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"
        mean, std = predict_acbr_candidate_scores(model, obs, candidates, context_features=context, device=device)
        distance = np.mean((candidates - action.reshape(1, -1)) ** 2, axis=1)
        scores = (
            mean
            + float(config.get("acbr_uncertainty_coef", 0.25)) * std
            + (
                float(config.get("acbr_anchor_penalty", 0.15))
                + float(config.get("acbr_policy_penalty", 0.05))
            )
            * distance
        )
        if np.isfinite(scores).any():
            selected_index = int(np.nanargmin(scores))
            if bool(config.get("acbr_benefit_gate_enabled", False)):
                policy_index = int(np.argmin(distance))
                predicted_improvement = float(scores[policy_index] - scores[selected_index])
                if selected_index == policy_index or predicted_improvement < float(config.get("acbr_benefit_margin", 0.0)):
                    selected_index = -1
            if selected_index >= 0:
                action = candidates[selected_index].astype(np.float32)
    weights = action_to_planning_weights(
        episode,
        action,
        action_mode=action_mode,
        action_gain=action_gain,
    )
    lambda_uncertainty = action_to_uncertainty_lambda(action, max_uncertainty_lambda=max_lambda)
    return weights, float(lambda_uncertainty)


def safe_nanmean(values: list[float], default: float = float("nan")) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.isnan(array).all():
        return float(default)
    return float(np.nanmean(array))


def safe_nanstd(values: list[float], default: float = float("nan")) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.isnan(array).all():
        return float(default)
    return float(np.nanstd(array))


def evaluate_checkpoint(
    checkpoint_path: Path,
    global_step: int,
    episodes_by_domain: dict[str, tuple[int, list[Any]]],
    map_size: int,
    env_attack: dict[str, Any],
    observation_attack: dict[str, Any] | None,
    seed: int,
) -> list[dict[str, Any]]:
    model_type, model, model_config = load_model(checkpoint_path, "auto")
    rows = []
    attack_cases: list[tuple[str, dict[str, Any] | None, dict[str, Any] | None]] = [
        ("none", None, None),
        ("environment", None, env_attack),
    ]
    if attack_enabled(observation_attack):
        attack_cases.append(("observation", observation_attack, None))
        attack_cases.append(("combined", observation_attack, env_attack))

    for eval_domain, (map_seed, episodes) in episodes_by_domain.items():
        for attack_type, obs_attack, active_env_attack in attack_cases:
            nominal_costs = []
            attacked_costs = []
            rewards = []
            successes = []
            path_lengths = []
            exposure_ratios = []
            hazard_exposures = []
            belief_hazard_exposures = []
            uncertainty_exposures = []
            belief_uncertainty_exposures = []
            belief_costs = []
            map_mismatch_penalties = []
            map_mismatch_abs_errors = []
            path_confidences = []
            true_belief_mismatch_flags = []
            mismatched_cells = []
            mean_belief_abs_errors = []
            mean_true_minus_belief_errors = []
            mean_selected_confidences = []
            lambda_values = []
            weight_values = []
            for episode_index, episode in enumerate(episodes):
                eval_episode = episode
                if attack_enabled(active_env_attack):
                    eval_episode = apply_environment_attack_to_episode(
                        episode,
                        active_env_attack,
                        np.random.default_rng(seed + 300_000 + episode_index),
                    )
                clean_weights, clean_lambda = policy_plan_config(
                    model_type,
                    model,
                    model_config,
                    episode,
                    map_size,
                    None,
                    None,
                    seed,
                )
                clean_result = plan_with_weights(
                    episode,
                    clean_weights,
                    lambda_uncertainty=clean_lambda,
                    allow_diagonal=True,
                )
                weights, lambda_uncertainty = policy_plan_config(
                    model_type,
                    model,
                    model_config,
                    eval_episode,
                    map_size,
                    obs_attack,
                    active_env_attack,
                    seed + 100_000 + episode_index,
                )
                result = plan_with_weights(
                    eval_episode,
                    weights,
                    lambda_uncertainty=lambda_uncertainty,
                    allow_diagonal=True,
                    attacker_temperature=float(active_env_attack.get("attacker_temperature", 0.5))
                    if attack_enabled(active_env_attack)
                    else 0.5,
                    attacker_response=str(active_env_attack.get("attacker_response", "zscore_topk"))
                    if attack_enabled(active_env_attack)
                    else "zscore_topk",
                    attacker_top_fraction=float(active_env_attack.get("attacker_top_fraction", 0.15))
                    if attack_enabled(active_env_attack)
                    else 0.15,
                    attacker_sharpness=float(active_env_attack.get("attacker_sharpness", 3.0))
                    if attack_enabled(active_env_attack)
                    else 3.0,
                    attack_strength=float(active_env_attack.get("attack_strength", 1.0))
                    if attack_enabled(active_env_attack)
                    else 1.0,
                )
                nominal_costs.append(float(clean_result.get("scalar_cost", np.nan)))
                attacked_cost = eval_attack_cost(result, active_env_attack)
                attacked_costs.append(attacked_cost)
                rewards.append(-attacked_cost)
                successes.append(1.0 if bool(result.get("success", False)) else 0.0)
                path_lengths.append(float(result.get("path_length", np.nan)))
                exposure_ratios.append(float(result.get("attacked_cell_exposure_ratio", 0.0)))
                hazard_exposures.append(float(result.get("hazard_exposure", np.nan)))
                belief_hazard_exposures.append(float(result.get("belief_hazard_exposure", np.nan)))
                uncertainty_exposures.append(float(result.get("uncertainty_exposure", np.nan)))
                belief_uncertainty_exposures.append(float(result.get("belief_uncertainty_exposure", np.nan)))
                belief_costs.append(float(result.get("belief_scalar_cost", result.get("scalar_cost", np.nan))))
                map_mismatch_penalties.append(float(result.get("map_mismatch_penalty", 0.0)))
                map_mismatch_abs_errors.append(float(result.get("map_mismatch_abs_error", 0.0)))
                path_confidences.append(float(result.get("mean_path_confidence", np.nan)))
                true_belief_mismatch_flags.append(1.0 if bool(result.get("true_belief_mismatch", False)) else 0.0)
                attack_metadata = result.get("attack_metadata", {}) if isinstance(result.get("attack_metadata", {}), dict) else {}
                mismatched_cells.append(float(attack_metadata.get("mismatched_cells", np.nan)))
                mean_belief_abs_errors.append(float(attack_metadata.get("mean_belief_abs_error", np.nan)))
                mean_true_minus_belief_errors.append(float(attack_metadata.get("mean_true_minus_belief_error", np.nan)))
                mean_selected_confidences.append(float(attack_metadata.get("mean_selected_confidence", np.nan)))
                lambda_values.append(float(lambda_uncertainty))
                weight_values.append(np.asarray(weights, dtype=np.float32))

            mean_nominal = safe_nanmean(nominal_costs)
            mean_attacked = safe_nanmean(attacked_costs)
            absolute = mean_attacked - mean_nominal
            mean_weights = (
                np.nanmean(np.stack(weight_values, axis=0), axis=0)
                if weight_values
                else np.full(5, np.nan)
            )
            rows.append(
                {
                    "global_step": int(global_step),
                    "eval_domain": eval_domain,
                    "map_pool_seed": int(map_seed),
                    "attack_type": attack_type,
                    "mean_nominal_scalar_cost": mean_nominal,
                    "mean_attacked_scalar_cost": mean_attacked,
                    "std_attacked_scalar_cost": safe_nanstd(attacked_costs),
                    "absolute_degradation": absolute,
                    "relative_degradation": float(absolute / (abs(mean_nominal) + 1e-8)),
                    "success_rate": safe_nanmean(successes),
                    "mean_reward": safe_nanmean(rewards),
                    "mean_path_length": safe_nanmean(path_lengths),
                    "mean_attacked_cell_exposure_ratio": safe_nanmean(exposure_ratios),
                    "mean_hazard_exposure": safe_nanmean(hazard_exposures),
                    "mean_belief_hazard_exposure": safe_nanmean(belief_hazard_exposures),
                    "mean_uncertainty_exposure": safe_nanmean(uncertainty_exposures),
                    "mean_belief_uncertainty_exposure": safe_nanmean(belief_uncertainty_exposures),
                    "mean_belief_scalar_cost": safe_nanmean(belief_costs),
                    "mean_map_mismatch_penalty": safe_nanmean(map_mismatch_penalties, default=0.0),
                    "mean_map_mismatch_abs_error": safe_nanmean(map_mismatch_abs_errors, default=0.0),
                    "mean_path_confidence": safe_nanmean(path_confidences),
                    "true_belief_mismatch_rate": safe_nanmean(true_belief_mismatch_flags, default=0.0),
                    "mean_mismatched_cells": safe_nanmean(mismatched_cells),
                    "mean_belief_abs_error": safe_nanmean(mean_belief_abs_errors),
                    "mean_true_minus_belief_error": safe_nanmean(mean_true_minus_belief_errors),
                    "mean_selected_confidence": safe_nanmean(mean_selected_confidences),
                    "mean_lambda_uncertainty": safe_nanmean(lambda_values),
                    "mean_weight_distance": float(mean_weights[0]),
                    "mean_weight_energy": float(mean_weights[1]),
                    "mean_weight_hazard": float(mean_weights[2]),
                    "mean_weight_communication": float(mean_weights[3]),
                    "mean_weight_illumination": float(mean_weights[4]),
                    "checkpoint_path": str(checkpoint_path),
                }
            )
    return rows


def plot_recovery(frame: pd.DataFrame, output_dir: Path) -> None:
    def save_figure(fig: plt.Figure, filename: str) -> None:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figures_dir / filename, dpi=180)
        # Keep the historical root-level filenames so older notes and links do
        # not break, while making figures/ the consistent primary location.
        fig.savefig(output_dir / filename, dpi=180)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (domain, attack_type), group in frame.groupby(["eval_domain", "attack_type"]):
        group = group.sort_values("global_step")
        ax.plot(group["global_step"], group["mean_attacked_scalar_cost"], marker="o", label=f"{domain}:{attack_type}")
    ax.set_xlabel("global_step")
    ax.set_ylabel("mean_attacked_scalar_cost")
    ax.set_title("Attack recovery curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_recovery_curve_attacked_cost.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for (domain, attack_type), group in frame.groupby(["eval_domain", "attack_type"]):
        group = group.sort_values("global_step")
        ax.plot(group["global_step"], group["relative_degradation"], marker="o", label=f"{domain}:{attack_type}")
    ax.set_xlabel("global_step")
    ax.set_ylabel("relative_degradation")
    ax.set_title("Attack degradation during fine-tuning")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_recovery_curve_degradation.png")
    plt.close(fig)

    attack_steps = sorted(
        int(value)
        for value in frame.loc[frame["attack_type"] == "environment", "global_step"].dropna().unique()
    )
    stage_labels = ["No attack\nstep 0"]
    stage_labels.extend(
        ["Attack\nstep 0" if step == 0 else f"FT\nstep {step}" for step in attack_steps]
    )
    x = np.arange(len(stage_labels))

    fig, ax = plt.subplots(figsize=(max(9.0, 0.65 * len(stage_labels)), 4.8))
    for domain, domain_frame in frame.groupby("eval_domain"):
        no_attack = domain_frame[
            (domain_frame["attack_type"] == "none") & (domain_frame["global_step"] == 0)
        ]
        env_rows = domain_frame[domain_frame["attack_type"] == "environment"].copy()
        if no_attack.empty or env_rows.empty:
            continue
        env_rows = env_rows.set_index("global_step")
        costs = [float(no_attack.iloc[0]["mean_attacked_scalar_cost"])]
        costs.extend(
            float(env_rows.loc[step]["mean_attacked_scalar_cost"])
            if step in env_rows.index
            else np.nan
            for step in attack_steps
        )
        ax.plot(x, costs, marker="o", linewidth=2.2, label=domain)
    ax.axvspan(0.5, 1.5, color="tab:red", alpha=0.08, label="attack applied")
    if len(stage_labels) > 2:
        ax.axvspan(1.5, len(stage_labels) - 0.5, color="tab:green", alpha=0.08, label="fine-tuning")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=8)
    ax.set_ylabel("Mean scalar cost (lower is better)")
    ax.set_title("Continuous attack and recovery stages - cost")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_continuous_stage_recovery_cost.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(9.0, 0.65 * len(stage_labels)), 4.8))
    for domain, domain_frame in frame.groupby("eval_domain"):
        no_attack = domain_frame[
            (domain_frame["attack_type"] == "none") & (domain_frame["global_step"] == 0)
        ]
        env_rows = domain_frame[domain_frame["attack_type"] == "environment"].copy()
        if no_attack.empty or env_rows.empty:
            continue
        base_cost = float(no_attack.iloc[0]["mean_attacked_scalar_cost"])
        env_rows = env_rows.set_index("global_step")
        costs = [base_cost]
        costs.extend(
            float(env_rows.loc[step]["mean_attacked_scalar_cost"])
            if step in env_rows.index
            else np.nan
            for step in attack_steps
        )
        performance = [base_cost / max(abs(cost), 1e-8) for cost in costs]
        ax.plot(x, performance, marker="o", linewidth=2.2, label=domain)
    ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1)
    ax.axvspan(0.5, 1.5, color="tab:red", alpha=0.08, label="attack drop")
    if len(stage_labels) > 2:
        ax.axvspan(1.5, len(stage_labels) - 0.5, color="tab:green", alpha=0.08, label="fine-tuning recovery")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=8)
    ax.set_ylabel("Normalized performance (higher is better)")
    ax.set_title("Continuous attack and recovery stages - performance")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_continuous_stage_recovery_performance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    env_frame = frame[frame["attack_type"] == "environment"].copy()
    if not env_frame.empty and "mean_attacked_cell_exposure_ratio" in env_frame:
        for domain, group in env_frame.groupby("eval_domain"):
            group = group.sort_values("global_step")
            ax.plot(
                group["global_step"],
                group["mean_attacked_cell_exposure_ratio"],
                marker="o",
                label=domain,
            )
        ax.set_xlabel("global_step")
        ax.set_ylabel("mean_attacked_cell_exposure_ratio")
        ax.set_title("Exposure to attacked corridor during fine-tuning")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, "fig_recovery_curve_attacked_exposure.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    if not env_frame.empty and "mean_lambda_uncertainty" in env_frame:
        for domain, group in env_frame.groupby("eval_domain"):
            group = group.sort_values("global_step")
            ax.plot(group["global_step"], group["mean_lambda_uncertainty"], marker="o", label=domain)
        ax.set_xlabel("global_step")
        ax.set_ylabel("mean_lambda_uncertainty")
        ax.set_title("Policy uncertainty sensitivity during fine-tuning")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_figure(fig, "fig_recovery_curve_lambda_uncertainty.png")
    plt.close(fig)


def write_output_guide(output_dir: Path) -> None:
    guide = """# Recovery Curve Outputs

## Main CSV

- `recovery_curve.csv`: one row per `global_step x eval_domain x attack_type`.
- `mean_attacked_scalar_cost`: main metric. Lower is better.
- `relative_degradation`: attack cost increase relative to the same checkpoint's no-attack cost.
- `mean_attacked_cell_exposure_ratio`: fraction of path cells inside the attacked corridor. Lower means the policy is avoiding the attacked region.
- `mean_lambda_uncertainty`: policy's uncertainty sensitivity. If this changes, the policy is changing planner behavior, not only sampling noise.
- `mean_weight_*`: average planner weights output by the policy at that checkpoint.

## Figures

Primary copies are under `figures/`. Root-level PNG copies are kept for backward compatibility.

- `fig_continuous_stage_recovery_performance.png`: best narrative figure. It starts with no attack, then applies attack at step 0, then shows fine-tuning recovery. Higher is better.
- `fig_continuous_stage_recovery_cost.png`: same story in raw cost. Lower is better.
- `fig_recovery_curve_attacked_cost.png`: attacked cost vs fine-tuning step. Use this to see whether recovery is monotonic or noisy.
- `fig_recovery_curve_degradation.png`: relative degradation vs step. Use this to see whether fine-tuning reduces the attack penalty.
- `fig_recovery_curve_attacked_exposure.png`: path exposure to the attacked corridor vs step. This tells whether recovery comes from route switching/avoidance.
- `fig_recovery_curve_lambda_uncertainty.png`: policy uncertainty sensitivity vs step. This helps diagnose whether policy outputs are changing.

## How to Read

The clean pattern you want is:

1. `fig_continuous_stage_recovery_performance.png`: performance drops from no-attack to attack step 0, then rises during fine-tuning.
2. `fig_recovery_curve_attacked_cost.png`: attacked cost trends downward.
3. `fig_recovery_curve_attacked_exposure.png`: exposure trends downward.
4. no-attack rows in `recovery_curve.csv`: no-attack cost should not increase too much, otherwise recovery trades away nominal performance.
"""
    (output_dir / "OUTPUT_GUIDE.md").write_text(guide, encoding="utf-8")


def write_diagnostic_report(output_root: Path, recovery: pd.DataFrame) -> None:
    report_path = output_root / "diagnostic_report.md"
    lines = ["# Robustness Diagnostic Report", ""]
    recommendation_path = output_root.parent / "attack_sweep" / "recommended_attack_config.json"
    if recommendation_path.exists():
        recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
        env_attack = recommendation["environment_attack"]
        basis = recommendation.get("selection_basis", {})
        lines.extend(
            [
                "## 1. Current environmental attack strength",
                "",
                f"- Recommended setting: top_fraction={env_attack['attacker_top_fraction']}, "
                f"sharpness={env_attack['attacker_sharpness']}, "
                f"temperature={env_attack['attacker_temperature']}, "
                f"attack_strength={env_attack.get('attack_strength', 1.0)}.",
                f"- Nominal PPO relative degradation: {basis.get('mean_relative_degradation', float('nan')):.4f}.",
                f"- Success_rate: {basis.get('min_success_rate', float('nan')):.4f}.",
                "",
            ]
        )
    else:
        lines.extend(["## 1. Current environmental attack strength", "", "- Attack sweep has not been run.", ""])

    def first_final(domain: str, attack_type: str) -> tuple[pd.Series | None, pd.Series | None]:
        subset = recovery[(recovery["eval_domain"] == domain) & (recovery["attack_type"] == attack_type)]
        if subset.empty:
            return None, None
        subset = subset.sort_values("global_step")
        return subset.iloc[0], subset.iloc[-1]

    in_first, in_final = first_final("in_domain_seed909", "environment")
    no_first, no_final = first_final("in_domain_seed909", "none")
    held_first, held_final = first_final("heldout_seed1919", "environment")

    lines.extend(["## 2. Fine-tuning recovery", ""])
    if in_first is not None and in_final is not None:
        improvement = float(in_first["mean_attacked_scalar_cost"] - in_final["mean_attacked_scalar_cost"])
        rel_improvement = improvement / (abs(float(in_first["mean_attacked_scalar_cost"])) + 1e-8)
        curve_decreases = bool(in_final["mean_attacked_scalar_cost"] < in_first["mean_attacked_scalar_cost"])
        lines.extend(
            [
                f"- Step 0 attacked cost: {float(in_first['mean_attacked_scalar_cost']):.4f}.",
                f"- Final attacked cost: {float(in_final['mean_attacked_scalar_cost']):.4f}.",
                f"- Absolute improvement: {improvement:.4f}.",
                f"- Relative improvement: {rel_improvement:.4f}.",
                f"- Generally decreases by endpoint: {curve_decreases}.",
            ]
        )
    else:
        lines.append("- Recovery curve is not available.")

    lines.extend(["", "## 3. Nominal performance trade-off", ""])
    if no_first is not None and no_final is not None:
        lines.extend(
            [
                f"- Step 0 no-attack cost: {float(no_first['mean_attacked_scalar_cost']):.4f}.",
                f"- Final no-attack cost: {float(no_final['mean_attacked_scalar_cost']):.4f}.",
            ]
        )
    else:
        lines.append("- No-attack recovery rows are not available.")

    lines.extend(["", "## 4. Held-out robustness", ""])
    if in_first is not None and in_final is not None and held_first is not None and held_final is not None:
        in_improvement = float(in_first["mean_attacked_scalar_cost"] - in_final["mean_attacked_scalar_cost"])
        held_improvement = float(held_first["mean_attacked_scalar_cost"] - held_final["mean_attacked_scalar_cost"])
        lines.extend(
            [
                f"- In-domain attacked improvement: {in_improvement:.4f}.",
                f"- Held-out attacked improvement: {held_improvement:.4f}.",
            ]
        )
    else:
        lines.append("- Held-out recovery rows are not available.")

    lines.extend(
        [
            "",
            "## 5. Policy output shift",
            "",
            "- Run analyze_policy_outputs.py to fill policy output comparisons.",
            "",
            "## 6. Observation attack impact",
            "",
            "- Observation stress testing is optional; if it remains weak, likely causes include weak noise, fixed-weight-like policy outputs, low terrain-observation reliance, or planner robustness to small context perturbations.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved diagnostic report to {report_path}")


def main() -> None:
    args = parse_args()
    if args.eval_interval <= 0 or args.finetune_timesteps <= 0:
        raise ValueError("--eval-interval and --finetune-timesteps must be positive")

    require_existing_file(args.base_config, "base config")
    require_existing_file(args.checkpoint, "checkpoint")
    require_existing_file(args.attack_config, "attack config")
    require_existing_file(args.observation_attack_config, "observation attack config")

    output_dir = Path(args.output_dir)
    if args.clean_output:
        clean_output_dir(output_dir, args.dry_run)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    base_args = load_base_args(args.base_config)
    env_attack = load_environment_attack(args.attack_config)
    observation_attack = load_attack_config(args.observation_attack_config)

    seed = int(args.seed if args.seed is not None else config_value(base_args, "seed", 0))
    map_size = int(config_value(base_args, "map-size", 48))
    scenario = str(config_value(base_args, "scenario", "lunar_rover"))
    min_distance_ratio = float(config_value(base_args, "min-start-goal-distance-ratio", 0.55))
    map_pool_size = int(
        args.map_pool_size
        if args.map_pool_size is not None
        else config_value(base_args, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE)
    )
    write_run_metadata(
        output_dir,
        args,
        base_args,
        env_attack,
        observation_attack,
        seed,
        map_size,
        scenario,
        map_pool_size,
    )

    episodes_by_domain = {
        f"in_domain_seed{args.in_domain_seed}": (
            int(args.in_domain_seed),
            generate_episodes(
                args.num_eval_episodes,
                seed + 222,
                map_size,
                scenario,
                args.in_domain_seed,
                map_pool_size,
                min_start_goal_distance_ratio=min_distance_ratio,
            ),
        ),
        f"heldout_seed{args.heldout_seed}": (
            int(args.heldout_seed),
            generate_episodes(
                args.num_eval_episodes,
                seed + 222,
                map_size,
                scenario,
                args.heldout_seed,
                map_pool_size,
                min_start_goal_distance_ratio=min_distance_ratio,
            ),
        ),
    }

    initial_checkpoint = Path(args.checkpoint)
    current_checkpoint = checkpoints_dir / "checkpoint_step_0.pt"
    if not args.dry_run:
        shutil.copy2(initial_checkpoint, current_checkpoint)
    rows = evaluate_checkpoint(
        current_checkpoint if current_checkpoint.exists() else initial_checkpoint,
        0,
        episodes_by_domain,
        map_size,
        env_attack,
        observation_attack,
        seed,
    )

    cumulative_step = 0
    chunk_index = 0
    while cumulative_step < args.finetune_timesteps:
        chunk_index += 1
        chunk_timesteps = min(args.eval_interval, args.finetune_timesteps - cumulative_step)
        command, chunk_final = build_train_command(
            args.python,
            base_args,
            current_checkpoint if current_checkpoint.exists() else initial_checkpoint,
            env_attack,
            observation_attack,
            output_dir,
            chunk_index,
            chunk_timesteps,
            seed,
        )
        print("Running:", " ".join(command))
        if args.dry_run:
            cumulative_step += chunk_timesteps
            continue
        completed = subprocess.run(command, cwd=Path(__file__).resolve().parent, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"training chunk {chunk_index} failed with code {completed.returncode}")
        actual_chunk_step = checkpoint_step(chunk_final)
        cumulative_step += actual_chunk_step if actual_chunk_step > 0 else chunk_timesteps
        current_checkpoint = checkpoints_dir / f"checkpoint_step_{cumulative_step}.pt"
        shutil.copy2(chunk_final, current_checkpoint)
        rows.extend(
            evaluate_checkpoint(
                current_checkpoint,
                cumulative_step,
                episodes_by_domain,
                map_size,
                env_attack,
                observation_attack,
                seed,
            )
        )

    if args.dry_run:
        print("Dry run complete; no recovery outputs written.")
        return

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "recovery_curve.csv"
    frame.to_csv(csv_path, index=False)
    plot_recovery(frame, output_dir)
    write_output_guide(output_dir)
    write_diagnostic_report(output_dir, frame)

    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved recovery curve CSV to {csv_path}")
    print(f"Saved attacked-cost figure to {output_dir / 'fig_recovery_curve_attacked_cost.png'}")
    print(f"Saved degradation figure to {output_dir / 'fig_recovery_curve_degradation.png'}")
    print(f"Saved output guide to {output_dir / 'OUTPUT_GUIDE.md'}")
    print(f"Saved checkpoints to {checkpoints_dir}")


if __name__ == "__main__":
    main()
