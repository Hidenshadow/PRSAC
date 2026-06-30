"""Robustness evaluation for PPO planner-parameter policies.

This script evaluates learned policies on the same episode set under four
attack conditions: no attack, observation attack, environmental attack, and
combined attack.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

from gymnasium import spaces
import numpy as np
import pandas as pd
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
from maps.map_generator import SCENARIO_NAMES
from utils.metrics import (
    ATTACKER_RESPONSE_MODES,
    DEFAULT_ATTACKER_RESPONSE,
    DEFAULT_ATTACKER_SHARPNESS,
    DEFAULT_ATTACKER_TEMPERATURE,
    DEFAULT_ATTACKER_TOP_FRACTION,
    DEFAULT_FIXED_MAP_SEED,
    DEFAULT_MAP_SEED_POOL_SIZE,
    MAP_SAMPLING_MODES,
    OBSERVATION_MODES,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    compute_observation,
    make_curriculum_planning_episode,
    plan_with_weights,
)


@dataclass
class LoadedPolicy:
    name: str
    checkpoint_path: str
    model_type: str
    model: Any
    model_config: dict[str, Any]
    observation_mode: str
    action_mode: str
    action_gain: float
    max_uncertainty_lambda: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--checkpoints",
        action="append",
        nargs="+",
        default=None,
        help="Policy checkpoints as name=path pairs.",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--map-size", type=int, default=None)
    parser.add_argument("--scenario", choices=SCENARIO_NAMES, default=None)
    parser.add_argument("--map-sampling-mode", choices=MAP_SAMPLING_MODES, default=None)
    parser.add_argument("--fixed-map-seed", type=int, default=None)
    parser.add_argument("--map-seed-pool-size", type=int, default=None)
    parser.add_argument("--eval-domain", type=str, default=None)
    parser.add_argument("--observation-mode", choices=("auto", *OBSERVATION_MODES), default=None)
    parser.add_argument("--action-mode", choices=("auto", "direct", "preference_delta"), default=None)
    parser.add_argument("--action-gain", type=float, default=None)
    parser.add_argument("--max-uncertainty-lambda", type=float, default=None)
    parser.add_argument("--attacker-response", choices=ATTACKER_RESPONSE_MODES, default=None)
    parser.add_argument("--attacker-temperature", type=float, default=None)
    parser.add_argument("--attacker-top-fraction", type=float, default=None)
    parser.add_argument("--attacker-sharpness", type=float, default=None)
    parser.add_argument(
        "--observation-attack-config",
        type=str,
        default=None,
        help="JSON text or path for observation attack config.",
    )
    parser.add_argument(
        "--environment-attack-config",
        type=str,
        default=None,
        help="JSON text or path for environmental attack config.",
    )
    return parser.parse_args()


def load_config(path_text: str | None) -> dict[str, Any]:
    if not path_text:
        return {}
    path = Path(path_text)
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("args", data))


def arg_value(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any) -> Any:
    attr = key.replace("-", "_")
    value = getattr(args, attr, None)
    if value is not None:
        return value
    return config.get(key, config.get(attr, default))


def parse_checkpoints(
    cli_specs: list[list[str]] | None,
    config: dict[str, Any],
) -> dict[str, str]:
    if cli_specs:
        specs = [item for group in cli_specs for item in group]
        checkpoints: dict[str, str] = {}
        for spec in specs:
            if "=" not in spec:
                raise ValueError("--checkpoints entries must be name=path")
            name, path = spec.split("=", 1)
            checkpoints[name.strip()] = path.strip()
        return checkpoints

    raw = config.get("checkpoints", {})
    if isinstance(raw, dict):
        return {str(name): str(path) for name, path in raw.items()}
    if isinstance(raw, list):
        checkpoints: dict[str, str] = {}
        for spec in raw:
            text = str(spec)
            if "=" not in text:
                raise ValueError("checkpoint list entries must be name=path")
            name, path = text.split("=", 1)
            checkpoints[name.strip()] = path.strip()
        return checkpoints
    raise ValueError("checkpoints must be provided by config or --checkpoints")


def load_policies(
    checkpoint_specs: dict[str, str],
    requested_observation_mode: str,
    requested_action_mode: str,
    requested_action_gain: float | None,
    requested_max_lambda: float | None,
) -> list[LoadedPolicy]:
    policies: list[LoadedPolicy] = []
    for name, path in checkpoint_specs.items():
        model_type, model, model_config = load_model(path, "auto")
        observation_mode = resolve_observation_mode(requested_observation_mode, model_config)
        action_mode, action_gain, max_uncertainty_lambda = resolve_action_config(
            requested_action_mode,
            requested_action_gain,
            requested_max_lambda,
            model_config,
        )
        policies.append(
            LoadedPolicy(
                name=name,
                checkpoint_path=path,
                model_type=model_type,
                model=model,
                model_config=model_config,
                observation_mode=observation_mode,
                action_mode=action_mode,
                action_gain=action_gain,
                max_uncertainty_lambda=max_uncertainty_lambda,
            )
        )
    return policies


def policy_planner_config(
    policy: LoadedPolicy,
    episode,
    map_size: int,
    observation_attack: dict[str, Any] | None,
    obs_seed: int,
) -> dict[str, Any]:
    obs = compute_observation(
        episode,
        map_size,
        observation_mode=policy.observation_mode,
        max_uncertainty_lambda=policy.max_uncertainty_lambda,
    )
    if attack_enabled(observation_attack):
        obs_space = spaces.Box(low=0.0, high=1.0, shape=obs.shape, dtype=np.float32)
        obs = apply_observation_attack(
            obs,
            observation_attack,
            np.random.default_rng(obs_seed),
            observation_space=obs_space,
        )

    action = predict_action(policy.model_type, policy.model, obs)
    weights = action_to_planning_weights(
        episode,
        action,
        action_mode=policy.action_mode,
        action_gain=policy.action_gain,
    )
    lambda_uncertainty = action_to_uncertainty_lambda(
        action,
        max_uncertainty_lambda=policy.max_uncertainty_lambda,
    )

    return {"weights": weights, "lambda_uncertainty": lambda_uncertainty}


def attack_eval_cost(result: dict[str, Any], environment_attack: dict[str, Any] | None) -> float:
    if not attack_enabled(environment_attack):
        return float(result.get("scalar_cost", np.nan))
    attack_type = str(environment_attack.get("type", "env_zscore_topk"))
    if attack_type == "env_zscore_topk":
        return float(result.get("soft_attacked_scalar_cost", result.get("attacked_scalar_cost", np.nan)))
    if attack_type in {"env_layer_noise", "env_layer_bias", "env_path_corridor_attack", "env_belief_mismatch"}:
        return float(result.get("scalar_cost", np.nan))
    return float(result.get("attacked_scalar_cost", result.get("scalar_cost", np.nan)))


def attack_scenarios(
    observation_attack: dict[str, Any],
    environment_attack: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "attack_type": "none",
            "observation_attack": None,
            "environment_attack": None,
        },
        {
            "attack_type": "observation",
            "observation_attack": observation_attack,
            "environment_attack": None,
        },
        {
            "attack_type": "environment",
            "observation_attack": None,
            "environment_attack": environment_attack,
        },
        {
            "attack_type": "combined",
            "observation_attack": observation_attack,
            "environment_attack": environment_attack,
        },
    ]


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    nominal = frame["nominal_scalar_cost"].to_numpy(dtype=np.float64)
    attacked = frame["eval_attacked_cost"].to_numpy(dtype=np.float64)
    success = frame["success"].to_numpy(dtype=np.float64)
    path_lengths = frame["path_length"].to_numpy(dtype=np.float64)
    planning_times = frame["planning_time"].to_numpy(dtype=np.float64)
    rewards = frame["reward"].to_numpy(dtype=np.float64)
    exposure = frame["attacked_cell_exposure_ratio"].to_numpy(dtype=np.float64) if "attacked_cell_exposure_ratio" in frame else np.zeros(len(frame))
    mean_nominal = float(np.nanmean(nominal))
    mean_attacked = float(np.nanmean(attacked))
    absolute_degradation = mean_attacked - mean_nominal
    relative_degradation = absolute_degradation / (abs(mean_nominal) + 1e-8)
    p90 = float(np.nanquantile(attacked, 0.90)) if len(attacked) else np.nan

    first = rows[0]
    return {
        "policy_name": first["policy_name"],
        "checkpoint_path": first["checkpoint_path"],
        "eval_domain": first["eval_domain"],
        "map_pool_seed": first["map_pool_seed"],
        "map_pool_size": first["map_pool_size"],
        "attack_type": first["attack_type"],
        "observation_attack_type": first["observation_attack_type"],
        "environment_attack_type": first["environment_attack_type"],
        "num_episodes": int(len(rows)),
        "mean_nominal_scalar_cost": mean_nominal,
        "std_nominal_scalar_cost": float(np.nanstd(nominal)),
        "mean_attacked_scalar_cost": mean_attacked,
        "std_attacked_scalar_cost": float(np.nanstd(attacked)),
        "mean_reward": float(np.nanmean(rewards)),
        "success_rate": float(np.nanmean(success)),
        "failure_rate": float(1.0 - np.nanmean(success)),
        "mean_path_length": float(np.nanmean(path_lengths)),
        "mean_planning_time": float(np.nanmean(planning_times)),
        "cvar90_attacked_cost": float(np.nanmean(attacked[attacked >= p90])) if np.isfinite(p90) else np.nan,
        "worst10_mean_attacked_cost": float(np.nanmean(attacked[attacked >= p90])) if np.isfinite(p90) else np.nan,
        "mean_attacked_cell_exposure_ratio": float(np.nanmean(exposure)),
        "absolute_degradation": float(absolute_degradation),
        "relative_degradation": float(relative_degradation),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    num_episodes = int(arg_value(args, config, "num-episodes", 500))
    seed = int(arg_value(args, config, "seed", 222))
    map_size = int(arg_value(args, config, "map-size", 48))
    scenario = str(arg_value(args, config, "scenario", "lunar_rover"))
    map_sampling_mode = str(arg_value(args, config, "map-sampling-mode", "map_seed_pool"))
    fixed_map_seed = int(arg_value(args, config, "fixed-map-seed", DEFAULT_FIXED_MAP_SEED))
    map_seed_pool_size = int(arg_value(args, config, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE))
    eval_domain = str(arg_value(args, config, "eval-domain", f"seed{fixed_map_seed}"))
    output = Path(arg_value(args, config, "output", "runs/robustness/results/robustness_summary.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)

    requested_observation_mode = str(arg_value(args, config, "observation-mode", "auto"))
    requested_action_mode = str(arg_value(args, config, "action-mode", "auto"))
    requested_action_gain = arg_value(args, config, "action-gain", None)
    requested_action_gain = None if requested_action_gain is None else float(requested_action_gain)
    requested_max_lambda = arg_value(args, config, "max-uncertainty-lambda", None)
    requested_max_lambda = None if requested_max_lambda is None else float(requested_max_lambda)

    observation_attack_raw = arg_value(args, config, "observation-attack-config", None)
    environment_attack_raw = arg_value(args, config, "environment-attack-config", None)
    observation_attack = (
        load_attack_config(observation_attack_raw)
        if observation_attack_raw is not None
        else dict(config.get("observation_attack", {}))
    )
    environment_attack = (
        load_attack_config(environment_attack_raw)
        if environment_attack_raw is not None
        else dict(config.get("environment_attack", {}))
    )
    if args.attacker_response is not None:
        environment_attack["attacker_response"] = args.attacker_response
    if args.attacker_temperature is not None:
        environment_attack["attacker_temperature"] = args.attacker_temperature
    if args.attacker_top_fraction is not None:
        environment_attack["attacker_top_fraction"] = args.attacker_top_fraction
    if args.attacker_sharpness is not None:
        environment_attack["attacker_sharpness"] = args.attacker_sharpness

    environment_attack.setdefault("attacker_response", DEFAULT_ATTACKER_RESPONSE)
    environment_attack.setdefault("attacker_temperature", DEFAULT_ATTACKER_TEMPERATURE)
    environment_attack.setdefault("attacker_top_fraction", DEFAULT_ATTACKER_TOP_FRACTION)
    environment_attack.setdefault("attacker_sharpness", DEFAULT_ATTACKER_SHARPNESS)

    checkpoint_specs = parse_checkpoints(args.checkpoints, config)
    policies = load_policies(
        checkpoint_specs,
        requested_observation_mode=requested_observation_mode,
        requested_action_mode=requested_action_mode,
        requested_action_gain=requested_action_gain,
        requested_max_lambda=requested_max_lambda,
    )

    rng = np.random.default_rng(seed)
    map_cache: dict[Any, Any] = {}
    episodes = [
        make_curriculum_planning_episode(
            map_size=map_size,
            rng=rng,
            allow_diagonal=True,
            scenario=scenario,
            map_sampling_mode=map_sampling_mode,
            fixed_map_seed=fixed_map_seed,
            map_seed_pool_size=map_seed_pool_size,
            map_cache=map_cache,
        )
        for _ in tqdm(range(num_episodes), desc="Generating shared episodes")
    ]

    detailed_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    scenarios = attack_scenarios(observation_attack, environment_attack)

    for scenario_index, attack in enumerate(scenarios):
        env_attack = attack["environment_attack"]
        obs_attack = attack["observation_attack"]
        env_attack_type = str(env_attack.get("type", "none")) if attack_enabled(env_attack) else "none"
        obs_attack_type = str(obs_attack.get("type", "none")) if attack_enabled(obs_attack) else "none"

        for policy in policies:
            group_rows: list[dict[str, Any]] = []
            for episode_index, episode in enumerate(
                tqdm(
                    episodes,
                    desc=f"{policy.name}:{attack['attack_type']}",
                    leave=False,
                )
            ):
                clean_config = policy_planner_config(
                    policy,
                    episode,
                    map_size,
                    observation_attack=None,
                    obs_seed=seed,
                )
                clean_result = plan_with_weights(
                    episode,
                    clean_config["weights"],
                    lambda_uncertainty=float(clean_config["lambda_uncertainty"]),
                    allow_diagonal=True,
                )

                eval_episode = episode
                if attack_enabled(env_attack):
                    env_rng = np.random.default_rng(seed + 500_000 * (scenario_index + 1) + episode_index)
                    eval_episode = apply_environment_attack_to_episode(episode, env_attack, env_rng)

                obs_seed = seed + 100_000 * (scenario_index + 1) + episode_index
                start_time = time.perf_counter()
                attacked_config = policy_planner_config(
                    policy,
                    eval_episode,
                    map_size,
                    observation_attack=obs_attack,
                    obs_seed=obs_seed,
                )
                result = plan_with_weights(
                    eval_episode,
                    attacked_config["weights"],
                    lambda_uncertainty=float(attacked_config["lambda_uncertainty"]),
                    allow_diagonal=True,
                    attacker_temperature=float(env_attack.get("attacker_temperature", DEFAULT_ATTACKER_TEMPERATURE))
                    if attack_enabled(env_attack)
                    else DEFAULT_ATTACKER_TEMPERATURE,
                    attacker_response=str(env_attack.get("attacker_response", DEFAULT_ATTACKER_RESPONSE))
                    if attack_enabled(env_attack)
                    else DEFAULT_ATTACKER_RESPONSE,
                    attacker_top_fraction=float(env_attack.get("attacker_top_fraction", DEFAULT_ATTACKER_TOP_FRACTION))
                    if attack_enabled(env_attack)
                    else DEFAULT_ATTACKER_TOP_FRACTION,
                    attacker_sharpness=float(env_attack.get("attacker_sharpness", DEFAULT_ATTACKER_SHARPNESS))
                    if attack_enabled(env_attack)
                    else DEFAULT_ATTACKER_SHARPNESS,
                )
                planning_time = time.perf_counter() - start_time
                eval_cost = attack_eval_cost(result, env_attack)
                row = {
                    "policy_name": policy.name,
                    "checkpoint_path": policy.checkpoint_path,
                    "eval_domain": eval_domain,
                    "map_pool_seed": fixed_map_seed,
                    "map_pool_size": map_seed_pool_size,
                    "attack_type": attack["attack_type"],
                    "observation_attack_type": obs_attack_type,
                    "environment_attack_type": env_attack_type,
                    "episode": episode_index,
                    "scenario": eval_episode.scenario,
                    "mission_regime": eval_episode.mission_regime,
                    "nominal_scalar_cost": float(clean_result.get("scalar_cost", np.nan)),
                    "eval_scalar_cost": float(result.get("scalar_cost", np.nan)),
                    "eval_hard_attacked_scalar_cost": float(result.get("attacked_scalar_cost", np.nan)),
                    "eval_soft_attacked_scalar_cost": float(result.get("soft_attacked_scalar_cost", np.nan)),
                    "eval_attacked_cost": eval_cost,
                    "reward": -eval_cost,
                    "success": 1.0 if bool(result.get("success", False)) else 0.0,
                    "path_length": float(result.get("path_length", np.nan)),
                    "attacked_cell_exposure": float(result.get("attacked_cell_exposure", 0.0)),
                    "attacked_cell_exposure_ratio": float(result.get("attacked_cell_exposure_ratio", 0.0)),
                    "attacked_corridor_cells": float(result.get("attacked_corridor_cells", 0.0)),
                    "hazard_exposure": float(result.get("hazard_exposure", np.nan)),
                    "uncertainty_exposure": float(result.get("uncertainty_exposure", np.nan)),
                    "planning_time": planning_time,
                    "lambda_uncertainty": float(result.get("lambda_uncertainty", np.nan)),
                }
                group_rows.append(row)
                detailed_rows.append(row)

            summary_rows.append(summarize_group(group_rows))

    summary = pd.DataFrame(summary_rows)
    detailed = pd.DataFrame(detailed_rows)
    summary.to_csv(output, index=False)
    json_path = output.with_suffix(".json")
    json_path.write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")
    detailed_path = output.with_name(f"{output.stem}_detailed.csv")
    detailed.to_csv(detailed_path, index=False)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved robustness summary to {output}")
    print(f"Saved robustness summary JSON to {json_path}")
    print(f"Saved detailed robustness rows to {detailed_path}")


if __name__ == "__main__":
    main()
