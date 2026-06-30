#!/usr/bin/env python
"""Evaluate non-learning planner baselines on the shock-recovery protocol.

The baselines in this script do not train a policy. They choose weighted-A*
planner parameters from deterministic rules, validation-selected presets, or a
model-based minimax rule over known attack variants. The output is normalized
onto the same performance-index scale used by the RL shock-recovery figures.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.attack_wrappers import apply_environment_attack_to_episode, attack_enabled  # noqa: E402
from run_attack_recovery_finetune import config_value, generate_episodes  # noqa: E402
from run_lunar_viper_staged_recovery import generate_real_episodes  # noqa: E402
from run_shock_recovery_experiment import (  # noqa: E402
    benchmark_attack_variant,
    component_attack_variants,
    local_attack_scales,
    scale_attack_config,
)
from utils.metrics import (  # noqa: E402
    DEFAULT_MAP_SEED_POOL_SIZE,
    OBJECTIVE_NAMES,
    candidate_planner_configs,
    plan_with_weights,
    safety_shield_planner_config,
)


LEVELS = ("level1", "level2", "level3")
DIFFICULTIES = ("easy", "medium", "hard")
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "runs" / "rl_baselines" / "ppo"
DEFAULT_RL_SUMMARY = PROJECT_ROOT / "runs" / "rl_baselines" / "paper_story_5seeds" / "rl_baseline_performance_story.csv"
DEFAULT_GAME_SUMMARY = (
    PROJECT_ROOT / "runs" / "game_recovery_protocol_analysis" / "paper_story_1seed" / "game_recovery_performance_story.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "nonlearning_planner_baselines"

FOCUS_METHODS = (
    "fixed",
    "heuristic",
    "safe_rover",
    "power_comms",
    "rover_guard",
    "mission_safe_blend",
    "mission_power_comms_blend",
    "emergency_uncertainty_rule",
)
PLOT_METHODS = (
    "fixed",
    "heuristic",
    "rover_guard",
    "emergency_uncertainty_rule",
    "risk_inflated_astar",
    "belief_sample_cvar",
    "validation_best_static",
    "model_minimax",
)
LABELS = {
    "ppo": "PPO",
    "sac": "SAC",
    "game_ppo": "Game-PPO",
    "fixed": "Fixed",
    "heuristic": "Mission",
    "safe_rover": "Safe",
    "power_comms": "Power/Comms",
    "rover_guard": "Guard",
    "mission_safe_blend": "Mission+Safe",
    "mission_power_comms_blend": "Mission+Power",
    "emergency_uncertainty_rule": "Emergency Rule",
    "risk_inflated_astar": "Risk-Inflated A*",
    "belief_sample_cvar": "Belief-CVaR A*",
    "validation_best_static": "Val-Best Rule",
    "model_minimax": "Model Minimax",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--source-seed-count", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--difficulties", nargs="+", choices=DIFFICULTIES, default=list(DIFFICULTIES))
    parser.add_argument("--num-eval-episodes", type=int, default=300)
    parser.add_argument("--validation-episodes", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rl-summary", type=Path, default=DEFAULT_RL_SUMMARY)
    parser.add_argument("--game-summary", type=Path, default=DEFAULT_GAME_SUMMARY)
    parser.add_argument("--include-all-presets", action="store_true")
    parser.add_argument("--disable-validation-best", action="store_true")
    parser.add_argument("--disable-model-minimax", action="store_true")
    parser.add_argument("--minimax-variant-mode", choices=("component", "scale", "scale_component"), default="component")
    parser.add_argument("--minimax-mixture-size", type=int, default=5)
    parser.add_argument("--minimax-jitter", type=float, default=0.18)
    parser.add_argument("--disable-risk-inflated", action="store_true")
    parser.add_argument("--disable-belief-cvar", action="store_true")
    parser.add_argument("--risk-inflation-scale", type=float, default=0.42)
    parser.add_argument("--risk-inflation-radius", type=int, default=2)
    parser.add_argument("--belief-cvar-samples", type=int, default=5)
    parser.add_argument("--belief-cvar-alpha", type=float, default=0.35)
    parser.add_argument("--belief-cvar-noise-scale", type=float, default=0.55)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save-episode-details", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values: list[float], default: float = float("nan")) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float(default)
    return float(np.mean(array))


def safe_std(values: list[float], default: float = float("nan")) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float(default)
    return float(np.std(array))


def source_run_dir(source_root: Path, level: str, difficulty: str, seed: int, seed_count: int) -> Path:
    root = resolve(source_root)
    nested = root / f"{level}_{difficulty}_shock_recovery_{seed_count}seeds" / f"seed{seed}"
    if nested.exists():
        return nested
    fallback = root / f"{level}_{difficulty}_shock_recovery_1seed" / f"seed{seed}"
    if fallback.exists():
        return fallback
    return nested


def source_run_dirs(args: argparse.Namespace) -> list[Path]:
    dirs: list[Path] = []
    for level in args.levels:
        for difficulty in args.difficulties:
            for seed in args.seeds:
                run_dir = source_run_dir(args.source_root, level, difficulty, int(seed), int(args.source_seed_count))
                if run_dir.exists():
                    dirs.append(run_dir)
                else:
                    print(f"WARNING: missing source run: {run_dir}", flush=True)
    return dirs


def run_label_from_dir(run_dir: Path) -> tuple[str, str, int]:
    name = run_dir.parent.name
    parts = name.split("_")
    if len(parts) < 2:
        raise ValueError(f"cannot infer level/difficulty from {run_dir}")
    seed_text = run_dir.name.replace("seed", "")
    return parts[0], parts[1], int(seed_text)


def reference_clean_costs(run_dir: Path) -> dict[str, float]:
    curve_path = run_dir / "shock_recovery_curve.csv"
    if not curve_path.exists():
        return {}
    frame = pd.read_csv(curve_path)
    required = {"phase", "attack_type", "eval_domain", "mean_attacked_scalar_cost"}
    if not required.issubset(frame.columns):
        return {}
    clean = frame[(frame["phase"] == "shock") & (frame["attack_type"] == "none")]
    return {
        str(row.eval_domain): float(row.mean_attacked_scalar_cost)
        for row in clean.itertuples(index=False)
        if math.isfinite(float(row.mean_attacked_scalar_cost))
    }


def synthetic_episodes_from_run(run_config: dict[str, Any], count: int) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    command_args = dict(run_config["command_args"])
    base_args = dict(run_config["base_config_args"])
    seed = int(command_args.get("seed", base_args.get("seed", 0)))
    map_size = int(config_value(base_args, "map-size", 48))
    scenario = str(config_value(base_args, "scenario", "lunar_rover_corridor"))
    map_pool_size = int(config_value(base_args, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE))
    min_distance_ratio = float(config_value(base_args, "min-start-goal-distance-ratio", 0.55))
    in_domain_seed = int(command_args.get("in_domain_seed", 909))
    heldout_seed = int(command_args.get("heldout_seed", 1919))
    num_eval = min(int(count), int(command_args.get("num_eval_episodes", count)))
    return map_size, {
        f"in_domain_seed{in_domain_seed}": (
            in_domain_seed,
            generate_episodes(
                num_eval,
                seed + 222,
                map_size,
                scenario,
                in_domain_seed,
                map_pool_size,
                min_start_goal_distance_ratio=min_distance_ratio,
            ),
        ),
        f"heldout_seed{heldout_seed}": (
            heldout_seed,
            generate_episodes(
                num_eval,
                seed + 222,
                map_size,
                scenario,
                heldout_seed,
                map_pool_size,
                min_start_goal_distance_ratio=min_distance_ratio,
            ),
        ),
    }


def real_episodes_from_run(run_dir: Path, run_config: dict[str, Any], count: int) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    level_config = dict(run_config["level_config"])
    command_args = dict(run_config["command_args"])
    seed = int(command_args.get("seed", 0))
    layers_path = resolve(Path(level_config["map_source"]))
    raw = np.load(layers_path)
    map_size = int(raw["layer_distance"].shape[0])
    raw.close()
    splits = {key: resolve(Path(value)) for key, value in dict(run_config.get("splits", {})).items()}
    if not splits:
        splits = {
            "train": run_dir / "splits" / "train_tasks.json",
            "heldout": run_dir / "splits" / "heldout_tasks.json",
            "validation": run_dir / "splits" / "validation_tasks.json",
        }
    scenario = str(level_config.get("scenario", "real_lunar_viper"))
    mission_profile = str(level_config.get("mission_profile_scenario", "lunar_polar_shadow"))
    num_eval = min(int(count), int(command_args.get("num_eval_episodes", count)))
    return map_size, {
        "train_tasks": (
            seed,
            generate_real_episodes(layers_path, splits["train"], scenario, mission_profile, seed + 11_000, num_eval),
        ),
        "heldout_tasks": (
            seed + 1,
            generate_real_episodes(layers_path, splits["heldout"], scenario, mission_profile, seed + 22_000, num_eval),
        ),
    }


def eval_episodes_from_run(run_dir: Path, run_config: dict[str, Any], count: int) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    if "map_source" in dict(run_config["level_config"]):
        return real_episodes_from_run(run_dir, run_config, count)
    return synthetic_episodes_from_run(run_config, count)


def validation_episodes_from_run(
    run_dir: Path,
    run_config: dict[str, Any],
    count: int,
) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    level_config = dict(run_config["level_config"])
    command_args = dict(run_config["command_args"])
    base_args = dict(run_config["base_config_args"])
    seed = int(command_args.get("seed", base_args.get("seed", 0)))
    if "map_source" in level_config:
        layers_path = resolve(Path(level_config["map_source"]))
        raw = np.load(layers_path)
        map_size = int(raw["layer_distance"].shape[0])
        raw.close()
        splits = {key: resolve(Path(value)) for key, value in dict(run_config.get("splits", {})).items()}
        validation_path = splits.get("validation", run_dir / "splits" / "validation_tasks.json")
        scenario = str(level_config.get("scenario", "real_lunar_viper"))
        mission_profile = str(level_config.get("mission_profile_scenario", "lunar_polar_shadow"))
        return map_size, {
            "validation_tasks": (
                seed + 2,
                generate_real_episodes(layers_path, validation_path, scenario, mission_profile, seed + 33_000, int(count)),
            )
        }

    map_size = int(config_value(base_args, "map-size", 48))
    scenario = str(config_value(base_args, "scenario", "lunar_rover_corridor"))
    map_pool_size = int(config_value(base_args, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE))
    min_distance_ratio = float(config_value(base_args, "min-start-goal-distance-ratio", 0.55))
    fixed_seed = int(config_value(base_args, "fixed-map-seed", 909))
    validation_seed = fixed_seed + 101
    return map_size, {
        f"validation_seed{validation_seed}": (
            validation_seed,
            generate_episodes(
                int(count),
                seed + 333 + validation_seed,
                map_size,
                scenario,
                validation_seed,
                map_pool_size,
                min_start_goal_distance_ratio=min_distance_ratio,
            ),
        )
    }


def _confidence_layers_for_episode(episode: Any) -> dict[str, np.ndarray]:
    if episode.confidence_layers:
        return {name: np.asarray(value, dtype=np.float32) for name, value in episode.confidence_layers.items()}
    return {
        name: np.clip(1.0 - np.asarray(episode.costmap.uncertainty_layers[name], dtype=np.float32), 0.0, 1.0)
        for name in OBJECTIVE_NAMES
    }


def _neighbor_pressure(mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    radius = max(int(radius), 0)
    if radius <= 0:
        return mask.astype(np.float32)
    pressure = np.zeros(mask.shape, dtype=np.float32)
    rows, cols = mask.shape
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            dist = math.hypot(dr, dc)
            if dist > radius + 1e-6:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(rows, rows - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(cols, cols - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(rows, rows + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(cols, cols + dc)
            if src_r0 >= src_r1 or src_c0 >= src_c1:
                continue
            weight = float((radius + 1.0 - dist) / (radius + 1.0))
            shifted = np.zeros_like(pressure)
            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1].astype(np.float32)
            pressure = np.maximum(pressure, weight * shifted)
    return np.clip(pressure, 0.0, 1.0).astype(np.float32)


def risk_inflated_planner_config(
    episode: Any,
    max_lambda: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    shielded = safety_shield_planner_config(
        episode,
        episode.mission_priority,
        lambda_uncertainty=0.78 * float(max_lambda),
        max_uncertainty_lambda=float(max_lambda),
        enabled=True,
        strength=0.70,
    )
    return {
        "weights": np.asarray(shielded["weights"], dtype=np.float32),
        "lambda_uncertainty": float(shielded["lambda_uncertainty"]),
        "robust_method": "risk_inflated_astar",
        "risk_inflation_scale": float(args.risk_inflation_scale),
        "risk_inflation_radius": int(args.risk_inflation_radius),
    }


def apply_risk_inflation_to_episode(
    episode: Any,
    scale: float,
    radius: int,
) -> Any:
    costmap = episode.costmap
    confidence_layers = _confidence_layers_for_episode(episode)
    obstacle_pressure = _neighbor_pressure(costmap.obstacle_mask, int(radius))
    gains = {
        "distance": 0.16,
        "energy": 0.92,
        "hazard": 1.18,
        "communication": 0.92,
        "illumination": 0.92,
    }
    layers: dict[str, np.ndarray] = {}
    uncertainty_layers: dict[str, np.ndarray] = {}
    for name in OBJECTIVE_NAMES:
        layer = np.asarray(costmap.layers[name], dtype=np.float32)
        uncertainty = np.asarray(costmap.uncertainty_layers[name], dtype=np.float32)
        confidence = np.asarray(confidence_layers.get(name, 1.0 - uncertainty), dtype=np.float32)
        low_confidence = np.clip(1.0 - confidence, 0.0, 1.0)
        belief_risk = np.clip(0.55 * uncertainty + 0.35 * low_confidence + 0.10 * obstacle_pressure, 0.0, 1.0)
        layers[name] = np.clip(layer + float(scale) * gains[name] * belief_risk, 0.0, 1.0).astype(np.float32)
        uncertainty_layers[name] = np.clip(np.maximum(uncertainty, belief_risk), 0.0, 1.0).astype(np.float32)

    inflated_costmap = replace(costmap, layers=layers, uncertainty_layers=uncertainty_layers)
    return replace(episode, costmap=inflated_costmap)


def belief_sample_cvar_config(
    max_lambda: float,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "weights": np.full(len(OBJECTIVE_NAMES), 1.0 / len(OBJECTIVE_NAMES), dtype=np.float32),
        "lambda_uncertainty": 0.0,
        "robust_method": "belief_sample_cvar",
        "selection_seed": int(seed),
        "max_uncertainty_lambda": float(max_lambda),
        "cvar_samples": int(args.belief_cvar_samples),
        "cvar_alpha": float(args.belief_cvar_alpha),
        "cvar_noise_scale": float(args.belief_cvar_noise_scale),
        "include_all_presets": bool(args.include_all_presets),
    }


def sample_plausible_belief_episode(
    episode: Any,
    rng: np.random.Generator,
    noise_scale: float,
) -> Any:
    """Sample a plausible map from the observed belief and uncertainty only."""

    costmap = episode.costmap
    confidence_layers = _confidence_layers_for_episode(episode)
    layers: dict[str, np.ndarray] = {}
    uncertainty_layers: dict[str, np.ndarray] = {}
    for name in OBJECTIVE_NAMES:
        layer = np.asarray(costmap.layers[name], dtype=np.float32)
        uncertainty = np.asarray(costmap.uncertainty_layers[name], dtype=np.float32)
        confidence = np.asarray(confidence_layers.get(name, 1.0 - uncertainty), dtype=np.float32)
        low_confidence = np.clip(1.0 - confidence, 0.0, 1.0)
        if name == "distance":
            layers[name] = layer.copy()
            uncertainty_layers[name] = uncertainty.copy()
            continue
        sigma = float(noise_scale) * np.clip(0.35 * uncertainty + 0.65 * low_confidence, 0.0, 1.0)
        zero_mean = rng.normal(loc=0.0, scale=np.maximum(sigma, 1e-6), size=layer.shape).astype(np.float32)
        one_sided_tail = rng.uniform(0.0, 0.35, size=layer.shape).astype(np.float32) * sigma
        sampled = layer + zero_mean + one_sided_tail
        layers[name] = np.clip(sampled, 0.0, 1.0).astype(np.float32)
        uncertainty_layers[name] = np.clip(np.maximum(uncertainty, np.abs(sampled - layer)), 0.0, 1.0).astype(np.float32)
    sampled_costmap = replace(costmap, layers=layers, uncertainty_layers=uncertainty_layers)
    # Important: CVaR selection must not score candidates with the hidden true map.
    return replace(episode, costmap=sampled_costmap, true_costmap=None, confidence_layers=None)


def upper_tail_cvar(costs: list[float], alpha: float) -> float:
    values = np.asarray([cost for cost in costs if math.isfinite(float(cost))], dtype=np.float64)
    if values.size == 0:
        return 10.0
    tail_fraction = float(np.clip(alpha, 1.0 / values.size, 1.0))
    tail_count = max(1, int(math.ceil(tail_fraction * values.size)))
    return float(np.sort(values)[-tail_count:].mean())


def select_belief_sample_cvar(
    episode: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    max_lambda = float(config.get("max_uncertainty_lambda", 2.0))
    candidates = baseline_configs(episode, max_lambda, bool(config.get("include_all_presets", False)))
    rng = np.random.default_rng(int(config.get("selection_seed", 0)))
    samples = max(1, int(config.get("cvar_samples", 5)))
    alpha = float(config.get("cvar_alpha", 0.35))
    noise_scale = float(config.get("cvar_noise_scale", 0.55))
    sampled_episodes = [sample_plausible_belief_episode(episode, rng, noise_scale) for _ in range(samples)]

    best_name = ""
    best_config: dict[str, Any] | None = None
    best_cvar = float("inf")
    best_mean = float("inf")
    for candidate_name, candidate in candidates.items():
        costs: list[float] = []
        for sampled_episode in sampled_episodes:
            result = plan_with_weights(
                sampled_episode,
                np.asarray(candidate["weights"], dtype=np.float32),
                lambda_uncertainty=float(candidate["lambda_uncertainty"]),
                allow_diagonal=True,
            )
            costs.append(float(result.get("scalar_cost", 10.0)))
        cvar = upper_tail_cvar(costs, alpha)
        mean_cost = safe_mean(costs, default=10.0)
        if cvar < best_cvar - 1e-12 or (abs(cvar - best_cvar) <= 1e-12 and mean_cost < best_mean):
            best_name = candidate_name
            best_config = dict(candidate)
            best_cvar = cvar
            best_mean = mean_cost
    if best_config is None:
        raise RuntimeError("belief-sample CVaR could not select a candidate")
    best_config["selected_candidate_id"] = best_name
    best_config["belief_cvar_score"] = float(best_cvar)
    best_config["belief_cvar_mean_score"] = float(best_mean)
    best_config["belief_cvar_samples"] = int(samples)
    best_config["belief_cvar_alpha"] = float(alpha)
    return best_config


def _reported_cost(result: dict[str, Any], active_attack: dict[str, Any]) -> float:
    if not attack_enabled(active_attack):
        return float(result.get("scalar_cost", np.nan))
    if str(active_attack.get("type", "")) == "env_zscore_topk":
        return float(result.get("soft_attacked_scalar_cost", result.get("attacked_scalar_cost", np.nan)))
    return float(result.get("scalar_cost", np.nan))


def plan_cost(episode: Any, config: dict[str, Any], env_attack: dict[str, Any] | None) -> tuple[float, dict[str, Any]]:
    active = env_attack if attack_enabled(env_attack) else {}
    robust_method = str(config.get("robust_method", ""))
    planning_episode = episode
    planning_config = config
    if robust_method == "risk_inflated_astar":
        planning_episode = apply_risk_inflation_to_episode(
            episode,
            float(config.get("risk_inflation_scale", 0.42)),
            int(config.get("risk_inflation_radius", 2)),
        )
    elif robust_method == "belief_sample_cvar":
        planning_config = select_belief_sample_cvar(episode, config)

    result = plan_with_weights(
        planning_episode,
        np.asarray(planning_config["weights"], dtype=np.float32),
        lambda_uncertainty=float(planning_config["lambda_uncertainty"]),
        allow_diagonal=True,
        attacker_temperature=float(active.get("attacker_temperature", 0.5)) if attack_enabled(active) else 0.5,
        attacker_response=str(active.get("attacker_response", "zscore_topk")) if attack_enabled(active) else "zscore_topk",
        attacker_top_fraction=float(active.get("attacker_top_fraction", 0.15)) if attack_enabled(active) else 0.15,
        attacker_sharpness=float(active.get("attacker_sharpness", 3.0)) if attack_enabled(active) else 3.0,
        attack_strength=float(active.get("attack_strength", 1.0)) if attack_enabled(active) else 1.0,
    )
    if robust_method:
        result["robust_method"] = robust_method
        result["selected_candidate_id"] = str(planning_config.get("selected_candidate_id", ""))
        for key in (
            "belief_cvar_score",
            "belief_cvar_mean_score",
            "belief_cvar_samples",
            "belief_cvar_alpha",
            "risk_inflation_scale",
            "risk_inflation_radius",
        ):
            if key in planning_config:
                result[key] = planning_config[key]
    cost = _reported_cost(result, active)
    return cost, result


def baseline_configs(episode: Any, max_lambda: float, include_all: bool) -> dict[str, dict[str, Any]]:
    configs = candidate_planner_configs(episode, max_uncertainty_lambda=max_lambda)
    if include_all:
        return configs
    return {name: configs[name] for name in FOCUS_METHODS if name in configs}


def attack_variants(env_attack: dict[str, Any], mode: str, mixture_size: int, jitter: float) -> list[dict[str, Any]]:
    variants = [benchmark_attack_variant(env_attack)]
    if mode in {"component", "scale_component"}:
        variants.extend(component_attack_variants(env_attack))
    if mode in {"scale", "scale_component"}:
        for scale in local_attack_scales(int(mixture_size), float(jitter)):
            variants.append(
                {
                    "variant_id": f"local_scale_{scale:.3f}",
                    "scale": float(scale),
                    "config": scale_attack_config(env_attack, float(scale)),
                }
            )
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for variant in variants:
        variant_id = str(variant.get("variant_id", f"variant_{len(unique)}"))
        if variant_id in seen:
            continue
        seen.add(variant_id)
        unique.append(variant)
    return unique


def select_model_minimax(
    episode: Any,
    configs: dict[str, dict[str, Any]],
    env_attack: dict[str, Any],
    seed: int,
    episode_index: int,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any], float]:
    variants = attack_variants(
        env_attack,
        str(args.minimax_variant_mode),
        int(args.minimax_mixture_size),
        float(args.minimax_jitter),
    )
    best_name = ""
    best_config: dict[str, Any] | None = None
    best_score = float("inf")
    for method_name, config in configs.items():
        costs: list[float] = []
        for variant_index, variant in enumerate(variants):
            variant_config = copy.deepcopy(variant["config"])
            attacked = apply_environment_attack_to_episode(
                episode,
                variant_config,
                np.random.default_rng(seed + 500_000 + 10_000 * episode_index + variant_index),
            )
            cost, _ = plan_cost(attacked, config, variant_config)
            costs.append(cost)
        score = float(np.nanmax(np.asarray(costs, dtype=np.float64)))
        if score < best_score:
            best_name = method_name
            best_config = config
            best_score = score
    if best_config is None:
        raise RuntimeError("model minimax could not select a candidate")
    return best_name, best_config, best_score


def evaluate_method_on_domains(
    run_dir: Path,
    run_config: dict[str, Any],
    episodes_by_domain: dict[str, tuple[int, list[Any]]],
    env_attack: dict[str, Any],
    reference_clean: dict[str, float],
    selected_validation_method: str | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    level, difficulty, seed = run_label_from_dir(run_dir)
    base_args = dict(run_config["base_config_args"])
    max_lambda = float(config_value(base_args, "max-uncertainty-lambda", 1.2))
    domain_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []

    for eval_domain, (map_seed, episodes) in episodes_by_domain.items():
        per_method: dict[str, list[dict[str, float]]] = {}
        for episode_index, episode in enumerate(episodes):
            configs = baseline_configs(episode, max_lambda, bool(args.include_all_presets))
            eval_configs = dict(configs)
            if selected_validation_method and selected_validation_method in configs:
                eval_configs["validation_best_static"] = configs[selected_validation_method]
            if not bool(args.disable_model_minimax):
                selected_name, selected_config, minimax_score = select_model_minimax(
                    episode,
                    configs,
                    env_attack,
                    int(seed),
                    int(episode_index),
                    args,
                )
                model_config = dict(selected_config)
                model_config["selected_candidate_id"] = selected_name
                model_config["minimax_score"] = minimax_score
                eval_configs["model_minimax"] = model_config
            if not bool(args.disable_risk_inflated):
                eval_configs["risk_inflated_astar"] = risk_inflated_planner_config(episode, max_lambda, args)
            if not bool(args.disable_belief_cvar):
                eval_configs["belief_sample_cvar"] = belief_sample_cvar_config(
                    max_lambda,
                    int(seed) + 730_000 + 10_000 * int(map_seed) + int(episode_index),
                    args,
                )

            attacked_episode = apply_environment_attack_to_episode(
                episode,
                env_attack,
                np.random.default_rng(seed + 300_000 + episode_index),
            )
            for method_name, config in eval_configs.items():
                clean_cost, clean_result = plan_cost(episode, config, {})
                attacked_cost, attacked_result = plan_cost(attacked_episode, config, env_attack)
                values = {
                    "clean_cost": float(clean_cost),
                    "attacked_cost": float(attacked_cost),
                    "success": 1.0 if bool(attacked_result.get("success", False)) else 0.0,
                    "path_length": float(attacked_result.get("path_length", np.nan)),
                    "lambda_uncertainty": float(config["lambda_uncertainty"]),
                    "clean_success": 1.0 if bool(clean_result.get("success", False)) else 0.0,
                }
                per_method.setdefault(method_name, []).append(values)
                if bool(args.save_episode_details):
                    weights = np.asarray(attacked_result.get("weights", config["weights"]), dtype=np.float32)
                    detail = {
                        "level": level,
                        "difficulty": difficulty,
                        "seed": int(seed),
                        "eval_domain": eval_domain,
                        "map_pool_seed": int(map_seed),
                        "episode_index": int(episode_index),
                        "method": method_name,
                        **values,
                    }
                    for index, objective_name in enumerate(OBJECTIVE_NAMES):
                        detail[f"weight_{objective_name}"] = float(weights[index])
                    if method_name == "model_minimax":
                        detail["selected_candidate_id"] = str(config.get("selected_candidate_id", ""))
                        detail["minimax_score"] = float(config.get("minimax_score", np.nan))
                    if method_name in {"risk_inflated_astar", "belief_sample_cvar"}:
                        detail["robust_method"] = str(attacked_result.get("robust_method", method_name))
                        detail["selected_candidate_id"] = str(attacked_result.get("selected_candidate_id", ""))
                        detail["belief_cvar_score"] = float(attacked_result.get("belief_cvar_score", np.nan))
                        detail["belief_cvar_mean_score"] = float(attacked_result.get("belief_cvar_mean_score", np.nan))
                        detail["risk_inflation_scale"] = float(attacked_result.get("risk_inflation_scale", np.nan))
                    detail_rows.append(detail)

        domain_reference = float(reference_clean.get(eval_domain, np.nan))
        if not math.isfinite(domain_reference):
            domain_reference = safe_mean(
                [item["clean_cost"] for values in per_method.values() for item in values],
            )
        for method_name, values in per_method.items():
            own_clean = safe_mean([item["clean_cost"] for item in values])
            attacked = safe_mean([item["attacked_cost"] for item in values])
            domain_rows.append(
                {
                    "level": level,
                    "difficulty": difficulty,
                    "seed": int(seed),
                    "eval_domain": eval_domain,
                    "map_pool_seed": int(map_seed),
                    "method": method_name,
                    "num_episodes": int(len(values)),
                    "reference_clean_cost": domain_reference,
                    "mean_clean_cost": own_clean,
                    "mean_attacked_cost": attacked,
                    "std_attacked_cost": safe_std([item["attacked_cost"] for item in values]),
                    "success_rate": safe_mean([item["success"] for item in values]),
                    "clean_success_rate": safe_mean([item["clean_success"] for item in values]),
                    "mean_path_length": safe_mean([item["path_length"] for item in values]),
                    "mean_lambda_uncertainty": safe_mean([item["lambda_uncertainty"] for item in values]),
                    "reference_performance_index": 100.0 * domain_reference / max(attacked, 1e-12),
                    "self_normalized_index": 100.0 * own_clean / max(attacked, 1e-12),
                    "selected_validation_method": selected_validation_method or "",
                }
            )
    return domain_rows, detail_rows


def select_validation_best(
    run_dir: Path,
    run_config: dict[str, Any],
    env_attack: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    _, validation_domains = validation_episodes_from_run(run_dir, run_config, int(args.validation_episodes))
    base_args = dict(run_config["base_config_args"])
    max_lambda = float(config_value(base_args, "max-uncertainty-lambda", 1.2))
    _, _, seed = run_label_from_dir(run_dir)
    scores: dict[str, list[float]] = {}
    for _, (_, episodes) in validation_domains.items():
        for episode_index, episode in enumerate(episodes):
            configs = baseline_configs(episode, max_lambda, bool(args.include_all_presets))
            attacked_episode = apply_environment_attack_to_episode(
                episode,
                env_attack,
                np.random.default_rng(seed + 410_000 + episode_index),
            )
            for method_name, config in configs.items():
                cost, _ = plan_cost(attacked_episode, config, env_attack)
                scores.setdefault(method_name, []).append(cost)
    if not scores:
        raise RuntimeError(f"no validation baseline scores for {run_dir}")
    return min(scores.items(), key=lambda item: safe_mean(item[1]))[0]


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    run_config = read_json(run_dir / "run_config.json")
    env_attack = dict(run_config["environment_attack"])
    count = int(args.num_eval_episodes)
    map_size, episodes_by_domain = eval_episodes_from_run(run_dir, run_config, count)
    reference_clean = reference_clean_costs(run_dir)
    selected = None
    if not bool(args.disable_validation_best):
        selected = select_validation_best(run_dir, run_config, env_attack, args)
    domain_rows, detail_rows = evaluate_method_on_domains(
        run_dir,
        run_config,
        episodes_by_domain,
        env_attack,
        reference_clean,
        selected,
        args,
    )
    meta = {
        "run_dir": str(run_dir),
        "map_size": int(map_size),
        "selected_validation_method": selected or "",
    }
    return pd.DataFrame(domain_rows), pd.DataFrame(detail_rows), meta


def panel_summary(domain_frame: pd.DataFrame) -> pd.DataFrame:
    if domain_frame.empty:
        return pd.DataFrame()
    grouped = (
        domain_frame.groupby(["method", "level", "difficulty"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            num_domains=("eval_domain", "count"),
            num_episodes=("num_episodes", "sum"),
            reference_clean_cost=("reference_clean_cost", "mean"),
            mean_clean_cost=("mean_clean_cost", "mean"),
            mean_attacked_cost=("mean_attacked_cost", "mean"),
            final_recovery_index=("reference_performance_index", "mean"),
            self_normalized_index=("self_normalized_index", "mean"),
            success_rate=("success_rate", "mean"),
            mean_lambda_uncertainty=("mean_lambda_uncertainty", "mean"),
        )
        .sort_values(["level", "difficulty", "method"])
    )
    grouped["algorithm"] = "planner_" + grouped["method"].astype(str)
    grouped["attack_shock_index"] = grouped["final_recovery_index"]
    grouped["best_recovery_index"] = grouped["final_recovery_index"]
    return grouped


def load_rl_rows(rl_summary: Path, game_summary: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    rl_path = resolve(rl_summary)
    if rl_path.exists():
        frame = pd.read_csv(rl_path)
        frames.append(frame[frame["algorithm"].isin(["ppo", "sac"])].copy())
    game_path = resolve(game_summary)
    if game_path.exists():
        frames.append(pd.read_csv(game_path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def compare_with_rl(panel: pd.DataFrame, rl: pd.DataFrame) -> pd.DataFrame:
    if panel.empty or rl.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    baseline = rl.set_index(["algorithm", "level", "difficulty"])
    for item in panel.itertuples(index=False):
        for baseline_algorithm in ("ppo", "sac", "game_ppo"):
            key = (baseline_algorithm, item.level, item.difficulty)
            if key not in baseline.index:
                continue
            base = baseline.loc[key]
            base_final = float(base["final_recovery_index"])
            base_best = float(base["best_recovery_index"])
            rows.append(
                {
                    "method": item.method,
                    "level": item.level,
                    "difficulty": item.difficulty,
                    "baseline_algorithm": baseline_algorithm,
                    "planner_final_recovery_index": float(item.final_recovery_index),
                    "baseline_final_recovery_index": base_final,
                    "delta_final_recovery_index": float(item.final_recovery_index) - base_final,
                    "planner_best_recovery_index": float(item.best_recovery_index),
                    "baseline_best_recovery_index": base_best,
                    "delta_best_recovery_index": float(item.best_recovery_index) - base_best,
                }
            )
    return pd.DataFrame(rows)


def plot_bars(panel: pd.DataFrame, rl: pd.DataFrame, output_dir: Path) -> None:
    if panel.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9.5), sharey=False)
    rl_lookup = rl.set_index(["algorithm", "level", "difficulty"]) if not rl.empty else pd.DataFrame()
    for row_index, level in enumerate(LEVELS):
        for col_index, difficulty in enumerate(DIFFICULTIES):
            ax = axes[row_index, col_index]
            labels: list[str] = []
            values: list[float] = []
            colors: list[str] = []
            for algorithm, color in (("ppo", "#2f6b9a"), ("sac", "#2a7f62"), ("game_ppo", "#7a3db8")):
                if not rl.empty and (algorithm, level, difficulty) in rl_lookup.index:
                    labels.append(LABELS.get(algorithm, algorithm))
                    values.append(float(rl_lookup.loc[(algorithm, level, difficulty), "final_recovery_index"]))
                    colors.append(color)
            subset = panel[(panel["level"] == level) & (panel["difficulty"] == difficulty)]
            for method in PLOT_METHODS:
                row = subset[subset["method"] == method]
                if row.empty:
                    continue
                labels.append(LABELS.get(method, method))
                values.append(float(row.iloc[0]["final_recovery_index"]))
                colors.append("#8a6f2a" if method != "model_minimax" else "#c45a1a")
            if values:
                ax.bar(np.arange(len(values)), values, color=colors, alpha=0.88)
                ax.set_xticks(np.arange(len(values)))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
            ax.axhline(100.0, color="0.35", linestyle="--", linewidth=0.9)
            ax.set_title(f"{level} {difficulty}", fontsize=10)
            ax.grid(axis="y", alpha=0.25)
            if col_index == 0:
                ax.set_ylabel("performance index")
    fig.suptitle("Non-learning Planner Baselines vs RL Recovery", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_dir / "fig_planner_baselines_vs_rl_recovery.png", dpi=220)
    plt.close(fig)


def write_report(panel: pd.DataFrame, delta: pd.DataFrame, output_dir: Path, metas: list[dict[str, Any]]) -> None:
    lines = [
        "# Non-learning Planner Baseline Evaluation",
        "",
        "These baselines do not train a policy. They reuse the same level configs, eval tasks, and benchmark attack as the RL shock-recovery protocol.",
        "",
        "Main deployable baselines:",
        "- `fixed`: equal objective weights.",
        "- `heuristic`: mission-priority weights.",
        "- `rover_guard`: hand-coded rover-state guard.",
        "- `emergency_uncertainty_rule`: hand-coded uncertainty-aware safety rule.",
        "- `risk_inflated_astar`: risk-aware A* that inflates belief costs using uncertainty, confidence, and hard-obstacle proximity.",
        "- `belief_sample_cvar`: CVaR-style planner selection over sampled plausible belief maps; it does not use the hidden true map.",
        "- `validation_best_static`: best preset selected on validation tasks, then reported on train/heldout eval domains.",
        "- `model_minimax`: model-based robust planner over known attack variants; non-learning but assumes an attack model.",
        "",
        f"- Evaluated panels: {int(panel[['level', 'difficulty']].drop_duplicates().shape[0]) if not panel.empty else 0}",
        f"- Source runs: {len(metas)}",
        "",
    ]
    if not delta.empty:
        for baseline in ("ppo", "sac", "game_ppo"):
            subset = delta[delta["baseline_algorithm"] == baseline]
            if subset.empty:
                continue
            best_rows = subset[subset["method"].isin(PLOT_METHODS)]
            wins = int((best_rows["delta_final_recovery_index"] > 0.0).sum())
            total = int(best_rows.shape[0])
            lines.append(f"- Planner plot methods beat {baseline} in {wins}/{total} method-panel comparisons.")
    lines.extend(
        [
            "",
            "Generated files:",
            "- `planner_baseline_domain_summary.csv`",
            "- `planner_baseline_panel_summary.csv`",
            "- `planner_baseline_vs_rl_deltas.csv`",
            "- `fig_planner_baselines_vs_rl_recovery.png`",
        ]
    )
    if not panel.empty:
        focus = panel[panel["method"].isin(PLOT_METHODS)].copy()
        focus = focus.sort_values(["level", "difficulty", "method"])
        lines.extend(["", "| Method | Level | Difficulty | Index | Self index | Success |", "|---|---|---|---:|---:|---:|"])
        for row in focus.itertuples(index=False):
            lines.append(
                f"| {row.method} | {row.level} | {row.difficulty} | "
                f"{float(row.final_recovery_index):.2f} | {float(row.self_normalized_index):.2f} | "
                f"{100.0 * float(row.success_rate):.1f}% |"
            )
    (output_dir / "nonlearning_baseline_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.quick:
        args.num_eval_episodes = min(int(args.num_eval_episodes), 8)
        args.validation_episodes = min(int(args.validation_episodes), 6)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = source_run_dirs(args)
    if args.dry_run:
        print("Non-learning planner baseline dry run:")
        for run_dir in run_dirs:
            print(f"  {run_dir}")
        return 0
    if not run_dirs:
        raise FileNotFoundError("no source runs found")

    all_domain: list[pd.DataFrame] = []
    all_detail: list[pd.DataFrame] = []
    metas: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        level, difficulty, seed = run_label_from_dir(run_dir)
        print(f"Evaluating non-learning baselines: {level}/{difficulty} seed={seed}", flush=True)
        domain_frame, detail_frame, meta = evaluate_run(run_dir, args)
        all_domain.append(domain_frame)
        if not detail_frame.empty:
            all_detail.append(detail_frame)
        metas.append(meta)

    domain = pd.concat(all_domain, ignore_index=True) if all_domain else pd.DataFrame()
    panel = panel_summary(domain)
    rl = load_rl_rows(args.rl_summary, args.game_summary)
    delta = compare_with_rl(panel, rl)

    domain.to_csv(output_dir / "planner_baseline_domain_summary.csv", index=False)
    panel.to_csv(output_dir / "planner_baseline_panel_summary.csv", index=False)
    delta.to_csv(output_dir / "planner_baseline_vs_rl_deltas.csv", index=False)
    if all_detail:
        pd.concat(all_detail, ignore_index=True).to_csv(output_dir / "planner_baseline_episode_details.csv", index=False)
    pd.DataFrame(metas).to_csv(output_dir / "planner_baseline_run_metadata.csv", index=False)
    plot_bars(panel, rl, output_dir)
    write_report(panel, delta, output_dir, metas)

    print(f"Saved domain summary: {output_dir / 'planner_baseline_domain_summary.csv'}")
    print(f"Saved panel summary: {output_dir / 'planner_baseline_panel_summary.csv'}")
    print(f"Saved deltas: {output_dir / 'planner_baseline_vs_rl_deltas.csv'}")
    print(f"Saved figure: {output_dir / 'fig_planner_baselines_vs_rl_recovery.png'}")
    print(f"Saved report: {output_dir / 'nonlearning_baseline_report.md'}")
    if not panel.empty:
        cols = ["method", "level", "difficulty", "final_recovery_index", "success_rate"]
        print(panel[panel["method"].isin(PLOT_METHODS)][cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
