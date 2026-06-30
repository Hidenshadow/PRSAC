#!/usr/bin/env python
"""Protocol-aware planner-oracle upper-bound check for shock recovery runs.

This script does not train. It reuses completed PPO shock-recovery runs and
asks whether a best-of-candidates planner search can beat the final recovery
checkpoint under the benchmark attack. The oracle is intentionally optimistic:
it selects the candidate with the lowest true attacked scalar cost per episode.
Use it as a headroom diagnostic, not as a deployable score.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.attack_wrappers import apply_environment_attack_to_episode, attack_enabled  # noqa: E402
from run_attack_recovery_finetune import config_value, eval_attack_cost, generate_episodes  # noqa: E402
from run_lunar_viper_staged_recovery import generate_real_episodes  # noqa: E402
from run_shock_recovery_experiment import (  # noqa: E402
    component_attack_variants,
    local_attack_scales,
    scale_attack_config,
)
from utils.evaluation_policy import (  # noqa: E402
    load_model,
    predict_action,
    resolve_action_config,
    resolve_observation_mode,
)
from utils.metrics import (  # noqa: E402
    DEFAULT_MAP_SEED_POOL_SIZE,
    OBJECTIVE_NAMES,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    candidate_planner_configs,
    compute_observation,
    normalize_weights,
    plan_with_weights,
)


DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "rl_baselines" / "ppo"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "planner_oracle_protocol_analysis" / "seed0_best_of_candidates"
DEFAULT_BASELINE_SUMMARY = PROJECT_ROOT / "runs" / "rl_baselines" / "paper_story_5seeds" / "rl_baseline_performance_story.csv"


LEVEL_ORDER = {"level1": 0, "level2": 1, "level3": 2}
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=32)
    parser.add_argument("--num-random-candidates", type=int, default=96)
    parser.add_argument("--num-local-candidates", type=int, default=24)
    parser.add_argument("--local-sigma", type=float, default=0.12)
    parser.add_argument("--num-structured-candidates", type=int, default=0)
    parser.add_argument("--disable-game-theory", action="store_true")
    parser.add_argument(
        "--game-attack-variant-mode",
        choices=("scale", "component", "scale_component"),
        default="scale_component",
    )
    parser.add_argument("--game-attack-mixture-size", type=int, default=5)
    parser.add_argument("--game-attack-jitter", type=float, default=0.18)
    parser.add_argument(
        "--game-softmax-temperature",
        type=float,
        default=0.08,
        help="Relative temperature for soft Stackelberg/log-sum-exp attacker response.",
    )
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_label_from_dir(run_dir: Path) -> tuple[str, str]:
    match = re.search(r"(level\d+)_(easy|medium|hard)_shock_recovery", run_dir.parent.name)
    if match is None:
        raise ValueError(f"cannot parse level/difficulty from {run_dir}")
    return match.group(1), match.group(2)


def discover_run_dirs(root: Path, seed: int) -> list[Path]:
    root = resolve_path(root)
    paths = sorted(root.glob(f"level*_shock_recovery_5seeds/seed{int(seed)}"))
    return [
        path
        for path in paths
        if (path / "run_config.json").exists()
        and (path / "checkpoints" / "checkpoint_nominal.pt").exists()
    ]


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_recovery_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def final_recovery_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(
        (run_dir / "checkpoints").glob("checkpoint_recovery_step_*.pt"),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(f"no recovery checkpoints in {run_dir / 'checkpoints'}")
    return checkpoints[-1]


def unique_game_attack_variants(env_attack: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build the attacker action set for the local zero-sum game."""

    variants: list[dict[str, Any]] = [
        {
            "variant_id": "benchmark",
            "scale": 1.0,
            "config": copy.deepcopy(env_attack),
        }
    ]
    mode = str(args.game_attack_variant_mode)
    if mode in {"component", "scale_component"}:
        variants.extend(component_attack_variants(env_attack))
    if mode in {"scale", "scale_component"}:
        for scale in local_attack_scales(int(args.game_attack_mixture_size), float(args.game_attack_jitter))[1:]:
            variants.append(
                {
                    "variant_id": f"local_scale_{scale:.3f}",
                    "scale": float(scale),
                    "config": scale_attack_config(env_attack, float(scale)),
                }
            )

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        variant_id = str(variant.get("variant_id", f"variant_{index}"))
        if variant_id in seen:
            suffix = 2
            base_id = variant_id
            while f"{base_id}_{suffix}" in seen:
                suffix += 1
            variant_id = f"{base_id}_{suffix}"
        seen.add(variant_id)
        item = dict(variant)
        item["variant_id"] = variant_id
        unique.append(item)
    return unique


def policy_action_config(
    policy: tuple[str, Any, dict[str, Any]],
    episode: Any,
    map_size: int,
) -> dict[str, Any]:
    model_type, model, model_config = policy
    observation_mode = resolve_observation_mode("auto", model_config)
    action_mode, action_gain, max_lambda = resolve_action_config("auto", None, None, model_config)
    obs = compute_observation(
        episode,
        map_size,
        observation_mode=observation_mode,
        max_uncertainty_lambda=max_lambda,
    )
    action = np.asarray(predict_action(model_type, model, obs), dtype=np.float32).reshape(-1)
    return {
        "action": action,
        "weights": action_to_planning_weights(episode, action, action_mode=action_mode, action_gain=action_gain),
        "lambda_uncertainty": action_to_uncertainty_lambda(action, max_uncertainty_lambda=max_lambda),
        "max_lambda": float(max_lambda),
    }


def make_candidate_key(weights: np.ndarray, lambda_uncertainty: float) -> tuple[tuple[float, ...], float]:
    clipped = normalize_weights(np.asarray(weights, dtype=np.float32))
    return tuple(float(round(value, 6)) for value in clipped), float(round(float(lambda_uncertainty), 6))


def add_candidate(
    candidates: list[dict[str, Any]],
    seen: set[tuple[tuple[float, ...], float]],
    method: str,
    candidate_id: str,
    weights: np.ndarray,
    lambda_uncertainty: float,
    max_lambda: float,
) -> None:
    lam = float(np.clip(lambda_uncertainty, 0.0, max(float(max_lambda), 0.0)))
    normalized = normalize_weights(np.asarray(weights, dtype=np.float32))
    key = make_candidate_key(normalized, lam)
    if key in seen:
        return
    seen.add(key)
    candidates.append(
        {
            "method": method,
            "candidate_id": candidate_id,
            "weights": normalized,
            "lambda_uncertainty": lam,
        }
    )


def add_policy_neighborhood(
    candidates: list[dict[str, Any]],
    seen: set[tuple[tuple[float, ...], float]],
    center: dict[str, Any],
    prefix: str,
    rng: np.random.Generator,
    num_local: int,
    local_sigma: float,
    max_lambda: float,
) -> None:
    weights = normalize_weights(np.asarray(center["weights"], dtype=np.float32))
    lam = float(center["lambda_uncertainty"])
    add_candidate(candidates, seen, prefix, "center", weights, lam, max_lambda)

    for axis, name in enumerate(OBJECTIVE_NAMES):
        one_hot = np.zeros(len(OBJECTIVE_NAMES), dtype=np.float32)
        one_hot[axis] = 1.0
        for mix in (0.25, 0.50):
            shifted = normalize_weights((1.0 - mix) * weights + mix * one_hot)
            add_candidate(candidates, seen, f"{prefix}_axis", f"{name}_{mix:.2f}", shifted, lam, max_lambda)
            add_candidate(
                candidates,
                seen,
                f"{prefix}_axis_high_lambda",
                f"{name}_{mix:.2f}",
                shifted,
                max(lam, 0.85 * max_lambda),
                max_lambda,
            )

    for index in range(max(0, int(num_local))):
        noisy = np.clip(
            weights + rng.normal(0.0, max(float(local_sigma), 0.0), size=len(OBJECTIVE_NAMES)),
            0.0,
            None,
        )
        if float(noisy.sum()) <= 1e-8:
            noisy = rng.dirichlet(np.ones(len(OBJECTIVE_NAMES))).astype(np.float32)
        lambda_noise = rng.normal(0.0, 0.18 * max_lambda)
        add_candidate(
            candidates,
            seen,
            f"{prefix}_local",
            f"local_{index:03d}",
            noisy,
            lam + lambda_noise,
            max_lambda,
        )


def build_candidates(
    eval_episode: Any,
    map_size: int,
    nominal_policy: tuple[str, Any, dict[str, Any]],
    recovery_policy: tuple[str, Any, dict[str, Any]],
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    nominal_cfg = policy_action_config(nominal_policy, eval_episode, map_size)
    recovery_cfg = policy_action_config(recovery_policy, eval_episode, map_size)
    max_lambda = max(float(nominal_cfg["max_lambda"]), float(recovery_cfg["max_lambda"]), 1e-6)

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[float, ...], float]] = set()
    add_policy_neighborhood(
        candidates,
        seen,
        nominal_cfg,
        "nominal_ppo",
        rng,
        args.num_local_candidates,
        args.local_sigma,
        max_lambda,
    )
    add_policy_neighborhood(
        candidates,
        seen,
        recovery_cfg,
        "final_recovery_ppo",
        rng,
        args.num_local_candidates,
        args.local_sigma,
        max_lambda,
    )

    structured = list(candidate_planner_configs(eval_episode, max_uncertainty_lambda=max_lambda).items())
    if int(args.num_structured_candidates) > 0:
        structured = structured[: int(args.num_structured_candidates)]
    for name, config in structured:
        add_candidate(
            candidates,
            seen,
            "structured",
            str(name),
            np.asarray(config["weights"], dtype=np.float32),
            float(config["lambda_uncertainty"]),
            max_lambda,
        )

    for index in range(max(0, int(args.num_random_candidates))):
        add_candidate(
            candidates,
            seen,
            "random",
            f"random_{index:03d}",
            rng.dirichlet(np.ones(len(OBJECTIVE_NAMES))).astype(np.float32),
            float(rng.uniform(0.0, max_lambda)),
            max_lambda,
        )

    return candidates, nominal_cfg, recovery_cfg


def plan_cost(
    episode: Any,
    weights: np.ndarray,
    lambda_uncertainty: float,
    env_attack: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    result = plan_with_weights(
        episode,
        weights,
        lambda_uncertainty=lambda_uncertainty,
        allow_diagonal=True,
        attacker_temperature=float(env_attack.get("attacker_temperature", 0.5))
        if attack_enabled(env_attack)
        else 0.5,
        attacker_response=str(env_attack.get("attacker_response", "zscore_topk"))
        if attack_enabled(env_attack)
        else "zscore_topk",
        attacker_top_fraction=float(env_attack.get("attacker_top_fraction", 0.15))
        if attack_enabled(env_attack)
        else 0.15,
        attacker_sharpness=float(env_attack.get("attacker_sharpness", 3.0))
        if attack_enabled(env_attack)
        else 3.0,
        attack_strength=float(env_attack.get("attack_strength", 1.0))
        if attack_enabled(env_attack)
        else 1.0,
    )
    return eval_attack_cost(result, env_attack), result


def _finite_cost_matrix(cost_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(cost_matrix, dtype=np.float64)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return np.ones_like(matrix, dtype=np.float64) * 10.0
    fallback = float(np.nanmax(finite) + max(abs(float(np.nanmax(finite))), 1.0))
    return np.where(np.isfinite(matrix), matrix, fallback)


def _logsumexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    max_value = float(np.max(array))
    return float(max_value + np.log(np.exp(array - max_value).sum()))


def solve_defender_mixed_strategy(cost_matrix: np.ndarray) -> tuple[np.ndarray, float, str]:
    """Solve min_p max_j E_p[cost(a, j)] for the defender mixed strategy."""

    matrix = _finite_cost_matrix(cost_matrix)
    num_actions, num_attacks = matrix.shape
    minimax_index = int(np.argmin(np.max(matrix, axis=1)))
    fallback = np.zeros(num_actions, dtype=np.float64)
    fallback[minimax_index] = 1.0
    fallback_value = float(np.max(matrix[minimax_index, :]))
    try:
        from scipy.optimize import linprog

        c = np.zeros(num_actions + 1, dtype=np.float64)
        c[-1] = 1.0
        a_ub = []
        b_ub = []
        for attack_index in range(num_attacks):
            row = np.zeros(num_actions + 1, dtype=np.float64)
            row[:num_actions] = matrix[:, attack_index]
            row[-1] = -1.0
            a_ub.append(row)
            b_ub.append(0.0)
        a_eq = np.zeros((1, num_actions + 1), dtype=np.float64)
        a_eq[0, :num_actions] = 1.0
        bounds = [(0.0, 1.0)] * num_actions + [(0.0, None)]
        result = linprog(
            c,
            A_ub=np.asarray(a_ub, dtype=np.float64),
            b_ub=np.asarray(b_ub, dtype=np.float64),
            A_eq=a_eq,
            b_eq=np.asarray([1.0], dtype=np.float64),
            bounds=bounds,
            method="highs",
        )
        if not result.success:
            return fallback, fallback_value, "linprog_failed"
        probs = np.asarray(result.x[:num_actions], dtype=np.float64)
        probs = np.clip(probs, 0.0, 1.0)
        probs = probs / max(float(probs.sum()), 1e-12)
        return probs, float(result.x[-1]), "linprog"
    except Exception:
        return fallback, fallback_value, "fallback_minimax"


def blended_candidate_from_strategy(candidates: list[dict[str, Any]], probs: np.ndarray) -> dict[str, Any]:
    weights = np.zeros(len(OBJECTIVE_NAMES), dtype=np.float64)
    lambda_uncertainty = 0.0
    for prob, candidate in zip(np.asarray(probs, dtype=np.float64), candidates):
        weights += float(prob) * np.asarray(candidate["weights"], dtype=np.float64)
        lambda_uncertainty += float(prob) * float(candidate["lambda_uncertainty"])
    return {
        "method": "game_nash_blend_teacher",
        "candidate_id": "mixed_strategy_blend",
        "weights": normalize_weights(weights.astype(np.float32)),
        "lambda_uncertainty": float(lambda_uncertainty),
    }


def evaluate_game_teachers(
    base_episode: Any,
    candidates: list[dict[str, Any]],
    env_attack: dict[str, Any],
    run_seed: int,
    episode_index: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate candidate actions as a zero-sum defender-vs-attacker game."""

    variants = unique_game_attack_variants(env_attack, args)
    variant_episodes = []
    for variant_index, variant in enumerate(variants):
        variant_seed = int(run_seed) + 300_000 + int(episode_index) + 9_973 * int(variant_index)
        variant_episodes.append(
            apply_environment_attack_to_episode(
                base_episode,
                dict(variant["config"]),
                np.random.default_rng(variant_seed),
            )
        )

    matrix = np.zeros((len(candidates), len(variants)), dtype=np.float64)
    for candidate_index, candidate in enumerate(candidates):
        for variant_index, (variant, variant_episode) in enumerate(zip(variants, variant_episodes)):
            cost, _ = plan_cost(
                variant_episode,
                candidate["weights"],
                float(candidate["lambda_uncertainty"]),
                dict(variant["config"]),
            )
            matrix[candidate_index, variant_index] = float(cost)

    finite_matrix = _finite_cost_matrix(matrix)
    worst_costs = np.max(finite_matrix, axis=1)
    minimax_index = int(np.argmin(worst_costs))
    scale = max(float(np.nanmedian(np.abs(finite_matrix))), 1e-6)
    temperature = max(float(args.game_softmax_temperature) * scale, 1e-8)
    soft_values = np.asarray(
        [
            temperature * (_logsumexp(row / temperature) - math.log(max(len(row), 1)))
            for row in finite_matrix
        ],
        dtype=np.float64,
    )
    soft_index = int(np.argmin(soft_values))
    strategy, mixed_value, solver = solve_defender_mixed_strategy(finite_matrix)
    nash_candidate = blended_candidate_from_strategy(candidates, strategy)

    benchmark_episode = variant_episodes[0]
    benchmark_config = dict(variants[0]["config"])
    nash_benchmark_cost, nash_benchmark_result = plan_cost(
        benchmark_episode,
        nash_candidate["weights"],
        float(nash_candidate["lambda_uncertainty"]),
        benchmark_config,
    )

    support = np.flatnonzero(strategy > 1e-4)
    entropy = -float(np.sum(strategy[support] * np.log(np.clip(strategy[support], 1e-12, 1.0)))) if support.size else 0.0
    diagnostics = {
        "num_game_attack_variants": int(len(variants)),
        "game_attack_variant_ids": ";".join(str(variant["variant_id"]) for variant in variants),
        "game_minimax_worst_cost": float(worst_costs[minimax_index]),
        "game_soft_stackelberg_value": float(soft_values[soft_index]),
        "game_nash_value": float(mixed_value),
        "game_nash_solver": solver,
        "game_nash_support_size": int(support.size),
        "game_nash_entropy": entropy,
        "game_nash_top_candidate": str(candidates[int(np.argmax(strategy))].get("candidate_id", "")),
    }

    teacher_specs = [
        {
            "method": "game_minimax_teacher",
            "candidate": candidates[minimax_index],
            "result": None,
            "benchmark_cost": float(finite_matrix[minimax_index, 0]),
            "source_method": str(candidates[minimax_index]["method"]),
            "game_value": float(worst_costs[minimax_index]),
        },
        {
            "method": "game_soft_stackelberg_teacher",
            "candidate": candidates[soft_index],
            "result": None,
            "benchmark_cost": float(finite_matrix[soft_index, 0]),
            "source_method": str(candidates[soft_index]["method"]),
            "game_value": float(soft_values[soft_index]),
        },
        {
            "method": "game_nash_blend_teacher",
            "candidate": nash_candidate,
            "result": nash_benchmark_result,
            "benchmark_cost": float(nash_benchmark_cost),
            "source_method": "mixed_strategy_blend",
            "game_value": float(mixed_value),
        },
    ]

    rows: list[dict[str, Any]] = []
    for spec in teacher_specs:
        result = spec["result"]
        if result is None:
            _, result = plan_cost(
                benchmark_episode,
                spec["candidate"]["weights"],
                lambda_uncertainty=float(spec["candidate"]["lambda_uncertainty"]),
                env_attack=benchmark_config,
            )
        rows.append(
            {
                "method": spec["method"],
                "candidate": spec["candidate"],
                "result": result,
                "benchmark_cost": float(spec["benchmark_cost"]),
                "source_method": str(spec["source_method"]),
                "game_value": float(spec["game_value"]),
            }
        )
    return rows, diagnostics


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


def real_episodes_from_run(run_config: dict[str, Any], count: int) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    level_config = dict(run_config["level_config"])
    command_args = dict(run_config["command_args"])
    seed = int(command_args.get("seed", 0))
    layers_path = resolve_path(level_config["map_source"])
    raw = np.load(layers_path)
    map_size = int(raw["layer_distance"].shape[0])
    raw.close()

    splits = {key: resolve_path(value) for key, value in dict(run_config["splits"]).items()}
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


def episodes_from_run(run_config: dict[str, Any], count: int) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    if "map_source" in dict(run_config["level_config"]):
        return real_episodes_from_run(run_config, count)
    return synthetic_episodes_from_run(run_config, count)


def summarize_method(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    values = [row for row in rows if row["method"] == method]
    costs = np.asarray([row["attacked_cost"] for row in values], dtype=np.float64)
    success = np.asarray([row["success"] for row in values], dtype=np.float64)
    exposure = np.asarray([row["attacked_cell_exposure_ratio"] for row in values], dtype=np.float64)
    return {
        f"{method}_cost": float(np.nanmean(costs)),
        f"{method}_success_rate": float(np.nanmean(success)),
        f"{method}_exposure_ratio": float(np.nanmean(exposure)),
    }


def evaluate_run(run_dir: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    level, difficulty = run_label_from_dir(run_dir)
    run_config = read_json(run_dir / "run_config.json")
    env_attack = dict(run_config["environment_attack"])
    run_seed = int(dict(run_config["command_args"]).get("seed", args.seed))
    map_size, episodes_by_domain = episodes_from_run(run_config, int(args.num_episodes))

    nominal_policy = load_model(run_dir / "checkpoints" / "checkpoint_nominal.pt", "auto")
    recovery_checkpoint = final_recovery_checkpoint(run_dir)
    recovery_policy = load_model(recovery_checkpoint, "auto")

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for eval_domain, (map_seed, episodes) in episodes_by_domain.items():
        domain_rows: list[dict[str, Any]] = []
        clean_costs: list[float] = []
        candidate_counts: list[int] = []
        for episode_index, episode in enumerate(episodes):
            attack_rng = np.random.default_rng(run_seed + 300_000 + episode_index)
            attacked_episode = apply_environment_attack_to_episode(episode, env_attack, attack_rng)
            candidate_rng = np.random.default_rng(
                run_seed
                + 700_000
                + 10_000 * LEVEL_ORDER.get(level, 9)
                + 1_000 * DIFFICULTY_ORDER.get(difficulty, 9)
                + episode_index
            )
            candidates, nominal_cfg, recovery_cfg = build_candidates(
                attacked_episode,
                map_size,
                nominal_policy,
                recovery_policy,
                candidate_rng,
                args,
            )
            candidate_counts.append(len(candidates))

            clean_cfg = policy_action_config(nominal_policy, episode, map_size)
            clean_cost, _ = plan_cost(
                episode,
                clean_cfg["weights"],
                float(clean_cfg["lambda_uncertainty"]),
                {},
            )
            clean_costs.append(clean_cost)

            nominal_cost, nominal_result = plan_cost(
                attacked_episode,
                nominal_cfg["weights"],
                float(nominal_cfg["lambda_uncertainty"]),
                env_attack,
            )
            recovery_cost, recovery_result = plan_cost(
                attacked_episode,
                recovery_cfg["weights"],
                float(recovery_cfg["lambda_uncertainty"]),
                env_attack,
            )

            evaluated_candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
            for candidate in candidates:
                cost, result = plan_cost(
                    attacked_episode,
                    candidate["weights"],
                    float(candidate["lambda_uncertainty"]),
                    env_attack,
                )
                evaluated_candidates.append((cost, candidate, result))
            oracle_cost, oracle_candidate, oracle_result = min(evaluated_candidates, key=lambda item: item[0])

            methods = [
                ("nominal_ppo", nominal_cost, nominal_cfg, nominal_result, "nominal_ppo"),
                ("final_recovery_ppo", recovery_cost, recovery_cfg, recovery_result, "final_recovery_ppo"),
                ("oracle_best_of_candidates", oracle_cost, oracle_candidate, oracle_result, str(oracle_candidate["method"])),
            ]
            game_diagnostics: dict[str, Any] = {}
            if not bool(args.disable_game_theory):
                game_teachers, game_diagnostics = evaluate_game_teachers(
                    episode,
                    candidates,
                    env_attack,
                    run_seed,
                    episode_index,
                    args,
                )
                for teacher in game_teachers:
                    methods.append(
                        (
                            str(teacher["method"]),
                            float(teacher["benchmark_cost"]),
                            teacher["candidate"],
                            teacher["result"],
                            str(teacher["source_method"]),
                        )
                    )
            for method, cost, config, result, source in methods:
                row = {
                    "level": level,
                    "difficulty": difficulty,
                    "run_dir": str(run_dir),
                    "eval_domain": eval_domain,
                    "map_pool_seed": int(map_seed),
                    "episode_index": int(episode_index),
                    "method": method,
                    "oracle_source_method": source if method == "oracle_best_of_candidates" else "",
                    "game_source_method": source if str(method).startswith("game_") else "",
                    "oracle_candidate_id": str(config.get("candidate_id", "")) if method == "oracle_best_of_candidates" else "",
                    "game_candidate_id": str(config.get("candidate_id", "")) if str(method).startswith("game_") else "",
                    "clean_nominal_cost": float(clean_cost),
                    "attacked_cost": float(cost),
                    "success": 1.0 if bool(result.get("success", False)) else 0.0,
                    "path_length": float(result.get("path_length", np.nan)),
                    "attacked_cell_exposure_ratio": float(result.get("attacked_cell_exposure_ratio", 0.0)),
                    "lambda_uncertainty": float(config["lambda_uncertainty"]),
                    "candidate_count": int(len(candidates)),
                    "recovery_checkpoint": str(recovery_checkpoint),
                    **game_diagnostics,
                }
                weights = np.asarray(config["weights"], dtype=np.float32)
                for index, name in enumerate(OBJECTIVE_NAMES):
                    row[f"weight_{name}"] = float(weights[index])
                domain_rows.append(row)
                detail_rows.append(row)

        domain_summary: dict[str, Any] = {
            "level": level,
            "difficulty": difficulty,
            "eval_domain": eval_domain,
            "map_pool_seed": int(map_seed),
            "num_episodes": int(len(episodes)),
            "mean_candidate_count": float(np.mean(candidate_counts)),
            "clean_nominal_cost": float(np.nanmean(clean_costs)),
            "run_dir": str(run_dir),
            "recovery_checkpoint": str(recovery_checkpoint),
        }
        method_names = ["nominal_ppo", "final_recovery_ppo", "oracle_best_of_candidates"]
        if not bool(args.disable_game_theory):
            method_names.extend(["game_minimax_teacher", "game_soft_stackelberg_teacher", "game_nash_blend_teacher"])
        for method in method_names:
            domain_summary.update(summarize_method(domain_rows, method))
            method_cost = float(domain_summary[f"{method}_cost"])
            domain_summary[f"{method}_index"] = 100.0 * domain_summary["clean_nominal_cost"] / max(method_cost, 1e-12)

        domain_summary["oracle_minus_recovery_index"] = (
            domain_summary["oracle_best_of_candidates_index"] - domain_summary["final_recovery_ppo_index"]
        )
        domain_summary["oracle_minus_nominal_index"] = (
            domain_summary["oracle_best_of_candidates_index"] - domain_summary["nominal_ppo_index"]
        )
        for method in ("game_minimax_teacher", "game_soft_stackelberg_teacher", "game_nash_blend_teacher"):
            if f"{method}_index" in domain_summary:
                domain_summary[f"{method}_minus_recovery_index"] = (
                    domain_summary[f"{method}_index"] - domain_summary["final_recovery_ppo_index"]
                )
        domain_summary["oracle_gap_vs_recovery_pct"] = (
            (domain_summary["final_recovery_ppo_cost"] - domain_summary["oracle_best_of_candidates_cost"])
            / max(abs(domain_summary["final_recovery_ppo_cost"]), 1e-12)
        )
        domain_summary["oracle_gap_vs_nominal_pct"] = (
            (domain_summary["nominal_ppo_cost"] - domain_summary["oracle_best_of_candidates_cost"])
            / max(abs(domain_summary["nominal_ppo_cost"]), 1e-12)
        )
        domain_frame = pd.DataFrame(domain_rows)
        paired = domain_frame.pivot_table(
            index="episode_index",
            columns="method",
            values="attacked_cost",
            aggfunc="first",
        )
        domain_summary["episode_oracle_beats_recovery_rate"] = float(
            (paired["oracle_best_of_candidates"] < paired["final_recovery_ppo"]).mean()
        )
        for method in ("game_minimax_teacher", "game_soft_stackelberg_teacher", "game_nash_blend_teacher"):
            if method in paired.columns:
                domain_summary[f"episode_{method}_beats_recovery_rate"] = float(
                    (paired[method] < paired["final_recovery_ppo"]).mean()
                )
        summary_rows.append(domain_summary)

    return summary_rows, detail_rows


def panel_summary(domain_summary: pd.DataFrame, baseline_summary: Path) -> pd.DataFrame:
    grouped = (
        domain_summary.groupby(["level", "difficulty"], as_index=False)
        .agg(
            num_domains=("eval_domain", "nunique"),
            num_episodes=("num_episodes", "sum"),
            mean_candidate_count=("mean_candidate_count", "mean"),
            clean_nominal_cost=("clean_nominal_cost", "mean"),
            nominal_ppo_index=("nominal_ppo_index", "mean"),
            final_recovery_ppo_index=("final_recovery_ppo_index", "mean"),
            oracle_best_of_candidates_index=("oracle_best_of_candidates_index", "mean"),
            oracle_minus_recovery_index=("oracle_minus_recovery_index", "mean"),
            oracle_minus_nominal_index=("oracle_minus_nominal_index", "mean"),
            oracle_gap_vs_recovery_pct=("oracle_gap_vs_recovery_pct", "mean"),
            oracle_gap_vs_nominal_pct=("oracle_gap_vs_nominal_pct", "mean"),
            episode_oracle_beats_recovery_rate=("episode_oracle_beats_recovery_rate", "mean"),
        )
    )
    grouped["oracle_beats_recovery"] = grouped["oracle_minus_recovery_index"] > 0.0
    for method in ("game_minimax_teacher", "game_soft_stackelberg_teacher", "game_nash_blend_teacher"):
        extra_columns = [
            column
            for column in (
                f"{method}_index",
                f"{method}_minus_recovery_index",
                f"episode_{method}_beats_recovery_rate",
            )
            if column in domain_summary.columns
        ]
        if extra_columns:
            extra = domain_summary.groupby(["level", "difficulty"], as_index=False)[extra_columns].mean()
            grouped = grouped.merge(extra, on=["level", "difficulty"], how="left")
            grouped[f"{method}_beats_recovery"] = grouped[f"{method}_minus_recovery_index"] > 0.0

    baseline_path = resolve_path(baseline_summary)
    if baseline_path.exists():
        baseline = pd.read_csv(baseline_path)
        baseline = baseline[baseline["algorithm"] == "ppo"][
            ["level", "difficulty", "attack_shock_index", "final_recovery_index", "best_recovery_index"]
        ].rename(
            columns={
                "attack_shock_index": "ppo_5seed_attack_shock_index",
                "final_recovery_index": "ppo_5seed_final_recovery_index",
                "best_recovery_index": "ppo_5seed_best_recovery_index",
            }
        )
        grouped = grouped.merge(baseline, on=["level", "difficulty"], how="left")
        grouped["oracle_minus_ppo_5seed_final_index"] = (
            grouped["oracle_best_of_candidates_index"] - grouped["ppo_5seed_final_recovery_index"]
        )
        for method in ("game_minimax_teacher", "game_soft_stackelberg_teacher", "game_nash_blend_teacher"):
            if f"{method}_index" in grouped.columns:
                grouped[f"{method}_minus_ppo_5seed_final_index"] = (
                    grouped[f"{method}_index"] - grouped["ppo_5seed_final_recovery_index"]
                )

    grouped["_level_order"] = grouped["level"].map(LEVEL_ORDER)
    grouped["_difficulty_order"] = grouped["difficulty"].map(DIFFICULTY_ORDER)
    return grouped.sort_values(["_level_order", "_difficulty_order"]).drop(
        columns=["_level_order", "_difficulty_order"]
    )


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.25",
            "axes.labelcolor": "0.15",
            "xtick.color": "0.15",
            "ytick.color": "0.15",
            "font.size": 10,
        }
    )


def plot_panel_summary(frame: pd.DataFrame, output_dir: Path) -> None:
    setup_matplotlib()
    labels = [f"{row.level}\n{row.difficulty}" for row in frame.itertuples()]
    x = np.arange(len(frame))
    bar_specs = [
        ("nominal_ppo_index", "PPO nominal under attack", "#c23b22"),
        ("final_recovery_ppo_index", "PPO final recovery", "#2f6b9a"),
    ]
    if "game_minimax_teacher_index" in frame.columns:
        bar_specs.append(("game_minimax_teacher_index", "Minimax game teacher", "#7a3db8"))
    if "game_nash_blend_teacher_index" in frame.columns:
        bar_specs.append(("game_nash_blend_teacher_index", "Nash mixed/blended teacher", "#c06b2c"))
    bar_specs.append(("oracle_best_of_candidates_index", "Single-attack oracle upper bound", "#2a7f62"))
    width = min(0.72 / max(len(bar_specs), 1), 0.18)
    offsets = (np.arange(len(bar_specs)) - (len(bar_specs) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    for offset, (column, label, color) in zip(offsets, bar_specs):
        ax.bar(x + offset, frame[column], width=width, label=label, color=color, alpha=0.82)
    if "ppo_5seed_final_recovery_index" in frame.columns:
        ax.scatter(
            x,
            frame["ppo_5seed_final_recovery_index"],
            color="black",
            marker="D",
            s=28,
            label="PPO 5-seed final",
            zorder=5,
        )
    ax.axhline(100.0, color="0.35", linestyle="--", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("performance index (clean nominal = 100)")
    ax.set_title("Planner teacher upper bounds under benchmark attack")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.20), frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_recovery_oracle_upper_bound_indices.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    headroom_specs = []
    if "game_minimax_teacher_minus_recovery_index" in frame.columns:
        headroom_specs.append(("game_minimax_teacher_minus_recovery_index", "Minimax game teacher", "#7a3db8"))
    if "game_nash_blend_teacher_minus_recovery_index" in frame.columns:
        headroom_specs.append(("game_nash_blend_teacher_minus_recovery_index", "Nash blend teacher", "#c06b2c"))
    headroom_specs.append(("oracle_minus_recovery_index", "Single-attack oracle", "#2a7f62"))
    width = min(0.72 / max(len(headroom_specs), 1), 0.22)
    offsets = (np.arange(len(headroom_specs)) - (len(headroom_specs) - 1) / 2.0) * width
    for offset, (column, label, color) in zip(offsets, headroom_specs):
        ax.bar(x + offset, frame[column], width=width, color=color, alpha=0.82, label=label)
    ax.axhline(0.0, color="0.25", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("teacher - final recovery index")
    ax.set_title("Headroom beyond PPO recovery")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_recovery_oracle_headroom.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    wins = int((frame["oracle_minus_recovery_index"] > 0.0).sum())
    total = int(len(frame))
    clear = int((frame["oracle_minus_recovery_index"] > 2.0).sum())
    mean_gain = float(frame["oracle_minus_recovery_index"].mean())
    has_game = "game_minimax_teacher_minus_recovery_index" in frame.columns
    game_wins = int((frame["game_minimax_teacher_minus_recovery_index"] > 0.0).sum()) if has_game else 0
    game_clear = int((frame["game_minimax_teacher_minus_recovery_index"] > 2.0).sum()) if has_game else 0
    game_mean_gain = float(frame["game_minimax_teacher_minus_recovery_index"].mean()) if has_game else math.nan
    lines = [
        "# Recovery Planner Oracle and Game-Teacher Upper-Bound Test",
        "",
        "This is a diagnostic upper bound, not a deployable algorithm result.",
        "The single-attack oracle chooses the lowest true attacked scalar cost from a candidate set per episode.",
        "The game teacher solves a local zero-sum defender-vs-attacker planner game over attack variants.",
        "",
        f"- Seed evaluated: {int(args.seed)}",
        f"- Episodes per eval domain cap: {int(args.num_episodes)}",
        f"- Random candidates per episode: {int(args.num_random_candidates)}",
        f"- Local candidates per policy neighborhood: {int(args.num_local_candidates)}",
        f"- Game attack variant mode: {args.game_attack_variant_mode if not args.disable_game_theory else 'disabled'}",
        f"- Oracle beats final PPO recovery in {wins}/{total} panels.",
        f"- Oracle gains more than 2 index points in {clear}/{total} panels.",
        f"- Mean oracle minus final-recovery index: {mean_gain:.2f}",
    ]
    if has_game:
        lines.extend(
            [
                f"- Minimax game teacher beats final PPO recovery in {game_wins}/{total} panels.",
                f"- Minimax game teacher gains more than 2 index points in {game_clear}/{total} panels.",
                f"- Mean minimax game teacher minus final-recovery index: {game_mean_gain:.2f}",
            ]
        )
    lines.extend(
        [
            "",
            "| Level | Difficulty | PPO attack | PPO final | Minimax game | Nash blend | Single-attack oracle | Game - final | Oracle - final |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in frame.itertuples():
        minimax_index = getattr(row, "game_minimax_teacher_index", math.nan)
        nash_index = getattr(row, "game_nash_blend_teacher_index", math.nan)
        minimax_gain = getattr(row, "game_minimax_teacher_minus_recovery_index", math.nan)
        lines.append(
            f"| {row.level} | {row.difficulty} | "
            f"{row.nominal_ppo_index:.2f} | {row.final_recovery_ppo_index:.2f} | "
            f"{minimax_index:.2f} | {nash_index:.2f} | "
            f"{row.oracle_best_of_candidates_index:.2f} | "
            f"{minimax_gain:.2f} | {row.oracle_minus_recovery_index:.2f} |"
        )

    lines.extend(
        [
            "",
            "Interpretation:",
            "- The single-attack oracle is the optimistic headroom on the benchmark attack.",
            "- The minimax game teacher is stricter: it chooses planner weights by minimizing worst-case cost over local attack variants.",
            "- If the minimax teacher remains above PPO recovery, the next algorithm should distill this game-theoretic best response during recovery.",
        ]
    )
    (output_dir / "oracle_upper_bound_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.quick:
        args.num_episodes = min(int(args.num_episodes), 8)
        args.num_random_candidates = min(int(args.num_random_candidates), 24)
        args.num_local_candidates = min(int(args.num_local_candidates), 8)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_run_dirs(args.runs_root, int(args.seed))
    if not run_dirs:
        raise FileNotFoundError(f"no PPO seed{args.seed} run dirs found under {args.runs_root}")

    all_domain_rows: list[dict[str, Any]] = []
    all_detail_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        level, difficulty = run_label_from_dir(run_dir)
        print(f"Evaluating {level}/{difficulty}: {run_dir}", flush=True)
        domain_rows, detail_rows = evaluate_run(run_dir, args)
        all_domain_rows.extend(domain_rows)
        all_detail_rows.extend(detail_rows)

    domain_frame = pd.DataFrame(all_domain_rows)
    detail_frame = pd.DataFrame(all_detail_rows)
    panel_frame = panel_summary(domain_frame, args.baseline_summary)

    domain_frame.to_csv(output_dir / "oracle_domain_summary.csv", index=False)
    detail_frame.to_csv(output_dir / "oracle_episode_details.csv", index=False)
    panel_frame.to_csv(output_dir / "oracle_panel_summary.csv", index=False)
    plot_panel_summary(panel_frame, output_dir)
    write_report(panel_frame, output_dir, args)

    print_columns = [
        "level",
        "difficulty",
        "nominal_ppo_index",
        "final_recovery_ppo_index",
    ]
    for column in (
        "game_minimax_teacher_index",
        "game_minimax_teacher_minus_recovery_index",
        "game_nash_blend_teacher_index",
    ):
        if column in panel_frame.columns:
            print_columns.append(column)
    print_columns.extend(
        [
            "oracle_best_of_candidates_index",
            "oracle_minus_recovery_index",
            "episode_oracle_beats_recovery_rate",
        ]
    )
    print(panel_frame[print_columns].to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved oracle outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
