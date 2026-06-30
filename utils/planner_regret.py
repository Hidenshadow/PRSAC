"""Legacy planner-regret utilities used by optional PR-PPO code paths.

The current paper-facing path does not enable PR-PPO, but train_cleanrl_ppo.py
keeps optional PR flags for backward-compatible checkpoints and diagnostics.
This module keeps those optional paths importable without carrying the old
runner scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np


@dataclass
class CandidateActionConfig:
    num_candidates: int = 16
    local_sigma: float = 0.10
    local_sigmas: tuple[float, ...] = ()
    num_random_candidates: int = 4
    include_policy_action: bool = True
    include_nominal_action: bool = True
    include_zero_delta: bool = True
    include_axis_perturbations: bool = True
    risk_dim_indices: list[int] | tuple[int, ...] = ()
    risk_local_sigma: float = 0.20
    include_risk_axis_perturbations: bool = True
    include_risk_block_perturbations: bool = True
    zero_action_value: float = 0.5
    dedup_tol: float = 1e-6


@dataclass
class PlannerRegretTargetConfig:
    target_type: str = "soft"
    soft_temperature: float = 0.10
    cost_normalization: str = "policy_abs"
    regret_weight_max: float = 2.0
    cost_epsilon: float = 1e-8
    hard_regret_threshold: float = 0.02
    soft_regret_threshold: float = 0.005
    random_target_control: bool = False


@dataclass
class CPAConfig:
    temperature: float = 0.03
    min_positive_adv: float = 0.001
    weighting: str = "softmax"
    sample_weight: str = "max_adv"
    regret_weight_max: float = 2.0
    cost_epsilon: float = 1e-8


@dataclass
class PairwisePreferenceConfig:
    adv_temperature: float = 0.03
    pref_temperature: float = 1.0
    min_positive_adv: float = 0.001
    regret_weight_max: float = 2.0
    cost_epsilon: float = 1e-8
    z_clip: float = 10.0


def _unwrap_env(env: Any) -> Any:
    current = env
    seen = set()
    while hasattr(current, "env") and id(current) not in seen:
        seen.add(id(current))
        current = current.env
    return getattr(env, "unwrapped", current)


def _env_attr(env: Any, name: str, default: Any) -> Any:
    base = _unwrap_env(env)
    return getattr(base, name, default)


def _current_base_episode(env: Any) -> Any:
    base = _unwrap_env(env)
    episode = base.current_episode
    true_costmap = getattr(episode, "true_costmap", None)
    if true_costmap is not None:
        return replace(episode, costmap=true_costmap, true_costmap=None, confidence_layers=None)
    return episode


def _candidate_cost(result: dict[str, Any], reward_cost_key: str, failure_cost: float) -> float:
    value = result.get(reward_cost_key, result.get("attacked_scalar_cost", result.get("scalar_cost", failure_cost)))
    try:
        cost = float(value)
    except (TypeError, ValueError):
        cost = float(failure_cost)
    return cost if np.isfinite(cost) else float(failure_cost)


def _action_to_result(
    env: Any,
    episode: Any,
    action: np.ndarray,
    reward_cost_key: str,
    failure_cost: float,
    attack_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from utils.metrics import action_to_planning_weights, action_to_uncertainty_lambda, plan_with_weights

    attack_config = dict(attack_config or {})
    weights = action_to_planning_weights(
        episode,
        action,
        action_mode=str(_env_attr(env, "action_mode", "preference_delta")),
        action_gain=float(_env_attr(env, "action_gain", 2.0)),
    )
    lambda_uncertainty = action_to_uncertainty_lambda(
        action,
        max_uncertainty_lambda=float(_env_attr(env, "max_uncertainty_lambda", 1.0)),
    )
    result = plan_with_weights(
        episode,
        weights,
        lambda_uncertainty=lambda_uncertainty,
        allow_diagonal=bool(_env_attr(env, "allow_diagonal", True)),
        attack_budget_fraction=float(
            attack_config.get("attack_budget_fraction", _env_attr(env, "attack_budget_fraction", 0.18))
        ),
        attack_strength=float(attack_config.get("attack_strength", _env_attr(env, "attack_strength", 1.0))),
        attacker_temperature=float(
            attack_config.get("attacker_temperature", _env_attr(env, "attacker_temperature", 0.5))
        ),
        attacker_response=str(attack_config.get("attacker_response", _env_attr(env, "attacker_response", "zscore_topk"))),
        attacker_top_fraction=float(
            attack_config.get("attacker_top_fraction", _env_attr(env, "attacker_top_fraction", 0.15))
        ),
        attacker_sharpness=float(
            attack_config.get("attacker_sharpness", _env_attr(env, "attacker_sharpness", 3.0))
        ),
    )
    result = dict(result)
    result["true_attacked_cost"] = _candidate_cost(result, reward_cost_key, failure_cost)
    result["candidate_action"] = np.asarray(action, dtype=np.float32).tolist()
    result["failure_flag"] = False
    return result


def evaluate_counterfactual_action_for_env(
    env: Any,
    action: np.ndarray,
    reward_cost_key: str = "scalar_cost",
    failure_cost: float = 1e6,
) -> dict[str, Any]:
    try:
        episode = _unwrap_env(env).current_episode
        return _action_to_result(env, episode, action, reward_cost_key, failure_cost)
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return {
            "candidate_action": np.asarray(action, dtype=np.float32).tolist(),
            "path_valid": False,
            "true_attacked_cost": float(failure_cost),
            "planner_internal_cost": float(failure_cost),
            "failure_flag": True,
            "failure_reason": str(exc),
        }


def evaluate_attack_variant_costs_for_env(
    env: Any,
    action: np.ndarray,
    environment_attack: dict[str, Any] | None,
    reward_cost_key: str = "scalar_cost",
    max_attack_variants: int = 6,
    rng: np.random.Generator | None = None,
    failure_cost: float = 1e6,
) -> dict[str, Any]:
    """Evaluate one action on the current base episode under attack variants."""

    from envs.attack_wrappers import apply_environment_attack_to_episode

    action_array = np.asarray(action, dtype=np.float32)
    variants = attack_variants_from_config(environment_attack, max_variants=max_attack_variants)
    if not variants:
        result = evaluate_counterfactual_action_for_env(
            env,
            action_array,
            reward_cost_key=reward_cost_key,
            failure_cost=failure_cost,
        )
        return {
            "variant_ids": ["current"],
            "costs": np.asarray([float(result.get("true_attacked_cost", failure_cost))], dtype=np.float64),
            "valid_mask": np.asarray([not bool(result.get("failure_flag", False))], dtype=bool),
            "failure_count": int(bool(result.get("failure_flag", False))),
            "results": [result],
        }

    random_state = rng if rng is not None else np.random.default_rng()
    base_episode = _current_base_episode(env)
    variant_ids: list[str] = []
    costs: list[float] = []
    valid: list[bool] = []
    results: list[dict[str, Any]] = []
    failure_count = 0
    for index, variant in enumerate(variants):
        variant_id = str(variant.get("variant_id", f"variant_{index}"))
        variant_config = dict(variant.get("config", {}))
        variant_ids.append(variant_id)
        try:
            episode = apply_environment_attack_to_episode(base_episode, variant_config, random_state)
            result = _action_to_result(
                env,
                episode,
                action_array,
                reward_cost_key,
                failure_cost,
                variant_config,
            )
            cost = float(result["true_attacked_cost"])
            is_valid = bool(np.isfinite(cost)) and not bool(result.get("failure_flag", False))
        except Exception as exc:
            result = {
                "candidate_action": action_array.tolist(),
                "path_valid": False,
                "true_attacked_cost": float(failure_cost),
                "planner_internal_cost": float(failure_cost),
                "failure_flag": True,
                "failure_reason": str(exc),
            }
            cost = float(failure_cost)
            is_valid = False
            failure_count += 1
        costs.append(cost)
        valid.append(is_valid)
        results.append(result)
    return {
        "variant_ids": variant_ids,
        "costs": np.asarray(costs, dtype=np.float64),
        "valid_mask": np.asarray(valid, dtype=bool),
        "failure_count": int(failure_count),
        "results": results,
    }


def attack_variants_from_config(environment_attack: dict[str, Any] | None, max_variants: int = 6) -> list[dict[str, Any]]:
    config = dict(environment_attack or {})
    if not config.get("enabled", False):
        return []
    if str(config.get("type", "")) == "env_attack_mixture":
        variants = []
        for index, variant in enumerate(config.get("variants", [])):
            if not isinstance(variant, dict):
                continue
            variant_config = dict(variant.get("config", {}))
            if not variant_config:
                continue
            variants.append(
                {
                    "variant_id": str(variant.get("variant_id", f"variant_{index}")),
                    "config": variant_config,
                }
            )
    else:
        variants = [{"variant_id": "benchmark", "config": config}]
    if max_variants > 0:
        variants = variants[: int(max_variants)]
    return variants


def _logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    max_value = float(np.max(values))
    return float(max_value + np.log(np.exp(values - max_value).sum()))


def robust_candidate_costs(
    cost_matrix: np.ndarray,
    mode: str = "minimax",
    softmax_temperature: float = 0.08,
) -> np.ndarray:
    matrix = np.asarray(cost_matrix, dtype=np.float64)
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return np.ones(matrix.shape[0], dtype=np.float64)
    fallback = float(np.max(finite) + max(abs(float(np.max(finite))), 1.0))
    matrix = np.where(np.isfinite(matrix), matrix, fallback)
    if str(mode) == "soft_stackelberg":
        scale = max(float(np.median(np.abs(matrix))), 1e-6)
        temperature = max(float(softmax_temperature) * scale, 1e-8)
        return np.asarray(
            [temperature * (_logsumexp(row / temperature) - np.log(max(row.size, 1))) for row in matrix],
            dtype=np.float64,
        )
    return np.max(matrix, axis=1)


def build_game_planner_regret_target(
    env: Any,
    candidates: np.ndarray,
    policy_action: np.ndarray,
    environment_attack: dict[str, Any] | None,
    reward_cost_key: str,
    target_config: PlannerRegretTargetConfig,
    rng: np.random.Generator,
    mode: str = "minimax",
    max_attack_variants: int = 6,
    softmax_temperature: float = 0.08,
    failure_cost: float = 1e6,
) -> dict[str, Any]:
    from envs.attack_wrappers import apply_environment_attack_to_episode

    actions = np.asarray(candidates, dtype=np.float32)
    variants = attack_variants_from_config(environment_attack, max_variants=max_attack_variants)
    if not variants:
        evaluator = PlannerCounterfactualEvaluator(reward_cost_key, failure_cost=failure_cost)
        eval_results = evaluator.evaluate_many(env, actions)
        costs = np.asarray([float(item["true_attacked_cost"]) for item in eval_results], dtype=np.float64)
        target = build_planner_regret_target(actions, costs, policy_action, target_config, rng=rng)
        target["game_num_attack_variants"] = 0
        target["game_teacher_mode"] = "disabled"
        target["planner_failure_count"] = int(sum(bool(item.get("failure_flag", False)) for item in eval_results))
        return target

    base_episode = _current_base_episode(env)
    cost_matrix = np.full((actions.shape[0], len(variants)), float(failure_cost), dtype=np.float64)
    failure_count = 0
    variant_ids: list[str] = []
    for variant_index, variant in enumerate(variants):
        variant_ids.append(str(variant.get("variant_id", f"variant_{variant_index}")))
        variant_config = dict(variant.get("config", {}))
        try:
            episode = apply_environment_attack_to_episode(base_episode, variant_config, rng)
        except Exception:
            episode = base_episode
            failure_count += int(actions.shape[0])
        for candidate_index, action in enumerate(actions):
            try:
                result = _action_to_result(env, episode, action, reward_cost_key, failure_cost, variant_config)
                cost_matrix[candidate_index, variant_index] = float(result["true_attacked_cost"])
            except Exception:
                failure_count += 1

    robust_costs = robust_candidate_costs(cost_matrix, mode=mode, softmax_temperature=softmax_temperature)
    target = build_planner_regret_target(actions, robust_costs, policy_action, target_config, rng=rng)
    target["game_teacher_mode"] = str(mode)
    target["game_num_attack_variants"] = int(len(variants))
    target["game_attack_variant_ids"] = ";".join(variant_ids)
    target["game_candidate_cost_matrix"] = cost_matrix.astype(float).tolist()
    target["game_robust_candidate_costs"] = robust_costs.astype(float).tolist()
    target["game_minimax_value"] = float(np.min(np.max(cost_matrix, axis=1)))
    target["planner_failure_count"] = int(failure_count)
    return target


def planning_config_to_action(
    episode: Any,
    weights: np.ndarray,
    lambda_uncertainty: float,
    action_dim: int,
    action_mode: str,
    action_gain: float,
    max_uncertainty_lambda: float,
) -> np.ndarray:
    from utils.metrics import normalize_weights

    target_weights = normalize_weights(np.asarray(weights, dtype=np.float32))
    action = np.zeros(int(action_dim), dtype=np.float32)
    if str(action_mode) == "direct":
        action[: target_weights.shape[0]] = target_weights
    else:
        mission = np.asarray(getattr(episode, "mission_priority", np.ones_like(target_weights)), dtype=np.float32)
        mission = normalize_weights(np.maximum(mission, 1e-6))
        logits = np.log(np.maximum(target_weights, 1e-8)) - np.log(np.maximum(mission, 1e-8))
        logits = logits - float(np.mean(logits))
        action[: target_weights.shape[0]] = np.clip(0.5 * (logits / max(float(action_gain), 1e-8) + 1.0), 0.0, 1.0)
    if int(action_dim) > target_weights.shape[0]:
        action[target_weights.shape[0]] = float(
            np.clip(float(lambda_uncertainty) / max(float(max_uncertainty_lambda), 1e-8), 0.0, 1.0)
        )
    return action.astype(np.float32)


def generate_structured_candidate_actions(
    env: Any,
    action_low: np.ndarray,
    action_high: np.ndarray,
    count: int,
    action_mode: str,
    action_gain: float,
    max_uncertainty_lambda: float,
    dedup_tol: float = 1e-6,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    from utils.metrics import candidate_planner_configs

    if int(count) <= 0:
        return np.zeros((0, len(action_low)), dtype=np.float32), []
    episode = _current_base_episode(env)
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
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    for name in names[: int(count)]:
        config = configs[name]
        action = planning_config_to_action(
            episode,
            np.asarray(config["weights"], dtype=np.float32),
            float(config["lambda_uncertainty"]),
            action_dim=int(low.shape[0]),
            action_mode=action_mode,
            action_gain=action_gain,
            max_uncertainty_lambda=max_uncertainty_lambda,
        )
        before = len(actions)
        _append_unique(actions, action, low, high, float(dedup_tol))
        if len(actions) > before:
            meta.append({"kind": f"structured:{name}"})
    if not actions:
        return np.zeros((0, len(action_low)), dtype=np.float32), []
    return np.stack(actions, axis=0).astype(np.float32), meta


def merge_candidate_action_sets(
    candidates: np.ndarray,
    metadata: list[dict[str, Any]],
    extra_candidates: np.ndarray,
    extra_metadata: list[dict[str, Any]],
    action_low: np.ndarray,
    action_high: np.ndarray,
    dedup_tol: float = 1e-6,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    items = [np.asarray(action, dtype=np.float32) for action in np.asarray(candidates, dtype=np.float32)]
    meta = [dict(item) for item in metadata]
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    for index, action in enumerate(np.asarray(extra_candidates, dtype=np.float32)):
        before = len(items)
        _append_unique(items, action, low, high, float(dedup_tol))
        if len(items) > before:
            meta.append(dict(extra_metadata[index]) if index < len(extra_metadata) else {"kind": "extra"})
    return np.stack(items, axis=0).astype(np.float32), meta


def _append_unique(items: list[np.ndarray], action: np.ndarray, low: np.ndarray, high: np.ndarray, tol: float) -> None:
    clipped = np.clip(np.asarray(action, dtype=np.float32), low, high)
    if not any(np.allclose(clipped, item, atol=tol, rtol=0.0) for item in items):
        items.append(clipped)


def generate_candidate_actions(
    policy_action: np.ndarray,
    nominal_action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    config: CandidateActionConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    policy = np.asarray(policy_action, dtype=np.float32)
    nominal = np.asarray(nominal_action, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32)
    high = np.asarray(action_high, dtype=np.float32)
    dim = int(policy.shape[0])
    candidates: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []

    def add(action: np.ndarray, kind: str) -> None:
        before = len(candidates)
        _append_unique(candidates, action, low, high, float(config.dedup_tol))
        if len(candidates) > before:
            meta.append({"kind": kind})

    if config.include_policy_action:
        add(policy, "policy")
    if config.include_nominal_action:
        add(nominal, "nominal")
    if config.include_zero_delta:
        add(np.full(dim, float(config.zero_action_value), dtype=np.float32), "zero")

    sigmas = tuple(config.local_sigmas) if config.local_sigmas else (float(config.local_sigma),)
    if config.include_axis_perturbations:
        for sigma in sigmas:
            for axis in range(dim):
                delta = np.zeros(dim, dtype=np.float32)
                delta[axis] = float(sigma)
                add(policy + delta, f"axis+:{axis}:{sigma}")
                add(policy - delta, f"axis-:{axis}:{sigma}")

    risk_indices = [idx for idx in config.risk_dim_indices if 0 <= int(idx) < dim]
    if risk_indices and config.include_risk_axis_perturbations:
        for axis in risk_indices:
            delta = np.zeros(dim, dtype=np.float32)
            delta[int(axis)] = float(config.risk_local_sigma)
            add(policy + delta, f"risk_axis+:{axis}")
            add(policy - delta, f"risk_axis-:{axis}")
    if risk_indices and config.include_risk_block_perturbations:
        delta = np.zeros(dim, dtype=np.float32)
        delta[np.asarray(risk_indices, dtype=np.int64)] = float(config.risk_local_sigma)
        add(policy + delta, "risk_block+")
        add(policy - delta, "risk_block-")

    while len(candidates) < int(config.num_candidates):
        add(policy + rng.normal(0.0, float(config.local_sigma), size=dim).astype(np.float32), "random_policy")

    if len(candidates) > int(config.num_candidates):
        candidates = candidates[: int(config.num_candidates)]
        meta = meta[: int(config.num_candidates)]
    return np.stack(candidates, axis=0).astype(np.float32), meta


class PlannerCounterfactualEvaluator:
    def __init__(self, reward_cost_key: str = "scalar_cost", failure_cost: float = 1e6) -> None:
        self.reward_cost_key = str(reward_cost_key)
        self.failure_cost = float(failure_cost)

    def evaluate_many(
        self,
        env: Any,
        candidates: np.ndarray,
        candidate_metadata: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not hasattr(env, "evaluate_counterfactual_action"):
            return [
                {
                    "candidate_action": np.asarray(action, dtype=np.float32).tolist(),
                    "path_valid": False,
                    "true_attacked_cost": self.failure_cost,
                    "planner_internal_cost": self.failure_cost,
                    "failure_flag": True,
                    "failure_reason": "env lacks evaluate_counterfactual_action",
                }
                for action in candidates
            ]
        out = []
        for index, action in enumerate(candidates):
            try:
                result = env.evaluate_counterfactual_action(action)
                cost = float(result.get(self.reward_cost_key, result.get("scalar_cost", self.failure_cost)))
                out.append(
                    {
                        **dict(result),
                        "candidate_action": np.asarray(action, dtype=np.float32).tolist(),
                        "true_attacked_cost": cost,
                        "failure_flag": False,
                        "candidate_metadata": (candidate_metadata or [{}])[index] if candidate_metadata else {},
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive legacy path
                out.append(
                    {
                        "candidate_action": np.asarray(action, dtype=np.float32).tolist(),
                        "path_valid": False,
                        "true_attacked_cost": self.failure_cost,
                        "planner_internal_cost": self.failure_cost,
                        "failure_flag": True,
                        "failure_reason": str(exc),
                    }
                )
        return out


def compute_cpa_advantages(
    policy_cost: float,
    candidate_costs: np.ndarray,
    valid_candidate_mask: np.ndarray | None = None,
    cost_epsilon: float = 1e-8,
) -> np.ndarray:
    costs = np.asarray(candidate_costs, dtype=np.float64)
    denom = max(abs(float(policy_cost)), float(cost_epsilon))
    adv = np.maximum(0.0, (float(policy_cost) - costs) / denom)
    if valid_candidate_mask is not None:
        adv = np.where(np.asarray(valid_candidate_mask, dtype=bool), adv, 0.0)
    return adv.astype(np.float32)


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-8)
    logits = np.asarray(values, dtype=np.float64) / temp
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    denom = np.sum(exp)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full_like(logits, 1.0 / max(logits.size, 1), dtype=np.float64)
    return exp / denom


def build_planner_regret_target(
    candidates: np.ndarray,
    candidate_costs: np.ndarray,
    policy_action: np.ndarray,
    config: PlannerRegretTargetConfig,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    actions = np.asarray(candidates, dtype=np.float32)
    costs = np.asarray(candidate_costs, dtype=np.float64)
    valid = np.isfinite(costs)
    if not np.any(valid):
        costs = np.full(actions.shape[0], float("inf"), dtype=np.float64)
        valid = np.zeros(actions.shape[0], dtype=bool)
        best_idx = 0
    else:
        masked = np.where(valid, costs, np.inf)
        best_idx = int(np.argmin(masked))

    distances = np.linalg.norm(actions - np.asarray(policy_action, dtype=np.float32)[None, :], axis=1)
    policy_idx = int(np.argmin(distances))
    policy_cost = float(costs[policy_idx]) if np.isfinite(costs[policy_idx]) else float(costs[best_idx])
    best_cost = float(costs[best_idx]) if np.isfinite(costs[best_idx]) else policy_cost
    valid_costs = costs[valid]
    candidate_cost_spread = (
        float(np.max(valid_costs) - np.min(valid_costs))
        if valid_costs.size > 0
        else 0.0
    )
    excluding_policy = valid.copy()
    if 0 <= policy_idx < excluding_policy.shape[0]:
        excluding_policy[policy_idx] = False
    if np.any(excluding_policy):
        oracle_cost = float(np.min(costs[excluding_policy]))
        oracle_regret_excluding_policy_action = float(max(0.0, policy_cost - oracle_cost))
    else:
        oracle_regret_excluding_policy_action = 0.0
    policy_is_best_candidate = bool(policy_idx == best_idx)
    raw_regret = float(policy_cost - best_cost)
    normalized_regret = raw_regret / max(abs(policy_cost), float(config.cost_epsilon))
    regret_weight = float(np.clip(normalized_regret, 0.0, float(config.regret_weight_max)))

    target_type = str(config.target_type)
    active = True
    if target_type == "hybrid":
        if normalized_regret >= float(config.hard_regret_threshold):
            target_type = "hard"
        elif normalized_regret >= float(config.soft_regret_threshold):
            target_type = "soft"
        else:
            active = False
            regret_weight = 0.0
            target_type = "inactive"

    if bool(config.random_target_control) and rng is not None and np.any(valid):
        valid_indices = np.flatnonzero(valid)
        best_idx = int(rng.choice(valid_indices))
        best_cost = float(costs[best_idx])
        target_type = "random"

    if target_type == "soft" and np.any(valid):
        valid_costs = costs[valid]
        if str(config.cost_normalization) == "range":
            denom = max(float(np.max(valid_costs) - np.min(valid_costs)), float(config.cost_epsilon))
        else:
            denom = max(abs(policy_cost), float(config.cost_epsilon))
        normalized = np.zeros_like(costs, dtype=np.float64)
        normalized[valid] = (costs[valid] - float(np.min(valid_costs))) / denom
        weights = np.zeros_like(costs, dtype=np.float64)
        weights[valid] = _softmax(-normalized[valid], float(config.soft_temperature))
        target_action = np.sum(actions * weights[:, None], axis=0)
        target_cost = float(np.sum(np.where(valid, costs, 0.0) * weights))
    else:
        target_action = actions[best_idx]
        target_cost = best_cost

    if not active:
        target_action = actions[policy_idx]
        target_cost = policy_cost

    return {
        "target_action": np.asarray(target_action, dtype=np.float32),
        "target_type": target_type,
        "policy_cost": float(policy_cost),
        "target_cost": float(target_cost),
        "best_candidate_index": int(best_idx),
        "policy_candidate_index": int(policy_idx),
        "raw_regret": float(raw_regret),
        "normalized_regret": float(normalized_regret),
        "regret_weight": float(regret_weight),
        "candidate_costs": costs.astype(float).tolist(),
        "valid_candidate_mask": valid.astype(bool).tolist(),
        "target_l2": float(np.linalg.norm(np.asarray(target_action, dtype=np.float32) - actions[policy_idx])),
        "best_candidate_l2_from_policy": float(distances[best_idx]),
        "candidate_cost_spread": float(candidate_cost_spread),
        "oracle_regret_excluding_policy_action": float(oracle_regret_excluding_policy_action),
        "policy_is_best_candidate": bool(policy_is_best_candidate),
        "fraction_policy_is_best_candidate": float(policy_is_best_candidate),
    }


class PlannerRegretBuffer:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.samples)

    def add(self, sample: dict[str, Any]) -> None:
        self.samples.append(dict(sample))

    def clear(self) -> None:
        self.samples.clear()

    def sample_indices(self, count: int, rng: np.random.Generator) -> np.ndarray:
        if not self.samples:
            return np.asarray([], dtype=np.int64)
        size = min(int(count), len(self.samples))
        return rng.choice(len(self.samples), size=size, replace=False).astype(np.int64)

    def arrays(self, indices: np.ndarray | list[int] | None = None) -> dict[str, np.ndarray]:
        selected = self.samples if indices is None else [self.samples[int(idx)] for idx in indices]
        if not selected:
            return {
                "observations": np.zeros((0, 0), dtype=np.float32),
                "target_actions": np.zeros((0, 0), dtype=np.float32),
                "nominal_actions": np.zeros((0, 0), dtype=np.float32),
                "policy_actions_at_query_time": np.zeros((0, 0), dtype=np.float32),
                "residual_features": np.zeros((0, 0), dtype=np.float32),
                "regret_weights": np.zeros((0,), dtype=np.float32),
                "raw_regrets": np.zeros((0,), dtype=np.float32),
                "normalized_regrets": np.zeros((0,), dtype=np.float32),
                "candidate_actions": np.zeros((0, 0, 0), dtype=np.float32),
                "candidate_costs": np.zeros((0, 0), dtype=np.float32),
                "valid_candidate_masks": np.zeros((0, 0), dtype=bool),
                "policy_costs": np.zeros((0,), dtype=np.float32),
                "policy_candidate_indices": np.zeros((0,), dtype=np.int64),
                "ref_logp_candidates": np.zeros((0, 0), dtype=np.float32),
            }
        return {
            "observations": np.asarray([s["observation"] for s in selected], dtype=np.float32),
            "target_actions": np.asarray([s["target_action"] for s in selected], dtype=np.float32),
            "nominal_actions": np.asarray([s.get("nominal_action", []) for s in selected], dtype=np.float32),
            "policy_actions_at_query_time": np.asarray(
                [s.get("policy_action_at_query_time", []) for s in selected],
                dtype=np.float32,
            ),
            "residual_features": np.asarray([s.get("residual_features", []) for s in selected], dtype=np.float32),
            "regret_weights": np.asarray([s.get("regret_weight", 0.0) for s in selected], dtype=np.float32),
            "raw_regrets": np.asarray([s.get("raw_regret", 0.0) for s in selected], dtype=np.float32),
            "normalized_regrets": np.asarray([s.get("normalized_regret", 0.0) for s in selected], dtype=np.float32),
            "candidate_actions": np.asarray([s.get("candidate_actions", []) for s in selected], dtype=np.float32),
            "candidate_costs": np.asarray([s.get("candidate_costs", []) for s in selected], dtype=np.float32),
            "valid_candidate_masks": np.asarray([s.get("valid_candidate_mask", []) for s in selected], dtype=bool),
            "policy_costs": np.asarray([s.get("policy_cost", 0.0) for s in selected], dtype=np.float32),
            "policy_candidate_indices": np.asarray([s.get("policy_candidate_index", 0) for s in selected], dtype=np.int64),
            "ref_logp_candidates": np.asarray([s.get("ref_logp_candidates", []) for s in selected], dtype=np.float32),
        }

    def summary(self) -> dict[str, float | int]:
        if not self.samples:
            return {
                "pr_num_query_states": 0,
                "pr_total_planner_queries": 0,
                "pr_planner_query_failures": 0,
                "pr_fraction_positive_regret": 0.0,
            }
        raw = np.asarray([s.get("raw_regret", 0.0) for s in self.samples], dtype=np.float64)
        norm = np.asarray([s.get("normalized_regret", 0.0) for s in self.samples], dtype=np.float64)
        weights = np.asarray([s.get("regret_weight", 0.0) for s in self.samples], dtype=np.float64)
        policy_costs = np.asarray([s.get("policy_cost", np.nan) for s in self.samples], dtype=np.float64)
        target_costs = np.asarray([s.get("target_cost", np.nan) for s in self.samples], dtype=np.float64)
        target_l2 = np.asarray([s.get("target_l2", np.nan) for s in self.samples], dtype=np.float64)
        best_l2 = np.asarray([s.get("best_candidate_l2_from_policy", np.nan) for s in self.samples], dtype=np.float64)
        spread = np.asarray([s.get("candidate_cost_spread", np.nan) for s in self.samples], dtype=np.float64)
        oracle = np.asarray([s.get("oracle_regret_excluding_policy_action", np.nan) for s in self.samples], dtype=np.float64)
        is_best = np.asarray([s.get("policy_is_best_candidate", False) for s in self.samples], dtype=bool)
        failures = int(sum(int(s.get("planner_failure_count", 0)) for s in self.samples))
        queries = int(sum(int(s.get("planner_query_count", 0)) for s in self.samples))
        positive = raw > 0.0
        return {
            "pr_num_query_states": int(len(self.samples)),
            "pr_total_planner_queries": queries,
            "pr_planner_query_failures": failures,
            "pr_mean_policy_cost": float(np.nanmean(policy_costs)),
            "pr_mean_target_cost": float(np.nanmean(target_costs)),
            "pr_mean_raw_regret": float(np.nanmean(raw)),
            "pr_mean_positive_regret": float(np.nanmean(np.maximum(norm, 0.0))),
            "pr_mean_normalized_regret": float(np.nanmean(norm)),
            "pr_mean_regret_weight": float(np.nanmean(weights)),
            "pr_fraction_positive_regret": float(np.mean(positive)),
            "pr_mean_action_target_l2": float(np.nanmean(target_l2)),
            "pr_mean_target_l2": float(np.nanmean(target_l2)),
            "pr_mean_best_candidate_l2_from_policy": float(np.nanmean(best_l2)),
            "pr_mean_candidate_cost_spread": float(np.nanmean(spread)),
            "pr_mean_oracle_regret_excluding_policy_action": float(np.nanmean(oracle)),
            "pr_fraction_policy_is_best_candidate": float(np.mean(is_best)),
        }
