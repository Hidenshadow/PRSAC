"""Local repair candidate generation and supervised target construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np


@dataclass(frozen=True)
class CandidateAction:
    action: np.ndarray
    candidate_type: str
    dim: int | None = None


@dataclass(frozen=True)
class LocalRepairConfig:
    improvement_epsilon: float = 0.25
    beta: float = 2.0
    w_max: float = 5.0
    target_mode: str = "best"
    soft_target_temperature: float = 2.0
    surface_ridge: float = 1e-3
    surface_max_step_fraction: float = 1.0
    surface_allow_diagonal: bool = True
    pairwise_candidate_mode: str = "none"
    target_blend: float = 1.0
    target_residual_norm_clip: float = 0.0
    failure_cost: float = 1e6
    reward_cost_key: str = "scalar_cost"


@dataclass(frozen=True)
class RepairLabel:
    clean_action: np.ndarray
    current_action: np.ndarray
    best_action: np.ndarray
    current_residual: np.ndarray
    target_residual: np.ndarray
    current_score: float
    best_score: float
    improvement: float
    weight: float
    chosen_candidate_type: str


@dataclass
class LocalRepairResult:
    candidates: list[CandidateAction]
    scores: np.ndarray
    raw_results: list[dict[str, Any]]
    label: RepairLabel | None


class EpisodePlannerContext:
    """Minimal env-like context for counterfactual planner evaluation.

    The existing planner-regret evaluator expects an object with a
    ``current_episode`` attribute and planner action configuration fields. This
    wrapper intentionally has no ``step`` method.
    """

    def __init__(
        self,
        episode: Any,
        *,
        action_mode: str = "preference_delta",
        action_gain: float = 2.0,
        max_uncertainty_lambda: float = 1.0,
        allow_diagonal: bool = True,
        attack_budget_fraction: float = 0.18,
        attack_strength: float = 1.0,
        attacker_temperature: float = 0.5,
        attacker_response: str = "zscore_topk",
        attacker_top_fraction: float = 0.15,
        attacker_sharpness: float = 3.0,
        reward_cost_key: str = "scalar_cost",
    ) -> None:
        self.current_episode = episode
        self.action_mode = str(action_mode)
        self.action_gain = float(action_gain)
        self.max_uncertainty_lambda = float(max_uncertainty_lambda)
        self.allow_diagonal = bool(allow_diagonal)
        self.attack_budget_fraction = float(attack_budget_fraction)
        self.attack_strength = float(attack_strength)
        self.attacker_temperature = float(attacker_temperature)
        self.attacker_response = str(attacker_response)
        self.attacker_top_fraction = float(attacker_top_fraction)
        self.attacker_sharpness = float(attacker_sharpness)
        self.reward_cost_key = str(reward_cost_key)


def _as_action_vector(value: float | Iterable[float] | np.ndarray, action_dim: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size == 1:
        array = np.repeat(array, int(action_dim)).astype(np.float32)
    if array.size != int(action_dim):
        raise ValueError(f"{name} must be scalar or length {action_dim}, got {array.size}")
    return array.astype(np.float32)


def action_range_fraction(
    fraction: float | Iterable[float] | np.ndarray,
    action_low: float | Iterable[float] | np.ndarray,
    action_high: float | Iterable[float] | np.ndarray,
    action_dim: int,
) -> np.ndarray:
    low = _as_action_vector(action_low, action_dim, "action_low")
    high = _as_action_vector(action_high, action_dim, "action_high")
    frac = _as_action_vector(fraction, action_dim, "fraction")
    return (frac * (high - low)).astype(np.float32)


def clamp_action(
    action: np.ndarray,
    action_low: float | Iterable[float] | np.ndarray,
    action_high: float | Iterable[float] | np.ndarray,
) -> np.ndarray:
    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    low = _as_action_vector(action_low, action_array.size, "action_low")
    high = _as_action_vector(action_high, action_array.size, "action_high")
    return np.clip(action_array, low, high).astype(np.float32)


def generate_local_repair_candidates(
    current_action: np.ndarray,
    clean_action: np.ndarray,
    action_low: float | Iterable[float] | np.ndarray,
    action_high: float | Iterable[float] | np.ndarray,
    search_radius: float | Iterable[float] | np.ndarray,
    pairwise_candidate_mode: str = "none",
) -> list[CandidateAction]:
    """Return the deterministic LRR candidate set: current, clean, +/- axes."""

    current = np.asarray(current_action, dtype=np.float32).reshape(-1)
    clean = np.asarray(clean_action, dtype=np.float32).reshape(-1)
    if current.shape != clean.shape:
        raise ValueError(f"current_action shape {current.shape} != clean_action shape {clean.shape}")
    dim = current.size
    low = _as_action_vector(action_low, dim, "action_low")
    high = _as_action_vector(action_high, dim, "action_high")
    radius = _as_action_vector(search_radius, dim, "search_radius")

    candidates = [
        CandidateAction(np.clip(current, low, high).astype(np.float32), "current"),
        CandidateAction(np.clip(clean, low, high).astype(np.float32), "clean_anchor"),
    ]
    for index in range(dim):
        plus = current.copy()
        plus[index] += radius[index]
        minus = current.copy()
        minus[index] -= radius[index]
        candidates.append(CandidateAction(np.clip(plus, low, high).astype(np.float32), f"plus_dim_{index}", index))
        candidates.append(CandidateAction(np.clip(minus, low, high).astype(np.float32), f"minus_dim_{index}", index))

    pairwise_mode = str(pairwise_candidate_mode).lower()
    if pairwise_mode not in {"none", "adjacent", "all"}:
        raise ValueError("pairwise_candidate_mode must be 'none', 'adjacent', or 'all'")
    if pairwise_mode != "none":
        if pairwise_mode == "adjacent":
            pairs = [(index, index + 1) for index in range(max(dim - 1, 0))]
        else:
            pairs = [(left, right) for left in range(dim) for right in range(left + 1, dim)]
        signs = ((1.0, 1.0, "pp"), (1.0, -1.0, "pm"), (-1.0, 1.0, "mp"), (-1.0, -1.0, "mm"))
        for left, right in pairs:
            for left_sign, right_sign, sign_name in signs:
                pair_action = current.copy()
                pair_action[left] += left_sign * radius[left]
                pair_action[right] += right_sign * radius[right]
                candidates.append(
                    CandidateAction(
                        np.clip(pair_action, low, high).astype(np.float32),
                        f"pair_{sign_name}_dim_{left}_{right}",
                        None,
                    )
                )
    return candidates


def performance_index(clean_nominal_cost: float, current_cost: float) -> float:
    clean = float(clean_nominal_cost)
    cost = float(current_cost)
    if not np.isfinite(clean) or not np.isfinite(cost) or cost <= 0.0:
        return float("nan")
    return float(100.0 * clean / max(cost, 1e-8))


def _candidate_cost(result: dict[str, Any], reward_cost_key: str, failure_cost: float) -> float:
    for key in (reward_cost_key, "true_attacked_cost", "attacked_scalar_cost", "scalar_cost"):
        if key in result:
            try:
                value = float(result[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
    return float(failure_cost)


def evaluate_planner_action(
    context: Any,
    action: np.ndarray,
    clean_nominal_cost: float,
    *,
    reward_cost_key: str = "scalar_cost",
    failure_cost: float = 1e6,
) -> tuple[float, dict[str, Any]]:
    """Evaluate one candidate without mutating a live rollout environment."""

    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if hasattr(context, "evaluate_counterfactual_action"):
        result = context.evaluate_counterfactual_action(action_array)
    elif callable(context):
        result = context(action_array)
    else:
        from utils.planner_regret import evaluate_counterfactual_action_for_env

        result = evaluate_counterfactual_action_for_env(
            context,
            action_array,
            reward_cost_key=reward_cost_key,
            failure_cost=failure_cost,
        )
    result = dict(result or {})
    cost = _candidate_cost(result, reward_cost_key, failure_cost)
    score = performance_index(clean_nominal_cost, cost)
    if not np.isfinite(score):
        score = float("-inf")
    result.setdefault("true_attacked_cost", float(cost))
    result.setdefault("performance_index", float(score))
    return float(score), result


def _surface_features(offsets: np.ndarray, pair_indices: list[tuple[int, int]]) -> np.ndarray:
    offsets = np.asarray(offsets, dtype=np.float64)
    parts = [np.ones((offsets.shape[0], 1), dtype=np.float64), offsets, offsets * offsets]
    if pair_indices:
        pair_features = [offsets[:, left : left + 1] * offsets[:, right : right + 1] for left, right in pair_indices]
        parts.append(np.concatenate(pair_features, axis=1))
    return np.concatenate(parts, axis=1)


def _surface_pair_indices(offsets: np.ndarray) -> list[tuple[int, int]]:
    pair_set: set[tuple[int, int]] = set()
    for row in np.asarray(offsets, dtype=np.float64):
        active = np.flatnonzero(np.abs(row) > 1e-6)
        if active.size < 2:
            continue
        for left_pos, left in enumerate(active[:-1]):
            for right in active[left_pos + 1 :]:
                pair_set.add((int(left), int(right)))
    return sorted(pair_set)


def _fit_response_surface(
    candidates: list[CandidateAction],
    scores: np.ndarray,
    current_action: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]], np.ndarray] | None:
    finite = np.isfinite(scores)
    if int(np.sum(finite)) < 3:
        return None
    actions = np.stack([np.asarray(candidate.action, dtype=np.float64).reshape(-1) for candidate in candidates], axis=0)
    current = np.asarray(current_action, dtype=np.float64).reshape(-1)
    local_radius = np.max(np.abs(actions - current[None, :]), axis=0)
    if not np.any(local_radius > 1e-8):
        return None
    local_radius = np.maximum(local_radius, 1e-8)
    offsets = np.clip((actions - current[None, :]) / local_radius[None, :], -1.0, 1.0)
    pair_indices = _surface_pair_indices(offsets[finite])
    design = _surface_features(offsets[finite], pair_indices)
    target = np.asarray(scores[finite], dtype=np.float64) - float(scores[0])
    penalty = np.eye(design.shape[1], dtype=np.float64) * max(float(ridge), 0.0)
    penalty[0, 0] = 0.0
    try:
        coeff = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    except np.linalg.LinAlgError:
        coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coeff.astype(np.float64), local_radius.astype(np.float64), pair_indices, offsets.astype(np.float64)


def _predict_response_surface(
    offsets: np.ndarray,
    coeff: np.ndarray,
    pair_indices: list[tuple[int, int]],
) -> np.ndarray:
    return _surface_features(np.asarray(offsets, dtype=np.float64), pair_indices) @ np.asarray(coeff, dtype=np.float64)


def _response_surface_target_action(
    candidates: list[CandidateAction],
    scores: np.ndarray,
    current_action: np.ndarray,
    *,
    ridge: float = 1e-3,
    max_step_fraction: float = 1.0,
    allow_diagonal: bool = True,
) -> tuple[np.ndarray, str] | None:
    fit = _fit_response_surface(candidates, scores, current_action, ridge=ridge)
    if fit is None:
        return None
    coeff, local_radius, pair_indices, candidate_offsets = fit
    current = np.asarray(current_action, dtype=np.float64).reshape(-1)
    dim = current.size
    max_step = float(np.clip(float(max_step_fraction), 0.0, 1.0))
    if max_step <= 0.0:
        return None

    linear = coeff[1 : 1 + dim]
    quadratic = coeff[1 + dim : 1 + 2 * dim]
    pair_coeffs = coeff[1 + 2 * dim :]
    pair_map = {pair: float(pair_coeffs[index]) for index, pair in enumerate(pair_indices)}

    proposals: list[tuple[np.ndarray, str]] = []
    for index, row in enumerate(candidate_offsets):
        proposals.append((np.clip(row, -max_step, max_step), f"surface_existing_{index}"))

    diagonal = np.zeros(dim, dtype=np.float64)
    for index in range(dim):
        if abs(linear[index]) <= 1e-10 and abs(quadratic[index]) <= 1e-10:
            value = 0.0
        elif quadratic[index] < -1e-8:
            value = -linear[index] / (2.0 * quadratic[index])
        else:
            value = max_step if linear[index] >= 0.0 else -max_step
        diagonal[index] = float(np.clip(value, -max_step, max_step))
        axis = np.zeros(dim, dtype=np.float64)
        axis[index] = diagonal[index]
        proposals.append((axis, f"surface_axis_{index}"))
    if bool(allow_diagonal):
        proposals.append((diagonal, "surface_diagonal"))

    for left, right in pair_indices:
        hessian = np.array(
            [
                [2.0 * quadratic[left], pair_map.get((left, right), 0.0)],
                [pair_map.get((left, right), 0.0), 2.0 * quadratic[right]],
            ],
            dtype=np.float64,
        )
        gradient = np.array([linear[left], linear[right]], dtype=np.float64)
        pair = np.zeros(dim, dtype=np.float64)
        try:
            pair_step = -np.linalg.solve(hessian, gradient)
            if np.all(np.isfinite(pair_step)):
                pair[left] = float(np.clip(pair_step[0], -max_step, max_step))
                pair[right] = float(np.clip(pair_step[1], -max_step, max_step))
                proposals.append((pair, f"surface_pair_newton_{left}_{right}"))
        except np.linalg.LinAlgError:
            pass

    offsets = np.stack([proposal for proposal, _name in proposals], axis=0)
    predicted = _predict_response_surface(offsets, coeff, pair_indices)
    best_index = int(np.nanargmax(predicted))
    if not np.isfinite(predicted[best_index]) or predicted[best_index] <= 0.0:
        return None
    best_offset, source = proposals[best_index]
    target_action = current + best_offset * local_radius
    return target_action.astype(np.float32), source


def build_repair_label(
    clean_action: np.ndarray,
    current_action: np.ndarray,
    candidates: list[CandidateAction],
    candidate_scores: np.ndarray,
    delta_max: float | Iterable[float] | np.ndarray,
    *,
    improvement_epsilon: float = 0.25,
    beta: float = 2.0,
    w_max: float = 5.0,
    target_mode: str = "best",
    soft_target_temperature: float = 2.0,
    surface_ridge: float = 1e-3,
    surface_max_step_fraction: float = 1.0,
    surface_allow_diagonal: bool = True,
    target_blend: float = 1.0,
    target_residual_norm_clip: float = 0.0,
    action_low: float | Iterable[float] | np.ndarray | None = None,
    action_high: float | Iterable[float] | np.ndarray | None = None,
) -> RepairLabel | None:
    if not candidates:
        return None
    scores = np.asarray(candidate_scores, dtype=np.float64).reshape(-1)
    if scores.size != len(candidates):
        raise ValueError(f"score count {scores.size} does not match candidate count {len(candidates)}")
    finite = np.isfinite(scores)
    if not finite.any() or not np.isfinite(scores[0]):
        return None

    best_index = int(np.nanargmax(scores))
    current_score = float(scores[0])
    best_score = float(scores[best_index])
    improvement = best_score - current_score
    if improvement <= float(improvement_epsilon):
        return None

    clean = np.asarray(clean_action, dtype=np.float32).reshape(-1)
    current = np.asarray(current_action, dtype=np.float32).reshape(-1)
    best = np.asarray(candidates[best_index].action, dtype=np.float32).reshape(-1)
    if clean.shape != current.shape or clean.shape != best.shape:
        raise ValueError("clean_action, current_action, and best_action must share shape")
    delta_limit = _as_action_vector(delta_max, clean.size, "delta_max")
    current_residual = (current - clean).astype(np.float32)
    residual_candidates = np.stack(
        [
            np.clip(np.asarray(candidate.action, dtype=np.float32).reshape(-1) - clean, -delta_limit, delta_limit)
            for candidate in candidates
        ],
        axis=0,
    ).astype(np.float32)
    mode = str(target_mode).lower()
    if mode == "best":
        raw_target_residual = residual_candidates[best_index].astype(np.float32)
        chosen_candidate_type = candidates[best_index].candidate_type
    elif mode == "soft":
        improvements = scores - current_score
        positive = finite & (improvements > float(improvement_epsilon))
        positive[0] = False
        if not bool(np.any(positive)):
            return None
        positive_improvements = improvements[positive].astype(np.float64)
        temperature = max(float(soft_target_temperature), 1e-6)
        logits = positive_improvements / temperature
        logits = logits - float(np.max(logits))
        soft_weights = np.exp(logits)
        soft_weights = soft_weights / max(float(np.sum(soft_weights)), 1e-12)
        raw_target_residual = np.sum(
            residual_candidates[positive] * soft_weights[:, None].astype(np.float32),
            axis=0,
        ).astype(np.float32)
        chosen_candidate_type = f"soft_improvement_{int(np.sum(positive))}"
    elif mode in {"surface", "response_surface"}:
        surface_target = _response_surface_target_action(
            candidates,
            scores,
            current,
            ridge=float(surface_ridge),
            max_step_fraction=float(surface_max_step_fraction),
            allow_diagonal=bool(surface_allow_diagonal),
        )
        if surface_target is None:
            raw_target_residual = residual_candidates[best_index].astype(np.float32)
            chosen_candidate_type = f"surface_fallback_{candidates[best_index].candidate_type}"
        else:
            surface_action, surface_source = surface_target
            if action_low is not None and action_high is not None:
                surface_action = clamp_action(surface_action, action_low, action_high)
            raw_target_residual = np.clip(
                np.asarray(surface_action, dtype=np.float32).reshape(-1) - clean,
                -delta_limit,
                delta_limit,
            ).astype(np.float32)
            chosen_candidate_type = surface_source
    else:
        raise ValueError("target_mode must be 'best', 'soft', or 'surface'")
    blend = float(np.clip(float(target_blend), 0.0, 1.0))
    target_residual = (
        current_residual + blend * (raw_target_residual - current_residual)
    ).astype(np.float32)
    target_residual = np.clip(target_residual, -delta_limit, delta_limit).astype(np.float32)
    norm_clip = float(target_residual_norm_clip)
    if norm_clip > 0.0:
        target_norm = float(np.linalg.norm(target_residual))
        if target_norm > norm_clip:
            target_residual = (target_residual * (norm_clip / max(target_norm, 1e-8))).astype(np.float32)
    weight = float(np.clip(improvement / max(float(beta), 1e-8), 0.0, float(w_max)))
    return RepairLabel(
        clean_action=clean.astype(np.float32),
        current_action=current.astype(np.float32),
        best_action=best.astype(np.float32),
        current_residual=current_residual,
        target_residual=target_residual,
        current_score=current_score,
        best_score=best_score,
        improvement=float(improvement),
        weight=weight,
        chosen_candidate_type=chosen_candidate_type,
    )


def evaluate_local_repair(
    context: Any,
    clean_nominal_cost: float,
    current_action: np.ndarray,
    clean_action: np.ndarray,
    action_low: float | Iterable[float] | np.ndarray,
    action_high: float | Iterable[float] | np.ndarray,
    search_radius: float | Iterable[float] | np.ndarray,
    delta_max: float | Iterable[float] | np.ndarray,
    config: LocalRepairConfig | None = None,
    evaluator: Callable[..., tuple[float, dict[str, Any]]] = evaluate_planner_action,
) -> LocalRepairResult:
    """Evaluate local candidates and return a supervised repair label if useful."""

    cfg = config or LocalRepairConfig()
    candidates = generate_local_repair_candidates(
        current_action,
        clean_action,
        action_low,
        action_high,
        search_radius,
        pairwise_candidate_mode=cfg.pairwise_candidate_mode,
    )
    scores: list[float] = []
    raw_results: list[dict[str, Any]] = []
    for candidate in candidates:
        score, result = evaluator(
            context,
            candidate.action,
            clean_nominal_cost,
            reward_cost_key=cfg.reward_cost_key,
            failure_cost=cfg.failure_cost,
        )
        result = dict(result)
        result["candidate_type"] = candidate.candidate_type
        scores.append(float(score))
        raw_results.append(result)
    score_array = np.asarray(scores, dtype=np.float64)
    label = build_repair_label(
        clean_action,
        current_action,
        candidates,
        score_array,
        delta_max,
        improvement_epsilon=cfg.improvement_epsilon,
        beta=cfg.beta,
        w_max=cfg.w_max,
        target_mode=cfg.target_mode,
        soft_target_temperature=cfg.soft_target_temperature,
        surface_ridge=cfg.surface_ridge,
        surface_max_step_fraction=cfg.surface_max_step_fraction,
        surface_allow_diagonal=cfg.surface_allow_diagonal,
        target_blend=cfg.target_blend,
        target_residual_norm_clip=cfg.target_residual_norm_clip,
        action_low=action_low,
        action_high=action_high,
    )
    return LocalRepairResult(candidates=candidates, scores=score_array, raw_results=raw_results, label=label)
