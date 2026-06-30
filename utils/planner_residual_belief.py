"""Planner-residual belief utilities for same-shock PPO recovery.

PRB-PPO uses planner-vs-true-cost diagnostics as representation inputs and
auxiliary prediction targets. This module is intentionally independent from
attack-variant and planner-action-imitation code paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


COMPONENT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("hazard", "belief_hazard_exposure", "hazard_exposure"),
    ("uncertainty", "belief_uncertainty_exposure", "uncertainty_exposure"),
    ("communication", "belief_communication_exposure", "communication_exposure"),
    ("illumination", "belief_illumination_exposure", "illumination_exposure"),
)


@dataclass(frozen=True)
class PlannerResidualFeatureConfig:
    action_dim: int
    normalize_features: bool = True
    feature_clip: float = 5.0
    epsilon: float = 1e-8
    use_component_costs: bool = True
    use_scalar_cost: bool = True
    use_attack_belief: bool = False
    attack_belief_dim: int = 0


@dataclass(frozen=True)
class PlannerResidualBatch:
    features: np.ndarray
    feature_mask: np.ndarray
    residual_total_target: float
    true_total_target: float
    component_residual_target: np.ndarray
    true_component_target: np.ndarray
    component_mask: np.ndarray
    component_cost_available: bool
    raw: dict[str, Any]


def _finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _result_cost(info: dict[str, Any], reward_cost_key: str) -> float:
    if reward_cost_key in info:
        return _finite_float(info.get(reward_cost_key))
    if reward_cost_key == "soft_attacked_scalar_cost":
        return _finite_float(info.get("soft_attacked_scalar_cost", info.get("attacked_scalar_cost", info.get("scalar_cost"))))
    return _finite_float(info.get("attacked_scalar_cost", info.get("scalar_cost")))


def neutral_probe_action(action_dim: int, zero_action_value: float = 0.5) -> np.ndarray:
    return np.full(int(action_dim), float(zero_action_value), dtype=np.float32)


def collect_planner_residual_record(
    info: dict[str, Any],
    action_delta: np.ndarray | list[float] | tuple[float, ...],
    reward_cost_key: str = "attacked_scalar_cost",
) -> dict[str, Any]:
    """Collect scalar/component residual diagnostics from an env info/result dict."""

    action = np.asarray(action_delta, dtype=np.float32).reshape(-1)
    planner_cost = _finite_float(info.get("belief_scalar_cost", info.get("scalar_cost")))
    true_cost = _result_cost(info, reward_cost_key)
    if not np.isfinite(true_cost):
        true_cost = _finite_float(info.get("attacked_scalar_cost", info.get("scalar_cost")))
    if not np.isfinite(planner_cost):
        planner_cost = true_cost
    residual_total = true_cost - planner_cost

    predicted_components: list[float] = []
    true_components: list[float] = []
    component_mask: list[bool] = []
    for _name, predicted_key, true_key in COMPONENT_SPECS:
        predicted = _finite_float(info.get(predicted_key))
        true = _finite_float(info.get(true_key))
        valid = bool(np.isfinite(predicted) and np.isfinite(true))
        predicted_components.append(float(predicted if valid else 0.0))
        true_components.append(float(true if valid else 0.0))
        component_mask.append(valid)

    success = bool(info.get("success", True))
    path_valid = 1.0 if success else 0.0
    planner_failure = 0.0 if success else 1.0
    record = {
        "planner_predicted_total_cost": float(planner_cost),
        "true_attacked_total_cost": float(true_cost),
        "residual_total_cost": float(residual_total),
        "planner_predicted_component_costs": np.asarray(predicted_components, dtype=np.float32),
        "true_attacked_component_costs": np.asarray(true_components, dtype=np.float32),
        "component_mask": np.asarray(component_mask, dtype=bool),
        "path_valid": float(path_valid),
        "planner_failure_flag": float(planner_failure),
        "path_length": _finite_float(info.get("path_length"), 0.0),
        "map_mismatch_penalty": _finite_float(info.get("map_mismatch_penalty"), 0.0),
        "map_mismatch_abs_error": _finite_float(info.get("map_mismatch_abs_error"), 0.0),
        "action_delta": action.astype(np.float32),
    }
    return record


class PlannerResidualFeatureBuilder:
    """Convert raw planner residual diagnostics into fixed-width features."""

    def __init__(self, config: PlannerResidualFeatureConfig) -> None:
        self.config = config
        self.component_names = tuple(name for name, _pred, _true in COMPONENT_SPECS)
        attack_belief_dim = max(int(config.attack_belief_dim), 0) if bool(config.use_attack_belief) else 0
        self.feature_names = (
            "residual_total_norm",
            "true_over_planner_cost",
            "planner_cost_log1p",
            "path_length_norm",
            "path_valid",
            "planner_failure_flag",
            "map_mismatch_penalty_norm",
            "map_mismatch_abs_error_norm",
            *(f"component_residual_norm_{name}" for name in self.component_names),
            *(f"action_delta_{index}" for index in range(int(config.action_dim))),
            *(
                (
                    "attack_belief_entropy_norm",
                    "attack_belief_confidence",
                    "attack_belief_expected_residual_norm",
                    "attack_belief_worst_residual_norm",
                    "attack_belief_cost_spread_norm",
                    "attack_belief_observed_cost_norm",
                )
                if attack_belief_dim > 0
                else ()
            ),
            *(f"attack_belief_prob_{index}" for index in range(attack_belief_dim)),
            *(f"attack_variant_residual_norm_{index}" for index in range(attack_belief_dim)),
        )
        self.feature_dim = len(self.feature_names)
        self.component_dim = len(self.component_names)
        self.attack_belief_dim = attack_belief_dim

    def zero_features(self) -> np.ndarray:
        return np.zeros(self.feature_dim, dtype=np.float32)

    def build_from_info(
        self,
        info: dict[str, Any],
        action_delta: np.ndarray | list[float] | tuple[float, ...],
        reward_cost_key: str = "attacked_scalar_cost",
    ) -> PlannerResidualBatch:
        return self.build(collect_planner_residual_record(info, action_delta, reward_cost_key))

    def build(self, record: dict[str, Any]) -> PlannerResidualBatch:
        cfg = self.config
        eps = float(cfg.epsilon)
        clip = float(cfg.feature_clip)

        planner_cost = _finite_float(record.get("planner_predicted_total_cost"), 0.0)
        true_cost = _finite_float(record.get("true_attacked_total_cost"), planner_cost)
        residual_total = true_cost - planner_cost
        denom = max(abs(planner_cost), eps)
        residual_total_norm = residual_total / denom
        true_total_target = true_cost / denom
        true_over_planner = true_total_target
        planner_cost_log = np.sign(planner_cost) * np.log1p(abs(planner_cost))
        path_length_norm = _finite_float(record.get("path_length"), 0.0) / max(abs(planner_cost), 1.0)
        map_mismatch_penalty_norm = _finite_float(record.get("map_mismatch_penalty"), 0.0) / denom
        map_mismatch_abs_error_norm = _finite_float(record.get("map_mismatch_abs_error"), 0.0) / denom

        predicted_components = np.asarray(record.get("planner_predicted_component_costs", []), dtype=np.float32)
        true_components = np.asarray(record.get("true_attacked_component_costs", []), dtype=np.float32)
        component_mask = np.asarray(record.get("component_mask", []), dtype=bool)
        if predicted_components.shape[0] != self.component_dim:
            predicted_components = np.zeros(self.component_dim, dtype=np.float32)
        if true_components.shape[0] != self.component_dim:
            true_components = np.zeros(self.component_dim, dtype=np.float32)
        if component_mask.shape[0] != self.component_dim:
            component_mask = np.zeros(self.component_dim, dtype=bool)
        if not bool(cfg.use_component_costs):
            component_mask = np.zeros(self.component_dim, dtype=bool)

        comp_den = np.maximum(np.abs(predicted_components), eps)
        component_residual = true_components - predicted_components
        component_residual_norm = np.divide(component_residual, comp_den, out=np.zeros_like(component_residual), where=component_mask)
        true_component_target = np.divide(true_components, comp_den, out=np.zeros_like(true_components), where=component_mask)

        action = np.asarray(record.get("action_delta", np.zeros(cfg.action_dim)), dtype=np.float32).reshape(-1)
        if action.shape[0] < cfg.action_dim:
            action = np.pad(action, (0, cfg.action_dim - action.shape[0]))
        action = action[: cfg.action_dim]

        value_list = [
            residual_total_norm if bool(cfg.use_scalar_cost) else 0.0,
            true_over_planner if bool(cfg.use_scalar_cost) else 0.0,
            planner_cost_log if bool(cfg.use_scalar_cost) else 0.0,
            path_length_norm,
            _finite_float(record.get("path_valid"), 1.0),
            _finite_float(record.get("planner_failure_flag"), 0.0),
            map_mismatch_penalty_norm,
            map_mismatch_abs_error_norm,
            *component_residual_norm.tolist(),
            *action.tolist(),
        ]
        if self.attack_belief_dim > 0:
            posterior = np.asarray(record.get("attack_belief_posterior", []), dtype=np.float32).reshape(-1)
            variant_residual = np.asarray(record.get("attack_variant_residual_norms", []), dtype=np.float32).reshape(-1)
            if posterior.shape[0] < self.attack_belief_dim:
                posterior = np.pad(posterior, (0, self.attack_belief_dim - posterior.shape[0]))
            if variant_residual.shape[0] < self.attack_belief_dim:
                variant_residual = np.pad(variant_residual, (0, self.attack_belief_dim - variant_residual.shape[0]))
            posterior = posterior[: self.attack_belief_dim]
            variant_residual = variant_residual[: self.attack_belief_dim]
            value_list.extend(
                [
                    _finite_float(record.get("attack_belief_entropy_norm"), 0.0),
                    _finite_float(record.get("attack_belief_confidence"), 0.0),
                    _finite_float(record.get("attack_belief_expected_residual_norm"), 0.0),
                    _finite_float(record.get("attack_belief_worst_residual_norm"), 0.0),
                    _finite_float(record.get("attack_belief_cost_spread_norm"), 0.0),
                    _finite_float(record.get("attack_belief_observed_cost_norm"), 0.0),
                    *posterior.tolist(),
                    *variant_residual.tolist(),
                ]
            )

        values = np.asarray(value_list, dtype=np.float32)
        feature_mask = np.ones(values.shape[0], dtype=np.float32)
        if not bool(cfg.use_scalar_cost):
            feature_mask[:3] = 0.0
        component_start = 8
        feature_mask[component_start : component_start + self.component_dim] = component_mask.astype(np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=clip, neginf=-clip)
        if bool(cfg.normalize_features):
            values = np.clip(values, -clip, clip)

        component_residual_norm = np.nan_to_num(component_residual_norm, nan=0.0, posinf=clip, neginf=-clip)
        true_component_target = np.nan_to_num(true_component_target, nan=0.0, posinf=clip, neginf=-clip)
        residual_total_target = float(np.clip(residual_total_norm, -clip, clip))
        true_total_target = float(np.clip(true_total_target, -clip, clip))

        return PlannerResidualBatch(
            features=values.astype(np.float32),
            feature_mask=feature_mask.astype(np.float32),
            residual_total_target=residual_total_target,
            true_total_target=true_total_target,
            component_residual_target=np.clip(component_residual_norm, -clip, clip).astype(np.float32),
            true_component_target=np.clip(true_component_target, -clip, clip).astype(np.float32),
            component_mask=component_mask.astype(np.float32),
            component_cost_available=bool(component_mask.any()),
            raw={
                **record,
                "residual_total_norm": residual_total_norm,
                "true_total_norm": true_total_target,
            },
        )


def attack_belief_record(
    env: Any,
    record: dict[str, Any],
    action: np.ndarray | list[float] | tuple[float, ...],
    environment_attack_config: dict[str, Any] | None,
    reward_cost_key: str,
    max_attack_variants: int = 6,
    temperature: float = 0.25,
    prior_mix: float = 0.10,
    failure_cost: float = 1e6,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Append a Bayes-game attack posterior inferred from variant costs."""

    from utils.planner_regret import evaluate_attack_variant_costs_for_env

    output = dict(record)
    max_variants = max(int(max_attack_variants), 0)
    if max_variants <= 0:
        return output
    try:
        variant_eval = evaluate_attack_variant_costs_for_env(
            env,
            np.asarray(action, dtype=np.float32),
            environment_attack_config,
            reward_cost_key=reward_cost_key,
            max_attack_variants=max_variants,
            rng=rng,
            failure_cost=failure_cost,
        )
    except Exception as exc:
        output.update(
            {
                "attack_belief_failure": True,
                "attack_belief_failure_reason": str(exc),
                "attack_belief_posterior": np.zeros(max_variants, dtype=np.float32),
                "attack_variant_residual_norms": np.zeros(max_variants, dtype=np.float32),
            }
        )
        return output

    costs = np.asarray(variant_eval.get("costs", []), dtype=np.float64).reshape(-1)
    valid = np.asarray(variant_eval.get("valid_mask", np.ones_like(costs, dtype=bool)), dtype=bool).reshape(-1)
    if costs.size == 0:
        return output
    observed_cost = _finite_float(record.get("true_attacked_total_cost"), np.nan)
    planner_cost = _finite_float(record.get("planner_predicted_total_cost"), observed_cost)
    finite_costs = costs[np.isfinite(costs) & valid]
    if not np.isfinite(observed_cost):
        observed_cost = float(np.nanmedian(finite_costs)) if finite_costs.size else float(failure_cost)
    if finite_costs.size == 0:
        finite_costs = np.asarray([observed_cost], dtype=np.float64)
    fallback_cost = float(np.max(finite_costs) + max(abs(float(np.max(finite_costs))), 1.0))
    safe_costs = np.where(np.isfinite(costs) & valid, costs, fallback_cost)
    cost_scale = max(float(np.std(safe_costs)), abs(float(observed_cost)) * 0.10, 1e-6)
    logits = -np.abs(safe_costs - float(observed_cost)) / (max(float(temperature), 1e-6) * cost_scale)
    logits = logits - float(np.max(logits))
    posterior = np.exp(logits)
    posterior = posterior / max(float(posterior.sum()), 1e-12)
    prior_weight = float(np.clip(prior_mix, 0.0, 1.0))
    posterior = (1.0 - prior_weight) * posterior + prior_weight / max(float(posterior.size), 1.0)
    posterior = posterior / max(float(posterior.sum()), 1e-12)

    denom = max(abs(float(planner_cost)), 1e-8)
    residual_norms = (safe_costs - float(planner_cost)) / denom
    residual_norms = np.nan_to_num(residual_norms, nan=0.0, posinf=0.0, neginf=0.0)
    entropy = -float(np.sum(posterior * np.log(np.clip(posterior, 1e-12, 1.0))))
    entropy_norm = entropy / max(float(np.log(max(posterior.size, 2))), 1e-8)
    cost_spread_norm = float((np.max(safe_costs) - np.min(safe_costs)) / denom)
    output.update(
        {
            "attack_belief_failure": False,
            "attack_belief_variant_ids": ";".join(str(item) for item in variant_eval.get("variant_ids", [])),
            "attack_belief_posterior": posterior.astype(np.float32),
            "attack_variant_costs": safe_costs.astype(np.float32),
            "attack_variant_residual_norms": residual_norms.astype(np.float32),
            "attack_belief_entropy_norm": float(np.clip(entropy_norm, 0.0, 1.0)),
            "attack_belief_confidence": float(np.max(posterior)),
            "attack_belief_expected_residual_norm": float(np.dot(posterior, residual_norms)),
            "attack_belief_worst_residual_norm": float(np.max(residual_norms)),
            "attack_belief_cost_spread_norm": cost_spread_norm,
            "attack_belief_observed_cost_norm": float(observed_cost / denom),
            "attack_belief_failure_count": int(variant_eval.get("failure_count", 0)),
        }
    )
    return output


def masked_huber_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if mask is None:
        return loss.mean()
    mask = mask.to(dtype=loss.dtype, device=loss.device)
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    denom = torch.clamp(mask.expand_as(loss).sum(), min=1.0)
    return (loss * mask).sum() / denom


def prb_auxiliary_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    lambdas: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    residual_total_loss = masked_huber_loss(
        predictions["residual_total"].view(-1),
        targets["residual_total"].view(-1),
    )
    true_total_loss = masked_huber_loss(
        predictions["true_total"].view(-1),
        targets["true_total"].view(-1),
    )
    component_mask = targets.get("component_mask")
    component_residual_loss = masked_huber_loss(
        predictions["component_residual"],
        targets["component_residual"],
        component_mask,
    )
    true_component_loss = masked_huber_loss(
        predictions["true_component"],
        targets["true_component"],
        component_mask,
    )
    total = (
        float(lambdas.get("residual_total", 1.0)) * residual_total_loss
        + float(lambdas.get("true_total", 0.5)) * true_total_loss
        + float(lambdas.get("component_residual", 1.0)) * component_residual_loss
        + float(lambdas.get("true_component", 0.5)) * true_component_loss
    )
    parts = {
        "residual_total_loss": residual_total_loss.detach(),
        "true_total_loss": true_total_loss.detach(),
        "component_residual_loss": component_residual_loss.detach(),
        "true_component_loss": true_component_loss.detach(),
    }
    return total, parts
