"""Optional robustness attack wrappers for rover planning experiments.

The wrappers in this module are deliberately small and config-driven. They do
not change the planner or policy classes; they only perturb observations or the
episode cost layers seen by an environment instance.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.ndimage import gaussian_filter

from maps.map_generator import GeneratedCostMap, OBJECTIVE_NAMES
from planners.weighted_astar import weighted_astar
from utils.metrics import (
    PlanningEpisode,
    compute_observation,
    evaluate_candidate_results,
    plan_with_weights,
)


OBSERVATION_ATTACK_TYPES = ("obs_gaussian_noise", "obs_dropout", "obs_bias")
ENVIRONMENT_ATTACK_TYPES = (
    "env_composite",
    "env_attack_mixture",
    "env_zscore_topk",
    "env_layer_noise",
    "env_layer_bias",
    "env_path_corridor_attack",
    "env_hazard_inflation",
    "env_uncertainty_inflation",
    "env_slope_risk_inflation",
    "env_belief_mismatch",
    "env_spatial_belief_mismatch",
    "env_confidence_degradation",
    "env_traversability_boundary_mismatch",
    "env_true_terrain_degradation",
)


def load_attack_config(config: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    """Load an attack config from a dict, JSON text, or JSON file path."""

    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)

    text = str(config).strip()
    if not text:
        return {}

    try:
        path = Path(text)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError:
        # JSON text can contain characters that are not valid in Windows paths.
        pass
    return json.loads(text)


def attack_enabled(config: dict[str, Any] | None) -> bool:
    return bool(config and config.get("enabled", False))


def _indices_from_config(size: int, indices: Any) -> np.ndarray | slice:
    if indices is None:
        return slice(None)
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size == 0:
        return np.asarray([], dtype=np.int64)
    return values[(values >= 0) & (values < size)]


def apply_observation_attack(
    observation: np.ndarray,
    config: dict[str, Any] | None,
    rng: np.random.Generator,
    observation_space: spaces.Space | None = None,
) -> np.ndarray:
    """Return a perturbed observation without changing environment state."""

    if not attack_enabled(config):
        return np.asarray(observation, dtype=np.float32).copy()

    attack_type = str(config.get("type", "obs_gaussian_noise"))
    if attack_type not in OBSERVATION_ATTACK_TYPES:
        raise ValueError(f"observation attack type must be one of {OBSERVATION_ATTACK_TYPES}")

    attacked = np.asarray(observation, dtype=np.float32).copy()
    indices = _indices_from_config(attacked.size, config.get("bias_indices"))

    if attack_type == "obs_gaussian_noise":
        noise_std = float(config.get("noise_std", 0.05))
        attacked = attacked + rng.normal(0.0, noise_std, size=attacked.shape).astype(np.float32)
    elif attack_type == "obs_dropout":
        dropout_prob = float(np.clip(config.get("dropout_prob", 0.1), 0.0, 1.0))
        fill_value = float(config.get("fill_value", 0.0))
        mask = rng.random(attacked.shape) < dropout_prob
        attacked[mask] = fill_value
    elif attack_type == "obs_bias":
        bias_value = float(config.get("bias_value", 0.0))
        attacked.reshape(-1)[indices] += bias_value

    should_clip = bool(config.get("clip_to_observation_space", True))
    if should_clip and isinstance(observation_space, spaces.Box):
        attacked = np.clip(attacked, observation_space.low, observation_space.high)

    return attacked.astype(np.float32)


class ObservationAttackWrapper(gym.ObservationWrapper):
    """Perturb observations returned to the policy while preserving true state."""

    def __init__(
        self,
        env: gym.Env,
        config: dict[str, Any] | None,
    ) -> None:
        super().__init__(env)
        self.config = dict(config or {})
        self.rng = np.random.default_rng(self.config.get("seed"))

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return apply_observation_attack(
            observation,
            self.config,
            self.rng,
            observation_space=self.observation_space,
        )


def _resolve_layer_family(layer_name: str, layer_family: str | None) -> tuple[str, str]:
    name = str(layer_name)
    family = layer_family
    if name.startswith("uncertainty:"):
        return "uncertainty", name.split(":", 1)[1]
    if name.startswith("uncertainty_"):
        return "uncertainty", name.removeprefix("uncertainty_")
    return str(family or "cost"), name


def _perturb_layer(
    layer: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    attacked = np.asarray(layer, dtype=np.float32).copy()
    attack_type = str(config.get("type", "env_layer_noise"))
    if attack_type == "env_layer_noise":
        noise_std = float(config.get("noise_std", 0.05))
        attacked = attacked + rng.normal(0.0, noise_std, size=attacked.shape).astype(np.float32)
    elif attack_type == "env_layer_bias":
        bias_value = float(config.get("bias_value", 0.0))
        mode = str(config.get("mode", "add"))
        if mode == "multiply":
            attacked = attacked * bias_value
        elif mode == "add":
            attacked = attacked + bias_value
        else:
            raise ValueError("env_layer_bias mode must be 'add' or 'multiply'")
    else:
        raise ValueError("layer perturbation only supports env_layer_noise/env_layer_bias")

    clip_low = float(config.get("clip_low", 0.0))
    clip_high = float(config.get("clip_high", 1.0))
    return np.clip(attacked, clip_low, clip_high).astype(np.float32)


def _corridor_mask_from_path(
    path: list[tuple[int, int]] | None,
    shape: tuple[int, int],
    radius: int,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if not path:
        return mask
    radius = max(int(radius), 0)
    rows, cols = np.indices(shape)
    for row, col in path:
        row_i, col_i = int(row), int(col)
        distance = np.maximum(np.abs(rows - row_i), np.abs(cols - col_i))
        mask |= distance <= radius
    return mask


def _reference_config_from_policy(
    episode: PlanningEpisode,
    config: dict[str, Any],
) -> dict[str, Any]:
    reference_policy = str(config.get("reference_policy", "heuristic"))
    fixed_reference_weights = {
        "distance_shortcut": np.array([0.92, 0.02, 0.02, 0.02, 0.02], dtype=np.float32),
        "distance_only": np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "energy_shortcut": np.array([0.35, 0.55, 0.04, 0.03, 0.03], dtype=np.float32),
        "hazard_shortcut": np.array([0.35, 0.04, 0.55, 0.03, 0.03], dtype=np.float32),
    }
    if reference_policy in fixed_reference_weights:
        weights = fixed_reference_weights[reference_policy]
        weights = weights / max(float(weights.sum()), 1e-8)
        return {
            "weights": weights.astype(np.float32),
            "lambda_uncertainty": 0.0,
            "reference_policy_resolved": reference_policy,
        }
    if reference_policy == "risk_neutral_mission":
        return {
            "weights": episode.mission_priority.astype(np.float32),
            "lambda_uncertainty": 0.0,
            "reference_policy_resolved": "risk_neutral_mission",
        }
    if reference_policy == "nominal_ppo":
        checkpoint_path = config.get("reference_checkpoint") or config.get("nominal_checkpoint")
        if checkpoint_path:
            try:
                from utils.evaluation_policy import (
                    load_model,
                    predict_action,
                    resolve_action_config,
                    resolve_observation_mode,
                )
                from utils.metrics import action_to_planning_weights, action_to_uncertainty_lambda

                model_type, model, model_config = load_model(checkpoint_path, "auto")
                observation_mode = resolve_observation_mode("auto", model_config)
                action_mode, action_gain, max_lambda = resolve_action_config("auto", None, None, model_config)
                map_size = int(episode.costmap.layers["distance"].shape[0])
                obs = compute_observation(
                    episode,
                    map_size,
                    observation_mode=observation_mode,
                    max_uncertainty_lambda=max_lambda,
                )
                action = predict_action(model_type, model, obs)
                return {
                    "weights": action_to_planning_weights(
                        episode,
                        action,
                        action_mode=action_mode,
                        action_gain=action_gain,
                    ),
                    "lambda_uncertainty": action_to_uncertainty_lambda(
                        action,
                        max_uncertainty_lambda=max_lambda,
                    ),
                    "reference_policy_resolved": "nominal_ppo",
                }
            except Exception as exc:  # pragma: no cover - diagnostic fallback
                print(f"WARNING: nominal_ppo corridor reference failed, using heuristic: {exc}")

    return {
        "weights": episode.mission_priority.astype(np.float32),
        "lambda_uncertainty": 0.0,
        "reference_policy_resolved": "heuristic",
    }


def _apply_path_corridor_attack(
    episode: PlanningEpisode,
    config: dict[str, Any],
) -> PlanningEpisode:
    costmap = episode.costmap
    reference_config = _reference_config_from_policy(episode, config)
    reference_result = plan_with_weights(
        episode,
        reference_config["weights"],
        lambda_uncertainty=float(reference_config["lambda_uncertainty"]),
        allow_diagonal=True,
    )
    reference_path = reference_result.get("path")
    shape = costmap.layers["distance"].shape
    corridor_radius = int(config.get("corridor_radius", 2))
    attack_mask = _corridor_mask_from_path(reference_path, shape, corridor_radius)
    attack_mask &= ~costmap.obstacle_mask

    layers = {name: value.copy() for name, value in costmap.layers.items()}
    uncertainty_layers = {
        name: value.copy()
        for name, value in costmap.uncertainty_layers.items()
    }
    affected_layers = tuple(config.get("affected_layers", ("hazard", "uncertainty")))
    attack_strength = float(config.get("attack_strength", 5.0))
    cost_delta = float(config.get("cost_delta", 0.08 * attack_strength))
    uncertainty_delta = float(config.get("uncertainty_delta", 0.12 * attack_strength))
    layer_cost_increase = 0.0

    for name in affected_layers:
        name = str(name)
        if name == "uncertainty":
            for uncertainty_name in uncertainty_layers:
                before = uncertainty_layers[uncertainty_name].copy()
                uncertainty_layers[uncertainty_name][attack_mask] = np.clip(
                    uncertainty_layers[uncertainty_name][attack_mask] + uncertainty_delta,
                    0.0,
                    1.0,
                )
                layer_cost_increase += float((uncertainty_layers[uncertainty_name] - before)[attack_mask].sum())
        elif name.startswith("uncertainty:"):
            uncertainty_name = name.split(":", 1)[1]
            if uncertainty_name in uncertainty_layers:
                before = uncertainty_layers[uncertainty_name].copy()
                uncertainty_layers[uncertainty_name][attack_mask] = np.clip(
                    uncertainty_layers[uncertainty_name][attack_mask] + uncertainty_delta,
                    0.0,
                    1.0,
                )
                layer_cost_increase += float((uncertainty_layers[uncertainty_name] - before)[attack_mask].sum())
        elif name in layers and name != "distance":
            before = layers[name].copy()
            layers[name][attack_mask] = np.clip(layers[name][attack_mask] + cost_delta, 0.0, 1.0)
            layer_cost_increase += float((layers[name] - before)[attack_mask].sum())

    metadata = {
        "environment_attack_type": "env_path_corridor_attack",
        "reference_policy": str(config.get("reference_policy", "heuristic")),
        "reference_policy_resolved": reference_config["reference_policy_resolved"],
        "corridor_radius": corridor_radius,
        "attack_strength": attack_strength,
        "affected_layers": list(affected_layers),
        "attacked_corridor_cells": int(attack_mask.sum()),
        "attacked_layer_cost_increase": float(layer_cost_increase),
        "reference_path_length": int(reference_result.get("path_length", 0)),
    }
    attacked_costmap = replace(
        costmap,
        layers=layers,
        uncertainty_layers=uncertainty_layers,
        attack_mask=attack_mask,
        attack_metadata=metadata,
    )
    return replace(episode, costmap=attacked_costmap)


def _top_fraction_mask(values: np.ndarray, obstacle_mask: np.ndarray, top_fraction: float) -> np.ndarray:
    free = ~np.asarray(obstacle_mask, dtype=bool)
    if not free.any():
        return np.zeros_like(free, dtype=bool)
    fraction = float(np.clip(top_fraction, 0.0, 1.0))
    if fraction <= 0.0:
        return np.zeros_like(free, dtype=bool)
    selector = np.asarray(values, dtype=np.float32)
    threshold = float(np.quantile(selector[free], max(0.0, 1.0 - fraction)))
    return (selector >= threshold) & free


def _confidence_layers_from_uncertainty(
    costmap: GeneratedCostMap,
    min_confidence: float,
    max_confidence: float,
) -> dict[str, np.ndarray]:
    """Convert objective uncertainty layers into planner-visible confidence."""

    min_value = float(np.clip(min_confidence, 0.0, 1.0))
    max_value = float(np.clip(max_confidence, min_value, 1.0))
    confidence_layers: dict[str, np.ndarray] = {}
    for name in OBJECTIVE_NAMES:
        uncertainty = np.asarray(costmap.uncertainty_layers[name], dtype=np.float32)
        confidence_layers[name] = np.clip(1.0 - uncertainty, min_value, max_value).astype(np.float32)
    return confidence_layers


def _smooth_unit_field(
    shape: tuple[int, int],
    rng: np.random.Generator,
    sigma: float,
    free_mask: np.ndarray,
) -> np.ndarray:
    """Generate a spatially correlated field in [0, 1] over free cells."""

    raw = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    sigma = max(float(sigma), 0.0)
    smooth = gaussian_filter(raw, sigma=sigma, mode="reflect") if sigma > 0.0 else raw
    smooth = np.asarray(smooth, dtype=np.float32)
    values = smooth[free_mask]
    if values.size == 0:
        return np.zeros(shape, dtype=np.float32)
    low = float(np.percentile(values, 5.0))
    high = float(np.percentile(values, 95.0))
    if high - low < 1e-8:
        return np.zeros(shape, dtype=np.float32)
    field = np.clip((smooth - low) / (high - low), 0.0, 1.0)
    field[~free_mask] = 0.0
    return field.astype(np.float32)


def _belief_mismatch_selector(
    episode: PlanningEpisode,
    true_costmap: GeneratedCostMap,
    config: dict[str, Any],
    confidence_layers: dict[str, np.ndarray],
    affected_cost_layers: tuple[str, ...],
) -> np.ndarray:
    """Select cells whose map values will be unreliable in the belief map."""

    selection_mode = str(config.get("selection_mode", "low_confidence_high_consequence"))
    shape = true_costmap.layers["distance"].shape
    free = ~np.asarray(true_costmap.obstacle_mask, dtype=bool)
    top_fraction = float(config.get("top_fraction", 0.25))

    if selection_mode == "path_corridor":
        reference_episode = replace(
            episode,
            costmap=true_costmap,
            true_costmap=None,
            confidence_layers=None,
        )
        reference_config = _reference_config_from_policy(reference_episode, config)
        reference_result = plan_with_weights(
            reference_episode,
            reference_config["weights"],
            lambda_uncertainty=float(reference_config["lambda_uncertainty"]),
            allow_diagonal=True,
        )
        corridor_radius = int(config.get("corridor_radius", 2))
        attack_mask = _corridor_mask_from_path(reference_result.get("path"), shape, corridor_radius)
        return attack_mask & free

    if selection_mode == "slope_risk":
        selector = getattr(true_costmap, "slope_layer", None)
        if selector is None:
            selector = true_costmap.layers.get("energy", np.zeros(shape, dtype=np.float32))
        return _top_fraction_mask(np.asarray(selector, dtype=np.float32), true_costmap.obstacle_mask, top_fraction)

    scoring_layers = [
        name
        for name in affected_cost_layers
        if name in true_costmap.layers and name in confidence_layers and name != "distance"
    ]
    if not scoring_layers:
        scoring_layers = [name for name in OBJECTIVE_NAMES if name in confidence_layers]

    unreliability = np.mean(
        np.stack([1.0 - confidence_layers[name] for name in scoring_layers], axis=0),
        axis=0,
    )
    if selection_mode == "low_confidence":
        selector = unreliability
    elif selection_mode in {"low_confidence_high_cost", "low_confidence_high_consequence"}:
        consequence = np.mean(
            np.stack([np.asarray(true_costmap.layers[name], dtype=np.float32) for name in scoring_layers], axis=0),
            axis=0,
        )
        selector = unreliability * (0.35 + 0.65 * consequence)
    elif selection_mode == "deceptive_shortcut":
        consequence = np.mean(
            np.stack([np.asarray(true_costmap.layers[name], dtype=np.float32) for name in scoring_layers], axis=0),
            axis=0,
        )
        selector = unreliability * (0.35 + 0.65 * consequence)
        distance = np.asarray(true_costmap.layers["distance"], dtype=np.float32)
        attraction = float(config.get("shortcut_attraction", 0.75))
        consequence_weight = float(config.get("shortcut_consequence_weight", 0.12))
        reference_cost = np.clip(
            distance + consequence_weight * consequence - attraction * selector,
            1e-4,
            1.0,
        ).astype(np.float32)
        path = weighted_astar(
            reference_cost,
            true_costmap.obstacle_mask,
            true_costmap.start,
            true_costmap.goal,
            allow_diagonal=True,
        )
        corridor_radius = int(config.get("corridor_radius", 3))
        return _corridor_mask_from_path(path, shape, corridor_radius) & free
    else:
        raise ValueError(
            "env_belief_mismatch selection_mode must be one of "
            "'low_confidence', 'low_confidence_high_consequence', 'low_confidence_high_cost', "
            "'path_corridor', 'deceptive_shortcut', or 'slope_risk'"
        )
    return _top_fraction_mask(selector, true_costmap.obstacle_mask, top_fraction)


def _unit_selector(values: np.ndarray, free_mask: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array) & np.asarray(free_mask, dtype=bool)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    finite_values = array[finite]
    if float(finite_values.min()) >= 0.0 and float(finite_values.max()) <= 1.0:
        out = np.clip(array, 0.0, 1.0)
    else:
        low = float(np.percentile(finite_values, 2.0))
        high = float(np.percentile(finite_values, 98.0))
        if high - low < 1e-8:
            return np.zeros_like(array, dtype=np.float32)
        out = np.clip((array - low) / (high - low), 0.0, 1.0)
    out = np.asarray(out, dtype=np.float32)
    out[~free_mask] = 0.0
    return out


def _traversability_boundary_pressure(
    costmap: GeneratedCostMap,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Score free cells close to the rover traversability boundary."""

    free_mask = ~np.asarray(costmap.obstacle_mask, dtype=bool)
    shape = costmap.layers["distance"].shape
    slope_ratio = np.zeros(shape, dtype=np.float32)
    slope_degrees = getattr(costmap, "slope_degrees", None)
    rover_max_slope = getattr(costmap, "rover_max_slope_degrees", None)
    if slope_degrees is not None and rover_max_slope is not None:
        slope_degrees = np.asarray(slope_degrees, dtype=np.float32)
        rover_max_slope = np.maximum(np.asarray(rover_max_slope, dtype=np.float32), 1e-6)
        slope_ratio = np.clip(slope_degrees / rover_max_slope, 0.0, 1.5).astype(np.float32)
        lower_ratio = float(config.get("lower_slope_ratio", 0.42))
        denominator = max(1.0 - lower_ratio, 1e-6)
        slope_pressure = np.clip((slope_ratio - lower_ratio) / denominator, 0.0, 1.0)
    else:
        slope_layer = getattr(costmap, "slope_layer", None)
        if slope_layer is None:
            slope_layer = costmap.layers.get("energy", np.zeros(shape, dtype=np.float32))
        slope_pressure = _unit_selector(np.asarray(slope_layer, dtype=np.float32), free_mask)
        slope_ratio = slope_pressure.copy()

    roughness = getattr(costmap, "roughness_layer", None)
    if roughness is None:
        roughness_pressure = np.zeros(shape, dtype=np.float32)
    else:
        roughness_pressure = _unit_selector(np.asarray(roughness, dtype=np.float32), free_mask)

    energy_pressure = _unit_selector(costmap.layers.get("energy", np.zeros(shape, dtype=np.float32)), free_mask)
    hazard_pressure = _unit_selector(costmap.layers.get("hazard", np.zeros(shape, dtype=np.float32)), free_mask)
    pressure = (
        0.48 * slope_pressure
        + 0.16 * roughness_pressure
        + 0.18 * energy_pressure
        + 0.18 * hazard_pressure
    ).astype(np.float32)
    pressure[~free_mask] = 0.0

    top_fraction = float(config.get("top_fraction", 0.22))
    focus_mask = _top_fraction_mask(pressure, costmap.obstacle_mask, top_fraction)
    min_boundary_score = config.get("min_boundary_score")
    if min_boundary_score is not None:
        threshold_mask = pressure >= float(min_boundary_score)
        if np.any(focus_mask & threshold_mask):
            focus_mask &= threshold_mask

    selected = pressure[focus_mask]
    free_slope_ratio = slope_ratio[free_mask]
    selected_slope_ratio = slope_ratio[focus_mask]
    metadata = {
        "mean_boundary_pressure": float(selected.mean()) if selected.size else 0.0,
        "max_boundary_pressure": float(selected.max()) if selected.size else 0.0,
        "mean_free_slope_ratio": float(free_slope_ratio.mean()) if free_slope_ratio.size else 0.0,
        "mean_selected_slope_ratio": float(selected_slope_ratio.mean()) if selected_slope_ratio.size else 0.0,
        "max_selected_slope_ratio": float(selected_slope_ratio.max()) if selected_slope_ratio.size else 0.0,
        "near_slope_limit_fraction": float(np.mean(selected_slope_ratio >= 0.55)) if selected_slope_ratio.size else 0.0,
    }
    if slope_degrees is not None and np.any(focus_mask):
        selected_slope_degrees = np.asarray(slope_degrees, dtype=np.float32)[focus_mask]
        metadata["mean_selected_slope_deg"] = float(selected_slope_degrees.mean())
        metadata["p90_selected_slope_deg"] = float(np.percentile(selected_slope_degrees, 90.0))
    return focus_mask & free_mask, pressure, metadata


def _apply_traversability_boundary_mismatch_attack(
    episode: PlanningEpisode,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> PlanningEpisode:
    """Underestimate cost near slope/roughness cells close to traversability limits."""

    true_costmap = episode.true_costmap or episode.costmap
    min_confidence = float(config.get("min_confidence", 0.10))
    max_confidence = float(config.get("max_confidence", 0.95))
    confidence_layers = (
        {name: value.copy() for name, value in episode.confidence_layers.items()}
        if episode.confidence_layers
        else _confidence_layers_from_uncertainty(
            true_costmap,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
        )
    )

    affected_layers = tuple(str(name) for name in config.get("affected_layers", ("energy", "hazard")))
    affected_cost_layers = tuple(
        name for name in affected_layers if name in true_costmap.layers and name != "distance"
    )
    if not affected_cost_layers:
        affected_cost_layers = ("energy", "hazard")

    free_mask = ~np.asarray(true_costmap.obstacle_mask, dtype=bool)
    focus_mask, boundary_pressure, boundary_metadata = _traversability_boundary_pressure(true_costmap, config)
    focus_mask &= free_mask

    layers = {name: value.copy() for name, value in episode.costmap.layers.items()}
    uncertainty_layers = {
        name: value.copy()
        for name, value in episode.costmap.uncertainty_layers.items()
    }
    mode = str(config.get("mode", "risk_underestimate"))
    attack_strength = float(config.get("attack_strength", 3.0))
    strength_scale = max(0.0, attack_strength) / 3.0
    error_scale = float(config.get("error_scale", 0.55))
    background_error_scale = float(np.clip(config.get("background_error_scale", 0.03), 0.0, 1.0))
    confidence_gamma = float(max(config.get("confidence_gamma", 1.0), 0.1))
    correlation_sigma = float(config.get("correlation_sigma", 2.5))
    confidence_penalty_scale = float(config.get("confidence_penalty_scale", 0.30))
    confidence_floor = float(np.clip(config.get("confidence_floor", 0.08), 0.0, 1.0))
    confidence_to_uncertainty = bool(config.get("confidence_to_uncertainty", True))
    spatial_field = _smooth_unit_field(true_costmap.layers["distance"].shape, rng, correlation_sigma, free_mask)

    mismatch_scale = np.zeros_like(free_mask, dtype=np.float32)
    mismatch_scale[free_mask] = background_error_scale
    mismatch_scale[focus_mask] = 1.0
    mismatch_mask = mismatch_scale > 0.0
    boundary_term = (0.25 + 0.75 * np.asarray(boundary_pressure, dtype=np.float32))
    spatial_term = 0.35 + 0.65 * spatial_field

    if confidence_to_uncertainty:
        for name in OBJECTIVE_NAMES:
            if name in uncertainty_layers and name in confidence_layers:
                uncertainty_layers[name] = np.maximum(
                    np.asarray(uncertainty_layers[name], dtype=np.float32),
                    1.0 - np.asarray(confidence_layers[name], dtype=np.float32),
                ).astype(np.float32)

    total_abs_error = 0.0
    total_signed_gap = 0.0
    total_count = 0
    total_confidence_drop = 0.0
    for name in affected_cost_layers:
        true_layer = np.asarray(true_costmap.layers[name], dtype=np.float32)
        base_layer = np.asarray(layers[name], dtype=np.float32)
        confidence = np.asarray(confidence_layers.get(name, np.ones_like(true_layer)), dtype=np.float32)
        unreliability = np.clip(1.0 - confidence, 0.0, 1.0) ** confidence_gamma
        delta = (
            error_scale
            * strength_scale
            * boundary_term
            * spatial_term
            * (0.40 + 0.60 * unreliability)
            * mismatch_scale
        )

        if mode == "risk_underestimate":
            layers[name][mismatch_mask] = np.clip(base_layer[mismatch_mask] - delta[mismatch_mask], 0.0, 1.0)
        elif mode == "risk_overestimate":
            layers[name][mismatch_mask] = np.clip(base_layer[mismatch_mask] + delta[mismatch_mask], 0.0, 1.0)
        elif mode == "zero_mean_noise":
            sign = np.sign(gaussian_filter(rng.normal(0.0, 1.0, size=true_layer.shape), sigma=correlation_sigma))
            sign = np.where(sign == 0.0, 1.0, sign).astype(np.float32)
            layers[name][mismatch_mask] = np.clip(
                base_layer[mismatch_mask] + sign[mismatch_mask] * delta[mismatch_mask],
                0.0,
                1.0,
            )
        else:
            raise ValueError(
                "env_traversability_boundary_mismatch mode must be one of "
                "'risk_underestimate', 'risk_overestimate', or 'zero_mean_noise'"
            )

        if confidence_penalty_scale > 0.0 and name in confidence_layers:
            old_confidence = confidence.copy()
            confidence_drop = confidence_penalty_scale * boundary_term * spatial_term * mismatch_scale
            confidence[mismatch_mask] = np.clip(
                confidence[mismatch_mask] - confidence_drop[mismatch_mask],
                confidence_floor,
                1.0,
            )
            confidence_layers[name] = confidence.astype(np.float32)
            total_confidence_drop += float((old_confidence - confidence)[mismatch_mask].sum())

        error = np.asarray(true_layer - layers[name], dtype=np.float32)
        total_abs_error += float(np.abs(error[mismatch_mask]).sum())
        total_signed_gap += float(error[mismatch_mask].sum())
        total_count += int(mismatch_mask.sum())
        if name in uncertainty_layers:
            uncertainty_layers[name][mismatch_mask] = np.maximum(
                uncertainty_layers[name][mismatch_mask],
                1.0 - confidence[mismatch_mask],
            )
        layers[name] = np.asarray(layers[name], dtype=np.float32)

    slope_underestimate_scale = float(config.get("slope_underestimate_scale", 0.45))
    belief_slope_degrees = getattr(episode.costmap, "slope_degrees", None)
    mean_slope_underestimate_deg = 0.0
    if belief_slope_degrees is not None and slope_underestimate_scale > 0.0:
        true_slope_degrees = np.asarray(getattr(true_costmap, "slope_degrees"), dtype=np.float32)
        belief_slope_degrees = np.asarray(belief_slope_degrees, dtype=np.float32).copy()
        slope_scale = slope_underestimate_scale * boundary_term * spatial_term * mismatch_scale
        belief_slope_degrees[mismatch_mask] = np.clip(
            belief_slope_degrees[mismatch_mask] * (1.0 - slope_scale[mismatch_mask]),
            0.0,
            None,
        )
        mean_slope_underestimate_deg = float(
            np.mean((true_slope_degrees - belief_slope_degrees)[mismatch_mask])
            if np.any(mismatch_mask)
            else 0.0
        )

    metadata = {
        "environment_attack_type": "env_traversability_boundary_mismatch",
        "mode": mode,
        "selection_mode": "traversability_boundary",
        "top_fraction": float(config.get("top_fraction", 0.22)),
        "attack_strength": attack_strength,
        "error_scale": error_scale,
        "background_error_scale": background_error_scale,
        "confidence_gamma": confidence_gamma,
        "correlation_sigma": correlation_sigma,
        "confidence_penalty_scale": confidence_penalty_scale,
        "confidence_floor": confidence_floor,
        "slope_underestimate_scale": slope_underestimate_scale,
        "affected_layers": list(affected_cost_layers),
        "attacked_corridor_cells": int(focus_mask.sum()),
        "mismatched_cells": int(mismatch_mask.sum()),
        "attacked_layer_cost_increase": float(total_abs_error),
        "mean_belief_abs_error": float(total_abs_error / max(total_count, 1)),
        "mean_true_minus_belief_error": float(total_signed_gap / max(total_count, 1)),
        "mean_confidence_drop": float(total_confidence_drop / max(total_count, 1)),
        "mean_slope_underestimate_deg": mean_slope_underestimate_deg,
        "true_belief_mismatch": True,
        **boundary_metadata,
    }
    belief_costmap = replace(
        episode.costmap,
        layers=layers,
        uncertainty_layers=uncertainty_layers,
        attack_mask=focus_mask,
        attack_metadata=metadata,
        slope_degrees=belief_slope_degrees,
    )
    return replace(
        episode,
        costmap=belief_costmap,
        true_costmap=true_costmap,
        confidence_layers=confidence_layers,
    )


def _apply_true_terrain_degradation_attack(
    episode: PlanningEpisode,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> PlanningEpisode:
    """Increase true traversal cost in terrain-degradation regions unseen by the belief map."""

    belief_costmap = episode.costmap
    base_true_costmap = episode.true_costmap or episode.costmap
    affected_layers = tuple(str(name) for name in config.get("affected_layers", ("energy", "hazard")))
    affected_cost_layers = tuple(
        name for name in affected_layers if name in base_true_costmap.layers and name != "distance"
    )
    if not affected_cost_layers:
        affected_cost_layers = ("energy", "hazard")

    free_mask = ~np.asarray(base_true_costmap.obstacle_mask, dtype=bool)
    focus_mask, boundary_pressure, boundary_metadata = _traversability_boundary_pressure(base_true_costmap, config)
    focus_mask &= free_mask

    true_layers = {name: value.copy() for name, value in base_true_costmap.layers.items()}
    true_uncertainty_layers = {
        name: value.copy()
        for name, value in base_true_costmap.uncertainty_layers.items()
    }
    degradation_scale = float(config.get("degradation_scale", 0.35))
    background_degradation_scale = float(np.clip(config.get("background_degradation_scale", 0.02), 0.0, 1.0))
    correlation_sigma = float(config.get("correlation_sigma", 3.0))
    spatial_field = _smooth_unit_field(base_true_costmap.layers["distance"].shape, rng, correlation_sigma, free_mask)
    degradation_weight = np.zeros_like(free_mask, dtype=np.float32)
    degradation_weight[free_mask] = background_degradation_scale
    degradation_weight[focus_mask] = 1.0
    degradation_mask = degradation_weight > 0.0
    terrain_term = 0.30 + 0.70 * np.asarray(boundary_pressure, dtype=np.float32)
    spatial_term = 0.35 + 0.65 * spatial_field

    total_abs_error = 0.0
    total_signed_gap = 0.0
    total_count = 0
    for name in affected_cost_layers:
        belief_layer = np.asarray(belief_costmap.layers[name], dtype=np.float32)
        true_layer = np.asarray(true_layers[name], dtype=np.float32)
        delta = degradation_scale * terrain_term * spatial_term * degradation_weight
        true_layer = true_layer.copy()
        true_layer[degradation_mask] = np.clip(true_layer[degradation_mask] + delta[degradation_mask], 0.0, 1.0)
        true_layers[name] = true_layer.astype(np.float32)
        error = np.asarray(true_layer - belief_layer, dtype=np.float32)
        total_abs_error += float(np.abs(error[degradation_mask]).sum())
        total_signed_gap += float(error[degradation_mask].sum())
        total_count += int(degradation_mask.sum())
        if bool(config.get("raise_true_uncertainty", True)) and name in true_uncertainty_layers:
            true_uncertainty_layers[name][degradation_mask] = np.maximum(
                np.asarray(true_uncertainty_layers[name], dtype=np.float32)[degradation_mask],
                np.clip(delta[degradation_mask], 0.0, 1.0),
            )

    metadata = {
        "environment_attack_type": "env_true_terrain_degradation",
        "selection_mode": "traversability_boundary",
        "top_fraction": float(config.get("top_fraction", 0.30)),
        "lower_slope_ratio": float(config.get("lower_slope_ratio", 0.32)),
        "degradation_scale": degradation_scale,
        "background_degradation_scale": background_degradation_scale,
        "correlation_sigma": correlation_sigma,
        "affected_layers": list(affected_cost_layers),
        "attacked_corridor_cells": int(focus_mask.sum()),
        "mismatched_cells": int(degradation_mask.sum()),
        "attacked_layer_cost_increase": float(total_abs_error),
        "mean_belief_abs_error": float(total_abs_error / max(total_count, 1)),
        "mean_true_minus_belief_error": float(total_signed_gap / max(total_count, 1)),
        "true_belief_mismatch": True,
        **boundary_metadata,
    }
    true_costmap = replace(
        base_true_costmap,
        layers=true_layers,
        uncertainty_layers=true_uncertainty_layers,
        attack_mask=focus_mask,
        attack_metadata=metadata,
    )
    belief_with_metadata = replace(
        belief_costmap,
        attack_mask=focus_mask,
        attack_metadata=metadata,
    )
    confidence_layers = (
        {name: value.copy() for name, value in episode.confidence_layers.items()}
        if episode.confidence_layers
        else _confidence_layers_from_uncertainty(
            belief_costmap,
            min_confidence=float(config.get("min_confidence", 0.10)),
            max_confidence=float(config.get("max_confidence", 0.95)),
        )
    )
    return replace(
        episode,
        costmap=belief_with_metadata,
        true_costmap=true_costmap,
        confidence_layers=confidence_layers,
    )


def _apply_belief_mismatch_attack(
    episode: PlanningEpisode,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> PlanningEpisode:
    """Create a planner belief map that can disagree with the true map.

    The original episode costmap remains the true terrain used for evaluation.
    The returned episode exposes a corrupted belief map to the policy/planner,
    plus confidence layers that explain where the belief is unreliable.
    """

    true_costmap = episode.true_costmap or episode.costmap
    min_confidence = float(config.get("min_confidence", 0.15))
    max_confidence = float(config.get("max_confidence", 0.95))
    confidence_layers = _confidence_layers_from_uncertainty(
        true_costmap,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )

    affected_layers = tuple(str(name) for name in config.get("affected_layers", ("hazard", "energy")))
    affected_cost_layers = tuple(
        name for name in affected_layers if name in true_costmap.layers and name != "distance"
    )
    if not affected_cost_layers:
        affected_cost_layers = ("hazard", "energy")

    attack_mask = _belief_mismatch_selector(
        episode,
        true_costmap,
        config,
        confidence_layers,
        affected_cost_layers,
    )
    attack_mask &= ~true_costmap.obstacle_mask

    layers = {name: value.copy() for name, value in true_costmap.layers.items()}
    uncertainty_layers = {
        name: value.copy()
        for name, value in true_costmap.uncertainty_layers.items()
    }
    mode = str(config.get("mode", "risk_underestimate"))
    attack_strength = float(config.get("attack_strength", 3.0))
    strength_scale = max(0.0, attack_strength) / 3.0
    error_scale = float(config.get("error_scale", 0.45))
    background_error_scale = float(np.clip(config.get("background_error_scale", 0.25), 0.0, 1.0))
    confidence_to_uncertainty = bool(config.get("confidence_to_uncertainty", True))
    free_mask = ~np.asarray(true_costmap.obstacle_mask, dtype=bool)
    mismatch_scale = np.zeros_like(free_mask, dtype=np.float32)
    mismatch_scale[free_mask] = background_error_scale
    mismatch_scale[attack_mask] = 1.0
    mismatch_mask = mismatch_scale > 0.0

    if confidence_to_uncertainty:
        for name in OBJECTIVE_NAMES:
            uncertainty_layers[name] = np.maximum(
                np.asarray(uncertainty_layers[name], dtype=np.float32),
                1.0 - confidence_layers[name],
            ).astype(np.float32)

    total_abs_error = 0.0
    total_signed_gap = 0.0
    total_count = 0
    for name in affected_cost_layers:
        true_layer = np.asarray(true_costmap.layers[name], dtype=np.float32)
        confidence = confidence_layers.get(name)
        if confidence is None:
            confidence = np.ones_like(true_layer, dtype=np.float32)
        unreliability = np.clip(1.0 - np.asarray(confidence, dtype=np.float32), 0.0, 1.0)
        delta = error_scale * strength_scale * unreliability * mismatch_scale

        if mode == "risk_underestimate":
            layers[name][mismatch_mask] = np.clip(true_layer[mismatch_mask] - delta[mismatch_mask], 0.0, 1.0)
        elif mode == "risk_overestimate":
            layers[name][mismatch_mask] = np.clip(true_layer[mismatch_mask] + delta[mismatch_mask], 0.0, 1.0)
        elif mode == "zero_mean_noise":
            noise = rng.normal(0.0, 1.0, size=true_layer.shape).astype(np.float32)
            layers[name][mismatch_mask] = np.clip(
                true_layer[mismatch_mask] + noise[mismatch_mask] * delta[mismatch_mask],
                0.0,
                1.0,
            )
        else:
            raise ValueError(
                "env_belief_mismatch mode must be one of "
                "'risk_underestimate', 'risk_overestimate', or 'zero_mean_noise'"
            )

        error = np.asarray(true_layer - layers[name], dtype=np.float32)
        total_abs_error += float(np.abs(error[mismatch_mask]).sum())
        total_signed_gap += float(error[mismatch_mask].sum())
        total_count += int(mismatch_mask.sum())
        if name in uncertainty_layers:
            uncertainty_layers[name][mismatch_mask] = np.maximum(
                uncertainty_layers[name][mismatch_mask],
                unreliability[mismatch_mask],
            )
        layers[name] = np.asarray(layers[name], dtype=np.float32)

    attacked_cells = int(attack_mask.sum())
    mismatched_cells = int(mismatch_mask.sum())
    confidence_values = []
    for name in affected_cost_layers:
        confidence_values.append(np.asarray(confidence_layers[name], dtype=np.float32)[attack_mask])
    if confidence_values and attacked_cells > 0:
        selected_confidence = np.concatenate([values.reshape(-1) for values in confidence_values])
        mean_confidence = float(selected_confidence.mean())
        min_selected_confidence = float(selected_confidence.min())
    else:
        mean_confidence = float("nan")
        min_selected_confidence = float("nan")

    metadata = {
        "environment_attack_type": "env_belief_mismatch",
        "mode": mode,
        "selection_mode": str(config.get("selection_mode", "low_confidence_high_consequence")),
        "top_fraction": float(config.get("top_fraction", 0.25)),
        "attack_strength": attack_strength,
        "error_scale": error_scale,
        "background_error_scale": background_error_scale,
        "affected_layers": list(affected_cost_layers),
        "attacked_corridor_cells": attacked_cells,
        "mismatched_cells": mismatched_cells,
        "attacked_layer_cost_increase": float(total_abs_error),
        "mean_belief_abs_error": float(total_abs_error / max(total_count, 1)),
        "mean_true_minus_belief_error": float(total_signed_gap / max(total_count, 1)),
        "mean_selected_confidence": mean_confidence,
        "min_selected_confidence": min_selected_confidence,
        "true_belief_mismatch": True,
    }
    belief_costmap = replace(
        true_costmap,
        layers=layers,
        uncertainty_layers=uncertainty_layers,
        attack_mask=attack_mask,
        attack_metadata=metadata,
    )
    return replace(
        episode,
        costmap=belief_costmap,
        true_costmap=true_costmap,
        confidence_layers=confidence_layers,
    )


def _apply_spatial_belief_mismatch_attack(
    episode: PlanningEpisode,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> PlanningEpisode:
    """Apply a spatially correlated true-vs-belief map error.

    This models region-scale DEM/cost-layer estimation error rather than
    independent cell noise. The true map is preserved for reward/evaluation.
    """

    true_costmap = episode.true_costmap or episode.costmap
    min_confidence = float(config.get("min_confidence", 0.10))
    max_confidence = float(config.get("max_confidence", 0.95))
    confidence_layers = (
        {name: value.copy() for name, value in episode.confidence_layers.items()}
        if episode.confidence_layers
        else _confidence_layers_from_uncertainty(
            true_costmap,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
        )
    )

    affected_layers = tuple(str(name) for name in config.get("affected_layers", ("hazard", "energy")))
    affected_cost_layers = tuple(
        name for name in affected_layers if name in true_costmap.layers and name != "distance"
    )
    if not affected_cost_layers:
        affected_cost_layers = ("hazard", "energy")

    free_mask = ~np.asarray(true_costmap.obstacle_mask, dtype=bool)
    focus_mask = _belief_mismatch_selector(
        episode,
        true_costmap,
        config,
        confidence_layers,
        affected_cost_layers,
    )
    focus_mask &= free_mask

    layers = {name: value.copy() for name, value in episode.costmap.layers.items()}
    uncertainty_layers = {
        name: value.copy()
        for name, value in episode.costmap.uncertainty_layers.items()
    }
    mode = str(config.get("mode", "risk_underestimate"))
    attack_strength = float(config.get("attack_strength", 3.0))
    strength_scale = max(0.0, attack_strength) / 3.0
    error_scale = float(config.get("error_scale", 0.30))
    background_error_scale = float(np.clip(config.get("background_error_scale", 0.15), 0.0, 1.0))
    confidence_gamma = float(max(config.get("confidence_gamma", 1.25), 0.1))
    correlation_sigma = float(config.get("correlation_sigma", 3.5))
    confidence_to_uncertainty = bool(config.get("confidence_to_uncertainty", True))
    mismatch_scale = np.zeros_like(free_mask, dtype=np.float32)
    mismatch_scale[free_mask] = background_error_scale
    mismatch_scale[focus_mask] = 1.0
    mismatch_mask = mismatch_scale > 0.0
    spatial_field = _smooth_unit_field(true_costmap.layers["distance"].shape, rng, correlation_sigma, free_mask)

    if confidence_to_uncertainty:
        for name in OBJECTIVE_NAMES:
            if name in uncertainty_layers and name in confidence_layers:
                uncertainty_layers[name] = np.maximum(
                    np.asarray(uncertainty_layers[name], dtype=np.float32),
                    1.0 - np.asarray(confidence_layers[name], dtype=np.float32),
                ).astype(np.float32)

    total_abs_error = 0.0
    total_signed_gap = 0.0
    total_count = 0
    for name in affected_cost_layers:
        true_layer = np.asarray(true_costmap.layers[name], dtype=np.float32)
        base_layer = np.asarray(layers[name], dtype=np.float32)
        confidence = np.asarray(confidence_layers.get(name, np.ones_like(true_layer)), dtype=np.float32)
        unreliability = np.clip(1.0 - confidence, 0.0, 1.0) ** confidence_gamma
        delta = error_scale * strength_scale * (0.25 + 0.75 * spatial_field) * unreliability * mismatch_scale

        if mode == "risk_underestimate":
            layers[name][mismatch_mask] = np.clip(base_layer[mismatch_mask] - delta[mismatch_mask], 0.0, 1.0)
        elif mode == "risk_overestimate":
            layers[name][mismatch_mask] = np.clip(base_layer[mismatch_mask] + delta[mismatch_mask], 0.0, 1.0)
        elif mode == "zero_mean_noise":
            sign = np.sign(gaussian_filter(rng.normal(0.0, 1.0, size=true_layer.shape), sigma=correlation_sigma))
            sign = np.where(sign == 0.0, 1.0, sign).astype(np.float32)
            layers[name][mismatch_mask] = np.clip(
                base_layer[mismatch_mask] + sign[mismatch_mask] * delta[mismatch_mask],
                0.0,
                1.0,
            )
        else:
            raise ValueError(
                "env_spatial_belief_mismatch mode must be one of "
                "'risk_underestimate', 'risk_overestimate', or 'zero_mean_noise'"
            )

        error = np.asarray(true_layer - layers[name], dtype=np.float32)
        total_abs_error += float(np.abs(error[mismatch_mask]).sum())
        total_signed_gap += float(error[mismatch_mask].sum())
        total_count += int(mismatch_mask.sum())
        if name in uncertainty_layers:
            uncertainty_layers[name][mismatch_mask] = np.maximum(
                uncertainty_layers[name][mismatch_mask],
                np.clip(1.0 - confidence[mismatch_mask], 0.0, 1.0),
            )
        layers[name] = np.asarray(layers[name], dtype=np.float32)

    confidence_values = [
        np.asarray(confidence_layers[name], dtype=np.float32)[focus_mask]
        for name in affected_cost_layers
        if name in confidence_layers
    ]
    if confidence_values and int(focus_mask.sum()) > 0:
        selected_confidence = np.concatenate([values.reshape(-1) for values in confidence_values])
        mean_confidence = float(selected_confidence.mean())
        min_selected_confidence = float(selected_confidence.min())
    else:
        mean_confidence = float("nan")
        min_selected_confidence = float("nan")

    metadata = {
        "environment_attack_type": "env_spatial_belief_mismatch",
        "mode": mode,
        "selection_mode": str(config.get("selection_mode", "low_confidence_high_consequence")),
        "top_fraction": float(config.get("top_fraction", 0.25)),
        "attack_strength": attack_strength,
        "error_scale": error_scale,
        "background_error_scale": background_error_scale,
        "confidence_gamma": confidence_gamma,
        "correlation_sigma": correlation_sigma,
        "affected_layers": list(affected_cost_layers),
        "attacked_corridor_cells": int(focus_mask.sum()),
        "mismatched_cells": int(mismatch_mask.sum()),
        "attacked_layer_cost_increase": float(total_abs_error),
        "mean_belief_abs_error": float(total_abs_error / max(total_count, 1)),
        "mean_true_minus_belief_error": float(total_signed_gap / max(total_count, 1)),
        "mean_selected_confidence": mean_confidence,
        "min_selected_confidence": min_selected_confidence,
        "spatial_error_mean": float(spatial_field[free_mask].mean()) if free_mask.any() else 0.0,
        "spatial_error_std": float(spatial_field[free_mask].std()) if free_mask.any() else 0.0,
        "true_belief_mismatch": True,
    }
    belief_costmap = replace(
        episode.costmap,
        layers=layers,
        uncertainty_layers=uncertainty_layers,
        attack_mask=focus_mask,
        attack_metadata=metadata,
    )
    return replace(
        episode,
        costmap=belief_costmap,
        true_costmap=true_costmap,
        confidence_layers=confidence_layers,
    )


def _apply_confidence_degradation_attack(
    episode: PlanningEpisode,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> PlanningEpisode:
    """Reduce visible map confidence and raise belief uncertainty in coherent regions."""

    true_costmap = episode.true_costmap or episode.costmap
    confidence_layers = (
        {name: value.copy() for name, value in episode.confidence_layers.items()}
        if episode.confidence_layers
        else _confidence_layers_from_uncertainty(
            true_costmap,
            min_confidence=float(config.get("min_confidence", 0.10)),
            max_confidence=float(config.get("max_confidence", 0.95)),
        )
    )
    affected_layers = tuple(str(name) for name in config.get("affected_layers", OBJECTIVE_NAMES))
    affected_cost_layers = tuple(name for name in affected_layers if name in confidence_layers)
    if not affected_cost_layers:
        affected_cost_layers = OBJECTIVE_NAMES

    free_mask = ~np.asarray(true_costmap.obstacle_mask, dtype=bool)
    focus_mask = _belief_mismatch_selector(
        episode,
        true_costmap,
        config,
        confidence_layers,
        tuple(name for name in affected_cost_layers if name in true_costmap.layers and name != "distance"),
    )
    focus_mask &= free_mask

    correlation_sigma = float(config.get("correlation_sigma", 3.5))
    spatial_field = _smooth_unit_field(true_costmap.layers["distance"].shape, rng, correlation_sigma, free_mask)
    degradation_scale = float(config.get("degradation_scale", 0.35))
    background_scale = float(np.clip(config.get("background_degradation_scale", 0.08), 0.0, 1.0))
    confidence_floor = float(np.clip(config.get("confidence_floor", 0.05), 0.0, 1.0))
    confidence_gamma = float(max(config.get("confidence_gamma", 1.0), 0.1))
    degrade_scale = np.zeros_like(free_mask, dtype=np.float32)
    degrade_scale[free_mask] = background_scale
    degrade_scale[focus_mask] = 1.0
    degrade_mask = degrade_scale > 0.0

    uncertainty_layers = {
        name: value.copy()
        for name, value in episode.costmap.uncertainty_layers.items()
    }
    total_degradation = 0.0
    total_count = 0
    for name in affected_cost_layers:
        confidence = np.asarray(confidence_layers[name], dtype=np.float32)
        unreliability = np.clip(1.0 - confidence, 0.0, 1.0) ** confidence_gamma
        degradation = degradation_scale * (0.25 + 0.75 * spatial_field) * (0.35 + 0.65 * unreliability) * degrade_scale
        updated_confidence = confidence.copy()
        updated_confidence[degrade_mask] = np.clip(
            updated_confidence[degrade_mask] - degradation[degrade_mask],
            confidence_floor,
            1.0,
        )
        confidence_layers[name] = updated_confidence.astype(np.float32)
        if name in uncertainty_layers:
            uncertainty_layers[name] = np.maximum(
                np.asarray(uncertainty_layers[name], dtype=np.float32),
                1.0 - updated_confidence,
            ).astype(np.float32)
        total_degradation += float((confidence - updated_confidence)[degrade_mask].sum())
        total_count += int(degrade_mask.sum())

    metadata = {
        "environment_attack_type": "env_confidence_degradation",
        "selection_mode": str(config.get("selection_mode", "low_confidence_high_consequence")),
        "top_fraction": float(config.get("top_fraction", 0.25)),
        "degradation_scale": degradation_scale,
        "background_degradation_scale": background_scale,
        "confidence_floor": confidence_floor,
        "confidence_gamma": confidence_gamma,
        "correlation_sigma": correlation_sigma,
        "affected_layers": list(affected_cost_layers),
        "attacked_corridor_cells": int(focus_mask.sum()),
        "mismatched_cells": int(degrade_mask.sum()),
        "mean_confidence_degradation": float(total_degradation / max(total_count, 1)),
        "true_belief_mismatch": episode.true_costmap is not None,
    }
    belief_costmap = replace(
        episode.costmap,
        uncertainty_layers=uncertainty_layers,
        attack_mask=focus_mask,
        attack_metadata=metadata,
    )
    return replace(
        episode,
        costmap=belief_costmap,
        true_costmap=true_costmap,
        confidence_layers=confidence_layers,
    )


def _summarize_true_belief_gap(
    true_costmap: GeneratedCostMap,
    belief_costmap: GeneratedCostMap,
    affected_layers: tuple[str, ...],
    mask: np.ndarray,
) -> dict[str, float]:
    total_abs_error = 0.0
    total_signed_gap = 0.0
    total_count = 0
    for name in affected_layers:
        if name not in true_costmap.layers or name not in belief_costmap.layers or name == "distance":
            continue
        true_layer = np.asarray(true_costmap.layers[name], dtype=np.float32)
        belief_layer = np.asarray(belief_costmap.layers[name], dtype=np.float32)
        error = true_layer - belief_layer
        total_abs_error += float(np.abs(error[mask]).sum())
        total_signed_gap += float(error[mask].sum())
        total_count += int(mask.sum())
    return {
        "attacked_layer_cost_increase": float(total_abs_error),
        "mean_belief_abs_error": float(total_abs_error / max(total_count, 1)),
        "mean_true_minus_belief_error": float(total_signed_gap / max(total_count, 1)),
    }


def _apply_composite_environment_attack(
    episode: PlanningEpisode,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> PlanningEpisode:
    """Apply a sequence of physically related map-belief attack components."""

    components = config.get("components", [])
    if not isinstance(components, list) or not components:
        raise ValueError("env_composite requires a non-empty 'components' list")

    true_costmap = episode.true_costmap or episode.costmap
    current = replace(episode, true_costmap=true_costmap)
    component_metadata: list[dict[str, Any]] = []
    combined_mask = np.zeros_like(true_costmap.obstacle_mask, dtype=bool)
    affected_layers: list[str] = []

    for index, component in enumerate(components):
        if not isinstance(component, dict):
            raise ValueError("env_composite components must be JSON objects")
        component_config = dict(component)
        component_config.setdefault("enabled", True)
        if not attack_enabled(component_config):
            continue
        if str(component_config.get("type", "")) == "env_composite":
            raise ValueError("nested env_composite attacks are not supported")
        current = apply_environment_attack_to_episode(current, component_config, rng)
        metadata = getattr(current.costmap, "attack_metadata", None) or {}
        component_metadata.append({"index": index, **metadata})
        mask = getattr(current.costmap, "attack_mask", None)
        if mask is not None:
            combined_mask |= np.asarray(mask, dtype=bool)
        for name in metadata.get("affected_layers", component_config.get("affected_layers", [])):
            if name not in affected_layers:
                affected_layers.append(str(name))

    if not component_metadata:
        return episode

    free_mask = ~np.asarray(true_costmap.obstacle_mask, dtype=bool)
    if not combined_mask.any():
        combined_mask = free_mask
    combined_mask &= free_mask
    affected_tuple = tuple(affected_layers or [name for name in OBJECTIVE_NAMES if name != "distance"])
    gap = _summarize_true_belief_gap(true_costmap, current.costmap, affected_tuple, combined_mask)

    selected_confidences = []
    for name in affected_tuple:
        if current.confidence_layers and name in current.confidence_layers:
            selected_confidences.append(np.asarray(current.confidence_layers[name], dtype=np.float32)[combined_mask])
    if selected_confidences and combined_mask.any():
        selected = np.concatenate([values.reshape(-1) for values in selected_confidences])
        mean_selected_confidence = float(selected.mean())
        min_selected_confidence = float(selected.min())
    else:
        mean_selected_confidence = float("nan")
        min_selected_confidence = float("nan")

    metadata = {
        "environment_attack_type": "env_composite",
        "composite_attack_name": str(config.get("name", "composite_map_belief_attack")),
        "component_types": [str(item.get("environment_attack_type", "")) for item in component_metadata],
        "component_metadata": component_metadata,
        "affected_layers": list(affected_tuple),
        "attacked_corridor_cells": int(combined_mask.sum()),
        "mismatched_cells": int(combined_mask.sum()),
        "mean_selected_confidence": mean_selected_confidence,
        "min_selected_confidence": min_selected_confidence,
        "true_belief_mismatch": True,
        **gap,
    }
    belief_costmap = replace(
        current.costmap,
        attack_mask=combined_mask,
        attack_metadata=metadata,
    )
    return replace(
        current,
        costmap=belief_costmap,
        true_costmap=true_costmap,
    )


def _apply_region_inflation_attack(
    episode: PlanningEpisode,
    config: dict[str, Any],
    attack_type: str,
) -> PlanningEpisode:
    """Inflate soft cost/uncertainty in selected high-risk real-terrain regions."""

    costmap = episode.costmap
    if attack_type == "env_hazard_inflation":
        selector = costmap.layers.get("hazard", np.zeros_like(costmap.obstacle_mask, dtype=np.float32))
        affected_layers = tuple(config.get("affected_layers", ("hazard",)))
    elif attack_type == "env_uncertainty_inflation":
        selector = np.mean(
            np.stack([np.asarray(value, dtype=np.float32) for value in costmap.uncertainty_layers.values()], axis=0),
            axis=0,
        )
        affected_layers = tuple(config.get("affected_layers", ("uncertainty",)))
    else:
        selector = getattr(costmap, "slope_layer", None)
        if selector is None:
            selector = costmap.layers.get("energy", np.zeros_like(costmap.obstacle_mask, dtype=np.float32))
        affected_layers = tuple(config.get("affected_layers", ("hazard", "energy", "uncertainty")))

    top_fraction = float(config.get("top_fraction", config.get("attacker_top_fraction", 0.20)))
    attack_mask = _top_fraction_mask(selector, costmap.obstacle_mask, top_fraction)
    layers = {name: value.copy() for name, value in costmap.layers.items()}
    uncertainty_layers = {name: value.copy() for name, value in costmap.uncertainty_layers.items()}
    attack_strength = float(config.get("attack_strength", 3.0))
    cost_delta = float(config.get("cost_delta", 0.06 * attack_strength))
    uncertainty_delta = float(config.get("uncertainty_delta", 0.10 * attack_strength))
    layer_cost_increase = 0.0

    for name in affected_layers:
        name = str(name)
        if name == "uncertainty":
            for uncertainty_name in uncertainty_layers:
                before = uncertainty_layers[uncertainty_name].copy()
                uncertainty_layers[uncertainty_name][attack_mask] = np.clip(
                    uncertainty_layers[uncertainty_name][attack_mask] + uncertainty_delta,
                    0.0,
                    1.0,
                )
                layer_cost_increase += float((uncertainty_layers[uncertainty_name] - before)[attack_mask].sum())
        elif name.startswith("uncertainty:"):
            uncertainty_name = name.split(":", 1)[1]
            if uncertainty_name in uncertainty_layers:
                before = uncertainty_layers[uncertainty_name].copy()
                uncertainty_layers[uncertainty_name][attack_mask] = np.clip(
                    uncertainty_layers[uncertainty_name][attack_mask] + uncertainty_delta,
                    0.0,
                    1.0,
                )
                layer_cost_increase += float((uncertainty_layers[uncertainty_name] - before)[attack_mask].sum())
        elif name in layers and name != "distance":
            before = layers[name].copy()
            layers[name][attack_mask] = np.clip(layers[name][attack_mask] + cost_delta, 0.0, 1.0)
            layer_cost_increase += float((layers[name] - before)[attack_mask].sum())

    metadata = {
        "environment_attack_type": attack_type,
        "top_fraction": top_fraction,
        "attack_strength": attack_strength,
        "affected_layers": list(affected_layers),
        "attacked_corridor_cells": int(attack_mask.sum()),
        "attacked_layer_cost_increase": float(layer_cost_increase),
    }
    attacked_costmap = replace(
        costmap,
        layers=layers,
        uncertainty_layers=uncertainty_layers,
        attack_mask=attack_mask,
        attack_metadata=metadata,
    )
    return replace(episode, costmap=attacked_costmap)


def apply_environment_attack_to_episode(
    episode: PlanningEpisode,
    config: dict[str, Any] | None,
    rng: np.random.Generator,
) -> PlanningEpisode:
    """Return an episode with attacked cost/uncertainty layers.

    ``env_zscore_topk`` is implemented by the existing planner evaluation path
    and therefore does not mutate the episode. Layer attacks copy the frozen
    costmap and replace only the requested layer; hard obstacles are preserved.
    """

    if not attack_enabled(config):
        return episode

    attack_type = str(config.get("type", "env_zscore_topk"))
    if attack_type == "env_composite":
        return _apply_composite_environment_attack(episode, config, rng)
    if attack_type == "env_zscore_topk":
        return episode
    if attack_type == "env_path_corridor_attack":
        return _apply_path_corridor_attack(episode, config)
    if attack_type == "env_belief_mismatch":
        return _apply_belief_mismatch_attack(episode, config, rng)
    if attack_type == "env_spatial_belief_mismatch":
        return _apply_spatial_belief_mismatch_attack(episode, config, rng)
    if attack_type == "env_confidence_degradation":
        return _apply_confidence_degradation_attack(episode, config, rng)
    if attack_type == "env_traversability_boundary_mismatch":
        return _apply_traversability_boundary_mismatch_attack(episode, config, rng)
    if attack_type == "env_true_terrain_degradation":
        return _apply_true_terrain_degradation_attack(episode, config, rng)
    if attack_type in {"env_hazard_inflation", "env_uncertainty_inflation", "env_slope_risk_inflation"}:
        return _apply_region_inflation_attack(episode, config, attack_type)
    if attack_type not in {"env_layer_noise", "env_layer_bias"}:
        raise ValueError(f"environment attack type must be one of {ENVIRONMENT_ATTACK_TYPES}")

    layer_name = str(config.get("layer_name", "hazard"))
    layer_family, resolved_name = _resolve_layer_family(layer_name, config.get("layer_family"))
    costmap = episode.costmap

    layers = {name: value.copy() for name, value in costmap.layers.items()}
    uncertainty_layers = {
        name: value.copy()
        for name, value in costmap.uncertainty_layers.items()
    }

    if layer_family == "uncertainty":
        if resolved_name not in uncertainty_layers:
            raise ValueError(f"unknown uncertainty layer: {resolved_name}")
        uncertainty_layers[resolved_name] = _perturb_layer(
            uncertainty_layers[resolved_name],
            config,
            rng,
        )
    else:
        if resolved_name not in layers:
            valid = ", ".join(OBJECTIVE_NAMES)
            raise ValueError(f"unknown cost layer: {resolved_name}; expected one of {valid}")
        layers[resolved_name] = _perturb_layer(layers[resolved_name], config, rng)

    attacked_costmap = replace(
        costmap,
        layers=layers,
        uncertainty_layers=uncertainty_layers,
    )
    return replace(episode, costmap=attacked_costmap)


class EnvironmentAttackWrapper(gym.Wrapper):
    """Apply layer-level environmental attacks at reset time."""

    def __init__(
        self,
        env: gym.Env,
        config: dict[str, Any] | None,
    ) -> None:
        super().__init__(env)
        self.config = dict(config or {})
        self.rng = np.random.default_rng(self.config.get("seed"))

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        if not attack_enabled(self.config):
            return obs, info

        attack_type = str(self.config.get("type", "env_zscore_topk"))
        active_config = self.config
        selected_variant_id = ""
        selected_variant_scale = np.nan
        if attack_type == "env_attack_mixture":
            variants = list(self.config.get("variants", []))
            if not variants:
                raise ValueError("env_attack_mixture requires a non-empty variants list")
            probs = np.asarray(self.config.get("probs", []), dtype=np.float64)
            if probs.shape != (len(variants),):
                raise ValueError("env_attack_mixture probs must match variants length")
            probs = probs / max(float(probs.sum()), 1e-12)
            index = int(self.rng.choice(len(variants), p=probs))
            variant = dict(variants[index])
            active_config = dict(variant.get("config", {}))
            selected_variant_id = str(variant.get("variant_id", f"variant_{index}"))
            selected_variant_scale = float(variant.get("scale", np.nan))
            attack_type = str(active_config.get("type", "env_zscore_topk"))

        if attack_type == "env_zscore_topk":
            info = dict(info)
            info["environment_attack_type"] = attack_type
            if selected_variant_id:
                info["environment_attack_mixture_variant_id"] = selected_variant_id
                info["environment_attack_mixture_variant_scale"] = selected_variant_scale
            return obs, info

        episode = self.unwrapped.current_episode
        attacked_episode = apply_environment_attack_to_episode(episode, active_config, self.rng)
        if selected_variant_id:
            metadata = dict(getattr(attacked_episode.costmap, "attack_metadata", None) or {})
            metadata["environment_attack_mixture_variant_id"] = selected_variant_id
            metadata["environment_attack_mixture_variant_scale"] = selected_variant_scale
            attacked_episode = replace(
                attacked_episode,
                costmap=replace(attacked_episode.costmap, attack_metadata=metadata),
            )
        obs, info = self.env.reset(options={"episode": attacked_episode})
        info = dict(info)
        info["environment_attack_type"] = attack_type
        info["environment_attack_layer"] = active_config.get("layer_name", "")
        if selected_variant_id:
            info["environment_attack_mixture_variant_id"] = selected_variant_id
            info["environment_attack_mixture_variant_scale"] = selected_variant_scale
        attack_metadata = getattr(attacked_episode.costmap, "attack_metadata", None) or {}
        info.update(attack_metadata)
        return obs, info


def configure_env_attacker_kwargs(config: dict[str, Any] | None) -> dict[str, Any]:
    """Translate env_zscore_topk config keys into MultiObjectiveCostmapEnv kwargs."""

    if not attack_enabled(config):
        return {}
    if str(config.get("type", "env_zscore_topk")) != "env_zscore_topk":
        return {}
    kwargs: dict[str, Any] = {}
    for key in (
        "attacker_temperature",
        "attacker_response",
        "attacker_top_fraction",
        "attacker_sharpness",
        "attack_budget_fraction",
        "attack_strength",
    ):
        if key in config:
            kwargs[key] = config[key]
    return kwargs


def wrap_env_with_attacks(
    env: gym.Env,
    observation_attack: dict[str, Any] | None = None,
    environment_attack: dict[str, Any] | None = None,
) -> gym.Env:
    """Apply configured attack wrappers in environment-then-observation order."""

    wrapped = env
    if attack_enabled(environment_attack):
        wrapped = EnvironmentAttackWrapper(wrapped, environment_attack)
    if attack_enabled(observation_attack):
        wrapped = ObservationAttackWrapper(wrapped, observation_attack)
    return wrapped
