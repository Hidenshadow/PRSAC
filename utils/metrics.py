"""Metrics and common rover planning evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd

from maps.map_generator import SCENARIO_NAMES, GeneratedCostMap, generate_costmap
from planners.weighted_astar import weighted_astar


OBJECTIVE_NAMES = ("distance", "energy", "hazard", "communication", "illumination")
ROVER_STATE_NAMES = (
    "battery_budget",
    "hazard_tolerance",
    "min_communication_quality",
    "illumination_requirement",
)
MISSION_REGIME_NAMES = (
    "nominal",
    "energy_limited",
    "hazard_avoidance",
    "communication_critical",
    "illumination_critical",
    "uncertainty_sensitive",
)
FIXED_WEIGHTS = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
OBSERVATION_MODES = ("basic", "terrain", "extended")
ACTION_MODES = ("direct", "preference_delta")
MAP_SAMPLING_MODES = ("random", "fixed_map", "map_seed_pool")
REWARD_MODES = (
    "raw",
    "advantage_heuristic",
    "relative_heuristic",
)
WEIGHT_ACTION_DIM = len(OBJECTIVE_NAMES)
ACTION_DIM = WEIGHT_ACTION_DIM + 1
DEFAULT_MAX_UNCERTAINTY_LAMBDA = 2.0
DEFAULT_FIXED_MAP_SEED = 424242
DEFAULT_MAP_SEED_POOL_SIZE = 32
DEFAULT_ATTACK_BUDGET_FRACTION = 0.18
DEFAULT_ATTACK_STRENGTH = 1.0
ATTACKER_RESPONSE_MODES = ("softmax", "zscore_softmax", "zscore_topk", "rank_topk")
DEFAULT_ATTACKER_RESPONSE = "zscore_topk"
DEFAULT_ATTACKER_TEMPERATURE = 0.50
DEFAULT_ATTACKER_TOP_FRACTION = 0.15
DEFAULT_ATTACKER_SHARPNESS = 3.0
ATTACK_OBJECTIVE_SCALE = {
    "distance": 0.08,
    "energy": 0.18,
    "hazard": 0.24,
    "communication": 0.20,
    "illumination": 0.18,
}


@dataclass(frozen=True)
class PlanningEpisode:
    """One contextual rover planning problem."""

    costmap: GeneratedCostMap
    mission_priority: np.ndarray
    rover_state: dict[str, float]
    scenario: str = "nominal"
    mission_regime: str = "nominal"
    mission_severity: float = 0.0
    true_costmap: GeneratedCostMap | None = None
    confidence_layers: dict[str, np.ndarray] | None = None


def normalize_weights(action: np.ndarray) -> np.ndarray:
    """Clip and normalize a continuous vector into a non-negative simplex."""

    weights = np.asarray(action, dtype=np.float32).reshape(-1)
    if weights.shape[0] != len(OBJECTIVE_NAMES):
        raise ValueError(f"expected {len(OBJECTIVE_NAMES)} weights, got {weights.shape[0]}")

    weights = np.clip(weights, 0.0, 1.0)
    weights = weights + 1e-8
    weights = weights / weights.sum()
    return weights.astype(np.float32)


MISSION_REGIME_PROBS: dict[str, dict[str, float]] = {
    "nominal": {
        "nominal": 0.35,
        "energy_limited": 0.14,
        "hazard_avoidance": 0.18,
        "communication_critical": 0.12,
        "illumination_critical": 0.12,
        "uncertainty_sensitive": 0.09,
    },
    "lunar_polar_shadow": {
        "illumination_critical": 0.46,
        "energy_limited": 0.22,
        "hazard_avoidance": 0.16,
        "communication_critical": 0.08,
        "uncertainty_sensitive": 0.08,
    },
    "mars_dust_low_power": {
        "energy_limited": 0.38,
        "illumination_critical": 0.28,
        "communication_critical": 0.18,
        "uncertainty_sensitive": 0.10,
        "hazard_avoidance": 0.06,
    },
    "crater_rim_traverse": {
        "hazard_avoidance": 0.52,
        "energy_limited": 0.24,
        "uncertainty_sensitive": 0.14,
        "illumination_critical": 0.06,
        "communication_critical": 0.04,
    },
    "comm_blackout": {
        "communication_critical": 0.58,
        "energy_limited": 0.16,
        "hazard_avoidance": 0.12,
        "uncertainty_sensitive": 0.08,
        "illumination_critical": 0.06,
    },
    "uncertain_hazard_corridor": {
        "uncertainty_sensitive": 0.38,
        "hazard_avoidance": 0.36,
        "energy_limited": 0.14,
        "communication_critical": 0.06,
        "illumination_critical": 0.06,
    },
    "lunar_rover_corridor": {
        "hazard_avoidance": 0.34,
        "uncertainty_sensitive": 0.30,
        "energy_limited": 0.18,
        "illumination_critical": 0.10,
        "communication_critical": 0.08,
    },
    "lunar_rover_bottleneck": {
        "hazard_avoidance": 0.38,
        "uncertainty_sensitive": 0.28,
        "energy_limited": 0.16,
        "communication_critical": 0.10,
        "illumination_critical": 0.08,
    },
}

MISSION_PRIORITY_TEMPLATES: dict[str, np.ndarray] = {
    "nominal": np.array([0.10, 0.23, 0.25, 0.20, 0.22], dtype=np.float32),
    "energy_limited": np.array([0.05, 0.56, 0.17, 0.09, 0.13], dtype=np.float32),
    "hazard_avoidance": np.array([0.05, 0.17, 0.58, 0.09, 0.11], dtype=np.float32),
    "communication_critical": np.array([0.05, 0.14, 0.15, 0.55, 0.11], dtype=np.float32),
    "illumination_critical": np.array([0.05, 0.18, 0.13, 0.09, 0.55], dtype=np.float32),
    "uncertainty_sensitive": np.array([0.04, 0.22, 0.43, 0.14, 0.17], dtype=np.float32),
}


def _normalized_probs(probabilities: dict[str, float]) -> tuple[list[str], np.ndarray]:
    names = list(probabilities.keys())
    values = np.asarray([max(float(probabilities[name]), 0.0) for name in names], dtype=np.float64)
    if values.sum() <= 1e-12:
        values = np.ones(len(names), dtype=np.float64)
    values = values / values.sum()
    return names, values


def sample_mission_profile(
    rng: np.random.Generator,
    scenario: str = "nominal",
) -> tuple[np.ndarray, str, float]:
    """Sample a regime-structured rover mission profile.

    The previous experiment sampled mission priorities directly from a broad
    Dirichlet distribution. That produced many near-average tasks, where a
    static preference policy can look competitive. This sampler models
    realistic rover operations more explicitly: each episode has a mission
    regime such as low power, hazard avoidance, relay-critical communication,
    or illumination survival. The regime is still stochastic, but the sampled
    priority vector and severity are coherent with the map scenario.
    """

    regime_probs = MISSION_REGIME_PROBS.get(scenario, MISSION_REGIME_PROBS["nominal"])
    regime_names, probabilities = _normalized_probs(regime_probs)
    regime = str(rng.choice(regime_names, p=probabilities))

    if scenario == "nominal" and regime == "nominal":
        severity = float(rng.uniform(0.15, 0.45))
    else:
        severity = float(rng.uniform(0.58, 0.95))

    base = MISSION_PRIORITY_TEMPLATES["nominal"]
    target = MISSION_PRIORITY_TEMPLATES.get(regime, base)
    profile = (1.0 - severity) * base + severity * target
    profile = np.maximum(profile, 1e-4)
    profile = profile / profile.sum()

    # Higher severity means the operator has a clearer primary mission mode.
    # Dirichlet noise keeps episodes varied without hiding the regime switch.
    concentration = 34.0 + 42.0 * severity
    priority = rng.dirichlet(profile * concentration).astype(np.float32)
    return priority, regime, severity


def sample_mission_priority(
    rng: np.random.Generator,
    scenario: str = "nominal",
) -> np.ndarray:
    """Backward-compatible helper returning only mission priorities."""

    priority, _, _ = sample_mission_profile(rng, scenario)
    return priority


def sample_rover_state(
    rng: np.random.Generator,
    scenario: str = "nominal",
    mission_regime: str = "nominal",
    mission_priority: np.ndarray | None = None,
    mission_severity: float = 0.0,
) -> dict[str, float]:
    """Sample simple rover mission constraints normalized to [0, 1]."""

    ranges_by_scenario = {
        "nominal": {
            "battery_budget": (0.34, 0.72),
            "hazard_tolerance": (0.28, 0.68),
            "min_communication_quality": (0.20, 0.62),
            "illumination_requirement": (0.20, 0.68),
        },
        "lunar_polar_shadow": {
            "battery_budget": (0.30, 0.58),
            "hazard_tolerance": (0.24, 0.52),
            "min_communication_quality": (0.18, 0.56),
            "illumination_requirement": (0.56, 0.88),
        },
        "mars_dust_low_power": {
            "battery_budget": (0.24, 0.50),
            "hazard_tolerance": (0.30, 0.66),
            "min_communication_quality": (0.38, 0.72),
            "illumination_requirement": (0.48, 0.82),
        },
        "crater_rim_traverse": {
            "battery_budget": (0.30, 0.62),
            "hazard_tolerance": (0.18, 0.44),
            "min_communication_quality": (0.22, 0.58),
            "illumination_requirement": (0.28, 0.68),
        },
        "comm_blackout": {
            "battery_budget": (0.32, 0.66),
            "hazard_tolerance": (0.26, 0.60),
            "min_communication_quality": (0.54, 0.86),
            "illumination_requirement": (0.24, 0.62),
        },
        "uncertain_hazard_corridor": {
            "battery_budget": (0.30, 0.62),
            "hazard_tolerance": (0.20, 0.50),
            "min_communication_quality": (0.26, 0.66),
            "illumination_requirement": (0.26, 0.70),
        },
        "lunar_rover_corridor": {
            "battery_budget": (0.30, 0.62),
            "hazard_tolerance": (0.18, 0.46),
            "min_communication_quality": (0.24, 0.64),
            "illumination_requirement": (0.28, 0.72),
        },
        "lunar_rover_bottleneck": {
            "battery_budget": (0.28, 0.60),
            "hazard_tolerance": (0.16, 0.44),
            "min_communication_quality": (0.24, 0.66),
            "illumination_requirement": (0.26, 0.70),
        },
    }
    ranges = ranges_by_scenario.get(scenario, ranges_by_scenario["nominal"])
    base_state = {name: float(rng.uniform(low, high)) for name, (low, high) in ranges.items()}

    regime_ranges = {
        "energy_limited": {
            "battery_budget": (0.18, 0.40),
            "hazard_tolerance": (0.25, 0.56),
            "min_communication_quality": (0.30, 0.70),
            "illumination_requirement": (0.34, 0.76),
        },
        "hazard_avoidance": {
            "battery_budget": (0.26, 0.58),
            "hazard_tolerance": (0.14, 0.34),
            "min_communication_quality": (0.26, 0.66),
            "illumination_requirement": (0.30, 0.72),
        },
        "communication_critical": {
            "battery_budget": (0.26, 0.58),
            "hazard_tolerance": (0.22, 0.54),
            "min_communication_quality": (0.66, 0.92),
            "illumination_requirement": (0.28, 0.68),
        },
        "illumination_critical": {
            "battery_budget": (0.20, 0.48),
            "hazard_tolerance": (0.24, 0.56),
            "min_communication_quality": (0.24, 0.64),
            "illumination_requirement": (0.66, 0.92),
        },
        "uncertainty_sensitive": {
            "battery_budget": (0.24, 0.56),
            "hazard_tolerance": (0.16, 0.40),
            "min_communication_quality": (0.30, 0.72),
            "illumination_requirement": (0.32, 0.76),
        },
    }
    severity = float(np.clip(mission_severity, 0.0, 1.0))
    blend = 0.78 * severity
    if mission_regime in regime_ranges:
        for name, (low, high) in regime_ranges[mission_regime].items():
            regime_value = float(rng.uniform(low, high))
            base_state[name] = float((1.0 - blend) * base_state[name] + blend * regime_value)

    if mission_priority is not None:
        priority = np.asarray(mission_priority, dtype=np.float32).reshape(-1)
        if priority.size >= len(OBJECTIVE_NAMES):
            # Treat high-priority support resources as operational constraints,
            # not merely soft preferences. The clipping keeps values normalized.
            base_state["battery_budget"] = float(
                np.clip(
                    base_state["battery_budget"] - 0.08 * severity * float(priority[1]),
                    0.14,
                    0.95,
                )
            )
            base_state["hazard_tolerance"] = float(
                np.clip(
                    base_state["hazard_tolerance"] - 0.10 * severity * float(priority[2]),
                    0.10,
                    0.95,
                )
            )
            base_state["min_communication_quality"] = float(
                np.clip(
                    base_state["min_communication_quality"] + 0.12 * severity * float(priority[3]),
                    0.05,
                    0.95,
                )
            )
            base_state["illumination_requirement"] = float(
                np.clip(
                    base_state["illumination_requirement"] + 0.12 * severity * float(priority[4]),
                    0.05,
                    0.95,
                )
            )

    return base_state


def rover_state_vector(rover_state: dict[str, float]) -> np.ndarray:
    return np.array([rover_state[name] for name in ROVER_STATE_NAMES], dtype=np.float32)


def make_planning_episode(
    map_size: int,
    rng: np.random.Generator,
    obstacle_threshold: float = 0.88,
    allow_diagonal: bool = True,
    scenario: str = "nominal",
    min_start_goal_distance_ratio: float = 0.55,
) -> PlanningEpisode:
    """Generate one map, one mission priority vector, and rover constraints."""

    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"scenario must be one of {SCENARIO_NAMES}, got {scenario!r}")
    costmap = generate_costmap(
        map_size=map_size,
        rng=rng,
        obstacle_threshold=obstacle_threshold,
        min_start_goal_distance_ratio=min_start_goal_distance_ratio,
        allow_diagonal=allow_diagonal,
        scenario=scenario,
    )
    return make_planning_episode_from_costmap(costmap, rng)


def make_planning_episode_from_costmap(
    costmap: GeneratedCostMap,
    rng: np.random.Generator,
) -> PlanningEpisode:
    """Generate one mission profile and rover state on a pre-generated map."""

    resolved_scenario = costmap.scenario
    mission_priority, mission_regime, mission_severity = sample_mission_profile(rng, resolved_scenario)
    return PlanningEpisode(
        costmap=costmap,
        mission_priority=mission_priority,
        rover_state=sample_rover_state(
            rng,
            resolved_scenario,
            mission_regime=mission_regime,
            mission_priority=mission_priority,
            mission_severity=mission_severity,
        ),
        scenario=resolved_scenario,
        mission_regime=mission_regime,
        mission_severity=mission_severity,
    )


def _curriculum_costmap_cache_key(
    map_size: int,
    obstacle_threshold: float,
    min_start_goal_distance_ratio: float,
    allow_diagonal: bool,
    scenario: str,
    map_seed: int,
) -> tuple[int, float, float, bool, str, int]:
    return (
        int(map_size),
        round(float(obstacle_threshold), 6),
        round(float(min_start_goal_distance_ratio), 6),
        bool(allow_diagonal),
        str(scenario),
        int(map_seed),
    )


def make_curriculum_planning_episode(
    map_size: int,
    rng: np.random.Generator,
    obstacle_threshold: float = 0.88,
    allow_diagonal: bool = True,
    scenario: str = "nominal",
    min_start_goal_distance_ratio: float = 0.55,
    map_sampling_mode: str = "random",
    fixed_map_seed: int = DEFAULT_FIXED_MAP_SEED,
    map_seed_pool_size: int = DEFAULT_MAP_SEED_POOL_SIZE,
    map_cache: dict[tuple[int, float, float, bool, str, int], GeneratedCostMap] | None = None,
) -> PlanningEpisode:
    """Generate an episode with optional fixed-map curriculum sampling.

    ``random`` preserves the original behavior: each episode samples a fresh
    map, mission, and rover state. ``fixed_map`` reuses one generated map while
    still sampling new mission contexts. ``map_seed_pool`` samples maps from a
    deterministic finite seed pool.
    """

    if map_sampling_mode not in MAP_SAMPLING_MODES:
        raise ValueError(f"map_sampling_mode must be one of {MAP_SAMPLING_MODES}")

    if map_sampling_mode == "random":
        return make_planning_episode(
            map_size=map_size,
            rng=rng,
            obstacle_threshold=obstacle_threshold,
            allow_diagonal=allow_diagonal,
            scenario=scenario,
            min_start_goal_distance_ratio=min_start_goal_distance_ratio,
        )

    if map_sampling_mode == "fixed_map":
        map_seed = int(fixed_map_seed)
    else:
        pool_size = max(int(map_seed_pool_size), 1)
        map_seed = int(fixed_map_seed) + int(rng.integers(0, pool_size))

    cache_key = _curriculum_costmap_cache_key(
        map_size,
        obstacle_threshold,
        min_start_goal_distance_ratio,
        allow_diagonal,
        scenario,
        map_seed,
    )
    if map_cache is not None and cache_key in map_cache:
        costmap = map_cache[cache_key]
    else:
        costmap = generate_costmap(
            map_size=map_size,
            rng=np.random.default_rng(map_seed),
            obstacle_threshold=obstacle_threshold,
            min_start_goal_distance_ratio=min_start_goal_distance_ratio,
            allow_diagonal=allow_diagonal,
            scenario=scenario,
        )
        if map_cache is not None:
            map_cache[cache_key] = costmap

    return make_planning_episode_from_costmap(costmap, rng)


def action_to_planning_weights(
    episode: PlanningEpisode,
    action: np.ndarray,
    action_mode: str = "direct",
    action_gain: float = 2.0,
) -> np.ndarray:
    """Convert an RL action into planner weights.

    In ``direct`` mode the action itself is normalized as weights. In
    ``preference_delta`` mode the action modulates the mission priority:

        weights = normalize(mission_priority * exp(gain * (2a - 1)))

    This keeps the policy anchored to the requested mission while allowing it to
    adapt preferences to terrain and rover state.
    """

    if action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}")

    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.shape[0] < len(OBJECTIVE_NAMES):
        raise ValueError(
            f"expected at least {len(OBJECTIVE_NAMES)} action values, got {action_array.shape[0]}"
        )
    weight_action = action_array[: len(OBJECTIVE_NAMES)]

    if action_mode == "direct":
        return normalize_weights(weight_action)

    clipped = np.clip(weight_action, 0.0, 1.0)
    delta = (2.0 * clipped - 1.0) * float(action_gain)
    weights = np.asarray(episode.mission_priority, dtype=np.float32) * np.exp(delta)
    weights = np.maximum(weights, 1e-8)
    return (weights / weights.sum()).astype(np.float32)


def action_to_uncertainty_lambda(
    action: np.ndarray,
    max_uncertainty_lambda: float = DEFAULT_MAX_UNCERTAINTY_LAMBDA,
) -> float:
    """Map the optional sixth action dimension to robust-planning sensitivity."""

    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.shape[0] <= len(OBJECTIVE_NAMES):
        return 0.0
    sensitivity = float(np.clip(action_array[len(OBJECTIVE_NAMES)], 0.0, 1.0))
    return float(sensitivity * max(float(max_uncertainty_lambda), 0.0))


def _project_simplex_with_lower_bounds(
    weights: np.ndarray,
    lower_bounds: np.ndarray,
    max_floor_mass: float,
) -> np.ndarray:
    """Project a simplex vector onto lower-bound constraints.

    The projection is intentionally lightweight and monotone: reserve mass for
    safety floors, then distribute the remaining mass according to the original
    excess preference. This is suitable for an action shield where robustness
    and predictable behavior matter more than exact Euclidean projection.
    """

    weights = normalize_weights(weights)
    floors = np.maximum(np.asarray(lower_bounds, dtype=np.float32), 0.0)
    floor_mass = float(floors.sum())
    max_mass = float(np.clip(max_floor_mass, 0.05, 0.95))
    if floor_mass > max_mass:
        floors = floors * (max_mass / max(floor_mass, 1e-8))
        floor_mass = float(floors.sum())

    remaining = max(1.0 - floor_mass, 1e-8)
    excess = np.maximum(weights - floors, 0.0)
    if float(excess.sum()) <= 1e-8:
        excess = weights
    projected = floors + remaining * excess / max(float(excess.sum()), 1e-8)
    return normalize_weights(projected)


def safety_shield_planner_config(
    episode: PlanningEpisode,
    weights: np.ndarray,
    lambda_uncertainty: float,
    max_uncertainty_lambda: float = DEFAULT_MAX_UNCERTAINTY_LAMBDA,
    enabled: bool = False,
    strength: float = 0.55,
    max_floor_mass: float = 0.72,
) -> dict[str, Any]:
    """Apply a rover safety shield to planner weights and uncertainty lambda.

    This is a safe-RL style action layer, not reward shaping. It keeps the RL
    policy responsible for preference selection while enforcing minimum
    preference mass on objectives whose rover-state or uncertainty pressure is
    high. The layer is generic across lunar/Mars scenarios and uses only
    mission priority, rover state, and map uncertainty statistics.
    """

    raw_weights = normalize_weights(weights)
    raw_lambda = float(np.clip(lambda_uncertainty, 0.0, max(float(max_uncertainty_lambda), 0.0)))
    if not enabled:
        return {
            "weights": raw_weights,
            "lambda_uncertainty": raw_lambda,
            "safety_shield_active": False,
        }

    shield_strength = float(np.clip(strength, 0.0, 1.0))
    rover_state = episode.rover_state
    mission = np.asarray(episode.mission_priority, dtype=np.float32)
    free_mask = ~episode.costmap.obstacle_mask
    if not bool(free_mask.any()):
        free_mask = np.ones_like(episode.costmap.obstacle_mask, dtype=bool)
    uncertainty_mean = np.array(
        [
            float(episode.costmap.uncertainty_layers[name][free_mask].mean())
            for name in OBJECTIVE_NAMES
        ],
        dtype=np.float32,
    )

    energy_pressure = np.clip(
        mission[1]
        + max(0.0, 0.56 - float(rover_state["battery_budget"]))
        + 0.35 * uncertainty_mean[1]
        + 0.18 * mission[4],
        0.0,
        1.0,
    )
    hazard_pressure = np.clip(
        mission[2]
        + max(0.0, 0.60 - float(rover_state["hazard_tolerance"]))
        + 0.45 * uncertainty_mean[2],
        0.0,
        1.0,
    )
    communication_pressure = np.clip(
        mission[3]
        + max(0.0, float(rover_state["min_communication_quality"]) - 0.42)
        + 0.35 * uncertainty_mean[3],
        0.0,
        1.0,
    )
    illumination_pressure = np.clip(
        mission[4]
        + max(0.0, float(rover_state["illumination_requirement"]) - 0.42)
        + 0.35 * uncertainty_mean[4]
        + 0.16 * energy_pressure,
        0.0,
        1.0,
    )

    lower_bounds = np.array(
        [
            0.015,
            0.035 + 0.20 * energy_pressure,
            0.035 + 0.22 * hazard_pressure,
            0.025 + 0.18 * communication_pressure,
            0.025 + 0.18 * illumination_pressure,
        ],
        dtype=np.float32,
    )
    lower_bounds *= shield_strength
    shielded_weights = _project_simplex_with_lower_bounds(
        raw_weights,
        lower_bounds,
        max_floor_mass=max_floor_mass,
    )

    uncertainty_pressure = float(
        np.clip(
            0.30 * float(uncertainty_mean.mean())
            + 0.28 * hazard_pressure
            + 0.18 * energy_pressure
            + 0.14 * communication_pressure
            + 0.10 * illumination_pressure,
            0.0,
            1.0,
        )
    )
    lambda_floor_fraction = shield_strength * np.clip(0.12 + 0.58 * uncertainty_pressure, 0.0, 0.85)
    lambda_floor = float(lambda_floor_fraction * max(float(max_uncertainty_lambda), 0.0))
    shielded_lambda = max(raw_lambda, lambda_floor)
    shield_active = bool(
        np.max(np.abs(shielded_weights - raw_weights)) > 1e-6
        or shielded_lambda > raw_lambda + 1e-6
    )
    return {
        "weights": shielded_weights.astype(np.float32),
        "lambda_uncertainty": float(shielded_lambda),
        "safety_shield_active": shield_active,
        "safety_shield_lambda_floor": lambda_floor,
        "safety_shield_floor_mass": float(np.maximum(lower_bounds, 0.0).sum()),
        "safety_pressure_energy": float(energy_pressure),
        "safety_pressure_hazard": float(hazard_pressure),
        "safety_pressure_communication": float(communication_pressure),
        "safety_pressure_illumination": float(illumination_pressure),
    }


def candidate_weight_sets(episode: PlanningEpisode) -> dict[str, np.ndarray]:
    """Deterministic baseline weight presets used by optional diagnostics."""

    candidates: dict[str, np.ndarray] = {
        "fixed": FIXED_WEIGHTS,
        "heuristic": episode.mission_priority.astype(np.float32),
    }
    for index, name in enumerate(OBJECTIVE_NAMES):
        one_hot = np.zeros(len(OBJECTIVE_NAMES), dtype=np.float32)
        one_hot[index] = 1.0
        candidates[f"{name}_only"] = one_hot

    safe = np.array([0.12, 0.18, 0.38, 0.18, 0.14], dtype=np.float32)
    power_comms = np.array([0.10, 0.20, 0.18, 0.24, 0.28], dtype=np.float32)
    candidates["safe_rover"] = normalize_weights(safe)
    candidates["power_comms"] = normalize_weights(power_comms)

    mission = episode.mission_priority.astype(np.float32)
    rover_state = episode.rover_state
    guard = np.array(
        [
            0.05,
            mission[1] + max(0.0, 0.56 - float(rover_state["battery_budget"])),
            mission[2] + max(0.0, 0.60 - float(rover_state["hazard_tolerance"])),
            mission[3] + max(0.0, float(rover_state["min_communication_quality"]) - 0.42),
            mission[4] + max(0.0, float(rover_state["illumination_requirement"]) - 0.42),
        ],
        dtype=np.float32,
    )
    candidates["rover_guard"] = normalize_weights(guard)
    candidates["mission_safe_blend"] = normalize_weights(0.60 * mission + 0.40 * candidates["safe_rover"])
    candidates["mission_power_comms_blend"] = normalize_weights(0.60 * mission + 0.40 * candidates["power_comms"])
    for index, name in enumerate(("energy", "hazard", "communication", "illumination"), start=1):
        one_hot = candidates[f"{name}_only"]
        candidates[f"mission_{name}_blend"] = normalize_weights(0.62 * mission + 0.38 * one_hot)
    return candidates


def heuristic_uncertainty_lambda(
    episode: PlanningEpisode,
    max_uncertainty_lambda: float = DEFAULT_MAX_UNCERTAINTY_LAMBDA,
) -> float:
    """Simple hand-designed uncertainty sensitivity for rover baselines."""

    costmap = episode.costmap
    free_mask = ~costmap.obstacle_mask
    uncertainty_layers = costmap.uncertainty_layers
    uncertainty_mean = float(
        np.mean([uncertainty_layers[name][free_mask].mean() for name in OBJECTIVE_NAMES])
    )
    hazard_pressure = float(episode.mission_priority[2]) + max(
        0.0,
        0.55 - float(episode.rover_state["hazard_tolerance"]),
    )
    power_pressure = 0.5 * float(episode.mission_priority[1] + episode.mission_priority[4])
    lambda_fraction = np.clip(0.15 + 0.55 * uncertainty_mean + 0.20 * hazard_pressure + 0.10 * power_pressure, 0.0, 1.0)
    return float(lambda_fraction * max(float(max_uncertainty_lambda), 0.0))


def candidate_planner_configs(
    episode: PlanningEpisode,
    max_uncertainty_lambda: float = DEFAULT_MAX_UNCERTAINTY_LAMBDA,
) -> dict[str, dict[str, Any]]:
    """Baseline planner settings including risk-neutral and uncertainty-aware variants."""

    weight_sets = candidate_weight_sets(episode)
    configs: dict[str, dict[str, Any]] = {
        name: {"weights": weights, "lambda_uncertainty": 0.0}
        for name, weights in weight_sets.items()
    }

    low_lambda = 0.35 * max_uncertainty_lambda
    high_lambda = 0.85 * max_uncertainty_lambda
    for name, weights in weight_sets.items():
        configs[f"{name}_uncertainty_low"] = {
            "weights": weights,
            "lambda_uncertainty": low_lambda,
        }
        configs[f"{name}_uncertainty_high"] = {
            "weights": weights,
            "lambda_uncertainty": high_lambda,
        }
    configs["fixed_uncertainty_low"] = {
        "weights": weight_sets["fixed"],
        "lambda_uncertainty": low_lambda,
    }
    configs["fixed_uncertainty_high"] = {
        "weights": weight_sets["fixed"],
        "lambda_uncertainty": high_lambda,
    }
    configs["heuristic_uncertainty_low"] = {
        "weights": weight_sets["heuristic"],
        "lambda_uncertainty": low_lambda,
    }
    configs["heuristic_uncertainty_high"] = {
        "weights": weight_sets["heuristic"],
        "lambda_uncertainty": high_lambda,
    }
    configs["emergency_uncertainty_rule"] = {
        "weights": weight_sets["safe_rover"],
        "lambda_uncertainty": heuristic_uncertainty_lambda(
            episode,
            max_uncertainty_lambda=max_uncertainty_lambda,
        ),
    }
    return configs


def _line_cells(
    start: tuple[int, int],
    goal: tuple[int, int],
    map_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    start_row, start_col = start
    goal_row, goal_col = goal
    num_samples = int(max(abs(goal_row - start_row), abs(goal_col - start_col), 1)) + 1
    rows = np.rint(np.linspace(start_row, goal_row, num_samples)).astype(np.int32)
    cols = np.rint(np.linspace(start_col, goal_col, num_samples)).astype(np.int32)
    rows = np.clip(rows, 0, map_size - 1)
    cols = np.clip(cols, 0, map_size - 1)
    cells = np.unique(np.column_stack([rows, cols]), axis=0)
    return cells[:, 0], cells[:, 1]


def path_grid_distance(path: list[tuple[int, int]]) -> float:
    """Return geometric length of a path in grid units."""

    if len(path) < 2:
        return 0.0
    distance = 0.0
    for (row_a, col_a), (row_b, col_b) in zip(path[:-1], path[1:]):
        distance += math.hypot(row_b - row_a, col_b - col_a)
    return float(distance)


def path_layer_integral(path: list[tuple[int, int]], layer: np.ndarray) -> float:
    """Return movement-distance-weighted layer integral along a path."""

    if len(path) < 2:
        return 0.0
    total = 0.0
    for (row_a, col_a), (row_b, col_b) in zip(path[:-1], path[1:]):
        move_distance = math.hypot(row_b - row_a, col_b - col_a)
        total += move_distance * float(layer[row_b, col_b])
    return float(total)


def build_weighted_cost_map(
    layers: dict[str, np.ndarray],
    weights: np.ndarray,
    uncertainty_layers: dict[str, np.ndarray] | None = None,
    lambda_uncertainty: float = 0.0,
    min_cost: float = 1e-6,
) -> np.ndarray:
    """Combine nominal rover costs and optional uncertainty penalties."""

    weights = normalize_weights(weights)
    weighted = np.zeros_like(layers["distance"], dtype=np.float32)
    for index, name in enumerate(OBJECTIVE_NAMES):
        weighted += weights[index] * np.asarray(layers[name], dtype=np.float32)

    if uncertainty_layers is not None and lambda_uncertainty > 0.0:
        uncertainty = build_weighted_uncertainty_map(uncertainty_layers, weights)
        weighted += float(lambda_uncertainty) * uncertainty
    return np.maximum(weighted, float(min_cost)).astype(np.float32)


def build_weighted_uncertainty_map(
    uncertainty_layers: dict[str, np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Combine objective uncertainty layers using the same preference weights."""

    weights = normalize_weights(weights)
    weighted = np.zeros_like(uncertainty_layers["distance"], dtype=np.float32)
    for index, name in enumerate(OBJECTIVE_NAMES):
        weighted += weights[index] * np.asarray(uncertainty_layers[name], dtype=np.float32)
    return np.clip(weighted, 0.0, 1.0).astype(np.float32)


def evaluate_path_objectives(
    path: list[tuple[int, int]],
    layers: dict[str, np.ndarray],
    map_size: int,
) -> dict[str, float]:
    """Compute normalized rover objective costs along a successful path."""

    rows = np.array([cell[0] for cell in path], dtype=np.int32)
    cols = np.array([cell[1] for cell in path], dtype=np.int32)
    max_grid_distance = math.sqrt(2.0) * max(map_size - 1, 1)

    distance_cost = path_grid_distance(path) / max_grid_distance
    energy_cost = path_layer_integral(path, layers["energy"]) / max_grid_distance

    return {
        "distance": float(np.clip(distance_cost, 0.0, 1.0)),
        "energy": float(np.clip(energy_cost, 0.0, 1.0)),
        "hazard": float(layers["hazard"][rows, cols].mean()),
        "communication": float(layers["communication"][rows, cols].mean()),
        "illumination": float(layers["illumination"][rows, cols].mean()),
    }


def path_constraint_metrics(
    path: list[tuple[int, int]],
    layers: dict[str, np.ndarray],
    costmap: GeneratedCostMap | None = None,
) -> dict[str, float]:
    rows = np.array([cell[0] for cell in path], dtype=np.int32)
    cols = np.array([cell[1] for cell in path], dtype=np.int32)
    communication_quality = 1.0 - layers["communication"][rows, cols]
    illumination_quality = 1.0 - layers["illumination"][rows, cols]
    metrics = {
        "max_hazard": float(layers["hazard"][rows, cols].max()),
        "min_communication_quality": float(communication_quality.min()),
        "mean_illumination_quality": float(illumination_quality.mean()),
    }
    if costmap is None:
        return metrics

    if getattr(costmap, "scenario", "") != "real_mars_dtm":
        return metrics

    obstacle_mask = getattr(costmap, "obstacle_mask", None)
    if obstacle_mask is not None:
        metrics["true_obstacle_fraction"] = float(np.asarray(obstacle_mask, dtype=bool)[rows, cols].mean())

    slope_degrees = getattr(costmap, "slope_degrees", None)
    rover_max_slope = getattr(costmap, "rover_max_slope_degrees", None)
    if slope_degrees is not None and rover_max_slope is not None:
        slope_values = np.asarray(slope_degrees, dtype=np.float32)[rows, cols]
        max_slope_values = np.maximum(np.asarray(rover_max_slope, dtype=np.float32)[rows, cols], 1e-6)
        slope_ratio = slope_values / max_slope_values
        metrics.update(
            {
                "max_slope_deg": float(slope_values.max()),
                "mean_slope_deg": float(slope_values.mean()),
                "max_slope_ratio": float(slope_ratio.max()),
                "mean_slope_ratio": float(slope_ratio.mean()),
                "near_slope_limit_fraction": float(np.mean(slope_ratio >= 0.55)),
                "slope_limit_violation_fraction": float(np.mean(slope_ratio > 1.0)),
            }
        )
    return metrics


def objective_vector(objectives: dict[str, float]) -> np.ndarray:
    return np.array([objectives[name] for name in OBJECTIVE_NAMES], dtype=np.float32)


def constraint_penalty(
    objectives: dict[str, float],
    constraint_metrics: dict[str, float],
    rover_state: dict[str, float],
    mission_priority: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    """Compute mission-conditioned rover feasibility penalties.

    These are soft penalties, not hard planner constraints. They make the
    scalar evaluation reflect how a real rover team treats resource limits:
    violating a communication floor during a relay-critical traverse, or a
    hazard tolerance during a crater-rim traverse, is more serious than the
    same raw deviation in a nominal survey task.
    """

    energy_violation = max(0.0, objectives["energy"] - rover_state["battery_budget"])
    hazard_violation = max(0.0, constraint_metrics["max_hazard"] - rover_state["hazard_tolerance"])
    comm_violation = max(
        0.0,
        rover_state["min_communication_quality"] - constraint_metrics["min_communication_quality"],
    )
    illumination_violation = max(
        0.0,
        rover_state["illumination_requirement"] - constraint_metrics["mean_illumination_quality"],
    )
    max_slope_ratio = float(constraint_metrics.get("max_slope_ratio", 0.0))
    near_slope_limit_fraction = float(constraint_metrics.get("near_slope_limit_fraction", 0.0))
    slope_limit_violation_fraction = float(constraint_metrics.get("slope_limit_violation_fraction", 0.0))
    true_obstacle_fraction = float(constraint_metrics.get("true_obstacle_fraction", 0.0))
    traversability_violation = (
        0.40 * max(0.0, max_slope_ratio - 0.58)
        + 0.75 * max(0.0, near_slope_limit_fraction - 0.04)
        + 2.50 * slope_limit_violation_fraction
        + 3.50 * true_obstacle_fraction
    )

    priority = np.asarray(
        mission_priority if mission_priority is not None else np.ones(len(OBJECTIVE_NAMES)) / len(OBJECTIVE_NAMES),
        dtype=np.float32,
    ).reshape(-1)
    if priority.size < len(OBJECTIVE_NAMES):
        priority = np.ones(len(OBJECTIVE_NAMES), dtype=np.float32) / len(OBJECTIVE_NAMES)
    priority = np.clip(priority[: len(OBJECTIVE_NAMES)], 0.0, None)
    priority = priority / max(float(priority.sum()), 1e-6)

    energy_tightness = max(0.0, 0.50 - float(rover_state["battery_budget"])) / 0.50
    hazard_tightness = max(0.0, 0.55 - float(rover_state["hazard_tolerance"])) / 0.55
    comm_tightness = max(0.0, float(rover_state["min_communication_quality"]) - 0.42) / 0.58
    illumination_tightness = max(0.0, float(rover_state["illumination_requirement"]) - 0.42) / 0.58
    weights = {
        "energy": 1.10 + 1.70 * float(priority[1]) + 1.20 * energy_tightness,
        "hazard": 1.35 + 1.90 * float(priority[2]) + 1.45 * hazard_tightness,
        "communication": 1.15 + 1.85 * float(priority[3]) + 1.30 * comm_tightness,
        "illumination": 1.05 + 1.85 * float(priority[4]) + 1.25 * illumination_tightness,
        "traversability": (
            1.20
            + 1.55 * float(priority[2])
            + 0.85 * hazard_tightness
            + 0.45 * float(priority[1])
        ),
    }
    violations_array = {
        "energy": energy_violation,
        "hazard": hazard_violation,
        "communication": comm_violation,
        "illumination": illumination_violation,
        "traversability": traversability_violation,
    }
    penalty = 0.0
    for name, violation in violations_array.items():
        weight = weights[name]
        penalty += weight * violation + 1.25 * weight * violation * violation

    violations = {
        "energy_violation": float(energy_violation),
        "hazard_violation": float(hazard_violation),
        "communication_violation": float(comm_violation),
        "illumination_violation": float(illumination_violation),
        "traversability_violation": float(traversability_violation),
        "energy_constraint_weight": float(weights["energy"]),
        "hazard_constraint_weight": float(weights["hazard"]),
        "communication_constraint_weight": float(weights["communication"]),
        "illumination_constraint_weight": float(weights["illumination"]),
        "traversability_constraint_weight": float(weights["traversability"]),
    }
    return float(penalty), violations


def scalar_path_cost(
    episode: PlanningEpisode,
    objectives: dict[str, float],
    constraint_metrics: dict[str, float],
) -> tuple[float, float, dict[str, float]]:
    base_cost = float(np.dot(episode.mission_priority, objective_vector(objectives)))
    penalty, violations = constraint_penalty(
        objectives,
        constraint_metrics,
        episode.rover_state,
        mission_priority=episode.mission_priority,
    )
    return base_cost + penalty, penalty, violations


def budgeted_path_uncertainty(
    path: list[tuple[int, int]],
    uncertainty_layers: dict[str, np.ndarray],
    budget_fraction: float = DEFAULT_ATTACK_BUDGET_FRACTION,
) -> dict[str, float]:
    """Measure top-budget uncertainty along a path.

    This is the attacker model: an adversary cannot corrupt the whole map. It
    only spends budget on the most uncertain path cells for each objective.
    """

    if not path:
        return {name: 1.0 for name in OBJECTIVE_NAMES}

    rows = np.array([cell[0] for cell in path], dtype=np.int32)
    cols = np.array([cell[1] for cell in path], dtype=np.int32)
    budget_count = max(1, int(math.ceil(len(path) * float(np.clip(budget_fraction, 0.0, 1.0)))))

    uncertainty: dict[str, float] = {}
    for name in OBJECTIVE_NAMES:
        values = np.asarray(uncertainty_layers[name][rows, cols], dtype=np.float32).reshape(-1)
        if values.size == 0:
            uncertainty[name] = 1.0
            continue
        if values.size > budget_count:
            selected = np.partition(values, -budget_count)[-budget_count:]
        else:
            selected = values
        uncertainty[name] = float(np.clip(selected.mean(), 0.0, 1.0))
    return uncertainty


def attacked_path_objectives(
    objectives: dict[str, float],
    path_uncertainty: dict[str, float],
    attack_strength: float = DEFAULT_ATTACK_STRENGTH,
) -> dict[str, float]:
    """Apply a bounded attacker perturbation to nominal path objectives."""

    attacked: dict[str, float] = {}
    for name in OBJECTIVE_NAMES:
        scale = ATTACK_OBJECTIVE_SCALE[name]
        increase = float(attack_strength) * scale * float(path_uncertainty[name])
        attacked[name] = float(np.clip(objectives[name] + increase, 0.0, 1.0))
    return attacked


def _attacker_probabilities(
    impact: np.ndarray,
    response_mode: str = DEFAULT_ATTACKER_RESPONSE,
    temperature: float = DEFAULT_ATTACKER_TEMPERATURE,
    top_fraction: float = DEFAULT_ATTACKER_TOP_FRACTION,
    sharpness: float = DEFAULT_ATTACKER_SHARPNESS,
) -> np.ndarray:
    """Convert per-cell impact into a bounded but sharp attacker response.

    ``softmax`` is the old raw-impact response. The new default, ``zscore_topk``,
    first standardizes impact values on the current path, keeps only the most
    attractive cells, and then applies a softmax. This models a rover-relevant
    adversary that concentrates limited disruption effort on a few plausible
    failure points instead of spreading probability nearly uniformly.
    """

    values = np.asarray(impact, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros(0, dtype=np.float32)
    if not np.isfinite(values).all():
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if response_mode not in ATTACKER_RESPONSE_MODES:
        raise ValueError(f"attacker response must be one of {ATTACKER_RESPONSE_MODES}")

    temp = max(float(temperature), 1e-4)
    gain = max(float(sharpness), 1e-6)
    logits = values.copy()

    if response_mode in {"zscore_softmax", "zscore_topk"}:
        std = float(values.std())
        if std > 1e-8:
            logits = (values - float(values.mean())) / std
        else:
            logits = np.zeros_like(values)
        logits = gain * logits
    elif response_mode == "rank_topk":
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty_like(values, dtype=np.float64)
        ranks[order] = np.linspace(0.0, 1.0, values.size)
        logits = gain * ranks

    if response_mode in {"zscore_topk", "rank_topk"} and values.size > 1:
        fraction = float(np.clip(top_fraction, 1.0 / float(values.size), 1.0))
        keep_count = max(1, int(math.ceil(values.size * fraction)))
        keep_indices = np.argpartition(values, -keep_count)[-keep_count:]
        mask = np.full(values.shape, False, dtype=bool)
        mask[keep_indices] = True
        logits = np.where(mask, logits, -np.inf)

    finite_logits = logits[np.isfinite(logits)]
    if finite_logits.size == 0:
        probabilities = np.full(values.size, 1.0 / float(values.size), dtype=np.float64)
    else:
        scaled_logits = logits / temp
        scaled_logits = scaled_logits - float(np.nanmax(scaled_logits))
        exp_logits = np.where(np.isfinite(scaled_logits), np.exp(scaled_logits), 0.0)
        total = float(exp_logits.sum())
        if total <= 1e-12:
            probabilities = np.full(values.size, 1.0 / float(values.size), dtype=np.float64)
        else:
            probabilities = exp_logits / total

    return probabilities.astype(np.float32)


def bounded_rational_attacker_response(
    path: list[tuple[int, int]],
    objectives: dict[str, float],
    uncertainty_layers: dict[str, np.ndarray],
    mission_priority: np.ndarray,
    constraint_penalty_value: float,
    temperature: float = DEFAULT_ATTACKER_TEMPERATURE,
    attack_strength: float = DEFAULT_ATTACK_STRENGTH,
    response_mode: str = DEFAULT_ATTACKER_RESPONSE,
    top_fraction: float = DEFAULT_ATTACKER_TOP_FRACTION,
    sharpness: float = DEFAULT_ATTACKER_SHARPNESS,
) -> dict[str, Any]:
    """Evaluate a soft Stackelberg follower on the selected path only.

    The attacker is bounded-rational: it prefers path cells with high
    mission-weighted uncertainty impact, but temperature/top-k controls prevent
    it from becoming either fully uniform or a brittle pure worst-case point
    mass. This keeps training at one A* call per step.
    """

    if not path:
        return {
            "soft_attacked_objectives": {},
            "soft_attacked_scalar_cost": 10.0,
            "soft_attack_penalty": 0.0,
            "soft_path_uncertainty": {},
            "soft_attacker_entropy": 0.0,
            "soft_attacker_peak_probability": 1.0,
            "soft_attacker_expected_impact": 1.0,
        }

    rows = np.array([cell[0] for cell in path], dtype=np.int32)
    cols = np.array([cell[1] for cell in path], dtype=np.int32)
    priority = np.asarray(mission_priority, dtype=np.float32)

    cell_uncertainties = np.stack(
        [np.asarray(uncertainty_layers[name][rows, cols], dtype=np.float32) for name in OBJECTIVE_NAMES],
        axis=1,
    )
    scales = np.array([ATTACK_OBJECTIVE_SCALE[name] for name in OBJECTIVE_NAMES], dtype=np.float32)
    impact = (cell_uncertainties * scales * priority.reshape(1, -1)).sum(axis=1)
    impact = np.asarray(impact, dtype=np.float32)

    probabilities = _attacker_probabilities(
        impact,
        response_mode=response_mode,
        temperature=temperature,
        top_fraction=top_fraction,
        sharpness=sharpness,
    )

    expected_uncertainty_values = probabilities.reshape(-1, 1) * cell_uncertainties
    expected_uncertainty = expected_uncertainty_values.sum(axis=0)

    soft_path_uncertainty = {
        name: float(np.clip(expected_uncertainty[index], 0.0, 1.0))
        for index, name in enumerate(OBJECTIVE_NAMES)
    }
    soft_attacked_objectives = attacked_path_objectives(
        objectives,
        soft_path_uncertainty,
        attack_strength=attack_strength,
    )
    soft_base_cost = float(np.dot(priority, objective_vector(soft_attacked_objectives)))
    soft_attacked_scalar_cost = soft_base_cost + float(constraint_penalty_value)

    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-8)))
    normalized_entropy = entropy / math.log(max(len(probabilities), 2))

    nominal_base_cost = float(np.dot(priority, objective_vector(objectives)))
    nominal_scalar_cost = nominal_base_cost + float(constraint_penalty_value)
    return {
        "soft_attacked_objectives": soft_attacked_objectives,
        "soft_attacked_scalar_cost": float(soft_attacked_scalar_cost),
        "soft_attack_penalty": max(0.0, float(soft_attacked_scalar_cost - nominal_scalar_cost)),
        "soft_path_uncertainty": soft_path_uncertainty,
        "soft_attacker_entropy": float(np.clip(normalized_entropy, 0.0, 1.0)),
        "soft_attacker_peak_probability": float(np.clip(probabilities.max(), 0.0, 1.0)),
        "soft_attacker_expected_impact": float(np.clip((probabilities * impact).sum(), 0.0, 1.0)),
        "soft_attacker_response_mode": response_mode,
        "soft_attacker_top_fraction": float(np.clip(top_fraction, 0.0, 1.0)),
        "soft_attacker_sharpness": float(sharpness),
    }


def counterfactual_lambda_target(
    episode: PlanningEpisode,
    weights: np.ndarray,
    lambda_fractions: np.ndarray,
    max_uncertainty_lambda: float = DEFAULT_MAX_UNCERTAINTY_LAMBDA,
    allow_diagonal: bool = True,
    overreaction_coef: float = 1.0,
    lambda_sparsity_coef: float = 0.03,
) -> dict[str, Any]:
    """Choose a planner-informed lambda target from local counterfactuals.

    For fixed objective weights, this function replans with several uncertainty
    sensitivities. The selected target minimizes robust attacked cost plus a
    penalty for nominal-cost degradation relative to risk-neutral planning. The
    label is used as an auxiliary signal by CRC-PPO.
    """

    fractions = np.asarray(lambda_fractions, dtype=np.float32).reshape(-1)
    if fractions.size == 0:
        fractions = np.array([0.0], dtype=np.float32)
    fractions = np.clip(fractions, 0.0, 1.0)

    neutral_result = plan_with_weights(
        episode,
        weights,
        lambda_uncertainty=0.0,
        allow_diagonal=allow_diagonal,
    )
    neutral_scalar_cost = float(neutral_result["scalar_cost"])

    best_score = float("inf")
    best_fraction = 0.0
    best_result = neutral_result
    records = []

    for fraction in fractions:
        lambda_uncertainty = float(fraction * max(float(max_uncertainty_lambda), 0.0))
        result = plan_with_weights(
            episode,
            weights,
            lambda_uncertainty=lambda_uncertainty,
            allow_diagonal=allow_diagonal,
        )
        nominal_overreaction = max(0.0, float(result["scalar_cost"]) - neutral_scalar_cost)
        score = (
            float(result["attacked_scalar_cost"])
            + float(overreaction_coef) * nominal_overreaction
            + float(lambda_sparsity_coef) * float(fraction)
        )
        records.append(
            {
                "lambda_fraction": float(fraction),
                "lambda_uncertainty": lambda_uncertainty,
                "score": float(score),
                "scalar_cost": float(result["scalar_cost"]),
                "attacked_scalar_cost": float(result["attacked_scalar_cost"]),
                "nominal_overreaction": float(nominal_overreaction),
                "success": bool(result["success"]),
            }
        )
        if score < best_score:
            best_score = float(score)
            best_fraction = float(fraction)
            best_result = result

    return {
        "target_lambda_fraction": float(best_fraction),
        "target_lambda_uncertainty": float(best_fraction * max(float(max_uncertainty_lambda), 0.0)),
        "counterfactual_score": float(best_score),
        "neutral_scalar_cost": neutral_scalar_cost,
        "best_result": best_result,
        "records": records,
    }


def path_overlap_ratio(
    path_a: list[tuple[int, int]] | None,
    path_b: list[tuple[int, int]] | None,
) -> float:
    """Jaccard overlap between two grid paths."""

    if not path_a or not path_b:
        return float("nan")
    cells_a = {tuple(map(int, cell)) for cell in path_a}
    cells_b = {tuple(map(int, cell)) for cell in path_b}
    union_size = len(cells_a | cells_b)
    if union_size <= 0:
        return float("nan")
    return float(len(cells_a & cells_b) / union_size)


def path_mask_exposure(
    path: list[tuple[int, int]] | None,
    mask: np.ndarray | None,
) -> tuple[int, float]:
    """Count and ratio of path cells inside a boolean mask."""

    if path is None or len(path) == 0 or mask is None:
        return 0, 0.0
    mask_array = np.asarray(mask, dtype=bool)
    count = 0
    for row, col in path:
        row_i, col_i = int(row), int(col)
        if 0 <= row_i < mask_array.shape[0] and 0 <= col_i < mask_array.shape[1]:
            count += int(mask_array[row_i, col_i])
    return int(count), float(count / max(len(path), 1))


def path_layer_mean(
    path: list[tuple[int, int]] | None,
    layer: np.ndarray,
) -> float:
    if path is None or len(path) == 0:
        return float("nan")
    values = []
    layer_array = np.asarray(layer, dtype=np.float32)
    for row, col in path:
        row_i, col_i = int(row), int(col)
        if 0 <= row_i < layer_array.shape[0] and 0 <= col_i < layer_array.shape[1]:
            values.append(float(layer_array[row_i, col_i]))
    return float(np.mean(values)) if values else float("nan")


def plan_with_weights(
    episode: PlanningEpisode,
    weights: np.ndarray,
    lambda_uncertainty: float = 0.0,
    allow_diagonal: bool = True,
    attack_budget_fraction: float = DEFAULT_ATTACK_BUDGET_FRACTION,
    attack_strength: float = DEFAULT_ATTACK_STRENGTH,
    attacker_temperature: float = DEFAULT_ATTACKER_TEMPERATURE,
    attacker_response: str = DEFAULT_ATTACKER_RESPONSE,
    attacker_top_fraction: float = DEFAULT_ATTACKER_TOP_FRACTION,
    attacker_sharpness: float = DEFAULT_ATTACKER_SHARPNESS,
) -> dict[str, Any]:
    """Plan and evaluate one method on one episode."""

    normalized_weights = normalize_weights(weights)
    lambda_uncertainty = float(max(lambda_uncertainty, 0.0))
    belief_costmap = episode.costmap
    true_costmap = episode.true_costmap or belief_costmap
    has_belief_mismatch = episode.true_costmap is not None
    weighted_uncertainty_map = build_weighted_uncertainty_map(
        belief_costmap.uncertainty_layers,
        normalized_weights,
    )
    weighted_cost_map = build_weighted_cost_map(
        belief_costmap.layers,
        normalized_weights,
        uncertainty_layers=belief_costmap.uncertainty_layers,
        lambda_uncertainty=lambda_uncertainty,
    )
    path = weighted_astar(
        weighted_cost_map,
        belief_costmap.obstacle_mask,
        belief_costmap.start,
        belief_costmap.goal,
        allow_diagonal=allow_diagonal,
    )

    if path is None:
        return {
            "success": False,
            "reward": -10.0,
            "scalar_cost": 10.0,
            "belief_scalar_cost": 10.0,
            "map_mismatch_penalty": 0.0,
            "map_mismatch_abs_error": 0.0,
            "attacked_scalar_cost": 10.0,
            "attack_penalty": 0.0,
            "path_uncertainty": {},
            "attacked_objectives": {},
            "soft_attacked_objectives": {},
            "soft_attacked_scalar_cost": 10.0,
            "soft_attack_penalty": 0.0,
            "soft_path_uncertainty": {},
            "soft_attacker_entropy": 0.0,
            "soft_attacker_peak_probability": 1.0,
            "soft_attacker_expected_impact": 1.0,
            "soft_attacker_response_mode": attacker_response,
            "soft_attacker_top_fraction": float(np.clip(attacker_top_fraction, 0.0, 1.0)),
            "soft_attacker_sharpness": float(attacker_sharpness),
            "constraint_penalty": 10.0,
            "constraint_metrics": {},
            "constraint_violations": {},
            "objectives": {},
            "belief_objectives": {},
            "belief_constraint_metrics": {},
            "belief_constraint_violations": {},
            "belief_constraint_penalty": 10.0,
            "weights": normalized_weights,
            "lambda_uncertainty": lambda_uncertainty,
            "path_length": 0,
            "path": None,
            "true_belief_mismatch": has_belief_mismatch,
            "mean_path_confidence": 0.0,
            "weighted_cost_map": weighted_cost_map,
            "weighted_uncertainty_map": weighted_uncertainty_map,
        }

    objectives = evaluate_path_objectives(path, true_costmap.layers, true_costmap.layers["distance"].shape[0])
    belief_objectives = evaluate_path_objectives(
        path,
        belief_costmap.layers,
        belief_costmap.layers["distance"].shape[0],
    )
    attack_mask = getattr(belief_costmap, "attack_mask", None)
    attacked_cell_exposure, attacked_cell_exposure_ratio = path_mask_exposure(path, attack_mask)
    constraint_metrics = path_constraint_metrics(path, true_costmap.layers, true_costmap)
    belief_constraint_metrics = path_constraint_metrics(path, belief_costmap.layers, belief_costmap)
    scalar_cost, penalty, violations = scalar_path_cost(episode, objectives, constraint_metrics)
    belief_scalar_cost, belief_penalty, belief_violations = scalar_path_cost(
        episode,
        belief_objectives,
        belief_constraint_metrics,
    )
    map_mismatch_penalty = float(scalar_cost - belief_scalar_cost)
    path_uncertainty = budgeted_path_uncertainty(
        path,
        belief_costmap.uncertainty_layers,
        budget_fraction=attack_budget_fraction,
    )
    attacked_objectives = attacked_path_objectives(
        objectives,
        path_uncertainty,
        attack_strength=attack_strength,
    )
    attacked_base_cost = float(
        np.dot(episode.mission_priority, objective_vector(attacked_objectives))
    )
    attacked_scalar_cost = attacked_base_cost + penalty
    soft_attack = bounded_rational_attacker_response(
        path,
        objectives,
        belief_costmap.uncertainty_layers,
        episode.mission_priority,
        constraint_penalty_value=penalty,
        temperature=attacker_temperature,
        attack_strength=attack_strength,
        response_mode=attacker_response,
        top_fraction=attacker_top_fraction,
        sharpness=attacker_sharpness,
    )

    confidence_layers = episode.confidence_layers or {}
    weighted_confidence_map = np.zeros_like(weighted_uncertainty_map, dtype=np.float32)
    for index, name in enumerate(OBJECTIVE_NAMES):
        confidence_layer = confidence_layers.get(name)
        if confidence_layer is None:
            confidence_layer = 1.0 - np.asarray(belief_costmap.uncertainty_layers[name], dtype=np.float32)
        weighted_confidence_map += normalized_weights[index] * np.asarray(confidence_layer, dtype=np.float32)
    weighted_confidence_map = np.clip(weighted_confidence_map, 0.0, 1.0).astype(np.float32)

    return {
        "success": True,
        "reward": -attacked_scalar_cost,
        "scalar_cost": scalar_cost,
        "belief_scalar_cost": belief_scalar_cost,
        "map_mismatch_penalty": map_mismatch_penalty,
        "map_mismatch_abs_error": abs(map_mismatch_penalty),
        "attacked_scalar_cost": attacked_scalar_cost,
        "attack_penalty": max(0.0, attacked_scalar_cost - scalar_cost),
        "path_uncertainty": path_uncertainty,
        "attacked_objectives": attacked_objectives,
        **soft_attack,
        "constraint_penalty": penalty,
        "constraint_metrics": constraint_metrics,
        "constraint_violations": violations,
        "objectives": objectives,
        "belief_objectives": belief_objectives,
        "belief_constraint_metrics": belief_constraint_metrics,
        "belief_constraint_violations": belief_violations,
        "belief_constraint_penalty": belief_penalty,
        "weights": normalized_weights,
        "lambda_uncertainty": lambda_uncertainty,
        "path_length": len(path),
        "path": path,
        "true_belief_mismatch": has_belief_mismatch,
        "attacked_cell_exposure": attacked_cell_exposure,
        "attacked_cell_exposure_ratio": attacked_cell_exposure_ratio,
        "attacked_corridor_cells": int(np.asarray(attack_mask, dtype=bool).sum()) if attack_mask is not None else 0,
        "hazard_exposure": path_layer_mean(path, true_costmap.layers["hazard"]),
        "belief_hazard_exposure": path_layer_mean(path, belief_costmap.layers["hazard"]),
        "uncertainty_exposure": path_layer_mean(path, weighted_uncertainty_map),
        "belief_uncertainty_exposure": path_layer_mean(path, weighted_uncertainty_map),
        "mean_path_confidence": path_layer_mean(path, weighted_confidence_map),
        "illumination_exposure": path_layer_mean(path, true_costmap.layers["illumination"]),
        "belief_illumination_exposure": path_layer_mean(path, belief_costmap.layers["illumination"]),
        "communication_exposure": path_layer_mean(path, true_costmap.layers["communication"]),
        "belief_communication_exposure": path_layer_mean(path, belief_costmap.layers["communication"]),
        "attack_metadata": getattr(belief_costmap, "attack_metadata", None) or {},
        "weighted_cost_map": weighted_cost_map,
        "weighted_uncertainty_map": weighted_uncertainty_map,
    }


def evaluate_candidate_results(
    episode: PlanningEpisode,
    allow_diagonal: bool = True,
    max_uncertainty_lambda: float = DEFAULT_MAX_UNCERTAINTY_LAMBDA,
    attacker_temperature: float = DEFAULT_ATTACKER_TEMPERATURE,
    attacker_response: str = DEFAULT_ATTACKER_RESPONSE,
    attacker_top_fraction: float = DEFAULT_ATTACKER_TOP_FRACTION,
    attacker_sharpness: float = DEFAULT_ATTACKER_SHARPNESS,
) -> dict[str, dict[str, Any]]:
    return {
        name: plan_with_weights(
            episode,
            config["weights"],
            lambda_uncertainty=float(config["lambda_uncertainty"]),
            allow_diagonal=allow_diagonal,
            attacker_temperature=attacker_temperature,
            attacker_response=attacker_response,
            attacker_top_fraction=attacker_top_fraction,
            attacker_sharpness=attacker_sharpness,
        )
        for name, config in candidate_planner_configs(
            episode,
            max_uncertainty_lambda=max_uncertainty_lambda,
        ).items()
    }


def best_candidate_result(
    episode: PlanningEpisode,
    allow_diagonal: bool = True,
    max_uncertainty_lambda: float = DEFAULT_MAX_UNCERTAINTY_LAMBDA,
    attacker_temperature: float = DEFAULT_ATTACKER_TEMPERATURE,
    attacker_response: str = DEFAULT_ATTACKER_RESPONSE,
    attacker_top_fraction: float = DEFAULT_ATTACKER_TOP_FRACTION,
    attacker_sharpness: float = DEFAULT_ATTACKER_SHARPNESS,
) -> tuple[str, dict[str, Any]]:
    candidates = evaluate_candidate_results(
        episode,
        allow_diagonal=allow_diagonal,
        max_uncertainty_lambda=max_uncertainty_lambda,
        attacker_temperature=attacker_temperature,
        attacker_response=attacker_response,
        attacker_top_fraction=attacker_top_fraction,
        attacker_sharpness=attacker_sharpness,
    )
    return min(candidates.items(), key=lambda item: float(item[1]["attacked_scalar_cost"]))


def candidate_feature_vector(
    episode: PlanningEpisode,
    allow_diagonal: bool = True,
    max_uncertainty_lambda: float = DEFAULT_MAX_UNCERTAINTY_LAMBDA,
    candidate_results: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Flatten candidate path trade-offs for the policy observation."""

    features: list[float] = []
    results = candidate_results or evaluate_candidate_results(
        episode,
        allow_diagonal=allow_diagonal,
        max_uncertainty_lambda=max_uncertainty_lambda,
    )
    for _, result in results.items():
        success = 1.0 if result["success"] else 0.0
        lambda_fraction = float(
            np.clip(
                result.get("lambda_uncertainty", 0.0) / max(float(max_uncertainty_lambda), 1e-8),
                0.0,
                1.0,
            )
        )
        if result["success"]:
            objectives = objective_vector(result["objectives"]).tolist()
            scalar_cost = float(np.clip(result["scalar_cost"], 0.0, 2.0) / 2.0)
            attacked_cost = float(np.clip(result["attacked_scalar_cost"], 0.0, 2.0) / 2.0)
            attack_penalty = float(np.clip(result["attack_penalty"], 0.0, 1.0))
            penalty = float(np.clip(result["constraint_penalty"], 0.0, 2.0) / 2.0)
        else:
            objectives = [1.0] * len(OBJECTIVE_NAMES)
            scalar_cost = 1.0
            attacked_cost = 1.0
            attack_penalty = 1.0
            penalty = 1.0
        features.extend([success, scalar_cost, attacked_cost, attack_penalty, penalty, lambda_fraction, *objectives])
    return np.asarray(features, dtype=np.float32)


def compute_observation(
    episode: PlanningEpisode,
    map_size: int,
    observation_mode: str = "basic",
    max_uncertainty_lambda: float = DEFAULT_MAX_UNCERTAINTY_LAMBDA,
    candidate_results: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Build the low-dimensional Gymnasium observation."""

    if observation_mode not in OBSERVATION_MODES:
        raise ValueError(f"observation_mode must be one of {OBSERVATION_MODES}")

    costmap = episode.costmap
    start_row, start_col = costmap.start
    goal_row, goal_col = costmap.goal

    free_mask = ~costmap.obstacle_mask
    if not free_mask.any():
        free_mask = np.ones_like(costmap.obstacle_mask, dtype=bool)

    layers = costmap.layers
    uncertainty_layers = costmap.uncertainty_layers
    euclidean = math.hypot(start_row - goal_row, start_col - goal_col)
    max_grid_distance = math.sqrt(2.0) * max(map_size - 1, 1)

    base_obs = [
        start_col / float(map_size),
        start_row / float(map_size),
        goal_col / float(map_size),
        goal_row / float(map_size),
        euclidean / max_grid_distance,
    ]
    for name in OBJECTIVE_NAMES[1:]:
        base_obs.extend([float(layers[name][free_mask].mean()), float(layers[name][free_mask].std())])
    for name in OBJECTIVE_NAMES:
        base_obs.extend(
            [
                float(uncertainty_layers[name][free_mask].mean()),
                float(uncertainty_layers[name][free_mask].std()),
            ]
        )
    base_obs.extend(episode.mission_priority.tolist())
    base_obs.extend(rover_state_vector(episode.rover_state).tolist())
    regime_one_hot = [
        1.0 if episode.mission_regime == name else 0.0
        for name in MISSION_REGIME_NAMES
    ]
    base_obs.extend(regime_one_hot)
    base_obs.append(float(np.clip(episode.mission_severity, 0.0, 1.0)))

    if observation_mode == "basic":
        obs = np.array(base_obs, dtype=np.float32)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    line_rows, line_cols = _line_cells(costmap.start, costmap.goal, map_size)
    line_free = ~costmap.obstacle_mask[line_rows, line_cols]
    obstacle_density = float(costmap.obstacle_mask.mean())
    line_obstacle_fraction = 1.0 - float(line_free.mean()) if line_free.size else 1.0

    direction_row = (goal_row - start_row) / max(float(map_size - 1), 1.0)
    direction_col = (goal_col - start_col) / max(float(map_size - 1), 1.0)

    extended_obs = [
        obstacle_density,
        line_obstacle_fraction,
        (direction_col + 1.0) * 0.5,
        (direction_row + 1.0) * 0.5,
    ]

    for name in OBJECTIVE_NAMES[1:]:
        line_values = layers[name][line_rows, line_cols]
        extended_obs.extend([float(line_values.mean()), float(line_values.std())])

    for name in OBJECTIVE_NAMES:
        line_values = uncertainty_layers[name][line_rows, line_cols]
        extended_obs.extend([float(line_values.mean()), float(line_values.std())])

    for name in OBJECTIVE_NAMES[1:]:
        extended_obs.append(float(layers[name][start_row, start_col]))

    for name in OBJECTIVE_NAMES[1:]:
        extended_obs.append(float(layers[name][goal_row, goal_col]))

    for name in OBJECTIVE_NAMES:
        extended_obs.append(float(uncertainty_layers[name][start_row, start_col]))

    for name in OBJECTIVE_NAMES:
        extended_obs.append(float(uncertainty_layers[name][goal_row, goal_col]))

    if observation_mode == "terrain":
        obs = np.array([*base_obs, *extended_obs], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0).astype(np.float32)

    obs = np.array(
        [
            *base_obs,
            *extended_obs,
            *candidate_feature_vector(
                episode,
                max_uncertainty_lambda=max_uncertainty_lambda,
                candidate_results=candidate_results,
            ).tolist(),
        ],
        dtype=np.float32,
    )
    return np.clip(obs, 0.0, 1.0).astype(np.float32)


def result_to_row(
    episode_index: int,
    method: str,
    episode: PlanningEpisode,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Flatten a planning result for CSV output."""

    row: dict[str, Any] = {
        "episode": episode_index,
        "scenario": episode.scenario,
        "mission_regime": episode.mission_regime,
        "mission_severity": float(episode.mission_severity),
        "method": method,
        "success": bool(result["success"]),
        "reward": float(result["reward"]),
        "scalar_cost": float(result["scalar_cost"]),
        "attacked_scalar_cost": float(result.get("attacked_scalar_cost", result["scalar_cost"])),
        "attack_penalty": float(result.get("attack_penalty", np.nan)),
        "soft_attacked_scalar_cost": float(result.get("soft_attacked_scalar_cost", np.nan)),
        "soft_attack_penalty": float(result.get("soft_attack_penalty", np.nan)),
        "soft_attacker_entropy": float(result.get("soft_attacker_entropy", np.nan)),
        "soft_attacker_peak_probability": float(result.get("soft_attacker_peak_probability", np.nan)),
        "soft_attacker_expected_impact": float(result.get("soft_attacker_expected_impact", np.nan)),
        "soft_attacker_response_mode": result.get("soft_attacker_response_mode", ""),
        "soft_attacker_top_fraction": float(result.get("soft_attacker_top_fraction", np.nan)),
        "soft_attacker_sharpness": float(result.get("soft_attacker_sharpness", np.nan)),
        "lambda_uncertainty": float(result.get("lambda_uncertainty", 0.0)),
        "constraint_penalty": float(result.get("constraint_penalty", np.nan)),
        "path_length": int(result["path_length"]),
        "attacked_cell_exposure": int(result.get("attacked_cell_exposure", 0)),
        "attacked_cell_exposure_ratio": float(result.get("attacked_cell_exposure_ratio", 0.0)),
        "attacked_corridor_cells": int(result.get("attacked_corridor_cells", 0)),
        "hazard_exposure": float(result.get("hazard_exposure", np.nan)),
        "uncertainty_exposure": float(result.get("uncertainty_exposure", np.nan)),
        "illumination_exposure": float(result.get("illumination_exposure", np.nan)),
        "communication_exposure": float(result.get("communication_exposure", np.nan)),
        "start_row": episode.costmap.start[0],
        "start_col": episode.costmap.start[1],
        "goal_row": episode.costmap.goal[0],
        "goal_col": episode.costmap.goal[1],
    }

    objectives = result.get("objectives", {})
    attacked_objectives = result.get("attacked_objectives", {})
    path_uncertainty = result.get("path_uncertainty", {})
    soft_path_uncertainty = result.get("soft_path_uncertainty", {})
    for name in OBJECTIVE_NAMES:
        row[f"{name}_cost"] = float(objectives[name]) if name in objectives else np.nan
        row[f"attacked_{name}_cost"] = (
            float(attacked_objectives[name]) if name in attacked_objectives else np.nan
        )
        row[f"path_uncertainty_{name}"] = (
            float(path_uncertainty[name]) if name in path_uncertainty else np.nan
        )
        row[f"soft_path_uncertainty_{name}"] = (
            float(soft_path_uncertainty[name]) if name in soft_path_uncertainty else np.nan
        )

    weights = np.asarray(result["weights"], dtype=np.float32)
    for index, name in enumerate(OBJECTIVE_NAMES):
        row[f"weight_{name}"] = float(weights[index])
        row[f"mission_{name}"] = float(episode.mission_priority[index])

    for name in ROVER_STATE_NAMES:
        row[name] = float(episode.rover_state[name])

    for name, value in result.get("constraint_metrics", {}).items():
        row[name] = float(value)
    for name, value in result.get("constraint_violations", {}).items():
        row[name] = float(value)

    return row


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Create mean/std metrics by method."""

    rows = []
    for method, group in results.groupby("method", sort=False):
        summary: dict[str, Any] = {"method": method}
        summary["success_rate"] = float(group["success"].mean())
        summary["std_success_rate"] = float(group["success"].astype(float).std(ddof=0))
        summary["mean_reward"] = float(group["reward"].mean())
        summary["std_reward"] = float(group["reward"].std(ddof=0))
        summary["mean_scalar_cost"] = float(group["scalar_cost"].mean())
        summary["std_scalar_cost"] = float(group["scalar_cost"].std(ddof=0))
        if "attacked_scalar_cost" in group:
            summary["mean_attacked_scalar_cost"] = float(group["attacked_scalar_cost"].mean())
            summary["std_attacked_scalar_cost"] = float(group["attacked_scalar_cost"].std(ddof=0))
            attacked_cost = group["attacked_scalar_cost"].dropna()
            if not attacked_cost.empty:
                p90 = float(attacked_cost.quantile(0.90))
                p95 = float(attacked_cost.quantile(0.95))
                summary["p90_attacked_scalar_cost"] = p90
                summary["p95_attacked_scalar_cost"] = p95
                summary["cvar90_attacked_scalar_cost"] = float(
                    attacked_cost[attacked_cost >= p90].mean()
                )
                summary["cvar95_attacked_scalar_cost"] = float(
                    attacked_cost[attacked_cost >= p95].mean()
                )
        reward_values = group["reward"].dropna()
        if not reward_values.empty:
            p10_reward = float(reward_values.quantile(0.10))
            summary["p10_reward"] = p10_reward
            summary["worst10_mean_reward"] = float(
                reward_values[reward_values <= p10_reward].mean()
            )
        if "attack_penalty" in group:
            summary["mean_attack_penalty"] = float(group["attack_penalty"].mean())
            summary["std_attack_penalty"] = float(group["attack_penalty"].std(ddof=0))
        if "soft_attacked_scalar_cost" in group:
            summary["mean_soft_attacked_scalar_cost"] = float(group["soft_attacked_scalar_cost"].mean())
            summary["std_soft_attacked_scalar_cost"] = float(group["soft_attacked_scalar_cost"].std(ddof=0))
        if "soft_attack_penalty" in group:
            summary["mean_soft_attack_penalty"] = float(group["soft_attack_penalty"].mean())
            summary["std_soft_attack_penalty"] = float(group["soft_attack_penalty"].std(ddof=0))
        if "soft_attacker_entropy" in group:
            summary["mean_soft_attacker_entropy"] = float(group["soft_attacker_entropy"].mean())
            summary["std_soft_attacker_entropy"] = float(group["soft_attacker_entropy"].std(ddof=0))
        if "soft_attacker_peak_probability" in group:
            summary["mean_soft_attacker_peak_probability"] = float(
                group["soft_attacker_peak_probability"].mean()
            )
            summary["std_soft_attacker_peak_probability"] = float(
                group["soft_attacker_peak_probability"].std(ddof=0)
            )
        if "lambda_uncertainty" in group:
            summary["mean_lambda_uncertainty"] = float(group["lambda_uncertainty"].mean())
            summary["std_lambda_uncertainty"] = float(group["lambda_uncertainty"].std(ddof=0))
        if "attacked_cell_exposure_ratio" in group:
            summary["mean_attacked_cell_exposure_ratio"] = float(
                group["attacked_cell_exposure_ratio"].mean(skipna=True)
            )
            summary["std_attacked_cell_exposure_ratio"] = float(
                group["attacked_cell_exposure_ratio"].std(skipna=True, ddof=0)
            )
        for exposure_column in (
            "hazard_exposure",
            "uncertainty_exposure",
            "illumination_exposure",
            "communication_exposure",
        ):
            if exposure_column in group:
                summary[f"mean_{exposure_column}"] = float(group[exposure_column].mean(skipna=True))
        if "constraint_penalty" in group:
            summary["mean_constraint_penalty"] = float(group["constraint_penalty"].mean())
            summary["std_constraint_penalty"] = float(group["constraint_penalty"].std(ddof=0))

        for name in OBJECTIVE_NAMES:
            column = f"{name}_cost"
            summary[f"mean_{name}_cost"] = float(group[column].mean(skipna=True))
            summary[f"std_{name}_cost"] = float(group[column].std(skipna=True, ddof=0))
            attacked_column = f"attacked_{name}_cost"
            if attacked_column in group:
                summary[f"mean_attacked_{name}_cost"] = float(group[attacked_column].mean(skipna=True))
                summary[f"std_attacked_{name}_cost"] = float(
                    group[attacked_column].std(skipna=True, ddof=0)
                )
            uncertainty_column = f"path_uncertainty_{name}"
            if uncertainty_column in group:
                summary[f"mean_path_uncertainty_{name}"] = float(
                    group[uncertainty_column].mean(skipna=True)
                )
                summary[f"std_path_uncertainty_{name}"] = float(
                    group[uncertainty_column].std(skipna=True, ddof=0)
                )

        rows.append(summary)

    return pd.DataFrame(rows)
