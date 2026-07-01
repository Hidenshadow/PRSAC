"""Helpers for real DEM/DTM rover benchmark tiles.

The synthetic map generator remains separate.  This module adapts exported
real-terrain layer files into the same ``GeneratedCostMap`` / ``PlanningEpisode``
interfaces used by the weighted-A* and Gymnasium code.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from maps.map_generator import GeneratedCostMap
from utils.metrics import (
    OBJECTIVE_NAMES,
    PlanningEpisode,
    sample_mission_profile,
    sample_rover_state,
)


REAL_SCENARIO_MISSION_PROXY = {
    "real_lunar_viper": "lunar_polar_shadow",
    "real_mars_dtm": "mars_dust_low_power",
}

REAL_TASK_SAMPLING_MODES = ("distance", "risk_corridor")
DEFAULT_CORRIDOR_RISK_WEIGHTS = {
    "slope_boundary": 0.28,
    "roughness_layer": 0.10,
    "layer_energy": 0.20,
    "layer_hazard": 0.24,
    "layer_communication": 0.04,
    "layer_illumination": 0.04,
    "uncertainty_energy": 0.04,
    "uncertainty_hazard": 0.04,
    "uncertainty_communication": 0.01,
    "uncertainty_illumination": 0.01,
}


@dataclass(frozen=True)
class RealTerrainSource:
    """One exported DEM/DTM tile and optional metadata."""

    tile_id: str
    layers_path: Path
    metadata_path: Path | None = None


def read_json(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    value = json.loads(resolved.read_text(encoding="utf-8-sig"))
    return value if isinstance(value, dict) else {}


def load_real_layers(path: str | Path) -> dict[str, np.ndarray]:
    """Load an exported ``real_map_layers.npz`` file."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"real terrain layer file not found: {resolved}")
    with np.load(resolved) as data:
        return {name: data[name] for name in data.files}


def real_layer_dicts(raw_layers: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return cost and uncertainty layer dictionaries expected by the planner."""

    layers = {
        name: np.asarray(raw_layers[f"layer_{name}"], dtype=np.float32)
        for name in OBJECTIVE_NAMES
    }
    uncertainty_layers = {
        name: np.asarray(raw_layers[f"uncertainty_{name}"], dtype=np.float32)
        for name in OBJECTIVE_NAMES
    }
    return layers, uncertainty_layers


def nearest_free_cell(mask: np.ndarray, target: tuple[int, int]) -> tuple[int, int]:
    """Snap a requested cell to the nearest non-obstacle cell."""

    obstacle_mask = np.asarray(mask, dtype=bool)
    free = np.argwhere(~obstacle_mask)
    if len(free) == 0:
        raise RuntimeError("real terrain tile has no free cells")
    target_arr = np.asarray(target, dtype=np.float32)
    distances = np.sum((free.astype(np.float32) - target_arr.reshape(1, 2)) ** 2, axis=1)
    row, col = free[int(np.argmin(distances))]
    return int(row), int(col)


def sample_start_goal(
    rng: np.random.Generator,
    obstacle_mask: np.ndarray,
    min_distance_ratio: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Sample a deterministic feasible start/goal pair on one tile."""

    free = np.argwhere(~np.asarray(obstacle_mask, dtype=bool))
    if len(free) < 2:
        raise RuntimeError("not enough free cells in real terrain tile")
    map_size = int(min(obstacle_mask.shape))
    min_distance = float(min_distance_ratio) * float(map_size)
    for _ in range(4000):
        i, j = rng.choice(len(free), size=2, replace=False)
        start = free[int(i)]
        goal = free[int(j)]
        if float(np.linalg.norm(start - goal)) >= min_distance:
            return tuple(map(int, start)), tuple(map(int, goal))
    return tuple(map(int, free[0])), tuple(map(int, free[-1]))


def _unit_layer(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    finite_values = array[finite]
    if float(finite_values.min()) >= 0.0 and float(finite_values.max()) <= 1.0:
        return np.clip(array, 0.0, 1.0).astype(np.float32)
    low = float(np.percentile(finite_values, 2.0))
    high = float(np.percentile(finite_values, 98.0))
    if high - low < 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    normalized = (array - low) / (high - low)
    normalized[~finite] = 0.0
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _risk_source_layer(raw_layers: dict[str, np.ndarray], name: str) -> np.ndarray | None:
    if name in raw_layers:
        return _unit_layer(raw_layers[name])
    if name == "slope_boundary":
        slope_degrees = raw_layers.get("slope_degrees")
        max_slope = raw_layers.get("rover_max_slope_deg")
        if slope_degrees is not None and max_slope is not None:
            denominator = np.maximum(np.asarray(max_slope, dtype=np.float32), 1e-6)
            return np.clip(np.asarray(slope_degrees, dtype=np.float32) / denominator, 0.0, 1.0).astype(np.float32)
        if "slope_layer" in raw_layers:
            return _unit_layer(raw_layers["slope_layer"])
    if name == "mean_uncertainty":
        fields = [
            _unit_layer(raw_layers[f"uncertainty_{objective}"])
            for objective in OBJECTIVE_NAMES
            if f"uncertainty_{objective}" in raw_layers
        ]
        if fields:
            return np.mean(np.stack(fields, axis=0), axis=0).astype(np.float32)
    return None


def build_corridor_risk_field(
    raw_layers: dict[str, np.ndarray],
    risk_weights: dict[str, float] | None = None,
) -> np.ndarray:
    """Build a normalized terrain risk field for start-goal task sampling."""

    weights = (
        {str(name): float(weight) for name, weight in risk_weights.items()}
        if risk_weights
        else dict(DEFAULT_CORRIDOR_RISK_WEIGHTS)
    )
    weighted_layers: list[np.ndarray] = []
    positive_weights: list[float] = []
    for name, weight in weights.items():
        if float(weight) <= 0.0:
            continue
        layer = _risk_source_layer(raw_layers, name)
        if layer is None:
            continue
        weighted_layers.append(layer.astype(np.float32))
        positive_weights.append(float(weight))
    if not weighted_layers:
        shape = np.asarray(raw_layers["obstacle_mask"]).shape
        return np.zeros(shape, dtype=np.float32)
    field = np.average(
        np.stack(weighted_layers, axis=0),
        axis=0,
        weights=np.asarray(positive_weights, dtype=np.float32),
    )
    obstacle_mask = np.asarray(raw_layers["obstacle_mask"], dtype=bool)
    field = np.asarray(field, dtype=np.float32)
    field[obstacle_mask] = 0.0
    return np.clip(field, 0.0, 1.0).astype(np.float32)


def corridor_mask_between(
    start: tuple[int, int],
    goal: tuple[int, int],
    shape: tuple[int, int],
    radius: int = 2,
) -> np.ndarray:
    """Return a square-radius mask around the straight-line start-goal corridor."""

    radius = max(int(radius), 0)
    start_row, start_col = int(start[0]), int(start[1])
    goal_row, goal_col = int(goal[0]), int(goal[1])
    steps = int(max(abs(goal_row - start_row), abs(goal_col - start_col))) + 1
    rows = np.rint(np.linspace(start_row, goal_row, max(steps, 2))).astype(np.int64)
    cols = np.rint(np.linspace(start_col, goal_col, max(steps, 2))).astype(np.int64)
    mask = np.zeros(shape, dtype=bool)
    for row, col in zip(rows, cols):
        row = int(np.clip(row, 0, shape[0] - 1))
        col = int(np.clip(col, 0, shape[1] - 1))
        row0 = max(0, row - radius)
        row1 = min(shape[0], row + radius + 1)
        col0 = max(0, col - radius)
        col1 = min(shape[1], col + radius + 1)
        mask[row0:row1, col0:col1] = True
    return mask


def corridor_difficulty(
    raw_layers: dict[str, np.ndarray],
    risk_field: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    corridor_radius: int,
) -> dict[str, float | int | None]:
    obstacle_mask = np.asarray(raw_layers["obstacle_mask"], dtype=bool)
    corridor = corridor_mask_between(start, goal, obstacle_mask.shape, radius=corridor_radius)
    corridor &= ~obstacle_mask
    values = np.asarray(risk_field, dtype=np.float32)[corridor]
    if values.size == 0:
        return {
            "corridor_radius": int(corridor_radius),
            "corridor_cells": 0,
            "corridor_risk_score": 0.0,
            "corridor_mean_risk": 0.0,
            "corridor_p90_risk": 0.0,
            "corridor_max_risk": 0.0,
            "corridor_slope_p90_deg": None,
            "corridor_near_slope_limit_fraction": None,
        }

    slope_p90_deg = None
    near_limit_fraction = None
    if "slope_degrees" in raw_layers:
        slope_values = np.asarray(raw_layers["slope_degrees"], dtype=np.float32)[corridor]
        slope_p90_deg = float(np.percentile(slope_values, 90.0))
        if "rover_max_slope_deg" in raw_layers:
            max_slope = np.asarray(raw_layers["rover_max_slope_deg"], dtype=np.float32)[corridor]
            slope_ratio = slope_values / np.maximum(max_slope, 1e-6)
            near_limit_fraction = float(np.mean(slope_ratio >= 0.55))

    mean_risk = float(values.mean())
    p90_risk = float(np.percentile(values, 90.0))
    max_risk = float(values.max())
    return {
        "corridor_radius": int(corridor_radius),
        "corridor_cells": int(corridor.sum()),
        "corridor_risk_score": float(0.65 * mean_risk + 0.35 * p90_risk),
        "corridor_mean_risk": mean_risk,
        "corridor_p90_risk": p90_risk,
        "corridor_max_risk": max_risk,
        "corridor_slope_p90_deg": slope_p90_deg,
        "corridor_near_slope_limit_fraction": near_limit_fraction,
    }


def build_real_costmap(
    raw_layers: dict[str, np.ndarray],
    start: tuple[int, int],
    goal: tuple[int, int],
    scenario: str,
) -> GeneratedCostMap:
    """Build a planner costmap from exported real-terrain layers."""

    layers, uncertainty_layers = real_layer_dicts(raw_layers)
    obstacle_mask = np.asarray(raw_layers["obstacle_mask"], dtype=bool)
    safe_start = nearest_free_cell(obstacle_mask, start)
    safe_goal = nearest_free_cell(obstacle_mask, goal)
    height_map = np.asarray(raw_layers.get("height_norm", raw_layers.get("height_map")), dtype=np.float32)
    slope_layer = np.asarray(raw_layers.get("slope_layer", np.zeros_like(height_map)), dtype=np.float32)
    roughness_layer = np.asarray(raw_layers.get("roughness_layer", np.zeros_like(height_map)), dtype=np.float32)
    slope_degrees = (
        np.asarray(raw_layers["slope_degrees"], dtype=np.float32)
        if "slope_degrees" in raw_layers
        else None
    )
    rover_max_slope_degrees = (
        np.asarray(raw_layers["rover_max_slope_deg"], dtype=np.float32)
        if "rover_max_slope_deg" in raw_layers
        else None
    )
    roughness_meters = (
        np.asarray(raw_layers["roughness_m"], dtype=np.float32)
        if "roughness_m" in raw_layers
        else None
    )
    if scenario == "real_mars_dtm" and slope_degrees is not None and rover_max_slope_degrees is not None:
        slope_ratio = slope_degrees / np.maximum(rover_max_slope_degrees, 1e-6)
        slope_pressure = np.clip((slope_ratio - 0.28) / 0.72, 0.0, 1.0).astype(np.float32)
        roughness_pressure = (
            _unit_layer(roughness_meters)
            if roughness_meters is not None
            else _unit_layer(roughness_layer)
        )
        slip_risk = np.clip(0.72 * slope_pressure + 0.28 * roughness_pressure, 0.0, 1.0).astype(np.float32)
        slip_risk[obstacle_mask] = 1.0
        layers = {name: value.copy() for name, value in layers.items()}
        uncertainty_layers = {name: value.copy() for name, value in uncertainty_layers.items()}
        layers["energy"] = np.clip(layers["energy"] + 0.32 * slip_risk, 0.0, 1.0).astype(np.float32)
        layers["hazard"] = np.clip(
            np.maximum(layers["hazard"], 0.08 + 0.62 * slip_risk),
            0.0,
            1.0,
        ).astype(np.float32)
        uncertainty_layers["energy"] = np.clip(
            np.maximum(uncertainty_layers["energy"], 0.08 + 0.34 * slip_risk),
            0.0,
            1.0,
        ).astype(np.float32)
        uncertainty_layers["hazard"] = np.clip(
            np.maximum(uncertainty_layers["hazard"], 0.10 + 0.38 * slip_risk),
            0.0,
            1.0,
        ).astype(np.float32)
    return GeneratedCostMap(
        layers=layers,
        uncertainty_layers=uncertainty_layers,
        obstacle_mask=obstacle_mask,
        start=safe_start,
        goal=safe_goal,
        height_map=height_map,
        slope_layer=slope_layer,
        roughness_layer=roughness_layer,
        communication_quality=(1.0 - layers["communication"]).astype(np.float32),
        illumination_quality=(1.0 - layers["illumination"]).astype(np.float32),
        beacons=np.empty((0, 2), dtype=np.int32),
        sun_direction=np.array([1.0, 1.0, 0.2], dtype=np.float32),
        scenario=scenario,
        slope_degrees=slope_degrees,
        rover_max_slope_degrees=rover_max_slope_degrees,
        roughness_meters=roughness_meters,
    )


def make_real_planning_episode(
    raw_layers: dict[str, np.ndarray],
    task: dict[str, Any],
    rng: np.random.Generator,
    scenario: str,
    mission_profile_scenario: str | None = None,
) -> PlanningEpisode:
    """Create a real-terrain planning episode with sampled mission context."""

    start = tuple(int(v) for v in task["start"])
    goal = tuple(int(v) for v in task["goal"])
    costmap = build_real_costmap(raw_layers, start, goal, scenario=scenario)
    proxy = mission_profile_scenario or REAL_SCENARIO_MISSION_PROXY.get(scenario, "nominal")
    mission_priority, mission_regime, mission_severity = sample_mission_profile(rng, proxy)
    rover_state = sample_rover_state(
        rng,
        proxy,
        mission_regime=mission_regime,
        mission_priority=mission_priority,
        mission_severity=mission_severity,
    )
    return PlanningEpisode(
        costmap=costmap,
        mission_priority=mission_priority,
        rover_state=rover_state,
        scenario=scenario,
        mission_regime=mission_regime,
        mission_severity=mission_severity,
    )


def sample_real_tasks(
    raw_layers: dict[str, np.ndarray],
    count: int,
    seed: int,
    tile_id: str,
    split: str,
    min_distance_ratio: float = 0.62,
    meters_per_pixel: float | None = None,
    task_sampling_mode: str = "distance",
    min_corridor_risk: float | None = None,
    corridor_radius: int = 2,
    candidate_pool_multiplier: int = 30,
    risk_weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Sample start-goal tasks for a train/validation/heldout split."""

    rng = np.random.default_rng(int(seed))
    obstacle_mask = np.asarray(raw_layers["obstacle_mask"], dtype=bool)
    mode = str(task_sampling_mode)
    if mode not in REAL_TASK_SAMPLING_MODES:
        raise ValueError(f"task_sampling_mode must be one of {REAL_TASK_SAMPLING_MODES}")
    risk_field = build_corridor_risk_field(raw_layers, risk_weights=risk_weights)

    def make_task(start: tuple[int, int], goal: tuple[int, int], index: int) -> dict[str, Any]:
        distance_cells = float(np.linalg.norm(np.asarray(start, dtype=np.float32) - np.asarray(goal, dtype=np.float32)))
        difficulty = {
            "euclidean_distance_cells": distance_cells,
            "euclidean_distance_m": (
                distance_cells * float(meters_per_pixel)
                if meters_per_pixel is not None
                else None
            ),
            "min_distance_ratio": float(min_distance_ratio),
            "task_sampling_mode": mode,
            "min_corridor_risk": (
                float(min_corridor_risk)
                if min_corridor_risk is not None
                else None
            ),
        }
        difficulty.update(corridor_difficulty(raw_layers, risk_field, start, goal, corridor_radius))
        return {
            "task_id": f"{split}_{index:04d}",
            "split": split,
            "tile_id": tile_id,
            "start": [int(start[0]), int(start[1])],
            "goal": [int(goal[0]), int(goal[1])],
            "seed": int(seed) + index,
            "difficulty": difficulty,
        }

    tasks: list[dict[str, Any]] = []
    if mode == "risk_corridor" and int(count) > 0:
        pool_size = max(int(count), int(count) * max(int(candidate_pool_multiplier), 1))
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for candidate_index in range(pool_size * 3):
            start, goal = sample_start_goal(rng, obstacle_mask, min_distance_ratio=min_distance_ratio)
            key = (start, goal)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(make_task(start, goal, candidate_index))
            if len(candidates) >= pool_size:
                break
        candidates.sort(
            key=lambda item: float(item["difficulty"].get("corridor_risk_score", 0.0)),
            reverse=True,
        )
        if min_corridor_risk is None:
            selected = candidates[: int(count)]
        else:
            threshold = float(min_corridor_risk)
            selected = [
                task
                for task in candidates
                if float(task["difficulty"].get("corridor_risk_score", 0.0)) >= threshold
            ][: int(count)]
            if len(selected) < int(count):
                selected_keys = {
                    (tuple(task["start"]), tuple(task["goal"]))
                    for task in selected
                }
                selected.extend(
                    task
                    for task in candidates
                    if (tuple(task["start"]), tuple(task["goal"])) not in selected_keys
                )
                selected = selected[: int(count)]
        for index, task in enumerate(selected):
            task["task_id"] = f"{split}_{index:04d}"
            task["seed"] = int(seed) + index
            task["difficulty"]["risk_sampling_rank"] = int(index)
            task["difficulty"]["risk_sampling_pool_size"] = int(len(candidates))
            task["difficulty"]["met_min_corridor_risk"] = (
                bool(float(task["difficulty"].get("corridor_risk_score", 0.0)) >= float(min_corridor_risk))
                if min_corridor_risk is not None
                else None
            )
            tasks.append(task)
        return tasks

    for index in range(max(int(count), 0)):
        start, goal = sample_start_goal(rng, obstacle_mask, min_distance_ratio=min_distance_ratio)
        tasks.append(make_task(start, goal, index))
    return tasks


def write_task_split(path: str | Path, tasks: list[dict[str, Any]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(tasks, indent=2), encoding="utf-8")


def load_task_split(path: str | Path) -> list[dict[str, Any]]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, list):
        raise ValueError(f"task split must be a list: {path}")
    return [dict(item) for item in value]


def generate_task_splits(
    layers_path: str | Path,
    output_dir: str | Path,
    seed: int,
    tile_id: str,
    num_train_tasks: int,
    num_validation_tasks: int,
    num_heldout_tasks: int,
    min_distance_ratio: float = 0.62,
    metadata_path: str | Path | None = None,
    task_sampling_mode: str = "distance",
    min_corridor_risk: float | None = None,
    corridor_radius: int = 2,
    candidate_pool_multiplier: int = 30,
    risk_weights: dict[str, float] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Generate deterministic train/validation/heldout split files."""

    raw_layers = load_real_layers(layers_path)
    metadata = read_json(metadata_path)
    meters_per_pixel = metadata.get("meters_per_pixel")
    split_counts = {
        "train": int(num_train_tasks),
        "validation": int(num_validation_tasks),
        "heldout": int(num_heldout_tasks),
    }
    splits: dict[str, list[dict[str, Any]]] = {}
    split_dir = Path(output_dir)
    for offset, (split, count) in enumerate(split_counts.items()):
        tasks = sample_real_tasks(
            raw_layers,
            count=count,
            seed=int(seed) + 100_000 * offset,
            tile_id=tile_id,
            split=split,
            min_distance_ratio=min_distance_ratio,
            meters_per_pixel=meters_per_pixel,
            task_sampling_mode=task_sampling_mode,
            min_corridor_risk=min_corridor_risk,
            corridor_radius=corridor_radius,
            candidate_pool_multiplier=candidate_pool_multiplier,
            risk_weights=risk_weights,
        )
        splits[split] = tasks
        write_task_split(split_dir / f"{split}_tasks.json", tasks)
    return splits
