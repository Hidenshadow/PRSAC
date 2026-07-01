"""Synthetic rover-style multi-layer 2D cost-map generation.

The generated map is still synthetic, but the layers are framed like a lunar or
Mars rover global planning problem: distance, energy, hazard, communication, and
illumination. Obstacles represent hard no-go terrain such as steep crater rims or
high-risk rock fields. Other layers remain soft costs for weighted A*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, label


OBJECTIVE_NAMES = ("distance", "energy", "hazard", "communication", "illumination")
EXTREME_SCENARIOS = (
    "lunar_polar_shadow",
    "mars_dust_low_power",
    "crater_rim_traverse",
    "comm_blackout",
    "uncertain_hazard_corridor",
)
LUNAR_SCENARIOS = (
    "lunar_polar_shadow",
    "crater_rim_traverse",
    "comm_blackout",
    "uncertain_hazard_corridor",
    "lunar_rover_corridor",
)
MARS_SCENARIOS = (
    "mars_dust_low_power",
    "crater_rim_traverse",
    "comm_blackout",
    "uncertain_hazard_corridor",
)
DOMAIN_SCENARIO_GROUPS = {
    "lunar_rover": LUNAR_SCENARIOS,
    "mars_rover": MARS_SCENARIOS,
    "mixed_rover_extreme": EXTREME_SCENARIOS,
}
SCENARIO_NAMES = (
    "nominal",
    *EXTREME_SCENARIOS,
    "lunar_rover",
    "lunar_rover_corridor",
    "lunar_rover_bottleneck",
    "mars_rover",
    "mixed_rover_extreme",
)


SCENARIO_CONFIGS: dict[str, dict] = {
    "nominal": {
        "num_craters_range": (3, 7),
        "crater_radius_range": (0.06, 0.16),
        "rock_threshold": 0.992,
        "slope_gain": 1.0,
        "roughness_gain": 1.0,
        "hazard_multiplier": 1.0,
        "hazard_crater_boost": 0.0,
        "energy_multiplier": 1.0,
        "energy_crater_boost": 0.0,
        "obstacle_threshold_delta": 0.0,
        "slope_obstacle_threshold": 0.96,
        "num_beacons_range": (1, 2),
        "comm_scale_divisor": 2.8,
        "comm_terrain_weight": 0.18,
        "comm_quality_scale": 1.0,
        "sun_elevation_degrees": (8.0, 35.0),
        "shadow_weight": 0.22,
        "illumination_quality_scale": 1.0,
        "uncertainty_multipliers": {},
        "corridor_hazard_strength": 0.0,
        "corridor_uncertainty_strength": 0.0,
        "route_conflicts": {},
    },
    "lunar_polar_shadow": {
        "num_craters_range": (4, 9),
        "crater_radius_range": (0.07, 0.18),
        "rock_threshold": 0.990,
        "slope_gain": 1.12,
        "roughness_gain": 1.08,
        "hazard_multiplier": 1.08,
        "hazard_crater_boost": 0.08,
        "energy_multiplier": 1.12,
        "energy_crater_boost": 0.05,
        "obstacle_threshold_delta": -0.02,
        "slope_obstacle_threshold": 0.94,
        "num_beacons_range": (1, 2),
        "comm_scale_divisor": 3.0,
        "comm_terrain_weight": 0.22,
        "comm_quality_scale": 0.92,
        "sun_elevation_degrees": (2.0, 11.0),
        "shadow_weight": 0.46,
        "illumination_quality_scale": 0.62,
        "uncertainty_multipliers": {"illumination": 1.55, "energy": 1.20, "hazard": 1.15},
        "corridor_hazard_strength": 0.0,
        "corridor_uncertainty_strength": 0.0,
        "route_conflicts": {
            "width_ratio": 0.060,
            "detour_offset_ratio": 0.32,
            "layer_delta": {
                # A shadowed cold-trap shortcut is flatter and shorter but power-poor.
                "energy": {"direct": -0.06, "detour": 0.20},
                "hazard": {"direct": -0.04, "detour": 0.08},
                "communication": {"direct": 0.08, "detour": -0.06},
                "illumination": {"direct": 0.42, "detour": -0.30},
            },
            "uncertainty_delta": {
                "energy": {"direct": 0.06, "detour": 0.14},
                "hazard": {"direct": 0.04, "detour": 0.10},
                "communication": {"direct": 0.10, "detour": -0.04},
                "illumination": {"direct": 0.34, "detour": 0.04},
            },
        },
    },
    "mars_dust_low_power": {
        "num_craters_range": (2, 6),
        "crater_radius_range": (0.06, 0.15),
        "rock_threshold": 0.991,
        "slope_gain": 0.95,
        "roughness_gain": 1.05,
        "hazard_multiplier": 0.98,
        "hazard_crater_boost": 0.03,
        "energy_multiplier": 1.18,
        "energy_crater_boost": 0.03,
        "obstacle_threshold_delta": 0.01,
        "slope_obstacle_threshold": 0.97,
        "num_beacons_range": (1, 1),
        "comm_scale_divisor": 3.4,
        "comm_terrain_weight": 0.26,
        "comm_quality_scale": 0.72,
        "sun_elevation_degrees": (10.0, 28.0),
        "shadow_weight": 0.30,
        "illumination_quality_scale": 0.54,
        "uncertainty_multipliers": {"communication": 1.35, "illumination": 1.45, "energy": 1.25},
        "corridor_hazard_strength": 0.0,
        "corridor_uncertainty_strength": 0.0,
        "route_conflicts": {
            "width_ratio": 0.065,
            "detour_offset_ratio": 0.30,
            "layer_delta": {
                # Dusty basins are comparatively safe but inefficient and power-limited.
                "energy": {"direct": 0.22, "detour": 0.10},
                "hazard": {"direct": -0.06, "detour": 0.12},
                "communication": {"direct": 0.18, "detour": -0.16},
                "illumination": {"direct": 0.34, "detour": -0.28},
            },
            "uncertainty_delta": {
                "energy": {"direct": 0.26, "detour": 0.08},
                "hazard": {"direct": 0.08, "detour": 0.14},
                "communication": {"direct": 0.18, "detour": -0.06},
                "illumination": {"direct": 0.28, "detour": -0.04},
            },
        },
    },
    "crater_rim_traverse": {
        "num_craters_range": (7, 13),
        "crater_radius_range": (0.08, 0.22),
        "rock_threshold": 0.989,
        "slope_gain": 1.25,
        "roughness_gain": 1.18,
        "hazard_multiplier": 1.18,
        "hazard_crater_boost": 0.16,
        "energy_multiplier": 1.22,
        "energy_crater_boost": 0.12,
        "obstacle_threshold_delta": -0.05,
        "slope_obstacle_threshold": 0.91,
        "num_beacons_range": (1, 2),
        "comm_scale_divisor": 2.8,
        "comm_terrain_weight": 0.24,
        "comm_quality_scale": 0.90,
        "sun_elevation_degrees": (7.0, 25.0),
        "shadow_weight": 0.32,
        "illumination_quality_scale": 0.86,
        "uncertainty_multipliers": {"hazard": 1.55, "energy": 1.35},
        "corridor_hazard_strength": 0.0,
        "corridor_uncertainty_strength": 0.0,
        "route_conflicts": {
            "width_ratio": 0.055,
            "detour_offset_ratio": 0.34,
            "layer_delta": {
                # Rim shortcuts preserve distance and visibility but cost energy and hazard margin.
                "energy": {"direct": 0.28, "detour": -0.12},
                "hazard": {"direct": 0.40, "detour": -0.16},
                "communication": {"direct": -0.08, "detour": 0.10},
                "illumination": {"direct": -0.06, "detour": 0.04},
            },
            "uncertainty_delta": {
                "energy": {"direct": 0.24, "detour": -0.08},
                "hazard": {"direct": 0.42, "detour": -0.12},
                "communication": {"direct": 0.04, "detour": 0.10},
                "illumination": {"direct": 0.02, "detour": 0.06},
            },
        },
    },
    "comm_blackout": {
        "num_craters_range": (4, 8),
        "crater_radius_range": (0.06, 0.16),
        "rock_threshold": 0.991,
        "slope_gain": 1.02,
        "roughness_gain": 1.08,
        "hazard_multiplier": 1.02,
        "hazard_crater_boost": 0.04,
        "energy_multiplier": 1.06,
        "energy_crater_boost": 0.02,
        "obstacle_threshold_delta": -0.01,
        "slope_obstacle_threshold": 0.95,
        "num_beacons_range": (1, 1),
        "comm_scale_divisor": 5.0,
        "comm_terrain_weight": 0.44,
        "comm_quality_scale": 0.56,
        "sun_elevation_degrees": (8.0, 30.0),
        "shadow_weight": 0.26,
        "illumination_quality_scale": 0.92,
        "uncertainty_multipliers": {"communication": 1.80, "hazard": 1.08},
        "corridor_hazard_strength": 0.0,
        "corridor_uncertainty_strength": 0.0,
        "route_conflicts": {
            "width_ratio": 0.060,
            "detour_offset_ratio": 0.30,
            "layer_delta": {
                # Low basins are easy to drive but lose line-of-sight to relay assets.
                "energy": {"direct": -0.10, "detour": 0.18},
                "hazard": {"direct": -0.06, "detour": 0.14},
                "communication": {"direct": 0.46, "detour": -0.34},
                "illumination": {"direct": 0.04, "detour": -0.06},
            },
            "uncertainty_delta": {
                "energy": {"direct": 0.04, "detour": 0.12},
                "hazard": {"direct": 0.04, "detour": 0.14},
                "communication": {"direct": 0.48, "detour": -0.12},
                "illumination": {"direct": 0.04, "detour": 0.02},
            },
        },
    },
    "uncertain_hazard_corridor": {
        "num_craters_range": (4, 8),
        "crater_radius_range": (0.06, 0.17),
        "rock_threshold": 0.990,
        "slope_gain": 1.04,
        "roughness_gain": 1.10,
        "hazard_multiplier": 1.05,
        "hazard_crater_boost": 0.06,
        "energy_multiplier": 1.08,
        "energy_crater_boost": 0.03,
        "obstacle_threshold_delta": -0.015,
        "slope_obstacle_threshold": 0.95,
        "num_beacons_range": (1, 2),
        "comm_scale_divisor": 3.0,
        "comm_terrain_weight": 0.22,
        "comm_quality_scale": 0.86,
        "sun_elevation_degrees": (6.0, 24.0),
        "shadow_weight": 0.32,
        "illumination_quality_scale": 0.82,
        "uncertainty_multipliers": {"hazard": 1.45, "communication": 1.20, "illumination": 1.20},
        "corridor_hazard_strength": 0.05,
        "corridor_uncertainty_strength": 0.55,
        "route_conflicts": {
            "width_ratio": 0.060,
            "detour_offset_ratio": 0.36,
            "layer_delta": {
                # The shortcut looks cheap in the nominal map but is poorly verified.
                "energy": {"direct": -0.14, "detour": 0.10},
                "hazard": {"direct": -0.08, "detour": 0.04},
                "communication": {"direct": -0.04, "detour": 0.04},
                "illumination": {"direct": 0.02, "detour": 0.04},
            },
            "uncertainty_delta": {
                "energy": {"direct": 0.26, "detour": -0.10},
                "hazard": {"direct": 0.62, "detour": -0.16},
                "communication": {"direct": 0.16, "detour": -0.06},
                "illumination": {"direct": 0.12, "detour": -0.04},
            },
        },
    },
    "lunar_rover_corridor": {
        "num_craters_range": (3, 6),
        "crater_radius_range": (0.05, 0.13),
        "rock_threshold": 0.993,
        "slope_gain": 0.94,
        "roughness_gain": 0.94,
        "hazard_multiplier": 0.92,
        "hazard_crater_boost": 0.02,
        "energy_multiplier": 0.94,
        "energy_crater_boost": 0.01,
        "obstacle_threshold_delta": 0.02,
        "slope_obstacle_threshold": 0.98,
        "num_beacons_range": (1, 2),
        "comm_scale_divisor": 3.0,
        "comm_terrain_weight": 0.18,
        "comm_quality_scale": 0.94,
        "sun_elevation_degrees": (7.0, 24.0),
        "shadow_weight": 0.30,
        "illumination_quality_scale": 0.88,
        "uncertainty_multipliers": {"hazard": 1.35, "energy": 1.15},
        "corridor_hazard_strength": 0.02,
        "corridor_uncertainty_strength": 0.20,
        "route_conflicts": {
            "width_ratio": 0.055,
            "detour_offset_ratio": 0.34,
            "layer_delta": {
                # Direct corridor is nominally attractive, but it is the
                # attack-prone route. The detour is longer/slightly more costly
                # nominally but has substantially lower uncertainty.
                "energy": {"direct": -0.20, "detour": 0.11},
                "hazard": {"direct": -0.18, "detour": 0.07},
                "communication": {"direct": -0.04, "detour": 0.03},
                "illumination": {"direct": -0.03, "detour": 0.04},
            },
            "uncertainty_delta": {
                "energy": {"direct": 0.36, "detour": -0.08},
                "hazard": {"direct": 0.72, "detour": -0.18},
                "communication": {"direct": 0.12, "detour": -0.04},
                "illumination": {"direct": 0.10, "detour": -0.04},
            },
        },
    },
    "lunar_rover_bottleneck": {
        "num_craters_range": (5, 9),
        "crater_radius_range": (0.06, 0.16),
        "rock_threshold": 0.991,
        "slope_gain": 1.02,
        "roughness_gain": 1.02,
        "hazard_multiplier": 1.00,
        "hazard_crater_boost": 0.04,
        "energy_multiplier": 1.00,
        "energy_crater_boost": 0.02,
        "obstacle_threshold_delta": 0.00,
        "slope_obstacle_threshold": 0.96,
        "num_beacons_range": (1, 2),
        "comm_scale_divisor": 3.1,
        "comm_terrain_weight": 0.20,
        "comm_quality_scale": 0.90,
        "sun_elevation_degrees": (6.0, 24.0),
        "shadow_weight": 0.34,
        "illumination_quality_scale": 0.84,
        "uncertainty_multipliers": {"hazard": 1.45, "communication": 1.15},
        "corridor_hazard_strength": 0.03,
        "corridor_uncertainty_strength": 0.28,
        "route_conflicts": {
            "width_ratio": 0.045,
            "detour_offset_ratio": 0.40,
            "layer_delta": {
                "energy": {"direct": -0.16, "detour": 0.13},
                "hazard": {"direct": -0.14, "detour": 0.10},
                "communication": {"direct": -0.02, "detour": 0.04},
                "illumination": {"direct": 0.00, "detour": 0.04},
            },
            "uncertainty_delta": {
                "energy": {"direct": 0.28, "detour": -0.06},
                "hazard": {"direct": 0.82, "detour": -0.16},
                "communication": {"direct": 0.16, "detour": -0.02},
                "illumination": {"direct": 0.08, "detour": -0.02},
            },
        },
    },
}


@dataclass(frozen=True)
class GeneratedCostMap:
    """Container for one generated rover planning map."""

    layers: dict[str, np.ndarray]
    uncertainty_layers: dict[str, np.ndarray]
    obstacle_mask: np.ndarray
    start: tuple[int, int]
    goal: tuple[int, int]
    height_map: np.ndarray
    slope_layer: np.ndarray
    roughness_layer: np.ndarray
    communication_quality: np.ndarray
    illumination_quality: np.ndarray
    beacons: np.ndarray
    sun_direction: np.ndarray
    scenario: str = "nominal"
    attack_mask: np.ndarray | None = None
    attack_metadata: dict[str, Any] | None = None
    slope_degrees: np.ndarray | None = None
    rover_max_slope_degrees: np.ndarray | None = None
    roughness_meters: np.ndarray | None = None

    @property
    def signal_quality(self) -> np.ndarray:
        """Backward-compatible alias for older plotting/debug code."""

        return self.communication_quality

    @property
    def solar_quality(self) -> np.ndarray:
        """Backward-compatible alias for older plotting/debug code."""

        return self.illumination_quality


def normalize01(values: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1], returning zeros for constant input."""

    array = np.asarray(values, dtype=np.float32)
    finite_mask = np.isfinite(array)
    if not finite_mask.any():
        return np.zeros_like(array, dtype=np.float32)

    min_value = float(array[finite_mask].min())
    max_value = float(array[finite_mask].max())
    span = max_value - min_value
    if span < 1e-8:
        return np.zeros_like(array, dtype=np.float32)

    out = (array - min_value) / span
    out[~finite_mask] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def resolve_scenario_name(
    scenario: str,
    rng: np.random.Generator,
) -> str:
    """Resolve scenario aliases into one concrete map-generation mode."""

    if scenario not in SCENARIO_NAMES:
        raise ValueError(f"scenario must be one of {SCENARIO_NAMES}, got {scenario!r}")
    if scenario in DOMAIN_SCENARIO_GROUPS:
        return str(rng.choice(DOMAIN_SCENARIO_GROUPS[scenario]))
    return scenario


def _scenario_config(scenario: str) -> dict:
    return SCENARIO_CONFIGS.get(scenario, SCENARIO_CONFIGS["nominal"])


def _sample_int_inclusive(
    rng: np.random.Generator,
    bounds: tuple[int, int],
) -> int:
    low, high = int(bounds[0]), int(bounds[1])
    if high < low:
        low, high = high, low
    return int(rng.integers(low, high + 1))


def _smooth_noise(
    rng: np.random.Generator,
    map_size: int,
    sigma: float,
) -> np.ndarray:
    noise = rng.standard_normal((map_size, map_size)).astype(np.float32)
    return normalize01(gaussian_filter(noise, sigma=sigma, mode="reflect"))


def _connectivity_structure(allow_diagonal: bool) -> np.ndarray:
    if allow_diagonal:
        return np.ones((3, 3), dtype=np.int8)
    return np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)


def _largest_free_component(
    free_mask: np.ndarray,
    allow_diagonal: bool,
) -> np.ndarray:
    labeled, num_labels = label(
        free_mask.astype(np.int8),
        structure=_connectivity_structure(allow_diagonal),
    )
    if num_labels == 0:
        return np.zeros_like(free_mask, dtype=bool)

    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = int(counts.argmax())
    return labeled == largest_label


def _sample_start_goal(
    rng: np.random.Generator,
    free_mask: np.ndarray,
    map_size: int,
    min_distance_ratio: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    coords = np.argwhere(free_mask)
    if len(coords) < 2:
        raise RuntimeError("not enough free cells to sample start and goal")

    requested_min_distance = min_distance_ratio * float(map_size)
    relaxed_distances = (
        requested_min_distance,
        0.45 * map_size,
        0.30 * map_size,
        0.15 * map_size,
        0.0,
    )

    for min_distance in relaxed_distances:
        for _ in range(4000):
            first_index, second_index = rng.choice(len(coords), size=2, replace=False)
            start_arr = coords[first_index]
            goal_arr = coords[second_index]
            if np.linalg.norm(start_arr - goal_arr) >= min_distance:
                return tuple(map(int, start_arr)), tuple(map(int, goal_arr))

    start_arr = coords[0]
    goal_arr = coords[-1]
    return tuple(map(int, start_arr)), tuple(map(int, goal_arr))


def _sample_beacons(
    rng: np.random.Generator,
    free_mask: np.ndarray,
    num_beacons: int,
) -> np.ndarray:
    free_coords = np.argwhere(free_mask)
    if len(free_coords) == 0:
        rows, cols = np.indices(free_mask.shape)
        free_coords = np.column_stack([rows.ravel(), cols.ravel()])

    replace = len(free_coords) < num_beacons
    indices = rng.choice(len(free_coords), size=num_beacons, replace=replace)
    return free_coords[indices].astype(np.int32)


def _communication_layers(
    rng: np.random.Generator,
    free_mask: np.ndarray,
    height_map: np.ndarray,
    map_size: int,
    num_beacons: int,
    range_scale_divisor: float = 2.8,
    terrain_weight: float = 0.18,
    quality_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    beacons = _sample_beacons(rng, free_mask, num_beacons=num_beacons)
    rows, cols = np.indices((map_size, map_size))

    distances = []
    for beacon_row, beacon_col in beacons:
        distances.append(np.hypot(rows - beacon_row, cols - beacon_col))
    nearest_distance = np.minimum.reduce(distances)

    scale = max(float(map_size) / max(float(range_scale_divisor), 1e-6), 1.0)
    range_quality = np.exp(-nearest_distance / scale).astype(np.float32)

    # A simple terrain attenuation term: locally high ridges reduce comm quality.
    terrain_occlusion = normalize01(gaussian_filter(height_map, sigma=max(map_size / 18.0, 1.0)))
    terrain_weight = float(np.clip(terrain_weight, 0.0, 1.0))
    communication_quality = normalize01(
        (1.0 - terrain_weight) * range_quality + terrain_weight * (1.0 - terrain_occlusion)
    )
    communication_quality = np.clip(communication_quality * float(quality_scale), 0.0, 1.0)
    communication_cost = (1.0 - communication_quality).astype(np.float32)
    return communication_quality.astype(np.float32), communication_cost, beacons


def _roughness_from_height(height_map: np.ndarray, map_size: int) -> np.ndarray:
    sigma = max(map_size / 48.0, 1.0)
    mean = gaussian_filter(height_map, sigma=sigma, mode="reflect")
    mean_sq = gaussian_filter(height_map * height_map, sigma=sigma, mode="reflect")
    variance = np.maximum(mean_sq - mean * mean, 0.0)
    return normalize01(np.sqrt(variance))


def _crater_and_rock_field(
    rng: np.random.Generator,
    map_size: int,
    num_craters_range: tuple[int, int] = (3, 7),
    crater_radius_range: tuple[float, float] = (0.06, 0.16),
    rock_threshold: float = 0.992,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices((map_size, map_size))
    crater_field = np.zeros((map_size, map_size), dtype=np.float32)

    num_craters = _sample_int_inclusive(rng, num_craters_range)
    for _ in range(num_craters):
        center_row = float(rng.uniform(0.12 * map_size, 0.88 * map_size))
        center_col = float(rng.uniform(0.12 * map_size, 0.88 * map_size))
        radius = float(rng.uniform(crater_radius_range[0] * map_size, crater_radius_range[1] * map_size))
        distance = np.hypot(rows - center_row, cols - center_col)

        rim = np.exp(-((distance - radius) ** 2) / (2.0 * (0.16 * radius) ** 2))
        interior = np.exp(-(distance**2) / (2.0 * (0.55 * radius) ** 2))
        crater_field += (0.85 * rim + 0.35 * interior).astype(np.float32)

    rock_noise = rng.random((map_size, map_size), dtype=np.float32)
    rock_seed = (rock_noise > float(rock_threshold)).astype(np.float32)
    rock_field = normalize01(gaussian_filter(rock_seed, sigma=1.1, mode="reflect"))
    return normalize01(crater_field), rock_field


def _illumination_layers(
    rng: np.random.Generator,
    height_map: np.ndarray,
    map_size: int,
    elevation_degrees: tuple[float, float] = (8.0, 35.0),
    shadow_weight: float = 0.22,
    quality_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grad_y, grad_x = np.gradient(height_map)
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = np.ones_like(height_map, dtype=np.float32)
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2) + 1e-8
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm

    azimuth = float(rng.uniform(0.0, 2.0 * np.pi))
    min_elevation, max_elevation = sorted((float(elevation_degrees[0]), float(elevation_degrees[1])))
    elevation = float(rng.uniform(np.deg2rad(min_elevation), np.deg2rad(max_elevation)))
    sun_direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float32,
    )

    incidence = (
        normal_x * sun_direction[0]
        + normal_y * sun_direction[1]
        + normal_z * sun_direction[2]
    )
    incidence = np.clip(incidence, 0.0, None)
    ridge_shadow = normalize01(gaussian_filter(height_map, sigma=max(map_size / 20.0, 1.0)))
    shadow_weight = float(np.clip(shadow_weight, 0.0, 1.0))
    illumination_quality = normalize01(
        (1.0 - shadow_weight) * incidence + shadow_weight * (1.0 - ridge_shadow)
    )
    illumination_quality = np.clip(illumination_quality * float(quality_scale), 0.0, 1.0)
    illumination_cost = (1.0 - illumination_quality).astype(np.float32)
    return illumination_quality.astype(np.float32), illumination_cost, sun_direction


def _corridor_field(
    rng: np.random.Generator,
    map_size: int,
) -> np.ndarray:
    """Create a traversable but uncertain corridor-like feature.

    This is used to model a rover stress case where orbital/sensor uncertainty
    is concentrated along a plausible traverse, such as a dust-obscured channel
    or shadowed crater-rim approach. The corridor is mostly a soft uncertainty
    feature; it should not become a hard wall by itself.
    """

    rows, cols = np.indices((map_size, map_size))
    width = float(rng.uniform(0.035 * map_size, 0.075 * map_size))
    mode = str(rng.choice(("horizontal", "vertical", "diagonal")))

    if mode == "horizontal":
        center = float(rng.uniform(0.25 * map_size, 0.75 * map_size))
        distance = np.abs(rows - center)
    elif mode == "vertical":
        center = float(rng.uniform(0.25 * map_size, 0.75 * map_size))
        distance = np.abs(cols - center)
    else:
        offset = float(rng.uniform(-0.25 * map_size, 0.25 * map_size))
        sign = float(rng.choice((-1.0, 1.0)))
        distance = np.abs((rows - 0.5 * map_size) - sign * (cols - 0.5 * map_size) - offset) / np.sqrt(2.0)

    field = np.exp(-(distance**2) / (2.0 * width**2)).astype(np.float32)
    modulation = _smooth_noise(rng, map_size, sigma=max(map_size / 18.0, 1.0))
    return normalize01(0.72 * field + 0.28 * modulation)


def _distance_to_segment(
    rows: np.ndarray,
    cols: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    """Distance from each grid cell center to a line segment."""

    segment = end - start
    segment_norm_sq = float(np.dot(segment, segment))
    if segment_norm_sq < 1e-8:
        return np.hypot(rows - start[0], cols - start[1])

    row_delta = rows - start[0]
    col_delta = cols - start[1]
    t = (row_delta * segment[0] + col_delta * segment[1]) / segment_norm_sq
    t = np.clip(t, 0.0, 1.0)
    closest_row = start[0] + t * segment[0]
    closest_col = start[1] + t * segment[1]
    return np.hypot(rows - closest_row, cols - closest_col)


def _polyline_tube(
    map_size: int,
    points: tuple[np.ndarray, ...],
    width: float,
) -> np.ndarray:
    """Create a normalized Gaussian tube around a polyline."""

    rows, cols = np.indices((map_size, map_size))
    min_distance = np.full((map_size, map_size), np.inf, dtype=np.float32)
    for start, end in zip(points[:-1], points[1:]):
        distance = _distance_to_segment(rows, cols, start, end)
        min_distance = np.minimum(min_distance, distance).astype(np.float32)

    width = max(float(width), 1.0)
    field = np.exp(-(min_distance**2) / (2.0 * width**2)).astype(np.float32)
    return normalize01(field)


def _route_conflict_fields(
    rng: np.random.Generator,
    map_size: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    width_ratio: float,
    detour_offset_ratio: float,
) -> dict[str, np.ndarray]:
    """Construct direct and offset traverse corridors for objective conflicts.

    The direct tube approximates the route a distance-minimizing planner would
    prefer. The detour tube is offset from the start-goal line and represents a
    plausible ridge, sunlit traverse, or surveyed bypass. Scenario-specific
    deltas decide which objectives each tube helps or hurts.
    """

    start_arr = np.array(start, dtype=np.float32)
    goal_arr = np.array(goal, dtype=np.float32)
    route = goal_arr - start_arr
    route_norm = float(np.linalg.norm(route))
    width = max(float(width_ratio) * float(map_size), 1.0)

    direct = _polyline_tube(map_size, (start_arr, goal_arr), width)
    if route_norm < 1e-6:
        return {"direct": direct, "detour": np.zeros_like(direct)}

    unit = route / route_norm
    perpendicular = np.array([-unit[1], unit[0]], dtype=np.float32)
    sign = float(rng.choice((-1.0, 1.0)))
    offset = sign * float(detour_offset_ratio) * float(map_size) * perpendicular
    midpoint = 0.5 * (start_arr + goal_arr) + offset
    midpoint = np.clip(midpoint, 1.0, float(map_size - 2)).astype(np.float32)

    detour = _polyline_tube(map_size, (start_arr, midpoint, goal_arr), width)
    return {"direct": direct, "detour": detour}


def _apply_route_conflict_deltas(
    layer: np.ndarray,
    direct: np.ndarray,
    detour: np.ndarray,
    objective_name: str,
    delta_config: dict[str, dict[str, float]],
) -> np.ndarray:
    """Apply objective-specific direct/detour deltas while keeping [0, 1]."""

    objective_delta = delta_config.get(objective_name, {})
    direct_delta = float(objective_delta.get("direct", 0.0))
    detour_delta = float(objective_delta.get("detour", 0.0))
    return np.clip(layer + direct_delta * direct + detour_delta * detour, 0.0, 1.0).astype(np.float32)


def _uncertainty_layers(
    rng: np.random.Generator,
    map_size: int,
    slope_layer: np.ndarray,
    roughness_layer: np.ndarray,
    crater_field: np.ndarray,
    rock_field: np.ndarray,
    hazard_layer: np.ndarray,
    communication_quality: np.ndarray,
    illumination_quality: np.ndarray,
    multipliers: dict[str, float] | None = None,
    corridor_field: np.ndarray | None = None,
    corridor_strength: float = 0.0,
) -> dict[str, np.ndarray]:
    """Create objective-specific uncertainty maps for robust planning tests.

    The uncertainty layers approximate where a rover would be less confident in
    the nominal cost map: crater rims, rough terrain, weak communication zones,
    and illumination/shadow boundaries. Values are normalized costs in [0, 1].
    """

    broad_noise = _smooth_noise(rng, map_size, sigma=max(map_size / 16.0, 1.0))
    fine_noise = _smooth_noise(rng, map_size, sigma=max(map_size / 36.0, 1.0))
    uncertainty_noise = normalize01(0.65 * broad_noise + 0.35 * fine_noise)

    hazard_grad_y, hazard_grad_x = np.gradient(hazard_layer)
    hazard_boundary = normalize01(np.hypot(hazard_grad_x, hazard_grad_y))

    illum_grad_y, illum_grad_x = np.gradient(illumination_quality)
    shadow_boundary = normalize01(np.hypot(illum_grad_x, illum_grad_y))

    distance_uncertainty = np.clip(0.04 + 0.06 * uncertainty_noise, 0.0, 1.0).astype(np.float32)
    energy_uncertainty = normalize01(
        0.32 * slope_layer
        + 0.28 * roughness_layer
        + 0.22 * crater_field
        + 0.18 * uncertainty_noise
    )
    hazard_uncertainty = normalize01(
        0.34 * hazard_boundary
        + 0.28 * crater_field
        + 0.22 * rock_field
        + 0.16 * uncertainty_noise
    )
    communication_uncertainty = normalize01(
        0.55 * (1.0 - communication_quality)
        + 0.20 * hazard_boundary
        + 0.25 * uncertainty_noise
    )
    illumination_uncertainty = normalize01(
        0.42 * shadow_boundary
        + 0.32 * (1.0 - illumination_quality)
        + 0.26 * uncertainty_noise
    )

    layers = {
        "distance": distance_uncertainty,
        "energy": energy_uncertainty.astype(np.float32),
        "hazard": hazard_uncertainty.astype(np.float32),
        "communication": communication_uncertainty.astype(np.float32),
        "illumination": illumination_uncertainty.astype(np.float32),
    }

    if corridor_field is not None and corridor_strength > 0.0:
        corridor = np.asarray(corridor_field, dtype=np.float32)
        for name in ("energy", "hazard", "communication", "illumination"):
            layers[name] = np.clip(
                layers[name] + float(corridor_strength) * corridor,
                0.0,
                1.0,
            ).astype(np.float32)

    for name, multiplier in (multipliers or {}).items():
        if name in layers:
            layers[name] = np.clip(
                layers[name] * float(multiplier),
                0.0,
                1.0,
            ).astype(np.float32)

    return layers


def generate_costmap(
    map_size: int = 64,
    rng: np.random.Generator | None = None,
    obstacle_threshold: float = 0.88,
    min_start_goal_distance_ratio: float = 0.55,
    allow_diagonal: bool = True,
    num_beacons: int | None = None,
    scenario: str = "nominal",
    max_attempts: int = 50,
) -> GeneratedCostMap:
    """Generate one synthetic lunar/Mars rover planning map."""

    if map_size < 8:
        raise ValueError("map_size must be at least 8")
    if rng is None:
        rng = np.random.default_rng()

    resolved_scenario = resolve_scenario_name(scenario, rng)
    config = _scenario_config(resolved_scenario)
    selected_num_beacons = int(
        num_beacons
        if num_beacons is not None
        else _sample_int_inclusive(rng, config["num_beacons_range"])
    )
    effective_obstacle_threshold = float(
        np.clip(
            obstacle_threshold + float(config["obstacle_threshold_delta"]),
            0.70,
            0.98,
        )
    )

    for _ in range(max_attempts):
        distance_layer = np.ones((map_size, map_size), dtype=np.float32)

        broad_terrain = _smooth_noise(rng, map_size, sigma=max(map_size / 10.0, 1.0))
        fine_terrain = _smooth_noise(rng, map_size, sigma=max(map_size / 28.0, 1.0))
        height_map = normalize01(0.75 * broad_terrain + 0.25 * fine_terrain)

        grad_y, grad_x = np.gradient(height_map)
        slope_layer = normalize01(np.hypot(grad_x, grad_y))
        roughness_layer = _roughness_from_height(height_map, map_size)
        slope_layer = np.clip(slope_layer * float(config["slope_gain"]), 0.0, 1.0).astype(np.float32)
        roughness_layer = np.clip(
            roughness_layer * float(config["roughness_gain"]),
            0.0,
            1.0,
        ).astype(np.float32)
        crater_field, rock_field = _crater_and_rock_field(
            rng,
            map_size,
            num_craters_range=config["num_craters_range"],
            crater_radius_range=config["crater_radius_range"],
            rock_threshold=float(config["rock_threshold"]),
        )
        hazard_noise = _smooth_noise(rng, map_size, sigma=max(map_size / 12.0, 1.0))
        corridor_field = (
            _corridor_field(rng, map_size)
            if float(config["corridor_uncertainty_strength"]) > 0.0
            else None
        )

        hazard_layer = normalize01(
            0.30 * hazard_noise
            + 0.25 * slope_layer
            + 0.20 * roughness_layer
            + 0.20 * crater_field
            + 0.05 * rock_field
        )
        if corridor_field is not None and float(config["corridor_hazard_strength"]) > 0.0:
            hazard_layer = np.clip(
                hazard_layer + float(config["corridor_hazard_strength"]) * corridor_field,
                0.0,
                1.0,
            )
        hazard_layer = np.clip(
            float(config["hazard_multiplier"]) * hazard_layer
            + float(config["hazard_crater_boost"]) * crater_field,
            0.0,
            1.0,
        ).astype(np.float32)
        energy_layer = normalize01(0.48 * slope_layer + 0.32 * roughness_layer + 0.20 * crater_field)
        energy_layer = np.clip(
            float(config["energy_multiplier"]) * energy_layer
            + float(config["energy_crater_boost"]) * crater_field,
            0.0,
            1.0,
        ).astype(np.float32)

        communication_quality, communication_cost, beacons = _communication_layers(
            rng,
            np.ones((map_size, map_size), dtype=bool),
            height_map,
            map_size,
            num_beacons=selected_num_beacons,
            range_scale_divisor=float(config["comm_scale_divisor"]),
            terrain_weight=float(config["comm_terrain_weight"]),
            quality_scale=float(config["comm_quality_scale"]),
        )
        illumination_quality, illumination_cost, sun_direction = _illumination_layers(
            rng,
            height_map,
            map_size,
            elevation_degrees=config["sun_elevation_degrees"],
            shadow_weight=float(config["shadow_weight"]),
            quality_scale=float(config["illumination_quality_scale"]),
        )

        obstacle_mask = (hazard_layer > effective_obstacle_threshold) | (
            slope_layer > float(config["slope_obstacle_threshold"])
        )
        largest_component = _largest_free_component(~obstacle_mask, allow_diagonal)
        component_fraction = float(largest_component.mean())
        if component_fraction < 0.25:
            continue

        try:
            start, goal = _sample_start_goal(
                rng,
                largest_component,
                map_size,
                min_start_goal_distance_ratio,
            )
        except RuntimeError:
            continue

        obstacle_mask[start] = False
        obstacle_mask[goal] = False

        route_fields = None
        route_conflicts = config.get("route_conflicts", {})
        if route_conflicts:
            route_fields = _route_conflict_fields(
                rng=rng,
                map_size=map_size,
                start=start,
                goal=goal,
                width_ratio=float(route_conflicts.get("width_ratio", 0.06)),
                detour_offset_ratio=float(route_conflicts.get("detour_offset_ratio", 0.30)),
            )
            direct_route = route_fields["direct"]
            detour_route = route_fields["detour"]
            layer_delta = route_conflicts.get("layer_delta", {})

            # These are soft objective trade-offs, not hard no-go regions. The
            # hard obstacle mask stays tied to the base terrain so the conflict
            # changes preference selection rather than simply blocking routes.
            energy_layer = _apply_route_conflict_deltas(
                energy_layer,
                direct_route,
                detour_route,
                "energy",
                layer_delta,
            )
            hazard_layer = _apply_route_conflict_deltas(
                hazard_layer,
                direct_route,
                detour_route,
                "hazard",
                layer_delta,
            )
            communication_cost = _apply_route_conflict_deltas(
                communication_cost,
                direct_route,
                detour_route,
                "communication",
                layer_delta,
            )
            illumination_cost = _apply_route_conflict_deltas(
                illumination_cost,
                direct_route,
                detour_route,
                "illumination",
                layer_delta,
            )
            communication_quality = (1.0 - communication_cost).astype(np.float32)
            illumination_quality = (1.0 - illumination_cost).astype(np.float32)

        layers = {
            "distance": distance_layer,
            "energy": energy_layer,
            "hazard": hazard_layer,
            "communication": communication_cost,
            "illumination": illumination_cost,
        }
        uncertainty_layers = _uncertainty_layers(
            rng=rng,
            map_size=map_size,
            slope_layer=slope_layer,
            roughness_layer=roughness_layer,
            crater_field=crater_field,
            rock_field=rock_field,
            hazard_layer=hazard_layer,
            communication_quality=communication_quality,
            illumination_quality=illumination_quality,
            multipliers=config["uncertainty_multipliers"],
            corridor_field=corridor_field,
            corridor_strength=float(config["corridor_uncertainty_strength"]),
        )
        if route_fields is not None:
            uncertainty_delta = route_conflicts.get("uncertainty_delta", {})
            direct_route = route_fields["direct"]
            detour_route = route_fields["detour"]
            for name in OBJECTIVE_NAMES:
                uncertainty_layers[name] = _apply_route_conflict_deltas(
                    uncertainty_layers[name],
                    direct_route,
                    detour_route,
                    name,
                    uncertainty_delta,
                )

        return GeneratedCostMap(
            layers=layers,
            uncertainty_layers=uncertainty_layers,
            obstacle_mask=obstacle_mask.astype(bool),
            start=start,
            goal=goal,
            height_map=height_map,
            slope_layer=slope_layer,
            roughness_layer=roughness_layer,
            communication_quality=communication_quality,
            illumination_quality=illumination_quality,
            beacons=beacons,
            sun_direction=sun_direction,
            scenario=resolved_scenario,
        )

    raise RuntimeError(
        "failed to generate a map with enough connected free space; "
        "try a larger map or higher obstacle_threshold"
    )
