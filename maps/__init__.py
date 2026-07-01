"""Synthetic map generation utilities."""

from .map_generator import (
    DOMAIN_SCENARIO_GROUPS,
    EXTREME_SCENARIOS,
    LUNAR_SCENARIOS,
    MARS_SCENARIOS,
    SCENARIO_NAMES,
    GeneratedCostMap,
    generate_costmap,
    normalize01,
)

__all__ = [
    "DOMAIN_SCENARIO_GROUPS",
    "EXTREME_SCENARIOS",
    "LUNAR_SCENARIOS",
    "MARS_SCENARIOS",
    "SCENARIO_NAMES",
    "GeneratedCostMap",
    "generate_costmap",
    "normalize01",
]
