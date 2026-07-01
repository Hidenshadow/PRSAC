from __future__ import annotations

import numpy as np

from maps.map_generator import generate_costmap
from planners.weighted_astar import weighted_astar
from utils.metrics import make_planning_episode, plan_with_weights


def test_synthetic_map_generation_and_astar() -> None:
    costmap = generate_costmap(map_size=24, rng=np.random.default_rng(123), scenario="lunar_rover_corridor")
    scalar_cost = costmap.layers["distance"] + 0.5 * costmap.layers["hazard"]

    path = weighted_astar(
        scalar_cost,
        start=(1, 1),
        goal=(22, 22),
        obstacle_mask=costmap.obstacle_mask,
    )

    assert path is not None
    assert path[0] == (1, 1)
    assert path[-1] == (22, 22)


def test_planner_preference_episode_runs() -> None:
    episode = make_planning_episode(map_size=24, rng=np.random.default_rng(456))
    result = plan_with_weights(
        episode,
        weights=np.array([1.0, 0.6, 0.8, 0.4, 0.4], dtype=np.float32),
        lambda_uncertainty=0.5,
    )

    assert result["success"]
    assert np.isfinite(float(result["scalar_cost"]))
