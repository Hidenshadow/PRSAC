from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from envs.attack_wrappers import apply_environment_attack_to_episode
from scripts.evaluate_nonlearning_planner_baselines import (
    apply_risk_inflation_to_episode,
    belief_sample_cvar_config,
    plan_cost,
    risk_inflated_planner_config,
    sample_plausible_belief_episode,
)
from utils.metrics import OBJECTIVE_NAMES, make_planning_episode


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        include_all_presets=False,
        risk_inflation_scale=0.42,
        risk_inflation_radius=2,
        belief_cvar_samples=3,
        belief_cvar_alpha=0.5,
        belief_cvar_noise_scale=0.4,
    )


def test_risk_inflation_preserves_true_map_and_increases_belief_risk() -> None:
    episode = make_planning_episode(map_size=24, rng=np.random.default_rng(10))
    attacked = apply_environment_attack_to_episode(
        episode,
        {"enabled": True, "type": "env_layer_bias", "layer_name": "hazard", "bias_value": 0.1, "mode": "add"},
        np.random.default_rng(11),
    )
    inflated = apply_risk_inflation_to_episode(attacked, scale=0.5, radius=2)

    assert inflated.true_costmap is attacked.true_costmap
    for name in OBJECTIVE_NAMES:
        assert inflated.costmap.layers[name].shape == attacked.costmap.layers[name].shape
        assert np.isfinite(inflated.costmap.layers[name]).all()
        assert float(inflated.costmap.uncertainty_layers[name].mean()) >= float(
            attacked.costmap.uncertainty_layers[name].mean()
        )


def test_risk_inflated_plan_cost_runs() -> None:
    episode = make_planning_episode(map_size=24, rng=np.random.default_rng(20))
    config = risk_inflated_planner_config(episode, max_lambda=1.2, args=_args())
    cost, result = plan_cost(episode, config, {})

    assert np.isfinite(cost)
    assert result["robust_method"] == "risk_inflated_astar"
    assert result["success"]


def test_belief_sample_cvar_does_not_use_hidden_true_map_for_samples() -> None:
    episode = make_planning_episode(map_size=24, rng=np.random.default_rng(30))
    attacked = replace(episode, true_costmap=episode.costmap)
    sampled = sample_plausible_belief_episode(attacked, np.random.default_rng(32), noise_scale=0.4)

    assert attacked.true_costmap is not None
    assert sampled.true_costmap is None


def test_belief_sample_cvar_plan_cost_selects_candidate() -> None:
    episode = make_planning_episode(map_size=24, rng=np.random.default_rng(40))
    config = belief_sample_cvar_config(max_lambda=1.2, seed=41, args=_args())
    cost, result = plan_cost(episode, config, {})

    assert np.isfinite(cost)
    assert result["robust_method"] == "belief_sample_cvar"
    assert result["selected_candidate_id"]
    assert np.isfinite(float(result["belief_cvar_score"]))
