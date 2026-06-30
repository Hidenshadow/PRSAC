"""Gymnasium environment for RL-selected objective weights.

Each episode is a one-step contextual bandit:
reset produces map statistics and mission priorities, step receives objective
preference actions plus uncertainty sensitivity, A* plans, and the environment
returns a robust path reward.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from utils.metrics import (
    ACTION_DIM,
    DEFAULT_ATTACK_BUDGET_FRACTION,
    DEFAULT_ATTACK_STRENGTH,
    DEFAULT_ATTACKER_RESPONSE,
    DEFAULT_ATTACKER_SHARPNESS,
    DEFAULT_ATTACKER_TEMPERATURE,
    DEFAULT_ATTACKER_TOP_FRACTION,
    DEFAULT_FIXED_MAP_SEED,
    DEFAULT_MAP_SEED_POOL_SIZE,
    DEFAULT_MAX_UNCERTAINTY_LAMBDA,
    MAP_SAMPLING_MODES,
    PlanningEpisode,
    REWARD_MODES,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    compute_observation,
    make_curriculum_planning_episode,
    make_planning_episode,
    plan_with_weights,
)


REWARD_COST_KEYS = ("scalar_cost", "attacked_scalar_cost", "soft_attacked_scalar_cost")


class MultiObjectiveCostmapEnv(gym.Env):
    """RL environment where actions are planning objective weights."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        map_size: int = 64,
        seed: int | None = None,
        allow_diagonal: bool = True,
        obstacle_threshold: float = 0.88,
        min_start_goal_distance_ratio: float = 0.55,
        scenario: str = "nominal",
        observation_mode: str = "terrain",
        reward_mode: str = "relative_heuristic",
        reward_scale: float = 10.0,
        reward_cost_key: str = "attacked_scalar_cost",
        action_mode: str = "preference_delta",
        action_gain: float = 2.0,
        max_uncertainty_lambda: float = DEFAULT_MAX_UNCERTAINTY_LAMBDA,
        attack_budget_fraction: float = DEFAULT_ATTACK_BUDGET_FRACTION,
        attack_strength: float = DEFAULT_ATTACK_STRENGTH,
        attacker_temperature: float = DEFAULT_ATTACKER_TEMPERATURE,
        attacker_response: str = DEFAULT_ATTACKER_RESPONSE,
        attacker_top_fraction: float = DEFAULT_ATTACKER_TOP_FRACTION,
        attacker_sharpness: float = DEFAULT_ATTACKER_SHARPNESS,
        map_sampling_mode: str = "random",
        fixed_map_seed: int = DEFAULT_FIXED_MAP_SEED,
        map_seed_pool_size: int = DEFAULT_MAP_SEED_POOL_SIZE,
    ) -> None:
        super().__init__()
        self.map_size = int(map_size)
        self.allow_diagonal = bool(allow_diagonal)
        self.obstacle_threshold = float(obstacle_threshold)
        self.min_start_goal_distance_ratio = float(min_start_goal_distance_ratio)
        self.scenario = scenario
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.reward_scale = float(reward_scale)
        self.reward_cost_key = reward_cost_key
        self.action_mode = action_mode
        self.action_gain = float(action_gain)
        self.max_uncertainty_lambda = float(max_uncertainty_lambda)
        self.attack_budget_fraction = float(attack_budget_fraction)
        self.attack_strength = float(attack_strength)
        self.attacker_temperature = float(attacker_temperature)
        self.attacker_response = attacker_response
        self.attacker_top_fraction = float(attacker_top_fraction)
        self.attacker_sharpness = float(attacker_sharpness)
        self.map_sampling_mode = map_sampling_mode
        self.fixed_map_seed = int(fixed_map_seed)
        self.map_seed_pool_size = max(int(map_seed_pool_size), 1)
        self._initial_seed = seed

        if self.map_sampling_mode not in MAP_SAMPLING_MODES:
            raise ValueError(f"map_sampling_mode must be one of {MAP_SAMPLING_MODES}")

        if self.reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")

        if self.reward_cost_key not in REWARD_COST_KEYS:
            raise ValueError(f"reward_cost_key must be one of {REWARD_COST_KEYS}")

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)

        # Keep the declared shape tied to the actual observation builder.
        dummy_rng = np.random.default_rng(0)
        dummy_episode = make_planning_episode(
            map_size=self.map_size,
            rng=dummy_rng,
            obstacle_threshold=self.obstacle_threshold,
            min_start_goal_distance_ratio=self.min_start_goal_distance_ratio,
            allow_diagonal=self.allow_diagonal,
            scenario=self.scenario,
        )
        obs_dim = int(
            compute_observation(
                dummy_episode,
                self.map_size,
                observation_mode=self.observation_mode,
                max_uncertainty_lambda=self.max_uncertainty_lambda,
            ).shape[0]
        )
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._episode: PlanningEpisode | None = None
        self._obs: np.ndarray | None = None
        self._map_cache = {}
        self._done = False

        if seed is not None:
            self.reset(seed=seed)

    @property
    def current_episode(self) -> PlanningEpisode:
        """Return the active planning problem for planner-in-the-loop algorithms."""

        if self._episode is None:
            raise RuntimeError("reset must be called before accessing current_episode")
        return self._episode

    def evaluate_counterfactual_action(self, action: np.ndarray) -> dict[str, Any]:
        """Evaluate a planner action on the current episode without stepping."""

        from utils.planner_regret import evaluate_counterfactual_action_for_env

        return evaluate_counterfactual_action_for_env(
            self,
            action,
            reward_cost_key=self.reward_cost_key,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        options = options or {}
        episode = options.get("episode") or options.get("problem")
        if episode is None:
            episode = make_curriculum_planning_episode(
                map_size=self.map_size,
                rng=self.np_random,
                obstacle_threshold=self.obstacle_threshold,
                min_start_goal_distance_ratio=self.min_start_goal_distance_ratio,
                allow_diagonal=self.allow_diagonal,
                scenario=self.scenario,
                map_sampling_mode=self.map_sampling_mode,
                fixed_map_seed=self.fixed_map_seed,
                map_seed_pool_size=self.map_seed_pool_size,
                map_cache=self._map_cache,
            )
        if not isinstance(episode, PlanningEpisode):
            raise TypeError("reset options must pass a PlanningEpisode under 'episode' or 'problem'")

        self._episode = episode
        self._obs = compute_observation(
            episode,
            self.map_size,
            observation_mode=self.observation_mode,
            max_uncertainty_lambda=self.max_uncertainty_lambda,
        )
        self._done = False

        info = {
            "start": episode.costmap.start,
            "goal": episode.costmap.goal,
            "mission_priority": episode.mission_priority.copy(),
            "rover_state": dict(episode.rover_state),
            "scenario": episode.scenario,
            "mission_regime": episode.mission_regime,
            "mission_severity": float(episode.mission_severity),
            "map_sampling_mode": self.map_sampling_mode,
            "fixed_map_seed": self.fixed_map_seed,
            "map_seed_pool_size": self.map_seed_pool_size,
        }
        return self._obs.copy(), info

    def _reward_cost_source(self) -> str:
        if self.reward_cost_key == "scalar_cost":
            if self._episode is not None and self._episode.true_costmap is not None:
                return "true_after_belief_mismatch"
            if self._episode is not None and getattr(self._episode.costmap, "attack_metadata", None):
                return "attacked"
            return "nominal"
        return "attacked"

    def _baseline_results(self) -> dict[str, dict[str, Any]]:
        if self._episode is None:
            raise RuntimeError("reset must be called before computing baselines")

        baselines: dict[str, dict[str, Any]]
        if self.reward_mode in {"advantage_heuristic", "relative_heuristic"}:
            baselines = {
                "heuristic": {
                    "weights": self._episode.mission_priority.astype(np.float32),
                    "lambda_uncertainty": 0.0,
                }
            }
        else:
            baselines = {}

        results = {}
        for name, config in baselines.items():
            results[name] = plan_with_weights(
                self._episode,
                config["weights"],
                lambda_uncertainty=float(config["lambda_uncertainty"]),
                allow_diagonal=self.allow_diagonal,
                attack_budget_fraction=self.attack_budget_fraction,
                attack_strength=self.attack_strength,
                attacker_temperature=self.attacker_temperature,
                attacker_response=self.attacker_response,
                attacker_top_fraction=self.attacker_top_fraction,
                attacker_sharpness=self.attacker_sharpness,
            )
        return results

    def _baseline_scalar_costs(self) -> dict[str, float]:
        return {
            name: self._result_cost(result)
            for name, result in self._baseline_results().items()
        }
        return costs

    def _result_cost(self, result: dict[str, Any]) -> float:
        return float(result.get(self.reward_cost_key, result.get("attacked_scalar_cost", result["scalar_cost"])))

    def _training_reward(self, result: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        raw_reward = float(result["reward"])
        if self.reward_mode == "raw":
            return raw_reward, {}

        if not bool(result["success"]):
            return -10.0, {"baseline_scalar_cost": np.nan, "baseline_costs": {}}

        baseline_results = self._baseline_results()
        if not baseline_results:
            return raw_reward, {}

        baseline_name, baseline_result = next(iter(baseline_results.items()))
        baseline_costs = {
            name: self._result_cost(baseline_result_item)
            for name, baseline_result_item in baseline_results.items()
        }
        baseline_scalar_cost = baseline_costs[baseline_name]

        policy_cost = self._result_cost(result)
        relative_reward = baseline_scalar_cost - policy_cost
        if self.reward_mode == "relative_heuristic":
            relative_reward = relative_reward / max(abs(baseline_scalar_cost), 1e-6)
        heuristic_attacked_cost = float(
            baseline_result.get(
                self.reward_cost_key,
                baseline_result.get("attacked_scalar_cost", baseline_result["scalar_cost"]),
            )
        )
        policy_attacked_cost = float(
            result.get(
                self.reward_cost_key,
                result.get("attacked_scalar_cost", result["scalar_cost"]),
            )
        )
        return self.reward_scale * relative_reward, {
            "baseline_scalar_cost": baseline_scalar_cost,
            "baseline_attacked_cost": baseline_scalar_cost,
            "baseline_costs": baseline_costs,
            "heuristic_nominal_cost": float(baseline_result.get("scalar_cost", np.nan)),
            "policy_nominal_cost": float(result.get("scalar_cost", np.nan)),
            "heuristic_attacked_cost": heuristic_attacked_cost,
            "policy_attacked_cost": policy_attacked_cost,
            "reward_cost_source": self._reward_cost_source(),
            "reward_selected_baseline_cost": float(baseline_scalar_cost),
            "reward_selected_policy_cost": float(policy_cost),
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._episode is None or self._obs is None:
            raise RuntimeError("reset must be called before step")

        weights = action_to_planning_weights(
            self._episode,
            action,
            action_mode=self.action_mode,
            action_gain=self.action_gain,
        )
        lambda_uncertainty = action_to_uncertainty_lambda(
            action,
            max_uncertainty_lambda=self.max_uncertainty_lambda,
        )
        result = plan_with_weights(
            self._episode,
            weights,
            lambda_uncertainty=lambda_uncertainty,
            allow_diagonal=self.allow_diagonal,
            attack_budget_fraction=self.attack_budget_fraction,
            attack_strength=self.attack_strength,
            attacker_temperature=self.attacker_temperature,
            attacker_response=self.attacker_response,
            attacker_top_fraction=self.attacker_top_fraction,
            attacker_sharpness=self.attacker_sharpness,
        )
        reward, reward_info = self._training_reward(result)

        self._done = True
        terminated = True
        truncated = False

        info: dict[str, Any] = {
            "success": result["success"],
            "objectives": result["objectives"],
            "weights": result["weights"],
            "mission_priority": self._episode.mission_priority.copy(),
            "rover_state": dict(self._episode.rover_state),
            "scenario": self._episode.scenario,
            "mission_regime": self._episode.mission_regime,
            "mission_severity": float(self._episode.mission_severity),
            "path_length": result["path_length"],
            "path": result["path"],
            "attacked_cell_exposure": result.get("attacked_cell_exposure", 0),
            "attacked_cell_exposure_ratio": result.get("attacked_cell_exposure_ratio", 0.0),
            "attacked_corridor_cells": result.get("attacked_corridor_cells", 0),
            "hazard_exposure": result.get("hazard_exposure", np.nan),
            "belief_hazard_exposure": result.get("belief_hazard_exposure", np.nan),
            "uncertainty_exposure": result.get("uncertainty_exposure", np.nan),
            "belief_uncertainty_exposure": result.get("belief_uncertainty_exposure", np.nan),
            "illumination_exposure": result.get("illumination_exposure", np.nan),
            "belief_illumination_exposure": result.get("belief_illumination_exposure", np.nan),
            "communication_exposure": result.get("communication_exposure", np.nan),
            "belief_communication_exposure": result.get("belief_communication_exposure", np.nan),
            "scalar_cost": result["scalar_cost"],
            "belief_scalar_cost": result.get("belief_scalar_cost", result["scalar_cost"]),
            "map_mismatch_penalty": result.get("map_mismatch_penalty", 0.0),
            "map_mismatch_abs_error": result.get("map_mismatch_abs_error", 0.0),
            "mean_path_confidence": result.get("mean_path_confidence", np.nan),
            "true_belief_mismatch": result.get("true_belief_mismatch", False),
            "attacked_scalar_cost": result.get("attacked_scalar_cost", result["scalar_cost"]),
            "attack_penalty": result.get("attack_penalty", 0.0),
            "path_uncertainty": result.get("path_uncertainty", {}),
            "attacked_objectives": result.get("attacked_objectives", {}),
            "soft_attacked_scalar_cost": result.get("soft_attacked_scalar_cost", result["scalar_cost"]),
            "soft_attack_penalty": result.get("soft_attack_penalty", 0.0),
            "soft_path_uncertainty": result.get("soft_path_uncertainty", {}),
            "soft_attacker_entropy": result.get("soft_attacker_entropy", 0.0),
            "soft_attacker_peak_probability": result.get("soft_attacker_peak_probability", 1.0),
            "soft_attacker_expected_impact": result.get("soft_attacker_expected_impact", 0.0),
            "soft_attacker_response_mode": result.get("soft_attacker_response_mode", self.attacker_response),
            "soft_attacker_top_fraction": result.get("soft_attacker_top_fraction", self.attacker_top_fraction),
            "soft_attacker_sharpness": result.get("soft_attacker_sharpness", self.attacker_sharpness),
            "lambda_uncertainty": result.get("lambda_uncertainty", lambda_uncertainty),
            "constraint_penalty": result.get("constraint_penalty", 0.0),
            "constraint_metrics": result.get("constraint_metrics", {}),
            "constraint_violations": result.get("constraint_violations", {}),
            "raw_reward": float(result["reward"]),
            "returned_reward": float(reward),
            "reward_mode": self.reward_mode,
            "reward_cost_key": self.reward_cost_key,
            "reward_cost_source": self._reward_cost_source(),
            "action_mode": self.action_mode,
            "max_uncertainty_lambda": self.max_uncertainty_lambda,
            "attack_budget_fraction": self.attack_budget_fraction,
            "attack_strength": self.attack_strength,
            "attacker_temperature": self.attacker_temperature,
            "attacker_response": self.attacker_response,
            "attacker_top_fraction": self.attacker_top_fraction,
            "attacker_sharpness": self.attacker_sharpness,
            "map_sampling_mode": self.map_sampling_mode,
            "fixed_map_seed": self.fixed_map_seed,
            "map_seed_pool_size": self.map_seed_pool_size,
        }
        attack_metadata = getattr(self._episode.costmap, "attack_metadata", None) or {}
        info.update(attack_metadata)
        info.update(reward_info)
        return self._obs.copy(), float(reward), terminated, truncated, info

    def evaluate_weights(self, weights: np.ndarray) -> dict[str, Any]:
        """Evaluate weights on the current episode without changing state."""

        if self._episode is None:
            raise RuntimeError("reset must be called before evaluate_weights")
        return plan_with_weights(
            self._episode,
            weights,
            allow_diagonal=self.allow_diagonal,
            attack_budget_fraction=self.attack_budget_fraction,
            attack_strength=self.attack_strength,
            attacker_temperature=self.attacker_temperature,
            attacker_response=self.attacker_response,
            attacker_top_fraction=self.attacker_top_fraction,
            attacker_sharpness=self.attacker_sharpness,
        )

    def evaluate_action(self, action: np.ndarray) -> dict[str, Any]:
        """Evaluate an RL action after converting it to planner weights."""

        if self._episode is None:
            raise RuntimeError("reset must be called before evaluate_action")
        weights = action_to_planning_weights(
            self._episode,
            action,
            action_mode=self.action_mode,
            action_gain=self.action_gain,
        )
        lambda_uncertainty = action_to_uncertainty_lambda(
            action,
            max_uncertainty_lambda=self.max_uncertainty_lambda,
        )
        return plan_with_weights(
            self._episode,
            weights,
            lambda_uncertainty=lambda_uncertainty,
            allow_diagonal=self.allow_diagonal,
            attack_budget_fraction=self.attack_budget_fraction,
            attack_strength=self.attack_strength,
            attacker_temperature=self.attacker_temperature,
            attacker_response=self.attacker_response,
            attacker_top_fraction=self.attacker_top_fraction,
            attacker_sharpness=self.attacker_sharpness,
        )
