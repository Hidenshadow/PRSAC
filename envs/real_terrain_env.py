"""Gymnasium environment for real DEM/DTM rover benchmark tiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from maps.real_terrain import (
    load_real_layers,
    load_task_split,
    make_real_planning_episode,
)
from utils.metrics import (
    ACTION_DIM,
    DEFAULT_ATTACK_BUDGET_FRACTION,
    DEFAULT_ATTACK_STRENGTH,
    DEFAULT_ATTACKER_RESPONSE,
    DEFAULT_ATTACKER_SHARPNESS,
    DEFAULT_ATTACKER_TEMPERATURE,
    DEFAULT_ATTACKER_TOP_FRACTION,
    DEFAULT_MAX_UNCERTAINTY_LAMBDA,
    OBJECTIVE_NAMES,
    PlanningEpisode,
    REWARD_MODES,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    compute_observation,
    plan_with_weights,
)


REWARD_COST_KEYS = ("scalar_cost", "attacked_scalar_cost", "soft_attacked_scalar_cost")


class RealTerrainPlanningEnv(gym.Env):
    """One-step planner-parameter environment backed by real terrain layers.

    This is intentionally parallel to ``MultiObjectiveCostmapEnv`` but samples
    from deterministic DEM/DTM start-goal tasks instead of synthetic maps.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        layers_path: str | Path,
        tasks: list[dict[str, Any]] | None = None,
        task_split_path: str | Path | None = None,
        seed: int | None = None,
        scenario: str = "real_lunar_viper",
        mission_profile_scenario: str | None = None,
        observation_mode: str = "terrain",
        reward_mode: str = "relative_heuristic",
        reward_scale: float = 10.0,
        reward_cost_key: str = "attacked_scalar_cost",
        action_mode: str = "preference_delta",
        action_gain: float = 3.0,
        max_uncertainty_lambda: float = DEFAULT_MAX_UNCERTAINTY_LAMBDA,
        allow_diagonal: bool = True,
        attack_budget_fraction: float = DEFAULT_ATTACK_BUDGET_FRACTION,
        attack_strength: float = DEFAULT_ATTACK_STRENGTH,
        attacker_temperature: float = DEFAULT_ATTACKER_TEMPERATURE,
        attacker_response: str = DEFAULT_ATTACKER_RESPONSE,
        attacker_top_fraction: float = DEFAULT_ATTACKER_TOP_FRACTION,
        attacker_sharpness: float = DEFAULT_ATTACKER_SHARPNESS,
    ) -> None:
        super().__init__()
        self.layers_path = Path(layers_path)
        self.raw_layers = load_real_layers(self.layers_path)
        if tasks is None and task_split_path is not None:
            tasks = load_task_split(task_split_path)
        self.tasks = [dict(task) for task in (tasks or [])]
        if not self.tasks:
            raise ValueError("RealTerrainPlanningEnv requires at least one task")

        self.scenario = scenario
        self.mission_profile_scenario = mission_profile_scenario
        self.observation_mode = observation_mode
        self.reward_mode = reward_mode
        self.reward_scale = float(reward_scale)
        self.reward_cost_key = reward_cost_key
        self.action_mode = action_mode
        self.action_gain = float(action_gain)
        self.max_uncertainty_lambda = float(max_uncertainty_lambda)
        self.allow_diagonal = bool(allow_diagonal)
        self.attack_budget_fraction = float(attack_budget_fraction)
        self.attack_strength = float(attack_strength)
        self.attacker_temperature = float(attacker_temperature)
        self.attacker_response = attacker_response
        self.attacker_top_fraction = float(attacker_top_fraction)
        self.attacker_sharpness = float(attacker_sharpness)
        self._initial_seed = seed

        if self.reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        if self.reward_cost_key not in REWARD_COST_KEYS:
            raise ValueError(f"reward_cost_key must be one of {REWARD_COST_KEYS}")

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        dummy_rng = np.random.default_rng(0)
        dummy_episode = make_real_planning_episode(
            self.raw_layers,
            self.tasks[0],
            dummy_rng,
            scenario=self.scenario,
            mission_profile_scenario=self.mission_profile_scenario,
        )
        self.map_size = int(dummy_episode.costmap.layers["distance"].shape[0])
        obs_dim = int(
            compute_observation(
                dummy_episode,
                self.map_size,
                observation_mode=self.observation_mode,
                max_uncertainty_lambda=self.max_uncertainty_lambda,
            ).shape[0]
        )
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self._episode: PlanningEpisode | None = None
        self._task: dict[str, Any] | None = None
        self._obs: np.ndarray | None = None
        self._done = False

        if seed is not None:
            self.reset(seed=seed)

    @property
    def current_episode(self) -> PlanningEpisode:
        if self._episode is None:
            raise RuntimeError("reset must be called before accessing current_episode")
        return self._episode

    def evaluate_counterfactual_action(self, action: np.ndarray) -> dict[str, Any]:
        """Evaluate a planner action on the current real-terrain episode without stepping."""

        from utils.planner_regret import evaluate_counterfactual_action_for_env

        return evaluate_counterfactual_action_for_env(
            self,
            action,
            reward_cost_key=self.reward_cost_key,
        )

    @property
    def current_task(self) -> dict[str, Any]:
        if self._task is None:
            return {}
        return dict(self._task)

    def get_observation_metadata(self) -> list[str]:
        """Return stable names for compact observation dimensions."""

        return [f"obs_{index}" for index in range(int(self.observation_space.shape[0]))]

    def _sample_task(self) -> dict[str, Any]:
        index = int(self.np_random.integers(0, len(self.tasks)))
        return dict(self.tasks[index])

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        episode = options.get("episode") or options.get("problem")
        task = options.get("task")
        if episode is None:
            task = dict(task or self._sample_task())
            task_seed = int(task.get("seed", 0))
            rng = np.random.default_rng(task_seed)
            episode = make_real_planning_episode(
                self.raw_layers,
                task,
                rng,
                scenario=self.scenario,
                mission_profile_scenario=self.mission_profile_scenario,
            )
        elif not isinstance(episode, PlanningEpisode):
            raise TypeError("reset options must pass a PlanningEpisode under 'episode' or 'problem'")

        self._episode = episode
        self._task = dict(task or getattr(self, "_task", {}) or {})
        self._obs = compute_observation(
            episode,
            self.map_size,
            observation_mode=self.observation_mode,
            max_uncertainty_lambda=self.max_uncertainty_lambda,
        )
        self._done = False
        attack_metadata = getattr(episode.costmap, "attack_metadata", None) or {}
        info = {
            "start": episode.costmap.start,
            "goal": episode.costmap.goal,
            "mission_priority": episode.mission_priority.copy(),
            "rover_state": dict(episode.rover_state),
            "scenario": episode.scenario,
            "mission_regime": episode.mission_regime,
            "mission_severity": float(episode.mission_severity),
            "task_id": self._task.get("task_id", ""),
            "split": self._task.get("split", ""),
            "tile_id": self._task.get("tile_id", ""),
            "layers_path": str(self.layers_path),
        }
        info.update(attack_metadata)
        return self._obs.copy(), info

    def _reward_cost_source(self) -> str:
        if self.reward_cost_key == "scalar_cost":
            if self._episode is not None and self._episode.true_costmap is not None:
                return "true_after_belief_mismatch"
            if self._episode is not None and getattr(self._episode.costmap, "attack_metadata", None):
                return "attacked"
            return "nominal"
        return "attacked"

    def _result_cost(self, result: dict[str, Any]) -> float:
        return float(result.get(self.reward_cost_key, result.get("attacked_scalar_cost", result["scalar_cost"])))

    def _baseline_results(self) -> dict[str, dict[str, Any]]:
        if self._episode is None:
            raise RuntimeError("reset must be called before computing baselines")
        if self.reward_mode not in {"advantage_heuristic", "relative_heuristic"}:
            return {}
        return {
            "heuristic": plan_with_weights(
                self._episode,
                self._episode.mission_priority.astype(np.float32),
                lambda_uncertainty=0.0,
                allow_diagonal=self.allow_diagonal,
                attack_budget_fraction=self.attack_budget_fraction,
                attack_strength=self.attack_strength,
                attacker_temperature=self.attacker_temperature,
                attacker_response=self.attacker_response,
                attacker_top_fraction=self.attacker_top_fraction,
                attacker_sharpness=self.attacker_sharpness,
            )
        }

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
        baseline_scalar_cost = self._result_cost(baseline_result)
        policy_cost = self._result_cost(result)
        relative_reward = baseline_scalar_cost - policy_cost
        if self.reward_mode == "relative_heuristic":
            relative_reward = relative_reward / max(abs(baseline_scalar_cost), 1e-6)
        return self.reward_scale * relative_reward, {
            "baseline_scalar_cost": float(baseline_scalar_cost),
            "baseline_costs": {baseline_name: float(baseline_scalar_cost)},
            "heuristic_nominal_cost": float(baseline_result.get("scalar_cost", np.nan)),
            "policy_nominal_cost": float(result.get("scalar_cost", np.nan)),
            "heuristic_attacked_cost": float(
                baseline_result.get(self.reward_cost_key, baseline_result.get("attacked_scalar_cost", np.nan))
            ),
            "policy_attacked_cost": float(
                result.get(self.reward_cost_key, result.get("attacked_scalar_cost", np.nan))
            ),
            "reward_cost_source": self._reward_cost_source(),
            "reward_selected_baseline_cost": float(baseline_scalar_cost),
            "reward_selected_policy_cost": float(policy_cost),
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
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
        info: dict[str, Any] = {
            "success": result["success"],
            "objectives": result["objectives"],
            "weights": result["weights"],
            "mission_priority": self._episode.mission_priority.copy(),
            "rover_state": dict(self._episode.rover_state),
            "scenario": self._episode.scenario,
            "mission_regime": self._episode.mission_regime,
            "mission_severity": float(self._episode.mission_severity),
            "task_id": self._task.get("task_id", "") if self._task else "",
            "split": self._task.get("split", "") if self._task else "",
            "tile_id": self._task.get("tile_id", "") if self._task else "",
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
            "attacked_objectives": result.get("attacked_objectives", {}),
            "soft_attacked_scalar_cost": result.get("soft_attacked_scalar_cost", result["scalar_cost"]),
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
        }
        attack_metadata = getattr(self._episode.costmap, "attack_metadata", None) or {}
        info.update(attack_metadata)
        info.update(reward_info)
        return self._obs.copy(), float(reward), True, False, info

    def evaluate_weights(self, weights: np.ndarray) -> dict[str, Any]:
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
