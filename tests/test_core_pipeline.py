"""Focused tests for the active PPO robustness workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from envs.attack_wrappers import (
    ObservationAttackWrapper,
    apply_environment_attack_to_episode,
    apply_observation_attack,
)
from envs.costmap_env import MultiObjectiveCostmapEnv
from maps.map_generator import LUNAR_SCENARIOS, generate_costmap
from planners.weighted_astar import weighted_astar
from run import replace_seed_tokens
from run_robustness_workflow import command_for_mode
from utils.cleanrl_policy import (
    CleanRLActorCritic,
    load_cleanrl_agent,
    save_cleanrl_checkpoint,
)
from utils.metrics import (
    OBJECTIVE_NAMES,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    compute_observation,
    make_planning_episode,
    normalize_weights,
    plan_with_weights,
)
from visualize_robustness_results import (
    ensure_columns,
    plot_cost_by_policy_attack,
    plot_forgetting_heatmap,
    plot_indomain_vs_heldout,
    plot_relative_degradation,
    write_markdown_report,
)


class CorePipelineTests(unittest.TestCase):
    def test_lunar_map_generation_and_curriculum_sampling(self) -> None:
        rng = np.random.default_rng(123)
        costmap = generate_costmap(map_size=24, rng=rng, scenario="lunar_rover")

        self.assertIn(costmap.scenario, LUNAR_SCENARIOS)
        self.assertEqual(tuple(costmap.layers.keys()), OBJECTIVE_NAMES)
        self.assertFalse(bool(costmap.obstacle_mask[costmap.start]))
        self.assertFalse(bool(costmap.obstacle_mask[costmap.goal]))
        for name in OBJECTIVE_NAMES:
            self.assertTrue(np.isfinite(costmap.layers[name]).all())
            self.assertTrue(np.isfinite(costmap.uncertainty_layers[name]).all())

        env = MultiObjectiveCostmapEnv(
            map_size=24,
            scenario="lunar_rover",
            observation_mode="basic",
            reward_mode="raw",
            map_sampling_mode="map_seed_pool",
            fixed_map_seed=909,
            map_seed_pool_size=2,
        )
        geometries = {
            (
                (info := env.reset(seed=2000 + index)[1])["scenario"],
                info["start"],
                info["goal"],
            )
            for index in range(10)
        }
        self.assertGreaterEqual(len(geometries), 1)
        self.assertLessEqual(len(geometries), 2)

    def test_weighted_astar_and_planning_result(self) -> None:
        cost_map = np.ones((5, 5), dtype=np.float32)
        obstacle_mask = np.zeros((5, 5), dtype=bool)
        path = weighted_astar(cost_map, obstacle_mask, (0, 0), (4, 4), allow_diagonal=True)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (4, 4))

        episode = make_planning_episode(map_size=24, rng=np.random.default_rng(789))
        result = plan_with_weights(
            episode,
            episode.mission_priority,
            lambda_uncertainty=0.0,
        )
        self.assertTrue(bool(result["success"]))
        self.assertGreater(float(result["scalar_cost"]), 0.0)

    def test_relative_reward_and_observation_shapes(self) -> None:
        env = MultiObjectiveCostmapEnv(
            map_size=24,
            seed=321,
            reward_mode="relative_heuristic",
            reward_scale=10.0,
        )
        env.reset(seed=321)
        action = np.full(env.action_space.shape, 0.5, dtype=np.float32)
        _, reward, terminated, truncated, info = env.step(action)

        baseline = float(info["baseline_scalar_cost"])
        policy_cost = float(info["attacked_scalar_cost"])
        expected = 10.0 * (baseline - policy_cost) / max(abs(baseline), 1e-6)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, expected, places=6)
        self.assertIn("heuristic", info["baseline_costs"])

        episode = make_planning_episode(map_size=24, rng=np.random.default_rng(111))
        basic_obs = compute_observation(episode, map_size=24, observation_mode="basic")
        terrain_obs = compute_observation(episode, map_size=24, observation_mode="terrain")
        self.assertGreater(terrain_obs.shape[0], basic_obs.shape[0])
        self.assertTrue(np.isfinite(terrain_obs).all())

    def test_observation_and_environment_attack_helpers(self) -> None:
        env = MultiObjectiveCostmapEnv(
            map_size=24,
            seed=404,
            reward_mode="raw",
            observation_mode="basic",
        )
        wrapped = ObservationAttackWrapper(
            env,
            {
                "enabled": True,
                "type": "obs_gaussian_noise",
                "noise_std": 0.05,
                "seed": 405,
                "clip_to_observation_space": True,
            },
        )
        obs, _ = wrapped.reset(seed=404)
        self.assertEqual(obs.shape, wrapped.observation_space.shape)
        self.assertGreaterEqual(float(obs.min()), 0.0)
        self.assertLessEqual(float(obs.max()), 1.0)

        base_obs = np.full(8, 0.5, dtype=np.float32)
        biased = apply_observation_attack(
            base_obs,
            {"enabled": True, "type": "obs_bias", "bias_value": 0.1, "bias_indices": [0, 2]},
            np.random.default_rng(1),
        )
        self.assertAlmostEqual(float(biased[0]), 0.6, places=6)
        self.assertAlmostEqual(float(biased[1]), 0.5, places=6)

        episode = make_planning_episode(map_size=24, rng=np.random.default_rng(406))
        attacked_episode = apply_environment_attack_to_episode(
            episode,
            {
                "enabled": True,
                "type": "env_layer_bias",
                "layer_name": "hazard",
                "bias_value": 0.1,
                "mode": "add",
            },
            np.random.default_rng(407),
        )
        self.assertFalse(
            np.allclose(
                episode.costmap.layers["hazard"],
                attacked_episode.costmap.layers["hazard"],
            )
        )
        np.testing.assert_array_equal(
            episode.costmap.obstacle_mask,
            attacked_episode.costmap.obstacle_mask,
        )

    def test_action_conversion_and_seed_rewrite(self) -> None:
        episode = make_planning_episode(map_size=24, rng=np.random.default_rng(456))
        direct = normalize_weights(np.array([0, 1, 2, 3, 4], dtype=np.float32))
        self.assertAlmostEqual(float(direct.sum()), 1.0, places=6)

        neutral_action = np.full(6, 0.5, dtype=np.float32)
        weights = action_to_planning_weights(
            episode,
            neutral_action,
            action_mode="preference_delta",
            action_gain=2.0,
        )
        np.testing.assert_allclose(weights, episode.mission_priority, atol=1e-6)
        self.assertAlmostEqual(action_to_uncertainty_lambda(neutral_action, 2.0), 1.0)

        path = (
            "runs/robustness/nominal_ppo/cleanrl_ppo_costmap_seed0/"
            "best_model.pt"
        )
        rewritten = replace_seed_tokens(path, seed=3)
        self.assertIn("nominal_ppo", rewritten)
        self.assertIn("cleanrl_ppo_costmap_seed3", rewritten)

    def test_active_configs_are_standalone(self) -> None:
        config_names = [
            "ppo_lunar_map_pool_relative_reward.json",
            "robustness_ppo_nominal.json",
            "robustness_ppo_env_attack_finetune.json",
            "robustness_ppo_obs_attack_finetune.json",
            "evaluate_robustness_lunar_in_domain.json",
            "evaluate_robustness_lunar_heldout.json",
        ]
        for name in config_names:
            config = json.loads((Path("configs") / name).read_text(encoding="utf-8"))
            self.assertIn("script", config, name)
            self.assertNotIn("extends", config, name)

    def test_robustness_workflow_reward_attack_switch(self) -> None:
        base_args = json.loads(
            Path("configs/ppo_lunar_map_pool_relative_reward.json").read_text(encoding="utf-8")
        )["args"]
        workflow_args = argparse.Namespace(
            mode="finetune_env_attack",
            checkpoint="runs/robustness/nominal_ppo/checkpoint.pt",
            output_root="runs/robustness_test",
            python="python",
            total_timesteps=None,
            finetune_timesteps=128,
        )

        command, _, _ = command_for_mode(workflow_args, dict(base_args))
        reward_index = command.index("--reward-cost-key")
        self.assertEqual(command[reward_index + 1], "soft_attacked_scalar_cost")
        self.assertIn("--environment-attack-config", command)

        nominal_reward_args = dict(base_args)
        nominal_reward_args["environment_attack"] = {
            "enabled": True,
            "type": "env_zscore_topk",
            "apply_during_training": True,
            "reward_uses_attacked_cost": False,
        }
        command, _, _ = command_for_mode(workflow_args, nominal_reward_args)
        reward_index = command.index("--reward-cost-key")
        self.assertEqual(command[reward_index + 1], "scalar_cost")
        self.assertNotIn("--environment-attack-config", command)

    def test_robustness_visualization_outputs(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "policy_name": "nominal",
                    "eval_domain": "in_domain",
                    "attack_type": "none",
                    "mean_nominal_scalar_cost": 10.0,
                    "mean_attacked_scalar_cost": 10.0,
                    "success_rate": 1.0,
                },
                {
                    "policy_name": "nominal",
                    "eval_domain": "in_domain",
                    "attack_type": "environment",
                    "mean_nominal_scalar_cost": 10.0,
                    "mean_attacked_scalar_cost": 15.0,
                    "success_rate": 0.9,
                },
                {
                    "policy_name": "env_ft",
                    "eval_domain": "in_domain",
                    "attack_type": "environment",
                    "mean_nominal_scalar_cost": 10.0,
                    "mean_attacked_scalar_cost": 12.0,
                    "success_rate": 0.95,
                },
                {
                    "policy_name": "obs_ft",
                    "eval_domain": "in_domain",
                    "attack_type": "observation",
                    "mean_nominal_scalar_cost": 10.0,
                    "mean_attacked_scalar_cost": 11.0,
                    "success_rate": 0.96,
                },
                {
                    "policy_name": "nominal",
                    "eval_domain": "heldout",
                    "attack_type": "environment",
                    "mean_nominal_scalar_cost": 11.0,
                    "mean_attacked_scalar_cost": 17.0,
                    "success_rate": 0.88,
                },
                {
                    "policy_name": "env_ft",
                    "eval_domain": "heldout",
                    "attack_type": "environment",
                    "mean_nominal_scalar_cost": 11.0,
                    "mean_attacked_scalar_cost": 14.0,
                    "success_rate": 0.92,
                },
            ]
        )
        df = ensure_columns(raw)
        self.assertIn("relative_degradation", df.columns)
        self.assertAlmostEqual(
            float(df.loc[df["policy_name"].eq("env_ft") & df["eval_domain"].eq("in_domain"), "relative_degradation"].iloc[0]),
            0.2,
            places=6,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            warnings: list[str] = []
            _, gap_df = plot_indomain_vs_heldout(df, output_dir, warnings)
            plot_cost_by_policy_attack(df, output_dir, warnings)
            plot_relative_degradation(df, output_dir, warnings)
            plot_forgetting_heatmap(df, output_dir, warnings)
            report_path = write_markdown_report(df, output_dir, warnings, gap_df)

            self.assertTrue((output_dir / "fig_cost_by_policy_attack_in_domain.png").exists())
            self.assertTrue((output_dir / "fig_relative_degradation_in_domain.png").exists())
            self.assertTrue((output_dir / "fig_indomain_vs_heldout_environment.png").exists())
            self.assertTrue((output_dir / "fig_forgetting_heatmap_in_domain.png").exists())
            self.assertTrue((output_dir / "generalization_gap.csv").exists())
            self.assertTrue(report_path.exists())

    def test_cleanrl_checkpoint_round_trip(self) -> None:
        agent = CleanRLActorCritic(
            obs_dim=50,
            action_dim=6,
            hidden_size=16,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cleanrl_ppo.pt"
            save_cleanrl_checkpoint(
                path,
                agent,
                {"observation_mode": "terrain", "action_mode": "preference_delta"},
                global_step=1,
                best_mean_reward=-1.0,
            )
            loaded, _ = load_cleanrl_agent(path)

        obs = torch.rand((3, 50), dtype=torch.float32)
        with torch.no_grad():
            action = loaded.get_deterministic_action(obs)

        self.assertEqual(tuple(action.shape), (3, 6))
        self.assertGreaterEqual(float(action.min()), 0.0)
        self.assertLessEqual(float(action.max()), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
