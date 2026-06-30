from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from utils.cleanrl_policy import (
    CleanRLActorCritic,
    CleanRLResidualBeliefActorCritic,
    load_cleanrl_agent,
    save_cleanrl_checkpoint,
)
from utils.planner_residual_belief import (
    PlannerResidualFeatureBuilder,
    PlannerResidualFeatureConfig,
    prb_auxiliary_loss,
)


def test_residual_feature_builder_scalar() -> None:
    builder = PlannerResidualFeatureBuilder(PlannerResidualFeatureConfig(action_dim=2))
    batch = builder.build_from_info(
        {
            "belief_scalar_cost": 10.0,
            "attacked_scalar_cost": 12.0,
            "success": True,
            "path_length": 5.0,
        },
        np.array([0.5, 0.5], dtype=np.float32),
        "attacked_scalar_cost",
    )
    assert np.isclose(batch.raw["residual_total_cost"], 2.0)
    assert np.isclose(batch.residual_total_target, 0.2)
    assert np.isclose(batch.features[0], 0.2)


def test_component_residual_and_mask() -> None:
    builder = PlannerResidualFeatureBuilder(PlannerResidualFeatureConfig(action_dim=1))
    batch = builder.build_from_info(
        {
            "belief_scalar_cost": 10.0,
            "attacked_scalar_cost": 10.0,
            "belief_hazard_exposure": 1.0,
            "hazard_exposure": 2.0,
            "belief_uncertainty_exposure": 2.0,
            "uncertainty_exposure": 1.0,
            "belief_communication_exposure": 4.0,
            "communication_exposure": 5.0,
        },
        np.array([0.5], dtype=np.float32),
    )
    assert np.allclose(batch.raw["true_attacked_component_costs"][:3], [2.0, 1.0, 5.0])
    assert np.allclose(batch.raw["planner_predicted_component_costs"][:3], [1.0, 2.0, 4.0])
    assert np.allclose(batch.component_residual_target[:3], [1.0, -0.5, 0.25])
    assert np.allclose(batch.component_mask[:3], [1.0, 1.0, 1.0])
    assert batch.component_mask[3] == 0.0


def test_component_mask_unavailable() -> None:
    builder = PlannerResidualFeatureBuilder(PlannerResidualFeatureConfig(action_dim=1, use_component_costs=True))
    batch = builder.build_from_info(
        {"belief_scalar_cost": 10.0, "attacked_scalar_cost": 12.0},
        np.array([0.5], dtype=np.float32),
    )
    assert not batch.component_cost_available
    assert np.all(batch.component_mask == 0.0)


def test_residual_belief_encoder_output_shape() -> None:
    obs_dim = 5
    action_dim = 3
    feature_dim = 12
    agent = CleanRLResidualBeliefActorCritic(obs_dim, action_dim, prb_feature_dim=feature_dim, prb_latent_dim=7)
    obs = torch.zeros((4, obs_dim), dtype=torch.float32)
    features = torch.zeros((4, feature_dim), dtype=torch.float32)
    latent = agent.encode_residual(obs, features)
    action, logprob, entropy, value = agent.get_action_and_value(obs, residual_features=features)
    assert latent.shape == (4, 7)
    assert action.shape == (4, action_dim)
    assert logprob.shape == (4,)
    assert entropy.shape == (4,)
    assert value.shape == (4,)


def test_prb_initially_matches_base_policy() -> None:
    torch.manual_seed(0)
    base = CleanRLActorCritic(4, 2, hidden_size=16)
    prb = CleanRLResidualBeliefActorCritic(4, 2, hidden_size=16, prb_feature_dim=6, prb_latent_dim=5, prb_hidden_dim=8)
    prb.copy_base_policy_from(base)
    obs = torch.rand((3, 4))
    features = torch.randn((3, 6))
    assert torch.allclose(base.get_deterministic_action(obs), prb.get_deterministic_action(obs, features), atol=1e-6)
    assert torch.allclose(base.get_value(obs), prb.get_value(obs, features), atol=1e-6)


def test_prb_auxiliary_loss_finite() -> None:
    agent = CleanRLResidualBeliefActorCritic(4, 2, hidden_size=16, prb_feature_dim=6, prb_latent_dim=5, prb_hidden_dim=8)
    obs = torch.rand((5, 4))
    features = torch.rand((5, 6))
    actions = torch.rand((5, 2))
    predictions = agent.predict_prb_aux(obs, features, actions)
    targets = {
        "residual_total": torch.rand(5),
        "true_total": torch.rand(5),
        "component_residual": torch.rand(5, 4),
        "true_component": torch.rand(5, 4),
        "component_mask": torch.ones(5, 4),
    }
    loss, parts = prb_auxiliary_loss(predictions, targets, {})
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_prb_checkpoint_round_trip(tmp_path: Path) -> None:
    agent = CleanRLResidualBeliefActorCritic(4, 2, hidden_size=16, prb_feature_dim=6, prb_latent_dim=5, prb_hidden_dim=8)
    path = tmp_path / "prb.pt"
    save_cleanrl_checkpoint(path, agent, {"prb_enabled": True}, global_step=123)
    loaded, checkpoint = load_cleanrl_agent(path)
    assert isinstance(loaded, CleanRLResidualBeliefActorCritic)
    assert checkpoint["policy_class"] == CleanRLResidualBeliefActorCritic.policy_class
    assert loaded.prb_feature_dim == 6
    assert loaded.prb_latent_dim == 5
