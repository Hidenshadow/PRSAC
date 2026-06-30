from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from algorithms.bvr import (
    BVRTrainer,
    BVRTrainerConfig,
    RouteVerifier,
    belief_only_episode,
    extract_route_verifier_features,
    generate_bvr_candidates,
    load_bvr_checkpoint,
    route_verifier_feature_names,
    save_bvr_checkpoint,
    select_bvr_action_from_features,
)


def test_bvr_candidate_generation_adjacent_pairs_and_bounds() -> None:
    clean = np.array([0.02, 0.50, 0.98], dtype=np.float32)
    candidates = generate_bvr_candidates(
        clean,
        action_low=0.0,
        action_high=1.0,
        search_radius=0.1,
        pairwise_candidate_mode="adjacent",
    )
    assert len(candidates) == 1 + 2 * clean.size + 4 * (clean.size - 1)
    assert candidates[0].candidate_type == "clean_anchor"
    assert np.allclose(candidates[0].action, clean)
    assert any(candidate.candidate_type == "pair_pp_dim_0_1" for candidate in candidates)
    stacked = np.stack([candidate.action for candidate in candidates], axis=0)
    assert np.all(stacked >= 0.0)
    assert np.all(stacked <= 1.0)


def test_route_features_use_belief_fields_not_true_route_fields() -> None:
    obs = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    clean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    action = np.array([0.6, 0.4, 0.5], dtype=np.float32)
    result = {
        "success": True,
        "path_length": 12,
        "scalar_cost": 99.0,
        "belief_scalar_cost": 4.0,
        "constraint_penalty": 88.0,
        "belief_constraint_penalty": 0.5,
        "hazard_exposure": 77.0,
        "belief_hazard_exposure": 0.2,
        "belief_illumination_exposure": 0.3,
        "belief_communication_exposure": 0.4,
        "belief_uncertainty_exposure": 0.5,
        "mean_path_confidence": 0.6,
        "weights": np.array([0.2, 0.3, 0.5], dtype=np.float32),
        "mission_priority": np.array([0.4, 0.4, 0.2], dtype=np.float32),
        "lambda_uncertainty": 0.25,
        "belief_objectives": {
            "distance": 0.1,
            "energy": 0.2,
            "hazard": 0.3,
            "communication": 0.4,
            "illumination": 0.5,
        },
        "path_uncertainty": {
            "distance": 0.01,
            "energy": 0.02,
            "hazard": 0.03,
            "communication": 0.04,
            "illumination": 0.05,
        },
        "belief_constraint_metrics": {
            "max_hazard": 0.7,
            "min_communication_quality": 0.8,
            "mean_illumination_quality": 0.9,
        },
    }
    features = extract_route_verifier_features(
        obs,
        clean,
        action,
        result,
        map_size=48,
        max_uncertainty_lambda=1.0,
    )
    names = route_verifier_feature_names(obs_dim=obs.size, action_dim=action.size)
    values = dict(zip(names, features))
    assert len(names) == features.size
    assert "scalar_cost" not in names
    assert "hazard_exposure" not in names
    assert np.isclose(values["belief_scalar_cost_scaled"], 0.4)
    assert np.isclose(values["belief_hazard_exposure"], 0.2)
    assert np.isclose(values["belief_constraint_penalty_scaled"], 0.05)


@dataclass(frozen=True)
class DummyEpisode:
    costmap: object
    true_costmap: object | None = None


def test_belief_only_episode_removes_true_map() -> None:
    episode = DummyEpisode(costmap=object(), true_costmap=object())
    belief_episode = belief_only_episode(episode)
    assert belief_episode.costmap is episode.costmap
    assert belief_episode.true_costmap is None


def test_bvr_trainer_update_and_checkpoint_roundtrip(tmp_path) -> None:
    verifier = RouteVerifier(4, hidden_sizes=(8,), final_init_std=0.0)
    trainer = BVRTrainer(
        verifier,
        BVRTrainerConfig(
            batch_size=2,
            epochs=2,
            learning_rate=1e-2,
            benefit_epsilon=2.0,
            benefit_loss_weight=1.0,
            benefit_positive_weight=2.0,
        ),
    )
    feature_sets = [
        np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32),
        np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0]], dtype=np.float32),
    ]
    score_sets = [
        np.array([90.0, 95.0, 91.0], dtype=np.float32),
        np.array([88.0, 89.0, 96.0], dtype=np.float32),
    ]
    metrics = trainer.update(feature_sets, score_sets)
    assert metrics["bvr/loss"] >= 0.0
    assert metrics["bvr/advantage_loss"] >= 0.0
    assert metrics["bvr/benefit_loss"] >= 0.0
    assert 0.49 <= metrics["bvr/positive_benefit_label_rate"] <= 0.51
    assert metrics["bvr/num_candidate_sets"] == 2

    checkpoint_path = tmp_path / "bvr.pt"
    save_bvr_checkpoint(checkpoint_path, verifier, {"iteration": 3})
    loaded, metadata = load_bvr_checkpoint(checkpoint_path)
    assert metadata["iteration"] == 3
    assert loaded.feature_dim == verifier.feature_dim
    assert loaded.hidden_sizes == verifier.hidden_sizes


def test_bvr_selection_uses_verifier_scores_only() -> None:
    verifier = RouteVerifier(2, hidden_sizes=(), final_init_std=0.0)
    final = verifier.net[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight[:] = torch.tensor([[1.0, -1.0]])
        final.bias.zero_()
    candidates = generate_bvr_candidates(np.array([0.5], dtype=np.float32), 0.0, 1.0, 0.1, "none")
    features = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.5]], dtype=np.float32)
    selection = select_bvr_action_from_features(
        verifier,
        candidates,
        features,
        clean_action=np.array([0.5], dtype=np.float32),
    )
    assert selection.selected_index == 1
    assert selection.selected_candidate_type == "plus_dim_0"


def test_bvr_selection_margin_falls_back_to_clean_anchor() -> None:
    verifier = RouteVerifier(1, hidden_sizes=(), final_init_std=0.0)
    final = verifier.net[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight[:] = torch.tensor([[1.0]])
        final.bias.zero_()
    candidates = generate_bvr_candidates(np.array([0.5], dtype=np.float32), 0.0, 1.0, 0.1, "none")
    low_margin_features = np.array([[0.00], [0.03], [0.01]], dtype=np.float32)
    high_margin_features = np.array([[0.00], [0.03], [0.20]], dtype=np.float32)

    conservative = select_bvr_action_from_features(
        verifier,
        candidates,
        low_margin_features,
        clean_action=np.array([0.5], dtype=np.float32),
        selection_margin=0.05,
    )
    assert conservative.selected_index == 0
    assert conservative.selected_candidate_type == "clean_anchor"

    confident = select_bvr_action_from_features(
        verifier,
        candidates,
        high_margin_features,
        clean_action=np.array([0.5], dtype=np.float32),
        selection_margin=0.05,
    )
    assert confident.selected_index == 2
    assert confident.selected_candidate_type == "minus_dim_0"


def test_bvr_belief_safety_penalizes_worse_belief_route() -> None:
    verifier = RouteVerifier(39, hidden_sizes=(), final_init_std=0.0)
    final = verifier.net[-1]
    assert isinstance(final, torch.nn.Linear)
    with torch.no_grad():
        final.weight.zero_()
        final.weight[0, 0] = 1.0
        final.bias.zero_()
    candidates = generate_bvr_candidates(np.array([0.5], dtype=np.float32), 0.0, 1.0, 0.1, "none")
    features = np.zeros((3, 39), dtype=np.float32)
    features[:, 0] = np.array([0.0, 1.0, 0.8], dtype=np.float32)
    scalar_start = 3 * 1 + 2 * 5 + 1
    features[:, scalar_start] = 1.0
    features[:, scalar_start + 2] = np.array([0.1, 2.0, 0.1], dtype=np.float32)
    features[:, scalar_start + 3] = np.array([0.1, 2.0, 0.1], dtype=np.float32)

    unsafe = select_bvr_action_from_features(
        verifier,
        candidates,
        features,
        clean_action=np.array([0.5], dtype=np.float32),
        selection_margin=0.0,
        belief_safety_penalty=0.0,
    )
    assert unsafe.selected_index == 1

    safe = select_bvr_action_from_features(
        verifier,
        candidates,
        features,
        clean_action=np.array([0.5], dtype=np.float32),
        selection_margin=0.0,
        belief_safety_penalty=1.0,
        belief_cost_margin=0.02,
        belief_constraint_margin=0.02,
    )
    assert safe.selected_index == 2
