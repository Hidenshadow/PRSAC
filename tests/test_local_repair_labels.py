from __future__ import annotations

import numpy as np

from algorithms.local_repair import CandidateAction, build_repair_label


def _candidates() -> list[CandidateAction]:
    return [
        CandidateAction(np.array([0.50, 0.50], dtype=np.float32), "current"),
        CandidateAction(np.array([0.45, 0.45], dtype=np.float32), "clean_anchor"),
        CandidateAction(np.array([0.70, 0.50], dtype=np.float32), "plus_dim_0", 0),
        CandidateAction(np.array([0.30, 0.50], dtype=np.float32), "minus_dim_0", 0),
    ]


def test_repair_label_skips_small_improvement() -> None:
    label = build_repair_label(
        clean_action=np.array([0.45, 0.45], dtype=np.float32),
        current_action=np.array([0.50, 0.50], dtype=np.float32),
        candidates=_candidates(),
        candidate_scores=np.array([90.0, 90.1, 90.2, 89.0]),
        delta_max=0.1,
        improvement_epsilon=0.25,
    )
    assert label is None


def test_repair_label_uses_best_action_and_clips_residual() -> None:
    label = build_repair_label(
        clean_action=np.array([0.45, 0.45], dtype=np.float32),
        current_action=np.array([0.50, 0.50], dtype=np.float32),
        candidates=_candidates(),
        candidate_scores=np.array([90.0, 88.0, 96.0, 89.0]),
        delta_max=np.array([0.1, 0.2], dtype=np.float32),
        improvement_epsilon=0.25,
        beta=2.0,
        w_max=2.5,
    )
    assert label is not None
    assert label.chosen_candidate_type == "plus_dim_0"
    assert np.allclose(label.target_residual, [0.1, 0.05])
    assert np.isclose(label.improvement, 6.0)
    assert np.isclose(label.weight, 2.5)


def test_repair_label_can_blend_target_from_current_residual() -> None:
    label = build_repair_label(
        clean_action=np.array([0.45, 0.45], dtype=np.float32),
        current_action=np.array([0.50, 0.50], dtype=np.float32),
        candidates=_candidates(),
        candidate_scores=np.array([90.0, 88.0, 96.0, 89.0]),
        delta_max=np.array([0.3, 0.3], dtype=np.float32),
        improvement_epsilon=0.25,
        target_blend=0.5,
    )
    assert label is not None
    assert np.allclose(label.current_residual, [0.05, 0.05])
    assert np.allclose(label.target_residual, [0.15, 0.05])


def test_repair_label_can_clip_target_residual_norm() -> None:
    label = build_repair_label(
        clean_action=np.array([0.45, 0.45], dtype=np.float32),
        current_action=np.array([0.50, 0.50], dtype=np.float32),
        candidates=_candidates(),
        candidate_scores=np.array([90.0, 88.0, 96.0, 89.0]),
        delta_max=np.array([0.3, 0.3], dtype=np.float32),
        improvement_epsilon=0.25,
        target_residual_norm_clip=0.1,
    )
    assert label is not None
    assert np.linalg.norm(label.target_residual) <= 0.100001


def test_repair_label_soft_target_averages_improved_candidates() -> None:
    label = build_repair_label(
        clean_action=np.array([0.45, 0.45], dtype=np.float32),
        current_action=np.array([0.50, 0.50], dtype=np.float32),
        candidates=_candidates(),
        candidate_scores=np.array([90.0, 96.0, 96.0, 89.0]),
        delta_max=np.array([0.3, 0.3], dtype=np.float32),
        improvement_epsilon=0.25,
        target_mode="soft",
        soft_target_temperature=1.0,
    )
    assert label is not None
    assert label.chosen_candidate_type == "soft_improvement_2"
    assert np.allclose(label.target_residual, [0.125, 0.025])


def test_repair_label_surface_target_can_choose_continuous_local_optimum() -> None:
    candidates = [
        CandidateAction(np.array([0.50], dtype=np.float32), "current"),
        CandidateAction(np.array([0.50], dtype=np.float32), "clean_anchor"),
        CandidateAction(np.array([0.60], dtype=np.float32), "plus_dim_0", 0),
        CandidateAction(np.array([0.40], dtype=np.float32), "minus_dim_0", 0),
    ]
    label = build_repair_label(
        clean_action=np.array([0.50], dtype=np.float32),
        current_action=np.array([0.50], dtype=np.float32),
        candidates=candidates,
        candidate_scores=np.array([90.0, 90.0, 90.2, 87.8]),
        delta_max=np.array([0.3], dtype=np.float32),
        improvement_epsilon=0.1,
        target_mode="surface",
        surface_ridge=0.0,
    )
    assert label is not None
    assert label.chosen_candidate_type == "surface_axis_0"
    assert np.allclose(label.target_residual, [0.06], atol=1e-5)
