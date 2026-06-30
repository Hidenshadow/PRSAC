from __future__ import annotations

import numpy as np

from algorithms.local_repair import generate_local_repair_candidates


def test_candidate_generation_count_and_contents() -> None:
    current = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    clean = np.array([0.45, 0.55, 0.65], dtype=np.float32)
    candidates = generate_local_repair_candidates(
        current,
        clean,
        action_low=0.0,
        action_high=1.0,
        search_radius=0.05,
    )
    assert len(candidates) == 2 * current.size + 2
    assert candidates[0].candidate_type == "current"
    assert candidates[1].candidate_type == "clean_anchor"
    assert np.allclose(candidates[0].action, current)
    assert np.allclose(candidates[1].action, clean)
    assert np.allclose(candidates[2].action, [0.45, 0.5, 0.6])
    assert np.allclose(candidates[3].action, [0.35, 0.5, 0.6])


def test_candidate_generation_clamps_to_bounds() -> None:
    current = np.array([0.98, 0.02], dtype=np.float32)
    clean = np.array([1.2, -0.2], dtype=np.float32)
    candidates = generate_local_repair_candidates(current, clean, 0.0, 1.0, 0.1)
    stacked = np.stack([candidate.action for candidate in candidates], axis=0)
    assert np.all(stacked >= 0.0)
    assert np.all(stacked <= 1.0)
    assert np.allclose(candidates[1].action, [1.0, 0.0])


def test_adjacent_pairwise_candidate_generation() -> None:
    current = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    clean = np.array([0.45, 0.55, 0.65], dtype=np.float32)
    candidates = generate_local_repair_candidates(
        current,
        clean,
        action_low=0.0,
        action_high=1.0,
        search_radius=0.05,
        pairwise_candidate_mode="adjacent",
    )
    assert len(candidates) == 2 * current.size + 2 + 4 * (current.size - 1)
    assert any(candidate.candidate_type == "pair_pp_dim_0_1" for candidate in candidates)
    assert any(candidate.candidate_type == "pair_mm_dim_1_2" for candidate in candidates)
    stacked = np.stack([candidate.action for candidate in candidates], axis=0)
    assert np.all(stacked >= 0.0)
    assert np.all(stacked <= 1.0)
