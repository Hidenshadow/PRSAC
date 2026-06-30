from __future__ import annotations

import numpy as np
import torch
from torch import nn

from algorithms.lrr_policy import LRRPolicy
from algorithms.local_repair import evaluate_local_repair


class CleanPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2))

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.full((obs.shape[0], 2), 0.5, dtype=obs.dtype, device=obs.device)


class CountingContext:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate_counterfactual_action(self, action: np.ndarray) -> dict[str, float]:
        self.calls += 1
        return {"scalar_cost": 10.0}


def test_eval_policy_does_not_call_local_search() -> None:
    policy = LRRPolicy(CleanPolicy(), obs_dim=3, action_dim=2, delta_max=0.1)
    context = CountingContext()
    with torch.no_grad():
        action = policy.predict_action(torch.zeros((1, 3))).cpu().numpy()
    assert action.shape == (1, 2)
    assert context.calls == 0


def test_local_repair_candidate_evaluation_is_train_only_explicit_call() -> None:
    context = CountingContext()
    result = evaluate_local_repair(
        context,
        clean_nominal_cost=8.0,
        current_action=np.array([0.5, 0.5], dtype=np.float32),
        clean_action=np.array([0.5, 0.5], dtype=np.float32),
        action_low=0.0,
        action_high=1.0,
        search_radius=0.05,
        delta_max=0.1,
    )
    assert context.calls == 2 * 2 + 2
    assert len(result.candidates) == 2 * 2 + 2
