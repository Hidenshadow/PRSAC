from __future__ import annotations

import torch
from torch import nn

from algorithms.lrr_policy import LRRPolicy


class ConstantCleanPolicy(nn.Module):
    def __init__(self, action: torch.Tensor) -> None:
        super().__init__()
        self.bias = nn.Parameter(action.clone())

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        return self.bias.unsqueeze(0).expand(obs.shape[0], -1)


def test_lrr_initially_matches_clean_policy() -> None:
    clean = ConstantCleanPolicy(torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32))
    policy = LRRPolicy(clean, obs_dim=4, action_dim=3, delta_max=0.1, final_init_std=0.0)
    obs = torch.rand((5, 4))
    out = policy(obs)
    assert torch.allclose(out.action, out.clean_action, atol=1e-7)
    assert torch.allclose(out.action, clean.get_deterministic_action(obs), atol=1e-7)
    assert all(not parameter.requires_grad for parameter in policy.clean_policy.parameters())


def test_lrr_action_bounds_are_clamped() -> None:
    clean = ConstantCleanPolicy(torch.tensor([0.97, 0.03], dtype=torch.float32))
    policy = LRRPolicy(clean, obs_dim=3, action_dim=2, action_low=0.0, action_high=1.0, delta_max=0.2)
    with torch.no_grad():
        for parameter in policy.residual_net.parameters():
            parameter.zero_()
        final = policy.residual_net.net[-1]
        final.bias.copy_(torch.tensor([10.0, -10.0]))

    out = policy(torch.ones((2, 3), dtype=torch.float32))
    assert torch.all(out.action <= 1.0)
    assert torch.all(out.action >= 0.0)
    assert torch.allclose(out.action[0], torch.tensor([1.0, 0.0]), atol=1e-5)


def test_gated_lrr_initially_matches_clean_policy() -> None:
    clean = ConstantCleanPolicy(torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32))
    policy = LRRPolicy(
        clean,
        obs_dim=4,
        action_dim=3,
        delta_max=0.1,
        final_init_std=0.0,
        use_repair_gate=True,
    )
    obs = torch.rand((5, 4))
    out = policy(obs)
    assert torch.allclose(out.action, out.clean_action, atol=1e-7)
    assert torch.allclose(out.raw_residual, torch.zeros_like(out.raw_residual), atol=1e-7)
    assert torch.allclose(out.repair_gate, torch.full_like(out.repair_gate, 0.5), atol=1e-7)


def test_repair_gate_can_suppress_raw_residual() -> None:
    clean = ConstantCleanPolicy(torch.tensor([0.5, 0.5], dtype=torch.float32))
    policy = LRRPolicy(clean, obs_dim=3, action_dim=2, delta_max=0.2, use_repair_gate=True)
    assert policy.gate_net is not None
    with torch.no_grad():
        for parameter in policy.residual_net.parameters():
            parameter.zero_()
        for parameter in policy.gate_net.parameters():
            parameter.zero_()
        policy.residual_net.net[-1].bias.copy_(torch.tensor([10.0, -10.0]))
        policy.gate_net.net[-1].bias.fill_(-20.0)

    out = policy(torch.ones((1, 3), dtype=torch.float32))
    assert torch.all(torch.abs(out.raw_residual) > 0.19)
    assert torch.all(out.repair_gate < 1e-7)
    assert torch.allclose(out.action, out.clean_action, atol=1e-6)
