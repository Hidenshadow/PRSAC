"""Local Residual Repair policy wrapper.

LRR keeps the clean PPO/SAC-style policy frozen and learns only a small bounded
residual in the same normalized planner-action space used by the environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class LRRPolicyOutput:
    action: torch.Tensor
    clean_action: torch.Tensor
    residual: torch.Tensor
    raw_residual: torch.Tensor
    repair_gate: torch.Tensor


def _as_1d_tensor(value: float | Iterable[float] | torch.Tensor, dim: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device).flatten()
    if tensor.numel() == 1:
        tensor = tensor.repeat(int(dim))
    if tensor.numel() != int(dim):
        raise ValueError(f"expected scalar or {dim} values, got {tensor.numel()}")
    return tensor


def action_range_fraction(
    fraction: float | Iterable[float] | torch.Tensor,
    action_low: float | Iterable[float] | torch.Tensor,
    action_high: float | Iterable[float] | torch.Tensor,
    action_dim: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert scalar/vector fractions of the action range into absolute deltas."""

    device = torch.device(device)
    low = _as_1d_tensor(action_low, action_dim, device)
    high = _as_1d_tensor(action_high, action_dim, device)
    frac = _as_1d_tensor(fraction, action_dim, device)
    return frac * (high - low)


class ResidualMLP(nn.Module):
    """Small residual network initialized so LRR initially matches clean policy."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
        activation: str = "tanh",
        final_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if activation == "tanh":
            activation_layer: type[nn.Module] = nn.Tanh
        elif activation == "relu":
            activation_layer = nn.ReLU
        else:
            raise ValueError("activation must be 'tanh' or 'relu'")

        layers: list[nn.Module] = []
        in_dim = int(obs_dim)
        for hidden in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(hidden)))
            layers.append(activation_layer())
            in_dim = int(hidden)
        final = nn.Linear(in_dim, int(action_dim))
        nn.init.normal_(final.weight, mean=0.0, std=float(final_init_std))
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class LRRPolicy(nn.Module):
    """Frozen clean-policy anchor plus bounded local residual correction."""

    policy_class = "local_residual_repair"

    def __init__(
        self,
        clean_policy: nn.Module,
        obs_dim: int,
        action_dim: int,
        action_low: float | Iterable[float] | torch.Tensor = 0.0,
        action_high: float | Iterable[float] | torch.Tensor = 1.0,
        delta_max: float | Iterable[float] | torch.Tensor = 0.10,
        hidden_sizes: tuple[int, ...] = (128, 128),
        activation: str = "tanh",
        final_init_std: float = 1e-4,
        use_repair_gate: bool = False,
        gate_initial_logit: float = 0.0,
    ) -> None:
        super().__init__()
        self.clean_policy = clean_policy
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.use_repair_gate = bool(use_repair_gate)
        for parameter in self.clean_policy.parameters():
            parameter.requires_grad_(False)
        self.clean_policy.eval()

        device = next(self.clean_policy.parameters(), torch.empty(0)).device
        self.register_buffer("action_low", _as_1d_tensor(action_low, self.action_dim, device))
        self.register_buffer("action_high", _as_1d_tensor(action_high, self.action_dim, device))
        delta = _as_1d_tensor(delta_max, self.action_dim, device)
        self.register_buffer("delta_max", torch.clamp(delta, min=0.0))
        self.residual_net = ResidualMLP(
            self.obs_dim,
            self.action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            final_init_std=final_init_std,
        ).to(device)
        self.gate_net: ResidualMLP | None = None
        if self.use_repair_gate:
            self.gate_net = ResidualMLP(
                self.obs_dim,
                1,
                hidden_sizes=hidden_sizes,
                activation=activation,
                final_init_std=final_init_std,
            ).to(device)
            with torch.no_grad():
                final = self.gate_net.net[-1]
                if isinstance(final, nn.Linear):
                    final.bias.fill_(float(gate_initial_logit))

    @classmethod
    def from_range_fraction(
        cls,
        clean_policy: nn.Module,
        obs_dim: int,
        action_dim: int,
        action_low: float | Iterable[float] | torch.Tensor = 0.0,
        action_high: float | Iterable[float] | torch.Tensor = 1.0,
        delta_max_fraction: float | Iterable[float] | torch.Tensor = 0.10,
        **kwargs: Any,
    ) -> "LRRPolicy":
        device = next(clean_policy.parameters(), torch.empty(0)).device
        delta_max = action_range_fraction(delta_max_fraction, action_low, action_high, action_dim, device)
        return cls(
            clean_policy=clean_policy,
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
            delta_max=delta_max,
            **kwargs,
        )

    @property
    def device(self) -> torch.device:
        return self.action_low.device

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic clean-policy action in normalized action space."""

        obs = obs.to(self.device)
        try:
            action = self.clean_policy.get_deterministic_action(obs)
        except TypeError:
            action = self.clean_policy.get_deterministic_action(obs, None)
        return torch.maximum(torch.minimum(action, self.action_high), self.action_low)

    def residual(self, obs: torch.Tensor) -> torch.Tensor:
        raw = self.raw_residual(obs)
        return raw * self.repair_gate(obs)

    def raw_residual(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        return self.delta_max * torch.tanh(self.residual_net(obs))

    def gate_logits(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)
        if not self.use_repair_gate or self.gate_net is None:
            return torch.full((obs.shape[0], 1), 20.0, dtype=obs.dtype, device=self.device)
        return self.gate_net(obs)

    def repair_gate(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.gate_logits(obs)
        gate = torch.sigmoid(logits)
        return gate.expand(-1, self.action_dim)

    def forward(self, obs: torch.Tensor) -> LRRPolicyOutput:
        obs = obs.to(self.device)
        with torch.no_grad():
            clean_action = self.mean_action(obs)
        raw_residual = self.raw_residual(obs)
        gate = self.repair_gate(obs)
        residual = raw_residual * gate
        action = torch.maximum(torch.minimum(clean_action + residual, self.action_high), self.action_low)
        return LRRPolicyOutput(
            action=action,
            clean_action=clean_action,
            residual=residual,
            raw_residual=raw_residual,
            repair_gate=gate,
        )

    @torch.no_grad()
    def predict_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Evaluation/deployment action. This never runs local search."""

        return self.forward(obs).action

    def checkpoint_payload(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "policy_class": self.policy_class,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low.detach().cpu().numpy().tolist(),
            "action_high": self.action_high.detach().cpu().numpy().tolist(),
            "delta_max": self.delta_max.detach().cpu().numpy().tolist(),
            "use_repair_gate": self.use_repair_gate,
            "residual_state_dict": self.residual_net.state_dict(),
            "gate_state_dict": self.gate_net.state_dict() if self.gate_net is not None else None,
            "metadata": dict(metadata or {}),
        }


def save_lrr_checkpoint(path: str | Path, policy: LRRPolicy, metadata: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.checkpoint_payload(metadata), path)


def load_lrr_residual(path: str | Path, policy: LRRPolicy, strict: bool = True) -> dict[str, Any]:
    try:
        checkpoint = torch.load(Path(path), map_location=policy.device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location=policy.device)
    if checkpoint.get("policy_class") != LRRPolicy.policy_class:
        raise ValueError(f"not an LRR checkpoint: {path}")
    checkpoint_uses_gate = bool(checkpoint.get("use_repair_gate", False))
    if checkpoint_uses_gate != bool(policy.use_repair_gate) and strict:
        raise ValueError(
            f"checkpoint use_repair_gate={checkpoint_uses_gate} does not match policy "
            f"use_repair_gate={policy.use_repair_gate}"
        )
    for buffer_name in ("action_low", "action_high", "delta_max"):
        if buffer_name in checkpoint:
            value = torch.as_tensor(checkpoint[buffer_name], dtype=torch.float32, device=policy.device).flatten()
            if value.shape != getattr(policy, buffer_name).shape:
                raise ValueError(
                    f"{buffer_name} shape {tuple(value.shape)} does not match policy "
                    f"{tuple(getattr(policy, buffer_name).shape)}"
                )
            getattr(policy, buffer_name).copy_(value)
    policy.residual_net.load_state_dict(checkpoint["residual_state_dict"], strict=strict)
    if checkpoint_uses_gate and policy.gate_net is not None:
        policy.gate_net.load_state_dict(checkpoint["gate_state_dict"], strict=strict)
    return checkpoint
