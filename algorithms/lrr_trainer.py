"""Weighted residual-regression trainer for Local Residual Repair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from algorithms.lrr_policy import LRRPolicy
from algorithms.local_repair import RepairLabel


@dataclass(frozen=True)
class LRRTrainerConfig:
    repair_batch_size: int = 64
    repair_epochs: int = 5
    learning_rate: float = 3e-4
    lambda_zero: float = 1e-3
    lambda_gate: float = 0.1
    positive_gate_weight: float = 1.0
    negative_gate_weight: float = 1.0


class LocalRepairDataset(Dataset):
    def __init__(self, observations: np.ndarray, labels: Iterable[RepairLabel]) -> None:
        self.observations = torch.as_tensor(np.asarray(observations, dtype=np.float32))
        label_list = list(labels)
        self.target_residuals = torch.as_tensor(
            np.stack([label.target_residual for label in label_list], axis=0).astype(np.float32)
            if label_list
            else np.zeros((0, 0), dtype=np.float32)
        )
        self.weights = torch.as_tensor(
            np.asarray([label.weight for label in label_list], dtype=np.float32)
            if label_list
            else np.zeros((0,), dtype=np.float32)
        )
        if len(self.observations) != len(label_list):
            raise ValueError("observations and labels must have the same length")

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.observations[index], self.target_residuals[index], self.weights[index]


class LRRTrainer:
    def __init__(self, policy: LRRPolicy, config: LRRTrainerConfig | None = None) -> None:
        self.policy = policy
        self.config = config or LRRTrainerConfig()
        trainable = [parameter for parameter in self.policy.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.Adam(trainable, lr=float(self.config.learning_rate))

    def _zero_loss(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=self.policy.device)
        residual = self.policy.residual(observations.to(self.policy.device))
        return residual.pow(2).mean()

    def _gate_loss(
        self,
        positive_observations: torch.Tensor,
        negative_observations: torch.Tensor,
    ) -> torch.Tensor:
        if not self.policy.use_repair_gate:
            return torch.zeros((), dtype=torch.float32, device=self.policy.device)
        losses = []
        weights = []
        if positive_observations.numel() > 0:
            pos_logits = self.policy.gate_logits(positive_observations.to(self.policy.device)).reshape(-1)
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
                pos_logits,
                torch.ones_like(pos_logits),
                reduction="none",
            ))
            weights.append(torch.full_like(pos_logits, float(self.config.positive_gate_weight)))
        if negative_observations.numel() > 0:
            neg_logits = self.policy.gate_logits(negative_observations.to(self.policy.device)).reshape(-1)
            losses.append(torch.nn.functional.binary_cross_entropy_with_logits(
                neg_logits,
                torch.zeros_like(neg_logits),
                reduction="none",
            ))
            weights.append(torch.full_like(neg_logits, float(self.config.negative_gate_weight)))
        if not losses:
            return torch.zeros((), dtype=torch.float32, device=self.policy.device)
        loss = torch.cat(losses)
        weight = torch.cat(weights)
        return (loss * weight).sum() / torch.clamp(weight.sum(), min=1.0)

    def update(
        self,
        repair_observations: np.ndarray,
        labels: list[RepairLabel],
        zero_observations: np.ndarray | None = None,
        negative_gate_observations: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Run weighted regression on positive labels plus residual shrinkage."""

        if zero_observations is None:
            zero_observations = repair_observations
        zero_tensor = torch.as_tensor(np.asarray(zero_observations, dtype=np.float32), device=self.policy.device)
        if negative_gate_observations is None:
            negative_gate_observations = np.zeros((0, zero_tensor.shape[-1] if zero_tensor.ndim == 2 else 0), dtype=np.float32)
        negative_gate_tensor = torch.as_tensor(
            np.asarray(negative_gate_observations, dtype=np.float32),
            device=self.policy.device,
        )
        metrics = {
            "lrr/loss_total": 0.0,
            "lrr/loss_repair": 0.0,
            "lrr/loss_zero": 0.0,
            "lrr/loss_gate": 0.0,
            "lrr/mean_repair_weight": float(np.mean([label.weight for label in labels])) if labels else 0.0,
        }

        if labels:
            dataset = LocalRepairDataset(repair_observations, labels)
            loader = DataLoader(
                dataset,
                batch_size=max(int(self.config.repair_batch_size), 1),
                shuffle=True,
                drop_last=False,
            )
            total_loss = []
            repair_losses = []
            zero_losses = []
            gate_losses = []
            for _ in range(max(int(self.config.repair_epochs), 1)):
                for obs_batch, target_batch, weight_batch in loader:
                    obs_batch = obs_batch.to(self.policy.device)
                    target_batch = target_batch.to(self.policy.device)
                    weight_batch = weight_batch.to(self.policy.device)
                    pred = self.policy.residual(obs_batch)
                    per_sample = (pred - target_batch).pow(2).mean(dim=-1)
                    loss_repair = (per_sample * weight_batch).sum() / torch.clamp(weight_batch.sum(), min=1.0)
                    loss_zero = self._zero_loss(zero_tensor)
                    loss_gate = self._gate_loss(obs_batch, negative_gate_tensor)
                    loss = (
                        loss_repair
                        + float(self.config.lambda_zero) * loss_zero
                        + float(self.config.lambda_gate) * loss_gate
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    total_loss.append(float(loss.detach().cpu()))
                    repair_losses.append(float(loss_repair.detach().cpu()))
                    zero_losses.append(float(loss_zero.detach().cpu()))
                    gate_losses.append(float(loss_gate.detach().cpu()))
            metrics.update(
                {
                    "lrr/loss_total": float(np.mean(total_loss)) if total_loss else 0.0,
                    "lrr/loss_repair": float(np.mean(repair_losses)) if repair_losses else 0.0,
                    "lrr/loss_zero": float(np.mean(zero_losses)) if zero_losses else 0.0,
                    "lrr/loss_gate": float(np.mean(gate_losses)) if gate_losses else 0.0,
                }
            )
        else:
            if zero_tensor.numel() > 0 or negative_gate_tensor.numel() > 0:
                loss_zero = self._zero_loss(zero_tensor)
                positive_gate_tensor = torch.zeros((0, zero_tensor.shape[-1] if zero_tensor.ndim == 2 else 0), dtype=torch.float32, device=self.policy.device)
                loss_gate = self._gate_loss(positive_gate_tensor, negative_gate_tensor)
                loss = float(self.config.lambda_zero) * loss_zero + float(self.config.lambda_gate) * loss_gate
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                metrics["lrr/loss_total"] = float(loss.detach().cpu())
                metrics["lrr/loss_zero"] = float(loss_zero.detach().cpu())
                metrics["lrr/loss_gate"] = float(loss_gate.detach().cpu())

        return metrics
