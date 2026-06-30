"""PyTorch actor-critic used by the CleanRL-style PPO trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Beta, Independent, Normal
import torch.nn.functional as F


MIN_BETA_CONCENTRATION = 0.3


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal initialization used by many PPO implementations."""

    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CleanRLActorCritic(nn.Module):
    """Bounded continuous-action actor-critic for PPO planner parameters.

    The action space is bounded in [0, 1]. The environment converts the first
    five action values into planner weights and the sixth value into uncertainty
    sensitivity.
    """

    policy_class = "cleanrl_actor_critic"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        acbr_context_dim: int = 16,
        acbr_hidden_dim: int = 64,
        acbr_ensemble_size: int = 3,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.acbr_context_dim = int(acbr_context_dim)
        self.acbr_hidden_dim = int(acbr_hidden_dim)
        self.acbr_ensemble_size = int(acbr_ensemble_size)

        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.alpha_head = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )
        self.acbr_critic = ACBRActionCriticEnsemble(
            obs_dim,
            action_dim,
            context_dim=self.acbr_context_dim,
            hidden_dim=self.acbr_hidden_dim,
            ensemble_size=self.acbr_ensemble_size,
        )

    def get_dist(self, obs: torch.Tensor) -> Independent:
        hidden = self.actor_body(obs)
        alpha = F.softplus(self.alpha_head(hidden)) + MIN_BETA_CONCENTRATION
        beta = F.softplus(self.beta_head(hidden)) + MIN_BETA_CONCENTRATION
        return Independent(Beta(alpha, beta), 1)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(obs)
        return action, log_prob, entropy, value

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.actor_body(obs)
        alpha = F.softplus(self.alpha_head(hidden)) + MIN_BETA_CONCENTRATION
        beta = F.softplus(self.beta_head(hidden)) + MIN_BETA_CONCENTRATION
        return alpha / (alpha + beta)

    def predict_acbr_costs(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.acbr_critic(obs, action, context)


class ACBRActionCriticEnsemble(nn.Module):
    """Small ensemble critic used for attack-context candidate reranking."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        context_dim: int = 16,
        hidden_dim: int = 64,
        ensemble_size: int = 3,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.context_dim = max(int(context_dim), 1)
        self.hidden_dim = max(int(hidden_dim), 8)
        self.ensemble_size = max(int(ensemble_size), 1)
        input_dim = self.obs_dim + self.action_dim + self.context_dim
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    layer_init(nn.Linear(input_dim, self.hidden_dim)),
                    nn.Tanh(),
                    layer_init(nn.Linear(self.hidden_dim, self.hidden_dim)),
                    nn.Tanh(),
                    layer_init(nn.Linear(self.hidden_dim, 1), std=0.01),
                )
                for _ in range(self.ensemble_size)
            ]
        )

    def _context_or_zeros(self, obs: torch.Tensor, context: torch.Tensor | None) -> torch.Tensor:
        if context is None:
            return torch.zeros((obs.shape[0], self.context_dim), dtype=obs.dtype, device=obs.device)
        context = context.to(dtype=obs.dtype, device=obs.device)
        if context.dim() == 1:
            context = context.unsqueeze(0)
        if context.shape[-1] == self.context_dim:
            return context
        if context.shape[-1] > self.context_dim:
            return context[..., : self.context_dim]
        return F.pad(context, (0, self.context_dim - context.shape[-1]))

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        context_tensor = self._context_or_zeros(obs, context)
        x = torch.cat([obs, action, context_tensor], dim=-1)
        return torch.cat([head(x) for head in self.heads], dim=-1)


class CleanRLResidualBeliefActorCritic(nn.Module):
    """PPO actor-critic augmented with a planner-residual belief latent.

    The baseline actor and critic paths are preserved exactly at initialization:
    residual adapters are zero-initialized, so a nominal PPO checkpoint copied
    into this module initially behaves like the original CleanRLActorCritic.
    """

    policy_class = "cleanrl_residual_belief_actor_critic"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        prb_feature_dim: int = 0,
        prb_latent_dim: int = 64,
        prb_hidden_dim: int = 64,
        prb_encoder_type: str = "mlp",
        prb_component_dim: int = 4,
        acbr_context_dim: int = 16,
        acbr_hidden_dim: int = 64,
        acbr_ensemble_size: int = 3,
    ) -> None:
        super().__init__()
        if str(prb_encoder_type) not in {"mlp", "gru"}:
            raise ValueError("CleanRLResidualBeliefActorCritic supports prb_encoder_type='mlp' or 'gru'")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.prb_feature_dim = int(prb_feature_dim)
        self.prb_latent_dim = int(prb_latent_dim)
        self.prb_hidden_dim = int(prb_hidden_dim)
        self.prb_encoder_type = str(prb_encoder_type)
        self.prb_component_dim = int(prb_component_dim)
        self.acbr_context_dim = int(acbr_context_dim)
        self.acbr_hidden_dim = int(acbr_hidden_dim)
        self.acbr_ensemble_size = int(acbr_ensemble_size)

        self.actor_body = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.alpha_head = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        self.beta_head = layer_init(nn.Linear(hidden_size, action_dim), std=0.01)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        feature_dim = max(int(prb_feature_dim), 1)
        if self.prb_encoder_type == "gru":
            self.residual_encoder = nn.GRU(feature_dim, self.prb_latent_dim, batch_first=True)
        else:
            self.residual_encoder = nn.Sequential(
                layer_init(nn.Linear(feature_dim, self.prb_hidden_dim)),
                nn.Tanh(),
                layer_init(nn.Linear(self.prb_hidden_dim, self.prb_latent_dim)),
                nn.Tanh(),
            )
        self.alpha_residual = nn.Linear(self.prb_latent_dim, action_dim)
        self.beta_residual = nn.Linear(self.prb_latent_dim, action_dim)
        self.value_residual = nn.Linear(self.prb_latent_dim, 1)
        nn.init.zeros_(self.alpha_residual.weight)
        nn.init.zeros_(self.alpha_residual.bias)
        nn.init.zeros_(self.beta_residual.weight)
        nn.init.zeros_(self.beta_residual.bias)
        nn.init.zeros_(self.value_residual.weight)
        nn.init.zeros_(self.value_residual.bias)

        aux_input_dim = obs_dim + self.prb_latent_dim + action_dim
        self.prb_aux_body = nn.Sequential(
            layer_init(nn.Linear(aux_input_dim, self.prb_hidden_dim)),
            nn.Tanh(),
        )
        self.prb_residual_total_head = layer_init(nn.Linear(self.prb_hidden_dim, 1), std=0.01)
        self.prb_true_total_head = layer_init(nn.Linear(self.prb_hidden_dim, 1), std=0.01)
        self.prb_component_residual_head = layer_init(
            nn.Linear(self.prb_hidden_dim, self.prb_component_dim),
            std=0.01,
        )
        self.prb_true_component_head = layer_init(
            nn.Linear(self.prb_hidden_dim, self.prb_component_dim),
            std=0.01,
        )
        self.acbr_critic = ACBRActionCriticEnsemble(
            obs_dim,
            action_dim,
            context_dim=self.acbr_context_dim,
            hidden_dim=self.acbr_hidden_dim,
            ensemble_size=self.acbr_ensemble_size,
        )

    def _residual_features_or_zeros(
        self,
        obs: torch.Tensor,
        residual_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if residual_features is None:
            return torch.zeros(
                (obs.shape[0], max(self.prb_feature_dim, 1)),
                dtype=obs.dtype,
                device=obs.device,
            )
        residual_features = residual_features.to(dtype=obs.dtype, device=obs.device)
        if residual_features.dim() == 1:
            residual_features = residual_features.unsqueeze(0)
        if residual_features.shape[-1] == max(self.prb_feature_dim, 1):
            return residual_features
        if residual_features.shape[-1] > max(self.prb_feature_dim, 1):
            return residual_features[..., : max(self.prb_feature_dim, 1)]
        pad_width = max(self.prb_feature_dim, 1) - residual_features.shape[-1]
        return F.pad(residual_features, (0, pad_width))

    def encode_residual(
        self,
        obs: torch.Tensor,
        residual_features: torch.Tensor | None = None,
        detach_latent: bool = False,
    ) -> torch.Tensor:
        features = self._residual_features_or_zeros(obs, residual_features)
        if self.prb_encoder_type == "gru":
            _output, hidden = self.residual_encoder(features.unsqueeze(1))
            latent = torch.tanh(hidden.squeeze(0))
        else:
            latent = self.residual_encoder(features)
        return latent.detach() if bool(detach_latent) else latent

    def get_dist(
        self,
        obs: torch.Tensor,
        residual_features: torch.Tensor | None = None,
        detach_residual_latent: bool = False,
    ) -> Independent:
        hidden = self.actor_body(obs)
        latent = self.encode_residual(obs, residual_features, detach_latent=detach_residual_latent)
        alpha_logits = self.alpha_head(hidden) + self.alpha_residual(latent)
        beta_logits = self.beta_head(hidden) + self.beta_residual(latent)
        alpha = F.softplus(alpha_logits) + MIN_BETA_CONCENTRATION
        beta = F.softplus(beta_logits) + MIN_BETA_CONCENTRATION
        return Independent(Beta(alpha, beta), 1)

    def get_value(
        self,
        obs: torch.Tensor,
        residual_features: torch.Tensor | None = None,
        detach_residual_latent: bool = False,
    ) -> torch.Tensor:
        latent = self.encode_residual(obs, residual_features, detach_latent=detach_residual_latent)
        return (self.critic(obs) + self.value_residual(latent)).squeeze(-1)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
        residual_features: torch.Tensor | None = None,
        detach_residual_latent: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self.get_dist(obs, residual_features, detach_residual_latent=detach_residual_latent)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.get_value(obs, residual_features, detach_residual_latent=detach_residual_latent)
        return action, log_prob, entropy, value

    def get_deterministic_action(
        self,
        obs: torch.Tensor,
        residual_features: torch.Tensor | None = None,
        detach_residual_latent: bool = False,
    ) -> torch.Tensor:
        hidden = self.actor_body(obs)
        latent = self.encode_residual(obs, residual_features, detach_latent=detach_residual_latent)
        alpha_logits = self.alpha_head(hidden) + self.alpha_residual(latent)
        beta_logits = self.beta_head(hidden) + self.beta_residual(latent)
        alpha = F.softplus(alpha_logits) + MIN_BETA_CONCENTRATION
        beta = F.softplus(beta_logits) + MIN_BETA_CONCENTRATION
        return alpha / (alpha + beta)

    def predict_prb_aux(
        self,
        obs: torch.Tensor,
        residual_features: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode_residual(obs, residual_features)
        aux_input = torch.cat([obs, latent, action], dim=-1)
        hidden = self.prb_aux_body(aux_input)
        return {
            "residual_total": self.prb_residual_total_head(hidden).squeeze(-1),
            "true_total": self.prb_true_total_head(hidden).squeeze(-1),
            "component_residual": self.prb_component_residual_head(hidden),
            "true_component": self.prb_true_component_head(hidden),
            "latent": latent,
        }

    def copy_base_policy_from(self, base_agent: CleanRLActorCritic) -> None:
        self.actor_body.load_state_dict(base_agent.actor_body.state_dict())
        self.alpha_head.load_state_dict(base_agent.alpha_head.state_dict())
        self.beta_head.load_state_dict(base_agent.beta_head.state_dict())
        self.critic.load_state_dict(base_agent.critic.state_dict())

    def predict_acbr_costs(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.acbr_critic(obs, action, context)


class CleanRLSACActor(nn.Module):
    """Bounded Gaussian actor used by the SAC baseline.

    The stochastic policy samples through a tanh squashing transform and then
    maps actions from [-1, 1] to the environment's [0, 1] action range.
    """

    policy_class = "cleanrl_sac_actor"

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(obs)
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), -5.0, 2.0)
        return mean, log_std

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        if deterministic:
            squashed = torch.tanh(mean)
            action = 0.5 * (squashed + 1.0)
            log_prob = torch.zeros(obs.shape[0], dtype=obs.dtype, device=obs.device)
            return action, log_prob

        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = normal.rsample()
        squashed = torch.tanh(raw_action)
        action = 0.5 * (squashed + 1.0)
        log_prob = normal.log_prob(raw_action)
        log_prob -= torch.log(0.5 * (1.0 - squashed.pow(2)) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        action, _ = self.get_action(obs, deterministic=True)
        return action


def save_cleanrl_checkpoint(
    path: str | Path,
    agent: CleanRLActorCritic | CleanRLResidualBeliefActorCritic,
    config: dict[str, Any],
    global_step: int,
    best_mean_reward: float | None = None,
) -> None:
    """Save a PPO checkpoint with enough metadata for evaluation/fine-tuning."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": agent.state_dict(),
        "obs_dim": agent.obs_dim,
        "action_dim": agent.action_dim,
        "hidden_size": agent.hidden_size,
        "acbr_context_dim": int(getattr(agent, "acbr_context_dim", 16)),
        "acbr_hidden_dim": int(getattr(agent, "acbr_hidden_dim", 64)),
        "acbr_ensemble_size": int(getattr(agent, "acbr_ensemble_size", 3)),
        "policy_class": getattr(agent, "policy_class", CleanRLActorCritic.policy_class),
        "config": config,
        "global_step": int(global_step),
        "best_mean_reward": best_mean_reward,
    }
    if isinstance(agent, CleanRLResidualBeliefActorCritic):
        checkpoint.update(
            {
                "prb_feature_dim": agent.prb_feature_dim,
                "prb_latent_dim": agent.prb_latent_dim,
                "prb_hidden_dim": agent.prb_hidden_dim,
                "prb_encoder_type": agent.prb_encoder_type,
                "prb_component_dim": agent.prb_component_dim,
            }
        )
    torch.save(checkpoint, path)


def save_cleanrl_sac_checkpoint(
    path: str | Path,
    actor: CleanRLSACActor,
    config: dict[str, Any],
    global_step: int,
    best_mean_reward: float | None = None,
    extra_state: dict[str, Any] | None = None,
) -> None:
    """Save a SAC checkpoint with an evaluator-compatible actor."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": actor.state_dict(),
        "obs_dim": actor.obs_dim,
        "action_dim": actor.action_dim,
        "hidden_size": actor.hidden_size,
        "policy_class": CleanRLSACActor.policy_class,
        "config": config,
        "global_step": int(global_step),
        "best_mean_reward": best_mean_reward,
    }
    if extra_state:
        checkpoint.update(extra_state)
    torch.save(checkpoint, path)


def load_cleanrl_agent(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[CleanRLActorCritic | CleanRLResidualBeliefActorCritic | CleanRLSACActor, dict[str, Any]]:
    """Load a checkpoint saved by train_cleanrl_ppo.py."""

    try:
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(Path(path), map_location=device)

    policy_class = checkpoint.get("policy_class", CleanRLActorCritic.policy_class)
    if policy_class == CleanRLSACActor.policy_class:
        actor = CleanRLSACActor(
            obs_dim=int(checkpoint["obs_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            hidden_size=int(checkpoint.get("hidden_size", 128)),
        ).to(device)
        actor.load_state_dict(checkpoint["model_state_dict"])
        actor.eval()
        return actor, checkpoint

    if policy_class == CleanRLResidualBeliefActorCritic.policy_class:
        agent = CleanRLResidualBeliefActorCritic(
            obs_dim=int(checkpoint["obs_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            hidden_size=int(checkpoint.get("hidden_size", 128)),
            prb_feature_dim=int(checkpoint.get("prb_feature_dim", 0)),
            prb_latent_dim=int(checkpoint.get("prb_latent_dim", 64)),
            prb_hidden_dim=int(checkpoint.get("prb_hidden_dim", 64)),
            prb_encoder_type=str(checkpoint.get("prb_encoder_type", "mlp")),
            prb_component_dim=int(checkpoint.get("prb_component_dim", 4)),
            acbr_context_dim=int(checkpoint.get("acbr_context_dim", 16)),
            acbr_hidden_dim=int(checkpoint.get("acbr_hidden_dim", 64)),
            acbr_ensemble_size=int(checkpoint.get("acbr_ensemble_size", 3)),
        ).to(device)
        agent.load_state_dict(checkpoint["model_state_dict"], strict=False)
        agent.eval()
        return agent, checkpoint

    if policy_class != CleanRLActorCritic.policy_class:
        raise ValueError(
            f"Unsupported checkpoint policy_class={policy_class!r}; "
            "supported checkpoints are PPO CleanRLActorCritic, PRB-PPO CleanRLResidualBeliefActorCritic, "
            "and SAC CleanRLSACActor."
        )

    agent = CleanRLActorCritic(
        obs_dim=int(checkpoint["obs_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        hidden_size=int(checkpoint.get("hidden_size", 128)),
        acbr_context_dim=int(checkpoint.get("acbr_context_dim", 16)),
        acbr_hidden_dim=int(checkpoint.get("acbr_hidden_dim", 64)),
        acbr_ensemble_size=int(checkpoint.get("acbr_ensemble_size", 3)),
    ).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"], strict=False)
    agent.eval()
    return agent, checkpoint


def predict_cleanrl_action(
    agent: CleanRLActorCritic | CleanRLResidualBeliefActorCritic | CleanRLSACActor,
    obs: np.ndarray,
    device: str | torch.device = "cpu",
    deterministic: bool = True,
    residual_features: np.ndarray | None = None,
) -> np.ndarray:
    """Predict one PPO planner-parameter action."""

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    residual_tensor = None
    if residual_features is not None:
        residual_tensor = torch.as_tensor(residual_features, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if deterministic:
            try:
                action = agent.get_deterministic_action(obs_tensor, residual_tensor)
            except TypeError:
                action = agent.get_deterministic_action(obs_tensor)
        else:
            try:
                action, _, _, _ = agent.get_action_and_value(obs_tensor, residual_features=residual_tensor)
            except TypeError:
                action, _, _, _ = agent.get_action_and_value(obs_tensor)
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def predict_acbr_candidate_scores(
    agent: CleanRLActorCritic | CleanRLResidualBeliefActorCritic,
    obs: np.ndarray,
    candidates: np.ndarray,
    context_features: np.ndarray | None = None,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ACBR critic ensemble mean/std for one state's action candidates."""

    candidate_array = np.asarray(candidates, dtype=np.float32)
    if candidate_array.ndim != 2:
        raise ValueError("candidates must be a 2D array")
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
    obs_batch = obs_tensor.expand(candidate_array.shape[0], -1)
    action_tensor = torch.as_tensor(candidate_array, dtype=torch.float32, device=device)
    context_tensor = None
    if context_features is not None:
        context = np.asarray(context_features, dtype=np.float32).reshape(1, -1)
        context_tensor = torch.as_tensor(context, dtype=torch.float32, device=device).expand(candidate_array.shape[0], -1)
    with torch.no_grad():
        scores = agent.predict_acbr_costs(obs_batch, action_tensor, context_tensor)
        mean = scores.mean(dim=-1)
        std = scores.std(dim=-1, unbiased=False)
    return mean.cpu().numpy().astype(np.float32), std.cpu().numpy().astype(np.float32)
