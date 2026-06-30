"""CleanRL-style SAC trainer for the cost-map weight-selection env."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from envs.attack_wrappers import load_attack_config
from train_cleanrl_ppo import (
    evaluate_agent,
    make_env,
    parse_game_action_indices,
    parse_args,
    reset_envs,
    resolve_device,
    set_global_seeds,
    step_envs,
)
from utils.cleanrl_policy import (
    CleanRLSACActor,
    load_cleanrl_agent,
    save_cleanrl_sac_checkpoint,
)
from utils.metrics import OBJECTIVE_NAMES


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_obs: np.ndarray,
        done: np.ndarray,
    ) -> None:
        count = int(obs.shape[0])
        for index in range(count):
            self.obs[self.pos] = obs[index]
            self.actions[self.pos] = action[index]
            self.rewards[self.pos] = reward[index]
            self.next_obs[self.pos] = next_obs[index]
            self.dones[self.pos] = done[index]
            self.pos = (self.pos + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        indices = np.random.randint(0, self.size, size=int(batch_size))
        return (
            torch.as_tensor(self.obs[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.actions[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.next_obs[indices], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.dones[indices], dtype=torch.float32, device=self.device),
        )


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for source_param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * source_param.data)


def sac_hparams(args: Any) -> dict[str, Any]:
    total_timesteps = int(args.total_timesteps)
    return {
        "buffer_size": max(10_000, min(200_000, total_timesteps * 10)),
        "learning_starts": max(128, min(1_000, total_timesteps // 10)),
        "batch_size": 256,
        "tau": 0.005,
        "policy_frequency": 2,
        "target_network_frequency": 1,
        "target_entropy": -float(6),
        "autotune": True,
    }


def load_sac_training_state(
    init_checkpoint: str | None,
    actor: CleanRLSACActor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    alpha_optimizer: optim.Optimizer,
    log_alpha: torch.Tensor,
    device: torch.device,
) -> dict[str, Any] | None:
    if not init_checkpoint:
        return None

    loaded_actor, checkpoint = load_cleanrl_agent(init_checkpoint, device=device)
    if not isinstance(loaded_actor, CleanRLSACActor):
        raise ValueError(f"SAC can only initialize from a SAC checkpoint: {init_checkpoint}")
    actor.load_state_dict(loaded_actor.state_dict())

    if "qf1_state_dict" in checkpoint:
        qf1.load_state_dict(checkpoint["qf1_state_dict"])
    if "qf2_state_dict" in checkpoint:
        qf2.load_state_dict(checkpoint["qf2_state_dict"])
    if "qf1_target_state_dict" in checkpoint:
        qf1_target.load_state_dict(checkpoint["qf1_target_state_dict"])
    else:
        qf1_target.load_state_dict(qf1.state_dict())
    if "qf2_target_state_dict" in checkpoint:
        qf2_target.load_state_dict(checkpoint["qf2_target_state_dict"])
    else:
        qf2_target.load_state_dict(qf2.state_dict())
    if "actor_optimizer_state_dict" in checkpoint:
        actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    if "q_optimizer_state_dict" in checkpoint:
        q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
    if "alpha_optimizer_state_dict" in checkpoint:
        alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer_state_dict"])
    if "log_alpha" in checkpoint:
        log_alpha.data.fill_(float(checkpoint["log_alpha"]))
    return checkpoint


def checkpoint_extra_state(
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    qf1_target: SoftQNetwork,
    qf2_target: SoftQNetwork,
    actor_optimizer: optim.Optimizer,
    q_optimizer: optim.Optimizer,
    alpha_optimizer: optim.Optimizer,
    log_alpha: torch.Tensor,
) -> dict[str, Any]:
    return {
        "qf1_state_dict": qf1.state_dict(),
        "qf2_state_dict": qf2.state_dict(),
        "qf1_target_state_dict": qf1_target.state_dict(),
        "qf2_target_state_dict": qf2_target.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "q_optimizer_state_dict": q_optimizer.state_dict(),
        "alpha_optimizer_state_dict": alpha_optimizer.state_dict(),
        "log_alpha": float(log_alpha.detach().cpu().item()),
    }


def load_sac_anchor_actor(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
) -> tuple[CleanRLSACActor, dict[str, Any]]:
    actor, checkpoint = load_cleanrl_agent(checkpoint_path, device=device)
    if not isinstance(actor, CleanRLSACActor):
        raise ValueError(f"SAC game-aware recovery requires a SAC anchor checkpoint: {checkpoint_path}")
    if int(actor.obs_dim) != int(obs_dim):
        raise ValueError(f"SAC anchor obs_dim={actor.obs_dim} does not match env obs_dim={obs_dim}")
    if int(actor.action_dim) != int(action_dim):
        raise ValueError(f"SAC anchor action_dim={actor.action_dim} does not match env action_dim={action_dim}")
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor, checkpoint


def sac_valt_schedule_value(
    start: float,
    end: float,
    step: int,
    schedule_steps: int,
    schedule: str,
) -> float:
    if schedule_steps <= 0:
        return float(end)
    progress = float(np.clip(float(step) / float(schedule_steps), 0.0, 1.0))
    if schedule == "constant":
        shaped = 1.0
    elif schedule == "cosine":
        shaped = 0.5 - 0.5 * math.cos(math.pi * progress)
    elif schedule == "exp":
        shaped = (math.exp(3.0 * progress) - 1.0) / (math.exp(3.0) - 1.0)
    else:
        shaped = progress
    return float(start + (end - start) * shaped)


def sac_valt_cumulative_step(args: Any, local_global_step: int) -> int:
    return int(getattr(args, "recovery_step_offset", 0)) + int(local_global_step)


def sac_valt_eps(args: Any, local_global_step: int) -> float:
    return sac_valt_schedule_value(
        float(args.sac_valt_eps_start),
        float(args.sac_valt_eps_end),
        sac_valt_cumulative_step(args, local_global_step),
        int(args.sac_valt_schedule_steps),
        str(args.sac_valt_schedule),
    )


def sac_valt_kappa(args: Any, local_global_step: int) -> float:
    return float(
        np.clip(
            sac_valt_schedule_value(
                float(args.sac_valt_kappa_start),
                float(args.sac_valt_kappa_end),
                sac_valt_cumulative_step(args, local_global_step),
                int(args.sac_valt_schedule_steps),
                str(args.sac_valt_schedule),
            ),
            0.0,
            1.0,
        )
    )


def sac_valt_project_linf(
    candidate: torch.Tensor,
    center: torch.Tensor,
    eps: float,
    clip_low: float,
    clip_high: float,
) -> torch.Tensor:
    if eps <= 0.0:
        return torch.clamp(center, float(clip_low), float(clip_high))
    lower = torch.clamp(center - float(eps), float(clip_low), float(clip_high))
    upper = torch.clamp(center + float(eps), float(clip_low), float(clip_high))
    return torch.max(torch.min(candidate, upper), lower)


def sac_valt_uniform_perturb(
    obs: torch.Tensor,
    eps: float,
    clip_low: float,
    clip_high: float,
) -> torch.Tensor:
    if eps <= 0.0:
        return torch.clamp(obs, float(clip_low), float(clip_high))
    noise = torch.empty_like(obs).uniform_(-float(eps), float(eps))
    return torch.clamp(obs + noise, float(clip_low), float(clip_high))


def sac_valt_worst_observation(
    actor: CleanRLSACActor,
    qf1: SoftQNetwork,
    qf2: SoftQNetwork,
    policy_obs: torch.Tensor,
    q_obs: torch.Tensor,
    eps: float,
    iters: int,
    step_size: float,
    noise_scale: float,
    random_start: bool,
    clip_low: float,
    clip_high: float,
    deterministic_action: bool = False,
) -> torch.Tensor:
    if eps <= 0.0 or iters <= 0:
        return torch.clamp(policy_obs.detach(), float(clip_low), float(clip_high))

    center = policy_obs.detach()
    reference_obs = q_obs.detach()
    if random_start:
        adv_obs = sac_valt_uniform_perturb(center, eps, clip_low, clip_high)
    else:
        adv_obs = torch.clamp(center, float(clip_low), float(clip_high))
    update_scale = float(step_size) if step_size > 0.0 else float(eps) / max(int(iters), 1)

    for iteration in range(int(iters)):
        del iteration
        adv_obs = adv_obs.detach().requires_grad_(True)
        if deterministic_action:
            adv_action = actor.get_deterministic_action(adv_obs)
        else:
            adv_action, _adv_log_pi = actor.get_action(adv_obs)
        q_value = torch.min(qf1(reference_obs, adv_action), qf2(reference_obs, adv_action)).mean()
        grad = torch.autograd.grad(q_value, adv_obs, only_inputs=True)[0]
        update = -update_scale * grad.sign()
        if noise_scale > 0.0:
            update = update + torch.randn_like(update) * float(noise_scale) * update_scale
        adv_obs = sac_valt_project_linf(
            adv_obs.detach() + update,
            center,
            eps,
            clip_low,
            clip_high,
        )
    return adv_obs.detach()


def sac_valt_policy_consistency_loss(
    actor: CleanRLSACActor,
    clean_obs: torch.Tensor,
    perturbed_obs: torch.Tensor,
) -> torch.Tensor:
    clean_mean, clean_log_std = actor.forward(clean_obs)
    perturbed_mean, _perturbed_log_std = actor.forward(perturbed_obs)
    clean_std = clean_log_std.exp().detach().clamp_min(1e-3)
    normalized_delta = (clean_mean - perturbed_mean) / clean_std
    return normalized_delta.pow(2).sum(dim=-1).mean()


def main() -> None:
    args = parse_args()
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if bool(args.sac_game_recovery_enabled):
        if not args.init_checkpoint:
            raise ValueError("--sac-game-recovery-enabled is recovery-only and requires --init-checkpoint")
        if not args.sac_game_anchor_checkpoint:
            args.sac_game_anchor_checkpoint = args.init_checkpoint
        if float(args.sac_game_anchor_coef) < 0.0 or float(args.sac_game_advantage_coef) < 0.0:
            raise ValueError("SAC game-aware coefficients must be non-negative")
        if float(args.sac_game_gate_temperature) <= 0.0:
            raise ValueError("--sac-game-gate-temperature must be positive")
        if float(args.sac_game_q_margin) < 0.0:
            raise ValueError("--sac-game-q-margin must be non-negative")
        if float(args.sac_game_lambda_drift_margin) < 0.0 or float(args.sac_game_risk_drift_margin) < 0.0:
            raise ValueError("SAC game-aware drift margins must be non-negative")
        if float(args.sac_game_anchor_barrier_coef) < 0.0:
            raise ValueError("--sac-game-anchor-barrier-coef must be non-negative")
        if float(args.sac_game_anchor_radius) < 0.0:
            raise ValueError("--sac-game-anchor-radius must be non-negative")
    if float(args.sac_target_entropy_scale) < 0.0:
        raise ValueError("--sac-target-entropy-scale must be non-negative")
    if args.sac_fixed_alpha is not None and float(args.sac_fixed_alpha) < 0.0:
        raise ValueError("--sac-fixed-alpha must be non-negative when set")
    if not 0.0 <= float(args.sac_rollout_deterministic_prob) <= 1.0:
        raise ValueError("--sac-rollout-deterministic-prob must be in [0, 1]")
    if float(args.sac_rollout_noise_std) < 0.0:
        raise ValueError("--sac-rollout-noise-std must be non-negative")
    if float(args.sac_log_std_penalty_coef) < 0.0:
        raise ValueError("--sac-log-std-penalty-coef must be non-negative")
    if bool(args.sac_valt_recovery_enabled):
        if float(args.sac_valt_eps_start) < 0.0 or float(args.sac_valt_eps_end) < 0.0:
            raise ValueError("VALT-SAC epsilon schedule values must be non-negative")
        if not 0.0 <= float(args.sac_valt_kappa_start) <= 1.0:
            raise ValueError("--sac-valt-kappa-start must be in [0, 1]")
        if not 0.0 <= float(args.sac_valt_kappa_end) <= 1.0:
            raise ValueError("--sac-valt-kappa-end must be in [0, 1]")
        if int(args.sac_valt_bound_iters) < 0:
            raise ValueError("--sac-valt-bound-iters must be non-negative")
        if float(args.sac_valt_worst_step_size) < 0.0:
            raise ValueError("--sac-valt-worst-step-size must be non-negative")
        if float(args.sac_valt_sgld_noise) < 0.0:
            raise ValueError("--sac-valt-sgld-noise must be non-negative")
        if float(args.sac_valt_policy_reg_coef) < 0.0:
            raise ValueError("--sac-valt-policy-reg-coef must be non-negative")
        if float(args.sac_valt_clip_low) >= float(args.sac_valt_clip_high):
            raise ValueError("--sac-valt-clip-low must be smaller than --sac-valt-clip-high")

    set_global_seeds(args.seed)
    random.seed(args.seed)
    device = resolve_device(args.device)
    observation_attack_config = load_attack_config(args.observation_attack_config)
    environment_attack_config = load_attack_config(args.environment_attack_config)
    hp = sac_hparams(args)

    run_name = f"cleanrl_sac_costmap_seed{args.seed}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tensorboard"))

    config = vars(args).copy()
    config.update(
        {
            "algorithm": (
                "valt_sac_recovery"
                if bool(args.sac_valt_recovery_enabled)
                else "game_aware_sac"
                if bool(args.sac_game_recovery_enabled)
                else "cleanrl_style_sac_tanh_gaussian"
            ),
            "device": str(device),
            "run_dir": str(run_dir),
            "observation_attack": observation_attack_config,
            "environment_attack": environment_attack_config,
            **hp,
        }
    )
    print(json.dumps(config, indent=2))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    envs = [
        make_env(
            args.map_size,
            args.seed + 1000 * index,
            args.scenario,
            args.observation_mode,
            args.reward_mode,
            args.reward_scale,
            args.reward_cost_key,
            args.action_mode,
            args.action_gain,
            args.max_uncertainty_lambda,
            args.attack_budget_fraction,
            args.attack_strength,
            args.map_sampling_mode,
            args.fixed_map_seed,
            args.map_seed_pool_size,
            min_start_goal_distance_ratio=args.min_start_goal_distance_ratio,
            attacker_temperature=args.attacker_temperature,
            attacker_response=args.attacker_response,
            attacker_top_fraction=args.attacker_top_fraction,
            attacker_sharpness=args.attacker_sharpness,
            observation_attack_config=observation_attack_config,
            environment_attack_config=environment_attack_config,
            env_kind=args.env_kind,
            layers_path=args.layers_path,
            task_split_path=args.train_tasks,
            mission_profile_scenario=args.mission_profile_scenario,
        )
        for index in range(args.num_envs)
    ]
    obs_np = reset_envs(envs, args.seed)
    obs_dim = int(envs[0].observation_space.shape[0])
    action_dim = int(envs[0].action_space.shape[0])
    hp["target_entropy"] = -float(action_dim) * float(args.sac_target_entropy_scale)
    config["target_entropy"] = hp["target_entropy"]

    actor = CleanRLSACActor(obs_dim, action_dim, hidden_size=args.hidden_size).to(device)
    qf1 = SoftQNetwork(obs_dim, action_dim, hidden_size=args.hidden_size).to(device)
    qf2 = SoftQNetwork(obs_dim, action_dim, hidden_size=args.hidden_size).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim, hidden_size=args.hidden_size).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim, hidden_size=args.hidden_size).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=args.learning_rate)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=args.learning_rate)
    if args.sac_fixed_alpha is not None:
        fixed_alpha = max(float(args.sac_fixed_alpha), 1e-8)
        log_alpha.data.fill_(float(np.log(fixed_alpha)))
        log_alpha.requires_grad_(False)

    def current_alpha() -> torch.Tensor:
        if args.sac_fixed_alpha is not None:
            return torch.as_tensor(float(args.sac_fixed_alpha), dtype=torch.float32, device=device)
        return log_alpha.exp()

    checkpoint = load_sac_training_state(
        args.init_checkpoint,
        actor,
        qf1,
        qf2,
        qf1_target,
        qf2_target,
        actor_optimizer,
        q_optimizer,
        alpha_optimizer,
        log_alpha,
        device,
    )
    if checkpoint is not None:
        print(
            f"Initialized SAC policy from {args.init_checkpoint} "
            f"(checkpoint_step={checkpoint.get('global_step', 'unknown')})"
        )
    if args.sac_fixed_alpha is not None:
        fixed_alpha = max(float(args.sac_fixed_alpha), 1e-8)
        log_alpha.data.fill_(float(np.log(fixed_alpha)))
        log_alpha.requires_grad_(False)

    sac_game_anchor_actor: CleanRLSACActor | None = None
    sac_game_risk_indices: list[int] = []
    sac_game_risk_index_tensor = torch.empty(0, dtype=torch.long, device=device)
    sac_game_lambda_index = len(OBJECTIVE_NAMES) if action_dim > len(OBJECTIVE_NAMES) else None
    if bool(args.sac_game_recovery_enabled):
        sac_game_anchor_actor, sac_game_anchor_checkpoint = load_sac_anchor_actor(
            str(args.sac_game_anchor_checkpoint),
            obs_dim,
            action_dim,
            device,
        )
        if float(args.sac_game_risk_drift_coef) > 0.0:
            sac_game_risk_indices = parse_game_action_indices(args.sac_game_risk_action_indices, action_dim)
            sac_game_risk_index_tensor = torch.as_tensor(sac_game_risk_indices, dtype=torch.long, device=device)
        print(
            "Game-aware SAC recovery enabled: "
            f"anchor={args.sac_game_anchor_checkpoint} "
            f"anchor_step={sac_game_anchor_checkpoint.get('global_step', 'unknown')} "
            f"anchor_coef={args.sac_game_anchor_coef} "
            f"advantage_coef={args.sac_game_advantage_coef} "
            f"q_margin={args.sac_game_q_margin} "
            f"gate_temperature={args.sac_game_gate_temperature} "
            f"lambda_drift={args.sac_game_lambda_drift_coef}@{args.sac_game_lambda_drift_margin} "
            f"risk_drift={args.sac_game_risk_drift_coef}@{args.sac_game_risk_drift_margin} "
            f"risk_indices={sac_game_risk_indices}"
        )
    if bool(args.sac_valt_recovery_enabled):
        print(
            "VALT-SAC recovery enabled: "
            f"eps={args.sac_valt_eps_start}->{args.sac_valt_eps_end} "
            f"kappa={args.sac_valt_kappa_start}->{args.sac_valt_kappa_end} "
            f"schedule={args.sac_valt_schedule}/{args.sac_valt_schedule_steps} "
            f"bound_iters={args.sac_valt_bound_iters} "
            f"policy_reg_coef={args.sac_valt_policy_reg_coef}"
        )

    replay = ReplayBuffer(int(hp["buffer_size"]), obs_dim, action_dim, device)
    global_step = 0
    start_time = time.time()
    last_eval_step = 0
    best_mean_reward: float | None = None
    stale_eval_count = 0
    eval_records: list[dict[str, float | int]] = []
    rollout_metric_records: list[dict[str, Any]] = []
    last_sac_game_anchor_loss = torch.tensor(0.0, device=device)
    last_sac_game_advantage_loss = torch.tensor(0.0, device=device)
    last_sac_game_gate_mean = torch.tensor(0.0, device=device)
    last_sac_game_q_advantage_mean = torch.tensor(0.0, device=device)
    last_sac_game_action_distance_mean = torch.tensor(0.0, device=device)
    last_sac_game_lambda_drift_loss = torch.tensor(0.0, device=device)
    last_sac_game_risk_drift_loss = torch.tensor(0.0, device=device)
    last_sac_game_anchor_barrier_loss = torch.tensor(0.0, device=device)
    last_sac_log_std_penalty = torch.tensor(0.0, device=device)
    last_sac_valt_eps = torch.tensor(0.0, device=device)
    last_sac_valt_kappa = torch.tensor(0.0, device=device)
    last_sac_valt_policy_reg_loss = torch.tensor(0.0, device=device)
    last_sac_valt_random_actor_loss = torch.tensor(0.0, device=device)
    last_sac_valt_worst_actor_loss = torch.tensor(0.0, device=device)
    last_sac_valt_worst_delta_linf = torch.tensor(0.0, device=device)

    def run_eval(step: int) -> dict[str, float]:
        return evaluate_agent(
            actor,
            map_size=args.map_size,
            scenario=args.scenario,
            seed=int(args.eval_seed) if args.eval_seed is not None else args.seed + 50_000 + step,
            num_episodes=args.n_eval_episodes,
            device=device,
            observation_mode=args.observation_mode,
            reward_mode=args.reward_mode,
            reward_scale=args.reward_scale,
            reward_cost_key=args.reward_cost_key,
            action_mode=args.action_mode,
            action_gain=args.action_gain,
            max_uncertainty_lambda=args.max_uncertainty_lambda,
            attack_budget_fraction=args.attack_budget_fraction,
            attack_strength=args.attack_strength,
            map_sampling_mode=args.map_sampling_mode,
            fixed_map_seed=args.fixed_map_seed,
            map_seed_pool_size=args.map_seed_pool_size,
            min_start_goal_distance_ratio=args.min_start_goal_distance_ratio,
            attacker_temperature=args.attacker_temperature,
            attacker_response=args.attacker_response,
            attacker_top_fraction=args.attacker_top_fraction,
            attacker_sharpness=args.attacker_sharpness,
            observation_attack_config=observation_attack_config,
            environment_attack_config=environment_attack_config,
            env_kind=args.env_kind,
            layers_path=args.layers_path,
            eval_tasks=args.eval_tasks,
            train_tasks=args.train_tasks,
            mission_profile_scenario=args.mission_profile_scenario,
        )

    def sac_game_diagnostics() -> dict[str, float]:
        diagnostics: dict[str, float] = {}
        if bool(args.sac_game_recovery_enabled):
            diagnostics.update(
                {
                    "sac_game_anchor_loss": float(last_sac_game_anchor_loss.detach().cpu().item()),
                    "sac_game_advantage_loss": float(last_sac_game_advantage_loss.detach().cpu().item()),
                    "sac_game_gate_mean": float(last_sac_game_gate_mean.detach().cpu().item()),
                    "sac_game_q_advantage_mean": float(last_sac_game_q_advantage_mean.detach().cpu().item()),
                    "sac_game_action_distance_mean": float(last_sac_game_action_distance_mean.detach().cpu().item()),
                    "sac_game_lambda_drift_loss": float(last_sac_game_lambda_drift_loss.detach().cpu().item()),
                    "sac_game_risk_drift_loss": float(last_sac_game_risk_drift_loss.detach().cpu().item()),
                    "sac_game_anchor_barrier_loss": float(last_sac_game_anchor_barrier_loss.detach().cpu().item()),
                    "sac_log_std_penalty": float(last_sac_log_std_penalty.detach().cpu().item()),
                }
            )
        if bool(args.sac_valt_recovery_enabled):
            diagnostics.update(
                {
                    "sac_valt_eps": float(last_sac_valt_eps.detach().cpu().item()),
                    "sac_valt_kappa": float(last_sac_valt_kappa.detach().cpu().item()),
                    "sac_valt_policy_reg_loss": float(last_sac_valt_policy_reg_loss.detach().cpu().item()),
                    "sac_valt_random_actor_loss": float(last_sac_valt_random_actor_loss.detach().cpu().item()),
                    "sac_valt_worst_actor_loss": float(last_sac_valt_worst_actor_loss.detach().cpu().item()),
                    "sac_valt_worst_delta_linf": float(last_sac_valt_worst_delta_linf.detach().cpu().item()),
                }
            )
        return diagnostics

    if args.n_eval_episodes > 0:
        initial_eval_metrics = run_eval(0)
        eval_records.append({"global_step": 0, **initial_eval_metrics, **sac_game_diagnostics()})
        pd.DataFrame(eval_records).to_csv(run_dir / "eval_metrics.csv", index=False)
        writer.add_scalar("eval/mean_reward", initial_eval_metrics["mean_reward"], 0)
        writer.add_scalar("eval/success_rate", initial_eval_metrics["success_rate"], 0)
        writer.add_scalar("eval/mean_scalar_cost", initial_eval_metrics["mean_scalar_cost"], 0)
        writer.add_scalar("eval/mean_attacked_scalar_cost", initial_eval_metrics["mean_attacked_scalar_cost"], 0)
        writer.add_scalar("eval/mean_lambda_uncertainty", initial_eval_metrics["mean_lambda_uncertainty"], 0)
        print(
            f"step=0 "
            f"eval_reward={initial_eval_metrics['mean_reward']:.4f} "
            f"eval_success={initial_eval_metrics['success_rate']:.3f} "
            f"eval_attacked_cost={initial_eval_metrics['mean_attacked_scalar_cost']:.4f} "
            f"eval_lambda={initial_eval_metrics['mean_lambda_uncertainty']:.3f}"
        )

    while global_step < int(args.total_timesteps):
        if global_step < int(hp["learning_starts"]):
            actions_np = np.stack([env.action_space.sample() for env in envs]).astype(np.float32)
        else:
            obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            with torch.no_grad():
                sampled_actions = actor.get_action(obs_tensor)[0].cpu().numpy().astype(np.float32)
                actions_np = sampled_actions
                deterministic_prob = float(args.sac_rollout_deterministic_prob)
                if deterministic_prob > 0.0:
                    deterministic_actions = actor.get_deterministic_action(obs_tensor).cpu().numpy().astype(np.float32)
                    mask = np.random.random(size=deterministic_actions.shape[0]) < deterministic_prob
                    if float(args.sac_rollout_noise_std) > 0.0:
                        noise = np.random.normal(
                            loc=0.0,
                            scale=float(args.sac_rollout_noise_std),
                            size=deterministic_actions.shape,
                        ).astype(np.float32)
                        deterministic_actions = np.clip(deterministic_actions + noise, 1e-6, 1.0 - 1e-6)
                    actions_np = np.where(mask[:, None], deterministic_actions, sampled_actions).astype(np.float32)

        next_obs_np, rewards_np, dones_np, infos = step_envs(envs, actions_np)
        replay.add(obs_np, actions_np, rewards_np, next_obs_np, dones_np)
        obs_np = next_obs_np
        global_step += len(envs)
        if args.rollout_metrics_path:
            for env_index, info in enumerate(infos):
                variant_id = info.get("environment_attack_mixture_variant_id", "")
                variant_scale = info.get("environment_attack_mixture_variant_scale", np.nan)
                policy_cost = info.get(args.reward_cost_key, info.get("scalar_cost", np.nan))
                rollout_metric_records.append(
                    {
                        "global_step": int(global_step),
                        "env_index": int(env_index),
                        "variant_id": str(variant_id),
                        "variant_scale": float(variant_scale) if variant_scale not in ("", None) else np.nan,
                        "policy_cost": float(policy_cost),
                        "scalar_cost": float(info.get("scalar_cost", np.nan)),
                        "attacked_scalar_cost": float(info.get("attacked_scalar_cost", np.nan)),
                        "soft_attacked_scalar_cost": float(info.get("soft_attacked_scalar_cost", np.nan)),
                        "returned_reward": float(info.get("returned_reward", rewards_np[env_index])),
                        "success": float(info.get("success", 0.0)),
                        "reward_cost_key": str(args.reward_cost_key),
                        "reward_cost_source": str(info.get("reward_cost_source", "")),
                        "environment_attack_type": str(info.get("environment_attack_type", "")),
                    }
                )

        qf1_loss = torch.tensor(0.0, device=device)
        qf2_loss = torch.tensor(0.0, device=device)
        actor_loss = torch.tensor(0.0, device=device)
        alpha_loss = torch.tensor(0.0, device=device)
        alpha = current_alpha().detach()

        if replay.size >= int(hp["learning_starts"]):
            batch_size = min(int(hp["batch_size"]), replay.size)
            b_obs, b_actions, b_rewards, b_next_obs, b_dones = replay.sample(batch_size)

            if bool(args.sac_valt_recovery_enabled):
                valt_eps = sac_valt_eps(args, global_step)
                valt_kappa = sac_valt_kappa(args, global_step)
                random_next_obs = sac_valt_uniform_perturb(
                    b_next_obs,
                    valt_eps,
                    float(args.sac_valt_clip_low),
                    float(args.sac_valt_clip_high),
                ).detach()
                if valt_eps > 0.0 and valt_kappa > 0.0 and int(args.sac_valt_bound_iters) > 0:
                    worst_next_obs = sac_valt_worst_observation(
                        actor,
                        qf1_target,
                        qf2_target,
                        b_next_obs,
                        b_next_obs,
                        eps=valt_eps,
                        iters=int(args.sac_valt_bound_iters),
                        step_size=float(args.sac_valt_worst_step_size),
                        noise_scale=float(args.sac_valt_sgld_noise),
                        random_start=bool(args.sac_valt_random_start),
                        clip_low=float(args.sac_valt_clip_low),
                        clip_high=float(args.sac_valt_clip_high),
                        deterministic_action=bool(args.sac_valt_attack_deterministic),
                    )
                else:
                    worst_next_obs = random_next_obs
                with torch.no_grad():
                    random_next_actions, random_next_log_pi = actor.get_action(random_next_obs)
                    random_target_q1 = qf1_target(b_next_obs, random_next_actions)
                    random_target_q2 = qf2_target(b_next_obs, random_next_actions)
                    random_target_q = (
                        torch.min(random_target_q1, random_target_q2)
                        - current_alpha().detach() * random_next_log_pi
                    )
                    worst_next_actions, worst_next_log_pi = actor.get_action(worst_next_obs)
                    worst_target_q1 = qf1_target(b_next_obs, worst_next_actions)
                    worst_target_q2 = qf2_target(b_next_obs, worst_next_actions)
                    worst_target_q = (
                        torch.min(worst_target_q1, worst_target_q2)
                        - current_alpha().detach() * worst_next_log_pi
                    )
                    min_target_q = (1.0 - valt_kappa) * random_target_q + valt_kappa * worst_target_q
                    next_q_value = b_rewards + (1.0 - b_dones) * float(args.gamma) * min_target_q
            else:
                with torch.no_grad():
                    next_actions, next_log_pi = actor.get_action(b_next_obs)
                    target_q1 = qf1_target(b_next_obs, next_actions)
                    target_q2 = qf2_target(b_next_obs, next_actions)
                    min_target_q = torch.min(target_q1, target_q2) - current_alpha().detach() * next_log_pi
                    next_q_value = b_rewards + (1.0 - b_dones) * float(args.gamma) * min_target_q

            qf1_a_values = qf1(b_obs, b_actions)
            qf2_a_values = qf2(b_obs, b_actions)
            qf1_loss = torch.nn.functional.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = torch.nn.functional.mse_loss(qf2_a_values, next_q_value)
            q_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            if global_step % int(hp["policy_frequency"]) == 0:
                sac_valt_policy_reg_loss = torch.tensor(0.0, device=device)
                sac_valt_random_actor_loss = torch.tensor(0.0, device=device)
                sac_valt_worst_actor_loss = torch.tensor(0.0, device=device)
                sac_valt_worst_delta_linf = torch.tensor(0.0, device=device)
                if bool(args.sac_valt_recovery_enabled):
                    valt_eps = sac_valt_eps(args, global_step)
                    valt_kappa = sac_valt_kappa(args, global_step)
                    random_obs = sac_valt_uniform_perturb(
                        b_obs,
                        valt_eps,
                        float(args.sac_valt_clip_low),
                        float(args.sac_valt_clip_high),
                    ).detach()
                    random_sampled_pi, random_log_pi = actor.get_action(random_obs)
                    random_pi = (
                        actor.get_deterministic_action(random_obs)
                        if bool(args.sac_deterministic_actor_update)
                        else random_sampled_pi
                    )
                    random_qf1_pi = qf1(b_obs, random_pi)
                    random_qf2_pi = qf2(b_obs, random_pi)
                    random_min_qf_pi = torch.min(random_qf1_pi, random_qf2_pi)
                    random_actor_loss_vec = current_alpha().detach() * random_log_pi - random_min_qf_pi

                    if valt_eps > 0.0 and valt_kappa > 0.0 and int(args.sac_valt_bound_iters) > 0:
                        worst_obs = sac_valt_worst_observation(
                            actor,
                            qf1,
                            qf2,
                            b_obs,
                            b_obs,
                            eps=valt_eps,
                            iters=int(args.sac_valt_bound_iters),
                            step_size=float(args.sac_valt_worst_step_size),
                            noise_scale=float(args.sac_valt_sgld_noise),
                            random_start=bool(args.sac_valt_random_start),
                            clip_low=float(args.sac_valt_clip_low),
                            clip_high=float(args.sac_valt_clip_high),
                            deterministic_action=bool(args.sac_valt_attack_deterministic),
                        )
                    else:
                        worst_obs = random_obs
                    worst_sampled_pi, worst_log_pi = actor.get_action(worst_obs)
                    worst_pi = (
                        actor.get_deterministic_action(worst_obs)
                        if bool(args.sac_deterministic_actor_update)
                        else worst_sampled_pi
                    )
                    worst_qf1_pi = qf1(b_obs, worst_pi)
                    worst_qf2_pi = qf2(b_obs, worst_pi)
                    worst_min_qf_pi = torch.min(worst_qf1_pi, worst_qf2_pi)
                    worst_actor_loss_vec = current_alpha().detach() * worst_log_pi - worst_min_qf_pi

                    actor_loss_vec = (1.0 - valt_kappa) * random_actor_loss_vec + valt_kappa * worst_actor_loss_vec
                    actor_loss = actor_loss_vec.mean()
                    log_pi = (1.0 - valt_kappa) * random_log_pi + valt_kappa * worst_log_pi
                    pi = worst_pi if valt_kappa >= 0.5 else random_pi
                    min_qf_pi = worst_min_qf_pi if valt_kappa >= 0.5 else random_min_qf_pi
                    sac_valt_random_actor_loss = random_actor_loss_vec.mean()
                    sac_valt_worst_actor_loss = worst_actor_loss_vec.mean()
                    sac_valt_worst_delta_linf = (worst_obs - b_obs).abs().amax(dim=-1).mean()
                    if float(args.sac_valt_policy_reg_coef) > 0.0:
                        sac_valt_policy_reg_loss = sac_valt_policy_consistency_loss(actor, b_obs, worst_obs)
                        actor_loss = actor_loss + float(args.sac_valt_policy_reg_coef) * sac_valt_policy_reg_loss
                else:
                    sampled_pi, log_pi = actor.get_action(b_obs)
                    pi = actor.get_deterministic_action(b_obs) if bool(args.sac_deterministic_actor_update) else sampled_pi
                    qf1_pi = qf1(b_obs, pi)
                    qf2_pi = qf2(b_obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((current_alpha().detach() * log_pi) - min_qf_pi).mean()
                sac_log_std_penalty = torch.tensor(0.0, device=device)
                if float(args.sac_log_std_penalty_coef) > 0.0:
                    _mean_for_std, log_std_for_penalty = actor.forward(b_obs)
                    sac_log_std_penalty = torch.relu(
                        log_std_for_penalty - float(args.sac_log_std_target)
                    ).pow(2).mean()
                    actor_loss = actor_loss + float(args.sac_log_std_penalty_coef) * sac_log_std_penalty

                sac_game_anchor_loss = torch.tensor(0.0, device=device)
                sac_game_advantage_loss = torch.tensor(0.0, device=device)
                sac_game_gate_mean = torch.tensor(0.0, device=device)
                sac_game_q_advantage_mean = torch.tensor(0.0, device=device)
                sac_game_action_distance_mean = torch.tensor(0.0, device=device)
                sac_game_lambda_drift_loss = torch.tensor(0.0, device=device)
                sac_game_risk_drift_loss = torch.tensor(0.0, device=device)
                sac_game_anchor_barrier_loss = torch.tensor(0.0, device=device)
                if sac_game_anchor_actor is not None:
                    with torch.no_grad():
                        anchor_action = sac_game_anchor_actor.get_deterministic_action(b_obs)
                        q_anchor = torch.min(qf1(b_obs, anchor_action), qf2(b_obs, anchor_action))
                    q_advantage = min_qf_pi - q_anchor
                    gate_logits = (
                        q_advantage - float(args.sac_game_q_margin)
                    ) / max(float(args.sac_game_gate_temperature), 1e-6)
                    gate = torch.sigmoid(gate_logits).detach()
                    action_distance = (pi - anchor_action).pow(2).mean(dim=-1)
                    sac_game_anchor_loss = ((1.0 - gate) * action_distance).mean()
                    sac_game_advantage_loss = torch.relu(
                        q_anchor + float(args.sac_game_q_margin) - min_qf_pi
                    ).mean()
                    sac_game_gate_mean = gate.mean()
                    sac_game_q_advantage_mean = q_advantage.mean()
                    sac_game_action_distance_mean = action_distance.mean()
                    if float(args.sac_game_anchor_barrier_coef) > 0.0:
                        rms_distance = torch.sqrt(action_distance + 1e-8)
                        sac_game_anchor_barrier_loss = torch.relu(
                            rms_distance - float(args.sac_game_anchor_radius)
                        ).pow(2).mean()
                        actor_loss = (
                            actor_loss
                            + float(args.sac_game_anchor_barrier_coef) * sac_game_anchor_barrier_loss
                        )
                    if float(args.sac_game_anchor_coef) > 0.0:
                        actor_loss = actor_loss + float(args.sac_game_anchor_coef) * sac_game_anchor_loss
                    if float(args.sac_game_advantage_coef) > 0.0:
                        actor_loss = actor_loss + float(args.sac_game_advantage_coef) * sac_game_advantage_loss
                    if (
                        sac_game_lambda_index is not None
                        and float(args.sac_game_lambda_drift_coef) > 0.0
                    ):
                        lambda_excess = torch.relu(
                            pi[:, sac_game_lambda_index]
                            - anchor_action[:, sac_game_lambda_index]
                            - float(args.sac_game_lambda_drift_margin)
                        )
                        sac_game_lambda_drift_loss = lambda_excess.pow(2).mean()
                        actor_loss = actor_loss + float(args.sac_game_lambda_drift_coef) * sac_game_lambda_drift_loss
                    if (
                        sac_game_risk_index_tensor.numel() > 0
                        and float(args.sac_game_risk_drift_coef) > 0.0
                    ):
                        risk_excess = torch.relu(
                            pi.index_select(1, sac_game_risk_index_tensor)
                            - anchor_action.index_select(1, sac_game_risk_index_tensor)
                            - float(args.sac_game_risk_drift_margin)
                        )
                        sac_game_risk_drift_loss = risk_excess.pow(2).mean()
                        actor_loss = actor_loss + float(args.sac_game_risk_drift_coef) * sac_game_risk_drift_loss

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                last_sac_game_anchor_loss = sac_game_anchor_loss.detach()
                last_sac_game_advantage_loss = sac_game_advantage_loss.detach()
                last_sac_game_gate_mean = sac_game_gate_mean.detach()
                last_sac_game_q_advantage_mean = sac_game_q_advantage_mean.detach()
                last_sac_game_action_distance_mean = sac_game_action_distance_mean.detach()
                last_sac_game_lambda_drift_loss = sac_game_lambda_drift_loss.detach()
                last_sac_game_risk_drift_loss = sac_game_risk_drift_loss.detach()
                last_sac_game_anchor_barrier_loss = sac_game_anchor_barrier_loss.detach()
                last_sac_log_std_penalty = sac_log_std_penalty.detach()
                if bool(args.sac_valt_recovery_enabled):
                    last_sac_valt_eps = torch.as_tensor(float(valt_eps), dtype=torch.float32, device=device)
                    last_sac_valt_kappa = torch.as_tensor(float(valt_kappa), dtype=torch.float32, device=device)
                    last_sac_valt_policy_reg_loss = sac_valt_policy_reg_loss.detach()
                    last_sac_valt_random_actor_loss = sac_valt_random_actor_loss.detach()
                    last_sac_valt_worst_actor_loss = sac_valt_worst_actor_loss.detach()
                    last_sac_valt_worst_delta_linf = sac_valt_worst_delta_linf.detach()

                if args.sac_fixed_alpha is None:
                    alpha_loss = (-(log_alpha.exp() * (log_pi + float(hp["target_entropy"])).detach())).mean()
                    alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    alpha_optimizer.step()
                else:
                    alpha_loss = torch.tensor(0.0, device=device)

            if global_step % int(hp["target_network_frequency"]) == 0:
                soft_update(qf1, qf1_target, float(hp["tau"]))
                soft_update(qf2, qf2_target, float(hp["tau"]))

        sps = int(global_step / max(time.time() - start_time, 1e-8))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("rollout/mean_reward", float(np.mean(rewards_np)), global_step)
        writer.add_scalar("rollout/success_rate", float(np.mean([info["success"] for info in infos])), global_step)
        writer.add_scalar("losses/qf1_loss", float(qf1_loss.detach().cpu().item()), global_step)
        writer.add_scalar("losses/qf2_loss", float(qf2_loss.detach().cpu().item()), global_step)
        writer.add_scalar("losses/actor_loss", float(actor_loss.detach().cpu().item()), global_step)
        writer.add_scalar("losses/alpha_loss", float(alpha_loss.detach().cpu().item()), global_step)
        writer.add_scalar("losses/alpha", float(alpha.detach().cpu().item()), global_step)
        writer.add_scalar("losses/sac_log_std_penalty", float(last_sac_log_std_penalty.item()), global_step)
        if bool(args.sac_game_recovery_enabled):
            writer.add_scalar("losses/sac_game_anchor_loss", float(last_sac_game_anchor_loss.item()), global_step)
            writer.add_scalar("losses/sac_game_advantage_loss", float(last_sac_game_advantage_loss.item()), global_step)
            writer.add_scalar("losses/sac_game_lambda_drift_loss", float(last_sac_game_lambda_drift_loss.item()), global_step)
            writer.add_scalar("losses/sac_game_risk_drift_loss", float(last_sac_game_risk_drift_loss.item()), global_step)
            writer.add_scalar("losses/sac_game_anchor_barrier_loss", float(last_sac_game_anchor_barrier_loss.item()), global_step)
            writer.add_scalar("game/sac_gate_mean", float(last_sac_game_gate_mean.item()), global_step)
            writer.add_scalar("game/sac_q_advantage_mean", float(last_sac_game_q_advantage_mean.item()), global_step)
            writer.add_scalar("game/sac_action_distance_mean", float(last_sac_game_action_distance_mean.item()), global_step)
        if bool(args.sac_valt_recovery_enabled):
            writer.add_scalar("valt/eps", float(last_sac_valt_eps.item()), global_step)
            writer.add_scalar("valt/kappa", float(last_sac_valt_kappa.item()), global_step)
            writer.add_scalar("losses/sac_valt_policy_reg_loss", float(last_sac_valt_policy_reg_loss.item()), global_step)
            writer.add_scalar("losses/sac_valt_random_actor_loss", float(last_sac_valt_random_actor_loss.item()), global_step)
            writer.add_scalar("losses/sac_valt_worst_actor_loss", float(last_sac_valt_worst_actor_loss.item()), global_step)
            writer.add_scalar("valt/worst_delta_linf", float(last_sac_valt_worst_delta_linf.item()), global_step)

        if global_step - last_eval_step >= args.eval_freq or global_step >= int(args.total_timesteps):
            last_eval_step = global_step
            eval_metrics = run_eval(global_step)
            eval_records.append({"global_step": int(global_step), **eval_metrics, **sac_game_diagnostics()})
            pd.DataFrame(eval_records).to_csv(run_dir / "eval_metrics.csv", index=False)
            writer.add_scalar("eval/mean_reward", eval_metrics["mean_reward"], global_step)
            writer.add_scalar("eval/success_rate", eval_metrics["success_rate"], global_step)
            writer.add_scalar("eval/mean_scalar_cost", eval_metrics["mean_scalar_cost"], global_step)
            writer.add_scalar("eval/mean_attacked_scalar_cost", eval_metrics["mean_attacked_scalar_cost"], global_step)
            writer.add_scalar("eval/mean_lambda_uncertainty", eval_metrics["mean_lambda_uncertainty"], global_step)

            min_eval_delta = max(float(args.min_eval_delta), 0.0)
            is_best = (
                best_mean_reward is None
                or eval_metrics["mean_reward"] > best_mean_reward + min_eval_delta
            )
            if is_best:
                best_mean_reward = eval_metrics["mean_reward"]
                stale_eval_count = 0
                save_cleanrl_sac_checkpoint(
                    run_dir / "best_model.pt",
                    actor,
                    config,
                    global_step,
                    best_mean_reward,
                    extra_state=checkpoint_extra_state(
                        qf1,
                        qf2,
                        qf1_target,
                        qf2_target,
                        actor_optimizer,
                        q_optimizer,
                        alpha_optimizer,
                        log_alpha,
                    ),
                )
            else:
                stale_eval_count += 1

            game_text = ""
            if bool(args.sac_game_recovery_enabled):
                game_text = (
                    f"sac_gate={float(last_sac_game_gate_mean.item()):.3f} "
                    f"q_adv={float(last_sac_game_q_advantage_mean.item()):.4f} "
                )
            if bool(args.sac_valt_recovery_enabled):
                game_text += (
                    f"valt_eps={float(last_sac_valt_eps.item()):.3f} "
                    f"valt_kappa={float(last_sac_valt_kappa.item()):.3f} "
                )
            print(
                f"step={global_step} "
                f"rollout_reward={np.mean(rewards_np):.4f} "
                f"eval_reward={eval_metrics['mean_reward']:.4f} "
                f"eval_success={eval_metrics['success_rate']:.3f} "
                f"eval_attacked_cost={eval_metrics['mean_attacked_scalar_cost']:.4f} "
                f"alpha={float(current_alpha().detach().cpu().item()):.3f} "
                f"{game_text}"
                f"stale={stale_eval_count} "
                f"sps={sps}"
            )
            if args.early_stop_patience > 0 and stale_eval_count >= args.early_stop_patience:
                print(
                    f"Early stopping at step={global_step}: "
                    f"no eval improvement for {stale_eval_count} evals "
                    f"(best_reward={best_mean_reward:.4f})."
                )
                break

    save_cleanrl_sac_checkpoint(
        run_dir / "final_model.pt",
        actor,
        config,
        global_step,
        best_mean_reward,
        extra_state=checkpoint_extra_state(
            qf1,
            qf2,
            qf1_target,
            qf2_target,
            actor_optimizer,
            q_optimizer,
            alpha_optimizer,
            log_alpha,
        ),
    )
    if args.rollout_metrics_path:
        rollout_metrics_path = Path(args.rollout_metrics_path)
        rollout_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rollout_metric_records).to_csv(rollout_metrics_path, index=False)

    for env in envs:
        env.close()
    writer.close()
    print(f"Saved final model to {run_dir / 'final_model.pt'}")
    print(f"Best model path: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
