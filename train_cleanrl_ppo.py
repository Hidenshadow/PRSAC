"""CleanRL-style single-file PPO trainer for the cost-map weight-selection env."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import time
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter

from envs.attack_wrappers import (
    configure_env_attacker_kwargs,
    load_attack_config,
    wrap_env_with_attacks,
)
from envs.costmap_env import MultiObjectiveCostmapEnv
from envs.real_terrain_env import RealTerrainPlanningEnv
from utils.cleanrl_policy import (
    CleanRLActorCritic,
    CleanRLResidualBeliefActorCritic,
    load_cleanrl_agent,
    save_cleanrl_checkpoint,
)
from utils.metrics import (
    ATTACKER_RESPONSE_MODES,
    DEFAULT_ATTACKER_RESPONSE,
    DEFAULT_ATTACKER_SHARPNESS,
    DEFAULT_ATTACKER_TEMPERATURE,
    DEFAULT_ATTACKER_TOP_FRACTION,
    DEFAULT_ATTACK_BUDGET_FRACTION,
    DEFAULT_ATTACK_STRENGTH,
    DEFAULT_FIXED_MAP_SEED,
    DEFAULT_MAP_SEED_POOL_SIZE,
    MAP_SAMPLING_MODES,
    OBSERVATION_MODES,
    OBJECTIVE_NAMES,
    REWARD_MODES,
)
from utils.planner_regret import (
    CPAConfig,
    CandidateActionConfig,
    PairwisePreferenceConfig,
    PlannerCounterfactualEvaluator,
    PlannerRegretBuffer,
    PlannerRegretTargetConfig,
    build_game_planner_regret_target,
    build_planner_regret_target,
    compute_cpa_advantages,
    generate_candidate_actions,
    generate_structured_candidate_actions,
    merge_candidate_action_sets,
)
from utils.planner_residual_belief import (
    PlannerResidualFeatureBuilder,
    PlannerResidualFeatureConfig,
    attack_belief_record,
    neutral_probe_action,
    prb_auxiliary_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--env-kind",
        choices=("synthetic", "real_terrain"),
        default="synthetic",
        help="Environment backend. The default preserves the synthetic map trainer.",
    )
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--min-start-goal-distance-ratio", type=float, default=0.55)
    parser.add_argument("--map-sampling-mode", choices=MAP_SAMPLING_MODES, default="random")
    parser.add_argument("--fixed-map-seed", type=int, default=DEFAULT_FIXED_MAP_SEED)
    parser.add_argument("--map-seed-pool-size", type=int, default=DEFAULT_MAP_SEED_POOL_SIZE)
    parser.add_argument("--scenario", type=str, default="nominal")
    parser.add_argument("--layers-path", type=str, default=None, help="Real terrain .npz layer file.")
    parser.add_argument("--train-tasks", type=str, default=None, help="Real terrain train task split JSON.")
    parser.add_argument("--eval-tasks", type=str, default=None, help="Real terrain eval task split JSON.")
    parser.add_argument(
        "--mission-profile-scenario",
        type=str,
        default=None,
        help="Synthetic mission-profile proxy used for real terrain context sampling.",
    )
    parser.add_argument("--log-dir", type=str, default="runs/ppo")
    parser.add_argument("--observation-mode", choices=OBSERVATION_MODES, default="terrain")
    parser.add_argument(
        "--reward-mode",
        choices=REWARD_MODES,
        default="relative_heuristic",
    )
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument(
        "--reward-cost-key",
        choices=("scalar_cost", "attacked_scalar_cost", "soft_attacked_scalar_cost"),
        default="attacked_scalar_cost",
        help="Planner result cost key used for relative/advantage rewards.",
    )
    parser.add_argument("--action-mode", choices=("direct", "preference_delta"), default="preference_delta")
    parser.add_argument("--action-gain", type=float, default=2.0)
    parser.add_argument("--max-uncertainty-lambda", type=float, default=2.0)
    parser.add_argument("--attack-budget-fraction", type=float, default=DEFAULT_ATTACK_BUDGET_FRACTION)
    parser.add_argument("--attack-strength", type=float, default=DEFAULT_ATTACK_STRENGTH)
    parser.add_argument("--attacker-temperature", type=float, default=DEFAULT_ATTACKER_TEMPERATURE)
    parser.add_argument("--attacker-response", choices=ATTACKER_RESPONSE_MODES, default=DEFAULT_ATTACKER_RESPONSE)
    parser.add_argument("--attacker-top-fraction", type=float, default=DEFAULT_ATTACKER_TOP_FRACTION)
    parser.add_argument("--attacker-sharpness", type=float, default=DEFAULT_ATTACKER_SHARPNESS)
    parser.add_argument(
        "--observation-attack-config",
        type=str,
        default=None,
        help="JSON text or path configuring ObservationAttackWrapper.",
    )
    parser.add_argument(
        "--environment-attack-config",
        type=str,
        default=None,
        help="JSON text or path configuring EnvironmentAttackWrapper/env attacker params.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="CleanRL checkpoint to initialize from before training/fine-tuning.",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--anneal-lr", action="store_true")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", dest="norm_adv", action="store_true")
    parser.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--clip-vloss", action="store_true")
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--mra-enabled",
        action="store_true",
        help="Enable Minimax Recovery Advantage PPO losses for robust attack recovery fine-tuning.",
    )
    parser.add_argument("--mra-cvar-quantile", type=float, default=0.75)
    parser.add_argument("--mra-cvar-weight", type=float, default=1.0)
    parser.add_argument("--mra-tail-excess-weight", type=float, default=0.5)
    parser.add_argument("--mra-risk-feature-weight", type=float, default=0.25)
    parser.add_argument("--mra-tail-reward-penalty", type=float, default=0.0)
    parser.add_argument("--mra-risk-reward-penalty", type=float, default=0.0)
    parser.add_argument("--mra-weight-cap", type=float, default=5.0)
    parser.add_argument("--mra-anchor-coef", type=float, default=0.0)
    parser.add_argument(
        "--mra-anchor-mask",
        choices=("all", "objective_only", "lambda_only", "hazard_lambda", "risk_layers"),
        default="risk_layers",
    )
    parser.add_argument(
        "--mirror-nominal-prior-coef",
        type=float,
        default=0.0,
        help="Optional weak action-mean prior to the initial nominal policy for MIRROR-PPO chunks.",
    )
    parser.add_argument(
        "--game-recovery-enabled",
        action="store_true",
        help="Enable local Stackelberg/game-regularized recovery losses. Intended only for attack recovery chunks.",
    )
    parser.add_argument(
        "--game-anchor-checkpoint",
        type=str,
        default=None,
        help="Nominal PPO checkpoint used as the recovery regularization anchor. Defaults to the initialized policy.",
    )
    parser.add_argument(
        "--game-nominal-prior-coef",
        type=float,
        default=0.0,
        help="L2 coefficient keeping the recovery policy near the nominal deterministic action.",
    )
    parser.add_argument(
        "--game-lambda-drift-coef",
        type=float,
        default=0.0,
        help="Penalty coefficient for uncertainty-lambda action increases beyond the nominal anchor plus margin.",
    )
    parser.add_argument("--game-lambda-drift-margin", type=float, default=0.10)
    parser.add_argument(
        "--game-risk-drift-coef",
        type=float,
        default=0.0,
        help="Penalty coefficient for risk-objective action increases beyond the nominal anchor plus margin.",
    )
    parser.add_argument("--game-risk-drift-margin", type=float, default=0.10)
    parser.add_argument(
        "--game-risk-action-indices",
        type=str,
        default="energy,hazard,communication,illumination",
        help="Comma-separated objective action indices or names regularized as risk-conservatism channels.",
    )
    parser.add_argument(
        "--sac-game-recovery-enabled",
        action="store_true",
        help=(
            "Enable conservative game-aware SAC actor regularization. "
            "This is consumed by train_cleanrl_sac.py during attacked recovery chunks."
        ),
    )
    parser.add_argument(
        "--sac-game-anchor-checkpoint",
        type=str,
        default=None,
        help="Nominal SAC checkpoint used as the incumbent anchor for game-aware SAC recovery.",
    )
    parser.add_argument(
        "--sac-game-anchor-coef",
        type=float,
        default=0.25,
        help="Penalty for deviating from the incumbent SAC action when Q improvement is not convincing.",
    )
    parser.add_argument(
        "--sac-game-advantage-coef",
        type=float,
        default=0.10,
        help="Hinge penalty that encourages sampled SAC actions to beat the incumbent action in twin-Q value.",
    )
    parser.add_argument(
        "--sac-game-q-margin",
        type=float,
        default=0.02,
        help="Minimum twin-Q advantage over the incumbent action before the gate opens.",
    )
    parser.add_argument(
        "--sac-game-gate-temperature",
        type=float,
        default=0.05,
        help="Temperature for the soft gate from incumbent-constrained to free SAC improvement.",
    )
    parser.add_argument("--sac-game-lambda-drift-coef", type=float, default=0.0)
    parser.add_argument("--sac-game-lambda-drift-margin", type=float, default=0.10)
    parser.add_argument("--sac-game-risk-drift-coef", type=float, default=0.0)
    parser.add_argument("--sac-game-risk-drift-margin", type=float, default=0.10)
    parser.add_argument("--sac-game-anchor-barrier-coef", type=float, default=0.0)
    parser.add_argument("--sac-game-anchor-radius", type=float, default=0.15)
    parser.add_argument(
        "--sac-game-risk-action-indices",
        type=str,
        default="energy,hazard,communication,illumination",
        help="Comma-separated action names or indices constrained for SAC game-aware recovery.",
    )
    parser.add_argument("--sac-deterministic-actor-update", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sac-target-entropy-scale", type=float, default=1.0)
    parser.add_argument("--sac-fixed-alpha", type=float, default=None)
    parser.add_argument("--sac-rollout-deterministic-prob", type=float, default=0.0)
    parser.add_argument("--sac-rollout-noise-std", type=float, default=0.0)
    parser.add_argument("--sac-log-std-penalty-coef", type=float, default=0.0)
    parser.add_argument("--sac-log-std-target", type=float, default=-1.5)
    parser.add_argument(
        "--sac-valt-recovery-enabled",
        action="store_true",
        help="Enable VALT-SAC style virtual alternative observation training in SAC recovery chunks.",
    )
    parser.add_argument("--sac-valt-eps-start", type=float, default=0.0)
    parser.add_argument("--sac-valt-eps-end", type=float, default=0.08)
    parser.add_argument("--sac-valt-kappa-start", type=float, default=0.0)
    parser.add_argument("--sac-valt-kappa-end", type=float, default=0.30)
    parser.add_argument("--sac-valt-schedule-steps", type=int, default=20_480)
    parser.add_argument("--sac-valt-schedule", choices=("constant", "linear", "cosine", "exp"), default="linear")
    parser.add_argument("--sac-valt-bound-iters", type=int, default=2)
    parser.add_argument("--sac-valt-worst-step-size", type=float, default=0.0)
    parser.add_argument("--sac-valt-sgld-noise", type=float, default=0.0)
    parser.add_argument("--sac-valt-policy-reg-coef", type=float, default=1.0)
    parser.add_argument("--sac-valt-random-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sac-valt-clip-low", type=float, default=0.0)
    parser.add_argument("--sac-valt-clip-high", type=float, default=1.0)
    parser.add_argument("--sac-valt-attack-deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--rollout-metrics-path",
        type=str,
        default=None,
        help="Optional CSV path for one-row-per-completed-episode rollout metrics.",
    )
    parser.add_argument("--pr-enabled", action="store_true", help="Enable Planner-Regret PPO auxiliary loss.")
    parser.add_argument(
        "--pr-enable-queries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect planner-regret counterfactual queries during rollouts.",
    )
    parser.add_argument("--pr-query-fraction", type=float, default=0.25)
    parser.add_argument("--pr-query-interval", type=int, default=4)
    parser.add_argument("--pr-num-candidates", type=int, default=16)
    parser.add_argument("--pr-aux-loss-type", choices=("mse", "nll", "cpa", "pairwise_pref"), default="mse")
    parser.add_argument("--pr-target-type", choices=("hard", "soft", "hybrid"), default="soft")
    parser.add_argument("--pr-game-teacher-enabled", action="store_true")
    parser.add_argument("--pr-game-teacher-mode", choices=("minimax", "soft_stackelberg"), default="minimax")
    parser.add_argument("--pr-game-max-attack-variants", type=int, default=6)
    parser.add_argument("--pr-game-softmax-temperature", type=float, default=0.08)
    parser.add_argument("--pr-include-structured-candidates", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pr-num-structured-candidates", type=int, default=8)
    parser.add_argument("--pr-soft-temperature", type=float, default=0.10)
    parser.add_argument("--pr-hard-regret-threshold", type=float, default=0.02)
    parser.add_argument("--pr-soft-regret-threshold", type=float, default=0.005)
    parser.add_argument("--pr-cpa-temperature", type=float, default=0.03)
    parser.add_argument("--pr-min-positive-adv", type=float, default=0.001)
    parser.add_argument("--pr-cpa-weighting", choices=("softmax", "linear"), default="softmax")
    parser.add_argument("--pr-cpa-sample-weight", choices=("max_adv",), default="max_adv")
    parser.add_argument("--pr-pair-reference", choices=("old_policy", "stored_logp", "none"), default="old_policy")
    parser.add_argument("--pr-pref-temperature", type=float, default=1.0)
    parser.add_argument("--pr-adv-temperature", type=float, default=0.03)
    parser.add_argument("--pr-store-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pr-local-sigma", type=float, default=0.10)
    parser.add_argument("--pr-local-sigmas", nargs="*", type=float, default=None)
    parser.add_argument("--pr-risk-dim-indices", type=str, default="")
    parser.add_argument("--pr-risk-local-sigma", type=float, default=0.20)
    parser.add_argument("--pr-include-risk-axis-perturbations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pr-include-risk-block-perturbations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pr-num-random-candidates", type=int, default=4)
    parser.add_argument("--pr-include-policy-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pr-include-nominal-action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pr-include-zero-delta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pr-include-axis-perturbations", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha-pr", type=float, default=0.0)
    parser.add_argument("--beta-nominal", type=float, default=0.0)
    parser.add_argument("--regret-weight-max", type=float, default=2.0)
    parser.add_argument("--recovery-step-offset", type=int, default=0)
    parser.add_argument("--pr-start-after-steps", type=int, default=0)
    parser.add_argument("--pr-ramp-steps", type=int, default=2048)
    parser.add_argument("--pr-guidance-schedule", choices=("constant", "early_decay"), default="constant")
    parser.add_argument("--pr-active-until-step", type=int, default=2048)
    parser.add_argument("--pr-decay-to-zero-by-step", type=int, default=3072)
    parser.add_argument("--pr-regret-adaptive-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pr-regret-low", type=float, default=0.005)
    parser.add_argument("--pr-regret-high", type=float, default=0.02)
    parser.add_argument("--pr-grad-ratio-controller", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pr-grad-ratio-target-low", type=float, default=0.05)
    parser.add_argument("--pr-grad-ratio-target-high", type=float, default=0.10)
    parser.add_argument("--pr-alpha-min", type=float, default=0.05)
    parser.add_argument("--pr-alpha-max", type=float, default=1.0)
    parser.add_argument("--pr-query-strategy", choices=("interval_random", "mixed"), default="interval_random")
    parser.add_argument("--pr-priority-fraction", type=float, default=0.5)
    parser.add_argument("--pr-priority-key", choices=("local_cost", "immediate_cost", "negative_advantage", "action_deviation_from_nominal"), default="local_cost")
    parser.add_argument("--pr-cost-normalization", choices=("policy_abs", "range"), default="policy_abs")
    parser.add_argument("--pr-random-target-control", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pr-failure-cost", type=float, default=1e6)
    parser.add_argument("--pr-grad-diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pr-grad-diagnostics-every", type=int, default=1)
    parser.add_argument("--pr-log-path", type=str, default=None)
    parser.add_argument("--prb-enabled", action="store_true", help="Enable Planner-Residual Belief PPO.")
    parser.add_argument("--prb-encoder-type", choices=("mlp", "gru"), default="mlp")
    parser.add_argument("--prb-latent-dim", type=int, default=64)
    parser.add_argument("--prb-hidden-dim", type=int, default=64)
    parser.add_argument("--prb-aux-coef", type=float, default=0.05)
    parser.add_argument("--prb-disable-aux-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prb-use-component-costs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prb-use-scalar-cost", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prb-normalize-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prb-feature-clip", type=float, default=5.0)
    parser.add_argument("--prb-lambda-residual-total", type=float, default=1.0)
    parser.add_argument("--prb-lambda-true-total", type=float, default=0.5)
    parser.add_argument("--prb-lambda-component-residual", type=float, default=1.0)
    parser.add_argument("--prb-lambda-true-component", type=float, default=0.5)
    parser.add_argument("--prb-stopgrad-residual-latent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prb-use-random-residual-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prb-probe-action", choices=("neutral",), default="neutral")
    parser.add_argument("--prb-log-path", type=str, default=None)
    parser.add_argument(
        "--bagr-enabled",
        action="store_true",
        help=(
            "Enable Belief-Adaptive Game Recovery PPO. This recovery-only mode augments "
            "planner-residual belief features with an attack-variant posterior and trains "
            "a constrained residual policy around the nominal checkpoint."
        ),
    )
    parser.add_argument("--bagr-anchor-checkpoint", type=str, default=None)
    parser.add_argument("--bagr-max-attack-variants", type=int, default=6)
    parser.add_argument("--bagr-belief-temperature", type=float, default=0.25)
    parser.add_argument("--bagr-belief-prior-mix", type=float, default=0.10)
    parser.add_argument("--bagr-freeze-base-actor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bagr-residual-l2-coef", type=float, default=0.10)
    parser.add_argument("--bagr-residual-barrier-coef", type=float, default=2.0)
    parser.add_argument("--bagr-residual-action-limit", type=float, default=0.18)
    parser.add_argument("--bagr-confidence-limit-scale", type=float, default=1.0)
    parser.add_argument(
        "--trr-enabled",
        action="store_true",
        help=(
            "Enable Teacher-Residual Recovery PPO. Recovery-only mode that keeps the "
            "clean PPO actor as an anchor and trains a residual adapter toward gated "
            "planner/game-teacher actions."
        ),
    )
    parser.add_argument("--trr-anchor-checkpoint", type=str, default=None)
    parser.add_argument("--trr-freeze-base-actor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trr-alpha", type=float, default=1.0)
    parser.add_argument("--trr-ramp-steps", type=int, default=1024)
    parser.add_argument("--trr-guidance-schedule", choices=("constant", "early_decay"), default="early_decay")
    parser.add_argument("--trr-active-until-step", type=int, default=4096)
    parser.add_argument("--trr-decay-to-zero-by-step", type=int, default=12288)
    parser.add_argument("--trr-min-normalized-regret", type=float, default=0.01)
    parser.add_argument("--trr-residual-l2-coef", type=float, default=0.05)
    parser.add_argument("--trr-residual-barrier-coef", type=float, default=1.0)
    parser.add_argument("--trr-residual-action-limit", type=float, default=0.22)
    parser.add_argument(
        "--acbr-enabled",
        action="store_true",
        help=(
            "Enable Attack-Context Belief Reranking PPO. Recovery-only mode that "
            "learns a context-conditioned action-cost critic from planner candidate "
            "queries and reranks candidate actions at decision time."
        ),
    )
    parser.add_argument("--acbr-anchor-checkpoint", type=str, default=None)
    parser.add_argument("--acbr-critic-coef", type=float, default=0.50)
    parser.add_argument("--acbr-uncertainty-coef", type=float, default=0.25)
    parser.add_argument("--acbr-anchor-penalty", type=float, default=0.15)
    parser.add_argument("--acbr-policy-penalty", type=float, default=0.05)
    parser.add_argument("--acbr-target-clip", type=float, default=3.0)
    parser.add_argument("--acbr-rerank-start-after-steps", type=int, default=1024)
    parser.add_argument("--acbr-benefit-gate-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--acbr-benefit-margin", type=float, default=0.0)
    parser.add_argument("--acbr-min-query-states", type=int, default=8)
    parser.add_argument("--acbr-context-dim", type=int, default=16)
    parser.add_argument("--acbr-hidden-dim", type=int, default=64)
    parser.add_argument("--acbr-ensemble-size", type=int, default=3)
    parser.add_argument("--bc-warmstart-steps", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--n-eval-episodes", type=int, default=25)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "Fixed validation seed used at every eval point. If unset, the old "
            "global-step-dependent eval seed is used. A fixed seed gives a much "
            "cleaner convergence signal for this one-step contextual bandit."
        ),
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop after this many evals without improvement. Zero disables early stopping.",
    )
    parser.add_argument(
        "--min-eval-delta",
        type=float,
        default=1e-4,
        help="Minimum eval reward improvement counted for best checkpoint and early stopping.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.set_defaults(norm_adv=True)
    return parser.parse_args()


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_env(
    map_size: int,
    seed: int,
    scenario: str,
    observation_mode: str,
    reward_mode: str,
    reward_scale: float,
    reward_cost_key: str,
    action_mode: str,
    action_gain: float,
    max_uncertainty_lambda: float,
    attack_budget_fraction: float,
    attack_strength: float,
    map_sampling_mode: str,
    fixed_map_seed: int,
    map_seed_pool_size: int,
    min_start_goal_distance_ratio: float = 0.55,
    attacker_temperature: float = DEFAULT_ATTACKER_TEMPERATURE,
    attacker_response: str = DEFAULT_ATTACKER_RESPONSE,
    attacker_top_fraction: float = DEFAULT_ATTACKER_TOP_FRACTION,
    attacker_sharpness: float = DEFAULT_ATTACKER_SHARPNESS,
    observation_attack_config: dict[str, Any] | None = None,
    environment_attack_config: dict[str, Any] | None = None,
    env_kind: str = "synthetic",
    layers_path: str | None = None,
    task_split_path: str | None = None,
    mission_profile_scenario: str | None = None,
) -> gym.Env:
    environment_attack_config_for_env = copy.deepcopy(environment_attack_config) if environment_attack_config else None
    if (
        environment_attack_config_for_env
        and str(environment_attack_config_for_env.get("type", "")) == "env_attack_mixture"
        and "seed" in environment_attack_config_for_env
    ):
        # Mixture-based recovery methods sample attack variants inside the
        # wrapper at episode reset. Offset by sub-env seed so vectorized envs do
        # not share identical variant sequences.
        environment_attack_config_for_env["seed"] = int(environment_attack_config_for_env["seed"]) + int(seed)
    attacker_kwargs = configure_env_attacker_kwargs(environment_attack_config_for_env)
    if env_kind == "real_terrain":
        if not layers_path:
            raise ValueError("--layers-path is required when --env-kind=real_terrain")
        if not task_split_path:
            raise ValueError("--train-tasks/--eval-tasks is required when --env-kind=real_terrain")
        env = RealTerrainPlanningEnv(
            layers_path=layers_path,
            task_split_path=task_split_path,
            seed=seed,
            scenario=scenario,
            mission_profile_scenario=mission_profile_scenario,
            observation_mode=observation_mode,
            reward_mode=reward_mode,
            reward_scale=reward_scale,
            reward_cost_key=reward_cost_key,
            action_mode=action_mode,
            action_gain=action_gain,
            max_uncertainty_lambda=max_uncertainty_lambda,
            attack_budget_fraction=float(attacker_kwargs.get("attack_budget_fraction", attack_budget_fraction)),
            attack_strength=float(attacker_kwargs.get("attack_strength", attack_strength)),
            attacker_temperature=float(attacker_kwargs.get("attacker_temperature", attacker_temperature)),
            attacker_response=str(attacker_kwargs.get("attacker_response", attacker_response)),
            attacker_top_fraction=float(attacker_kwargs.get("attacker_top_fraction", attacker_top_fraction)),
            attacker_sharpness=float(attacker_kwargs.get("attacker_sharpness", attacker_sharpness)),
        )
    else:
        env = MultiObjectiveCostmapEnv(
            map_size=map_size,
            seed=seed,
            scenario=scenario,
            observation_mode=observation_mode,
            reward_mode=reward_mode,
            reward_scale=reward_scale,
            reward_cost_key=reward_cost_key,
            action_mode=action_mode,
            action_gain=action_gain,
            max_uncertainty_lambda=max_uncertainty_lambda,
            attack_budget_fraction=float(attacker_kwargs.get("attack_budget_fraction", attack_budget_fraction)),
            attack_strength=float(attacker_kwargs.get("attack_strength", attack_strength)),
            attacker_temperature=float(attacker_kwargs.get("attacker_temperature", attacker_temperature)),
            attacker_response=str(attacker_kwargs.get("attacker_response", attacker_response)),
            attacker_top_fraction=float(attacker_kwargs.get("attacker_top_fraction", attacker_top_fraction)),
            attacker_sharpness=float(attacker_kwargs.get("attacker_sharpness", attacker_sharpness)),
            map_sampling_mode=map_sampling_mode,
            fixed_map_seed=fixed_map_seed,
            map_seed_pool_size=map_seed_pool_size,
            min_start_goal_distance_ratio=float(min_start_goal_distance_ratio),
        )
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return wrap_env_with_attacks(
        env,
        observation_attack=observation_attack_config,
        environment_attack=environment_attack_config_for_env,
    )


def reset_envs(envs: list[gym.Env], seed: int) -> np.ndarray:
    observations = []
    for index, env in enumerate(envs):
        obs, _ = env.reset(seed=seed + index)
        observations.append(obs)
    return np.stack(observations).astype(np.float32)


def mra_action_mask(action_dim: int, mode: str) -> torch.Tensor:
    mask = np.ones(int(action_dim), dtype=np.float32)
    if mode == "all":
        return torch.as_tensor(mask, dtype=torch.float32)
    mask[:] = 0.0
    if mode == "objective_only":
        mask[: len(OBJECTIVE_NAMES)] = 1.0
    elif mode == "lambda_only":
        mask[len(OBJECTIVE_NAMES)] = 1.0
    elif mode == "hazard_lambda":
        mask[2] = 1.0
        mask[len(OBJECTIVE_NAMES)] = 1.0
    elif mode == "risk_layers":
        mask[1 : len(OBJECTIVE_NAMES)] = 1.0
        mask[len(OBJECTIVE_NAMES)] = 1.0
    else:
        raise ValueError(f"unknown MRA anchor mask: {mode}")
    return torch.as_tensor(mask, dtype=torch.float32)


def parse_game_action_indices(text: str, action_dim: int) -> list[int]:
    """Parse objective names or integer action indices for game recovery losses."""

    indices: list[int] = []
    for raw_token in str(text or "").split(","):
        token = raw_token.strip().lower()
        if not token:
            continue
        if token in OBJECTIVE_NAMES:
            index = int(OBJECTIVE_NAMES.index(token))
        elif token in {"lambda", "uncertainty", "uncertainty_lambda"}:
            index = len(OBJECTIVE_NAMES)
        else:
            index = int(token)
        if index < 0 or index >= int(action_dim):
            raise ValueError(f"game action index {index} is outside action_dim={action_dim}")
        if index not in indices:
            indices.append(index)
    return indices


def deterministic_policy_action(
    agent: Any,
    obs: torch.Tensor,
    residual_features: torch.Tensor | None = None,
    detach_residual_latent: bool = False,
) -> torch.Tensor:
    if isinstance(agent, CleanRLResidualBeliefActorCritic):
        return agent.get_deterministic_action(
            obs,
            residual_features=residual_features,
            detach_residual_latent=bool(detach_residual_latent),
        )
    return agent.get_deterministic_action(obs)


def policy_distribution(
    agent: Any,
    obs: torch.Tensor,
    residual_features: torch.Tensor | None = None,
    detach_residual_latent: bool = False,
):
    if isinstance(agent, CleanRLResidualBeliefActorCritic):
        return agent.get_dist(
            obs,
            residual_features=residual_features,
            detach_residual_latent=bool(detach_residual_latent),
        )
    return agent.get_dist(obs)


def finite_info_float(info: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = float(info.get(key, default))
    return value if np.isfinite(value) else float(default)


def mra_rollout_risk(info: dict[str, Any], reward_cost_key: str) -> tuple[float, float]:
    if reward_cost_key == "soft_attacked_scalar_cost":
        cost = finite_info_float(
            info,
            "soft_attacked_scalar_cost",
            finite_info_float(info, "attacked_scalar_cost", finite_info_float(info, "scalar_cost", 0.0)),
        )
    elif reward_cost_key == "attacked_scalar_cost":
        cost = finite_info_float(info, "attacked_scalar_cost", finite_info_float(info, "scalar_cost", 0.0))
    else:
        cost = finite_info_float(info, "scalar_cost", 0.0)
    confidence = finite_info_float(info, "mean_path_confidence", 1.0)
    low_confidence = 1.0 - float(np.clip(confidence, 0.0, 1.0))
    feature = (
        finite_info_float(info, "attacked_cell_exposure_ratio", 0.0)
        + finite_info_float(info, "map_mismatch_penalty", 0.0)
        + finite_info_float(info, "belief_hazard_exposure", finite_info_float(info, "hazard_exposure", 0.0))
        + finite_info_float(info, "belief_uncertainty_exposure", finite_info_float(info, "uncertainty_exposure", 0.0))
        + low_confidence
    )
    return float(cost), float(feature)


def normalize_robust_signal(values: torch.Tensor) -> torch.Tensor:
    center = torch.quantile(values.detach(), 0.50)
    q25 = torch.quantile(values.detach(), 0.25)
    q75 = torch.quantile(values.detach(), 0.75)
    scale = torch.clamp(q75 - q25, min=1e-6)
    normalized = (values - center) / scale
    return torch.nan_to_num(normalized, nan=0.0, posinf=5.0, neginf=-5.0)


def mra_batch_weights(
    risk_costs: torch.Tensor,
    risk_features: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not bool(args.mra_enabled):
        ones = torch.ones_like(risk_costs)
        zeros = torch.zeros_like(risk_costs)
        return ones, zeros, zeros, torch.tensor(float("nan"), device=risk_costs.device)

    costs = torch.nan_to_num(risk_costs.float(), nan=0.0, posinf=0.0, neginf=0.0)
    features = torch.nan_to_num(risk_features.float(), nan=0.0, posinf=0.0, neginf=0.0)
    quantile = float(np.clip(args.mra_cvar_quantile, 0.0, 1.0))
    cutoff = torch.quantile(costs.detach(), quantile)
    cost_scale = torch.clamp(costs.detach().std(unbiased=False), min=1e-6)
    tail_indicator = (costs >= cutoff).float()
    tail_excess = torch.clamp((costs - cutoff) / cost_scale, min=0.0)
    feature_signal = torch.clamp(normalize_robust_signal(features), min=0.0)
    weights = (
        1.0
        + float(args.mra_cvar_weight) * tail_indicator
        + float(args.mra_tail_excess_weight) * tail_excess
        + float(args.mra_risk_feature_weight) * feature_signal
    )
    weights = torch.clamp(weights, min=1.0, max=max(float(args.mra_weight_cap), 1.0))
    weights = weights / torch.clamp(weights.mean(), min=1e-6)
    return weights.detach(), tail_excess.detach(), feature_signal.detach(), cutoff.detach()


def parse_int_csv(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for item in str(text or "").replace(",", " ").split():
        if item.strip():
            values.append(int(item))
    return tuple(values)


def pr_cumulative_step(args: argparse.Namespace, local_global_step: int) -> int:
    return int(args.recovery_step_offset) + int(local_global_step)


def pr_alpha_components(
    args: argparse.Namespace,
    local_global_step: int,
    mean_positive_regret: float | None = None,
    alpha_runtime: float | None = None,
) -> dict[str, float | str]:
    alpha_target = float(alpha_runtime if alpha_runtime is not None else getattr(args, "alpha_pr", 0.0))
    cumulative_step = pr_cumulative_step(args, local_global_step)
    if not bool(getattr(args, "pr_enabled", False)) or alpha_target <= 0.0:
        return {
            "pr_alpha_base": 0.0,
            "pr_alpha_eff": 0.0,
            "pr_ramp_mult": 0.0,
            "pr_schedule_mult": 0.0,
            "pr_regret_gate": 0.0,
            "pr_guidance_schedule": str(getattr(args, "pr_guidance_schedule", "constant")),
        }

    start_after = int(getattr(args, "pr_start_after_steps", 0))
    if cumulative_step < start_after:
        ramp_mult = 0.0
    elif int(getattr(args, "pr_ramp_steps", 2048)) <= 0:
        ramp_mult = 1.0
    else:
        ramp_steps = max(int(getattr(args, "pr_ramp_steps", 2048)), 1)
        ramp_mult = min(1.0, max(0.0, (cumulative_step - start_after) / float(ramp_steps)))

    schedule = str(getattr(args, "pr_guidance_schedule", "constant"))
    schedule_mult = 1.0
    if schedule == "early_decay":
        active_until = int(getattr(args, "pr_active_until_step", 2048))
        decay_to_zero_by = int(getattr(args, "pr_decay_to_zero_by_step", 3072))
        if cumulative_step <= active_until:
            schedule_mult = 1.0
        elif cumulative_step >= decay_to_zero_by:
            schedule_mult = 0.0
        else:
            denom = max(float(decay_to_zero_by - active_until), 1.0)
            schedule_mult = 1.0 - (float(cumulative_step - active_until) / denom)
            schedule_mult = float(np.clip(schedule_mult, 0.0, 1.0))

    regret_gate = 1.0
    if bool(getattr(args, "pr_regret_adaptive_gate", False)):
        regret = 0.0 if mean_positive_regret is None else float(mean_positive_regret)
        low = float(getattr(args, "pr_regret_low", 0.005))
        high = float(getattr(args, "pr_regret_high", 0.02))
        regret_gate = float(np.clip((regret - low) / max(high - low, 1e-8), 0.0, 1.0))

    alpha_base = float(alpha_target * ramp_mult)
    alpha_eff = float(alpha_base * schedule_mult * regret_gate)
    return {
        "pr_alpha_base": alpha_base,
        "pr_alpha_eff": alpha_eff,
        "pr_ramp_mult": float(ramp_mult),
        "pr_schedule_mult": float(schedule_mult),
        "pr_regret_gate": float(regret_gate),
        "pr_guidance_schedule": schedule,
    }


def pr_alpha_eff(args: argparse.Namespace, local_global_step: int) -> float:
    return float(pr_alpha_components(args, local_global_step)["pr_alpha_eff"])


def trr_alpha_components(args: argparse.Namespace, local_global_step: int) -> dict[str, float | str]:
    cumulative_step = pr_cumulative_step(args, local_global_step)
    alpha_target = float(getattr(args, "trr_alpha", 0.0))
    if not bool(getattr(args, "trr_enabled", False)) or alpha_target <= 0.0:
        return {
            "trr_alpha_eff": 0.0,
            "trr_ramp_mult": 0.0,
            "trr_schedule_mult": 0.0,
            "trr_guidance_schedule": str(getattr(args, "trr_guidance_schedule", "constant")),
        }

    ramp_steps = int(getattr(args, "trr_ramp_steps", 1024))
    if ramp_steps <= 0:
        ramp_mult = 1.0
    else:
        ramp_mult = float(np.clip(cumulative_step / max(float(ramp_steps), 1.0), 0.0, 1.0))

    schedule = str(getattr(args, "trr_guidance_schedule", "constant"))
    schedule_mult = 1.0
    if schedule == "early_decay":
        active_until = int(getattr(args, "trr_active_until_step", 4096))
        decay_to_zero_by = int(getattr(args, "trr_decay_to_zero_by_step", 12288))
        if cumulative_step <= active_until:
            schedule_mult = 1.0
        elif cumulative_step >= decay_to_zero_by:
            schedule_mult = 0.0
        else:
            denom = max(float(decay_to_zero_by - active_until), 1.0)
            schedule_mult = 1.0 - (float(cumulative_step - active_until) / denom)
            schedule_mult = float(np.clip(schedule_mult, 0.0, 1.0))

    return {
        "trr_alpha_eff": float(alpha_target * ramp_mult * schedule_mult),
        "trr_ramp_mult": float(ramp_mult),
        "trr_schedule_mult": float(schedule_mult),
        "trr_guidance_schedule": schedule,
    }


def acbr_context_array(features: np.ndarray | list[float] | None, context_dim: int) -> np.ndarray:
    """Pad/truncate residual belief features into the fixed ACBR context vector."""

    dim = max(int(context_dim), 1)
    if features is None:
        return np.zeros(dim, dtype=np.float32)
    array = np.asarray(features, dtype=np.float32).reshape(-1)
    if array.size >= dim:
        return array[:dim].astype(np.float32)
    out = np.zeros(dim, dtype=np.float32)
    out[: array.size] = array
    return out


def acbr_context_tensor(features: torch.Tensor | None, context_dim: int, device: torch.device) -> torch.Tensor:
    dim = max(int(context_dim), 1)
    if features is None or features.numel() == 0:
        return torch.zeros((0, dim), dtype=torch.float32, device=device)
    features = features.to(dtype=torch.float32, device=device)
    if features.dim() == 1:
        features = features.unsqueeze(0)
    if features.shape[-1] >= dim:
        return features[..., :dim]
    return F.pad(features, (0, dim - features.shape[-1]))


def acbr_normalized_cost_targets(
    candidate_costs: torch.Tensor,
    policy_costs: torch.Tensor,
    clip_value: float,
) -> torch.Tensor:
    denom = torch.clamp(policy_costs.abs(), min=1e-6).unsqueeze(1)
    targets = (candidate_costs - policy_costs.unsqueeze(1)) / denom
    return torch.clamp(torch.nan_to_num(targets, nan=0.0, posinf=clip_value, neginf=-clip_value), -clip_value, clip_value)


def acbr_rerank_candidates(
    agent: Any,
    obs_np: np.ndarray,
    candidates: np.ndarray,
    context_np: np.ndarray,
    policy_action: np.ndarray,
    anchor_action: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fallback_action: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    candidate_array = np.asarray(candidates, dtype=np.float32)
    fallback_array = np.asarray(
        policy_action if fallback_action is None else fallback_action,
        dtype=np.float32,
    )
    if candidate_array.ndim != 2 or candidate_array.shape[0] == 0:
        return fallback_array, {
            "selected_index": 0,
            "changed": 0.0,
            "score": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "gated": 0.0,
            "predicted_improvement": float("nan"),
            "policy_score": float("nan"),
        }
    obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
    obs_batch = obs_tensor.expand(candidate_array.shape[0], -1)
    action_tensor = torch.as_tensor(candidate_array, dtype=torch.float32, device=device)
    context_tensor = torch.as_tensor(context_np, dtype=torch.float32, device=device).unsqueeze(0)
    context_batch = context_tensor.expand(candidate_array.shape[0], -1)
    with torch.no_grad():
        critic_values = agent.predict_acbr_costs(obs_batch, action_tensor, context_batch)
        mean = critic_values.mean(dim=-1)
        std = critic_values.std(dim=-1, unbiased=False)
        policy_tensor = torch.as_tensor(policy_action, dtype=torch.float32, device=device).unsqueeze(0)
        anchor_tensor = torch.as_tensor(anchor_action, dtype=torch.float32, device=device).unsqueeze(0)
        anchor_penalty = (action_tensor - anchor_tensor).pow(2).mean(dim=-1)
        policy_penalty = (action_tensor - policy_tensor).pow(2).mean(dim=-1)
        score = (
            mean
            + float(args.acbr_uncertainty_coef) * std
            + float(args.acbr_anchor_penalty) * anchor_penalty
            + float(args.acbr_policy_penalty) * policy_penalty
        )
        selected_index = int(torch.argmin(score).item())
        policy_index = int(
            torch.argmin((action_tensor - policy_tensor).pow(2).sum(dim=-1)).item()
        )
        policy_score = score[policy_index]
        predicted_improvement = policy_score - score[selected_index]
        gated = bool(getattr(args, "acbr_benefit_gate_enabled", False)) and (
            selected_index == policy_index
            or float(predicted_improvement.detach().cpu().item()) < float(getattr(args, "acbr_benefit_margin", 0.0))
        )
        if gated:
            selected_index = -1
    selected = fallback_array if selected_index < 0 else candidate_array[selected_index].astype(np.float32)
    changed = float(np.linalg.norm(selected - fallback_array) > 1e-6)
    report_index = int(policy_index if selected_index < 0 else selected_index)
    return selected, {
        "selected_index": report_index,
        "changed": changed,
        "score": float(score[report_index].detach().cpu().item()),
        "mean": float(mean[report_index].detach().cpu().item()),
        "std": float(std[report_index].detach().cpu().item()),
        "gated": float(gated),
        "predicted_improvement": float(predicted_improvement.detach().cpu().item()),
        "policy_score": float(policy_score.detach().cpu().item()),
    }


def actor_parameters(agent: CleanRLActorCritic) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    modules = [agent.actor_body, agent.alpha_head, agent.beta_head]
    if isinstance(agent, CleanRLResidualBeliefActorCritic):
        modules.extend([agent.residual_encoder, agent.alpha_residual, agent.beta_residual])
    for module in modules:
        params.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
    return params


def prb_zero_action_value(action_mode: str) -> float:
    return 0.5 if action_mode == "preference_delta" else 0.5


def prb_probe_features_for_envs(
    envs: list[gym.Env],
    builder: PlannerResidualFeatureBuilder,
    action_dim: int,
    reward_cost_key: str,
    action_mode: str,
    environment_attack_config: dict[str, Any] | None = None,
    bagr_enabled: bool = False,
    bagr_max_attack_variants: int = 6,
    bagr_belief_temperature: float = 0.25,
    bagr_belief_prior_mix: float = 0.10,
    failure_cost: float = 1e6,
    rng: np.random.Generator | None = None,
    random_features: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    features = []
    records = []
    probe_action = neutral_probe_action(action_dim, prb_zero_action_value(action_mode))
    for env in envs:
        if random_features:
            feature = (rng if rng is not None else np.random.default_rng()).normal(
                0.0,
                1.0,
                size=builder.feature_dim,
            ).astype(np.float32)
            features.append(feature)
            records.append({"component_cost_available": False, "probe_failure": False})
            continue
        try:
            evaluator = env if hasattr(env, "evaluate_action") else getattr(env, "env", env)
            result = evaluator.evaluate_action(probe_action)
            record = builder.build_from_info(result, probe_action, reward_cost_key).raw
            if bool(bagr_enabled):
                record = attack_belief_record(
                    env,
                    record,
                    probe_action,
                    environment_attack_config,
                    reward_cost_key,
                    max_attack_variants=int(bagr_max_attack_variants),
                    temperature=float(bagr_belief_temperature),
                    prior_mix=float(bagr_belief_prior_mix),
                    failure_cost=float(failure_cost),
                    rng=rng,
                )
            batch = builder.build(record)
            features.append(batch.features)
            records.append(
                {
                    "component_cost_available": bool(batch.component_cost_available),
                    "probe_failure": False,
                    "mean_planner_predicted_cost": float(batch.raw.get("planner_predicted_total_cost", np.nan)),
                    "mean_true_attacked_cost": float(batch.raw.get("true_attacked_total_cost", np.nan)),
                    "mean_residual_total": float(batch.raw.get("residual_total_cost", np.nan)),
                    "mean_residual_norm": float(batch.raw.get("residual_total_norm", np.nan)),
                    "attack_belief_confidence": float(batch.raw.get("attack_belief_confidence", np.nan)),
                    "attack_belief_entropy_norm": float(batch.raw.get("attack_belief_entropy_norm", np.nan)),
                    "attack_belief_variant_ids": str(batch.raw.get("attack_belief_variant_ids", "")),
                    "attack_belief_failure": bool(batch.raw.get("attack_belief_failure", False)),
                }
            )
        except Exception as exc:
            features.append(builder.zero_features())
            records.append({"component_cost_available": False, "probe_failure": True, "probe_failure_reason": str(exc)})
    return np.stack(features).astype(np.float32), records


def prb_targets_from_infos(
    infos: list[dict[str, Any]],
    actions_np: np.ndarray,
    builder: PlannerResidualFeatureBuilder,
    reward_cost_key: str,
) -> dict[str, np.ndarray | list[dict[str, Any]]]:
    residual_total = []
    true_total = []
    component_residual = []
    true_component = []
    component_mask = []
    records = []
    for info, action in zip(infos, actions_np):
        batch = builder.build_from_info(info, action, reward_cost_key)
        residual_total.append(batch.residual_total_target)
        true_total.append(batch.true_total_target)
        component_residual.append(batch.component_residual_target)
        true_component.append(batch.true_component_target)
        component_mask.append(batch.component_mask)
        records.append(batch.raw | {"component_cost_available": bool(batch.component_cost_available)})
    return {
        "residual_total": np.asarray(residual_total, dtype=np.float32),
        "true_total": np.asarray(true_total, dtype=np.float32),
        "component_residual": np.asarray(component_residual, dtype=np.float32),
        "true_component": np.asarray(true_component, dtype=np.float32),
        "component_mask": np.asarray(component_mask, dtype=np.float32),
        "records": records,
    }


def grad_norm_for_loss(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    retain_graph: bool = True,
) -> float:
    if not parameters or not torch.isfinite(loss.detach()):
        return float("nan")
    if not bool(loss.requires_grad):
        return 0.0
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    total = torch.tensor(0.0, dtype=torch.float32, device=loss.device)
    for grad in grads:
        if grad is None:
            continue
        total = total + grad.detach().pow(2).sum()
    return float(torch.sqrt(total).item())


def step_envs(
    envs: list[gym.Env],
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Step all one-step envs and immediately reset completed episodes."""

    next_observations = []
    rewards = []
    dones = []
    infos = []

    for env, action in zip(envs, actions):
        _, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        next_obs, reset_info = env.reset()
        info["reset_info"] = reset_info

        next_observations.append(next_obs)
        rewards.append(float(reward))
        dones.append(done)
        infos.append(info)

    return (
        np.stack(next_observations).astype(np.float32),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(dones, dtype=np.float32),
        infos,
    )


def evaluate_agent(
    agent: CleanRLActorCritic,
    map_size: int,
    scenario: str,
    seed: int,
    num_episodes: int,
    device: torch.device,
    observation_mode: str,
    reward_mode: str,
    reward_scale: float,
    reward_cost_key: str,
    action_mode: str,
    action_gain: float,
    max_uncertainty_lambda: float,
    attack_budget_fraction: float,
    attack_strength: float,
    map_sampling_mode: str,
    fixed_map_seed: int,
    map_seed_pool_size: int,
    min_start_goal_distance_ratio: float = 0.55,
    attacker_temperature: float = DEFAULT_ATTACKER_TEMPERATURE,
    attacker_response: str = DEFAULT_ATTACKER_RESPONSE,
    attacker_top_fraction: float = DEFAULT_ATTACKER_TOP_FRACTION,
    attacker_sharpness: float = DEFAULT_ATTACKER_SHARPNESS,
    observation_attack_config: dict[str, Any] | None = None,
    environment_attack_config: dict[str, Any] | None = None,
    env_kind: str = "synthetic",
    layers_path: str | None = None,
    eval_tasks: str | None = None,
    train_tasks: str | None = None,
    mission_profile_scenario: str | None = None,
    bagr_enabled: bool = False,
    bagr_max_attack_variants: int = 6,
    bagr_belief_temperature: float = 0.25,
    bagr_belief_prior_mix: float = 0.10,
    pr_failure_cost: float = 1e6,
    acbr_enabled: bool = False,
    acbr_anchor_agent: Any | None = None,
    acbr_context_dim: int = 16,
    acbr_num_candidates: int = 24,
    acbr_num_random_candidates: int = 6,
    acbr_num_structured_candidates: int = 10,
    acbr_local_sigma: float = 0.12,
    acbr_risk_local_sigma: float = 0.20,
    acbr_uncertainty_coef: float = 0.25,
    acbr_anchor_penalty: float = 0.15,
    acbr_policy_penalty: float = 0.05,
    acbr_benefit_gate_enabled: bool = False,
    acbr_benefit_margin: float = 0.0,
) -> dict[str, float]:
    env = make_env(
        map_size,
        seed,
        scenario,
        observation_mode,
        reward_mode,
        reward_scale,
        reward_cost_key,
        action_mode,
        action_gain,
        max_uncertainty_lambda,
        attack_budget_fraction,
        attack_strength,
        map_sampling_mode,
        fixed_map_seed,
        map_seed_pool_size,
        min_start_goal_distance_ratio=min_start_goal_distance_ratio,
        attacker_temperature=attacker_temperature,
        attacker_response=attacker_response,
        attacker_top_fraction=attacker_top_fraction,
        attacker_sharpness=attacker_sharpness,
        observation_attack_config=observation_attack_config,
        environment_attack_config=environment_attack_config,
        env_kind=env_kind,
        layers_path=layers_path,
        task_split_path=eval_tasks or train_tasks,
        mission_profile_scenario=mission_profile_scenario,
    )
    rewards = []
    successes = []
    scalar_costs = []
    attacked_scalar_costs = []
    lambdas = []
    prb_builder: PlannerResidualFeatureBuilder | None = None
    if isinstance(agent, CleanRLResidualBeliefActorCritic):
        prb_builder = PlannerResidualFeatureBuilder(
            PlannerResidualFeatureConfig(
                action_dim=agent.action_dim,
                normalize_features=True,
                feature_clip=5.0,
                use_component_costs=True,
                use_scalar_cost=True,
                use_attack_belief=bool(bagr_enabled),
                attack_belief_dim=int(bagr_max_attack_variants) if bool(bagr_enabled) else 0,
            )
        )

    for episode_index in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode_index)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        residual_tensor = None
        if prb_builder is not None:
            feature_np, _ = prb_probe_features_for_envs(
                [env],
                prb_builder,
                agent.action_dim,
                reward_cost_key,
                action_mode,
                environment_attack_config=environment_attack_config,
                bagr_enabled=bool(bagr_enabled),
                bagr_max_attack_variants=int(bagr_max_attack_variants),
                bagr_belief_temperature=float(bagr_belief_temperature),
                bagr_belief_prior_mix=float(bagr_belief_prior_mix),
                failure_cost=float(pr_failure_cost),
            )
            residual_tensor = torch.as_tensor(feature_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            if residual_tensor is not None:
                action = agent.get_deterministic_action(obs_tensor, residual_tensor).squeeze(0).cpu().numpy()
            else:
                action = agent.get_deterministic_action(obs_tensor).squeeze(0).cpu().numpy()
            if bool(acbr_enabled) and hasattr(agent, "predict_acbr_costs"):
                if acbr_anchor_agent is not None:
                    anchor_action = deterministic_policy_action(acbr_anchor_agent, obs_tensor).squeeze(0).cpu().numpy()
                else:
                    anchor_action = action
                low = np.asarray(env.action_space.low, dtype=np.float32)
                high = np.asarray(env.action_space.high, dtype=np.float32)
                candidate_config = CandidateActionConfig(
                    num_candidates=int(acbr_num_candidates),
                    local_sigma=float(acbr_local_sigma),
                    num_random_candidates=int(acbr_num_random_candidates),
                    include_policy_action=True,
                    include_nominal_action=True,
                    include_zero_delta=True,
                    include_axis_perturbations=True,
                    risk_local_sigma=float(acbr_risk_local_sigma),
                    include_risk_axis_perturbations=True,
                    include_risk_block_perturbations=True,
                    zero_action_value=0.5 if action_mode == "preference_delta" else 0.0,
                )
                candidates, candidate_meta = generate_candidate_actions(
                    action,
                    anchor_action,
                    low,
                    high,
                    candidate_config,
                    np.random.default_rng(seed + 900_000 + episode_index),
                )
                if int(acbr_num_structured_candidates) > 0:
                    structured_candidates, structured_meta = generate_structured_candidate_actions(
                        env,
                        low,
                        high,
                        int(acbr_num_structured_candidates),
                        action_mode=action_mode,
                        action_gain=float(action_gain),
                        max_uncertainty_lambda=float(max_uncertainty_lambda),
                        dedup_tol=float(candidate_config.dedup_tol),
                    )
                    if structured_candidates.size > 0:
                        candidates, candidate_meta = merge_candidate_action_sets(
                            candidates,
                            candidate_meta,
                            structured_candidates,
                            structured_meta,
                            low,
                            high,
                            dedup_tol=float(candidate_config.dedup_tol),
                        )
                context_np = acbr_context_array(
                    residual_tensor.squeeze(0).detach().cpu().numpy() if residual_tensor is not None else None,
                    int(acbr_context_dim),
                )
                selected_action, _diag = acbr_rerank_candidates(
                    agent,
                    obs,
                    candidates,
                    context_np,
                    action,
                    anchor_action,
                    argparse.Namespace(
                        acbr_uncertainty_coef=float(acbr_uncertainty_coef),
                        acbr_anchor_penalty=float(acbr_anchor_penalty),
                        acbr_policy_penalty=float(acbr_policy_penalty),
                        acbr_benefit_gate_enabled=bool(acbr_benefit_gate_enabled),
                        acbr_benefit_margin=float(acbr_benefit_margin),
                    ),
                    device,
                )
                action = selected_action
        _, reward, _, _, info = env.step(action)
        rewards.append(float(reward))
        successes.append(float(info["success"]))
        scalar_costs.append(float(info["scalar_cost"]))
        attacked_scalar_costs.append(float(info["attacked_scalar_cost"]))
        lambdas.append(float(info["lambda_uncertainty"]))

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_scalar_cost": float(np.mean(scalar_costs)),
        "mean_attacked_scalar_cost": float(np.mean(attacked_scalar_costs)),
        "mean_lambda_uncertainty": float(np.mean(lambdas)),
    }


def heuristic_target_from_obs(obs: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Warm-start toward neutral preference deltas and mild uncertainty caution."""

    target = torch.full((obs.shape[0], action_dim), 0.5, dtype=obs.dtype, device=obs.device)
    if action_dim > 5:
        target[:, 5] = 0.25
    return target


def behavior_cloning_warmstart(
    agent: CleanRLActorCritic,
    optimizer: optim.Optimizer,
    envs: list[gym.Env],
    steps: int,
    seed: int,
    device: torch.device,
    writer: SummaryWriter,
) -> None:
    """Optionally initialize the actor to the mission-priority heuristic."""

    if steps <= 0:
        return

    for step in range(steps):
        obs_np = reset_envs(envs, seed + 100_000 + step * len(envs))
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        target = heuristic_target_from_obs(obs, agent.action_dim)
        prediction = agent.get_deterministic_action(obs)
        loss = torch.mean((prediction - target) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()

        if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
            writer.add_scalar("warmstart/bc_loss", loss.item(), step + 1)
            print(f"warmstart_step={step + 1} bc_loss={loss.item():.6f}")


def main() -> None:
    args = parse_args()
    if bool(args.trr_enabled):
        if bool(args.bagr_enabled):
            raise ValueError("--trr-enabled cannot be combined with --bagr-enabled")
        if bool(args.mra_enabled):
            raise ValueError("--trr-enabled cannot be combined with AP-CVaR/MRA")
        if not args.init_checkpoint:
            raise ValueError("--trr-enabled is recovery-only and requires --init-checkpoint")
        args.prb_enabled = True
        args.pr_enabled = True
        args.pr_enable_queries = True
        args.pr_include_policy_action = True
        args.pr_include_nominal_action = True
        args.pr_include_structured_candidates = True
        args.prb_stopgrad_residual_latent = False
        if not args.trr_anchor_checkpoint:
            args.trr_anchor_checkpoint = args.init_checkpoint
        if float(args.trr_alpha) < 0.0:
            raise ValueError("--trr-alpha must be non-negative")
        if int(args.trr_ramp_steps) < 0:
            raise ValueError("--trr-ramp-steps must be non-negative")
        if int(args.trr_decay_to_zero_by_step) < int(args.trr_active_until_step):
            raise ValueError("--trr-decay-to-zero-by-step must be >= --trr-active-until-step")
        if float(args.trr_min_normalized_regret) < 0.0:
            raise ValueError("--trr-min-normalized-regret must be non-negative")
        if float(args.trr_residual_l2_coef) < 0.0 or float(args.trr_residual_barrier_coef) < 0.0:
            raise ValueError("TRR residual coefficients must be non-negative")
        if float(args.trr_residual_action_limit) < 0.0:
            raise ValueError("--trr-residual-action-limit must be non-negative")
    if bool(args.bagr_enabled):
        if bool(args.pr_enabled) or bool(args.pr_game_teacher_enabled):
            raise ValueError("--bagr-enabled is a standalone recovery method and cannot be combined with PR/game-teacher losses")
        if bool(args.mra_enabled):
            raise ValueError("--bagr-enabled is a standalone recovery method and cannot be combined with AP-CVaR/MRA")
        args.prb_enabled = True
        args.prb_stopgrad_residual_latent = True
        if not args.init_checkpoint:
            raise ValueError("--bagr-enabled is recovery-only and requires --init-checkpoint")
        if int(args.bagr_max_attack_variants) <= 0:
            raise ValueError("--bagr-max-attack-variants must be positive")
        if float(args.bagr_belief_temperature) <= 0.0:
            raise ValueError("--bagr-belief-temperature must be positive")
        if not 0.0 <= float(args.bagr_belief_prior_mix) <= 1.0:
            raise ValueError("--bagr-belief-prior-mix must be between 0 and 1")
        if float(args.bagr_residual_l2_coef) < 0.0 or float(args.bagr_residual_barrier_coef) < 0.0:
            raise ValueError("BAGR residual coefficients must be non-negative")
        if float(args.bagr_residual_action_limit) < 0.0:
            raise ValueError("--bagr-residual-action-limit must be non-negative")
        if float(args.bagr_confidence_limit_scale) < 0.0:
            raise ValueError("--bagr-confidence-limit-scale must be non-negative")
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.num_envs <= 0 or args.num_steps <= 0:
        raise ValueError("--num-envs and --num-steps must be positive")
    if args.num_minibatches <= 0 or args.update_epochs <= 0:
        raise ValueError("--num-minibatches and --update-epochs must be positive")
    if bool(args.pr_enabled) and str(args.pr_aux_loss_type) in {"cpa", "pairwise_pref"} and not bool(args.pr_store_candidates):
        raise ValueError("--pr-store-candidates must be true for cpa or pairwise_pref PR losses")
    if bool(args.pr_game_teacher_enabled) and not bool(args.pr_enabled):
        raise ValueError("--pr-game-teacher-enabled requires --pr-enabled")
    if int(args.pr_game_max_attack_variants) <= 0:
        raise ValueError("--pr-game-max-attack-variants must be positive")
    if int(args.pr_num_structured_candidates) < 0:
        raise ValueError("--pr-num-structured-candidates must be non-negative")
    if float(args.game_lambda_drift_margin) < 0.0 or float(args.game_risk_drift_margin) < 0.0:
        raise ValueError("game recovery drift margins must be non-negative")
    if bool(args.acbr_enabled):
        if bool(args.trr_enabled) or bool(args.bagr_enabled) or bool(args.mra_enabled):
            raise ValueError("--acbr-enabled cannot be combined with TRR, BAGR, or AP-CVaR/MRA")
        if not args.init_checkpoint:
            raise ValueError("--acbr-enabled is recovery-only and requires --init-checkpoint")
        if not args.acbr_anchor_checkpoint:
            args.acbr_anchor_checkpoint = args.init_checkpoint
        if float(args.acbr_critic_coef) <= 0.0:
            raise ValueError("--acbr-critic-coef must be positive when ACBR is enabled")
        if float(args.acbr_uncertainty_coef) < 0.0:
            raise ValueError("--acbr-uncertainty-coef must be non-negative")
        if float(args.acbr_anchor_penalty) < 0.0 or float(args.acbr_policy_penalty) < 0.0:
            raise ValueError("ACBR rerank penalties must be non-negative")
        if float(args.acbr_target_clip) <= 0.0:
            raise ValueError("--acbr-target-clip must be positive")
        if float(args.acbr_benefit_margin) < 0.0:
            raise ValueError("--acbr-benefit-margin must be non-negative")
        if int(args.acbr_context_dim) <= 0 or int(args.acbr_hidden_dim) <= 0 or int(args.acbr_ensemble_size) <= 0:
            raise ValueError("ACBR critic dimensions must be positive")
        if int(args.acbr_min_query_states) < 0:
            raise ValueError("--acbr-min-query-states must be non-negative")
        args.pr_enabled = True
        args.pr_enable_queries = True
        args.prb_enabled = True

    set_global_seeds(args.seed)
    device = resolve_device(args.device)
    observation_attack_config = load_attack_config(args.observation_attack_config)
    environment_attack_config = load_attack_config(args.environment_attack_config)

    batch_size = args.num_envs * args.num_steps
    minibatch_size = max(batch_size // args.num_minibatches, 1)
    num_updates = max(args.total_timesteps // batch_size, 1)

    run_name = f"cleanrl_ppo_costmap_seed{args.seed}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(run_dir / "tensorboard"))

    config = vars(args).copy()
    config.update(
        {
            "algorithm": (
                "acbr_ppo"
                if bool(args.acbr_enabled)
                else (
                    "trr_ppo"
                    if bool(args.trr_enabled)
                    else (
                        "bagr_ppo"
                        if bool(args.bagr_enabled)
                        else ("prb_ppo" if bool(args.prb_enabled) else "cleanrl_style_ppo_beta_policy")
                    )
                )
            ),
            "batch_size": batch_size,
            "minibatch_size": minibatch_size,
            "num_updates": num_updates,
            "device": str(device),
            "run_dir": str(run_dir),
            "observation_attack": observation_attack_config,
            "environment_attack": environment_attack_config,
        }
    )
    print(json.dumps(config, indent=2))
    reward_cost_source = "nominal" if args.reward_cost_key == "scalar_cost" else "attacked"
    if (
        bool(environment_attack_config.get("enabled", False))
        and str(environment_attack_config.get("type", "env_zscore_topk")) != "env_zscore_topk"
        and args.reward_cost_key == "scalar_cost"
    ):
        attack_type = str(environment_attack_config.get("type", ""))
        if attack_type in {
            "env_belief_mismatch",
            "env_spatial_belief_mismatch",
            "env_traversability_boundary_mismatch",
            "env_composite",
        }:
            reward_cost_source = "true_after_belief_mismatch"
        else:
            # Layer-mutating environmental attacks make scalar_cost refer to
            # the attacked map even though the metric key is still scalar_cost.
            reward_cost_source = "attacked"
    print("Reward cost-source check:")
    print(f"  reward_mode={args.reward_mode}")
    print(f"  reward_scale={args.reward_scale}")
    print(f"  environment_attack.enabled={bool(environment_attack_config.get('enabled', False))}")
    print(f"  environment_attack.type={environment_attack_config.get('type', 'none')}")
    print(
        "  environment_attack.reward_uses_attacked_cost="
        f"{bool(environment_attack_config.get('reward_uses_attacked_cost', False))}"
    )
    print(f"  actual reward cost source: {reward_cost_source}")
    print(f"  reward_cost_key={args.reward_cost_key}")
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
    prb_builder: PlannerResidualFeatureBuilder | None = None
    if bool(args.prb_enabled):
        prb_builder = PlannerResidualFeatureBuilder(
            PlannerResidualFeatureConfig(
                action_dim=action_dim,
                normalize_features=bool(args.prb_normalize_features),
                feature_clip=float(args.prb_feature_clip),
                use_component_costs=bool(args.prb_use_component_costs),
                use_scalar_cost=bool(args.prb_use_scalar_cost),
                use_attack_belief=bool(args.bagr_enabled),
                attack_belief_dim=int(args.bagr_max_attack_variants) if bool(args.bagr_enabled) else 0,
            )
        )

    if args.init_checkpoint:
        loaded_agent, checkpoint = load_cleanrl_agent(args.init_checkpoint, device=device)
        if int(getattr(loaded_agent, "obs_dim", -1)) != obs_dim:
            raise ValueError(
                f"init checkpoint obs_dim={getattr(loaded_agent, 'obs_dim', None)} "
                f"does not match env obs_dim={obs_dim}"
            )
        if int(getattr(loaded_agent, "action_dim", -1)) != action_dim:
            raise ValueError(
                f"init checkpoint action_dim={getattr(loaded_agent, 'action_dim', None)} "
                f"does not match env action_dim={action_dim}"
            )
        if bool(args.prb_enabled):
            if isinstance(loaded_agent, CleanRLResidualBeliefActorCritic):
                agent = loaded_agent.to(device)
            else:
                if prb_builder is None:
                    raise RuntimeError("PRB feature builder was not initialized")
                agent = CleanRLResidualBeliefActorCritic(
                    obs_dim,
                    action_dim,
                    hidden_size=int(getattr(loaded_agent, "hidden_size", args.hidden_size)),
                    prb_feature_dim=prb_builder.feature_dim,
                    prb_latent_dim=int(args.prb_latent_dim),
                    prb_hidden_dim=int(args.prb_hidden_dim),
                    prb_encoder_type=str(args.prb_encoder_type),
                    prb_component_dim=prb_builder.component_dim,
                    acbr_context_dim=int(args.acbr_context_dim),
                    acbr_hidden_dim=int(args.acbr_hidden_dim),
                    acbr_ensemble_size=int(args.acbr_ensemble_size),
                ).to(device)
                agent.copy_base_policy_from(loaded_agent)
        else:
            agent = loaded_agent.to(device)
        agent.train()
        print(
            f"Initialized policy from {args.init_checkpoint} "
            f"(checkpoint_step={checkpoint.get('global_step', 'unknown')})"
        )
    else:
        if bool(args.prb_enabled):
            if prb_builder is None:
                raise RuntimeError("PRB feature builder was not initialized")
            agent = CleanRLResidualBeliefActorCritic(
                obs_dim,
                action_dim,
                hidden_size=args.hidden_size,
                prb_feature_dim=prb_builder.feature_dim,
                prb_latent_dim=int(args.prb_latent_dim),
                prb_hidden_dim=int(args.prb_hidden_dim),
                prb_encoder_type=str(args.prb_encoder_type),
                prb_component_dim=prb_builder.component_dim,
                acbr_context_dim=int(args.acbr_context_dim),
                acbr_hidden_dim=int(args.acbr_hidden_dim),
                acbr_ensemble_size=int(args.acbr_ensemble_size),
            ).to(device)
        else:
            agent = CleanRLActorCritic(
                obs_dim,
                action_dim,
                hidden_size=args.hidden_size,
                acbr_context_dim=int(args.acbr_context_dim),
                acbr_hidden_dim=int(args.acbr_hidden_dim),
                acbr_ensemble_size=int(args.acbr_ensemble_size),
            ).to(device)
    anchor_agent: CleanRLActorCritic | None = None
    if bool(args.mra_enabled) and float(args.mra_anchor_coef) > 0.0:
        anchor_agent = copy.deepcopy(agent).to(device)
        anchor_agent.eval()
        for parameter in anchor_agent.parameters():
            parameter.requires_grad_(False)
        print(
            "MRA-PPO enabled with nominal action anchor: "
            f"coef={args.mra_anchor_coef} mask={args.mra_anchor_mask}"
        )
    mirror_anchor_agent: CleanRLActorCritic | None = None
    if float(args.mirror_nominal_prior_coef) > 0.0:
        mirror_anchor_agent = copy.deepcopy(agent).to(device)
        mirror_anchor_agent.eval()
        for parameter in mirror_anchor_agent.parameters():
            parameter.requires_grad_(False)
        print(f"MIRROR nominal prior enabled: coef={args.mirror_nominal_prior_coef}")
    game_regularizer_active = bool(args.game_recovery_enabled) and (
        float(args.game_nominal_prior_coef) > 0.0
        or float(args.game_lambda_drift_coef) > 0.0
        or float(args.game_risk_drift_coef) > 0.0
    )
    game_anchor_agent: Any | None = None
    game_risk_indices = (
        parse_game_action_indices(args.game_risk_action_indices, action_dim)
        if game_regularizer_active and float(args.game_risk_drift_coef) > 0.0
        else []
    )
    game_risk_index_tensor = torch.as_tensor(game_risk_indices, dtype=torch.long, device=device)
    game_lambda_index = len(OBJECTIVE_NAMES) if action_dim > len(OBJECTIVE_NAMES) else None
    if game_regularizer_active:
        if args.game_anchor_checkpoint:
            loaded_anchor, anchor_checkpoint = load_cleanrl_agent(args.game_anchor_checkpoint, device=device)
            if int(getattr(loaded_anchor, "obs_dim", -1)) != obs_dim:
                raise ValueError(
                    f"game anchor checkpoint obs_dim={getattr(loaded_anchor, 'obs_dim', None)} "
                    f"does not match env obs_dim={obs_dim}"
                )
            if int(getattr(loaded_anchor, "action_dim", -1)) != action_dim:
                raise ValueError(
                    f"game anchor checkpoint action_dim={getattr(loaded_anchor, 'action_dim', None)} "
                    f"does not match env action_dim={action_dim}"
                )
            game_anchor_agent = loaded_anchor.to(device)
            print(
                "Game recovery anchor loaded: "
                f"{args.game_anchor_checkpoint} checkpoint_step={anchor_checkpoint.get('global_step', 'unknown')}"
            )
        else:
            game_anchor_agent = copy.deepcopy(agent).to(device)
            print("Game recovery anchor copied from initialized policy")
        game_anchor_agent.eval()
        for parameter in game_anchor_agent.parameters():
            parameter.requires_grad_(False)
        print(
            "Game recovery regularizer enabled: "
            f"nominal_prior={args.game_nominal_prior_coef} "
            f"lambda_drift={args.game_lambda_drift_coef}@{args.game_lambda_drift_margin} "
            f"risk_drift={args.game_risk_drift_coef}@{args.game_risk_drift_margin} "
            f"risk_indices={game_risk_indices}"
        )
    trr_anchor_agent: Any | None = None
    if bool(args.trr_enabled):
        anchor_path = str(args.trr_anchor_checkpoint or args.init_checkpoint)
        loaded_anchor, anchor_checkpoint = load_cleanrl_agent(anchor_path, device=device)
        if int(getattr(loaded_anchor, "obs_dim", -1)) != obs_dim:
            raise ValueError(
                f"TRR anchor checkpoint obs_dim={getattr(loaded_anchor, 'obs_dim', None)} "
                f"does not match env obs_dim={obs_dim}"
            )
        if int(getattr(loaded_anchor, "action_dim", -1)) != action_dim:
            raise ValueError(
                f"TRR anchor checkpoint action_dim={getattr(loaded_anchor, 'action_dim', None)} "
                f"does not match env action_dim={action_dim}"
            )
        trr_anchor_agent = loaded_anchor.to(device)
        trr_anchor_agent.eval()
        for parameter in trr_anchor_agent.parameters():
            parameter.requires_grad_(False)
        if bool(args.trr_freeze_base_actor) and isinstance(agent, CleanRLResidualBeliefActorCritic):
            for module in (agent.actor_body, agent.alpha_head, agent.beta_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            print("TRR base actor frozen; residual adapter remains trainable")
        print(
            "TRR-PPO enabled: "
            f"anchor={anchor_path} checkpoint_step={anchor_checkpoint.get('global_step', 'unknown')} "
            f"alpha={args.trr_alpha} residual_limit={args.trr_residual_action_limit} "
            f"min_regret={args.trr_min_normalized_regret}"
        )
    acbr_anchor_agent: Any | None = None
    if bool(args.acbr_enabled):
        anchor_path = str(args.acbr_anchor_checkpoint or args.init_checkpoint)
        loaded_anchor, anchor_checkpoint = load_cleanrl_agent(anchor_path, device=device)
        if int(getattr(loaded_anchor, "obs_dim", -1)) != obs_dim:
            raise ValueError(
                f"ACBR anchor checkpoint obs_dim={getattr(loaded_anchor, 'obs_dim', None)} "
                f"does not match env obs_dim={obs_dim}"
            )
        if int(getattr(loaded_anchor, "action_dim", -1)) != action_dim:
            raise ValueError(
                f"ACBR anchor checkpoint action_dim={getattr(loaded_anchor, 'action_dim', None)} "
                f"does not match env action_dim={action_dim}"
            )
        acbr_anchor_agent = loaded_anchor.to(device)
        acbr_anchor_agent.eval()
        for parameter in acbr_anchor_agent.parameters():
            parameter.requires_grad_(False)
        print(
            "ACBR-PPO enabled: "
            f"anchor={anchor_path} checkpoint_step={anchor_checkpoint.get('global_step', 'unknown')} "
            f"critic_coef={args.acbr_critic_coef} uncertainty={args.acbr_uncertainty_coef} "
            f"anchor_penalty={args.acbr_anchor_penalty}"
        )
    pr_nominal_agent: CleanRLActorCritic | None = None
    if bool(args.acbr_enabled):
        pr_nominal_agent = acbr_anchor_agent
    elif bool(args.trr_enabled):
        pr_nominal_agent = trr_anchor_agent
    elif bool(args.pr_enabled) and (bool(args.pr_enable_queries) or float(args.beta_nominal) > 0.0):
        pr_nominal_agent = copy.deepcopy(agent).to(device)
        pr_nominal_agent.eval()
        for parameter in pr_nominal_agent.parameters():
            parameter.requires_grad_(False)
        print(
            "PR-PPO enabled: "
            f"alpha_pr={args.alpha_pr} beta_nominal={args.beta_nominal} "
            f"queries={bool(args.pr_enable_queries)}"
        )
    bagr_anchor_agent: Any | None = None
    if bool(args.bagr_enabled):
        anchor_path = str(args.bagr_anchor_checkpoint or args.init_checkpoint)
        loaded_anchor, anchor_checkpoint = load_cleanrl_agent(anchor_path, device=device)
        if int(getattr(loaded_anchor, "obs_dim", -1)) != obs_dim:
            raise ValueError(
                f"BAGR anchor checkpoint obs_dim={getattr(loaded_anchor, 'obs_dim', None)} "
                f"does not match env obs_dim={obs_dim}"
            )
        if int(getattr(loaded_anchor, "action_dim", -1)) != action_dim:
            raise ValueError(
                f"BAGR anchor checkpoint action_dim={getattr(loaded_anchor, 'action_dim', None)} "
                f"does not match env action_dim={action_dim}"
            )
        bagr_anchor_agent = loaded_anchor.to(device)
        bagr_anchor_agent.eval()
        for parameter in bagr_anchor_agent.parameters():
            parameter.requires_grad_(False)
        if bool(args.bagr_freeze_base_actor) and isinstance(agent, CleanRLResidualBeliefActorCritic):
            for module in (agent.actor_body, agent.alpha_head, agent.beta_head):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            print("BAGR base actor frozen; residual adapter remains trainable")
        print(
            "BAGR-PPO enabled: "
            f"anchor={anchor_path} checkpoint_step={anchor_checkpoint.get('global_step', 'unknown')} "
            f"attack_variants={args.bagr_max_attack_variants} "
            f"belief_temperature={args.bagr_belief_temperature} "
            f"residual_limit={args.bagr_residual_action_limit}"
        )
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    behavior_cloning_warmstart(
        agent,
        optimizer,
        envs,
        steps=args.bc_warmstart_steps,
        seed=args.seed,
        device=device,
        writer=writer,
    )
    obs_np = reset_envs(envs, args.seed + 200_000)

    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), dtype=torch.float32, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    risk_costs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    risk_features = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    prb_feature_dim = int(prb_builder.feature_dim) if prb_builder is not None else 1
    prb_component_dim = int(prb_builder.component_dim) if prb_builder is not None else 4
    bagr_confidence_feature_index = -1
    bagr_entropy_feature_index = -1
    if prb_builder is not None:
        try:
            bagr_confidence_feature_index = int(prb_builder.feature_names.index("attack_belief_confidence"))
            bagr_entropy_feature_index = int(prb_builder.feature_names.index("attack_belief_entropy_norm"))
        except ValueError:
            bagr_confidence_feature_index = -1
            bagr_entropy_feature_index = -1
    prb_features = torch.zeros((args.num_steps, args.num_envs, prb_feature_dim), dtype=torch.float32, device=device)
    prb_residual_total_targets = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    prb_true_total_targets = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    prb_component_residual_targets = torch.zeros(
        (args.num_steps, args.num_envs, prb_component_dim),
        dtype=torch.float32,
        device=device,
    )
    prb_true_component_targets = torch.zeros(
        (args.num_steps, args.num_envs, prb_component_dim),
        dtype=torch.float32,
        device=device,
    )
    prb_component_masks = torch.zeros((args.num_steps, args.num_envs, prb_component_dim), dtype=torch.float32, device=device)
    prb_rng = np.random.default_rng(args.seed + 900_000)

    global_step = 0
    start_time = time.time()
    best_mean_reward: float | None = None
    stale_eval_count = 0
    last_eval_step = 0
    eval_records: list[dict[str, float | int]] = []
    rollout_metric_records: list[dict[str, Any]] = []
    pr_query_rng = np.random.default_rng(args.seed + 700_000)
    pr_buffer = PlannerRegretBuffer()
    pr_chunk_records: list[dict[str, Any]] = []
    pr_local_sigmas = tuple(float(value) for value in (args.pr_local_sigmas or []) if float(value) > 0.0)
    pr_risk_dim_indices = parse_int_csv(args.pr_risk_dim_indices)
    pr_candidate_config = CandidateActionConfig(
        num_candidates=int(args.pr_num_candidates),
        local_sigma=float(args.pr_local_sigma),
        local_sigmas=pr_local_sigmas,
        num_random_candidates=int(args.pr_num_random_candidates),
        include_policy_action=bool(args.pr_include_policy_action),
        include_nominal_action=bool(args.pr_include_nominal_action),
        include_zero_delta=bool(args.pr_include_zero_delta),
        include_axis_perturbations=bool(args.pr_include_axis_perturbations),
        risk_dim_indices=pr_risk_dim_indices,
        risk_local_sigma=float(args.pr_risk_local_sigma),
        include_risk_axis_perturbations=bool(args.pr_include_risk_axis_perturbations),
        include_risk_block_perturbations=bool(args.pr_include_risk_block_perturbations),
        zero_action_value=0.5 if args.action_mode == "preference_delta" else 0.0,
    )
    pr_target_config = PlannerRegretTargetConfig(
        target_type=str(args.pr_target_type),
        soft_temperature=float(args.pr_soft_temperature),
        cost_normalization=str(args.pr_cost_normalization),
        regret_weight_max=float(args.regret_weight_max),
        hard_regret_threshold=float(args.pr_hard_regret_threshold),
        soft_regret_threshold=float(args.pr_soft_regret_threshold),
        random_target_control=bool(args.pr_random_target_control),
    )
    pr_cpa_config = CPAConfig(
        temperature=float(args.pr_cpa_temperature),
        min_positive_adv=float(args.pr_min_positive_adv),
        weighting=str(args.pr_cpa_weighting),
        sample_weight=str(args.pr_cpa_sample_weight),
        regret_weight_max=float(args.regret_weight_max),
    )
    pr_pair_config = PairwisePreferenceConfig(
        adv_temperature=float(args.pr_adv_temperature),
        pref_temperature=float(args.pr_pref_temperature),
        min_positive_adv=float(args.pr_min_positive_adv),
        regret_weight_max=float(args.regret_weight_max),
    )
    pr_evaluator = PlannerCounterfactualEvaluator(args.reward_cost_key, failure_cost=float(args.pr_failure_cost))
    pr_actor_params = actor_parameters(agent)
    alpha_pr_runtime = float(args.alpha_pr)

    next_obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    if prb_builder is not None:
        next_prb_features_np, _ = prb_probe_features_for_envs(
            envs,
            prb_builder,
            action_dim,
            args.reward_cost_key,
            args.action_mode,
            environment_attack_config=environment_attack_config,
            bagr_enabled=bool(args.bagr_enabled),
            bagr_max_attack_variants=int(args.bagr_max_attack_variants),
            bagr_belief_temperature=float(args.bagr_belief_temperature),
            bagr_belief_prior_mix=float(args.bagr_belief_prior_mix),
            failure_cost=float(args.pr_failure_cost),
            rng=prb_rng,
            random_features=bool(args.prb_use_random_residual_features),
        )
    else:
        next_prb_features_np = np.zeros((args.num_envs, prb_feature_dim), dtype=np.float32)
    next_prb_features = torch.as_tensor(next_prb_features_np, dtype=torch.float32, device=device)

    if args.n_eval_episodes > 0:
        initial_eval_metrics = evaluate_agent(
            agent,
            map_size=args.map_size,
            scenario=args.scenario,
            seed=int(args.eval_seed) if args.eval_seed is not None else args.seed + 50_000,
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
            bagr_enabled=bool(args.bagr_enabled),
            bagr_max_attack_variants=int(args.bagr_max_attack_variants),
            bagr_belief_temperature=float(args.bagr_belief_temperature),
            bagr_belief_prior_mix=float(args.bagr_belief_prior_mix),
            pr_failure_cost=float(args.pr_failure_cost),
            acbr_enabled=bool(args.acbr_enabled),
            acbr_anchor_agent=acbr_anchor_agent,
            acbr_context_dim=int(args.acbr_context_dim),
            acbr_num_candidates=int(args.pr_num_candidates),
            acbr_num_random_candidates=int(args.pr_num_random_candidates),
            acbr_num_structured_candidates=int(args.pr_num_structured_candidates),
            acbr_local_sigma=float(args.pr_local_sigma),
            acbr_risk_local_sigma=float(args.pr_risk_local_sigma),
            acbr_uncertainty_coef=float(args.acbr_uncertainty_coef),
            acbr_anchor_penalty=float(args.acbr_anchor_penalty),
            acbr_policy_penalty=float(args.acbr_policy_penalty),
        )
        initial_record = {"global_step": 0, **initial_eval_metrics}
        eval_records.append(initial_record)
        pd.DataFrame(eval_records).to_csv(run_dir / "eval_metrics.csv", index=False)
        writer.add_scalar("eval/mean_reward", initial_eval_metrics["mean_reward"], 0)
        writer.add_scalar("eval/success_rate", initial_eval_metrics["success_rate"], 0)
        writer.add_scalar("eval/mean_scalar_cost", initial_eval_metrics["mean_scalar_cost"], 0)
        writer.add_scalar(
            "eval/mean_attacked_scalar_cost",
            initial_eval_metrics["mean_attacked_scalar_cost"],
            0,
        )
        writer.add_scalar(
            "eval/mean_lambda_uncertainty",
            initial_eval_metrics["mean_lambda_uncertainty"],
            0,
        )
        print(
            f"step=0 "
            f"eval_reward={initial_eval_metrics['mean_reward']:.4f} "
            f"eval_success={initial_eval_metrics['success_rate']:.3f} "
            f"eval_attacked_cost={initial_eval_metrics['mean_attacked_scalar_cost']:.4f} "
            f"eval_lambda={initial_eval_metrics['mean_lambda_uncertainty']:.3f}"
        )

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / float(num_updates)
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        rollout_rewards = []
        rollout_successes = []
        acbr_rerank_count = 0
        acbr_rerank_changed = 0
        acbr_rerank_gated = 0
        acbr_rerank_score_sum = 0.0
        acbr_rerank_std_sum = 0.0
        acbr_predicted_improvement_sum = 0.0

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            if prb_builder is not None:
                prb_features[step] = next_prb_features

            with torch.no_grad():
                if prb_builder is not None:
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs,
                        residual_features=next_prb_features,
                        detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                    )
                else:
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value
                pr_policy_mean_np = None
                pr_nominal_mean_np = None
                pr_policy_std_np = None
                if bool(args.pr_enabled) and bool(args.pr_enable_queries):
                    pr_query_residual_features = next_prb_features if prb_builder is not None else None
                    pr_policy_mean = deterministic_policy_action(
                        agent,
                        next_obs,
                        residual_features=pr_query_residual_features,
                        detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                    )
                    pr_policy_mean_np = pr_policy_mean.detach().cpu().numpy()
                    try:
                        pr_dist = policy_distribution(
                            agent,
                            next_obs,
                            residual_features=pr_query_residual_features,
                            detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                        )
                        pr_policy_std_np = pr_dist.base_dist.stddev.detach().cpu().numpy()
                    except Exception:
                        pr_policy_std_np = None
                    if pr_nominal_agent is not None:
                        pr_nominal_mean_np = deterministic_policy_action(pr_nominal_agent, next_obs).detach().cpu().numpy()

            actions[step] = action
            logprobs[step] = logprob

            if (
                bool(args.pr_enabled)
                and bool(args.pr_enable_queries)
                and pr_policy_mean_np is not None
                and int(args.pr_query_interval) > 0
                and step % int(args.pr_query_interval) == 0
            ):
                current_obs_np = next_obs.detach().cpu().numpy()
                low = np.asarray(envs[0].action_space.low, dtype=np.float32)
                high = np.asarray(envs[0].action_space.high, dtype=np.float32)
                priority_fallback = str(args.pr_query_strategy) == "mixed"
                priority_fallback_reason = (
                    f"priority key {args.pr_priority_key!r} unavailable during rollout"
                    if priority_fallback
                    else ""
                )
                for env_index, env in enumerate(envs):
                    if pr_query_rng.random() > float(np.clip(args.pr_query_fraction, 0.0, 1.0)):
                        continue
                    policy_mean_action = pr_policy_mean_np[env_index]
                    nominal_mean_action = (
                        pr_nominal_mean_np[env_index]
                        if pr_nominal_mean_np is not None
                        else policy_mean_action
                    )
                    candidates, candidate_meta = generate_candidate_actions(
                        policy_mean_action,
                        nominal_mean_action,
                        low,
                        high,
                        pr_candidate_config,
                        pr_query_rng,
                    )
                    if bool(args.pr_include_structured_candidates) and int(args.pr_num_structured_candidates) > 0:
                        structured_candidates, structured_meta = generate_structured_candidate_actions(
                            env,
                            low,
                            high,
                            int(args.pr_num_structured_candidates),
                            action_mode=str(args.action_mode),
                            action_gain=float(args.action_gain),
                            max_uncertainty_lambda=float(args.max_uncertainty_lambda),
                            dedup_tol=float(pr_candidate_config.dedup_tol),
                        )
                        if structured_candidates.size > 0:
                            candidates, candidate_meta = merge_candidate_action_sets(
                                candidates,
                                candidate_meta,
                                structured_candidates,
                                structured_meta,
                                low,
                                high,
                                dedup_tol=float(pr_candidate_config.dedup_tol),
                            )
                    if bool(args.pr_game_teacher_enabled):
                        target = build_game_planner_regret_target(
                            env,
                            candidates,
                            policy_mean_action,
                            environment_attack_config,
                            str(args.reward_cost_key),
                            pr_target_config,
                            rng=pr_query_rng,
                            mode=str(args.pr_game_teacher_mode),
                            max_attack_variants=int(args.pr_game_max_attack_variants),
                            softmax_temperature=float(args.pr_game_softmax_temperature),
                            failure_cost=float(args.pr_failure_cost),
                        )
                        planner_failures = int(target.get("planner_failure_count", 0))
                    else:
                        eval_results = pr_evaluator.evaluate_many(env, candidates, candidate_meta)
                        candidate_costs = np.asarray(
                            [float(item["true_attacked_cost"]) for item in eval_results],
                            dtype=np.float64,
                        )
                        target = build_planner_regret_target(
                            candidates,
                            candidate_costs,
                            policy_mean_action,
                            pr_target_config,
                            rng=pr_query_rng,
                        )
                        planner_failures = int(sum(bool(item.get("failure_flag", False)) for item in eval_results))
                    target_action = np.asarray(target["target_action"], dtype=np.float32)
                    action_std = float("nan")
                    if pr_policy_std_np is not None:
                        action_std = float(np.mean(np.asarray(pr_policy_std_np[env_index], dtype=np.float64)))
                    target_l2 = float(target.get("target_l2", np.linalg.norm(policy_mean_action - target_action)))
                    target_l2_over_action_std = (
                        target_l2 / max(action_std, 1e-8)
                        if np.isfinite(action_std)
                        else float("nan")
                    )
                    ref_logp_candidates: list[float] = []
                    if str(args.pr_aux_loss_type) == "pairwise_pref" and str(args.pr_pair_reference) == "stored_logp":
                        with torch.no_grad():
                            obs_one = torch.as_tensor(
                                current_obs_np[env_index],
                                dtype=torch.float32,
                                device=device,
                            ).unsqueeze(0)
                            candidate_tensor = torch.as_tensor(candidates, dtype=torch.float32, device=device)
                            ref_obs = obs_one.expand(candidate_tensor.shape[0], -1)
                            ref_residual_features = None
                            if prb_builder is not None:
                                ref_feature = next_prb_features[env_index].unsqueeze(0)
                                ref_residual_features = ref_feature.expand(candidate_tensor.shape[0], -1)
                            ref_dist = policy_distribution(
                                agent,
                                ref_obs,
                                residual_features=ref_residual_features,
                                detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                            )
                            ref_logp_candidates = (
                                ref_dist.log_prob(torch.clamp(candidate_tensor, 1e-6, 1.0 - 1e-6))
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                                .tolist()
                            )
                    pr_buffer.add(
                        {
                            "observation": current_obs_np[env_index].astype(np.float32),
                            "policy_action_at_query_time": policy_mean_action.astype(np.float32),
                            "nominal_action": nominal_mean_action.astype(np.float32),
                            "residual_features": (
                                next_prb_features_np[env_index].astype(np.float32).tolist()
                                if prb_builder is not None
                                else []
                            ),
                            "target_action": target_action,
                            "regret_weight": float(target["regret_weight"]),
                            "raw_regret": float(target["raw_regret"]),
                            "normalized_regret": float(target["normalized_regret"]),
                            "policy_cost": float(target["policy_cost"]),
                            "target_cost": float(target["target_cost"]),
                            "best_candidate_index": int(target["best_candidate_index"]),
                            "policy_candidate_index": int(target["policy_candidate_index"]),
                            "candidate_costs": list(target["candidate_costs"]),
                            "valid_candidate_mask": list(target["valid_candidate_mask"]),
                            "candidate_actions": candidates.astype(np.float32).tolist(),
                            "ref_logp_candidates": ref_logp_candidates,
                            "candidate_metadata": candidate_meta,
                            "state_id": f"u{update}_s{step}_e{env_index}",
                            "timestep": int(step),
                            "planner_query_count": int(len(candidates)),
                            "planner_failure_count": planner_failures,
                            "action_target_l2": target_l2,
                            "target_l2": target_l2,
                            "target_l2_over_action_std": float(target_l2_over_action_std),
                            "best_candidate_l2_from_policy": float(target["best_candidate_l2_from_policy"]),
                            "candidate_cost_spread": float(target["candidate_cost_spread"]),
                            "oracle_regret_excluding_policy_action": float(target["oracle_regret_excluding_policy_action"]),
                            "policy_is_best_candidate": bool(target["policy_is_best_candidate"]),
                            "fraction_policy_is_best_candidate": float(target["fraction_policy_is_best_candidate"]),
                            "pr_game_teacher_enabled": bool(args.pr_game_teacher_enabled),
                            "pr_game_teacher_mode": str(target.get("game_teacher_mode", "")),
                            "pr_game_num_attack_variants": int(target.get("game_num_attack_variants", 0)),
                            "pr_game_attack_variant_ids": str(target.get("game_attack_variant_ids", "")),
                            "pr_game_minimax_value": float(target.get("game_minimax_value", np.nan)),
                            "priority_fallback": bool(priority_fallback),
                            "priority_fallback_reason": priority_fallback_reason,
                        }
                    )

            if (
                bool(args.acbr_enabled)
                and pr_policy_mean_np is not None
                and pr_cumulative_step(args, global_step) >= int(args.acbr_rerank_start_after_steps)
            ):
                current_obs_np = next_obs.detach().cpu().numpy()
                selected_actions_np = action.detach().cpu().numpy().astype(np.float32)
                sampled_actions_np = selected_actions_np.copy()
                low = np.asarray(envs[0].action_space.low, dtype=np.float32)
                high = np.asarray(envs[0].action_space.high, dtype=np.float32)
                for env_index, env in enumerate(envs):
                    policy_mean_action = pr_policy_mean_np[env_index]
                    anchor_action = (
                        pr_nominal_mean_np[env_index]
                        if pr_nominal_mean_np is not None
                        else policy_mean_action
                    )
                    candidates, candidate_meta = generate_candidate_actions(
                        policy_mean_action,
                        anchor_action,
                        low,
                        high,
                        pr_candidate_config,
                        pr_query_rng,
                    )
                    sampled_candidate = sampled_actions_np[env_index : env_index + 1]
                    candidates, candidate_meta = merge_candidate_action_sets(
                        candidates,
                        candidate_meta,
                        sampled_candidate,
                        [{"kind": "sampled_policy"}],
                        low,
                        high,
                        dedup_tol=float(pr_candidate_config.dedup_tol),
                    )
                    if bool(args.pr_include_structured_candidates) and int(args.pr_num_structured_candidates) > 0:
                        structured_candidates, structured_meta = generate_structured_candidate_actions(
                            env,
                            low,
                            high,
                            int(args.pr_num_structured_candidates),
                            action_mode=str(args.action_mode),
                            action_gain=float(args.action_gain),
                            max_uncertainty_lambda=float(args.max_uncertainty_lambda),
                            dedup_tol=float(pr_candidate_config.dedup_tol),
                        )
                        if structured_candidates.size > 0:
                            candidates, candidate_meta = merge_candidate_action_sets(
                                candidates,
                                candidate_meta,
                                structured_candidates,
                                structured_meta,
                                low,
                                high,
                                dedup_tol=float(pr_candidate_config.dedup_tol),
                            )
                    context_np = acbr_context_array(
                        next_prb_features_np[env_index] if prb_builder is not None else None,
                        int(args.acbr_context_dim),
                    )
                    selected_action, acbr_diag = acbr_rerank_candidates(
                        agent,
                        current_obs_np[env_index],
                        candidates,
                        context_np,
                        policy_mean_action,
                        anchor_action,
                        args,
                        device,
                        fallback_action=sampled_actions_np[env_index],
                    )
                    selected_actions_np[env_index] = selected_action
                    acbr_rerank_count += 1
                    acbr_rerank_changed += int(float(acbr_diag.get("changed", 0.0)) > 0.0)
                    acbr_rerank_gated += int(float(acbr_diag.get("gated", 0.0)) > 0.0)
                    if np.isfinite(float(acbr_diag.get("score", np.nan))):
                        acbr_rerank_score_sum += float(acbr_diag["score"])
                    if np.isfinite(float(acbr_diag.get("std", np.nan))):
                        acbr_rerank_std_sum += float(acbr_diag["std"])
                    if np.isfinite(float(acbr_diag.get("predicted_improvement", np.nan))):
                        acbr_predicted_improvement_sum += float(acbr_diag["predicted_improvement"])
                action = torch.as_tensor(
                    np.clip(selected_actions_np, 1e-6, 1.0 - 1e-6),
                    dtype=torch.float32,
                    device=device,
                )
                with torch.no_grad():
                    rerank_dist = policy_distribution(
                        agent,
                        next_obs,
                        residual_features=next_prb_features if prb_builder is not None else None,
                        detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                    )
                    logprob = rerank_dist.log_prob(action)
                actions[step] = action
                logprobs[step] = logprob

            action_np = action.cpu().numpy()
            next_obs_np, reward_np, done_np, infos = step_envs(envs, action_np)
            rewards[step] = torch.as_tensor(reward_np, dtype=torch.float32, device=device)
            dones[step] = torch.as_tensor(done_np, dtype=torch.float32, device=device)
            risk_pairs = [mra_rollout_risk(info, args.reward_cost_key) for info in infos]
            risk_costs[step] = torch.as_tensor(
                [item[0] for item in risk_pairs],
                dtype=torch.float32,
                device=device,
            )
            risk_features[step] = torch.as_tensor(
                [item[1] for item in risk_pairs],
                dtype=torch.float32,
                device=device,
            )
            if prb_builder is not None:
                prb_target_batch = prb_targets_from_infos(infos, action_np, prb_builder, args.reward_cost_key)
                prb_residual_total_targets[step] = torch.as_tensor(
                    prb_target_batch["residual_total"],
                    dtype=torch.float32,
                    device=device,
                )
                prb_true_total_targets[step] = torch.as_tensor(
                    prb_target_batch["true_total"],
                    dtype=torch.float32,
                    device=device,
                )
                prb_component_residual_targets[step] = torch.as_tensor(
                    prb_target_batch["component_residual"],
                    dtype=torch.float32,
                    device=device,
                )
                prb_true_component_targets[step] = torch.as_tensor(
                    prb_target_batch["true_component"],
                    dtype=torch.float32,
                    device=device,
                )
                prb_component_masks[step] = torch.as_tensor(
                    prb_target_batch["component_mask"],
                    dtype=torch.float32,
                    device=device,
                )
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32, device=device)
            if prb_builder is not None:
                next_prb_features_np, _ = prb_probe_features_for_envs(
                    envs,
                    prb_builder,
                    action_dim,
                    args.reward_cost_key,
                    args.action_mode,
                    environment_attack_config=environment_attack_config,
                    bagr_enabled=bool(args.bagr_enabled),
                    bagr_max_attack_variants=int(args.bagr_max_attack_variants),
                    bagr_belief_temperature=float(args.bagr_belief_temperature),
                    bagr_belief_prior_mix=float(args.bagr_belief_prior_mix),
                    failure_cost=float(args.pr_failure_cost),
                    rng=prb_rng,
                    random_features=bool(args.prb_use_random_residual_features),
                )
                next_prb_features = torch.as_tensor(next_prb_features_np, dtype=torch.float32, device=device)

            rollout_rewards.extend(reward_np.tolist())
            rollout_successes.extend(float(info["success"]) for info in infos)
            if args.rollout_metrics_path:
                for env_index, info in enumerate(infos):
                    variant_id = info.get("environment_attack_mixture_variant_id", "")
                    variant_scale = info.get("environment_attack_mixture_variant_scale", np.nan)
                    policy_cost = info.get(args.reward_cost_key, info.get("scalar_cost", np.nan))
                    rollout_metric_records.append(
                        {
                            "global_step": int(global_step),
                            "update": int(update),
                            "rollout_step": int(step),
                            "env_index": int(env_index),
                            "variant_id": str(variant_id),
                            "variant_scale": float(variant_scale) if variant_scale not in ("", None) else np.nan,
                            "policy_cost": float(policy_cost),
                            "scalar_cost": float(info.get("scalar_cost", np.nan)),
                            "attacked_scalar_cost": float(info.get("attacked_scalar_cost", np.nan)),
                            "soft_attacked_scalar_cost": float(info.get("soft_attacked_scalar_cost", np.nan)),
                            "returned_reward": float(info.get("returned_reward", reward_np[env_index])),
                            "success": float(info.get("success", 0.0)),
                            "reward_cost_key": str(args.reward_cost_key),
                            "reward_cost_source": str(info.get("reward_cost_source", "")),
                            "environment_attack_type": str(info.get("environment_attack_type", "")),
                        }
                    )

        b_risk_costs_pre = risk_costs.reshape(-1)
        b_risk_features_pre = risk_features.reshape(-1)
        b_mra_weights, b_tail_excess, b_risk_feature_signal, mra_cutoff = mra_batch_weights(
            b_risk_costs_pre,
            b_risk_features_pre,
            args,
        )
        training_rewards = rewards
        if bool(args.mra_enabled) and (
            float(args.mra_tail_reward_penalty) > 0.0
            or float(args.mra_risk_reward_penalty) > 0.0
        ):
            training_rewards = rewards - (
                float(args.mra_tail_reward_penalty) * b_tail_excess.reshape_as(rewards)
                + float(args.mra_risk_reward_penalty) * b_risk_feature_signal.reshape_as(rewards)
            )

        with torch.no_grad():
            if prb_builder is not None:
                next_value = agent.get_value(next_obs, next_prb_features)
            else:
                next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards)
            last_gae_lam = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
            for step in reversed(range(args.num_steps)):
                if step == args.num_steps - 1:
                    next_values = next_value
                else:
                    next_values = values[step + 1]
                next_non_terminal = 1.0 - dones[step]
                delta = training_rewards[step] + args.gamma * next_values * next_non_terminal - values[step]
                last_gae_lam = (
                    delta
                    + args.gamma
                    * args.gae_lambda
                    * next_non_terminal
                    * last_gae_lam
                )
                advantages[step] = last_gae_lam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_actions = actions.reshape((-1, action_dim))
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_mra_weights = b_mra_weights.reshape(-1)
        b_prb_features = prb_features.reshape((-1, prb_feature_dim))
        b_prb_residual_total_targets = prb_residual_total_targets.reshape(-1)
        b_prb_true_total_targets = prb_true_total_targets.reshape(-1)
        b_prb_component_residual_targets = prb_component_residual_targets.reshape((-1, prb_component_dim))
        b_prb_true_component_targets = prb_true_component_targets.reshape((-1, prb_component_dim))
        b_prb_component_masks = prb_component_masks.reshape((-1, prb_component_dim))

        b_indices = np.arange(batch_size)
        clipfracs = []
        pre_update_pr_summary = pr_buffer.summary()
        last_anchor_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_mirror_prior_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_game_nominal_prior_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_game_lambda_drift_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_game_risk_drift_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_game_lambda_excess_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_game_risk_excess_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_bagr_residual_l2_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_bagr_residual_barrier_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_bagr_residual_abs_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_bagr_adaptive_limit_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_bagr_belief_confidence_mean = torch.tensor(float("nan"), dtype=torch.float32, device=device)
        last_bagr_belief_entropy_mean = torch.tensor(float("nan"), dtype=torch.float32, device=device)
        last_trr_teacher_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_trr_residual_l2_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_trr_residual_barrier_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_trr_residual_abs_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_trr_target_residual_abs_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_trr_active_fraction = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_acbr_critic_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_acbr_target_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_acbr_pred_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_acbr_pred_std = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_acbr_active_fraction = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_planner_regret_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_pr_nominal_prior_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_prb_aux_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_prb_residual_total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_prb_true_total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_prb_component_residual_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_prb_true_component_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_prb_latent_norm_mean = float("nan")
        last_prb_latent_norm_std = float("nan")
        last_total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        last_grad_norm_ppo_actor = float("nan")
        last_grad_norm_pr_actor = float("nan")
        last_grad_norm_nominal_actor = float("nan")
        last_grad_ratio_pr = float("nan")
        last_grad_ratio_nominal = float("nan")
        alpha_components = pr_alpha_components(
            args,
            global_step,
            mean_positive_regret=float(pre_update_pr_summary.get("pr_mean_positive_regret", 0.0)),
            alpha_runtime=alpha_pr_runtime,
        )
        alpha_pr_eff = float(alpha_components["pr_alpha_eff"])
        cumulative_recovery_step = pr_cumulative_step(args, global_step)
        trr_components = trr_alpha_components(args, global_step)
        trr_alpha_eff = float(trr_components["trr_alpha_eff"])
        last_mra_weight_mean = torch.tensor(1.0, dtype=torch.float32, device=device)
        anchor_mask = mra_action_mask(action_dim, args.mra_anchor_mask).to(device)
        grad_diag_done = False
        pr_pair_ref_agent: CleanRLActorCritic | None = None
        if (
            bool(args.pr_enabled)
            and str(args.pr_aux_loss_type) == "pairwise_pref"
            and str(args.pr_pair_reference) == "old_policy"
            and len(pr_buffer) > 0
        ):
            pr_pair_ref_agent = copy.deepcopy(agent).to(device)
            pr_pair_ref_agent.eval()
            for parameter in pr_pair_ref_agent.parameters():
                parameter.requires_grad_(False)
        last_pairwise_pref_loss = float("nan")
        last_pairwise_num_states = 0
        last_pairwise_num_pairs = 0
        last_pairwise_pairs_per_state = 0.0
        last_pairwise_fraction_states = 0.0
        last_pairwise_mean_advantage = float("nan")
        last_pairwise_max_advantage = float("nan")
        last_pairwise_mean_logp_margin = float("nan")
        last_pairwise_mean_ref_logp_margin = float("nan")
        last_pairwise_mean_z = float("nan")

        for _ in range(args.update_epochs):
            np.random.shuffle(b_indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = b_indices[start:end]

                if prb_builder is not None:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_indices],
                        b_actions[mb_indices],
                        residual_features=b_prb_features[mb_indices],
                        detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                    )
                else:
                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                        b_obs[mb_indices],
                        b_actions[mb_indices],
                    )
                logratio = newlogprob - b_logprobs[mb_indices]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_indices]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std(unbiased=False) + 1e-8
                    )

                mb_weights = b_mra_weights[mb_indices]
                mb_weights = mb_weights / torch.clamp(mb_weights.mean(), min=1e-6)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio,
                    1.0 - args.clip_coef,
                    1.0 + args.clip_coef,
                )
                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_weights).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_indices]) ** 2
                    v_clipped = b_values[mb_indices] + torch.clamp(
                        newvalue - b_values[mb_indices],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_indices]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_indices]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
                anchor_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if anchor_agent is not None and float(args.mra_anchor_coef) > 0.0:
                    current_mean_action = agent.get_deterministic_action(b_obs[mb_indices])
                    with torch.no_grad():
                        anchor_action = anchor_agent.get_deterministic_action(b_obs[mb_indices])
                    masked_delta = (current_mean_action - anchor_action) * anchor_mask
                    per_sample_anchor = masked_delta.pow(2).mean(dim=-1)
                    anchor_loss = (per_sample_anchor * mb_weights).mean()
                    loss = loss + float(args.mra_anchor_coef) * anchor_loss
                mirror_prior_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if mirror_anchor_agent is not None and float(args.mirror_nominal_prior_coef) > 0.0:
                    current_mean_action = agent.get_deterministic_action(b_obs[mb_indices])
                    with torch.no_grad():
                        nominal_mean_action = mirror_anchor_agent.get_deterministic_action(b_obs[mb_indices])
                    mirror_prior_loss = (current_mean_action - nominal_mean_action).pow(2).mean()
                    loss = loss + float(args.mirror_nominal_prior_coef) * mirror_prior_loss
                game_nominal_prior_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                game_lambda_drift_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                game_risk_drift_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                game_lambda_excess_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
                game_risk_excess_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
                if game_anchor_agent is not None:
                    mb_residual_features = b_prb_features[mb_indices] if prb_builder is not None else None
                    current_mean_action = deterministic_policy_action(
                        agent,
                        b_obs[mb_indices],
                        residual_features=mb_residual_features,
                    )
                    with torch.no_grad():
                        nominal_mean_action = deterministic_policy_action(
                            game_anchor_agent,
                            b_obs[mb_indices],
                            residual_features=mb_residual_features,
                        )
                    if float(args.game_nominal_prior_coef) > 0.0:
                        game_nominal_prior_loss = (current_mean_action - nominal_mean_action).pow(2).mean()
                        loss = loss + float(args.game_nominal_prior_coef) * game_nominal_prior_loss
                    if (
                        game_lambda_index is not None
                        and float(args.game_lambda_drift_coef) > 0.0
                    ):
                        lambda_excess = torch.relu(
                            current_mean_action[:, game_lambda_index]
                            - nominal_mean_action[:, game_lambda_index]
                            - float(args.game_lambda_drift_margin)
                        )
                        game_lambda_drift_loss = lambda_excess.pow(2).mean()
                        game_lambda_excess_mean = lambda_excess.mean()
                        loss = loss + float(args.game_lambda_drift_coef) * game_lambda_drift_loss
                    if (
                        game_risk_index_tensor.numel() > 0
                        and float(args.game_risk_drift_coef) > 0.0
                    ):
                        risk_excess = torch.relu(
                            current_mean_action.index_select(1, game_risk_index_tensor)
                            - nominal_mean_action.index_select(1, game_risk_index_tensor)
                            - float(args.game_risk_drift_margin)
                        )
                        game_risk_drift_loss = risk_excess.pow(2).mean()
                        game_risk_excess_mean = risk_excess.mean()
                        loss = loss + float(args.game_risk_drift_coef) * game_risk_drift_loss
                bagr_residual_l2_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                bagr_residual_barrier_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                bagr_residual_abs_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
                bagr_adaptive_limit_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
                bagr_belief_confidence_mean = torch.tensor(float("nan"), dtype=torch.float32, device=device)
                bagr_belief_entropy_mean = torch.tensor(float("nan"), dtype=torch.float32, device=device)
                if (
                    bool(args.bagr_enabled)
                    and bagr_anchor_agent is not None
                    and prb_builder is not None
                    and isinstance(agent, CleanRLResidualBeliefActorCritic)
                ):
                    mb_residual_features = b_prb_features[mb_indices]
                    current_mean_action = deterministic_policy_action(
                        agent,
                        b_obs[mb_indices],
                        residual_features=mb_residual_features,
                        detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                    )
                    with torch.no_grad():
                        nominal_mean_action = deterministic_policy_action(
                            bagr_anchor_agent,
                            b_obs[mb_indices],
                        )
                    residual_delta = current_mean_action - nominal_mean_action
                    confidence = torch.zeros(
                        residual_delta.shape[0],
                        dtype=torch.float32,
                        device=device,
                    )
                    entropy = torch.ones_like(confidence)
                    if 0 <= bagr_confidence_feature_index < mb_residual_features.shape[-1]:
                        confidence = torch.clamp(mb_residual_features[:, bagr_confidence_feature_index], 0.0, 1.0)
                    if 0 <= bagr_entropy_feature_index < mb_residual_features.shape[-1]:
                        entropy = torch.clamp(mb_residual_features[:, bagr_entropy_feature_index], 0.0, 1.0)
                    limit_multiplier = torch.clamp(
                        1.0 + float(args.bagr_confidence_limit_scale) * (confidence - 0.5),
                        min=0.25,
                        max=2.0,
                    )
                    adaptive_limit = float(args.bagr_residual_action_limit) * limit_multiplier.unsqueeze(-1)
                    l2_weight = (1.0 + entropy).unsqueeze(-1)
                    bagr_residual_l2_loss = (l2_weight * residual_delta.pow(2)).mean()
                    bagr_residual_barrier_loss = torch.relu(residual_delta.abs() - adaptive_limit).pow(2).mean()
                    bagr_residual_abs_mean = residual_delta.abs().mean()
                    bagr_adaptive_limit_mean = adaptive_limit.mean()
                    bagr_belief_confidence_mean = confidence.mean()
                    bagr_belief_entropy_mean = entropy.mean()
                    if float(args.bagr_residual_l2_coef) > 0.0:
                        loss = loss + float(args.bagr_residual_l2_coef) * bagr_residual_l2_loss
                    if float(args.bagr_residual_barrier_coef) > 0.0:
                        loss = loss + float(args.bagr_residual_barrier_coef) * bagr_residual_barrier_loss
                trr_teacher_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                trr_residual_l2_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                trr_residual_barrier_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                trr_residual_abs_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
                trr_target_residual_abs_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
                trr_active_fraction = torch.tensor(0.0, dtype=torch.float32, device=device)
                if (
                    bool(args.trr_enabled)
                    and trr_anchor_agent is not None
                    and prb_builder is not None
                    and isinstance(agent, CleanRLResidualBeliefActorCritic)
                ):
                    mb_residual_features = b_prb_features[mb_indices]
                    current_mean_action = deterministic_policy_action(
                        agent,
                        b_obs[mb_indices],
                        residual_features=mb_residual_features,
                        detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                    )
                    with torch.no_grad():
                        nominal_mean_action = deterministic_policy_action(trr_anchor_agent, b_obs[mb_indices])
                    residual_delta = current_mean_action - nominal_mean_action
                    trr_residual_l2_loss = residual_delta.pow(2).mean()
                    trr_residual_barrier_loss = torch.relu(
                        residual_delta.abs() - float(args.trr_residual_action_limit)
                    ).pow(2).mean()
                    trr_residual_abs_mean = residual_delta.abs().mean()
                    if float(args.trr_residual_l2_coef) > 0.0:
                        loss = loss + float(args.trr_residual_l2_coef) * trr_residual_l2_loss
                    if float(args.trr_residual_barrier_coef) > 0.0:
                        loss = loss + float(args.trr_residual_barrier_coef) * trr_residual_barrier_loss

                    if trr_alpha_eff > 0.0 and len(pr_buffer) > 0:
                        trr_indices = pr_buffer.sample_indices(minibatch_size, pr_query_rng)
                        trr_arrays = pr_buffer.arrays(trr_indices)
                        trr_obs = torch.as_tensor(trr_arrays["observations"], dtype=torch.float32, device=device)
                        trr_targets = torch.as_tensor(trr_arrays["target_actions"], dtype=torch.float32, device=device)
                        trr_nominal = torch.as_tensor(trr_arrays["nominal_actions"], dtype=torch.float32, device=device)
                        trr_features_np = np.asarray(trr_arrays["residual_features"], dtype=np.float32)
                        trr_norm_regret = torch.as_tensor(
                            trr_arrays["normalized_regrets"],
                            dtype=torch.float32,
                            device=device,
                        )
                        trr_regret_weights = torch.as_tensor(
                            trr_arrays["regret_weights"],
                            dtype=torch.float32,
                            device=device,
                        )
                        if (
                            trr_obs.numel() > 0
                            and trr_targets.shape == trr_nominal.shape
                            and trr_features_np.ndim == 2
                            and trr_features_np.shape[1] > 0
                        ):
                            trr_features = torch.as_tensor(trr_features_np, dtype=torch.float32, device=device)
                            active = trr_norm_regret >= float(args.trr_min_normalized_regret)
                            if active.any():
                                trr_current = deterministic_policy_action(
                                    agent,
                                    trr_obs,
                                    residual_features=trr_features,
                                    detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                                )
                                with torch.no_grad():
                                    if trr_nominal.shape != trr_current.shape:
                                        trr_nominal = deterministic_policy_action(trr_anchor_agent, trr_obs)
                                    target_residual = torch.clamp(
                                        trr_targets - trr_nominal,
                                        min=-float(args.trr_residual_action_limit),
                                        max=float(args.trr_residual_action_limit),
                                    )
                                current_residual = trr_current - trr_nominal
                                per_sample = (current_residual - target_residual).pow(2).mean(dim=-1)
                                sample_weights = torch.clamp(
                                    torch.maximum(trr_regret_weights, torch.clamp(trr_norm_regret, min=0.0)),
                                    min=0.0,
                                    max=float(args.regret_weight_max),
                                )
                                sample_weights = torch.where(active, sample_weights, torch.zeros_like(sample_weights))
                                weight_sum = torch.clamp(sample_weights.sum(), min=1e-8)
                                trr_teacher_loss = (per_sample * sample_weights).sum() / weight_sum
                                loss = loss + trr_alpha_eff * trr_teacher_loss
                                trr_target_residual_abs_mean = target_residual[active].abs().mean()
                            trr_active_fraction = active.float().mean()
                acbr_critic_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                acbr_target_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
                acbr_pred_mean = torch.tensor(0.0, dtype=torch.float32, device=device)
                acbr_pred_std = torch.tensor(0.0, dtype=torch.float32, device=device)
                acbr_active_fraction = torch.tensor(0.0, dtype=torch.float32, device=device)
                if (
                    bool(args.acbr_enabled)
                    and len(pr_buffer) >= int(args.acbr_min_query_states)
                    and hasattr(agent, "predict_acbr_costs")
                ):
                    acbr_indices = pr_buffer.sample_indices(minibatch_size, pr_query_rng)
                    acbr_arrays = pr_buffer.arrays(acbr_indices)
                    acbr_obs = torch.as_tensor(acbr_arrays["observations"], dtype=torch.float32, device=device)
                    acbr_candidates = torch.as_tensor(acbr_arrays["candidate_actions"], dtype=torch.float32, device=device)
                    acbr_costs = torch.as_tensor(acbr_arrays["candidate_costs"], dtype=torch.float32, device=device)
                    acbr_valid = torch.as_tensor(acbr_arrays["valid_candidate_masks"], dtype=torch.bool, device=device)
                    acbr_policy_costs = torch.as_tensor(acbr_arrays["policy_costs"], dtype=torch.float32, device=device)
                    acbr_features_np = np.asarray(acbr_arrays["residual_features"], dtype=np.float32)
                    if acbr_candidates.numel() > 0 and acbr_obs.numel() > 0 and acbr_candidates.dim() == 3:
                        batch_count, candidate_count, action_count = acbr_candidates.shape
                        if acbr_valid.shape != acbr_costs.shape:
                            acbr_valid = torch.isfinite(acbr_costs)
                        else:
                            acbr_valid = acbr_valid & torch.isfinite(acbr_costs)
                        acbr_targets = acbr_normalized_cost_targets(
                            acbr_costs,
                            acbr_policy_costs,
                            float(args.acbr_target_clip),
                        )
                        flat_obs = acbr_obs[:, None, :].expand(batch_count, candidate_count, acbr_obs.shape[-1]).reshape(
                            batch_count * candidate_count,
                            acbr_obs.shape[-1],
                        )
                        flat_actions = acbr_candidates.reshape(batch_count * candidate_count, action_count)
                        acbr_features = None
                        if acbr_features_np.ndim == 2 and acbr_features_np.shape[0] == batch_count:
                            acbr_features = torch.as_tensor(acbr_features_np, dtype=torch.float32, device=device)
                        context = acbr_context_tensor(acbr_features, int(args.acbr_context_dim), device)
                        if context.shape[0] != batch_count:
                            context = torch.zeros(
                                (batch_count, int(args.acbr_context_dim)),
                                dtype=torch.float32,
                                device=device,
                            )
                        flat_context = context[:, None, :].expand(
                            batch_count,
                            candidate_count,
                            context.shape[-1],
                        ).reshape(batch_count * candidate_count, context.shape[-1])
                        flat_valid = acbr_valid.reshape(-1)
                        if flat_valid.any():
                            acbr_predictions = agent.predict_acbr_costs(flat_obs, flat_actions, flat_context)
                            flat_targets = acbr_targets.reshape(-1)
                            target_matrix = flat_targets.unsqueeze(-1).expand_as(acbr_predictions)
                            per_head_loss = F.smooth_l1_loss(acbr_predictions, target_matrix, reduction="none")
                            acbr_critic_loss = per_head_loss[flat_valid].mean()
                            loss = loss + float(args.acbr_critic_coef) * acbr_critic_loss
                            with torch.no_grad():
                                valid_predictions = acbr_predictions[flat_valid]
                                valid_targets = flat_targets[flat_valid]
                                acbr_target_mean = valid_targets.mean()
                                acbr_pred_mean = valid_predictions.mean()
                                acbr_pred_std = valid_predictions.std(unbiased=False)
                                acbr_active_fraction = flat_valid.float().mean()
                planner_regret_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if bool(args.pr_enabled) and alpha_pr_eff > 0.0 and len(pr_buffer) > 0:
                    pr_indices = pr_buffer.sample_indices(minibatch_size, pr_query_rng)
                    pr_arrays = pr_buffer.arrays(pr_indices)
                    pr_obs = torch.as_tensor(pr_arrays["observations"], dtype=torch.float32, device=device)
                    pr_targets = torch.as_tensor(pr_arrays["target_actions"], dtype=torch.float32, device=device)
                    pr_weights = torch.as_tensor(pr_arrays["regret_weights"], dtype=torch.float32, device=device)
                    pr_residual_features = None
                    pr_residual_features_np = np.asarray(pr_arrays["residual_features"], dtype=np.float32)
                    if (
                        isinstance(agent, CleanRLResidualBeliefActorCritic)
                        and pr_residual_features_np.ndim == 2
                        and pr_residual_features_np.shape[1] > 0
                    ):
                        pr_residual_features = torch.as_tensor(
                            pr_residual_features_np,
                            dtype=torch.float32,
                            device=device,
                        )
                    if pr_obs.numel() > 0:
                        if str(args.pr_aux_loss_type) == "nll":
                            pr_dist = policy_distribution(
                                agent,
                                pr_obs,
                                residual_features=pr_residual_features,
                                detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                            )
                            per_sample_pr = -pr_dist.log_prob(torch.clamp(pr_targets, 1e-6, 1.0 - 1e-6))
                            planner_regret_loss = (per_sample_pr * pr_weights).mean()
                        elif str(args.pr_aux_loss_type) == "cpa":
                            pr_candidates = torch.as_tensor(pr_arrays["candidate_actions"], dtype=torch.float32, device=device)
                            pr_costs = torch.as_tensor(pr_arrays["candidate_costs"], dtype=torch.float32, device=device)
                            pr_valid = torch.as_tensor(pr_arrays["valid_candidate_masks"], dtype=torch.bool, device=device)
                            pr_policy_costs = torch.as_tensor(pr_arrays["policy_costs"], dtype=torch.float32, device=device)
                            if pr_candidates.numel() > 0:
                                batch_count, candidate_count, action_count = pr_candidates.shape
                                denom = torch.clamp(pr_policy_costs.abs(), min=float(pr_cpa_config.cost_epsilon)).unsqueeze(1)
                                cpa_adv = torch.clamp((pr_policy_costs.unsqueeze(1) - pr_costs) / denom, min=0.0)
                                cpa_adv = torch.where(pr_valid, cpa_adv, torch.zeros_like(cpa_adv))
                                positive = cpa_adv >= float(pr_cpa_config.min_positive_adv)
                                active = positive.any(dim=1)
                                if active.any():
                                    if str(pr_cpa_config.weighting) == "linear":
                                        weight_values = torch.where(positive, cpa_adv, torch.zeros_like(cpa_adv))
                                        weight_sum = torch.clamp(weight_values.sum(dim=1, keepdim=True), min=1e-8)
                                        cpa_weights = weight_values / weight_sum
                                    else:
                                        logits = cpa_adv / max(float(pr_cpa_config.temperature), 1e-8)
                                        logits = torch.where(positive, logits, torch.full_like(logits, -1e9))
                                        cpa_weights = torch.softmax(logits, dim=1)
                                        cpa_weights = torch.where(positive, cpa_weights, torch.zeros_like(cpa_weights))
                                    flat_obs = pr_obs[:, None, :].expand(batch_count, candidate_count, pr_obs.shape[-1]).reshape(
                                        batch_count * candidate_count,
                                        pr_obs.shape[-1],
                                    )
                                    flat_actions = pr_candidates.reshape(batch_count * candidate_count, action_count)
                                    flat_residual_features = None
                                    if pr_residual_features is not None:
                                        flat_residual_features = pr_residual_features[:, None, :].expand(
                                            batch_count,
                                            candidate_count,
                                            pr_residual_features.shape[-1],
                                        ).reshape(batch_count * candidate_count, pr_residual_features.shape[-1])
                                    flat_dist = policy_distribution(
                                        agent,
                                        flat_obs,
                                        residual_features=flat_residual_features,
                                        detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                                    )
                                    flat_log_prob = flat_dist.log_prob(torch.clamp(flat_actions, 1e-6, 1.0 - 1e-6))
                                    candidate_log_prob = flat_log_prob.reshape(batch_count, candidate_count)
                                    max_adv = torch.clamp(cpa_adv.max(dim=1).values, min=0.0, max=float(pr_cpa_config.regret_weight_max))
                                    per_sample_pr = (cpa_weights * (-candidate_log_prob)).sum(dim=1) * max_adv
                                    planner_regret_loss = per_sample_pr[active].mean()
                                else:
                                    planner_regret_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                        elif str(args.pr_aux_loss_type) == "pairwise_pref":
                            pr_candidates = torch.as_tensor(pr_arrays["candidate_actions"], dtype=torch.float32, device=device)
                            pr_costs = torch.as_tensor(pr_arrays["candidate_costs"], dtype=torch.float32, device=device)
                            pr_valid = torch.as_tensor(pr_arrays["valid_candidate_masks"], dtype=torch.bool, device=device)
                            pr_policy_costs = torch.as_tensor(pr_arrays["policy_costs"], dtype=torch.float32, device=device)
                            pr_policy_indices = torch.as_tensor(
                                pr_arrays["policy_candidate_indices"],
                                dtype=torch.long,
                                device=device,
                            )
                            if pr_candidates.numel() > 0:
                                batch_count, candidate_count, action_count = pr_candidates.shape
                                pr_policy_indices = torch.clamp(pr_policy_indices, 0, candidate_count - 1)
                                denom = torch.clamp(pr_policy_costs.abs(), min=float(pr_pair_config.cost_epsilon)).unsqueeze(1)
                                pair_adv = torch.clamp((pr_policy_costs.unsqueeze(1) - pr_costs) / denom, min=0.0)
                                pair_adv = torch.where(pr_valid, pair_adv, torch.zeros_like(pair_adv))
                                positive = pair_adv >= float(pr_pair_config.min_positive_adv)
                                active = positive.any(dim=1)
                                if active.any():
                                    logits = pair_adv / max(float(pr_pair_config.adv_temperature), 1e-8)
                                    logits = torch.where(positive, logits, torch.full_like(logits, -1e9))
                                    pair_weights = torch.softmax(logits, dim=1)
                                    pair_weights = torch.where(positive, pair_weights, torch.zeros_like(pair_weights))

                                    flat_obs = pr_obs[:, None, :].expand(batch_count, candidate_count, pr_obs.shape[-1]).reshape(
                                        batch_count * candidate_count,
                                        pr_obs.shape[-1],
                                    )
                                    flat_actions = pr_candidates.reshape(batch_count * candidate_count, action_count)
                                    flat_residual_features = None
                                    if pr_residual_features is not None:
                                        flat_residual_features = pr_residual_features[:, None, :].expand(
                                            batch_count,
                                            candidate_count,
                                            pr_residual_features.shape[-1],
                                        ).reshape(batch_count * candidate_count, pr_residual_features.shape[-1])
                                    flat_dist = policy_distribution(
                                        agent,
                                        flat_obs,
                                        residual_features=flat_residual_features,
                                        detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                                    )
                                    candidate_log_prob = flat_dist.log_prob(
                                        torch.clamp(flat_actions, 1e-6, 1.0 - 1e-6)
                                    ).reshape(batch_count, candidate_count)
                                    base_log_prob = candidate_log_prob.gather(1, pr_policy_indices.unsqueeze(1))
                                    current_margin = candidate_log_prob - base_log_prob

                                    reference_mode = str(args.pr_pair_reference)
                                    if reference_mode == "old_policy":
                                        if pr_pair_ref_agent is None:
                                            raise RuntimeError("old_policy pairwise reference was not initialized")
                                        with torch.no_grad():
                                            ref_dist = policy_distribution(
                                                pr_pair_ref_agent,
                                                flat_obs,
                                                residual_features=flat_residual_features,
                                                detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                                            )
                                            ref_candidate_log_prob = ref_dist.log_prob(
                                                torch.clamp(flat_actions, 1e-6, 1.0 - 1e-6)
                                            ).reshape(batch_count, candidate_count)
                                    elif reference_mode == "stored_logp":
                                        ref_candidate_log_prob = torch.as_tensor(
                                            pr_arrays["ref_logp_candidates"],
                                            dtype=torch.float32,
                                            device=device,
                                        )
                                        active = active & torch.isfinite(ref_candidate_log_prob).all(dim=1)
                                    else:
                                        ref_candidate_log_prob = candidate_log_prob.detach()

                                    if active.any():
                                        if reference_mode == "none":
                                            ref_margin = torch.zeros_like(current_margin)
                                        else:
                                            ref_base_log_prob = ref_candidate_log_prob.gather(1, pr_policy_indices.unsqueeze(1))
                                            ref_margin = ref_candidate_log_prob - ref_base_log_prob
                                        z = current_margin - ref_margin
                                        z = torch.clamp(z, -float(pr_pair_config.z_clip), float(pr_pair_config.z_clip))
                                        per_pair_loss = F.softplus(-z / max(float(pr_pair_config.pref_temperature), 1e-8))
                                        sample_weight = torch.clamp(
                                            pair_adv.max(dim=1).values,
                                            min=0.0,
                                            max=float(pr_pair_config.regret_weight_max),
                                        )
                                        per_sample_pr = (pair_weights * per_pair_loss).sum(dim=1) * sample_weight
                                        planner_regret_loss = per_sample_pr[active].mean()

                                        with torch.no_grad():
                                            positive_active = positive & active.unsqueeze(1)
                                            num_active = int(active.sum().item())
                                            num_pairs = int(positive_active.sum().item())
                                            last_pairwise_pref_loss = float(planner_regret_loss.detach().item())
                                            last_pairwise_num_states = num_active
                                            last_pairwise_num_pairs = num_pairs
                                            last_pairwise_pairs_per_state = float(num_pairs / max(num_active, 1))
                                            last_pairwise_fraction_states = float(active.float().mean().item())
                                            if num_pairs > 0:
                                                last_pairwise_mean_advantage = float(pair_adv[positive_active].mean().item())
                                                last_pairwise_max_advantage = float(pair_adv[positive_active].max().item())
                                                last_pairwise_mean_logp_margin = float(current_margin[positive_active].mean().item())
                                                last_pairwise_mean_ref_logp_margin = float(ref_margin[positive_active].mean().item())
                                                last_pairwise_mean_z = float(z[positive_active].mean().item())
                                    else:
                                        planner_regret_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                                else:
                                    planner_regret_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                        else:
                            pr_current_mean = deterministic_policy_action(
                                agent,
                                pr_obs,
                                residual_features=pr_residual_features,
                                detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                            )
                            per_sample_pr = (pr_current_mean - pr_targets).pow(2).mean(dim=-1)
                            planner_regret_loss = (per_sample_pr * pr_weights).mean()
                        loss = loss + float(alpha_pr_eff) * planner_regret_loss
                pr_nominal_prior_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                if (
                    bool(args.pr_enabled)
                    and pr_nominal_agent is not None
                    and float(args.beta_nominal) > 0.0
                ):
                    mb_residual_features = b_prb_features[mb_indices] if prb_builder is not None else None
                    current_mean_action = deterministic_policy_action(
                        agent,
                        b_obs[mb_indices],
                        residual_features=mb_residual_features,
                        detach_residual_latent=bool(args.prb_stopgrad_residual_latent),
                    )
                    with torch.no_grad():
                        nominal_mean_action = deterministic_policy_action(pr_nominal_agent, b_obs[mb_indices])
                    pr_nominal_prior_loss = (current_mean_action - nominal_mean_action).pow(2).mean()
                    loss = loss + float(args.beta_nominal) * pr_nominal_prior_loss

                prb_aux_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
                prb_parts: dict[str, torch.Tensor] = {}
                if (
                    prb_builder is not None
                    and isinstance(agent, CleanRLResidualBeliefActorCritic)
                    and not bool(args.prb_disable_aux_loss)
                    and float(args.prb_aux_coef) > 0.0
                ):
                    prb_predictions = agent.predict_prb_aux(
                        b_obs[mb_indices],
                        b_prb_features[mb_indices],
                        b_actions[mb_indices],
                    )
                    prb_targets = {
                        "residual_total": b_prb_residual_total_targets[mb_indices],
                        "true_total": b_prb_true_total_targets[mb_indices],
                        "component_residual": b_prb_component_residual_targets[mb_indices],
                        "true_component": b_prb_true_component_targets[mb_indices],
                        "component_mask": b_prb_component_masks[mb_indices],
                    }
                    prb_aux_loss, prb_parts = prb_auxiliary_loss(
                        prb_predictions,
                        prb_targets,
                        {
                            "residual_total": float(args.prb_lambda_residual_total),
                            "true_total": float(args.prb_lambda_true_total),
                            "component_residual": float(args.prb_lambda_component_residual),
                            "true_component": float(args.prb_lambda_true_component),
                        },
                    )
                    loss = loss + float(args.prb_aux_coef) * prb_aux_loss
                    with torch.no_grad():
                        latent_norm = prb_predictions["latent"].detach().norm(dim=-1)
                        last_prb_latent_norm_mean = float(latent_norm.mean().item())
                        last_prb_latent_norm_std = float(latent_norm.std(unbiased=False).item())

                if (
                    bool(args.pr_enabled)
                    and bool(args.pr_grad_diagnostics)
                    and not grad_diag_done
                    and (int(args.pr_grad_diagnostics_every) <= 1 or update % int(args.pr_grad_diagnostics_every) == 0)
                ):
                    last_grad_norm_ppo_actor = grad_norm_for_loss(pg_loss, pr_actor_params, retain_graph=True)
                    last_grad_norm_pr_actor = grad_norm_for_loss(
                        float(alpha_pr_eff) * planner_regret_loss,
                        pr_actor_params,
                        retain_graph=True,
                    )
                    last_grad_norm_nominal_actor = grad_norm_for_loss(
                        float(args.beta_nominal) * pr_nominal_prior_loss,
                        pr_actor_params,
                        retain_graph=True,
                    )
                    denom = max(last_grad_norm_ppo_actor, 1e-12) if np.isfinite(last_grad_norm_ppo_actor) else float("nan")
                    last_grad_ratio_pr = last_grad_norm_pr_actor / denom if np.isfinite(denom) else float("nan")
                    last_grad_ratio_nominal = last_grad_norm_nominal_actor / denom if np.isfinite(denom) else float("nan")
                    if bool(args.pr_grad_ratio_controller) and np.isfinite(last_grad_ratio_pr):
                        if last_grad_ratio_pr < float(args.pr_grad_ratio_target_low):
                            alpha_pr_runtime *= 1.25
                        elif last_grad_ratio_pr > float(args.pr_grad_ratio_target_high) * 1.5:
                            alpha_pr_runtime *= 0.8
                        alpha_pr_runtime = float(
                            np.clip(
                                alpha_pr_runtime,
                                float(args.pr_alpha_min),
                                float(args.pr_alpha_max),
                            )
                        )
                    grad_diag_done = True

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                last_anchor_loss = anchor_loss.detach()
                last_mirror_prior_loss = mirror_prior_loss.detach()
                last_game_nominal_prior_loss = game_nominal_prior_loss.detach()
                last_game_lambda_drift_loss = game_lambda_drift_loss.detach()
                last_game_risk_drift_loss = game_risk_drift_loss.detach()
                last_game_lambda_excess_mean = game_lambda_excess_mean.detach()
                last_game_risk_excess_mean = game_risk_excess_mean.detach()
                last_bagr_residual_l2_loss = bagr_residual_l2_loss.detach()
                last_bagr_residual_barrier_loss = bagr_residual_barrier_loss.detach()
                last_bagr_residual_abs_mean = bagr_residual_abs_mean.detach()
                last_bagr_adaptive_limit_mean = bagr_adaptive_limit_mean.detach()
                last_bagr_belief_confidence_mean = bagr_belief_confidence_mean.detach()
                last_bagr_belief_entropy_mean = bagr_belief_entropy_mean.detach()
                last_trr_teacher_loss = trr_teacher_loss.detach()
                last_trr_residual_l2_loss = trr_residual_l2_loss.detach()
                last_trr_residual_barrier_loss = trr_residual_barrier_loss.detach()
                last_trr_residual_abs_mean = trr_residual_abs_mean.detach()
                last_trr_target_residual_abs_mean = trr_target_residual_abs_mean.detach()
                last_trr_active_fraction = trr_active_fraction.detach()
                last_acbr_critic_loss = acbr_critic_loss.detach()
                last_acbr_target_mean = acbr_target_mean.detach()
                last_acbr_pred_mean = acbr_pred_mean.detach()
                last_acbr_pred_std = acbr_pred_std.detach()
                last_acbr_active_fraction = acbr_active_fraction.detach()
                last_planner_regret_loss = planner_regret_loss.detach()
                last_pr_nominal_prior_loss = pr_nominal_prior_loss.detach()
                last_prb_aux_loss = prb_aux_loss.detach()
                if prb_parts:
                    last_prb_residual_total_loss = prb_parts["residual_total_loss"].detach()
                    last_prb_true_total_loss = prb_parts["true_total_loss"].detach()
                    last_prb_component_residual_loss = prb_parts["component_residual_loss"].detach()
                    last_prb_true_component_loss = prb_parts["true_component_loss"].detach()
                last_total_loss = loss.detach()
                last_mra_weight_mean = mb_weights.detach().mean()

            if args.target_kl is not None and approx_kl.item() > args.target_kl:
                break

        explained_var = np.nan
        y_pred = b_values.detach().cpu().numpy()
        y_true = b_returns.detach().cpu().numpy()
        var_y = np.var(y_true)
        if var_y > 0:
            explained_var = 1.0 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / max(time.time() - start_time, 1e-8))
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar("rollout/mean_reward", float(np.mean(rollout_rewards)), global_step)
        writer.add_scalar("rollout/success_rate", float(np.mean(rollout_successes)), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/mra_anchor_loss", float(last_anchor_loss.item()), global_step)
        writer.add_scalar("losses/mirror_nominal_prior_loss", float(last_mirror_prior_loss.item()), global_step)
        writer.add_scalar("losses/game_nominal_prior_loss", float(last_game_nominal_prior_loss.item()), global_step)
        writer.add_scalar("losses/game_lambda_drift_loss", float(last_game_lambda_drift_loss.item()), global_step)
        writer.add_scalar("losses/game_risk_drift_loss", float(last_game_risk_drift_loss.item()), global_step)
        writer.add_scalar("losses/bagr_residual_l2_loss", float(last_bagr_residual_l2_loss.item()), global_step)
        writer.add_scalar("losses/bagr_residual_barrier_loss", float(last_bagr_residual_barrier_loss.item()), global_step)
        writer.add_scalar("game/lambda_excess_mean", float(last_game_lambda_excess_mean.item()), global_step)
        writer.add_scalar("game/risk_excess_mean", float(last_game_risk_excess_mean.item()), global_step)
        writer.add_scalar("bagr/residual_abs_mean", float(last_bagr_residual_abs_mean.item()), global_step)
        writer.add_scalar("bagr/adaptive_limit_mean", float(last_bagr_adaptive_limit_mean.item()), global_step)
        writer.add_scalar("bagr/belief_confidence_mean", float(last_bagr_belief_confidence_mean.item()), global_step)
        writer.add_scalar("bagr/belief_entropy_mean", float(last_bagr_belief_entropy_mean.item()), global_step)
        writer.add_scalar("losses/trr_teacher_loss", float(last_trr_teacher_loss.item()), global_step)
        writer.add_scalar("losses/trr_residual_l2_loss", float(last_trr_residual_l2_loss.item()), global_step)
        writer.add_scalar("losses/trr_residual_barrier_loss", float(last_trr_residual_barrier_loss.item()), global_step)
        writer.add_scalar("trr/alpha_eff", float(trr_alpha_eff), global_step)
        writer.add_scalar("trr/ramp_mult", float(trr_components["trr_ramp_mult"]), global_step)
        writer.add_scalar("trr/schedule_mult", float(trr_components["trr_schedule_mult"]), global_step)
        writer.add_scalar("trr/residual_abs_mean", float(last_trr_residual_abs_mean.item()), global_step)
        writer.add_scalar("trr/target_residual_abs_mean", float(last_trr_target_residual_abs_mean.item()), global_step)
        writer.add_scalar("trr/active_fraction", float(last_trr_active_fraction.item()), global_step)
        writer.add_scalar("losses/acbr_critic_loss", float(last_acbr_critic_loss.item()), global_step)
        writer.add_scalar("acbr/rerank_count", float(acbr_rerank_count), global_step)
        writer.add_scalar(
            "acbr/rerank_changed_fraction",
            float(acbr_rerank_changed) / max(float(acbr_rerank_count), 1.0),
            global_step,
        )
        writer.add_scalar(
            "acbr/rerank_gated_fraction",
            float(acbr_rerank_gated) / max(float(acbr_rerank_count), 1.0),
            global_step,
        )
        writer.add_scalar(
            "acbr/rerank_score_mean",
            float(acbr_rerank_score_sum) / max(float(acbr_rerank_count), 1.0) if acbr_rerank_count > 0 else np.nan,
            global_step,
        )
        writer.add_scalar(
            "acbr/rerank_std_mean",
            float(acbr_rerank_std_sum) / max(float(acbr_rerank_count), 1.0) if acbr_rerank_count > 0 else np.nan,
            global_step,
        )
        writer.add_scalar(
            "acbr/predicted_improvement_mean",
            float(acbr_predicted_improvement_sum) / max(float(acbr_rerank_count), 1.0)
            if acbr_rerank_count > 0
            else np.nan,
            global_step,
        )
        writer.add_scalar("acbr/target_mean", float(last_acbr_target_mean.item()), global_step)
        writer.add_scalar("acbr/pred_mean", float(last_acbr_pred_mean.item()), global_step)
        writer.add_scalar("acbr/pred_std", float(last_acbr_pred_std.item()), global_step)
        writer.add_scalar("acbr/active_fraction", float(last_acbr_active_fraction.item()), global_step)
        writer.add_scalar("losses/planner_regret_loss", float(last_planner_regret_loss.item()), global_step)
        writer.add_scalar("losses/pr_nominal_prior_loss", float(last_pr_nominal_prior_loss.item()), global_step)
        writer.add_scalar("losses/prb_aux_loss", float(last_prb_aux_loss.item()), global_step)
        writer.add_scalar("prb/residual_total_loss", float(last_prb_residual_total_loss.item()), global_step)
        writer.add_scalar("prb/true_total_loss", float(last_prb_true_total_loss.item()), global_step)
        writer.add_scalar("prb/component_residual_loss", float(last_prb_component_residual_loss.item()), global_step)
        writer.add_scalar("prb/true_component_loss", float(last_prb_true_component_loss.item()), global_step)
        writer.add_scalar("prb/latent_norm_mean", float(last_prb_latent_norm_mean), global_step)
        writer.add_scalar("pr/alpha_eff", float(alpha_pr_eff), global_step)
        writer.add_scalar("pr/alpha_base", float(alpha_components["pr_alpha_base"]), global_step)
        writer.add_scalar("pr/schedule_mult", float(alpha_components["pr_schedule_mult"]), global_step)
        writer.add_scalar("pr/regret_gate", float(alpha_components["pr_regret_gate"]), global_step)
        writer.add_scalar("pr/cumulative_recovery_step", float(cumulative_recovery_step), global_step)
        writer.add_scalar("pr/grad_ratio_pr", float(last_grad_ratio_pr), global_step)
        writer.add_scalar("pr/grad_ratio_nominal", float(last_grad_ratio_nominal), global_step)
        writer.add_scalar("mra/rollout_cost_cutoff", float(mra_cutoff.item()) if torch.isfinite(mra_cutoff) else np.nan, global_step)
        writer.add_scalar("mra/rollout_risk_cost_mean", float(b_risk_costs_pre.mean().item()), global_step)
        writer.add_scalar("mra/rollout_risk_feature_mean", float(b_risk_features_pre.mean().item()), global_step)
        writer.add_scalar("mra/weight_max", float(b_mra_weights.max().item()), global_step)
        writer.add_scalar("mra/weight_mean", float(last_mra_weight_mean.item()), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        prb_diag: dict[str, float | int | bool | str] = {}
        if prb_builder is not None and isinstance(agent, CleanRLResidualBeliefActorCritic):
            with torch.no_grad():
                prb_predictions_full = agent.predict_prb_aux(b_obs, b_prb_features, b_actions)
                residual_error = prb_predictions_full["residual_total"].view(-1) - b_prb_residual_total_targets
                true_error = prb_predictions_full["true_total"].view(-1) - b_prb_true_total_targets
                component_mask_sum = float(b_prb_component_masks.sum().item())
                if component_mask_sum > 0.0:
                    component_residual_error = (
                        (prb_predictions_full["component_residual"] - b_prb_component_residual_targets).abs()
                        * b_prb_component_masks
                    ).sum() / max(component_mask_sum, 1.0)
                    component_true_error = (
                        (prb_predictions_full["true_component"] - b_prb_true_component_targets).abs()
                        * b_prb_component_masks
                    ).sum() / max(component_mask_sum, 1.0)
                else:
                    component_residual_error = torch.tensor(float("nan"), device=device)
                    component_true_error = torch.tensor(float("nan"), device=device)
                latent_norm = prb_predictions_full["latent"].norm(dim=-1)
                prb_diag = {
                    "prb_aux_loss": float(last_prb_aux_loss.item()),
                    "prb_residual_total_loss": float(last_prb_residual_total_loss.item()),
                    "prb_true_total_loss": float(last_prb_true_total_loss.item()),
                    "prb_component_residual_loss": float(last_prb_component_residual_loss.item()),
                    "prb_true_component_loss": float(last_prb_true_component_loss.item()),
                    "prb_residual_feature_dim": int(prb_feature_dim),
                    "prb_latent_dim": int(agent.prb_latent_dim),
                    "prb_latent_norm_mean": float(latent_norm.mean().item()),
                    "prb_latent_norm_std": float(latent_norm.std(unbiased=False).item()),
                    "prb_component_cost_available": bool(component_mask_sum > 0.0),
                    "prb_component_mask_fraction": float(b_prb_component_masks.mean().item()),
                    "residual_total_mae": float(residual_error.abs().mean().item()),
                    "residual_total_rmse": float(torch.sqrt((residual_error**2).mean()).item()),
                    "true_total_mae": float(true_error.abs().mean().item()),
                    "true_total_rmse": float(torch.sqrt((true_error**2).mean()).item()),
                    "component_residual_mae": float(component_residual_error.item()),
                    "component_true_cost_mae": float(component_true_error.item()),
                    "mean_residual_norm": float(b_prb_residual_total_targets.mean().item()),
                    "residual_norm_std": float(b_prb_residual_total_targets.std(unbiased=False).item()),
                    "mean_true_total_norm": float(b_prb_true_total_targets.mean().item()),
                    "bagr_enabled": bool(args.bagr_enabled),
                    "bagr_max_attack_variants": int(args.bagr_max_attack_variants),
                    "bagr_belief_temperature": float(args.bagr_belief_temperature),
                    "bagr_belief_prior_mix": float(args.bagr_belief_prior_mix),
                    "bagr_residual_l2_coef": float(args.bagr_residual_l2_coef),
                    "bagr_residual_barrier_coef": float(args.bagr_residual_barrier_coef),
                    "bagr_residual_action_limit": float(args.bagr_residual_action_limit),
                    "bagr_freeze_base_actor": bool(args.bagr_freeze_base_actor),
                    "bagr_residual_l2_loss": float(last_bagr_residual_l2_loss.item()),
                    "bagr_residual_barrier_loss": float(last_bagr_residual_barrier_loss.item()),
                    "bagr_residual_abs_mean": float(last_bagr_residual_abs_mean.item()),
                    "bagr_adaptive_limit_mean": float(last_bagr_adaptive_limit_mean.item()),
                    "bagr_belief_confidence_mean": float(last_bagr_belief_confidence_mean.item()),
                    "bagr_belief_entropy_mean": float(last_bagr_belief_entropy_mean.item()),
                    "trr_enabled": bool(args.trr_enabled),
                    "trr_anchor_checkpoint": str(args.trr_anchor_checkpoint or ""),
                    "trr_alpha": float(args.trr_alpha),
                    "trr_alpha_eff": float(trr_alpha_eff),
                    "trr_guidance_schedule": str(args.trr_guidance_schedule),
                    "trr_ramp_steps": int(args.trr_ramp_steps),
                    "trr_active_until_step": int(args.trr_active_until_step),
                    "trr_decay_to_zero_by_step": int(args.trr_decay_to_zero_by_step),
                    "trr_min_normalized_regret": float(args.trr_min_normalized_regret),
                    "trr_residual_l2_coef": float(args.trr_residual_l2_coef),
                    "trr_residual_barrier_coef": float(args.trr_residual_barrier_coef),
                    "trr_residual_action_limit": float(args.trr_residual_action_limit),
                    "trr_freeze_base_actor": bool(args.trr_freeze_base_actor),
                    "trr_teacher_loss": float(last_trr_teacher_loss.item()),
                    "trr_residual_l2_loss": float(last_trr_residual_l2_loss.item()),
                    "trr_residual_barrier_loss": float(last_trr_residual_barrier_loss.item()),
                    "trr_residual_abs_mean": float(last_trr_residual_abs_mean.item()),
                    "trr_target_residual_abs_mean": float(last_trr_target_residual_abs_mean.item()),
                    "trr_active_fraction": float(last_trr_active_fraction.item()),
                    "official_attack_only": bool(
                        bool(environment_attack_config.get("enabled", False))
                        and str(environment_attack_config.get("type", "")) != "env_attack_mixture"
                    ),
                }
        pr_summary = dict(pre_update_pr_summary)
        pr_summary.update(
            {
                "algo": (
                    "acbr_ppo"
                    if bool(args.acbr_enabled)
                    else (
                        "trr_ppo"
                        if bool(args.trr_enabled)
                        else (
                            "bagr_ppo"
                            if bool(args.bagr_enabled)
                            else ("prb_ppo" if bool(args.prb_enabled) else ("pr_ppo" if bool(args.pr_enabled) else "ppo"))
                        )
                    )
                ),
                "global_step": int(global_step),
                "update": int(update),
                "recovery_step_offset": int(args.recovery_step_offset),
                "local_global_step": int(global_step),
                "local_train_step": int(global_step),
                "cumulative_recovery_step": int(cumulative_recovery_step),
                "official_attack_only": bool(
                    bool(environment_attack_config.get("enabled", False))
                    and str(environment_attack_config.get("type", "")) != "env_attack_mixture"
                ),
                "pr_num_candidates_per_state": int(args.pr_num_candidates),
                "pr_game_teacher_enabled": bool(args.pr_game_teacher_enabled),
                "pr_game_teacher_mode": str(args.pr_game_teacher_mode),
                "pr_game_max_attack_variants": int(args.pr_game_max_attack_variants),
                "pr_include_structured_candidates": bool(args.pr_include_structured_candidates),
                "pr_num_structured_candidates": int(args.pr_num_structured_candidates),
                "pr_alpha_base": float(alpha_components["pr_alpha_base"]),
                "pr_alpha_eff": float(alpha_pr_eff),
                "pr_alpha_target": float(args.alpha_pr),
                "pr_alpha_runtime": float(alpha_pr_runtime),
                "alpha_pr": float(args.alpha_pr),
                "pr_ramp_steps": int(args.pr_ramp_steps),
                "pr_ramp_mult": float(alpha_components["pr_ramp_mult"]),
                "pr_schedule_mult": float(alpha_components["pr_schedule_mult"]),
                "pr_regret_gate": float(alpha_components["pr_regret_gate"]),
                "pr_guidance_schedule": str(args.pr_guidance_schedule),
                "pr_active_until_step": int(args.pr_active_until_step),
                "pr_decay_to_zero_by_step": int(args.pr_decay_to_zero_by_step),
                "beta_nominal": float(args.beta_nominal),
                "pr_aux_loss_type": str(args.pr_aux_loss_type),
                "pr_pair_reference": str(args.pr_pair_reference),
                "pr_target_type": str(args.pr_target_type),
                "pr_soft_temperature": float(args.pr_soft_temperature),
                "pr_cpa_temperature": float(args.pr_cpa_temperature),
                "pr_pref_temperature": float(args.pr_pref_temperature),
                "pr_adv_temperature": float(args.pr_adv_temperature),
                "pr_local_sigma": float(args.pr_local_sigma),
                "pr_local_sigmas": ",".join(str(value) for value in pr_local_sigmas),
                "pr_query_strategy": str(args.pr_query_strategy),
                "ppo_loss": float((pg_loss + args.vf_coef * v_loss - args.ent_coef * entropy_loss).detach().item()),
                "policy_loss": float(pg_loss.detach().item()),
                "value_loss": float(v_loss.detach().item()),
                "entropy_loss": float(entropy_loss.detach().item()),
                "planner_regret_loss": float(last_planner_regret_loss.item()),
                "nominal_prior_loss": float(last_pr_nominal_prior_loss.item()),
                "trr_enabled": bool(args.trr_enabled),
                "trr_alpha": float(args.trr_alpha),
                "trr_alpha_eff": float(trr_alpha_eff),
                "trr_ramp_mult": float(trr_components["trr_ramp_mult"]),
                "trr_schedule_mult": float(trr_components["trr_schedule_mult"]),
                "trr_guidance_schedule": str(args.trr_guidance_schedule),
                "trr_min_normalized_regret": float(args.trr_min_normalized_regret),
                "trr_teacher_loss": float(last_trr_teacher_loss.item()),
                "trr_residual_l2_loss": float(last_trr_residual_l2_loss.item()),
                "trr_residual_barrier_loss": float(last_trr_residual_barrier_loss.item()),
                "trr_residual_abs_mean": float(last_trr_residual_abs_mean.item()),
                "trr_target_residual_abs_mean": float(last_trr_target_residual_abs_mean.item()),
                "trr_active_fraction": float(last_trr_active_fraction.item()),
                "acbr_enabled": bool(args.acbr_enabled),
                "acbr_anchor_checkpoint": str(args.acbr_anchor_checkpoint or ""),
                "acbr_critic_coef": float(args.acbr_critic_coef),
                "acbr_uncertainty_coef": float(args.acbr_uncertainty_coef),
                "acbr_anchor_penalty": float(args.acbr_anchor_penalty),
                "acbr_policy_penalty": float(args.acbr_policy_penalty),
                "acbr_target_clip": float(args.acbr_target_clip),
                "acbr_rerank_start_after_steps": int(args.acbr_rerank_start_after_steps),
                "acbr_benefit_gate_enabled": bool(args.acbr_benefit_gate_enabled),
                "acbr_benefit_margin": float(args.acbr_benefit_margin),
                "acbr_critic_loss": float(last_acbr_critic_loss.item()),
                "acbr_target_mean": float(last_acbr_target_mean.item()),
                "acbr_pred_mean": float(last_acbr_pred_mean.item()),
                "acbr_pred_std": float(last_acbr_pred_std.item()),
                "acbr_active_fraction": float(last_acbr_active_fraction.item()),
                "acbr_rerank_count": int(acbr_rerank_count),
                "acbr_rerank_changed_fraction": float(acbr_rerank_changed) / max(float(acbr_rerank_count), 1.0),
                "acbr_rerank_gated_fraction": float(acbr_rerank_gated) / max(float(acbr_rerank_count), 1.0),
                "acbr_rerank_score_mean": (
                    float(acbr_rerank_score_sum) / max(float(acbr_rerank_count), 1.0)
                    if acbr_rerank_count > 0
                    else float("nan")
                ),
                "acbr_rerank_std_mean": (
                    float(acbr_rerank_std_sum) / max(float(acbr_rerank_count), 1.0)
                    if acbr_rerank_count > 0
                    else float("nan")
                ),
                "acbr_predicted_improvement_mean": (
                    float(acbr_predicted_improvement_sum) / max(float(acbr_rerank_count), 1.0)
                    if acbr_rerank_count > 0
                    else float("nan")
                ),
                "total_loss": float(last_total_loss.item()),
                "grad_norm_ppo_actor": float(last_grad_norm_ppo_actor),
                "grad_norm_pr_actor": float(last_grad_norm_pr_actor),
                "grad_norm_nominal_actor": float(last_grad_norm_nominal_actor),
                "grad_ratio_pr": float(last_grad_ratio_pr),
                "grad_ratio_nominal": float(last_grad_ratio_nominal),
                "pr_pairwise_pref_loss": float(last_pairwise_pref_loss),
                "pr_num_pairwise_states": int(last_pairwise_num_states),
                "pr_num_positive_pairs": int(last_pairwise_num_pairs),
                "pr_mean_positive_pairs_per_state": float(last_pairwise_pairs_per_state),
                "pr_fraction_states_with_positive_pairs": float(last_pairwise_fraction_states),
                "pr_mean_pair_advantage": float(last_pairwise_mean_advantage),
                "pr_max_pair_advantage": float(last_pairwise_max_advantage),
                "pr_mean_pair_logp_margin": float(last_pairwise_mean_logp_margin),
                "pr_mean_reference_logp_margin": float(last_pairwise_mean_ref_logp_margin),
                "pr_mean_pair_z": float(last_pairwise_mean_z),
                **prb_diag,
            }
        )
        pr_chunk_records.append(pr_summary)
        pr_buffer.clear()

        if args.n_eval_episodes > 0 and (global_step - last_eval_step >= args.eval_freq or update == num_updates):
            last_eval_step = global_step
            eval_metrics = evaluate_agent(
                agent,
                map_size=args.map_size,
                scenario=args.scenario,
                seed=int(args.eval_seed) if args.eval_seed is not None else args.seed + 50_000 + global_step,
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
                bagr_enabled=bool(args.bagr_enabled),
                bagr_max_attack_variants=int(args.bagr_max_attack_variants),
                bagr_belief_temperature=float(args.bagr_belief_temperature),
                bagr_belief_prior_mix=float(args.bagr_belief_prior_mix),
                pr_failure_cost=float(args.pr_failure_cost),
                acbr_enabled=bool(args.acbr_enabled),
                acbr_anchor_agent=acbr_anchor_agent,
                acbr_context_dim=int(args.acbr_context_dim),
                acbr_num_candidates=int(args.pr_num_candidates),
                acbr_num_random_candidates=int(args.pr_num_random_candidates),
                acbr_num_structured_candidates=int(args.pr_num_structured_candidates),
                acbr_local_sigma=float(args.pr_local_sigma),
                acbr_risk_local_sigma=float(args.pr_risk_local_sigma),
                acbr_uncertainty_coef=float(args.acbr_uncertainty_coef),
                acbr_anchor_penalty=float(args.acbr_anchor_penalty),
                acbr_policy_penalty=float(args.acbr_policy_penalty),
                acbr_benefit_gate_enabled=bool(args.acbr_benefit_gate_enabled),
                acbr_benefit_margin=float(args.acbr_benefit_margin),
            )
            writer.add_scalar("eval/mean_reward", eval_metrics["mean_reward"], global_step)
            writer.add_scalar("eval/success_rate", eval_metrics["success_rate"], global_step)
            writer.add_scalar("eval/mean_scalar_cost", eval_metrics["mean_scalar_cost"], global_step)
            writer.add_scalar(
                "eval/mean_attacked_scalar_cost",
                eval_metrics["mean_attacked_scalar_cost"],
                global_step,
            )
            writer.add_scalar(
                "eval/mean_lambda_uncertainty",
                eval_metrics["mean_lambda_uncertainty"],
                global_step,
            )
            eval_records.append({"global_step": int(global_step), **eval_metrics})
            pd.DataFrame(eval_records).to_csv(run_dir / "eval_metrics.csv", index=False)

            min_eval_delta = max(float(args.min_eval_delta), 0.0)
            is_best = (
                best_mean_reward is None
                or eval_metrics["mean_reward"] > best_mean_reward + min_eval_delta
            )
            if is_best:
                best_mean_reward = eval_metrics["mean_reward"]
                stale_eval_count = 0
                save_cleanrl_checkpoint(
                    run_dir / "best_model.pt",
                    agent,
                    config,
                    global_step,
                    best_mean_reward,
                )
            else:
                stale_eval_count += 1

            print(
                f"step={global_step} "
                f"rollout_reward={np.mean(rollout_rewards):.4f} "
                f"eval_reward={eval_metrics['mean_reward']:.4f} "
                f"eval_success={eval_metrics['success_rate']:.3f} "
                f"eval_attacked_cost={eval_metrics['mean_attacked_scalar_cost']:.4f} "
                f"eval_lambda={eval_metrics['mean_lambda_uncertainty']:.3f} "
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

    save_cleanrl_checkpoint(
        run_dir / "final_model.pt",
        agent,
        config,
        global_step,
        best_mean_reward,
    )
    if args.rollout_metrics_path:
        rollout_metrics_path = Path(args.rollout_metrics_path)
        rollout_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rollout_metric_records).to_csv(rollout_metrics_path, index=False)
    if args.pr_log_path:
        pr_log_path = Path(args.pr_log_path)
        pr_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pr_chunk_records).to_csv(pr_log_path, index=False)
    if args.prb_log_path:
        prb_log_path = Path(args.prb_log_path)
        prb_log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pr_chunk_records).to_csv(prb_log_path, index=False)

    for env in envs:
        env.close()
    writer.close()

    print(f"Saved final model to {run_dir / 'final_model.pt'}")
    print(f"Best model path: {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
