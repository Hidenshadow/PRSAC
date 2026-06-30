"""Main RL-baseline shock-recovery experiment.

Protocol:

1. Train an RL policy on the clean environment.
2. Evaluate the trained nominal policy on clean and attacked maps.
3. Fine-tune the same policy under the environment/map attack and evaluate
   recovery checkpoints.

This is intentionally simpler than sequential staged recovery. It is the main
RL baseline protocol for the belief-vs-true map robustness benchmark.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.attack_wrappers import attack_enabled
from maps.real_terrain import load_real_layers
from run_attack_recovery_finetune import (
    build_train_command,
    checkpoint_step,
    clean_output_dir,
    config_value,
    evaluate_checkpoint,
    generate_episodes,
    load_base_args,
    run_name_for_algo,
    training_script_for_algo,
)
from run_lunar_viper_staged_recovery import (
    command_from_args,
    generate_real_episodes,
    load_environment_attack,
    prepare_splits,
    read_json,
    resolved_base_args,
    write_json,
)
from utils.metrics import DEFAULT_MAP_SEED_POOL_SIZE
from utils.paper_metrics import summarize_shock_recovery_frame


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "ppo_shock_recovery"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean RL training -> attack shock -> attack recovery.")
    parser.add_argument("--algo", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--level-config", type=Path, default=PROJECT_ROOT / "configs" / "levels" / "synthetic_corridor.json")
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nominal-timesteps", type=int, default=50_000)
    parser.add_argument("--recovery-timesteps", type=int, default=20_480)
    parser.add_argument("--eval-interval", type=int, default=1024)
    parser.add_argument("--num-eval-episodes", type=int, default=300)
    parser.add_argument("--train-eval-episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--in-domain-seed", type=int, default=909)
    parser.add_argument("--heldout-seed", type=int, default=1919)
    parser.add_argument("--map-pool-size", type=int, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional trainer device override, for example cpu, cuda, or cuda:0.",
    )
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--game-recovery-enabled",
        action="store_true",
        help="Use local Stackelberg/game-regularized recovery only during attack recovery.",
    )
    parser.add_argument(
        "--game-attack-mixture-size",
        type=int,
        default=5,
        help="Number of local attack variants sampled during recovery training, including the base attack.",
    )
    parser.add_argument(
        "--game-attack-jitter",
        type=float,
        default=0.15,
        help="Relative scale range for local attack variants around the benchmark attack.",
    )
    parser.add_argument(
        "--game-attack-sampler",
        choices=("fixed", "adaptive_bandit", "qre_minimax"),
        default="adaptive_bandit",
        help=(
            "Recovery attack sampler. adaptive_bandit shifts probability toward variants with higher rollout cost; "
            "qre_minimax uses a temperature-controlled soft worst-case adversary."
        ),
    )
    parser.add_argument(
        "--game-attack-variant-mode",
        choices=("scale", "component", "scale_component"),
        default="scale",
        help=(
            "Local attack population construction. scale varies attack strength; "
            "component samples individual composite attack mechanisms."
        ),
    )
    parser.add_argument(
        "--cdr-recovery-enabled",
        action="store_true",
        help=(
            "Enable curriculum domain-randomized recovery. Clean training and shock evaluation "
            "remain vanilla; recovery training samples a scheduled local attack population."
        ),
    )
    parser.add_argument(
        "--cdr-attack-mixture-size",
        type=int,
        default=5,
        help="Number of CDR attack variants sampled during recovery training, including the benchmark attack.",
    )
    parser.add_argument(
        "--cdr-attack-jitter-start",
        type=float,
        default=0.05,
        help="Initial relative scale range for CDR local attack variants.",
    )
    parser.add_argument(
        "--cdr-attack-jitter-end",
        type=float,
        default=0.25,
        help="Final relative scale range for CDR local attack variants.",
    )
    parser.add_argument(
        "--cdr-attack-variant-mode",
        choices=("scale", "component", "scale_component"),
        default="scale",
        help="CDR local attack population construction.",
    )
    parser.add_argument(
        "--cdr-benchmark-prob-start",
        type=float,
        default=0.70,
        help="Initial probability assigned to the benchmark attack in the CDR mixture.",
    )
    parser.add_argument(
        "--cdr-benchmark-prob-end",
        type=float,
        default=0.20,
        help="Final probability assigned to the benchmark attack in the CDR mixture.",
    )
    parser.add_argument(
        "--cdr-schedule",
        choices=("linear", "exp"),
        default="linear",
        help="CDR curriculum schedule from benchmark-heavy/easy to broader/harder attack randomization.",
    )
    parser.add_argument(
        "--game-bandit-eta",
        type=float,
        default=0.8,
        help="Exponentiated-gradient step size for adaptive attack probabilities.",
    )
    parser.add_argument(
        "--game-bandit-min-prob",
        type=float,
        default=0.05,
        help="Minimum probability floor for every local attack variant.",
    )
    parser.add_argument(
        "--game-bandit-prior-mix",
        type=float,
        default=0.10,
        help="Fraction of the fixed prior mixed back into adaptive probabilities after every update.",
    )
    parser.add_argument(
        "--game-bandit-cost-key",
        choices=("policy_cost", "scalar_cost", "attacked_scalar_cost", "soft_attacked_scalar_cost"),
        default="policy_cost",
        help="Rollout metric used by adaptive_bandit; higher cost receives higher future probability.",
    )
    parser.add_argument(
        "--game-bandit-benchmark-floor",
        type=float,
        default=0.0,
        help="Minimum probability reserved for the benchmark attack variant during adaptive recovery.",
    )
    parser.add_argument(
        "--qre-minimax-recovery-enabled",
        action="store_true",
        help=(
            "Enable standalone QRE-Minimax recovery: a bounded-rational adversary "
            "updates attack-variant probabilities during attacked recovery only."
        ),
    )
    parser.add_argument("--qre-temperature-start", type=float, default=2.0)
    parser.add_argument("--qre-temperature-end", type=float, default=0.25)
    parser.add_argument(
        "--qre-temperature-schedule",
        choices=("linear", "exp"),
        default="exp",
        help="Temperature curriculum from high-entropy adversary to near worst-case adversary.",
    )
    parser.add_argument(
        "--qre-response-rate",
        type=float,
        default=0.75,
        help="Blend rate from previous attack probabilities toward the current QRE soft best response.",
    )
    parser.add_argument(
        "--qre-cost-ema-beta",
        type=float,
        default=0.60,
        help="EMA retention for per-variant rollout costs used by the QRE adversary.",
    )
    parser.add_argument(
        "--qre-prior-mix",
        type=float,
        default=0.10,
        help="Fraction of initial attack prior mixed into the QRE adversary after every update.",
    )
    parser.add_argument("--qre-min-prob", type=float, default=0.03)
    parser.add_argument(
        "--qre-max-prob-cap",
        type=float,
        default=0.55,
        help="Maximum probability assigned to any single attack variant during QRE recovery; use 0 to disable.",
    )
    parser.add_argument(
        "--qre-exploration-bonus",
        type=float,
        default=0.10,
        help="Count-based score bonus for under-sampled QRE attack variants.",
    )
    parser.add_argument("--qre-benchmark-floor", type=float, default=0.20)
    parser.add_argument(
        "--qre-cost-normalization",
        choices=("std", "range", "none"),
        default="std",
        help="Scale cost estimates before the quantal response softmax.",
    )
    parser.add_argument(
        "--ap-cvar-enabled",
        action="store_true",
        help=(
            "Enable Adversarial-Population CVaR recovery: adaptive attack population "
            "plus tail-weighted PPO updates during recovery chunks only."
        ),
    )
    parser.add_argument("--ap-cvar-quantile", type=float, default=0.75)
    parser.add_argument("--ap-cvar-weight", type=float, default=1.5)
    parser.add_argument("--ap-cvar-tail-excess-weight", type=float, default=0.75)
    parser.add_argument("--ap-cvar-risk-feature-weight", type=float, default=0.25)
    parser.add_argument("--ap-cvar-weight-cap", type=float, default=4.0)
    parser.add_argument("--ap-cvar-tail-reward-penalty", type=float, default=0.0)
    parser.add_argument("--ap-cvar-risk-reward-penalty", type=float, default=0.0)
    parser.add_argument(
        "--planner-regret-recovery-enabled",
        action="store_true",
        help=(
            "Enable recovery-only planner-regret guidance. During attacked recovery, "
            "candidate planner actions are evaluated and better candidates add an auxiliary PPO loss."
        ),
    )
    parser.add_argument("--pr-recovery-alpha", type=float, default=0.50)
    parser.add_argument("--pr-recovery-beta-nominal", type=float, default=0.0)
    parser.add_argument("--pr-recovery-aux-loss-type", choices=("mse", "nll", "cpa", "pairwise_pref"), default="cpa")
    parser.add_argument("--pr-recovery-query-fraction", type=float, default=0.25)
    parser.add_argument("--pr-recovery-query-interval", type=int, default=4)
    parser.add_argument("--pr-recovery-num-candidates", type=int, default=24)
    parser.add_argument("--pr-recovery-num-random-candidates", type=int, default=8)
    parser.add_argument("--pr-recovery-local-sigma", type=float, default=0.10)
    parser.add_argument("--pr-recovery-risk-local-sigma", type=float, default=0.20)
    parser.add_argument("--pr-recovery-cpa-temperature", type=float, default=0.03)
    parser.add_argument("--pr-recovery-min-positive-adv", type=float, default=0.001)
    parser.add_argument("--pr-recovery-regret-weight-max", type=float, default=3.0)
    parser.add_argument("--pr-recovery-ramp-steps", type=int, default=1024)
    parser.add_argument("--pr-recovery-grad-ratio-controller", action="store_true")
    parser.add_argument(
        "--game-teacher-recovery-enabled",
        action="store_true",
        help=(
            "Enable recovery-only game-theoretic planner-teacher distillation. "
            "Candidate planner actions are scored by a local minimax game over attack variants."
        ),
    )
    parser.add_argument("--gt-recovery-alpha", type=float, default=0.70)
    parser.add_argument("--gt-recovery-aux-loss-type", choices=("mse", "nll", "cpa", "pairwise_pref"), default="cpa")
    parser.add_argument("--gt-recovery-query-fraction", type=float, default=0.12)
    parser.add_argument("--gt-recovery-query-interval", type=int, default=16)
    parser.add_argument("--gt-recovery-num-candidates", type=int, default=16)
    parser.add_argument("--gt-recovery-num-random-candidates", type=int, default=4)
    parser.add_argument("--gt-recovery-num-structured-candidates", type=int, default=8)
    parser.add_argument("--gt-recovery-local-sigma", type=float, default=0.10)
    parser.add_argument("--gt-recovery-risk-local-sigma", type=float, default=0.18)
    parser.add_argument("--gt-recovery-cpa-temperature", type=float, default=0.04)
    parser.add_argument("--gt-recovery-min-positive-adv", type=float, default=0.001)
    parser.add_argument("--gt-recovery-regret-weight-max", type=float, default=3.0)
    parser.add_argument("--gt-recovery-ramp-steps", type=int, default=1024)
    parser.add_argument("--gt-recovery-max-attack-variants", type=int, default=6)
    parser.add_argument("--gt-recovery-teacher-mode", choices=("minimax", "soft_stackelberg"), default="minimax")
    parser.add_argument("--gt-recovery-softmax-temperature", type=float, default=0.08)
    parser.add_argument(
        "--teacher-residual-recovery-enabled",
        action="store_true",
        help=(
            "Enable Teacher-Residual Recovery PPO during attacked recovery only. "
            "The clean PPO checkpoint remains the anchor; a residual adapter is trained "
            "toward gated planner/game-teacher actions."
        ),
    )
    parser.add_argument("--trr-recovery-alpha", type=float, default=1.0)
    parser.add_argument("--trr-recovery-query-fraction", type=float, default=0.25)
    parser.add_argument("--trr-recovery-query-interval", type=int, default=8)
    parser.add_argument("--trr-recovery-num-candidates", type=int, default=20)
    parser.add_argument("--trr-recovery-num-random-candidates", type=int, default=4)
    parser.add_argument("--trr-recovery-num-structured-candidates", type=int, default=10)
    parser.add_argument("--trr-recovery-local-sigma", type=float, default=0.10)
    parser.add_argument("--trr-recovery-risk-local-sigma", type=float, default=0.18)
    parser.add_argument("--trr-recovery-min-normalized-regret", type=float, default=0.01)
    parser.add_argument("--trr-recovery-regret-weight-max", type=float, default=3.0)
    parser.add_argument("--trr-recovery-ramp-steps", type=int, default=1024)
    parser.add_argument("--trr-recovery-active-until-step", type=int, default=4096)
    parser.add_argument("--trr-recovery-decay-to-zero-by-step", type=int, default=12288)
    parser.add_argument("--trr-recovery-residual-l2-coef", type=float, default=0.05)
    parser.add_argument("--trr-recovery-residual-barrier-coef", type=float, default=1.0)
    parser.add_argument("--trr-recovery-residual-action-limit", type=float, default=0.22)
    parser.add_argument("--trr-recovery-aux-coef", type=float, default=0.05)
    parser.add_argument("--trr-recovery-latent-dim", type=int, default=64)
    parser.add_argument("--trr-recovery-hidden-dim", type=int, default=64)
    parser.add_argument("--trr-recovery-feature-clip", type=float, default=5.0)
    parser.add_argument("--trr-recovery-max-attack-variants", type=int, default=6)
    parser.add_argument("--trr-recovery-teacher-mode", choices=("minimax", "soft_stackelberg"), default="minimax")
    parser.add_argument("--trr-recovery-softmax-temperature", type=float, default=0.08)
    parser.add_argument("--trr-recovery-freeze-base-actor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--acbr-recovery-enabled",
        action="store_true",
        help=(
            "Enable Attack-Context Belief Reranking PPO during attacked recovery only. "
            "A planner-trained action critic reranks local PPO/anchor/structured candidates."
        ),
    )
    parser.add_argument(
        "--bvr-recovery-enabled",
        action="store_true",
        help=(
            "Enable same-protocol BVR-PPO recovery. This starts from the clean nominal "
            "checkpoint at attack recovery time and uses a benefit-gated local route "
            "verifier inside the PPO recovery loop."
        ),
    )
    parser.add_argument("--acbr-recovery-critic-coef", type=float, default=0.50)
    parser.add_argument("--acbr-recovery-query-fraction", type=float, default=0.50)
    parser.add_argument("--acbr-recovery-query-interval", type=int, default=4)
    parser.add_argument("--acbr-recovery-num-candidates", type=int, default=24)
    parser.add_argument("--acbr-recovery-num-random-candidates", type=int, default=6)
    parser.add_argument("--acbr-recovery-num-structured-candidates", type=int, default=10)
    parser.add_argument("--acbr-recovery-local-sigma", type=float, default=0.12)
    parser.add_argument("--acbr-recovery-risk-local-sigma", type=float, default=0.20)
    parser.add_argument("--acbr-recovery-uncertainty-coef", type=float, default=0.25)
    parser.add_argument("--acbr-recovery-anchor-penalty", type=float, default=0.15)
    parser.add_argument("--acbr-recovery-policy-penalty", type=float, default=0.05)
    parser.add_argument("--acbr-recovery-target-clip", type=float, default=3.0)
    parser.add_argument("--acbr-recovery-rerank-start-after-steps", type=int, default=1024)
    parser.add_argument("--acbr-recovery-benefit-gate-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--acbr-recovery-benefit-margin", type=float, default=0.005)
    parser.add_argument("--acbr-recovery-aux-coef", type=float, default=0.03)
    parser.add_argument("--acbr-recovery-latent-dim", type=int, default=64)
    parser.add_argument("--acbr-recovery-hidden-dim", type=int, default=64)
    parser.add_argument("--acbr-recovery-feature-clip", type=float, default=5.0)
    parser.add_argument("--acbr-recovery-max-attack-variants", type=int, default=6)
    parser.add_argument("--acbr-recovery-teacher-mode", choices=("minimax", "soft_stackelberg"), default="minimax")
    parser.add_argument("--acbr-recovery-softmax-temperature", type=float, default=0.08)
    parser.add_argument(
        "--sac-game-recovery-enabled",
        action="store_true",
        help=(
            "Enable conservative game-aware SAC recovery. "
            "Clean training and shock evaluation remain vanilla SAC; only attacked recovery chunks use this."
        ),
    )
    parser.add_argument("--sac-game-anchor-coef", type=float, default=0.25)
    parser.add_argument("--sac-game-advantage-coef", type=float, default=0.10)
    parser.add_argument("--sac-game-q-margin", type=float, default=0.02)
    parser.add_argument("--sac-game-gate-temperature", type=float, default=0.05)
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
    )
    parser.add_argument("--sac-recovery-deterministic-actor-update", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sac-recovery-target-entropy-scale", type=float, default=1.0)
    parser.add_argument("--sac-recovery-fixed-alpha", type=float, default=None)
    parser.add_argument("--sac-recovery-rollout-deterministic-prob", type=float, default=0.0)
    parser.add_argument("--sac-recovery-rollout-noise-std", type=float, default=0.0)
    parser.add_argument("--sac-recovery-log-std-penalty-coef", type=float, default=0.0)
    parser.add_argument("--sac-recovery-log-std-target", type=float, default=-1.5)
    parser.add_argument(
        "--valt-sac-recovery-enabled",
        action="store_true",
        help=(
            "Enable VALT-SAC recovery-only training: virtual alternative observation perturbations "
            "regularize SAC under corrupted terrain belief."
        ),
    )
    parser.add_argument("--valt-sac-eps-start", type=float, default=0.0)
    parser.add_argument("--valt-sac-eps-end", type=float, default=0.08)
    parser.add_argument("--valt-sac-kappa-start", type=float, default=0.0)
    parser.add_argument("--valt-sac-kappa-end", type=float, default=0.30)
    parser.add_argument("--valt-sac-schedule-steps", type=int, default=20_480)
    parser.add_argument("--valt-sac-schedule", choices=("constant", "linear", "cosine", "exp"), default="linear")
    parser.add_argument("--valt-sac-bound-iters", type=int, default=2)
    parser.add_argument("--valt-sac-worst-step-size", type=float, default=0.0)
    parser.add_argument("--valt-sac-sgld-noise", type=float, default=0.0)
    parser.add_argument("--valt-sac-policy-reg-coef", type=float, default=1.0)
    parser.add_argument("--valt-sac-random-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--valt-sac-clip-low", type=float, default=0.0)
    parser.add_argument("--valt-sac-clip-high", type=float, default=1.0)
    parser.add_argument("--valt-sac-attack-deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--bagr-recovery-enabled",
        action="store_true",
        help=(
            "Enable Belief-Adaptive Game Recovery PPO during attacked recovery only. "
            "This uses attack-posterior residual features and a constrained residual policy."
        ),
    )
    parser.add_argument("--bagr-recovery-aux-coef", type=float, default=0.10)
    parser.add_argument("--bagr-recovery-latent-dim", type=int, default=64)
    parser.add_argument("--bagr-recovery-hidden-dim", type=int, default=64)
    parser.add_argument("--bagr-recovery-feature-clip", type=float, default=5.0)
    parser.add_argument("--bagr-recovery-max-attack-variants", type=int, default=6)
    parser.add_argument("--bagr-recovery-belief-temperature", type=float, default=0.25)
    parser.add_argument("--bagr-recovery-belief-prior-mix", type=float, default=0.10)
    parser.add_argument("--bagr-recovery-residual-l2-coef", type=float, default=0.10)
    parser.add_argument("--bagr-recovery-residual-barrier-coef", type=float, default=2.0)
    parser.add_argument("--bagr-recovery-residual-action-limit", type=float, default=0.18)
    parser.add_argument("--bagr-recovery-confidence-limit-scale", type=float, default=1.0)
    parser.add_argument("--bagr-recovery-freeze-base-actor", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--game-nominal-prior-coef", type=float, default=0.0)
    parser.add_argument("--game-lambda-drift-coef", type=float, default=0.0)
    parser.add_argument("--game-lambda-drift-margin", type=float, default=0.10)
    parser.add_argument("--game-risk-drift-coef", type=float, default=0.0)
    parser.add_argument("--game-risk-drift-margin", type=float, default=0.10)
    parser.add_argument(
        "--game-risk-action-indices",
        type=str,
        default="energy,hazard,communication,illumination",
        help="Comma-separated objective action names or indices regularized as conservative-risk channels.",
    )
    return parser.parse_args()


def disabled_attack() -> dict[str, Any]:
    return {"enabled": False}


GAME_ATTACK_SCALE_KEYS = {
    "attack_strength",
    "error_scale",
    "background_error_scale",
    "degradation_scale",
    "background_degradation_scale",
    "confidence_penalty_scale",
    "slope_underestimate_scale",
}
GAME_ATTACK_FRACTION_KEYS = {
    "top_fraction",
    "lower_slope_ratio",
}
GAME_ATTACK_UNIT_INTERVAL_KEYS = {
    "shortcut_attraction",
    "shortcut_consequence_weight",
}


def _scaled_numeric_attack_value(key: str, value: Any, scale: float) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    if key in GAME_ATTACK_SCALE_KEYS:
        return float(max(0.0, float(value) * float(scale)))
    if key in GAME_ATTACK_UNIT_INTERVAL_KEYS:
        return float(np.clip(float(value) * float(scale), 0.0, 1.0))
    if key in GAME_ATTACK_FRACTION_KEYS:
        # Keep spatial selections local: vary fractions, but less aggressively than strengths.
        fraction_scale = 1.0 + 0.5 * (float(scale) - 1.0)
        return float(np.clip(float(value) * fraction_scale, 0.01, 0.95))
    return value


def scale_attack_config(config: Any, scale: float) -> Any:
    if isinstance(config, dict):
        return {
            key: scale_attack_config(_scaled_numeric_attack_value(key, value, scale), scale)
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [scale_attack_config(item, scale) for item in config]
    return config


def local_attack_scales(size: int, jitter: float) -> list[float]:
    size = max(int(size), 1)
    jitter = max(float(jitter), 0.0)
    scales = [1.0]
    if size == 1 or jitter <= 0.0:
        return scales
    magnitudes = np.linspace(jitter, jitter / max(size - 1, 1), num=size - 1)
    for magnitude in magnitudes:
        for offset in (float(magnitude), -float(magnitude)):
            scale = float(max(0.05, 1.0 + offset))
            scales.append(scale)
            if len(scales) >= size:
                return scales
    while len(scales) < size:
        scales.append(1.0)
    return scales


def attack_variant_slug(*parts: Any) -> str:
    raw = "_".join(str(part) for part in parts if part not in (None, ""))
    slug = "".join(char.lower() if char.isalnum() else "_" for char in raw)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")[:64] or "attack"


def benchmark_attack_variant(env_attack: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant_id": "benchmark",
        "scale": 1.0,
        "config": copy.deepcopy(env_attack),
    }


def component_attack_variants(env_attack: dict[str, Any]) -> list[dict[str, Any]]:
    if str(env_attack.get("type", "")) != "env_composite":
        return []
    components = env_attack.get("components", [])
    if not isinstance(components, list):
        return []
    variants: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        if not isinstance(component, dict) or not bool(component.get("enabled", True)):
            continue
        component_config = copy.deepcopy(env_attack)
        component_copy = copy.deepcopy(component)
        component_config["components"] = [component_copy]
        component_config["name"] = (
            f"{env_attack.get('name', 'composite_attack')}_component_{index:02d}_"
            f"{attack_variant_slug(component_copy.get('type'), component_copy.get('mode'), component_copy.get('selection_mode'))}"
        )
        variants.append(
            {
                "variant_id": (
                    f"component_{index:02d}_"
                    f"{attack_variant_slug(component_copy.get('type'), component_copy.get('mode'), component_copy.get('selection_mode'))}"
                ),
                "scale": 1.0,
                "component_index": int(index),
                "component_type": str(component_copy.get("type", "")),
                "component_mode": str(component_copy.get("mode", "")),
                "config": component_config,
            }
        )
    return variants


def attack_variants_for_mode(
    env_attack: dict[str, Any],
    mode: str,
    mixture_size: int,
    jitter: float,
) -> list[dict[str, Any]]:
    mode = str(mode)
    variants = [benchmark_attack_variant(env_attack)]
    if mode in {"component", "scale_component"}:
        variants.extend(component_attack_variants(env_attack))
    if mode in {"scale", "scale_component"}:
        scales = local_attack_scales(mixture_size, jitter)
        for scale in scales[1:]:
            variant_config = scale_attack_config(env_attack, scale)
            variants.append(
                {
                    "variant_id": f"local_scale_{scale:.3f}",
                    "scale": float(scale),
                    "config": variant_config,
                }
            )
    if len(variants) == 1 and mode == "component":
        scales = local_attack_scales(mixture_size, jitter)
        for scale in scales[1:]:
            variants.append(
                {
                    "variant_id": f"local_scale_{scale:.3f}",
                    "scale": float(scale),
                    "config": scale_attack_config(env_attack, scale),
                }
            )
    seen: set[str] = set()
    unique_variants: list[dict[str, Any]] = []
    for variant in variants:
        variant_id = str(variant.get("variant_id", "variant"))
        if variant_id in seen:
            suffix = 2
            base_id = variant_id
            while f"{base_id}_{suffix}" in seen:
                suffix += 1
            variant_id = f"{base_id}_{suffix}"
            variant["variant_id"] = variant_id
        seen.add(variant_id)
        unique_variants.append(variant)
    return unique_variants


def game_attack_variants(env_attack: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    return attack_variants_for_mode(
        env_attack,
        str(args.game_attack_variant_mode),
        int(args.game_attack_mixture_size),
        float(args.game_attack_jitter),
    )


def normalized_probs(weights: list[float] | np.ndarray) -> list[float]:
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        return []
    values = np.where(np.isfinite(values), values, 0.0)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        values = np.ones_like(values, dtype=np.float64)
        total = float(values.sum())
    return (values / total).astype(float).tolist()


def enforce_benchmark_floor(probs: list[float] | np.ndarray, floor: float) -> list[float]:
    values = np.asarray(normalized_probs(probs), dtype=np.float64)
    if values.size <= 1:
        return values.astype(float).tolist()
    floor = float(np.clip(float(floor), 0.0, 1.0))
    if floor <= 0.0 or values[0] >= floor:
        return values.astype(float).tolist()
    rest = values[1:]
    rest_total = float(rest.sum())
    if rest_total <= 1e-12:
        rest = np.ones_like(rest, dtype=np.float64) / max(rest.size, 1)
    else:
        rest = rest / rest_total
    values = np.concatenate([[floor], rest * (1.0 - floor)])
    return normalized_probs(values)


def enforce_probability_cap(probs: list[float] | np.ndarray, cap: float) -> list[float]:
    values = np.asarray(normalized_probs(probs), dtype=np.float64)
    if values.size <= 1:
        return values.astype(float).tolist()
    cap = float(cap)
    if cap <= 0.0 or cap >= 1.0:
        return values.astype(float).tolist()
    cap = max(cap, 1.0 / float(values.size))
    for _ in range(values.size + 1):
        over = values > cap
        if not bool(over.any()):
            break
        excess = float(np.sum(values[over] - cap))
        values[over] = cap
        under = ~over
        capacity = np.clip(cap - values[under], 0.0, None)
        capacity_total = float(capacity.sum())
        if capacity_total <= 1e-12:
            break
        values[under] += excess * capacity / capacity_total
    return normalized_probs(values)


def initial_game_attack_probs(variant_count: int) -> list[float]:
    if int(variant_count) <= 0:
        return []
    return normalized_probs([2.0] + [1.0] * (int(variant_count) - 1))


def attack_probability_benchmark_floor(args: argparse.Namespace) -> float:
    if str(args.game_attack_sampler) == "qre_minimax" or bool(getattr(args, "qre_minimax_recovery_enabled", False)):
        return float(args.qre_benchmark_floor)
    return float(args.game_bandit_benchmark_floor)


def cdr_curriculum_progress(
    args: argparse.Namespace,
    chunk_index: int = 1,
    total_chunks: int = 1,
    recovery_step_offset: int = 0,
) -> float:
    del recovery_step_offset
    total = max(int(total_chunks), 1)
    if total <= 1:
        raw = 1.0
    else:
        raw = float(int(chunk_index) - 1) / float(total - 1)
    raw = float(np.clip(raw, 0.0, 1.0))
    if str(getattr(args, "cdr_schedule", "linear")) == "exp":
        scale = np.exp(3.0) - 1.0
        return float((np.exp(3.0 * raw) - 1.0) / scale)
    return raw


def cdr_attack_jitter(args: argparse.Namespace, progress: float) -> float:
    start = max(float(getattr(args, "cdr_attack_jitter_start", 0.05)), 0.0)
    end = max(float(getattr(args, "cdr_attack_jitter_end", 0.25)), 0.0)
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(start + (end - start) * progress)


def cdr_attack_variants(
    env_attack: dict[str, Any],
    args: argparse.Namespace,
    progress: float = 0.0,
) -> list[dict[str, Any]]:
    return attack_variants_for_mode(
        env_attack,
        str(getattr(args, "cdr_attack_variant_mode", "scale")),
        int(getattr(args, "cdr_attack_mixture_size", 5)),
        cdr_attack_jitter(args, progress),
    )


def cdr_attack_probs(variant_count: int, args: argparse.Namespace, progress: float) -> list[float]:
    count = int(variant_count)
    if count <= 0:
        return []
    if count == 1:
        return [1.0]
    progress = float(np.clip(progress, 0.0, 1.0))
    start = float(np.clip(float(getattr(args, "cdr_benchmark_prob_start", 0.70)), 0.0, 1.0))
    end = float(np.clip(float(getattr(args, "cdr_benchmark_prob_end", 0.20)), 0.0, 1.0))
    benchmark_prob = float(np.clip(start + (end - start) * progress, 0.0, 1.0))
    rest_prob = (1.0 - benchmark_prob) / float(count - 1)
    return normalized_probs([benchmark_prob] + [rest_prob] * (count - 1))


def build_game_recovery_attack(
    env_attack: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
    probs: list[float] | None = None,
    recovery_step_offset: int = 0,
    chunk_index: int = 1,
    total_chunks: int = 1,
) -> dict[str, Any]:
    if bool(getattr(args, "cdr_recovery_enabled", False)):
        progress = cdr_curriculum_progress(
            args,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            recovery_step_offset=recovery_step_offset,
        )
        variants = cdr_attack_variants(env_attack, args, progress)
        if len(variants) <= 1:
            return env_attack
        attack_probs = cdr_attack_probs(len(variants), args, progress)
        return {
            "enabled": True,
            "type": "env_attack_mixture",
            "name": "cdr_sac_recovery_mixture" if str(getattr(args, "algo", "")).lower() == "sac" else "cdr_recovery_mixture",
            "seed": int(seed),
            "sampler": "cdr_curriculum",
            "base_attack_name": str(env_attack.get("name", env_attack.get("type", "environment_attack"))),
            "variants": variants,
            "probs": attack_probs,
            "cdr_progress": float(progress),
            "cdr_attack_jitter": float(cdr_attack_jitter(args, progress)),
        }
    if not bool(args.game_recovery_enabled):
        return env_attack
    variants = game_attack_variants(env_attack, args)
    if len(variants) <= 1:
        return env_attack
    sampler = str(args.game_attack_sampler)
    attack_probs = enforce_benchmark_floor(
        probs if probs is not None else initial_game_attack_probs(len(variants)),
        attack_probability_benchmark_floor(args),
    )
    if sampler == "qre_minimax" and float(getattr(args, "qre_max_prob_cap", 0.0)) > 0.0:
        attack_probs = enforce_probability_cap(attack_probs, float(args.qre_max_prob_cap))
        attack_probs = enforce_benchmark_floor(attack_probs, attack_probability_benchmark_floor(args))
    if sampler == "qre_minimax":
        name = "qre_minimax_recovery_mixture"
    elif sampler == "adaptive_bandit":
        name = "adaptive_bandit_recovery_mixture"
    else:
        name = "local_stackelberg_recovery_mixture"
    return {
        "enabled": True,
        "type": "env_attack_mixture",
        "name": name,
        "seed": int(seed),
        "sampler": sampler,
        "base_attack_name": str(env_attack.get("name", env_attack.get("type", "environment_attack"))),
        "variants": variants,
        "probs": attack_probs,
    }


def attack_sampler_dir_name(sampler: str) -> str:
    return "qre_minimax" if str(sampler) == "qre_minimax" else "attack_bandit"


def attack_bandit_rollout_metrics_path(output_dir: Path, chunk_index: int, sampler: str = "adaptive_bandit") -> Path:
    return output_dir / attack_sampler_dir_name(sampler) / f"chunk_{chunk_index:04d}_rollout_metrics.csv"


def attack_bandit_history_path(output_dir: Path, sampler: str = "adaptive_bandit") -> Path:
    filename = "qre_minimax_history.csv" if str(sampler) == "qre_minimax" else "attack_bandit_history.csv"
    return output_dir / attack_sampler_dir_name(sampler) / filename


def qre_temperature_for_chunk(args: argparse.Namespace, chunk_index: int, total_chunks: int) -> float:
    start = max(float(args.qre_temperature_start), 1e-6)
    end = max(float(args.qre_temperature_end), 1e-6)
    total = max(int(total_chunks), 1)
    progress = float(np.clip(float(chunk_index) / float(total), 0.0, 1.0))
    if str(args.qre_temperature_schedule) == "linear":
        return float((1.0 - progress) * start + progress * end)
    return float(start * ((end / start) ** progress))


def normalized_qre_scores(cost_estimates: np.ndarray, mode: str) -> np.ndarray:
    scores = np.asarray(cost_estimates, dtype=np.float64)
    valid = np.isfinite(scores)
    if not bool(valid.any()):
        return np.zeros_like(scores, dtype=np.float64)
    filled = scores.copy()
    filled[~valid] = float(np.mean(scores[valid]))
    centered = filled - float(np.mean(filled))
    if str(mode) == "none":
        return centered
    if str(mode) == "range":
        scale = float(np.max(filled) - np.min(filled))
    else:
        scale = float(np.std(filled))
    if scale <= 1e-9:
        scale = max(float(np.max(np.abs(centered))), 1.0)
    return centered / scale


def qre_soft_best_response(scores: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    logits = np.asarray(scores, dtype=np.float64) / temp
    logits = logits - float(np.max(logits))
    probs = np.exp(logits)
    return probs / max(float(probs.sum()), 1e-12)


def update_attack_bandit_probs(
    current_probs: list[float],
    variants: list[dict[str, Any]],
    rollout_metrics_path: Path,
    args: argparse.Namespace,
    qre_state: dict[str, Any] | None = None,
    chunk_index: int = 1,
    total_chunks: int = 1,
) -> tuple[list[float], list[dict[str, Any]]]:
    sampler = str(args.game_attack_sampler)
    if sampler not in {"adaptive_bandit", "qre_minimax"} or not rollout_metrics_path.exists():
        return current_probs, []
    frame = pd.read_csv(rollout_metrics_path)
    cost_key = str(args.game_bandit_cost_key)
    if frame.empty or "variant_id" not in frame.columns or cost_key not in frame.columns:
        return current_probs, []

    frame = frame.copy()
    frame["variant_id"] = frame["variant_id"].astype(str)
    frame[cost_key] = pd.to_numeric(frame[cost_key], errors="coerce")
    frame = frame[(frame["variant_id"] != "") & frame[cost_key].notna()]
    if frame.empty:
        return current_probs, []

    grouped = frame.groupby("variant_id")[cost_key].agg(["mean", "count"])
    variant_ids = [str(variant.get("variant_id", f"variant_{index}")) for index, variant in enumerate(variants)]
    costs = np.asarray(
        [
            float(grouped.loc[variant_id, "mean"]) if variant_id in grouped.index else np.nan
            for variant_id in variant_ids
        ],
        dtype=np.float64,
    )
    counts = np.asarray(
        [
            int(grouped.loc[variant_id, "count"]) if variant_id in grouped.index else 0
            for variant_id in variant_ids
        ],
        dtype=np.int64,
    )
    valid = np.isfinite(costs) & (counts > 0)
    if not bool(valid.any()):
        return current_probs, []

    advantages = np.zeros_like(costs, dtype=np.float64)

    old_probs = np.asarray(normalized_probs(current_probs), dtype=np.float64)
    if old_probs.shape != costs.shape:
        old_probs = np.asarray(initial_game_attack_probs(len(variants)), dtype=np.float64)

    qre_temperature = np.nan
    qre_response_probs = np.full_like(old_probs, np.nan, dtype=np.float64)
    qre_score_before_bonus = np.full_like(old_probs, np.nan, dtype=np.float64)
    qre_exploration_bonus_values = np.zeros_like(old_probs, dtype=np.float64)
    qre_max_prob_cap = np.nan
    cost_estimates = costs.copy()
    update_mode = sampler
    if sampler == "qre_minimax":
        state = qre_state if qre_state is not None else {}
        old_ema = np.asarray(state.get("cost_ema", np.full_like(costs, np.nan)), dtype=np.float64)
        if old_ema.shape != costs.shape:
            old_ema = np.full_like(costs, np.nan)
        beta = float(np.clip(float(args.qre_cost_ema_beta), 0.0, 0.999))
        cost_estimates = old_ema.copy()
        for index, is_valid in enumerate(valid):
            if bool(is_valid):
                if np.isfinite(old_ema[index]):
                    cost_estimates[index] = beta * float(old_ema[index]) + (1.0 - beta) * float(costs[index])
                else:
                    cost_estimates[index] = float(costs[index])
        if np.isfinite(cost_estimates).any():
            fill_value = float(np.nanmean(cost_estimates))
        else:
            fill_value = float(np.mean(costs[valid]))
        cost_estimates = np.where(np.isfinite(cost_estimates), cost_estimates, fill_value)
        if qre_state is not None:
            qre_state["cost_ema"] = cost_estimates.astype(float).tolist()
        advantages = normalized_qre_scores(cost_estimates, str(args.qre_cost_normalization))
        qre_score_before_bonus = advantages.copy()
        exploration_bonus = max(float(args.qre_exploration_bonus), 0.0)
        if exploration_bonus > 0.0:
            total_count = max(float(np.sum(np.clip(counts, 0, None))), 1.0)
            denominators = np.maximum(counts.astype(np.float64), 0.0) + 1.0
            qre_exploration_bonus_values = exploration_bonus * np.sqrt(np.log(total_count + 1.0) / denominators)
            advantages = advantages + qre_exploration_bonus_values
        qre_temperature = qre_temperature_for_chunk(args, int(chunk_index), int(total_chunks))
        qre_response_probs = qre_soft_best_response(advantages, qre_temperature)
        response_rate = float(np.clip(float(args.qre_response_rate), 0.0, 1.0))
        updated = (1.0 - response_rate) * old_probs + response_rate * qre_response_probs
        prior_mix = float(np.clip(float(args.qre_prior_mix), 0.0, 1.0))
        min_prob = max(float(args.qre_min_prob), 0.0)
        benchmark_floor = float(args.qre_benchmark_floor)
        qre_max_prob_cap = float(args.qre_max_prob_cap)
    else:
        centered = costs[valid] - float(np.mean(costs[valid]))
        scale = float(np.std(costs[valid]))
        if scale <= 1e-9:
            scale = max(float(np.max(np.abs(centered))), 1.0)
        advantages[valid] = centered / scale
        eta = max(float(args.game_bandit_eta), 0.0)
        logits = np.log(np.clip(old_probs, 1e-12, 1.0)) + eta * advantages
        logits = logits - float(np.max(logits))
        updated = np.exp(logits)
        updated = updated / max(float(updated.sum()), 1e-12)
        prior_mix = float(np.clip(float(args.game_bandit_prior_mix), 0.0, 1.0))
        min_prob = max(float(args.game_bandit_min_prob), 0.0)
        benchmark_floor = float(args.game_bandit_benchmark_floor)

    if len(updated) > 0:
        min_prob = min(min_prob, 0.49 / len(updated))
        updated = (1.0 - min_prob * len(updated)) * updated + min_prob
    prior = np.asarray(initial_game_attack_probs(len(variants)), dtype=np.float64)
    updated = (1.0 - prior_mix) * updated + prior_mix * prior
    updated = np.asarray(
        enforce_benchmark_floor(updated, benchmark_floor),
        dtype=np.float64,
    )
    if sampler == "qre_minimax" and np.isfinite(qre_max_prob_cap) and qre_max_prob_cap > 0.0:
        updated = np.asarray(enforce_probability_cap(updated, qre_max_prob_cap), dtype=np.float64)
        updated = np.asarray(enforce_benchmark_floor(updated, benchmark_floor), dtype=np.float64)

    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(variants):
        rows.append(
            {
                "variant_id": variant_ids[index],
                "variant_scale": float(variant.get("scale", np.nan)),
                "rollout_count": int(counts[index]),
                "mean_cost": float(costs[index]) if np.isfinite(costs[index]) else np.nan,
                "advantage_z": float(advantages[index]),
                "qre_score_before_bonus": (
                    float(qre_score_before_bonus[index]) if np.isfinite(qre_score_before_bonus[index]) else np.nan
                ),
                "qre_exploration_bonus_value": float(qre_exploration_bonus_values[index]),
                "cost_estimate": float(cost_estimates[index]) if np.isfinite(cost_estimates[index]) else np.nan,
                "prob_before": float(old_probs[index]),
                "prob_after": float(updated[index]),
                "qre_response_prob": float(qre_response_probs[index]) if np.isfinite(qre_response_probs[index]) else np.nan,
                "qre_temperature": float(qre_temperature) if np.isfinite(qre_temperature) else np.nan,
                "qre_response_rate": float(args.qre_response_rate) if sampler == "qre_minimax" else np.nan,
                "qre_cost_ema_beta": float(args.qre_cost_ema_beta) if sampler == "qre_minimax" else np.nan,
                "qre_exploration_bonus": float(args.qre_exploration_bonus) if sampler == "qre_minimax" else np.nan,
                "qre_max_prob_cap": float(qre_max_prob_cap) if np.isfinite(qre_max_prob_cap) else np.nan,
                "benchmark_floor": float(benchmark_floor),
                "cost_key": cost_key,
                "sampler": sampler,
                "update_mode": update_mode,
                "rollout_metrics_path": str(rollout_metrics_path),
            }
        )
    return updated.astype(float).tolist(), rows


def append_attack_bandit_history(
    output_dir: Path,
    chunk_index: int,
    rows: list[dict[str, Any]],
    sampler: str = "adaptive_bandit",
) -> None:
    if not rows:
        return
    history = attack_bandit_history_path(output_dir, sampler)
    history.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{**row, "chunk_index": int(chunk_index)} for row in rows])
    if history.exists():
        previous = pd.read_csv(history)
        frame = pd.concat([previous, frame], ignore_index=True)
    frame.to_csv(history, index=False)


def game_recovery_train_args(
    args: argparse.Namespace,
    nominal_checkpoint: Path,
    rollout_metrics_path: Path | None = None,
    recovery_step_offset: int = 0,
    chunk_index: int = 0,
) -> dict[str, Any]:
    train_args: dict[str, Any] = {}
    if bool(getattr(args, "valt_sac_recovery_enabled", False)):
        train_args.update(
            {
                "sac-valt-recovery-enabled": True,
                "sac-valt-eps-start": float(args.valt_sac_eps_start),
                "sac-valt-eps-end": float(args.valt_sac_eps_end),
                "sac-valt-kappa-start": float(args.valt_sac_kappa_start),
                "sac-valt-kappa-end": float(args.valt_sac_kappa_end),
                "sac-valt-schedule-steps": int(args.valt_sac_schedule_steps),
                "sac-valt-schedule": str(args.valt_sac_schedule),
                "sac-valt-bound-iters": int(args.valt_sac_bound_iters),
                "sac-valt-worst-step-size": float(args.valt_sac_worst_step_size),
                "sac-valt-sgld-noise": float(args.valt_sac_sgld_noise),
                "sac-valt-policy-reg-coef": float(args.valt_sac_policy_reg_coef),
                "sac-valt-random-start": bool(args.valt_sac_random_start),
                "sac-valt-clip-low": float(args.valt_sac_clip_low),
                "sac-valt-clip-high": float(args.valt_sac_clip_high),
                "sac-valt-attack-deterministic": bool(args.valt_sac_attack_deterministic),
                "sac-deterministic-actor-update": bool(args.sac_recovery_deterministic_actor_update),
                "sac-target-entropy-scale": float(args.sac_recovery_target_entropy_scale),
                "sac-rollout-deterministic-prob": float(args.sac_recovery_rollout_deterministic_prob),
                "sac-rollout-noise-std": float(args.sac_recovery_rollout_noise_std),
                "sac-log-std-penalty-coef": float(args.sac_recovery_log_std_penalty_coef),
                "sac-log-std-target": float(args.sac_recovery_log_std_target),
                "recovery-step-offset": int(recovery_step_offset),
            }
        )
        if args.sac_recovery_fixed_alpha is not None:
            train_args["sac-fixed-alpha"] = float(args.sac_recovery_fixed_alpha)
        if rollout_metrics_path is not None:
            train_args["rollout-metrics-path"] = str(rollout_metrics_path)
    if not bool(args.game_recovery_enabled):
        return train_args
    if bool(getattr(args, "sac_game_recovery_enabled", False)):
        train_args.update(
            {
                "sac-game-recovery-enabled": True,
                "sac-game-anchor-checkpoint": str(nominal_checkpoint),
                "sac-game-anchor-coef": float(args.sac_game_anchor_coef),
                "sac-game-advantage-coef": float(args.sac_game_advantage_coef),
                "sac-game-q-margin": float(args.sac_game_q_margin),
                "sac-game-gate-temperature": float(args.sac_game_gate_temperature),
                "sac-game-lambda-drift-coef": float(args.sac_game_lambda_drift_coef),
                "sac-game-lambda-drift-margin": float(args.sac_game_lambda_drift_margin),
                "sac-game-risk-drift-coef": float(args.sac_game_risk_drift_coef),
                "sac-game-risk-drift-margin": float(args.sac_game_risk_drift_margin),
                "sac-game-anchor-barrier-coef": float(args.sac_game_anchor_barrier_coef),
                "sac-game-anchor-radius": float(args.sac_game_anchor_radius),
                "sac-game-risk-action-indices": str(args.sac_game_risk_action_indices),
                "sac-deterministic-actor-update": bool(args.sac_recovery_deterministic_actor_update),
                "sac-target-entropy-scale": float(args.sac_recovery_target_entropy_scale),
                "sac-rollout-deterministic-prob": float(args.sac_recovery_rollout_deterministic_prob),
                "sac-rollout-noise-std": float(args.sac_recovery_rollout_noise_std),
                "sac-log-std-penalty-coef": float(args.sac_recovery_log_std_penalty_coef),
                "sac-log-std-target": float(args.sac_recovery_log_std_target),
            }
        )
        if args.sac_recovery_fixed_alpha is not None:
            train_args["sac-fixed-alpha"] = float(args.sac_recovery_fixed_alpha)
        if rollout_metrics_path is not None:
            train_args["rollout-metrics-path"] = str(rollout_metrics_path)
        return train_args
    train_args.update(
        {
            "game-recovery-enabled": True,
            "game-anchor-checkpoint": str(nominal_checkpoint),
            "game-nominal-prior-coef": float(args.game_nominal_prior_coef),
            "game-lambda-drift-coef": float(args.game_lambda_drift_coef),
            "game-lambda-drift-margin": float(args.game_lambda_drift_margin),
            "game-risk-drift-coef": float(args.game_risk_drift_coef),
            "game-risk-drift-margin": float(args.game_risk_drift_margin),
            "game-risk-action-indices": str(args.game_risk_action_indices),
        }
    )
    if rollout_metrics_path is not None:
        train_args["rollout-metrics-path"] = str(rollout_metrics_path)
    if bool(args.ap_cvar_enabled):
        train_args.update(
            {
                "mra-enabled": True,
                "mra-cvar-quantile": float(args.ap_cvar_quantile),
                "mra-cvar-weight": float(args.ap_cvar_weight),
                "mra-tail-excess-weight": float(args.ap_cvar_tail_excess_weight),
                "mra-risk-feature-weight": float(args.ap_cvar_risk_feature_weight),
                "mra-weight-cap": float(args.ap_cvar_weight_cap),
                "mra-tail-reward-penalty": float(args.ap_cvar_tail_reward_penalty),
                "mra-risk-reward-penalty": float(args.ap_cvar_risk_reward_penalty),
            }
        )
    if bool(args.planner_regret_recovery_enabled):
        train_args.update(
            {
                "pr-enabled": True,
                "pr-enable-queries": True,
                "alpha-pr": float(args.pr_recovery_alpha),
                "beta-nominal": float(args.pr_recovery_beta_nominal),
                "pr-aux-loss-type": str(args.pr_recovery_aux_loss_type),
                "pr-query-fraction": float(args.pr_recovery_query_fraction),
                "pr-query-interval": int(args.pr_recovery_query_interval),
                "pr-num-candidates": int(args.pr_recovery_num_candidates),
                "pr-num-random-candidates": int(args.pr_recovery_num_random_candidates),
                "pr-local-sigma": float(args.pr_recovery_local_sigma),
                "pr-risk-local-sigma": float(args.pr_recovery_risk_local_sigma),
                "pr-cpa-temperature": float(args.pr_recovery_cpa_temperature),
                "pr-min-positive-adv": float(args.pr_recovery_min_positive_adv),
                "regret-weight-max": float(args.pr_recovery_regret_weight_max),
                "pr-ramp-steps": int(args.pr_recovery_ramp_steps),
                "pr-guidance-schedule": "constant",
                "recovery-step-offset": int(recovery_step_offset),
                "pr-log-path": str(args.output_dir / "planner_regret" / f"chunk_{int(chunk_index):04d}_pr_summary.csv"),
            }
        )
        if bool(args.pr_recovery_grad_ratio_controller):
            train_args["pr-grad-ratio-controller"] = True
            train_args["pr-grad-diagnostics"] = True
    if bool(args.game_teacher_recovery_enabled):
        train_args.update(
            {
                "pr-enabled": True,
                "pr-enable-queries": True,
                "pr-game-teacher-enabled": True,
                "pr-game-teacher-mode": str(args.gt_recovery_teacher_mode),
                "pr-game-max-attack-variants": int(args.gt_recovery_max_attack_variants),
                "pr-game-softmax-temperature": float(args.gt_recovery_softmax_temperature),
                "pr-include-structured-candidates": True,
                "pr-num-structured-candidates": int(args.gt_recovery_num_structured_candidates),
                "alpha-pr": float(args.gt_recovery_alpha),
                "beta-nominal": 0.0,
                "pr-aux-loss-type": str(args.gt_recovery_aux_loss_type),
                "pr-query-fraction": float(args.gt_recovery_query_fraction),
                "pr-query-interval": int(args.gt_recovery_query_interval),
                "pr-num-candidates": int(args.gt_recovery_num_candidates),
                "pr-num-random-candidates": int(args.gt_recovery_num_random_candidates),
                "pr-local-sigma": float(args.gt_recovery_local_sigma),
                "pr-risk-local-sigma": float(args.gt_recovery_risk_local_sigma),
                "pr-cpa-temperature": float(args.gt_recovery_cpa_temperature),
                "pr-min-positive-adv": float(args.gt_recovery_min_positive_adv),
                "regret-weight-max": float(args.gt_recovery_regret_weight_max),
                "pr-ramp-steps": int(args.gt_recovery_ramp_steps),
                "pr-guidance-schedule": "constant",
                "recovery-step-offset": int(recovery_step_offset),
                "pr-log-path": str(
                    args.output_dir
                    / "game_teacher_planner_regret"
                    / f"chunk_{int(chunk_index):04d}_pr_summary.csv"
                ),
            }
        )
    if bool(args.acbr_recovery_enabled):
        train_args.update(
            {
                "acbr-enabled": True,
                "acbr-anchor-checkpoint": str(nominal_checkpoint),
                "acbr-critic-coef": float(args.acbr_recovery_critic_coef),
                "acbr-uncertainty-coef": float(args.acbr_recovery_uncertainty_coef),
                "acbr-anchor-penalty": float(args.acbr_recovery_anchor_penalty),
                "acbr-policy-penalty": float(args.acbr_recovery_policy_penalty),
                "acbr-target-clip": float(args.acbr_recovery_target_clip),
                "acbr-rerank-start-after-steps": int(args.acbr_recovery_rerank_start_after_steps),
                "acbr-benefit-gate-enabled": bool(args.acbr_recovery_benefit_gate_enabled),
                "acbr-benefit-margin": float(args.acbr_recovery_benefit_margin),
                "pr-enabled": True,
                "pr-enable-queries": True,
                "pr-game-teacher-enabled": True,
                "pr-game-teacher-mode": str(args.acbr_recovery_teacher_mode),
                "pr-game-max-attack-variants": int(args.acbr_recovery_max_attack_variants),
                "pr-game-softmax-temperature": float(args.acbr_recovery_softmax_temperature),
                "pr-include-structured-candidates": True,
                "pr-num-structured-candidates": int(args.acbr_recovery_num_structured_candidates),
                "alpha-pr": 0.0,
                "beta-nominal": 0.0,
                "pr-aux-loss-type": "mse",
                "pr-target-type": "hybrid",
                "pr-query-fraction": float(args.acbr_recovery_query_fraction),
                "pr-query-interval": int(args.acbr_recovery_query_interval),
                "pr-num-candidates": int(args.acbr_recovery_num_candidates),
                "pr-num-random-candidates": int(args.acbr_recovery_num_random_candidates),
                "pr-local-sigma": float(args.acbr_recovery_local_sigma),
                "pr-risk-local-sigma": float(args.acbr_recovery_risk_local_sigma),
                "pr-soft-regret-threshold": 0.005,
                "pr-hard-regret-threshold": 0.02,
                "regret-weight-max": 3.0,
                "pr-ramp-steps": 1024,
                "pr-guidance-schedule": "constant",
                "prb-enabled": True,
                "prb-encoder-type": "mlp",
                "prb-latent-dim": int(args.acbr_recovery_latent_dim),
                "prb-hidden-dim": int(args.acbr_recovery_hidden_dim),
                "prb-aux-coef": float(args.acbr_recovery_aux_coef),
                "prb-feature-clip": float(args.acbr_recovery_feature_clip),
                "prb-use-component-costs": True,
                "prb-use-scalar-cost": True,
                "prb-normalize-features": True,
                "recovery-step-offset": int(recovery_step_offset),
                "pr-log-path": str(args.output_dir / "acbr" / f"chunk_{int(chunk_index):04d}_acbr_summary.csv"),
                "prb-log-path": str(args.output_dir / "acbr" / f"chunk_{int(chunk_index):04d}_acbr_prb_summary.csv"),
            }
        )
    if bool(args.teacher_residual_recovery_enabled):
        train_args.update(
            {
                "trr-enabled": True,
                "trr-anchor-checkpoint": str(nominal_checkpoint),
                "trr-freeze-base-actor": bool(args.trr_recovery_freeze_base_actor),
                "trr-alpha": float(args.trr_recovery_alpha),
                "trr-ramp-steps": int(args.trr_recovery_ramp_steps),
                "trr-guidance-schedule": "early_decay",
                "trr-active-until-step": int(args.trr_recovery_active_until_step),
                "trr-decay-to-zero-by-step": int(args.trr_recovery_decay_to_zero_by_step),
                "trr-min-normalized-regret": float(args.trr_recovery_min_normalized_regret),
                "trr-residual-l2-coef": float(args.trr_recovery_residual_l2_coef),
                "trr-residual-barrier-coef": float(args.trr_recovery_residual_barrier_coef),
                "trr-residual-action-limit": float(args.trr_recovery_residual_action_limit),
                "pr-enabled": True,
                "pr-enable-queries": True,
                "pr-game-teacher-enabled": True,
                "pr-game-teacher-mode": str(args.trr_recovery_teacher_mode),
                "pr-game-max-attack-variants": int(args.trr_recovery_max_attack_variants),
                "pr-game-softmax-temperature": float(args.trr_recovery_softmax_temperature),
                "pr-include-structured-candidates": True,
                "pr-num-structured-candidates": int(args.trr_recovery_num_structured_candidates),
                "alpha-pr": 0.0,
                "beta-nominal": 0.0,
                "pr-aux-loss-type": "mse",
                "pr-target-type": "hybrid",
                "pr-query-fraction": float(args.trr_recovery_query_fraction),
                "pr-query-interval": int(args.trr_recovery_query_interval),
                "pr-num-candidates": int(args.trr_recovery_num_candidates),
                "pr-num-random-candidates": int(args.trr_recovery_num_random_candidates),
                "pr-local-sigma": float(args.trr_recovery_local_sigma),
                "pr-risk-local-sigma": float(args.trr_recovery_risk_local_sigma),
                "pr-soft-regret-threshold": float(args.trr_recovery_min_normalized_regret),
                "pr-hard-regret-threshold": max(0.02, float(args.trr_recovery_min_normalized_regret)),
                "regret-weight-max": float(args.trr_recovery_regret_weight_max),
                "pr-ramp-steps": int(args.trr_recovery_ramp_steps),
                "pr-guidance-schedule": "constant",
                "prb-enabled": True,
                "prb-encoder-type": "mlp",
                "prb-latent-dim": int(args.trr_recovery_latent_dim),
                "prb-hidden-dim": int(args.trr_recovery_hidden_dim),
                "prb-aux-coef": float(args.trr_recovery_aux_coef),
                "prb-feature-clip": float(args.trr_recovery_feature_clip),
                "prb-use-component-costs": True,
                "prb-use-scalar-cost": True,
                "prb-normalize-features": True,
                "recovery-step-offset": int(recovery_step_offset),
                "pr-log-path": str(args.output_dir / "teacher_residual" / f"chunk_{int(chunk_index):04d}_trr_summary.csv"),
                "prb-log-path": str(args.output_dir / "teacher_residual" / f"chunk_{int(chunk_index):04d}_trr_prb_summary.csv"),
            }
        )
    if bool(args.bagr_recovery_enabled):
        train_args.update(
            {
                "bagr-enabled": True,
                "bagr-anchor-checkpoint": str(nominal_checkpoint),
                "bagr-max-attack-variants": int(args.bagr_recovery_max_attack_variants),
                "bagr-belief-temperature": float(args.bagr_recovery_belief_temperature),
                "bagr-belief-prior-mix": float(args.bagr_recovery_belief_prior_mix),
                "bagr-freeze-base-actor": bool(args.bagr_recovery_freeze_base_actor),
                "bagr-residual-l2-coef": float(args.bagr_recovery_residual_l2_coef),
                "bagr-residual-barrier-coef": float(args.bagr_recovery_residual_barrier_coef),
                "bagr-residual-action-limit": float(args.bagr_recovery_residual_action_limit),
                "bagr-confidence-limit-scale": float(args.bagr_recovery_confidence_limit_scale),
                "prb-enabled": True,
                "prb-encoder-type": "mlp",
                "prb-latent-dim": int(args.bagr_recovery_latent_dim),
                "prb-hidden-dim": int(args.bagr_recovery_hidden_dim),
                "prb-aux-coef": float(args.bagr_recovery_aux_coef),
                "prb-feature-clip": float(args.bagr_recovery_feature_clip),
                "prb-use-component-costs": True,
                "prb-use-scalar-cost": True,
                "prb-normalize-features": True,
                "prb-stopgrad-residual-latent": True,
                "recovery-step-offset": int(recovery_step_offset),
                "prb-log-path": str(args.output_dir / "bagr_belief" / f"chunk_{int(chunk_index):04d}_bagr_summary.csv"),
            }
        )
    return train_args


def is_real_level(level_config: dict[str, Any]) -> bool:
    return "map_source" in level_config


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def split_paths(output_dir: Path) -> dict[str, Path]:
    splits_dir = output_dir / "splits"
    return {
        "train": splits_dir / "train_tasks.json",
        "validation": splits_dir / "validation_tasks.json",
        "heldout": splits_dir / "heldout_tasks.json",
    }


def command_from_train_args(python_exe: str, train_args: dict[str, Any], algo: str) -> list[str]:
    command = [python_exe, training_script_for_algo(algo)]
    for key, value in train_args.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    return command


def synthetic_base_args(
    level_config: dict[str, Any],
    base_config: Path | None,
    output_dir: Path,
    seed: int,
    nominal_timesteps: int,
    eval_interval: int,
    train_eval_episodes: int,
) -> dict[str, Any]:
    base_path = base_config or resolve_path(level_config["base_config"])
    args = load_base_args(str(base_path))
    args["seed"] = int(seed)
    args["map-size"] = int(level_config.get("map_size", args.get("map-size", 48)))
    args["scenario"] = str(level_config.get("scenario", args.get("scenario", "lunar_rover_corridor")))
    args["min-start-goal-distance-ratio"] = float(
        level_config.get(
            "min_distance_ratio",
            args.get("min-start-goal-distance-ratio", 0.55),
        )
    )
    args["observation-mode"] = str(level_config.get("observation_mode", args.get("observation-mode", "terrain")))
    args["reward-mode"] = str(level_config.get("reward_mode", args.get("reward-mode", "relative_heuristic")))
    args["reward-scale"] = float(level_config.get("reward_scale", args.get("reward-scale", 10.0)))
    args["reward-cost-key"] = "scalar_cost"
    args["action-mode"] = str(level_config.get("action_mode", args.get("action-mode", "preference_delta")))
    args["action-gain"] = float(level_config.get("action_gain", args.get("action-gain", 3.0)))
    args["max-uncertainty-lambda"] = float(
        level_config.get("max_uncertainty_lambda", args.get("max-uncertainty-lambda", 1.2))
    )
    args["map-sampling-mode"] = str(level_config.get("map_sampling_mode", args.get("map-sampling-mode", "map_seed_pool")))
    args["fixed-map-seed"] = int(level_config.get("fixed_map_seed", args.get("fixed-map-seed", 909)))
    args["map-seed-pool-size"] = int(level_config.get("map_seed_pool_size", args.get("map-seed-pool-size", 32)))
    args["log-dir"] = str(output_dir / "nominal_train")
    args["total-timesteps"] = int(nominal_timesteps)
    args["eval-freq"] = min(int(eval_interval), max(int(nominal_timesteps), 1))
    args["n-eval-episodes"] = int(train_eval_episodes)
    args["eval-seed"] = int(level_config.get("fixed_map_seed", args.get("fixed-map-seed", 909)))
    args.pop("environment_attack", None)
    args.pop("observation_attack", None)
    return args


def real_base_args(
    level_config: dict[str, Any],
    base_config: Path | None,
    output_dir: Path,
    seed: int,
    nominal_timesteps: int,
    eval_interval: int,
    train_eval_episodes: int,
    quick: bool,
    dry_run: bool,
) -> tuple[dict[str, Any], dict[str, Path]]:
    base_path = base_config or (PROJECT_ROOT / "configs" / "ppo_lunar_viper_relative_reward.json")
    splits = split_paths(output_dir) if dry_run else prepare_splits(level_config, output_dir, seed, quick=quick)
    args = resolved_base_args(base_path, level_config, splits, output_dir, seed, nominal_timesteps)
    args["reward-cost-key"] = "scalar_cost"
    args["eval-freq"] = min(int(eval_interval), max(int(nominal_timesteps), 1))
    args["n-eval-episodes"] = int(train_eval_episodes)
    args["eval-seed"] = int(seed + 50_000)
    return args, splits


def train_nominal(
    python_exe: str,
    base_args: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
    algo: str,
) -> tuple[Path, Path | None]:
    command = command_from_train_args(python_exe, base_args, algo)
    print(" ".join(str(part) for part in command), flush=True)
    seed = int(config_value(base_args, "seed", 0))
    run_dir = Path(base_args["log-dir"]) / run_name_for_algo(algo, seed)
    final_model = run_dir / "final_model.pt"
    if not dry_run:
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        if not final_model.exists():
            raise FileNotFoundError(f"nominal checkpoint not found: {final_model}")
    eval_csv = run_dir / "eval_metrics.csv"
    return final_model, eval_csv if eval_csv.exists() else None


def copy_checkpoint(source: Path, target: Path, dry_run: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"Would copy checkpoint {source} -> {target}")
        return
    shutil.copy2(source, target)


def build_synthetic_eval_episodes(
    args: argparse.Namespace,
    base_args: dict[str, Any],
    map_pool_size: int,
) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    map_size = int(config_value(base_args, "map-size", 48))
    scenario = str(config_value(base_args, "scenario", "lunar_rover_corridor"))
    min_distance_ratio = float(config_value(base_args, "min-start-goal-distance-ratio", 0.55))
    episodes_by_domain = {
        f"in_domain_seed{args.in_domain_seed}": (
            int(args.in_domain_seed),
            generate_episodes(
                args.num_eval_episodes,
                args.seed + 222,
                map_size,
                scenario,
                args.in_domain_seed,
                map_pool_size,
                min_start_goal_distance_ratio=min_distance_ratio,
            ),
        ),
        f"heldout_seed{args.heldout_seed}": (
            int(args.heldout_seed),
            generate_episodes(
                args.num_eval_episodes,
                args.seed + 222,
                map_size,
                scenario,
                args.heldout_seed,
                map_pool_size,
                min_start_goal_distance_ratio=min_distance_ratio,
            ),
        ),
    }
    return map_size, episodes_by_domain


def build_real_eval_episodes(
    level_config: dict[str, Any],
    splits: dict[str, Path],
    seed: int,
    num_eval_episodes: int,
) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    layers_path = resolve_path(level_config["map_source"])
    scenario = str(level_config.get("scenario", "real_lunar_viper"))
    mission_profile = str(level_config.get("mission_profile_scenario", "lunar_polar_shadow"))
    map_size = int(load_real_layers(layers_path)["layer_distance"].shape[0])
    return map_size, {
        "train_tasks": (
            seed,
            generate_real_episodes(layers_path, splits["train"], scenario, mission_profile, seed + 11_000, num_eval_episodes),
        ),
        "heldout_tasks": (
            seed + 1,
            generate_real_episodes(layers_path, splits["heldout"], scenario, mission_profile, seed + 22_000, num_eval_episodes),
        ),
    }


def add_protocol_columns(rows: list[dict[str, Any]], phase: str, recovery_step: int, checkpoint_role: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        enriched = dict(row)
        enriched["phase"] = phase
        enriched["recovery_step"] = int(recovery_step)
        enriched["checkpoint_role"] = checkpoint_role
        output.append(enriched)
    return output


def write_nominal_eval(eval_csv: Path | None, output_dir: Path) -> None:
    if eval_csv is None or not eval_csv.exists():
        return
    frame = pd.read_csv(eval_csv)
    frame.insert(0, "phase", "nominal_training")
    frame.to_csv(output_dir / "nominal_training_eval.csv", index=False)


def numeric_stats(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "max": None,
            "p25": None,
            "p50": None,
            "p75": None,
        }
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "p25": float(np.percentile(array, 25)),
        "p50": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
    }


def task_split_summary(splits: dict[str, Path] | None) -> dict[str, Any]:
    if not splits:
        return {}
    summary: dict[str, Any] = {}
    for split_name, split_path in splits.items():
        path = Path(split_path)
        if not path.exists():
            summary[split_name] = {"path": str(path), "exists": False}
            continue
        tasks = json.loads(path.read_text(encoding="utf-8-sig"))
        difficulty = [dict(task.get("difficulty", {})) for task in tasks]
        met_flags = [
            bool(item.get("met_min_corridor_risk"))
            for item in difficulty
            if item.get("met_min_corridor_risk") is not None
        ]
        summary[split_name] = {
            "path": str(path),
            "exists": True,
            "num_tasks": int(len(tasks)),
            "euclidean_distance_cells": numeric_stats(
                [float(item["euclidean_distance_cells"]) for item in difficulty if item.get("euclidean_distance_cells") is not None]
            ),
            "euclidean_distance_m": numeric_stats(
                [float(item["euclidean_distance_m"]) for item in difficulty if item.get("euclidean_distance_m") is not None]
            ),
            "corridor_risk_score": numeric_stats(
                [float(item["corridor_risk_score"]) for item in difficulty if item.get("corridor_risk_score") is not None]
            ),
            "met_min_corridor_risk_rate": (
                float(np.mean(met_flags)) if met_flags else None
            ),
        }
    return summary


def map_metadata_summary(level_config: dict[str, Any]) -> dict[str, Any]:
    metadata_path = level_config.get("metadata")
    if metadata_path is None:
        return {
            "map_size": level_config.get("map_size"),
            "scenario": level_config.get("scenario"),
            "fixed_map_seed": level_config.get("fixed_map_seed"),
            "map_seed_pool_size": level_config.get("map_seed_pool_size"),
        }
    path = resolve_path(str(metadata_path))
    if not path.exists():
        return {"metadata": str(path), "exists": False}
    metadata = read_json(path)
    keys = [
        "tile_id",
        "source_name",
        "difficulty",
        "archetype",
        "meters_per_pixel",
        "requested_meters",
        "obstacle_cell_ratio",
        "slope_mean_deg",
        "slope_p95_deg",
        "slope_max_deg",
        "roughness_mean_m",
        "roughness_p95_m",
        "relief_p95_p05_m",
        "elevation_std_m",
        "rover_name",
    ]
    return {
        "metadata": str(path),
        "exists": True,
        **{key: metadata.get(key) for key in keys if key in metadata},
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def memory_total_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                kb = float(line.split()[1])
                return kb / (1024.0 * 1024.0)
    except Exception:
        return None
    return None


def torch_cuda_environment() -> dict[str, Any]:
    import warnings

    try:
        import torch
    except Exception as exc:
        return {"torch_import_error": str(exc)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cuda_available = bool(torch.cuda.is_available())
        cudnn_available = bool(torch.backends.cudnn.is_available())

    output: dict[str, Any] = {
        "cuda_available": cuda_available,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if cudnn_available else None,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            output["cuda_device_count"] = int(torch.cuda.device_count())
        if output["cuda_available"]:
            output["cuda_device_names"] = [
                torch.cuda.get_device_name(index) for index in range(output["cuda_device_count"])
            ]
    except Exception as exc:
        output["cuda_query_error"] = str(exc)
    return output


def git_snapshot() -> dict[str, Any]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(PROJECT_ROOT),
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            return result.stdout.strip()
        except Exception:
            return None

    status = run_git("status", "--short")
    status_lines = status.splitlines() if status else []
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "is_dirty": bool(status),
        "status_short_count": len(status_lines),
        "status_short_preview": status_lines[:50],
    }


def software_environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hardware": {
            "cpu_count": os.cpu_count(),
            "memory_total_gib": memory_total_gib(),
            **torch_cuda_environment(),
        },
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": package_version("matplotlib"),
            "torch": package_version("torch"),
            "gymnasium": package_version("gymnasium"),
        },
        "git": git_snapshot(),
    }


def summarize_recovery(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = summarize_shock_recovery_frame(frame)
    summary.to_csv(output_dir / "shock_recovery_summary.csv", index=False)
    return summary


def plot_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    nominal_csv = output_dir / "nominal_training_eval.csv"
    nominal = pd.read_csv(nominal_csv) if nominal_csv.exists() else pd.DataFrame()
    nominal_end = int(frame["global_step"].max()) if "global_step" in frame.columns and not frame.empty else 0
    if not nominal.empty and "global_step" in nominal.columns:
        nominal_end = max(nominal_end, int(nominal["global_step"].max()))
    shock_offset = max(1, int(frame["recovery_step"].max()) // 20 if "recovery_step" in frame.columns else 1)

    summary = summarize_shock_recovery_frame(frame)
    if not summary.empty:
        fig, ax = plt.subplots(figsize=(9.4, 5.0))
        x = np.arange(len(summary), dtype=np.float32)
        width = 0.20
        bars = [
            ("Clean", "clean_nominal_cost", -1.5 * width, "0.35"),
            ("Attack", "attacked_nominal_cost", -0.5 * width, "tab:red"),
            ("Final recovery", "final_recovery_cost", 0.5 * width, "tab:blue"),
            ("Best recovery", "best_recovery_cost", 1.5 * width, "tab:green"),
        ]
        for label, column, offset, color in bars:
            ax.bar(x + offset, summary[column].astype(float), width=width, label=label, color=color, alpha=0.88)
        labels = [
            str(row["eval_domain"])
            + (
                "\nattack <5%"
                if str(row.get("attack_effect_status", "")) != "meaningful_attack"
                else ""
            )
            for _, row in summary.iterrows()
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("true scalar cost (lower is better)")
        ax.set_title("Clean, attack shock, and recovery cost")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "fig_paper_clean_attack_recovery_cost.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9.4, 5.0))
        x = np.arange(len(summary), dtype=np.float32)
        width = 0.24
        ax.bar(
            x - width,
            summary["attack_degradation_pct"].astype(float),
            width=width,
            label="Attack degradation",
            color="tab:red",
            alpha=0.88,
        )
        ax.bar(
            x,
            summary["final_residual_degradation_pct"].astype(float),
            width=width,
            label="Final residual",
            color="tab:blue",
            alpha=0.88,
        )
        ax.bar(
            x + width,
            summary["best_residual_degradation_pct"].astype(float),
            width=width,
            label="Best residual",
            color="tab:green",
            alpha=0.88,
        )
        ax.axhline(5.0, color="0.35", linestyle="--", linewidth=1.0, label="5% report threshold")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["eval_domain"].astype(str), fontsize=8)
        ax.set_ylabel("degradation vs clean (%)")
        ax.set_title("Attack and residual degradation")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "fig_paper_degradation_summary.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    if {"global_step", "mean_scalar_cost"}.issubset(nominal.columns):
        clean_training = nominal.sort_values("global_step")
        final_training_cost = float(clean_training.iloc[-1]["mean_scalar_cost"])
        if np.isfinite(final_training_cost) and final_training_cost > 0:
            training_performance = 100.0 * final_training_cost / clean_training["mean_scalar_cost"].astype(float)
            ax.plot(
                clean_training["global_step"].astype(int),
                training_performance,
                color="0.20",
                linewidth=2.2,
                marker="o",
                markersize=4,
                label="clean PPO training",
            )

    for eval_domain, group in frame.groupby("eval_domain"):
        clean = group[(group["phase"] == "shock") & (group["attack_type"] == "none")]
        env_rows = group[group["attack_type"] == "environment"].sort_values("recovery_step")
        if clean.empty or env_rows.empty:
            continue
        clean_cost = float(clean.iloc[0]["mean_attacked_scalar_cost"])
        if not np.isfinite(clean_cost) or clean_cost <= 0:
            continue
        shock_x = nominal_end + shock_offset
        recovery_x = (nominal_end + shock_offset + env_rows["recovery_step"].astype(int)).tolist()
        x_values = [nominal_end, shock_x] + recovery_x[1:]
        y_values = [100.0] + (
            100.0 * clean_cost / env_rows["mean_attacked_scalar_cost"].astype(float)
        ).tolist()
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.0,
            label=f"{eval_domain} shock/recovery",
        )

    ax.axvline(nominal_end, color="tab:red", linestyle="--", linewidth=1)
    ax.axhline(100.0, color="0.35", linestyle="--", linewidth=1, alpha=0.75)
    ax.text(
        nominal_end,
        ax.get_ylim()[0],
        "map attack introduced",
        va="bottom",
        ha="left",
        fontsize=8,
        color="tab:red",
    )
    ax.set_xlabel("experiment timeline")
    ax.set_ylabel("performance index (clean nominal = 100; higher is better)")
    ax.set_title("Clean training, attack shock, and PPO recovery")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_main_clean_shock_recovery_performance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for eval_domain, group in frame.groupby("eval_domain"):
        clean = group[(group["phase"] == "shock") & (group["attack_type"] == "none")]
        env_rows = group[group["attack_type"] == "environment"].sort_values("recovery_step")
        if clean.empty or env_rows.empty:
            continue
        x = [-1] + env_rows["recovery_step"].astype(int).tolist()
        y = [float(clean.iloc[0]["mean_attacked_scalar_cost"])] + env_rows["mean_attacked_scalar_cost"].astype(float).tolist()
        ax.plot(x, y, marker="o", linewidth=2.0, label=eval_domain)
    ax.axvline(-0.5, color="0.45", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="tab:red", linestyle="--", linewidth=1)
    ax.text(-1, ax.get_ylim()[1], "clean nominal", va="top", fontsize=8)
    ax.text(0, ax.get_ylim()[1], "attack shock", va="top", fontsize=8, color="tab:red")
    ax.set_xlabel("recovery step (-1 is clean nominal)")
    ax.set_ylabel("true scalar cost (lower is better)")
    ax.set_title("PPO shock-recovery curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_main_shock_recovery_cost.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for eval_domain, group in frame[frame["attack_type"] == "environment"].groupby("eval_domain"):
        group = group.sort_values("recovery_step")
        ax.plot(
            group["recovery_step"],
            100.0 * group["relative_degradation"].astype(float),
            marker="o",
            linewidth=2.0,
            label=eval_domain,
        )
    ax.axvline(0.0, color="tab:red", linestyle="--", linewidth=1)
    ax.set_xlabel("recovery step")
    ax.set_ylabel("attack degradation (%)")
    ax.set_title("Attack degradation during recovery")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_attack_degradation_recovery.png", dpi=180)
    plt.close(fig)

    if not nominal.empty:
        if {"global_step", "mean_scalar_cost"}.issubset(nominal.columns):
            fig, ax = plt.subplots(figsize=(8.5, 4.8))
            ax.plot(nominal["global_step"], nominal["mean_scalar_cost"], marker="o", linewidth=2.0)
            ax.set_xlabel("nominal training step")
            ax.set_ylabel("clean eval scalar cost")
            ax.set_title("Clean PPO nominal training")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(figures_dir / "fig_nominal_training_clean_cost.png", dpi=180)
            plt.close(fig)


def write_output_guide(output_dir: Path) -> None:
    guide = """# RL Shock-Recovery Outputs

This is the main RL robustness baseline:

1. train the policy in the clean environment,
2. evaluate the nominal checkpoint with no attack and with map attack,
3. fine-tune under the map attack and evaluate recovery checkpoints.

Main files:

- `nominal_training_eval.csv`: clean-environment PPO training curve.
- `shock_recovery_curve.csv`: clean nominal, attack shock, and attack recovery evaluations.
- `shock_recovery_summary.csv`: paper-facing attack and recovery metrics per eval domain.
- `attack_bandit/attack_bandit_history.csv`: adaptive attack sampler probabilities per recovery chunk, when enabled.
- `figures/fig_main_clean_shock_recovery_performance.png`: main narrative plot; higher is better.
- `figures/fig_main_shock_recovery_cost.png`: cost diagnostic; lower is better.
- `figures/fig_attack_degradation_recovery.png`: attack degradation over recovery.
- `figures/fig_paper_clean_attack_recovery_cost.png`: clean/attack/final/best cost bars.
- `figures/fig_paper_degradation_summary.png`: attack and residual degradation bars.

Important fields:

- `attack_drop = attacked_nominal_cost - clean_nominal_cost`
- `attack_degradation_pct = 100 * attack_drop / clean_nominal_cost`
- `final_residual_degradation_pct = 100 * (final_recovery_cost - clean_nominal_cost) / clean_nominal_cost`
- `final_recovery_closure_pct = 100 * (attacked_nominal_cost - final_recovery_cost) / attack_drop`
- `final_recovery_closure_pct` is only reported when `attack_degradation_pct >= 5`.
- `performance_index = 100 * clean_nominal_cost / true_scalar_cost`
- `mean_map_mismatch_penalty = true cost - belief cost`

Adaptive game recovery:

- `--game-attack-sampler adaptive_bandit` keeps the clean training and shock evaluation unchanged.
- During recovery training only, local attack variants are sampled from a probability distribution.
- After each recovery chunk, variants with higher rollout cost receive higher probability for the next chunk.
- `--qre-minimax-recovery-enabled` switches the recovery attacker to a quantal-response minimax
  adversary: high temperature starts broad, then the schedule anneals toward a soft worst-case
  attack distribution. `--qre-max-prob-cap` and `--qre-exploration-bonus` keep the adversary from
  collapsing onto one attack before the recovering policy has seen enough of the attack population.
- `--ap-cvar-enabled` additionally enables tail-weighted PPO updates during recovery only: high-cost
  rollout samples receive larger policy-gradient weight, approximating an EPOpt/CVaR robust update.
- `--game-bandit-benchmark-floor` reserves probability mass for the benchmark attack so adaptive
  recovery remains aligned with the fixed shock-recovery evaluation protocol.
- `--cdr-recovery-enabled` is a recovery-only curriculum domain-randomization baseline: it starts
  benchmark-heavy and gradually increases local attack scale jitter plus non-benchmark sampling.
"""
    (output_dir / "OUTPUT_GUIDE.md").write_text(guide, encoding="utf-8")


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    level_config: dict[str, Any],
    base_args: dict[str, Any],
    env_attack: dict[str, Any],
    splits: dict[str, Path] | None,
) -> None:
    total_recovery_chunks = int(np.ceil(max(int(args.recovery_timesteps), 1) / max(int(args.eval_interval), 1)))
    recovery_training_attack = build_game_recovery_attack(
        env_attack,
        args,
        int(args.seed) + 20_000,
        probs=initial_game_attack_probs(len(game_attack_variants(env_attack, args))),
        chunk_index=1,
        total_chunks=total_recovery_chunks,
    )
    cdr_progress0 = cdr_curriculum_progress(args, chunk_index=1, total_chunks=total_recovery_chunks)
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": "clean_train_attack_shock_attack_recovery",
        "algorithm": str(args.algo),
        "command_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "level_config": level_config,
        "base_config_args": base_args,
        "environment_attack": env_attack,
        "recovery_training_environment_attack": recovery_training_attack,
        "game_recovery_attack_variants": game_attack_variants(env_attack, args)
        if bool(args.game_recovery_enabled)
        else [],
        "cdr_recovery_attack_variants": cdr_attack_variants(env_attack, args, cdr_progress0)
        if bool(getattr(args, "cdr_recovery_enabled", False))
        else [],
        "splits": {key: str(value) for key, value in (splits or {}).items()},
        "task_split_summary": task_split_summary(splits),
        "map_metadata_summary": map_metadata_summary(level_config),
        "software_environment": software_environment(),
    }
    write_json(output_dir / "run_config.json", metadata)


def main() -> int:
    args = parse_args()
    if args.quick:
        args.nominal_timesteps = 2048
        args.recovery_timesteps = 2048
        args.eval_interval = 1024
        args.num_eval_episodes = min(args.num_eval_episodes, 20)
        args.train_eval_episodes = min(args.train_eval_episodes, 10)
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            args.output_dir = PROJECT_ROOT / "runs" / "debug_ppo_shock_recovery"
    if str(args.game_attack_sampler) == "qre_minimax":
        args.qre_minimax_recovery_enabled = True
    if bool(args.qre_minimax_recovery_enabled):
        args.game_attack_sampler = "qre_minimax"
    if bool(args.bvr_recovery_enabled):
        if str(args.algo).lower() != "ppo":
            raise ValueError("--bvr-recovery-enabled requires --algo ppo")
        args.game_recovery_enabled = True
        args.acbr_recovery_enabled = True
        args.acbr_recovery_benefit_gate_enabled = True
    if bool(args.sac_game_recovery_enabled):
        if str(args.algo).lower() != "sac":
            raise ValueError("--sac-game-recovery-enabled requires --algo sac")
        args.game_recovery_enabled = True
    if bool(getattr(args, "cdr_recovery_enabled", False)):
        if bool(args.game_recovery_enabled):
            raise ValueError("--cdr-recovery-enabled is a standalone recovery baseline; do not combine it with game recovery")
        if int(args.cdr_attack_mixture_size) < 1:
            raise ValueError("--cdr-attack-mixture-size must be at least 1")
        if float(args.cdr_attack_jitter_start) < 0.0 or float(args.cdr_attack_jitter_end) < 0.0:
            raise ValueError("--cdr-attack-jitter-start/end must be non-negative")
        if not (0.0 <= float(args.cdr_benchmark_prob_start) <= 1.0):
            raise ValueError("--cdr-benchmark-prob-start must be in [0, 1]")
        if not (0.0 <= float(args.cdr_benchmark_prob_end) <= 1.0):
            raise ValueError("--cdr-benchmark-prob-end must be in [0, 1]")
    if bool(args.game_recovery_enabled) and str(args.algo).lower() != "ppo" and not bool(args.sac_game_recovery_enabled):
        raise ValueError("game recovery is implemented for PPO, or for SAC with --sac-game-recovery-enabled")
    if bool(args.sac_game_recovery_enabled) and (
        bool(args.ap_cvar_enabled)
        or bool(args.game_teacher_recovery_enabled)
        or bool(args.teacher_residual_recovery_enabled)
        or bool(args.bagr_recovery_enabled)
        or bool(args.planner_regret_recovery_enabled)
        or bool(args.acbr_recovery_enabled)
        or bool(args.bvr_recovery_enabled)
    ):
        raise ValueError(
            "--sac-game-recovery-enabled cannot be combined with PPO-specific "
            "AP-CVaR, game-teacher, TRR, BAGR, planner-regret, ACBR, or BVR recovery"
        )
    if bool(args.ap_cvar_enabled) and not bool(args.game_recovery_enabled):
        raise ValueError("--ap-cvar-enabled must be used with --game-recovery-enabled")
    if bool(args.game_teacher_recovery_enabled) and not bool(args.game_recovery_enabled):
        raise ValueError("--game-teacher-recovery-enabled must be used with --game-recovery-enabled")
    if bool(args.teacher_residual_recovery_enabled) and not bool(args.game_recovery_enabled):
        raise ValueError("--teacher-residual-recovery-enabled must be used with --game-recovery-enabled")
    if bool(args.bagr_recovery_enabled) and not bool(args.game_recovery_enabled):
        raise ValueError("--bagr-recovery-enabled must be used with --game-recovery-enabled")
    if bool(args.qre_minimax_recovery_enabled) and not bool(args.game_recovery_enabled):
        raise ValueError("--qre-minimax-recovery-enabled must be used with --game-recovery-enabled")
    if bool(args.acbr_recovery_enabled) and not bool(args.game_recovery_enabled):
        raise ValueError("--acbr-recovery-enabled must be used with --game-recovery-enabled")
    if bool(args.bagr_recovery_enabled) and (
        bool(args.game_teacher_recovery_enabled)
        or bool(args.teacher_residual_recovery_enabled)
        or bool(args.planner_regret_recovery_enabled)
        or bool(args.ap_cvar_enabled)
        or bool(args.qre_minimax_recovery_enabled)
        or bool(args.acbr_recovery_enabled)
    ):
        raise ValueError(
            "--bagr-recovery-enabled is a standalone recovery method and cannot be combined "
            "with game-teacher, planner-regret, AP-CVaR, QRE-Minimax, or ACBR recovery"
        )
    if bool(args.qre_minimax_recovery_enabled) and (
        bool(args.game_teacher_recovery_enabled)
        or bool(args.teacher_residual_recovery_enabled)
        or bool(args.planner_regret_recovery_enabled)
        or bool(args.ap_cvar_enabled)
        or bool(args.bagr_recovery_enabled)
        or bool(args.acbr_recovery_enabled)
    ):
        raise ValueError(
            "--qre-minimax-recovery-enabled is a standalone recovery method and cannot be combined "
            "with game-teacher, planner-regret, AP-CVaR, BAGR, or ACBR recovery"
        )
    if bool(args.acbr_recovery_enabled) and (
        bool(args.game_teacher_recovery_enabled)
        or bool(args.teacher_residual_recovery_enabled)
        or bool(args.planner_regret_recovery_enabled)
        or bool(args.ap_cvar_enabled)
        or bool(args.bagr_recovery_enabled)
        or bool(args.qre_minimax_recovery_enabled)
    ):
        raise ValueError(
            "--acbr-recovery-enabled is a standalone recovery method and cannot be combined "
            "with game-teacher, planner-regret, AP-CVaR, BAGR, QRE-Minimax, or TRR recovery"
        )
    if bool(args.teacher_residual_recovery_enabled) and (
        bool(args.game_teacher_recovery_enabled)
        or bool(args.planner_regret_recovery_enabled)
        or bool(args.ap_cvar_enabled)
        or bool(args.qre_minimax_recovery_enabled)
        or bool(args.bagr_recovery_enabled)
        or bool(args.acbr_recovery_enabled)
    ):
        raise ValueError(
            "--teacher-residual-recovery-enabled is a standalone recovery method and cannot be combined "
            "with old game-teacher, planner-regret, AP-CVaR, QRE-Minimax, BAGR, or ACBR recovery"
        )
    if int(args.game_attack_mixture_size) <= 0:
        raise ValueError("--game-attack-mixture-size must be positive")
    if float(args.game_attack_jitter) < 0.0:
        raise ValueError("--game-attack-jitter must be non-negative")
    if float(args.game_bandit_eta) < 0.0:
        raise ValueError("--game-bandit-eta must be non-negative")
    if float(args.game_bandit_min_prob) < 0.0:
        raise ValueError("--game-bandit-min-prob must be non-negative")
    if not 0.0 <= float(args.game_bandit_prior_mix) <= 1.0:
        raise ValueError("--game-bandit-prior-mix must be between 0 and 1")
    if not 0.0 <= float(args.game_bandit_benchmark_floor) <= 1.0:
        raise ValueError("--game-bandit-benchmark-floor must be between 0 and 1")
    if bool(args.sac_game_recovery_enabled):
        if float(args.sac_game_anchor_coef) < 0.0 or float(args.sac_game_advantage_coef) < 0.0:
            raise ValueError("SAC game-aware coefficients must be non-negative")
        if float(args.sac_game_q_margin) < 0.0:
            raise ValueError("--sac-game-q-margin must be non-negative")
        if float(args.sac_game_gate_temperature) <= 0.0:
            raise ValueError("--sac-game-gate-temperature must be positive")
        if float(args.sac_game_lambda_drift_coef) < 0.0 or float(args.sac_game_risk_drift_coef) < 0.0:
            raise ValueError("SAC game-aware drift coefficients must be non-negative")
        if float(args.sac_game_lambda_drift_margin) < 0.0 or float(args.sac_game_risk_drift_margin) < 0.0:
            raise ValueError("SAC game-aware drift margins must be non-negative")
        if float(args.sac_game_anchor_barrier_coef) < 0.0:
            raise ValueError("--sac-game-anchor-barrier-coef must be non-negative")
        if float(args.sac_game_anchor_radius) < 0.0:
            raise ValueError("--sac-game-anchor-radius must be non-negative")
        if float(args.sac_recovery_target_entropy_scale) < 0.0:
            raise ValueError("--sac-recovery-target-entropy-scale must be non-negative")
        if args.sac_recovery_fixed_alpha is not None and float(args.sac_recovery_fixed_alpha) < 0.0:
            raise ValueError("--sac-recovery-fixed-alpha must be non-negative when set")
        if not 0.0 <= float(args.sac_recovery_rollout_deterministic_prob) <= 1.0:
            raise ValueError("--sac-recovery-rollout-deterministic-prob must be in [0, 1]")
        if float(args.sac_recovery_rollout_noise_std) < 0.0:
            raise ValueError("--sac-recovery-rollout-noise-std must be non-negative")
        if float(args.sac_recovery_log_std_penalty_coef) < 0.0:
            raise ValueError("--sac-recovery-log-std-penalty-coef must be non-negative")
    if bool(args.qre_minimax_recovery_enabled):
        if float(args.qre_temperature_start) <= 0.0 or float(args.qre_temperature_end) <= 0.0:
            raise ValueError("QRE temperatures must be positive")
        if not 0.0 <= float(args.qre_response_rate) <= 1.0:
            raise ValueError("--qre-response-rate must be between 0 and 1")
        if not 0.0 <= float(args.qre_cost_ema_beta) < 1.0:
            raise ValueError("--qre-cost-ema-beta must be in [0, 1)")
        if not 0.0 <= float(args.qre_prior_mix) <= 1.0:
            raise ValueError("--qre-prior-mix must be between 0 and 1")
        if float(args.qre_min_prob) < 0.0:
            raise ValueError("--qre-min-prob must be non-negative")
        if float(args.qre_max_prob_cap) < 0.0:
            raise ValueError("--qre-max-prob-cap must be non-negative; use 0 to disable")
        if float(args.qre_max_prob_cap) > 0.0 and float(args.qre_max_prob_cap) < float(args.qre_benchmark_floor):
            raise ValueError("--qre-max-prob-cap must be at least --qre-benchmark-floor, or 0 to disable")
        if float(args.qre_exploration_bonus) < 0.0:
            raise ValueError("--qre-exploration-bonus must be non-negative")
        if not 0.0 <= float(args.qre_benchmark_floor) <= 1.0:
            raise ValueError("--qre-benchmark-floor must be between 0 and 1")
    if not 0.0 <= float(args.ap_cvar_quantile) <= 1.0:
        raise ValueError("--ap-cvar-quantile must be between 0 and 1")
    if float(args.ap_cvar_weight_cap) < 1.0:
        raise ValueError("--ap-cvar-weight-cap must be at least 1")
    if bool(args.planner_regret_recovery_enabled):
        if float(args.pr_recovery_alpha) <= 0.0:
            raise ValueError("--pr-recovery-alpha must be positive when planner-regret recovery is enabled")
        if int(args.pr_recovery_query_interval) <= 0:
            raise ValueError("--pr-recovery-query-interval must be positive")
        if int(args.pr_recovery_num_candidates) <= 0:
            raise ValueError("--pr-recovery-num-candidates must be positive")
        if not 0.0 <= float(args.pr_recovery_query_fraction) <= 1.0:
            raise ValueError("--pr-recovery-query-fraction must be between 0 and 1")
    if bool(args.game_teacher_recovery_enabled):
        if float(args.gt_recovery_alpha) <= 0.0:
            raise ValueError("--gt-recovery-alpha must be positive when game-teacher recovery is enabled")
        if int(args.gt_recovery_query_interval) <= 0:
            raise ValueError("--gt-recovery-query-interval must be positive")
        if int(args.gt_recovery_num_candidates) <= 0:
            raise ValueError("--gt-recovery-num-candidates must be positive")
        if int(args.gt_recovery_num_random_candidates) < 0:
            raise ValueError("--gt-recovery-num-random-candidates must be non-negative")
        if int(args.gt_recovery_num_structured_candidates) < 0:
            raise ValueError("--gt-recovery-num-structured-candidates must be non-negative")
        if int(args.gt_recovery_max_attack_variants) <= 0:
            raise ValueError("--gt-recovery-max-attack-variants must be positive")
        if int(args.gt_recovery_ramp_steps) < 0:
            raise ValueError("--gt-recovery-ramp-steps must be non-negative")
        if not 0.0 <= float(args.gt_recovery_query_fraction) <= 1.0:
            raise ValueError("--gt-recovery-query-fraction must be between 0 and 1")
        if float(args.gt_recovery_softmax_temperature) <= 0.0:
            raise ValueError("--gt-recovery-softmax-temperature must be positive")
        if float(args.gt_recovery_regret_weight_max) <= 0.0:
            raise ValueError("--gt-recovery-regret-weight-max must be positive")
    if bool(args.teacher_residual_recovery_enabled):
        if float(args.trr_recovery_alpha) <= 0.0:
            raise ValueError("--trr-recovery-alpha must be positive when teacher-residual recovery is enabled")
        if int(args.trr_recovery_query_interval) <= 0:
            raise ValueError("--trr-recovery-query-interval must be positive")
        if int(args.trr_recovery_num_candidates) <= 0:
            raise ValueError("--trr-recovery-num-candidates must be positive")
        if int(args.trr_recovery_num_random_candidates) < 0:
            raise ValueError("--trr-recovery-num-random-candidates must be non-negative")
        if int(args.trr_recovery_num_structured_candidates) < 0:
            raise ValueError("--trr-recovery-num-structured-candidates must be non-negative")
        if int(args.trr_recovery_max_attack_variants) <= 0:
            raise ValueError("--trr-recovery-max-attack-variants must be positive")
        if int(args.trr_recovery_ramp_steps) < 0:
            raise ValueError("--trr-recovery-ramp-steps must be non-negative")
        if int(args.trr_recovery_decay_to_zero_by_step) < int(args.trr_recovery_active_until_step):
            raise ValueError("--trr-recovery-decay-to-zero-by-step must be >= --trr-recovery-active-until-step")
        if not 0.0 <= float(args.trr_recovery_query_fraction) <= 1.0:
            raise ValueError("--trr-recovery-query-fraction must be between 0 and 1")
        if float(args.trr_recovery_softmax_temperature) <= 0.0:
            raise ValueError("--trr-recovery-softmax-temperature must be positive")
        if float(args.trr_recovery_regret_weight_max) <= 0.0:
            raise ValueError("--trr-recovery-regret-weight-max must be positive")
        if float(args.trr_recovery_min_normalized_regret) < 0.0:
            raise ValueError("--trr-recovery-min-normalized-regret must be non-negative")
        if float(args.trr_recovery_residual_l2_coef) < 0.0:
            raise ValueError("--trr-recovery-residual-l2-coef must be non-negative")
        if float(args.trr_recovery_residual_barrier_coef) < 0.0:
            raise ValueError("--trr-recovery-residual-barrier-coef must be non-negative")
        if float(args.trr_recovery_residual_action_limit) < 0.0:
            raise ValueError("--trr-recovery-residual-action-limit must be non-negative")
        if float(args.trr_recovery_aux_coef) < 0.0:
            raise ValueError("--trr-recovery-aux-coef must be non-negative")
        if int(args.trr_recovery_latent_dim) <= 0:
            raise ValueError("--trr-recovery-latent-dim must be positive")
        if int(args.trr_recovery_hidden_dim) <= 0:
            raise ValueError("--trr-recovery-hidden-dim must be positive")
    if bool(args.acbr_recovery_enabled):
        if float(args.acbr_recovery_critic_coef) <= 0.0:
            raise ValueError("--acbr-recovery-critic-coef must be positive when ACBR recovery is enabled")
        if int(args.acbr_recovery_query_interval) <= 0:
            raise ValueError("--acbr-recovery-query-interval must be positive")
        if int(args.acbr_recovery_num_candidates) <= 0:
            raise ValueError("--acbr-recovery-num-candidates must be positive")
        if int(args.acbr_recovery_num_random_candidates) < 0:
            raise ValueError("--acbr-recovery-num-random-candidates must be non-negative")
        if int(args.acbr_recovery_num_structured_candidates) < 0:
            raise ValueError("--acbr-recovery-num-structured-candidates must be non-negative")
        if not 0.0 <= float(args.acbr_recovery_query_fraction) <= 1.0:
            raise ValueError("--acbr-recovery-query-fraction must be between 0 and 1")
        if float(args.acbr_recovery_local_sigma) < 0.0 or float(args.acbr_recovery_risk_local_sigma) < 0.0:
            raise ValueError("ACBR recovery candidate sigmas must be non-negative")
        if float(args.acbr_recovery_uncertainty_coef) < 0.0:
            raise ValueError("--acbr-recovery-uncertainty-coef must be non-negative")
        if float(args.acbr_recovery_anchor_penalty) < 0.0 or float(args.acbr_recovery_policy_penalty) < 0.0:
            raise ValueError("ACBR recovery rerank penalties must be non-negative")
        if float(args.acbr_recovery_target_clip) <= 0.0:
            raise ValueError("--acbr-recovery-target-clip must be positive")
        if int(args.acbr_recovery_rerank_start_after_steps) < 0:
            raise ValueError("--acbr-recovery-rerank-start-after-steps must be non-negative")
        if float(args.acbr_recovery_benefit_margin) < 0.0:
            raise ValueError("--acbr-recovery-benefit-margin must be non-negative")
        if float(args.acbr_recovery_aux_coef) < 0.0:
            raise ValueError("--acbr-recovery-aux-coef must be non-negative")
        if int(args.acbr_recovery_latent_dim) <= 0 or int(args.acbr_recovery_hidden_dim) <= 0:
            raise ValueError("ACBR recovery PRB dimensions must be positive")
        if int(args.acbr_recovery_max_attack_variants) <= 0:
            raise ValueError("--acbr-recovery-max-attack-variants must be positive")
        if float(args.acbr_recovery_softmax_temperature) <= 0.0:
            raise ValueError("--acbr-recovery-softmax-temperature must be positive")
    if bool(args.bagr_recovery_enabled):
        if float(args.bagr_recovery_aux_coef) < 0.0:
            raise ValueError("--bagr-recovery-aux-coef must be non-negative")
        if int(args.bagr_recovery_latent_dim) <= 0:
            raise ValueError("--bagr-recovery-latent-dim must be positive")
        if int(args.bagr_recovery_hidden_dim) <= 0:
            raise ValueError("--bagr-recovery-hidden-dim must be positive")
        if int(args.bagr_recovery_max_attack_variants) <= 0:
            raise ValueError("--bagr-recovery-max-attack-variants must be positive")
        if float(args.bagr_recovery_belief_temperature) <= 0.0:
            raise ValueError("--bagr-recovery-belief-temperature must be positive")
        if not 0.0 <= float(args.bagr_recovery_belief_prior_mix) <= 1.0:
            raise ValueError("--bagr-recovery-belief-prior-mix must be between 0 and 1")
        if float(args.bagr_recovery_residual_l2_coef) < 0.0:
            raise ValueError("--bagr-recovery-residual-l2-coef must be non-negative")
        if float(args.bagr_recovery_residual_barrier_coef) < 0.0:
            raise ValueError("--bagr-recovery-residual-barrier-coef must be non-negative")
        if float(args.bagr_recovery_residual_action_limit) < 0.0:
            raise ValueError("--bagr-recovery-residual-action-limit must be non-negative")
        if float(args.bagr_recovery_confidence_limit_scale) < 0.0:
            raise ValueError("--bagr-recovery-confidence-limit-scale must be non-negative")

    if args.clean_output:
        clean_output_dir(args.output_dir, args.dry_run)
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.output_dir / "checkpoints"
    if not args.dry_run:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    level_config = read_json(args.level_config)
    env_attack = load_environment_attack(level_config)
    if not attack_enabled(env_attack):
        raise ValueError("environment attack must be enabled for shock-recovery")

    splits: dict[str, Path] | None = None
    if is_real_level(level_config):
        base_args, splits = real_base_args(
            level_config,
            args.base_config,
            args.output_dir,
            args.seed,
            args.nominal_timesteps,
            args.eval_interval,
            args.train_eval_episodes,
            args.quick,
            args.dry_run,
        )
        if not args.dry_run:
            map_size, episodes_by_domain = build_real_eval_episodes(level_config, splits, args.seed, args.num_eval_episodes)
    else:
        base_args = synthetic_base_args(
            level_config,
            args.base_config,
            args.output_dir,
            args.seed,
            args.nominal_timesteps,
            args.eval_interval,
            args.train_eval_episodes,
        )
        map_pool_size = int(
            args.map_pool_size
            if args.map_pool_size is not None
            else config_value(base_args, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE)
        )
        if not args.dry_run:
            map_size, episodes_by_domain = build_synthetic_eval_episodes(args, base_args, map_pool_size)

    if args.device:
        base_args["device"] = str(args.device)

    if not args.dry_run:
        write_run_config(args.output_dir, args, level_config, base_args, env_attack, splits)

    game_variants = game_attack_variants(env_attack, args) if bool(args.game_recovery_enabled) else []
    attack_bandit_probs = enforce_benchmark_floor(
        initial_game_attack_probs(len(game_variants)),
        attack_probability_benchmark_floor(args),
    )
    use_adaptive_attack_sampler = (
        bool(args.game_recovery_enabled)
        and str(args.game_attack_sampler) in {"adaptive_bandit", "qre_minimax"}
        and len(game_variants) > 1
    )
    qre_state: dict[str, Any] = {}
    total_recovery_chunks = int(np.ceil(max(int(args.recovery_timesteps), 1) / max(int(args.eval_interval), 1)))

    nominal_model, nominal_eval_csv = train_nominal(args.python, base_args, args.output_dir, args.dry_run, args.algo)
    nominal_checkpoint = checkpoints_dir / "checkpoint_nominal.pt"
    if args.dry_run:
        print(f"Would copy checkpoint {nominal_model} -> {nominal_checkpoint}")
        current_checkpoint = nominal_checkpoint
        recovery_step = 0
        chunk_index = 0
        while recovery_step < int(args.recovery_timesteps):
            chunk_index += 1
            chunk_timesteps = min(int(args.eval_interval), int(args.recovery_timesteps) - recovery_step)
            recovery_train_attack = build_game_recovery_attack(
                env_attack,
                args,
                args.seed + 20_000 + chunk_index,
                probs=attack_bandit_probs,
                recovery_step_offset=recovery_step,
                chunk_index=chunk_index,
                total_chunks=total_recovery_chunks,
            )
            rollout_metrics_path = (
                attack_bandit_rollout_metrics_path(args.output_dir, chunk_index, str(args.game_attack_sampler))
                if use_adaptive_attack_sampler
                else None
            )
            command, _chunk_final = build_train_command(
                args.python,
                base_args,
                current_checkpoint,
                recovery_train_attack,
                disabled_attack(),
                args.output_dir,
                chunk_index,
                chunk_timesteps,
                args.seed + 10_000 + chunk_index,
                args.algo,
                extra_train_args=game_recovery_train_args(
                    args,
                    nominal_checkpoint,
                    rollout_metrics_path,
                    recovery_step_offset=recovery_step,
                    chunk_index=chunk_index,
                ),
            )
            print(" ".join(str(part) for part in command), flush=True)
            if use_adaptive_attack_sampler:
                print(
                    f"Would update {args.game_attack_sampler} attack sampler from "
                    f"{rollout_metrics_path} after chunk {chunk_index}.",
                    flush=True,
                )
            recovery_step += chunk_timesteps
            current_checkpoint = checkpoints_dir / f"checkpoint_recovery_step_{recovery_step:05d}.pt"
        print("Dry run complete.")
        return 0

    if not args.dry_run:
        copy_checkpoint(nominal_model, nominal_checkpoint, args.dry_run)
        write_nominal_eval(nominal_eval_csv, args.output_dir)

    rows: list[dict[str, Any]] = []
    eval_checkpoint_path = nominal_checkpoint if nominal_checkpoint.exists() else nominal_model
    rows.extend(
        add_protocol_columns(
            evaluate_checkpoint(
                eval_checkpoint_path,
                0,
                episodes_by_domain,
                map_size,
                env_attack,
                disabled_attack(),
                args.seed,
            ),
            phase="shock",
            recovery_step=0,
            checkpoint_role="nominal",
        )
    )

    current_checkpoint = eval_checkpoint_path
    recovery_step = 0
    chunk_index = 0
    while recovery_step < int(args.recovery_timesteps):
        chunk_index += 1
        chunk_timesteps = min(int(args.eval_interval), int(args.recovery_timesteps) - recovery_step)
        recovery_train_attack = build_game_recovery_attack(
            env_attack,
            args,
            args.seed + 20_000 + chunk_index,
            probs=attack_bandit_probs,
            recovery_step_offset=recovery_step,
            chunk_index=chunk_index,
            total_chunks=total_recovery_chunks,
        )
        rollout_metrics_path = (
            attack_bandit_rollout_metrics_path(args.output_dir, chunk_index, str(args.game_attack_sampler))
            if use_adaptive_attack_sampler
            else None
        )
        command, chunk_final = build_train_command(
            args.python,
            base_args,
            current_checkpoint,
            recovery_train_attack,
            disabled_attack(),
            args.output_dir,
            chunk_index,
            chunk_timesteps,
            args.seed + 10_000 + chunk_index,
            args.algo,
            extra_train_args=game_recovery_train_args(
                args,
                nominal_checkpoint,
                rollout_metrics_path,
                recovery_step_offset=recovery_step,
                chunk_index=chunk_index,
            ),
        )
        print(" ".join(str(part) for part in command), flush=True)
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        if not chunk_final.exists():
            raise FileNotFoundError(f"expected recovery checkpoint not found: {chunk_final}")
        actual_step = checkpoint_step(chunk_final)
        recovery_step += actual_step if actual_step > 0 else chunk_timesteps
        current_checkpoint = checkpoints_dir / f"checkpoint_recovery_step_{recovery_step:05d}.pt"
        copy_checkpoint(chunk_final, current_checkpoint, args.dry_run)
        if use_adaptive_attack_sampler and rollout_metrics_path is not None:
            attack_bandit_probs, bandit_rows = update_attack_bandit_probs(
                attack_bandit_probs,
                game_variants,
                rollout_metrics_path,
                args,
                qre_state=qre_state,
                chunk_index=chunk_index,
                total_chunks=total_recovery_chunks,
            )
            append_attack_bandit_history(args.output_dir, chunk_index, bandit_rows, str(args.game_attack_sampler))
            if bandit_rows:
                best_variant = max(bandit_rows, key=lambda row: float(row["prob_after"]))
                sampler_label = "QRE-Minimax" if str(args.game_attack_sampler) == "qre_minimax" else "Adaptive attack"
                qre_suffix = ""
                if str(args.game_attack_sampler) == "qre_minimax":
                    qre_suffix = f" temp={float(best_variant.get('qre_temperature', np.nan)):.3f}"
                print(
                    f"{sampler_label} sampler updated: "
                    f"chunk={chunk_index} top_variant={best_variant['variant_id']} "
                    f"prob={float(best_variant['prob_after']):.3f} "
                    f"mean_cost={float(best_variant['mean_cost']):.4f}{qre_suffix}",
                    flush=True,
                )
        rows.extend(
            add_protocol_columns(
                evaluate_checkpoint(
                    current_checkpoint,
                    recovery_step,
                    episodes_by_domain,
                    map_size,
                    env_attack,
                    disabled_attack(),
                    args.seed,
                ),
                phase="recovery",
                recovery_step=recovery_step,
                checkpoint_role="recovery",
            )
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "shock_recovery_curve.csv", index=False)
    summarize_recovery(frame, args.output_dir)
    plot_outputs(frame, args.output_dir)
    write_output_guide(args.output_dir)
    print(f"Saved shock-recovery outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
