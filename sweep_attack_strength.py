"""Sweep environmental attack strength for a trained PPO planner policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from envs.attack_wrappers import attack_enabled
from utils.evaluation_policy import (
    load_model,
    predict_action,
    resolve_action_config,
    resolve_observation_mode,
)
from utils.metrics import (
    DEFAULT_FIXED_MAP_SEED,
    DEFAULT_MAP_SEED_POOL_SIZE,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    compute_observation,
    make_curriculum_planning_episode,
    plan_with_weights,
)


FULL_TOP_FRACTIONS = [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]
FULL_SHARPNESS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
FULL_TEMPERATURES = [0.5, 0.3, 0.2]
QUICK_TOP_FRACTIONS = [0.15, 0.25, 0.35, 0.50]
QUICK_SHARPNESS = [3.0, 5.0, 8.0]
QUICK_TEMPERATURES = [0.5]
FULL_ATTACK_STRENGTHS = [1.0, 2.0, 3.0, 5.0, 8.0]
QUICK_ATTACK_STRENGTHS = [1.0, 2.0, 3.0, 5.0]


def parse_args() -> argparse.Namespace:
    default_checkpoint = Path("runs/robustness/seed0/nominal_ppo/checkpoint.pt")
    if not default_checkpoint.exists():
        default_checkpoint = Path("runs/robustness/nominal_ppo/checkpoint.pt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, default="configs/ppo_lunar_map_pool_relative_reward.json")
    parser.add_argument("--checkpoint", type=str, default=str(default_checkpoint))
    parser.add_argument("--output-dir", type=str, default="runs/robustness/attack_sweep")
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--domains", choices=("both", "in_domain", "heldout"), default="both")
    parser.add_argument("--in-domain-seed", type=int, default=909)
    parser.add_argument("--heldout-seed", type=int, default=1919)
    parser.add_argument("--map-pool-size", type=int, default=None)
    parser.add_argument("--map-size", type=int, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--target-low", type=float, default=0.10)
    parser.add_argument("--target-high", type=float, default=0.30)
    parser.add_argument(
        "--attack-strength-values",
        type=str,
        default=None,
        help="Comma-separated attack_strength values. Defaults depend on --quick.",
    )
    return parser.parse_args()


def load_base_args(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("args", data))


def config_value(config: dict[str, Any], hyphen_key: str, default: Any) -> Any:
    return config.get(hyphen_key, config.get(hyphen_key.replace("-", "_"), default))


def domain_specs(args: argparse.Namespace) -> list[tuple[str, int]]:
    specs = []
    if args.domains in {"both", "in_domain"}:
        specs.append((f"in_domain_seed{args.in_domain_seed}", int(args.in_domain_seed)))
    if args.domains in {"both", "heldout"}:
        specs.append((f"heldout_seed{args.heldout_seed}", int(args.heldout_seed)))
    return specs


def parse_float_list(text: str | None, default: list[float]) -> list[float]:
    if not text:
        return default
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def attack_config(top_fraction: float, sharpness: float, temperature: float, attack_strength: float) -> dict[str, Any]:
    return {
        "enabled": True,
        "type": "env_zscore_topk",
        "attacker_response": "zscore_topk",
        "attacker_temperature": float(temperature),
        "attacker_top_fraction": float(top_fraction),
        "attacker_sharpness": float(sharpness),
        "attack_strength": float(attack_strength),
        "apply_during_training": True,
        "reward_uses_attacked_cost": True,
    }


def eval_cost(result: dict[str, Any], env_attack: dict[str, Any]) -> float:
    if not attack_enabled(env_attack):
        return float(result.get("scalar_cost", np.nan))
    if str(env_attack.get("type", "env_zscore_topk")) == "env_zscore_topk":
        return float(result.get("soft_attacked_scalar_cost", result.get("attacked_scalar_cost", np.nan)))
    return float(result.get("attacked_scalar_cost", result.get("scalar_cost", np.nan)))


def policy_action_to_plan(
    model_type: str,
    model: Any,
    model_config: dict[str, Any],
    episode: Any,
    map_size: int,
) -> tuple[np.ndarray, float]:
    observation_mode = resolve_observation_mode("auto", model_config)
    action_mode, action_gain, max_lambda = resolve_action_config("auto", None, None, model_config)
    obs = compute_observation(
        episode,
        map_size,
        observation_mode=observation_mode,
        max_uncertainty_lambda=max_lambda,
    )
    action = predict_action(model_type, model, obs)
    weights = action_to_planning_weights(
        episode,
        action,
        action_mode=action_mode,
        action_gain=action_gain,
    )
    lambda_uncertainty = action_to_uncertainty_lambda(action, max_uncertainty_lambda=max_lambda)
    return weights, float(lambda_uncertainty)


def generate_episodes(
    num_episodes: int,
    seed: int,
    map_size: int,
    scenario: str,
    fixed_map_seed: int,
    map_pool_size: int,
) -> list[Any]:
    rng = np.random.default_rng(seed)
    map_cache: dict[Any, Any] = {}
    return [
        make_curriculum_planning_episode(
            map_size=map_size,
            rng=rng,
            allow_diagonal=True,
            scenario=scenario,
            map_sampling_mode="map_seed_pool",
            fixed_map_seed=fixed_map_seed,
            map_seed_pool_size=map_pool_size,
            map_cache=map_cache,
        )
        for _ in tqdm(range(num_episodes), desc=f"episodes seed{fixed_map_seed}", leave=False)
    ]


def evaluate_setting(
    episodes: list[Any],
    model_type: str,
    model: Any,
    model_config: dict[str, Any],
    map_size: int,
    env_attack: dict[str, Any],
) -> dict[str, float]:
    nominal_costs = []
    attacked_costs = []
    successes = []
    path_lengths = []
    planning_times = []
    attacked_cost_std_values = []

    for episode in episodes:
        weights, lambda_uncertainty = policy_action_to_plan(
            model_type,
            model,
            model_config,
            episode,
            map_size,
        )
        clean_result = plan_with_weights(
            episode,
            weights,
            lambda_uncertainty=lambda_uncertainty,
            allow_diagonal=True,
        )
        start = time.perf_counter()
        result = plan_with_weights(
            episode,
            weights,
            lambda_uncertainty=lambda_uncertainty,
            allow_diagonal=True,
            attacker_temperature=float(env_attack["attacker_temperature"]),
            attacker_response=str(env_attack["attacker_response"]),
            attacker_top_fraction=float(env_attack["attacker_top_fraction"]),
            attacker_sharpness=float(env_attack["attacker_sharpness"]),
            attack_strength=float(env_attack.get("attack_strength", 1.0)),
        )
        planning_times.append(time.perf_counter() - start)
        nominal_costs.append(float(clean_result.get("scalar_cost", np.nan)))
        cost = eval_cost(result, env_attack)
        attacked_costs.append(cost)
        attacked_cost_std_values.append(cost)
        successes.append(1.0 if bool(result.get("success", False)) else 0.0)
        path_lengths.append(float(result.get("path_length", np.nan)))

    mean_nominal = float(np.nanmean(nominal_costs))
    mean_attacked = float(np.nanmean(attacked_costs))
    absolute = mean_attacked - mean_nominal
    return {
        "mean_nominal_scalar_cost": mean_nominal,
        "std_nominal_scalar_cost": float(np.nanstd(nominal_costs)),
        "mean_attacked_scalar_cost": mean_attacked,
        "std_attacked_scalar_cost": float(np.nanstd(attacked_cost_std_values)),
        "absolute_degradation": absolute,
        "relative_degradation": float(absolute / (abs(mean_nominal) + 1e-8)),
        "success_rate": float(np.nanmean(successes)),
        "failure_rate": float(1.0 - np.nanmean(successes)),
        "mean_path_length": float(np.nanmean(path_lengths)),
        "mean_planning_time": float(np.nanmean(planning_times)),
    }


def recommend_setting(frame: pd.DataFrame, target_low: float, target_high: float) -> dict[str, Any]:
    grouped = (
        frame.groupby(
            ["attacker_top_fraction", "attacker_sharpness", "attacker_temperature", "attack_strength"],
            as_index=False,
        )
        .agg(
            relative_degradation=("relative_degradation", "mean"),
            success_rate=("success_rate", "min"),
            mean_attacked_scalar_cost=("mean_attacked_scalar_cost", "mean"),
            std_attacked_scalar_cost=("std_attacked_scalar_cost", "mean"),
        )
    )
    candidates = grouped[
        (grouped["relative_degradation"] >= target_low)
        & (grouped["relative_degradation"] <= target_high)
        & (grouped["success_rate"] >= 0.95)
    ].copy()
    used_target_range = True
    if candidates.empty:
        candidates = grouped[grouped["success_rate"] >= 0.95].copy()
        used_target_range = False
    if candidates.empty:
        candidates = grouped.copy()
        used_target_range = False

    candidates["score"] = (
        (candidates["success_rate"] - 1.0).abs() * 100.0
        + (candidates["relative_degradation"] - 0.20).abs()
        + candidates["attacker_top_fraction"] * 0.01
        + candidates["std_attacked_scalar_cost"].fillna(0.0) * 0.001
    )
    row = candidates.sort_values("score").iloc[0].to_dict()
    config = attack_config(
        float(row["attacker_top_fraction"]),
        float(row["attacker_sharpness"]),
        float(row["attacker_temperature"]),
        float(row["attack_strength"]),
    )
    return {
        "environment_attack": config,
        "selection_basis": {
            "target_relative_degradation_range": [float(target_low), float(target_high)],
            "target_range_satisfied": bool(used_target_range),
            "mean_relative_degradation": float(row["relative_degradation"]),
            "min_success_rate": float(row["success_rate"]),
            "mean_attacked_scalar_cost": float(row["mean_attacked_scalar_cost"]),
            "mean_std_attacked_scalar_cost": float(row["std_attacked_scalar_cost"]),
        },
    }


def plot_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for (domain, temperature, sharpness, attack_strength), group in frame.groupby(
        ["eval_domain", "attacker_temperature", "attacker_sharpness", "attack_strength"]
    ):
        group = group.sort_values("attacker_top_fraction")
        label = f"{domain} T={temperature:g} S={sharpness:g} A={attack_strength:g}"
        ax.plot(group["attacker_top_fraction"], group["relative_degradation"], marker="o", label=label)
    ax.axhspan(0.10, 0.30, color="tab:green", alpha=0.10, label="target 10-30%")
    ax.set_xlabel("attacker_top_fraction")
    ax.set_ylabel("relative_degradation")
    ax.set_title("Environmental attack strength curve")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_attack_strength_curve.png", dpi=180)
    plt.close(fig)

    domains = list(frame["eval_domain"].unique())
    temps = list(frame["attacker_temperature"].unique())
    ncols = max(len(temps), 1)
    nrows = max(len(domains), 1)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    for row_index, domain in enumerate(domains):
        for col_index, temp in enumerate(temps):
            ax = axes[row_index][col_index]
            subset = frame[(frame["eval_domain"] == domain) & (frame["attacker_temperature"] == temp)]
            if "attack_strength" in subset:
                strongest = subset["attack_strength"].max()
                subset = subset[subset["attack_strength"] == strongest]
            heat = subset.pivot_table(
                index="attacker_sharpness",
                columns="attacker_top_fraction",
                values="relative_degradation",
                aggfunc="mean",
            ).sort_index(ascending=False)
            im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
            ax.set_title(f"{domain}, T={temp:g}")
            ax.set_xlabel("top_fraction")
            ax.set_ylabel("sharpness")
            ax.set_xticks(range(len(heat.columns)))
            ax.set_xticklabels([f"{value:g}" for value in heat.columns], rotation=45)
            ax.set_yticks(range(len(heat.index)))
            ax.set_yticklabels([f"{value:g}" for value in heat.index])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_attack_heatmap.png", dpi=180)
    plt.close(fig)


def update_diagnostic_report(output_dir: Path, recommendation: dict[str, Any]) -> None:
    report_path = output_dir.parent / "diagnostic_report.md"
    env_attack = recommendation["environment_attack"]
    basis = recommendation["selection_basis"]
    report_path.write_text(
        "\n".join(
            [
                "# Robustness Diagnostic Report",
                "",
                "## 1. Current environmental attack strength",
                "",
                f"- Recommended setting: top_fraction={env_attack['attacker_top_fraction']}, "
                f"sharpness={env_attack['attacker_sharpness']}, "
                f"temperature={env_attack['attacker_temperature']}, "
                f"attack_strength={env_attack.get('attack_strength', 1.0)}.",
                f"- Nominal PPO relative degradation under this setting: "
                f"{basis['mean_relative_degradation']:.4f}.",
                f"- Minimum success_rate across evaluated domains: {basis['min_success_rate']:.4f}.",
                "",
                "Other sections are filled after running recovery fine-tuning and policy analysis.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved diagnostic report stub to {report_path}")


def main() -> None:
    args = parse_args()
    base_args = load_base_args(args.base_config)
    seed = int(args.seed if args.seed is not None else config_value(base_args, "seed", 222))
    map_size = int(args.map_size if args.map_size is not None else config_value(base_args, "map-size", 48))
    scenario = str(args.scenario if args.scenario is not None else config_value(base_args, "scenario", "lunar_rover"))
    map_pool_size = int(
        args.map_pool_size
        if args.map_pool_size is not None
        else config_value(base_args, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_type, model, model_config = load_model(args.checkpoint, "auto")

    if args.quick:
        top_fractions = QUICK_TOP_FRACTIONS
        sharpness_values = QUICK_SHARPNESS
        temperature_values = QUICK_TEMPERATURES
        attack_strength_values = parse_float_list(args.attack_strength_values, QUICK_ATTACK_STRENGTHS)
    else:
        top_fractions = FULL_TOP_FRACTIONS
        sharpness_values = FULL_SHARPNESS
        temperature_values = FULL_TEMPERATURES
        attack_strength_values = parse_float_list(args.attack_strength_values, FULL_ATTACK_STRENGTHS)

    rows: list[dict[str, Any]] = []
    for eval_domain, map_seed in domain_specs(args):
        episodes = generate_episodes(
            num_episodes=args.num_episodes,
            seed=seed,
            map_size=map_size,
            scenario=scenario,
            fixed_map_seed=map_seed,
            map_pool_size=map_pool_size,
        )
        grid = [
            (top_fraction, sharpness, temperature, attack_strength)
            for temperature in temperature_values
            for attack_strength in attack_strength_values
            for sharpness in sharpness_values
            for top_fraction in top_fractions
        ]
        for top_fraction, sharpness, temperature, attack_strength in tqdm(grid, desc=eval_domain):
            env_attack = attack_config(top_fraction, sharpness, temperature, attack_strength)
            metrics = evaluate_setting(episodes, model_type, model, model_config, map_size, env_attack)
            rows.append(
                {
                    "eval_domain": eval_domain,
                    "map_pool_seed": map_seed,
                    "map_pool_size": map_pool_size,
                    "num_episodes": args.num_episodes,
                    "attacker_top_fraction": top_fraction,
                    "attacker_sharpness": sharpness,
                    "attacker_temperature": temperature,
                    "attack_strength": attack_strength,
                    **metrics,
                }
            )

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "attack_strength_sweep.csv"
    frame.to_csv(csv_path, index=False)
    recommendation = recommend_setting(frame, args.target_low, args.target_high)
    recommendation_path = output_dir / "recommended_attack_config.json"
    recommendation_path.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")
    plot_outputs(frame, output_dir)
    update_diagnostic_report(output_dir, recommendation)

    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved attack strength sweep CSV to {csv_path}")
    print(f"Saved recommendation JSON to {recommendation_path}")
    print(f"Saved attack strength curve to {output_dir / 'fig_attack_strength_curve.png'}")
    print(f"Saved attack heatmap to {output_dir / 'fig_attack_heatmap.png'}")


if __name__ == "__main__":
    main()
