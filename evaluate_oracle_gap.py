"""Oracle gap analysis for attacked rover planning benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from envs.attack_wrappers import apply_environment_attack_to_episode, attack_enabled, load_attack_config
from utils.evaluation_policy import (
    load_model,
    predict_action,
    resolve_action_config,
    resolve_observation_mode,
)
from utils.metrics import (
    DEFAULT_MAP_SEED_POOL_SIZE,
    OBJECTIVE_NAMES,
    candidate_planner_configs,
    compute_observation,
    make_curriculum_planning_episode,
    normalize_weights,
    path_overlap_ratio,
    plan_with_weights,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--nominal-checkpoint", type=str, default="runs/robustness/seed0/nominal_ppo/checkpoint.pt")
    parser.add_argument("--env-ft-checkpoint", type=str, default=None)
    parser.add_argument("--obs-ft-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/robustness/oracle_gap")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--num-random-candidates", type=int, default=128)
    parser.add_argument("--num-structured-candidates", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--map-size", type=int, default=None)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--map-sampling-mode", type=str, default=None)
    parser.add_argument("--fixed-map-seed", type=int, default=None)
    parser.add_argument("--map-seed-pool-size", type=int, default=None)
    parser.add_argument("--eval-domain", type=str, default=None)
    parser.add_argument("--heldout-fixed-map-seed", type=int, default=1919)
    parser.add_argument("--domains", choices=("in_domain", "heldout", "both"), default="in_domain")
    parser.add_argument("--attack-types", type=str, default="none,zscore_topk,path_corridor")
    parser.add_argument("--environment-attack-config", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def load_config(path_text: str | None) -> dict[str, Any]:
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.exists():
        print(f"WARNING: config not found, using defaults: {path}")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data.get("args", data))


def cfg_value(args: argparse.Namespace, config: dict[str, Any], key: str, default: Any) -> Any:
    attr = key.replace("-", "_")
    value = getattr(args, attr, None)
    if value is not None:
        return value
    return config.get(key, config.get(attr, default))


def load_optional_policy(path_text: str | None) -> tuple[str, Any, dict[str, Any]] | None:
    if not path_text:
        return None
    path = Path(path_text)
    if not path.exists():
        print(f"WARNING: checkpoint not found, skipping: {path}")
        return None
    return load_model(path, "auto")


def ppo_candidate(
    policy: tuple[str, Any, dict[str, Any]],
    episode: Any,
    map_size: int,
) -> dict[str, Any]:
    model_type, model, model_config = policy
    observation_mode = resolve_observation_mode("auto", model_config)
    action_mode, action_gain, max_lambda = resolve_action_config("auto", None, None, model_config)
    obs = compute_observation(
        episode,
        map_size,
        observation_mode=observation_mode,
        max_uncertainty_lambda=max_lambda,
    )
    action = predict_action(model_type, model, obs)
    from utils.metrics import action_to_planning_weights, action_to_uncertainty_lambda

    return {
        "weights": action_to_planning_weights(episode, action, action_mode=action_mode, action_gain=action_gain),
        "lambda_uncertainty": action_to_uncertainty_lambda(action, max_uncertainty_lambda=max_lambda),
    }


def attack_config_for_type(attack_type: str, base: dict[str, Any]) -> dict[str, Any] | None:
    if attack_type == "none":
        return None
    if attack_type == "zscore_topk":
        cfg = {
            "enabled": True,
            "type": "env_zscore_topk",
            "attacker_response": "zscore_topk",
            "attacker_temperature": 0.5,
            "attacker_top_fraction": 0.15,
            "attacker_sharpness": 3.0,
            "attack_strength": 1.0,
        }
        allowed = {
            "attacker_response",
            "attacker_temperature",
            "attacker_top_fraction",
            "attacker_sharpness",
            "attack_strength",
            "attack_budget_fraction",
        }
        cfg.update({k: v for k, v in base.items() if k in allowed})
        cfg["enabled"] = True
        cfg["type"] = "env_zscore_topk"
        return cfg
    if attack_type == "path_corridor":
        cfg = {
            "enabled": True,
            "type": "env_path_corridor_attack",
            "reference_policy": "heuristic",
            "corridor_radius": 2,
            "attack_strength": 5.0,
            "affected_layers": ["hazard", "uncertainty"],
        }
        cfg.update(base)
        cfg["type"] = "env_path_corridor_attack"
        cfg.setdefault("enabled", True)
        return cfg
    raise ValueError(f"unknown attack type: {attack_type}")


def eval_cost(result: dict[str, Any], attack_type: str) -> float:
    if attack_type == "zscore_topk":
        return float(result.get("soft_attacked_scalar_cost", result.get("attacked_scalar_cost", np.nan)))
    return float(result.get("scalar_cost", np.nan))


def random_candidates(
    rng: np.random.Generator,
    count: int,
    max_lambda: float,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "method": "random_weight_candidate",
                "candidate_id": f"random_{index}",
                "weights": normalize_weights(rng.dirichlet(np.ones(len(OBJECTIVE_NAMES))).astype(np.float32)),
                "lambda_uncertainty": float(rng.uniform(0.0, max_lambda)),
            }
        )
    return rows


def structured_candidates(
    episode: Any,
    max_lambda: float,
    count: int,
) -> list[dict[str, Any]]:
    configs = candidate_planner_configs(episode, max_uncertainty_lambda=max_lambda)
    rows = []
    preferred = [
        "heuristic",
        "safe_rover",
        "mission_safe_blend",
        "hazard_only_uncertainty_high",
        "mission_hazard_blend_uncertainty_high",
        "illumination_only_uncertainty_high",
        "communication_only_uncertainty_high",
        "power_comms",
        "rover_guard",
        "emergency_uncertainty_rule",
    ]
    names = [name for name in preferred if name in configs]
    if len(names) < count:
        names.extend([name for name in configs if name not in names])
    for name in names[:count]:
        cfg = configs[name]
        rows.append(
            {
                "method": "structured_candidate",
                "candidate_id": name,
                "weights": np.asarray(cfg["weights"], dtype=np.float32),
                "lambda_uncertainty": float(cfg["lambda_uncertainty"]),
            }
        )
    return rows


def base_candidate_rows(
    episode: Any,
    map_size: int,
    policies: dict[str, tuple[str, Any, dict[str, Any]] | None],
) -> list[dict[str, Any]]:
    rows = [
        {
            "method": "fixed_heuristic",
            "candidate_id": "mission_priority",
            "weights": episode.mission_priority.astype(np.float32),
            "lambda_uncertainty": 0.0,
        }
    ]
    for method, policy in policies.items():
        if policy is None:
            continue
        cfg = ppo_candidate(policy, episode, map_size)
        rows.append({"method": method, "candidate_id": method, **cfg})
    return rows


def generate_episodes(
    num_episodes: int,
    seed: int,
    map_size: int,
    scenario: str,
    map_sampling_mode: str,
    fixed_map_seed: int,
    map_pool_size: int,
) -> list[Any]:
    rng = np.random.default_rng(seed)
    cache: dict[Any, Any] = {}
    return [
        make_curriculum_planning_episode(
            map_size=map_size,
            rng=rng,
            allow_diagonal=True,
            scenario=scenario,
            map_sampling_mode=map_sampling_mode,
            fixed_map_seed=fixed_map_seed,
            map_seed_pool_size=map_pool_size,
            map_cache=cache,
        )
        for _ in tqdm(range(num_episodes), desc=f"episodes seed{fixed_map_seed}", leave=False)
    ]


def method_summary_rows(
    episode_rows: list[dict[str, Any]],
    eval_domain: str,
    attack_type: str,
    map_seed: int,
    num_episodes: int,
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(episode_rows)
    rows = []
    for method, group in frame.groupby("method", sort=False):
        costs = group["attacked_scalar_cost"].to_numpy(dtype=np.float64)
        nominal = group["nominal_scalar_cost"].to_numpy(dtype=np.float64)
        p90 = float(np.nanquantile(costs, 0.90)) if len(costs) else np.nan
        mean_nominal = float(np.nanmean(nominal))
        mean_attacked = float(np.nanmean(costs))
        absolute = mean_attacked - mean_nominal
        oracle_gap = float(group["oracle_gap_vs_nominal"].mean(skipna=True))
        rows.append(
            {
                "eval_domain": eval_domain,
                "map_pool_seed": map_seed,
                "attack_type": attack_type,
                "episode_id": "all",
                "method": method,
                "num_episodes": num_episodes,
                "mean_attacked_scalar_cost": mean_attacked,
                "mean_nominal_scalar_cost": mean_nominal,
                "relative_degradation": float(absolute / (abs(mean_nominal) + 1e-8)),
                "oracle_gap_vs_nominal": oracle_gap,
                "oracle_gap_pct": float(group["oracle_gap_pct"].mean(skipna=True)),
                "success_rate": float(group["success"].mean()),
                "path_length": float(group["path_length"].mean()),
                "cvar90_attacked_cost": float(np.nanmean(costs[costs >= p90])) if np.isfinite(p90) else np.nan,
                "worst10_mean_attacked_cost": float(np.nanmean(costs[costs >= p90])) if np.isfinite(p90) else np.nan,
                "mean_attacked_cell_exposure_ratio": float(group["attacked_cell_exposure_ratio"].mean()),
                "mean_path_overlap_vs_nominal": float(group["path_overlap_vs_nominal"].mean(skipna=True)),
                "mean_path_overlap_vs_oracle": float(group["path_overlap_vs_oracle"].mean(skipna=True)),
                "path_switch_rate_vs_nominal": float((group["path_overlap_vs_nominal"] < 0.95).mean()),
            }
        )
    return rows


def plot_oracle_outputs(summary: pd.DataFrame, details: pd.DataFrame, output_dir: Path) -> None:
    figure_dir = output_dir
    figure_dir.mkdir(parents=True, exist_ok=True)

    key_methods = ["nominal_ppo", "env_ft_ppo", "obs_ft_ppo", "oracle_best_of_candidates"]
    subset = summary[summary["method"].isin(key_methods)].copy()
    subset["label"] = subset["eval_domain"] + "\n" + subset["attack_type"]
    labels = list(subset["label"].drop_duplicates())
    x = np.arange(len(labels))
    width = 0.18
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, method in enumerate(key_methods):
        values = []
        for label in labels:
            row = subset[(subset["label"] == label) & (subset["method"] == method)]
            values.append(float(row["mean_attacked_scalar_cost"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (index - 1.5) * width, values, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("mean_attacked_scalar_cost")
    ax.set_title("Oracle gap by attack")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_oracle_gap_by_attack.png", dpi=180)
    plt.close(fig)

    oracle_rows = details[details["method"] == "oracle_best_of_candidates"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for attack_type, group in oracle_rows.groupby("attack_type"):
        ax.hist(group["oracle_gap_pct"], bins=24, alpha=0.55, label=attack_type)
    ax.axvline(0.01, color="tab:orange", linestyle="--", linewidth=1, label="1%")
    ax.axvline(0.05, color="tab:green", linestyle="--", linewidth=1, label="5%")
    ax.set_xlabel("oracle_gap_pct vs nominal PPO")
    ax.set_ylabel("episode count")
    ax.set_title("Oracle gap distribution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_oracle_gap_distribution.png", dpi=180)
    plt.close(fig)

    exposure = summary[summary["method"].isin(key_methods)].copy()
    exposure["label"] = exposure["method"] + "\n" + exposure["attack_type"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(np.arange(len(exposure)), exposure["mean_attacked_cell_exposure_ratio"])
    ax.set_xticks(np.arange(len(exposure)))
    ax.set_xticklabels(exposure["label"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("mean_attacked_cell_exposure_ratio")
    ax.set_title("Exposure to attacked corridor")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_attacked_cell_exposure.png", dpi=180)
    plt.close(fig)


def plot_path_examples(examples: list[dict[str, Any]], output_dir: Path) -> None:
    example_dir = output_dir / "path_examples"
    example_dir.mkdir(parents=True, exist_ok=True)
    for index, example in enumerate(examples[:6]):
        episode = example["episode"]
        paths = example["paths"]
        mask = getattr(episode.costmap, "attack_mask", None)
        background = episode.costmap.layers["hazard"]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(background, cmap="gray", origin="upper")
        if mask is not None:
            ax.imshow(np.ma.masked_where(~mask, mask), cmap="Reds", alpha=0.35, origin="upper")
        colors = {
            "nominal_ppo": "tab:blue",
            "env_ft_ppo": "tab:green",
            "oracle_best_of_candidates": "tab:red",
        }
        for method, path in paths.items():
            if not path:
                continue
            cells = np.asarray(path)
            ax.plot(cells[:, 1], cells[:, 0], color=colors.get(method, "tab:purple"), linewidth=1.8, label=method)
        start = episode.costmap.start
        goal = episode.costmap.goal
        ax.scatter([start[1]], [start[0]], c="cyan", s=35, marker="o", label="start")
        ax.scatter([goal[1]], [goal[0]], c="yellow", s=45, marker="*", label="goal")
        ax.set_title(f"{example['eval_domain']} {example['attack_type']} ep{example['episode_id']}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(example_dir / f"path_example_{index:02d}.png", dpi=180)
        plt.close(fig)


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Oracle Gap Report", ""]
    oracle = summary[summary["method"] == "oracle_best_of_candidates"].copy()
    nominal = summary[summary["method"] == "nominal_ppo"].copy()
    for _, row in oracle.iterrows():
        gap_pct = float(row["oracle_gap_pct"])
        verdict = "small recoverable space" if gap_pct < 0.01 else "clear adaptive potential" if gap_pct > 0.05 else "moderate recoverable space"
        lines.append(
            f"- {row['eval_domain']} / {row['attack_type']}: oracle gap vs nominal = "
            f"{gap_pct:.2%} ({verdict}), exposure={row['mean_attacked_cell_exposure_ratio']:.3f}."
        )
    lines.extend(["", "## PPO distance to oracle", ""])
    for _, nrow in nominal.iterrows():
        matching = oracle[(oracle["eval_domain"] == nrow["eval_domain"]) & (oracle["attack_type"] == nrow["attack_type"])]
        if matching.empty:
            continue
        orow = matching.iloc[0]
        lines.append(
            f"- {nrow['eval_domain']} / {nrow['attack_type']}: nominal={nrow['mean_attacked_scalar_cost']:.4f}, "
            f"oracle={orow['mean_attacked_scalar_cost']:.4f}, gap={orow['oracle_gap_pct']:.2%}."
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "- oracle_gap_pct < 1% means the current setting has little recoverable space.",
            "- oracle_gap_pct > 5% means adaptive planner-parameter selection has meaningful headroom.",
            "- If PPO remains close to nominal while oracle is much better, the benchmark exposes a learning gap.",
        ]
    )
    (output_dir / "oracle_gap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.quick:
        args.num_random_candidates = min(args.num_random_candidates, 16)
        if args.num_episodes is None:
            args.num_episodes = 20

    num_episodes = int(cfg_value(args, config, "num-episodes", 300))
    seed = int(cfg_value(args, config, "seed", 222))
    map_size = int(cfg_value(args, config, "map-size", 48))
    scenario = str(cfg_value(args, config, "scenario", "lunar_rover"))
    map_sampling_mode = str(cfg_value(args, config, "map-sampling-mode", "map_seed_pool"))
    fixed_map_seed = int(cfg_value(args, config, "fixed-map-seed", 909))
    map_pool_size = int(cfg_value(args, config, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_lambda = float(config.get("max-uncertainty-lambda", config.get("max_uncertainty_lambda", 1.2)))

    env_attack_base = load_attack_config(args.environment_attack_config)
    policies = {
        "nominal_ppo": load_optional_policy(args.nominal_checkpoint),
        "env_ft_ppo": load_optional_policy(args.env_ft_checkpoint),
        "obs_ft_ppo": load_optional_policy(args.obs_ft_checkpoint),
    }

    domain_specs = []
    if args.domains in {"in_domain", "both"}:
        domain_specs.append((str(args.eval_domain or f"in_domain_seed{fixed_map_seed}"), fixed_map_seed))
    if args.domains in {"heldout", "both"}:
        domain_specs.append((f"heldout_seed{args.heldout_fixed_map_seed}", int(args.heldout_fixed_map_seed)))
    attack_types = [item.strip() for item in args.attack_types.split(",") if item.strip()]

    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    for eval_domain, map_seed in domain_specs:
        episodes = generate_episodes(num_episodes, seed, map_size, scenario, map_sampling_mode, map_seed, map_pool_size)
        for attack_type in attack_types:
            env_attack = attack_config_for_type(attack_type, env_attack_base)
            episode_method_rows: list[dict[str, Any]] = []
            for episode_id, base_episode in enumerate(tqdm(episodes, desc=f"{eval_domain}:{attack_type}", leave=False)):
                eval_episode = base_episode
                if attack_enabled(env_attack):
                    eval_episode = apply_environment_attack_to_episode(
                        base_episode,
                        env_attack,
                        np.random.default_rng(seed + 100_000 + episode_id),
                    )

                candidates = base_candidate_rows(eval_episode, map_size, policies)
                candidates.extend(structured_candidates(eval_episode, max_lambda, args.num_structured_candidates))
                randoms = random_candidates(rng, args.num_random_candidates, max_lambda)
                candidates.extend(randoms)

                evaluated = []
                for candidate in candidates:
                    clean = plan_with_weights(
                        base_episode,
                        candidate["weights"],
                        lambda_uncertainty=float(candidate["lambda_uncertainty"]),
                    )
                    result = plan_with_weights(
                        eval_episode,
                        candidate["weights"],
                        lambda_uncertainty=float(candidate["lambda_uncertainty"]),
                    )
                    cost = eval_cost(result, attack_type)
                    evaluated.append((candidate, clean, result, cost))

                nominal_candidates = [item for item in evaluated if item[0]["method"] == "nominal_ppo"]
                if nominal_candidates:
                    nominal_item = nominal_candidates[0]
                else:
                    nominal_item = next(item for item in evaluated if item[0]["method"] == "fixed_heuristic")
                nominal_cost = float(nominal_item[3])
                nominal_path = nominal_item[2].get("path")
                oracle_item = min(evaluated, key=lambda item: float(item[3]))
                oracle_cost = float(oracle_item[3])
                oracle_path = oracle_item[2].get("path")

                method_items = [item for item in evaluated if item[0]["method"] in {"fixed_heuristic", "nominal_ppo", "env_ft_ppo", "obs_ft_ppo"}]
                random_items = [item for item in evaluated if item[0]["method"] == "random_weight_candidate"]
                if random_items:
                    random_costs = np.asarray([item[3] for item in random_items], dtype=np.float64)
                    best_random = min(random_items, key=lambda item: float(item[3]))
                    mean_random = best_random
                    mean_random = (
                        {
                            "method": "random_weights_mean",
                            "candidate_id": "random_mean",
                            "weights": np.mean([item[0]["weights"] for item in random_items], axis=0),
                            "lambda_uncertainty": float(np.mean([item[0]["lambda_uncertainty"] for item in random_items])),
                        },
                        best_random[1],
                        best_random[2],
                        float(random_costs.mean()),
                    )
                    best_random = (
                        {**best_random[0], "method": "random_weights_best_of_N", "candidate_id": best_random[0]["candidate_id"]},
                        best_random[1],
                        best_random[2],
                        best_random[3],
                    )
                    method_items.extend([mean_random, best_random])
                method_items.append(
                    (
                        {**oracle_item[0], "method": "oracle_best_of_candidates", "candidate_id": oracle_item[0]["candidate_id"]},
                        oracle_item[1],
                        oracle_item[2],
                        oracle_item[3],
                    )
                )

                for candidate, clean, result, cost in method_items:
                    gap = nominal_cost - float(cost)
                    row = {
                        "eval_domain": eval_domain,
                        "map_pool_seed": map_seed,
                        "attack_type": attack_type,
                        "episode_id": episode_id,
                        "map_seed": map_seed,
                        "start": json.dumps(list(eval_episode.costmap.start)),
                        "goal": json.dumps(list(eval_episode.costmap.goal)),
                        "method": candidate["method"],
                        "candidate_id": candidate["candidate_id"],
                        "weights": json.dumps([float(v) for v in np.asarray(candidate["weights"]).reshape(-1)]),
                        "lambda_uncertainty": float(candidate["lambda_uncertainty"]),
                        "nominal_scalar_cost": float(clean.get("scalar_cost", np.nan)),
                        "attacked_scalar_cost": float(cost),
                        "relative_degradation": float((float(cost) - float(clean.get("scalar_cost", np.nan))) / (abs(float(clean.get("scalar_cost", np.nan))) + 1e-8)),
                        "oracle_gap_vs_nominal": float(gap),
                        "oracle_gap_pct": float(gap / (abs(nominal_cost) + 1e-8)),
                        "path_length": float(result.get("path_length", np.nan)),
                        "success": 1.0 if bool(result.get("success", False)) else 0.0,
                        "attacked_cell_exposure_ratio": float(result.get("attacked_cell_exposure_ratio", 0.0)),
                        "path_overlap_vs_nominal": path_overlap_ratio(result.get("path"), nominal_path),
                        "path_overlap_vs_oracle": path_overlap_ratio(result.get("path"), oracle_path),
                        "attacked_corridor_cells": int(result.get("attacked_corridor_cells", 0)),
                    }
                    episode_method_rows.append(row)
                    detail_rows.append(row)

                if attack_type == "path_corridor" and len(examples) < 6:
                    paths = {}
                    for candidate, _, result, _ in method_items:
                        if candidate["method"] in {"nominal_ppo", "env_ft_ppo", "oracle_best_of_candidates"}:
                            paths[candidate["method"]] = result.get("path")
                    examples.append(
                        {
                            "episode": eval_episode,
                            "paths": paths,
                            "eval_domain": eval_domain,
                            "attack_type": attack_type,
                            "episode_id": episode_id,
                        }
                    )

            summary_rows.extend(
                method_summary_rows(episode_method_rows, eval_domain, attack_type, map_seed, num_episodes)
            )

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)
    summary_path = output_dir / "oracle_gap_summary.csv"
    details_path = output_dir / "oracle_gap_episode_details.csv"
    summary.to_csv(summary_path, index=False)
    details.to_csv(details_path, index=False)
    plot_oracle_outputs(summary, details, output_dir)
    plot_path_examples(examples, output_dir)
    write_report(summary, output_dir)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Saved oracle gap summary to {summary_path}")
    print(f"Saved oracle gap episode details to {details_path}")
    print(f"Saved oracle gap report to {output_dir / 'oracle_gap_report.md'}")


if __name__ == "__main__":
    main()
