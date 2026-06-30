"""Run a real DEM + VIPER planning benchmark.

This is intentionally an evaluation/planning script, not a PPO trainer. The
synthetic Gym environment still owns the RL training path; this script loads the
real DEM tile exported by ``extract_real_dem_tile.py`` and checks whether the
same weighted-A* objective machinery produces meaningful route differences on a
VIPER-like lunar map.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from envs.attack_wrappers import apply_environment_attack_to_episode
from maps.map_generator import GeneratedCostMap
from utils.cleanrl_policy import load_cleanrl_agent
from utils.metrics import (
    DEFAULT_MAX_UNCERTAINTY_LAMBDA,
    OBJECTIVE_NAMES,
    PlanningEpisode,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    compute_observation,
    normalize_weights,
    path_overlap_ratio,
    plan_with_weights,
)


DEFAULT_LAYERS = Path("maps") / "real_dem_tiles" / "viper_200m_tile" / "real_map_layers.npz"
DEFAULT_METADATA = Path("maps") / "real_dem_tiles" / "viper_200m_tile" / "tile_metadata.json"
DEFAULT_ROVER_PROFILE = Path("configs") / "rovers" / "viper.json"
DEFAULT_OUTPUT_DIR = Path("runs") / "real_viper_scenario"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate weighted-A* methods on a real VIPER DEM tile.")
    parser.add_argument("--layers", type=Path, default=DEFAULT_LAYERS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--rover-profile", type=Path, default=DEFAULT_ROVER_PROFILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--num-random-candidates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-distance-ratio", type=float, default=0.62)
    parser.add_argument("--attack-strength", type=float, default=3.0)
    parser.add_argument("--corridor-radius", type=int, default=2)
    parser.add_argument("--max-uncertainty-lambda", type=float, default=DEFAULT_MAX_UNCERTAINTY_LAMBDA)
    parser.add_argument(
        "--ppo-checkpoint",
        action="append",
        default=[],
        help=(
            "Evaluate a PPO checkpoint as a named method and include it in the oracle candidates. "
            "Use NAME=PATH or just PATH. Can be repeated."
        ),
    )
    parser.add_argument(
        "--ppo-candidate-dir",
        type=Path,
        default=None,
        help="Directory of PPO checkpoints to include as oracle-only candidates.",
    )
    parser.add_argument(
        "--ppo-candidate-glob",
        type=str,
        default="*.pt",
        help="Glob used with --ppo-candidate-dir.",
    )
    parser.add_argument("--observation-mode", type=str, default="terrain")
    parser.add_argument("--action-mode", type=str, default=None)
    parser.add_argument("--action-gain", type=float, default=None)
    parser.add_argument(
        "--ppo-max-uncertainty-lambda",
        type=float,
        default=None,
        help="Override max uncertainty lambda for PPO checkpoints. Defaults to each checkpoint config.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        help="Print progress every N episodes. Set to 0 to disable.",
    )
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test.")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def checkpoint_config_value(
    checkpoint: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        return default
    return config.get(key, config.get(key.replace("-", "_"), default))


def parse_checkpoint_spec(spec: str) -> tuple[str | None, Path]:
    if "=" in spec:
        name, raw_path = spec.split("=", 1)
        return name.strip() or None, Path(raw_path.strip())
    return None, Path(spec)


def unique_policy_name(name: str, used: set[str]) -> str:
    base = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in name).strip("_")
    if not base:
        base = "ppo_checkpoint"
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate


def load_ppo_policies(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[tuple[str | None, Path, bool]] = []
    for item in args.ppo_checkpoint:
        name, path = parse_checkpoint_spec(str(item))
        specs.append((name, path, True))

    if args.ppo_candidate_dir is not None:
        candidate_dir = Path(args.ppo_candidate_dir)
        if not candidate_dir.exists():
            raise FileNotFoundError(f"PPO candidate directory does not exist: {candidate_dir}")
        for path in sorted(candidate_dir.glob(str(args.ppo_candidate_glob))):
            specs.append((None, path, False))

    if not specs:
        return []

    device = resolve_device(str(args.device))
    policies: list[dict[str, Any]] = []
    used_names: set[str] = set()
    seen_paths: set[Path] = set()
    for raw_name, raw_path, as_method in specs:
        path = Path(raw_path)
        resolved_path = path.resolve()
        if resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        if not path.exists():
            raise FileNotFoundError(f"PPO checkpoint not found: {path}")
        agent, checkpoint = load_cleanrl_agent(path, device=device)
        method_name = unique_policy_name(raw_name or f"ppo_{path.stem}", used_names)
        action_mode = str(
            args.action_mode
            if args.action_mode is not None
            else checkpoint_config_value(checkpoint, "action_mode", "preference_delta")
        )
        action_gain = float(
            args.action_gain
            if args.action_gain is not None
            else checkpoint_config_value(checkpoint, "action_gain", 3.0)
        )
        max_uncertainty_lambda = float(
            args.ppo_max_uncertainty_lambda
            if args.ppo_max_uncertainty_lambda is not None
            else checkpoint_config_value(checkpoint, "max_uncertainty_lambda", DEFAULT_MAX_UNCERTAINTY_LAMBDA)
        )
        policies.append(
            {
                "name": method_name,
                "path": path,
                "agent": agent,
                "checkpoint": checkpoint,
                "device": device,
                "as_method": bool(as_method),
                "action_mode": action_mode,
                "action_gain": action_gain,
                "max_uncertainty_lambda": max_uncertainty_lambda,
            }
        )
    return policies


def ppo_policy_candidate(
    policy: dict[str, Any],
    episode: PlanningEpisode,
    observation_mode: str,
) -> dict[str, Any]:
    map_size = int(episode.costmap.layers["distance"].shape[0])
    obs = compute_observation(
        episode,
        map_size=map_size,
        observation_mode=str(observation_mode),
        max_uncertainty_lambda=float(policy["max_uncertainty_lambda"]),
    )
    agent = policy["agent"]
    if int(obs.shape[0]) != int(agent.obs_dim):
        raise ValueError(
            f"PPO checkpoint {policy['path']} obs_dim={agent.obs_dim}, "
            f"but real observation has shape {obs.shape}. Use the same observation mode as training."
        )
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=policy["device"]).unsqueeze(0)
    with torch.no_grad():
        action = agent.get_deterministic_action(obs_tensor).squeeze(0).cpu().numpy()
    weights = action_to_planning_weights(
        episode,
        action,
        action_mode=str(policy["action_mode"]),
        action_gain=float(policy["action_gain"]),
    )
    lambda_uncertainty = action_to_uncertainty_lambda(
        action,
        max_uncertainty_lambda=float(policy["max_uncertainty_lambda"]),
    )
    return {
        "weights": weights,
        "lambda_uncertainty": float(lambda_uncertainty),
    }


def load_layers(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Real map layers not found: {path}. Run extract_real_dem_tile.py with the VIPER profile first."
        )
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def layer_dicts(raw_layers: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    layers = {
        name: np.asarray(raw_layers[f"layer_{name}"], dtype=np.float32)
        for name in OBJECTIVE_NAMES
    }
    uncertainty_layers = {
        name: np.asarray(raw_layers[f"uncertainty_{name}"], dtype=np.float32)
        for name in OBJECTIVE_NAMES
    }
    return layers, uncertainty_layers


def nearest_free_cell(mask: np.ndarray, target: tuple[int, int]) -> tuple[int, int]:
    free = np.argwhere(~mask)
    if len(free) == 0:
        raise RuntimeError("real DEM tile has no free cells")
    target_arr = np.asarray(target, dtype=np.float32)
    distances = np.sum((free.astype(np.float32) - target_arr.reshape(1, 2)) ** 2, axis=1)
    row, col = free[int(np.argmin(distances))]
    return int(row), int(col)


def sample_start_goal(
    rng: np.random.Generator,
    obstacle_mask: np.ndarray,
    min_distance_ratio: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    free = np.argwhere(~obstacle_mask)
    if len(free) < 2:
        raise RuntimeError("not enough free cells in real DEM tile")
    map_size = int(obstacle_mask.shape[0])
    min_distance = float(min_distance_ratio) * float(map_size)
    for _ in range(4000):
        i, j = rng.choice(len(free), size=2, replace=False)
        start = free[int(i)]
        goal = free[int(j)]
        if float(np.linalg.norm(start - goal)) >= min_distance:
            return tuple(map(int, start)), tuple(map(int, goal))
    return tuple(map(int, free[0])), tuple(map(int, free[-1]))


def build_costmap(
    raw_layers: dict[str, np.ndarray],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> GeneratedCostMap:
    layers, uncertainty_layers = layer_dicts(raw_layers)
    obstacle_mask = np.asarray(raw_layers["obstacle_mask"], dtype=bool)
    height_map = np.asarray(raw_layers["height_norm"], dtype=np.float32)
    slope_layer = np.asarray(raw_layers["slope_layer"], dtype=np.float32)
    roughness_layer = np.asarray(raw_layers["roughness_layer"], dtype=np.float32)
    return GeneratedCostMap(
        layers=layers,
        uncertainty_layers=uncertainty_layers,
        obstacle_mask=obstacle_mask,
        start=start,
        goal=goal,
        height_map=height_map,
        slope_layer=slope_layer,
        roughness_layer=roughness_layer,
        communication_quality=(1.0 - layers["communication"]).astype(np.float32),
        illumination_quality=(1.0 - layers["illumination"]).astype(np.float32),
        beacons=np.empty((0, 2), dtype=np.int32),
        sun_direction=np.array([1.0, 1.0, 0.2], dtype=np.float32),
        scenario="real_viper_dem",
    )


def viper_rover_state() -> dict[str, float]:
    return {
        "battery_budget": 0.62,
        "hazard_tolerance": 0.58,
        "min_communication_quality": 0.28,
        "illumination_requirement": 0.30,
    }


def mission_priority() -> np.ndarray:
    return normalize_weights(np.array([0.06, 0.29, 0.43, 0.08, 0.14], dtype=np.float32))


def deterministic_methods(max_uncertainty_lambda: float) -> dict[str, dict[str, Any]]:
    methods = {
        "heuristic": {
            "weights": mission_priority(),
            "lambda_uncertainty": 0.0,
        },
        "viper_safe": {
            "weights": normalize_weights(np.array([0.05, 0.22, 0.52, 0.08, 0.13], dtype=np.float32)),
            "lambda_uncertainty": 0.35 * max_uncertainty_lambda,
        },
        "viper_energy": {
            "weights": normalize_weights(np.array([0.05, 0.54, 0.25, 0.06, 0.10], dtype=np.float32)),
            "lambda_uncertainty": 0.20 * max_uncertainty_lambda,
        },
        "viper_uncertainty_high": {
            "weights": mission_priority(),
            "lambda_uncertainty": 0.80 * max_uncertainty_lambda,
        },
        "balanced": {
            "weights": normalize_weights(np.ones(len(OBJECTIVE_NAMES), dtype=np.float32)),
            "lambda_uncertainty": 0.25 * max_uncertainty_lambda,
        },
    }
    return methods


def random_candidate_configs(
    rng: np.random.Generator,
    count: int,
    max_uncertainty_lambda: float,
) -> dict[str, dict[str, Any]]:
    candidates = {}
    for index in range(max(int(count), 0)):
        candidates[f"random_{index:03d}"] = {
            "weights": normalize_weights(rng.dirichlet(np.ones(len(OBJECTIVE_NAMES))).astype(np.float32)),
            "lambda_uncertainty": float(rng.uniform(0.0, max_uncertainty_lambda)),
        }
    return candidates


def attack_config(attack_strength: float, corridor_radius: int) -> dict[str, Any]:
    return {
        "enabled": True,
        "type": "env_path_corridor_attack",
        "reference_policy": "heuristic",
        "corridor_radius": int(corridor_radius),
        "attack_strength": float(attack_strength),
        "affected_layers": ["hazard", "energy"],
        "apply_during_training": False,
        "reward_uses_attacked_cost": True,
    }


def result_row(
    episode_id: int,
    attack_type: str,
    method: str,
    result: dict[str, Any],
    baseline_path: list[tuple[int, int]] | None,
    oracle_path: list[tuple[int, int]] | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    weights = np.asarray(result.get("weights", np.full(len(OBJECTIVE_NAMES), np.nan)), dtype=np.float32)
    row = {
        "episode_id": int(episode_id),
        "attack_type": attack_type,
        "method": method,
        "oracle_source": str(result.get("oracle_source", "")),
        "success": bool(result.get("success", False)),
        "scalar_cost": float(result.get("scalar_cost", np.nan)),
        "belief_scalar_cost": float(result.get("belief_scalar_cost", result.get("scalar_cost", np.nan))),
        "map_mismatch_penalty": float(result.get("map_mismatch_penalty", 0.0)),
        "map_mismatch_abs_error": float(result.get("map_mismatch_abs_error", 0.0)),
        "mean_path_confidence": float(result.get("mean_path_confidence", np.nan)),
        # For map-reliability attacks, scalar_cost is the true-map cost after
        # planning on the belief map. Keep the soft attacker metric separately
        # so it cannot be mistaken for the belief-vs-true mismatch effect.
        "attacked_scalar_cost": float(result.get("scalar_cost", np.nan)),
        "soft_attacked_scalar_cost": float(result.get("attacked_scalar_cost", np.nan)),
        "relative_degradation": np.nan,
        "path_length": int(result.get("path_length", 0)),
        "attacked_cell_exposure_ratio": float(result.get("attacked_cell_exposure_ratio", 0.0)),
        "mismatched_cells": int(result.get("attack_metadata", {}).get("mismatched_cells", 0)),
        "hazard_exposure": float(result.get("hazard_exposure", np.nan)),
        "belief_hazard_exposure": float(result.get("belief_hazard_exposure", np.nan)),
        "uncertainty_exposure": float(result.get("uncertainty_exposure", np.nan)),
        "path_overlap_vs_heuristic": path_overlap_ratio(result.get("path"), baseline_path),
        "path_overlap_vs_oracle": path_overlap_ratio(result.get("path"), oracle_path),
        "lambda_uncertainty": float(result.get("lambda_uncertainty", np.nan)),
        "map_row": metadata.get("row", np.nan),
        "map_col": metadata.get("col", np.nan),
    }
    for index, name in enumerate(OBJECTIVE_NAMES):
        row[f"weight_{name}"] = float(weights[index]) if index < len(weights) else np.nan
        objectives = result.get("objectives", {})
        attacked_objectives = result.get("attacked_objectives", {})
        row[f"{name}_cost"] = float(objectives.get(name, np.nan)) if isinstance(objectives, dict) else np.nan
        row[f"attacked_{name}_cost"] = (
            float(attacked_objectives.get(name, np.nan)) if isinstance(attacked_objectives, dict) else np.nan
        )
    return row


def summarize(details: pd.DataFrame) -> pd.DataFrame:
    rows = []
    none_baseline: dict[str, float] = {}
    if not details.empty:
        none_rows = details[details["attack_type"] == "none"]
        for method, group in none_rows.groupby("method", dropna=False):
            none_baseline[str(method)] = float(group["attacked_scalar_cost"].astype(float).mean())

    grouped = details.groupby(["attack_type", "method"], dropna=False)
    for (attack_type, method), group in grouped:
        scalar = group["scalar_cost"].astype(float)
        belief = group["belief_scalar_cost"].astype(float) if "belief_scalar_cost" in group else scalar
        attacked = group["attacked_scalar_cost"].astype(float)
        soft_attacked = group["soft_attacked_scalar_cost"].astype(float)
        baseline = none_baseline.get(str(method), float(scalar.mean()))
        relative_degradation = (
            0.0
            if str(attack_type) == "none"
            else float((attacked.mean() - baseline) / (abs(baseline) + 1e-8))
        )
        rows.append(
            {
                "attack_type": attack_type,
                "method": method,
                "num_episodes": int(len(group)),
                "success_rate": float(group["success"].astype(float).mean()),
                "mean_scalar_cost": float(scalar.mean()),
                "mean_belief_scalar_cost": float(belief.mean()),
                "mean_map_mismatch_penalty": (
                    float(group["map_mismatch_penalty"].astype(float).mean())
                    if "map_mismatch_penalty" in group
                    else 0.0
                ),
                "mean_attacked_scalar_cost": float(attacked.mean()),
                "mean_soft_attacked_scalar_cost": float(soft_attacked.mean()),
                "relative_degradation": relative_degradation,
                "mean_path_length": float(group["path_length"].mean()),
                "mean_attacked_cell_exposure_ratio": float(group["attacked_cell_exposure_ratio"].mean()),
                "mean_mismatched_cells": (
                    float(group["mismatched_cells"].astype(float).mean())
                    if "mismatched_cells" in group
                    else 0.0
                ),
                "mean_path_confidence": (
                    float(group["mean_path_confidence"].astype(float).mean(skipna=True))
                    if "mean_path_confidence" in group
                    else float("nan")
                ),
                "mean_path_overlap_vs_heuristic": float(group["path_overlap_vs_heuristic"].mean(skipna=True)),
                "mean_path_overlap_vs_oracle": float(group["path_overlap_vs_oracle"].mean(skipna=True)),
                "mean_lambda_uncertainty": float(group["lambda_uncertainty"].mean(skipna=True)),
                "mean_hazard_exposure": float(group["hazard_exposure"].mean(skipna=True)),
                "mean_belief_hazard_exposure": (
                    float(group["belief_hazard_exposure"].mean(skipna=True))
                    if "belief_hazard_exposure" in group
                    else float("nan")
                ),
                "mean_uncertainty_exposure": float(group["uncertainty_exposure"].mean(skipna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["attack_type", "mean_attacked_scalar_cost"])


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for attack_type, group in summary.groupby("attack_type"):
        group = group.sort_values("mean_attacked_scalar_cost")
        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        ax.bar(group["method"], group["mean_attacked_scalar_cost"], color="#5577aa")
        ax.set_title(f"Real VIPER DEM attacked cost ({attack_type})")
        ax.set_ylabel("mean attacked scalar cost")
        ax.tick_params(axis="x", rotation=30)
        fig.savefig(figures_dir / f"fig_mean_attacked_cost_{attack_type}.png", dpi=180)
        plt.close(fig)

    attack_group = summary[summary["attack_type"] == "path_corridor"].sort_values(
        "mean_attacked_cell_exposure_ratio"
    )
    if not attack_group.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        ax.bar(attack_group["method"], attack_group["mean_attacked_cell_exposure_ratio"], color="#aa6655")
        ax.set_title("Real VIPER DEM exposure to attacked corridor")
        ax.set_ylabel("mean attacked cell exposure ratio")
        ax.tick_params(axis="x", rotation=30)
        fig.savefig(figures_dir / "fig_attacked_cell_exposure.png", dpi=180)
        plt.close(fig)


def plot_path_example(
    raw_layers: dict[str, np.ndarray],
    attacked_episode: PlanningEpisode,
    example_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    background = np.asarray(raw_layers.get("height_map", raw_layers["height_norm"]), dtype=np.float32)
    attack_mask = getattr(attacked_episode.costmap, "attack_mask", None)
    meters = float(read_json(DEFAULT_METADATA).get("tile_meters", background.shape[0]))
    extent = [0.0, meters, meters, 0.0]

    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    ax.imshow(background, cmap="terrain", extent=extent)
    if attack_mask is not None:
        masked = np.ma.masked_where(~np.asarray(attack_mask, dtype=bool), attack_mask)
        ax.imshow(masked, cmap="Reds", alpha=0.35, extent=extent)

    colors = {
        "heuristic": "#111111",
        "viper_safe": "#277da1",
        "oracle_best": "#43aa8b",
    }
    for method, color in colors.items():
        result = example_results.get(method)
        if not result or not result.get("path"):
            continue
        path = np.asarray(result["path"], dtype=np.float32)
        scale = meters / float(background.shape[0])
        ax.plot((path[:, 1] + 0.5) * scale, (path[:, 0] + 0.5) * scale, color=color, linewidth=2.2, label=method)

    start = attacked_episode.costmap.start
    goal = attacked_episode.costmap.goal
    scale = meters / float(background.shape[0])
    ax.scatter([(start[1] + 0.5) * scale], [(start[0] + 0.5) * scale], marker="o", s=60, c="lime", edgecolors="black", label="start")
    ax.scatter([(goal[1] + 0.5) * scale], [(goal[0] + 0.5) * scale], marker="*", s=110, c="yellow", edgecolors="black", label="goal")
    ax.set_title("Real VIPER DEM path-corridor attack example")
    ax.set_xlabel("meters")
    ax.set_ylabel("meters")
    ax.legend(loc="best")
    fig.savefig(figures_dir / "fig_path_example_overlay.png", dpi=180)
    plt.close(fig)


def write_report(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    output_dir: Path,
    args: argparse.Namespace,
    metadata: dict[str, Any],
) -> None:
    attack_rows = summary[summary["attack_type"] == "path_corridor"].copy()
    oracle = attack_rows[attack_rows["method"] == "oracle_best"]
    heuristic = attack_rows[attack_rows["method"] == "heuristic"]
    safe = attack_rows[attack_rows["method"] == "viper_safe"]
    lines = [
        "# Real VIPER Scenario Report",
        "",
        f"- Episodes: `{args.num_episodes}`",
        f"- DEM tile origin: row `{metadata.get('row')}`, col `{metadata.get('col')}`",
        f"- Attack: `env_path_corridor_attack`, strength `{args.attack_strength}`, radius `{args.corridor_radius}`",
        "",
    ]
    if not heuristic.empty and not oracle.empty:
        h_cost = float(heuristic.iloc[0]["mean_attacked_scalar_cost"])
        o_cost = float(oracle.iloc[0]["mean_attacked_scalar_cost"])
        gap = (h_cost - o_cost) / (abs(h_cost) + 1e-8)
        lines.extend(
            [
                "## Oracle Gap",
                "",
                f"- Heuristic attacked cost: `{h_cost:.4f}`",
                f"- Oracle attacked cost: `{o_cost:.4f}`",
                f"- Oracle gap vs heuristic: `{100.0 * gap:.2f}%`",
                "",
            ]
        )
    oracle_rows = details[
        (details["attack_type"] == "path_corridor")
        & (details["method"] == "oracle_best")
        & (details["oracle_source"].astype(str) != "")
    ].copy()
    if not oracle_rows.empty:
        source_counts = oracle_rows["oracle_source"].astype(str).value_counts().head(8)
        lines.extend(["## Oracle Source", ""])
        for source, count in source_counts.items():
            lines.append(f"- `{source}`: `{int(count)}` episodes")
        lines.append("")
    if not heuristic.empty and not safe.empty:
        h_exp = float(heuristic.iloc[0]["mean_attacked_cell_exposure_ratio"])
        s_exp = float(safe.iloc[0]["mean_attacked_cell_exposure_ratio"])
        lines.extend(
            [
                "## Exposure",
                "",
                f"- Heuristic attacked-cell exposure: `{h_exp:.4f}`",
                f"- VIPER-safe attacked-cell exposure: `{s_exp:.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Outputs",
            "",
            "- `real_viper_summary.csv`",
            "- `real_viper_episode_details.csv`",
            "- `figures/fig_path_example_overlay.png`",
            "- `figures/fig_attacked_cell_exposure.png`",
            "",
        ]
    )
    (output_dir / "real_viper_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_output_guide(output_dir: Path) -> None:
    guide = """# Real VIPER Scenario Outputs

## Main CSV

- `real_viper_summary.csv`: one row per `attack_type x method`.
- `real_viper_episode_details.csv`: one row per sampled start-goal episode, attack type, and method.
- `mean_attacked_scalar_cost`: main scalar objective for the evaluated map condition. For `path_corridor`, this is the scalar cost on the attacked costmap. Lower is better.
- `mean_soft_attacked_scalar_cost`: optional soft uncertainty-attacker metric from the planner utilities; it is not the primary path-corridor attack metric.
- `relative_degradation`: attacked cost increase relative to the same method's no-attack mean cost.
- `oracle_source`: candidate that won when `method == oracle_best`.
- `mean_attacked_cell_exposure_ratio`: fraction of path cells inside the attacked corridor. Lower usually means better avoidance.
- `mean_path_overlap_vs_heuristic`: route similarity to the heuristic path. Lower means the method changed route more.
- `mean_path_overlap_vs_oracle`: route similarity to the candidate oracle path. Higher means closer to the oracle route.

## Figures

Figures are stored under `figures/`.

- `fig_mean_attacked_cost_none.png`: baseline no-attack cost comparison.
- `fig_mean_attacked_cost_path_corridor.png`: attacked cost comparison under path-corridor attack.
- `fig_attacked_cell_exposure.png`: exposure to attacked cells under path-corridor attack.
- `fig_path_example_overlay.png`: one visual path example over DEM and attack mask.

## How to Read

The useful pattern is:

1. `oracle_best` lower than `heuristic`: real DEM tile has recoverable route-choice headroom.
2. `viper_safe` lower than `heuristic`: hand-designed VIPER-aware weighting is already beneficial.
3. lower exposure and lower path overlap with heuristic: the method is actually rerouting, not just changing cost labels.
"""
    (output_dir / "OUTPUT_GUIDE.md").write_text(guide, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.quick:
        args.num_episodes = 20
        args.num_random_candidates = min(int(args.num_random_candidates), 24)
    if args.clean_output and args.output_dir.exists():
        resolved = args.output_dir.resolve()
        cwd = Path.cwd().resolve()
        if cwd not in [resolved, *resolved.parents]:
            shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_layers = load_layers(args.layers)
    metadata = read_json(args.metadata)
    rover_profile = read_json(args.rover_profile)
    rng = np.random.default_rng(int(args.seed))
    obstacle_mask = np.asarray(raw_layers["obstacle_mask"], dtype=bool)
    methods = deterministic_methods(float(args.max_uncertainty_lambda))
    ppo_policies = load_ppo_policies(args)
    rows: list[dict[str, Any]] = []
    example_results: dict[str, dict[str, Any]] | None = None
    example_attacked_episode: PlanningEpisode | None = None
    start_time = time.time()

    print("Starting real VIPER scenario benchmark.", flush=True)
    print(f"  layers: {args.layers}", flush=True)
    print(f"  output_dir: {args.output_dir}", flush=True)
    print(f"  num_episodes: {args.num_episodes}", flush=True)
    print(f"  num_random_candidates: {args.num_random_candidates}", flush=True)
    print(f"  attack_strength: {args.attack_strength}", flush=True)
    print(f"  corridor_radius: {args.corridor_radius}", flush=True)
    print(f"  rover_profile: {rover_profile.get('name', args.rover_profile)}", flush=True)
    if ppo_policies:
        direct_count = sum(1 for policy in ppo_policies if policy["as_method"])
        print(
            f"  ppo_policies: {len(ppo_policies)} total "
            f"({direct_count} direct methods, {len(ppo_policies) - direct_count} oracle-only)",
            flush=True,
        )

    for episode_id in range(int(args.num_episodes)):
        start, goal = sample_start_goal(rng, obstacle_mask, float(args.min_distance_ratio))
        start = nearest_free_cell(obstacle_mask, start)
        goal = nearest_free_cell(obstacle_mask, goal)
        costmap = build_costmap(raw_layers, start=start, goal=goal)
        episode = PlanningEpisode(
            costmap=costmap,
            mission_priority=mission_priority(),
            rover_state=viper_rover_state(),
            scenario="real_viper_dem",
            mission_regime="viper_hazard_energy",
            mission_severity=0.75,
        )
        attacked_episode = apply_environment_attack_to_episode(
            episode,
            attack_config(float(args.attack_strength), int(args.corridor_radius)),
            rng=rng,
        )

        for attack_type, eval_episode in (("none", episode), ("path_corridor", attacked_episode)):
            base_results: dict[str, dict[str, Any]] = {}
            for method_name, method_config in methods.items():
                base_results[method_name] = plan_with_weights(
                    eval_episode,
                    method_config["weights"],
                    lambda_uncertainty=float(method_config["lambda_uncertainty"]),
                )
            ppo_candidate_configs: dict[str, dict[str, Any]] = {}
            for policy in ppo_policies:
                candidate_name = f"ppo_{policy['name']}"
                candidate_config = ppo_policy_candidate(policy, eval_episode, str(args.observation_mode))
                ppo_candidate_configs[candidate_name] = candidate_config
                if bool(policy["as_method"]):
                    base_results[candidate_name] = plan_with_weights(
                        eval_episode,
                        candidate_config["weights"],
                        lambda_uncertainty=float(candidate_config["lambda_uncertainty"]),
                    )
            oracle_candidates = {
                **methods,
                **ppo_candidate_configs,
                **random_candidate_configs(rng, int(args.num_random_candidates), float(args.max_uncertainty_lambda)),
            }
            oracle_name = ""
            oracle_result: dict[str, Any] | None = None
            for candidate_name, candidate_config in oracle_candidates.items():
                result = plan_with_weights(
                    eval_episode,
                    candidate_config["weights"],
                    lambda_uncertainty=float(candidate_config["lambda_uncertainty"]),
                )
                if oracle_result is None or float(result["attacked_scalar_cost"]) < float(
                    oracle_result["attacked_scalar_cost"]
                ):
                    oracle_name = candidate_name
                    oracle_result = result
            if oracle_result is None:
                raise RuntimeError("oracle candidate set is empty")
            oracle_result = dict(oracle_result)
            oracle_result["oracle_source"] = oracle_name
            base_results["oracle_best"] = oracle_result
            heuristic_path = base_results["heuristic"].get("path")
            oracle_path = oracle_result.get("path")
            for method_name, result in base_results.items():
                rows.append(
                    result_row(
                        episode_id=episode_id,
                        attack_type=attack_type,
                        method=method_name,
                        result=result,
                        baseline_path=heuristic_path,
                        oracle_path=oracle_path,
                        metadata=metadata,
                    )
                )
            if episode_id == 0 and attack_type == "path_corridor":
                example_results = base_results
                example_attacked_episode = attacked_episode

        completed = episode_id + 1
        interval = int(args.progress_interval)
        if interval > 0 and (completed == 1 or completed % interval == 0 or completed == int(args.num_episodes)):
            elapsed = max(time.time() - start_time, 1e-8)
            rate = completed / elapsed
            remaining = (int(args.num_episodes) - completed) / max(rate, 1e-8)
            print(
                f"progress {completed}/{args.num_episodes} episodes "
                f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                flush=True,
            )

    details = pd.DataFrame(rows)
    summary = summarize(details)
    details_path = args.output_dir / "real_viper_episode_details.csv"
    summary_path = args.output_dir / "real_viper_summary.csv"
    details.to_csv(details_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, args.output_dir)
    if example_results is not None and example_attacked_episode is not None:
        plot_path_example(raw_layers, example_attacked_episode, example_results, args.output_dir)
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "layers": str(args.layers),
                "metadata": str(args.metadata),
                "rover_profile": str(args.rover_profile),
                "rover_name": rover_profile.get("name", "unknown"),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    write_report(summary, details, args.output_dir, args, metadata)
    write_output_guide(args.output_dir)

    print(f"Real VIPER scenario complete.")
    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")
    print(f"Figures: {args.output_dir / 'figures'}")
    print(f"Report: {args.output_dir / 'real_viper_report.md'}")
    print(f"Output guide: {args.output_dir / 'OUTPUT_GUIDE.md'}")


if __name__ == "__main__":
    main()
