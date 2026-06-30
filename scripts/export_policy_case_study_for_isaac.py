#!/usr/bin/env python
"""Export trained-policy path-planning case-study assets for Isaac visualization.

This script does not launch Isaac. It evaluates a trained CleanRL PPO/SAC
planner policy on one real-terrain task, exports the policy-selected route,
and writes heightfield/waypoint files that Isaac can replay with a robot.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.attack_wrappers import apply_environment_attack_to_episode  # noqa: E402
from maps.real_terrain import load_real_layers, load_task_split, make_real_planning_episode  # noqa: E402
from run_lunar_viper_staged_recovery import load_environment_attack, read_json  # noqa: E402
from utils.cleanrl_policy import load_cleanrl_agent, predict_cleanrl_action  # noqa: E402
from utils.metrics import (  # noqa: E402
    DEFAULT_ATTACK_BUDGET_FRACTION,
    DEFAULT_ATTACK_STRENGTH,
    DEFAULT_ATTACKER_RESPONSE,
    DEFAULT_ATTACKER_SHARPNESS,
    DEFAULT_ATTACKER_TEMPERATURE,
    DEFAULT_ATTACKER_TOP_FRACTION,
    DEFAULT_MAX_UNCERTAINTY_LAMBDA,
    OBJECTIVE_NAMES,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    compute_observation,
    plan_with_weights,
)


DEFAULT_LEVEL_CONFIG = PROJECT_ROOT / "configs" / "levels" / "ppo_difficulty" / "level2_medium.json"
DEFAULT_TASKS = (
    PROJECT_ROOT
    / "runs"
    / "rl_baselines"
    / "ppo"
    / "level2_medium_shock_recovery_5seeds"
    / "seed0"
    / "splits"
    / "validation_tasks.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "exports" / "isaac_policy_case_study"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained CleanRL PPO/SAC checkpoint.")
    parser.add_argument("--level-config", type=Path, default=DEFAULT_LEVEL_CONFIG)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--belief", choices=("clean", "corrupted"), default="corrupted")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--map-size-m", type=float, default=20.0)
    parser.add_argument("--height-scale-m", type=float, default=1.0)
    parser.add_argument("--waypoint-stride", type=int, default=2)
    parser.add_argument("--allow-diagonal", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def checkpoint_config_value(config: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in config:
            return config[key]
    return default


def as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): as_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def write_waypoints(path: Path, route_rc: list[tuple[int, int]], map_size_m: float, waypoint_stride: int) -> None:
    if not route_rc:
        raise ValueError("cannot write empty route")
    stride = max(int(waypoint_stride), 1)
    keep = list(range(0, len(route_rc), stride))
    if keep[-1] != len(route_rc) - 1:
        keep.append(len(route_rc) - 1)
    rows = [cell[0] for cell in route_rc]
    cols = [cell[1] for cell in route_rc]
    grid_size = max(max(rows), max(cols)) + 1
    scale = float(map_size_m) / float(max(grid_size - 1, 1))

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["index", "x_global_m", "y_global_m", "row_global", "col_global"],
        )
        writer.writeheader()
        for output_index, route_index in enumerate(keep):
            row, col = route_rc[route_index]
            writer.writerow(
                {
                    "index": output_index,
                    "x_global_m": f"{float(col) * scale:.6f}",
                    "y_global_m": f"{float(row) * scale:.6f}",
                    "row_global": int(row),
                    "col_global": int(col),
                }
            )


def write_preview(
    path: Path,
    terrain_cost: np.ndarray,
    obstacle_mask: np.ndarray,
    route_rc: list[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#222222")
    image = np.ma.array(np.asarray(terrain_cost, dtype=np.float32), mask=np.asarray(obstacle_mask, dtype=bool))
    ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
    if route_rc:
        route = np.asarray(route_rc, dtype=np.float32)
        ax.plot(route[:, 1], route[:, 0], color="#1f77ff", linewidth=2.4)
        ax.plot(route[:, 1], route[:, 0], color="white", linewidth=4.0, alpha=0.7, zorder=1)
        ax.plot(route[:, 1], route[:, 0], color="#1f77ff", linewidth=2.4, zorder=2)
    ax.scatter([start[1]], [start[0]], s=90, c="#2ca02c", edgecolors="white", linewidths=1.0, zorder=4)
    ax.scatter([goal[1]], [goal[0]], s=110, c="#d62728", edgecolors="white", linewidths=1.0, zorder=4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve(args.checkpoint)
    level_config_path = resolve(args.level_config)
    tasks_path = resolve(args.tasks)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    level_config = read_json(level_config_path)
    raw_layers = load_real_layers(resolve(Path(str(level_config["map_source"]))))
    tasks = load_task_split(tasks_path)
    if not tasks:
        raise ValueError(f"no tasks found in {tasks_path}")
    task_index = int(np.clip(args.task_index, 0, len(tasks) - 1))
    task = dict(tasks[task_index])

    rng = np.random.default_rng(int(args.seed) + int(task.get("seed", 0)) + task_index)
    clean_episode = make_real_planning_episode(
        raw_layers,
        task,
        rng,
        scenario=str(level_config.get("scenario", "real_lunar_viper")),
        mission_profile_scenario=str(level_config.get("mission_profile_scenario", "lunar_polar_shadow")),
    )

    episode = clean_episode
    env_attack: dict[str, Any] | None = None
    if args.belief == "corrupted":
        env_attack = load_environment_attack(level_config)
        episode = apply_environment_attack_to_episode(
            clean_episode,
            env_attack,
            np.random.default_rng(int(args.seed) + 20_000 + task_index),
        )

    agent, checkpoint = load_cleanrl_agent(checkpoint_path, device=args.device)
    checkpoint_config = dict(checkpoint.get("config", {}) or {})
    map_size = int(episode.costmap.layers["distance"].shape[0])
    observation_mode = str(
        checkpoint_config_value(
            checkpoint_config,
            "observation_mode",
            "observation-mode",
            default=level_config.get("observation_mode", "terrain"),
        )
    )
    action_mode = str(
        checkpoint_config_value(
            checkpoint_config,
            "action_mode",
            "action-mode",
            default=level_config.get("action_mode", "preference_delta"),
        )
    )
    action_gain = float(
        checkpoint_config_value(
            checkpoint_config,
            "action_gain",
            "action-gain",
            default=level_config.get("action_gain", 3.0),
        )
    )
    max_uncertainty_lambda = float(
        checkpoint_config_value(
            checkpoint_config,
            "max_uncertainty_lambda",
            "max-uncertainty-lambda",
            default=level_config.get("max_uncertainty_lambda", DEFAULT_MAX_UNCERTAINTY_LAMBDA),
        )
    )

    obs = compute_observation(
        episode,
        map_size,
        observation_mode=observation_mode,
        max_uncertainty_lambda=max_uncertainty_lambda,
    )
    if int(getattr(agent, "obs_dim", obs.shape[0])) != int(obs.shape[0]):
        raise ValueError(
            f"checkpoint obs_dim={getattr(agent, 'obs_dim', None)} but built observation has {obs.shape[0]} dims. "
            "Use the matching level config / task split for this checkpoint."
        )

    action = np.clip(
        predict_cleanrl_action(agent, obs, device=args.device, deterministic=True),
        0.0,
        1.0,
    ).astype(np.float32)
    weights = action_to_planning_weights(
        episode,
        action,
        action_mode=action_mode,
        action_gain=action_gain,
    )
    lambda_uncertainty = action_to_uncertainty_lambda(action, max_uncertainty_lambda=max_uncertainty_lambda)
    result = plan_with_weights(
        episode,
        weights,
        lambda_uncertainty=lambda_uncertainty,
        allow_diagonal=bool(args.allow_diagonal),
        attack_budget_fraction=float(
            checkpoint_config_value(
                checkpoint_config,
                "attack_budget_fraction",
                "attack-budget-fraction",
                default=DEFAULT_ATTACK_BUDGET_FRACTION,
            )
        ),
        attack_strength=float(
            checkpoint_config_value(
                checkpoint_config,
                "attack_strength",
                "attack-strength",
                default=DEFAULT_ATTACK_STRENGTH,
            )
        ),
        attacker_temperature=float(
            checkpoint_config_value(
                checkpoint_config,
                "attacker_temperature",
                "attacker-temperature",
                default=DEFAULT_ATTACKER_TEMPERATURE,
            )
        ),
        attacker_response=str(
            checkpoint_config_value(
                checkpoint_config,
                "attacker_response",
                "attacker-response",
                default=DEFAULT_ATTACKER_RESPONSE,
            )
        ),
        attacker_top_fraction=float(
            checkpoint_config_value(
                checkpoint_config,
                "attacker_top_fraction",
                "attacker-top-fraction",
                default=DEFAULT_ATTACKER_TOP_FRACTION,
            )
        ),
        attacker_sharpness=float(
            checkpoint_config_value(
                checkpoint_config,
                "attacker_sharpness",
                "attacker-sharpness",
                default=DEFAULT_ATTACKER_SHARPNESS,
            )
        ),
    )

    route_rc = [(int(row), int(col)) for row, col in (result.get("path") or [])]
    if not route_rc:
        raise RuntimeError("policy/planner did not produce a valid route")

    height_source = np.asarray(raw_layers.get("height_norm", raw_layers.get("height_map")), dtype=np.float32)
    heightfield = (height_source - float(np.nanmean(height_source))) * float(args.height_scale_m)
    obstacle_mask = np.asarray(episode.costmap.obstacle_mask, dtype=np.bool_)
    composite_cost = np.mean(
        np.stack([np.asarray(episode.costmap.layers[name], dtype=np.float32) for name in OBJECTIVE_NAMES], axis=0),
        axis=0,
    )

    np.save(output_dir / "terrain_heightfield.npy", heightfield.astype(np.float32))
    np.save(output_dir / "obstacle_mask.npy", obstacle_mask)
    np.save(output_dir / "composite_cost.npy", composite_cost.astype(np.float32))
    for name in OBJECTIVE_NAMES:
        np.save(output_dir / f"layer_{name}.npy", np.asarray(episode.costmap.layers[name], dtype=np.float32))

    waypoint_path = output_dir / "policy_waypoints.csv"
    write_waypoints(waypoint_path, route_rc, float(args.map_size_m), int(args.waypoint_stride))
    write_preview(
        output_dir / "policy_case_study_preview.png",
        composite_cost,
        obstacle_mask,
        route_rc,
        tuple(episode.costmap.start),
        tuple(episode.costmap.goal),
    )

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_policy_class": checkpoint.get("policy_class"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "level_config": str(level_config_path),
        "tasks": str(tasks_path),
        "task_index": task_index,
        "task": task,
        "belief": args.belief,
        "environment_attack": env_attack,
        "map_size_m": float(args.map_size_m),
        "height_scale_m": float(args.height_scale_m),
        "grid_shape": list(heightfield.shape),
        "start_rc": list(episode.costmap.start),
        "goal_rc": list(episode.costmap.goal),
        "observation_mode": observation_mode,
        "action_mode": action_mode,
        "action_gain": action_gain,
        "max_uncertainty_lambda": max_uncertainty_lambda,
        "action": action,
        "objective_names": list(OBJECTIVE_NAMES),
        "weights": weights,
        "lambda_uncertainty": lambda_uncertainty,
        "success": bool(result.get("success", False)),
        "path_length": result.get("path_length"),
        "scalar_cost": result.get("scalar_cost"),
        "belief_scalar_cost": result.get("belief_scalar_cost"),
        "attacked_scalar_cost": result.get("attacked_scalar_cost"),
        "soft_attacked_scalar_cost": result.get("soft_attacked_scalar_cost"),
        "map_mismatch_penalty": result.get("map_mismatch_penalty"),
        "attacked_cell_exposure": result.get("attacked_cell_exposure"),
        "attacked_cell_exposure_ratio": result.get("attacked_cell_exposure_ratio"),
        "hazard_exposure": result.get("hazard_exposure"),
        "uncertainty_exposure": result.get("uncertainty_exposure"),
        "waypoint_file": str(waypoint_path),
        "terrain_heightfield": str(output_dir / "terrain_heightfield.npy"),
        "obstacle_mask": str(output_dir / "obstacle_mask.npy"),
        "preview": str(output_dir / "policy_case_study_preview.png"),
        "isaac_note": (
            "In Isaac, import continuum_description/urdf/continuum.urdf, load terrain_heightfield.npy "
            "as the terrain, and track policy_waypoints.csv with the rover controller."
        ),
    }
    (output_dir / "metadata.json").write_text(json.dumps(as_jsonable(metadata), indent=2), encoding="utf-8")

    print(output_dir / "terrain_heightfield.npy")
    print(waypoint_path)
    print(output_dir / "metadata.json")
    print(output_dir / "policy_case_study_preview.png")


if __name__ == "__main__":
    main()
