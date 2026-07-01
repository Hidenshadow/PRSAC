"""Real-terrain helpers used by the paper shock-recovery protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from maps.real_terrain import generate_task_splits, load_real_layers, load_task_split, make_real_planning_episode
from utils.recovery_protocol import load_base_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    extends = value.pop("extends", None)
    if extends is None:
        return value

    parent_paths = [extends] if isinstance(extends, str) else list(extends)
    merged: dict[str, Any] = {}
    for parent in parent_paths:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = _deep_merge(merged, read_json(parent_path.resolve()))
    return _deep_merge(merged, value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_environment_attack(level_config: dict[str, Any]) -> dict[str, Any]:
    attack = dict(level_config.get("attacks", {}).get("environment", {}))
    attack.setdefault("enabled", True)
    if str(attack.get("type", "")) == "env_composite":
        return attack
    attack.setdefault("type", "env_belief_mismatch")
    attack.setdefault("mode", "risk_underestimate")
    attack.setdefault("selection_mode", "low_confidence_high_consequence")
    attack.setdefault("top_fraction", 0.25)
    attack.setdefault("attack_strength", 3.0)
    attack.setdefault("error_scale", 0.20)
    attack.setdefault("background_error_scale", 0.20)
    attack.setdefault("min_confidence", 0.15)
    attack.setdefault("max_confidence", 0.95)
    attack.setdefault("affected_layers", ["energy", "hazard", "communication", "illumination"])
    attack.setdefault("confidence_to_uncertainty", True)
    attack.setdefault("apply_during_training", True)
    attack.setdefault("reward_uses_attacked_cost", True)
    return attack


def prepare_splits(
    level_config: dict[str, Any],
    output_dir: Path,
    seed: int,
    quick: bool,
) -> dict[str, Path]:
    splits_dir = output_dir / "splits"
    train_count = int(level_config.get("num_train_tasks", 128))
    validation_count = int(level_config.get("num_validation_tasks", 64))
    heldout_count = int(level_config.get("num_heldout_tasks", 128))
    if quick:
        train_count = min(train_count, 32)
        validation_count = min(validation_count, 16)
        heldout_count = min(heldout_count, 32)

    generate_task_splits(
        layers_path=PROJECT_ROOT / str(level_config["map_source"]),
        output_dir=splits_dir,
        seed=seed,
        tile_id=str(level_config.get("tile_id", "real_terrain_tile")),
        num_train_tasks=train_count,
        num_validation_tasks=validation_count,
        num_heldout_tasks=heldout_count,
        min_distance_ratio=float(level_config.get("min_distance_ratio", 0.62)),
        metadata_path=PROJECT_ROOT / str(level_config.get("metadata", "")),
        task_sampling_mode=str(level_config.get("task_sampling_mode", "distance")),
        min_corridor_risk=(
            float(level_config["min_corridor_risk"])
            if level_config.get("min_corridor_risk") is not None
            else None
        ),
        corridor_radius=int(level_config.get("corridor_radius", 2)),
        candidate_pool_multiplier=int(level_config.get("candidate_pool_multiplier", 30)),
        risk_weights=level_config.get("corridor_risk_weights"),
    )
    return {
        "train": splits_dir / "train_tasks.json",
        "validation": splits_dir / "validation_tasks.json",
        "heldout": splits_dir / "heldout_tasks.json",
    }


def resolved_base_args(
    base_config: Path,
    level_config: dict[str, Any],
    splits: dict[str, Path],
    output_dir: Path,
    seed: int,
    nominal_timesteps: int,
) -> dict[str, Any]:
    args = load_base_args(str(base_config))
    args["env-kind"] = "real_terrain"
    args["layers-path"] = str(PROJECT_ROOT / str(level_config["map_source"]))
    args["train-tasks"] = str(splits["train"])
    args["eval-tasks"] = str(splits["validation"])
    args["scenario"] = str(level_config.get("scenario", "real_lunar_viper"))
    args["mission-profile-scenario"] = str(level_config.get("mission_profile_scenario", "lunar_polar_shadow"))
    args["log-dir"] = str(output_dir / "nominal_train")
    args["total-timesteps"] = int(nominal_timesteps)
    args["seed"] = int(seed)
    args["observation-mode"] = str(level_config.get("observation_mode", args.get("observation-mode", "terrain")))
    args["reward-mode"] = str(level_config.get("reward_mode", args.get("reward-mode", "relative_heuristic")))
    args["reward-scale"] = float(level_config.get("reward_scale", args.get("reward-scale", 10.0)))
    args["reward-cost-key"] = str(level_config.get("reward_cost_key", args.get("reward-cost-key", "attacked_scalar_cost")))
    args["action-mode"] = str(level_config.get("action_mode", args.get("action-mode", "preference_delta")))
    args["action-gain"] = float(level_config.get("action_gain", args.get("action-gain", 3.0)))
    args["max-uncertainty-lambda"] = float(
        level_config.get("max_uncertainty_lambda", args.get("max-uncertainty-lambda", 1.2))
    )
    args["num-envs"] = int(level_config.get("num_envs", args.get("num-envs", 8)))
    args["num-steps"] = int(level_config.get("num_steps", args.get("num-steps", 64)))
    args["learning-rate"] = float(level_config.get("learning_rate", args.get("learning-rate", 0.0002)))
    args["eval-freq"] = min(int(args.get("eval-freq", 1024)), max(int(nominal_timesteps), 1))
    return args


def command_from_args(python_exe: str, train_args: dict[str, Any]) -> list[str]:
    command = [python_exe, "train_cleanrl_ppo.py"]
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


def generate_real_episodes(
    layers_path: Path,
    tasks_path: Path,
    scenario: str,
    mission_profile_scenario: str,
    seed: int,
    count: int,
) -> list[Any]:
    raw_layers = load_real_layers(layers_path)
    tasks = load_task_split(tasks_path)
    selected = tasks[: min(int(count), len(tasks))]
    episodes = []
    for index, task in enumerate(selected):
        rng = np.random.default_rng(int(seed) + int(task.get("seed", 0)) + index)
        episodes.append(
            make_real_planning_episode(
                raw_layers,
                task,
                rng,
                scenario=scenario,
                mission_profile_scenario=mission_profile_scenario,
            )
        )
    return episodes

