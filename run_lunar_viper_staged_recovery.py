"""Train and stage-fine-tune PPO directly on the lunar DEM / VIPER benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

from envs.attack_wrappers import attack_enabled, load_attack_config
from maps.real_terrain import generate_task_splits, load_real_layers, load_task_split, make_real_planning_episode
from run_attack_recovery_finetune import (
    build_train_command,
    checkpoint_step,
    clean_output_dir,
    config_value,
    evaluate_checkpoint,
    load_base_args,
)
from run_staged_attack_recovery import (
    append_stage_columns,
    disabled_attack,
    plot_staged,
    save_checkpoint_copy,
    stage_definitions,
    write_output_guide,
    write_report,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LEVEL_CONFIG = PROJECT_ROOT / "configs" / "levels" / "lunar_viper.json"
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "ppo_lunar_viper_relative_reward.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "lunar_viper_staged_recovery"
DEFAULT_OBSERVATION_ATTACK = {
    "enabled": True,
    "type": "obs_dropout",
    "dropout_prob": 0.25,
    "fill_value": 0.0,
    "clip_to_observation_space": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPO staged recovery directly on lunar DEM / VIPER tasks.")
    parser.add_argument("--level-config", type=Path, default=DEFAULT_LEVEL_CONFIG)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--nominal-timesteps", type=int, default=50000)
    parser.add_argument("--stage-timesteps", type=int, default=20480)
    parser.add_argument("--eval-interval", type=int, default=1024)
    parser.add_argument("--num-eval-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


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
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def load_observation_attack(level_config: dict[str, Any]) -> dict[str, Any]:
    attack = dict(level_config.get("attacks", {}).get("observation", DEFAULT_OBSERVATION_ATTACK))
    attack.setdefault("enabled", True)
    return attack


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
        tile_id=str(level_config.get("tile_id", "viper_200m_tile")),
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


def train_nominal(
    python_exe: str,
    base_args: dict[str, Any],
    output_dir: Path,
    dry_run: bool,
) -> Path:
    command = command_from_args(python_exe, base_args)
    print(" ".join(str(part) for part in command), flush=True)
    seed = int(config_value(base_args, "seed", 0))
    final_model = Path(base_args["log-dir"]) / f"cleanrl_ppo_costmap_seed{seed}" / "final_model.pt"
    if not dry_run:
        subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
        if not final_model.exists():
            raise FileNotFoundError(f"expected nominal checkpoint not found: {final_model}")
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    nominal_checkpoint = checkpoints_dir / "checkpoint_nominal.pt"
    if dry_run:
        print(f"Would copy nominal checkpoint {final_model} -> {nominal_checkpoint}")
    else:
        shutil.copy2(final_model, nominal_checkpoint)
    return nominal_checkpoint


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


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    level_config: dict[str, Any],
    base_args: dict[str, Any],
    env_attack: dict[str, Any],
    obs_attack: dict[str, Any],
    splits: dict[str, Path],
) -> None:
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "level_config": level_config,
        "base_config_args": base_args,
        "environment_attack": env_attack,
        "observation_attack": obs_attack,
        "splits": {key: str(value) for key, value in splits.items()},
        "framing": "independent training on lunar DEM / VIPER benchmark",
    }
    write_json(output_dir / "run_config.json", metadata)


def main() -> int:
    args = parse_args()
    if args.quick:
        args.nominal_timesteps = 2048
        args.stage_timesteps = 2048
        args.eval_interval = 1024
        args.num_eval_episodes = 20
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            args.output_dir = PROJECT_ROOT / "runs" / "debug_lunar_viper_staged_recovery"

    if args.clean_output:
        clean_output_dir(args.output_dir, args.dry_run)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    level_config = read_json(args.level_config)
    layers_path = PROJECT_ROOT / str(level_config["map_source"])
    if not layers_path.exists():
        raise FileNotFoundError(f"lunar DEM layer file not found: {layers_path}")

    splits = prepare_splits(level_config, args.output_dir, args.seed, args.quick)
    base_args = resolved_base_args(
        args.base_config,
        level_config,
        splits,
        args.output_dir,
        args.seed,
        args.nominal_timesteps,
    )
    env_attack = load_environment_attack(level_config)
    obs_attack = load_observation_attack(level_config)
    if not attack_enabled(obs_attack):
        raise ValueError("observation attack must be enabled for staged recovery")
    write_run_config(args.output_dir, args, level_config, base_args, env_attack, obs_attack, splits)

    scenario = str(level_config.get("scenario", "real_lunar_viper"))
    mission_profile = str(level_config.get("mission_profile_scenario", "lunar_polar_shadow"))
    map_size = int(load_real_layers(layers_path)["layer_distance"].shape[0])
    episodes_by_domain = {
        "train_tasks": (
            args.seed,
            generate_real_episodes(
                layers_path,
                splits["train"],
                scenario,
                mission_profile,
                args.seed + 11_000,
                args.num_eval_episodes,
            ),
        ),
        "heldout_tasks": (
            args.seed + 1,
            generate_real_episodes(
                layers_path,
                splits["heldout"],
                scenario,
                mission_profile,
                args.seed + 22_000,
                args.num_eval_episodes,
            ),
        ),
    }

    current_checkpoint = train_nominal(args.python, base_args, args.output_dir, args.dry_run)
    checkpoints_dir = args.output_dir / "checkpoints"
    rows: list[dict[str, Any]] = []
    cumulative_step = 0
    chunk_index = 0

    for stage in stage_definitions(env_attack, obs_attack):
        stage_step = 0
        stage_checkpoint = checkpoints_dir / (
            f"checkpoint_stage{stage['stage_index']:02d}_{stage['active_attack']}_step_{stage_step:05d}.pt"
        )
        save_checkpoint_copy(current_checkpoint, stage_checkpoint, args.dry_run)
        if not args.dry_run:
            eval_rows = evaluate_checkpoint(
                stage_checkpoint,
                cumulative_step,
                episodes_by_domain,
                map_size,
                env_attack,
                obs_attack,
                args.seed,
            )
            rows.extend(append_stage_columns(eval_rows, stage, stage_step, cumulative_step))

        while stage_step < int(args.stage_timesteps):
            chunk_index += 1
            chunk_timesteps = min(int(args.eval_interval), int(args.stage_timesteps) - stage_step)
            command, chunk_final = build_train_command(
                args.python,
                base_args,
                current_checkpoint,
                stage["env_attack"],
                stage["obs_attack"],
                args.output_dir,
                chunk_index,
                chunk_timesteps,
                args.seed + 10_000 * int(stage["stage_index"]) + chunk_index,
            )
            print(" ".join(str(part) for part in command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
                if not chunk_final.exists():
                    raise FileNotFoundError(f"expected chunk checkpoint not found: {chunk_final}")
                actual_step = checkpoint_step(chunk_final)
            else:
                actual_step = chunk_timesteps

            stage_step += actual_step if actual_step > 0 else chunk_timesteps
            cumulative_step += actual_step if actual_step > 0 else chunk_timesteps
            current_checkpoint = checkpoints_dir / (
                f"checkpoint_stage{stage['stage_index']:02d}_{stage['active_attack']}_step_{stage_step:05d}.pt"
            )
            save_checkpoint_copy(chunk_final if not args.dry_run else stage_checkpoint, current_checkpoint, args.dry_run)
            if not args.dry_run:
                eval_rows = evaluate_checkpoint(
                    current_checkpoint,
                    cumulative_step,
                    episodes_by_domain,
                    map_size,
                    env_attack,
                    obs_attack,
                    args.seed,
                )
                rows.extend(append_stage_columns(eval_rows, stage, stage_step, cumulative_step))

    csv_path = args.output_dir / "staged_recovery_curve.csv"
    if not args.dry_run:
        frame = pd.DataFrame(rows)
        frame.to_csv(csv_path, index=False)
        plot_staged(frame, args.output_dir)
        write_output_guide(args.output_dir)
        write_report(frame, args.output_dir)
    print(f"Saved lunar VIPER staged recovery CSV to {csv_path}")
    print(f"Saved figures to {args.output_dir / 'figures'}")
    print(f"Saved checkpoints to {checkpoints_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
