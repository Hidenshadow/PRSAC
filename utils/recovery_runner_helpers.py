"""Shared helpers for shock-recovery runner scripts.

The helpers in this module are intentionally algorithm-neutral. They keep
PRB-PPO and analysis scripts from depending on obsolete exploratory runner
files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from maps.real_terrain import load_real_layers
from run_attack_recovery_finetune import config_value, evaluate_checkpoint, generate_episodes
from run_lunar_viper_staged_recovery import generate_real_episodes
from run_shock_recovery_experiment import PROJECT_ROOT
from utils.metrics import DEFAULT_MAP_SEED_POOL_SIZE


def resolve_project_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


def disabled_attack() -> dict[str, Any]:
    return {"enabled": False}


def infer_experiment_name(source_run_dir: Path) -> str:
    return source_run_dir.parent.name


def infer_seed(source_run_dir: Path) -> int:
    name = source_run_dir.name
    if name.startswith("seed"):
        try:
            return int(name[4:])
        except ValueError:
            pass
    raise ValueError(f"cannot infer seed from source run directory: {source_run_dir}")


def load_source(source_run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    source_run_dir = source_run_dir if source_run_dir.is_absolute() else PROJECT_ROOT / source_run_dir
    run_config_path = source_run_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"source run_config.json not found: {run_config_path}")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    level_config = dict(run_config.get("level_config", {}))
    base_args = dict(run_config.get("base_config_args", {}))
    env_attack = dict(run_config.get("environment_attack", {}))

    checkpoint_candidates = [
        source_run_dir / "checkpoints" / "checkpoint_nominal.pt",
        source_run_dir / "nominal_ppo" / "checkpoint.pt",
    ]
    checkpoint_candidates.extend(sorted((source_run_dir / "nominal_train").glob("*/final_model.pt")))
    for candidate in checkpoint_candidates:
        if candidate.exists():
            return level_config, base_args, env_attack, candidate
    raise FileNotFoundError(f"nominal checkpoint not found under {source_run_dir}")


def source_num_eval_episodes(source_run_dir: Path, fallback: int = 128) -> int:
    """Return the evaluation episode count used by the source PPO run.

    A non-positive ``--eval-episodes`` in recovery scripts means "match the
    source PPO setting" so BVR/LRR and PPO are compared with the same episode
    count for each scenario/seed.
    """

    source_run_dir = source_run_dir if source_run_dir.is_absolute() else PROJECT_ROOT / source_run_dir
    run_config_path = source_run_dir / "run_config.json"
    if not run_config_path.exists():
        return int(fallback)
    try:
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return int(fallback)
    command_args = run_config.get("command_args", {})
    if not isinstance(command_args, dict):
        return int(fallback)
    try:
        value = int(command_args.get("num_eval_episodes", fallback))
    except (TypeError, ValueError):
        value = int(fallback)
    return int(value) if value > 0 else int(fallback)


def _split_path(source_run_dir: Path, split_name: str, run_config_splits: dict[str, Any] | None = None) -> Path:
    local = source_run_dir / "splits" / f"{split_name}_tasks.json"
    if local.exists():
        return local
    if run_config_splits and split_name in run_config_splits:
        return resolve_project_path(run_config_splits[split_name])
    return local


def _real_eval_domains(
    source_run_dir: Path,
    level_config: dict[str, Any],
    seed: int,
    num_episodes: int,
    split_names: tuple[str, ...],
) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    run_config = json.loads((source_run_dir / "run_config.json").read_text(encoding="utf-8"))
    splits = run_config.get("splits", {}) if isinstance(run_config.get("splits"), dict) else {}
    layers_path = resolve_project_path(level_config["map_source"])
    scenario = str(level_config.get("scenario", "real_lunar_viper"))
    mission_profile = str(level_config.get("mission_profile_scenario", "lunar_polar_shadow"))
    map_size = int(load_real_layers(layers_path)["layer_distance"].shape[0])
    domains: dict[str, tuple[int, list[Any]]] = {}
    seed_offsets = {"train": 11_000, "validation": 33_000, "heldout": 22_000}
    for split_name in split_names:
        task_path = _split_path(source_run_dir, split_name, splits)
        if not task_path.exists():
            continue
        domains[f"{split_name}_tasks"] = (
            int(seed),
            generate_real_episodes(
                layers_path,
                task_path,
                scenario,
                mission_profile,
                seed + seed_offsets.get(split_name, 44_000),
                int(num_episodes),
            ),
        )
    if not domains:
        raise FileNotFoundError(f"no requested task splits {split_names} found under {source_run_dir}")
    return map_size, domains


def _synthetic_eval_domains(
    source_run_dir: Path,
    level_config: dict[str, Any],
    base_args: dict[str, Any],
    seed: int,
    num_episodes: int,
    validation: bool,
) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    map_size = int(config_value(base_args, "map-size", level_config.get("map_size", 48)))
    scenario = str(config_value(base_args, "scenario", level_config.get("scenario", "lunar_rover_corridor")))
    min_distance_ratio = float(
        config_value(base_args, "min-start-goal-distance-ratio", level_config.get("min_distance_ratio", 0.55))
    )
    map_pool_size = int(config_value(base_args, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE))
    fixed_seed = int(config_value(base_args, "fixed-map-seed", level_config.get("fixed_map_seed", 909)))
    if validation:
        seeds = {"validation_seed": fixed_seed + 101}
        episode_seed = seed + 222 + int(fixed_seed + 101)
    else:
        run_config_path = source_run_dir / "run_config.json"
        command_args: dict[str, Any] = {}
        if run_config_path.exists():
            try:
                run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
                if isinstance(run_config.get("command_args"), dict):
                    command_args = dict(run_config["command_args"])
            except json.JSONDecodeError:
                command_args = {}
        in_domain_seed = int(command_args.get("in_domain_seed", fixed_seed))
        heldout_seed = int(command_args.get("heldout_seed", fixed_seed + 1010))
        seeds = {f"in_domain_seed{in_domain_seed}": in_domain_seed, f"heldout_seed{heldout_seed}": heldout_seed}
        # Match run_shock_recovery_experiment.build_synthetic_eval_episodes:
        # both domains are generated from the same RNG seed.
        episode_seed = seed + 222
    domains = {
        name: (
            int(map_seed),
            generate_episodes(
                int(num_episodes),
                int(episode_seed),
                map_size,
                scenario,
                int(map_seed),
                map_pool_size,
                min_start_goal_distance_ratio=min_distance_ratio,
            ),
        )
        for name, map_seed in seeds.items()
    }
    return map_size, domains


def build_eval_episodes(
    source_run_dir: Path,
    level_config: dict[str, Any],
    base_args: dict[str, Any],
    seed: int,
    num_episodes: int,
) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    source_run_dir = source_run_dir if source_run_dir.is_absolute() else PROJECT_ROOT / source_run_dir
    if "map_source" in level_config:
        return _real_eval_domains(source_run_dir, level_config, seed, num_episodes, ("train", "heldout"))
    return _synthetic_eval_domains(source_run_dir, level_config, base_args, seed, num_episodes, validation=False)


def make_validation_eval_domains(
    source_run_dir: Path,
    level_config: dict[str, Any],
    base_args: dict[str, Any],
    seed: int,
    num_episodes: int,
    validation_map_seeds: list[int] | None = None,
) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    source_run_dir = source_run_dir if source_run_dir.is_absolute() else PROJECT_ROOT / source_run_dir
    if "map_source" in level_config:
        return _real_eval_domains(source_run_dir, level_config, seed, num_episodes, ("validation",))
    if validation_map_seeds:
        map_size = int(config_value(base_args, "map-size", level_config.get("map_size", 48)))
        scenario = str(config_value(base_args, "scenario", level_config.get("scenario", "lunar_rover_corridor")))
        min_distance_ratio = float(
            config_value(base_args, "min-start-goal-distance-ratio", level_config.get("min_distance_ratio", 0.55))
        )
        map_pool_size = int(config_value(base_args, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE))
        domains = {
            f"validation_seed{map_seed}": (
                int(map_seed),
                generate_episodes(
                    int(num_episodes),
                    seed + 333 + int(map_seed),
                    map_size,
                    scenario,
                    int(map_seed),
                    map_pool_size,
                    min_start_goal_distance_ratio=min_distance_ratio,
                ),
            )
            for map_seed in validation_map_seeds
        }
        return map_size, domains
    return _synthetic_eval_domains(source_run_dir, level_config, base_args, seed, num_episodes, validation=True)


def add_protocol_columns(rows: list[dict[str, Any]], phase: str, recovery_step: int, checkpoint_role: str) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        enriched = dict(row)
        enriched["phase"] = phase
        enriched["recovery_step"] = int(recovery_step)
        enriched["checkpoint_role"] = checkpoint_role
        output.append(enriched)
    return output


def total_domain_episodes(episodes_by_domain: dict[str, tuple[int, list[Any]]]) -> int:
    return int(sum(len(episodes) for _, episodes in episodes_by_domain.values()))


def mean_costs(rows: list[dict[str, Any]]) -> tuple[float, float]:
    none_costs = [
        float(row["mean_attacked_scalar_cost"])
        for row in rows
        if str(row.get("attack_type")) == "none" and np.isfinite(float(row.get("mean_attacked_scalar_cost", np.nan)))
    ]
    env_costs = [
        float(row["mean_attacked_scalar_cost"])
        for row in rows
        if str(row.get("attack_type")) == "environment" and np.isfinite(float(row.get("mean_attacked_scalar_cost", np.nan)))
    ]
    clean = float(np.mean(none_costs)) if none_costs else float("nan")
    attacked = float(np.mean(env_costs)) if env_costs else float("nan")
    return clean, attacked


def validation_score(
    checkpoint_path: Path,
    recovery_step: int,
    validation_domains: dict[str, tuple[int, list[Any]]],
    map_size: int,
    env_attack: dict[str, Any],
    clean_validation_cost: float,
    attacked_validation_cost: float,
    seed: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    rows = add_protocol_columns(
        evaluate_checkpoint(
            checkpoint_path,
            recovery_step,
            validation_domains,
            map_size,
            env_attack,
            disabled_attack(),
            seed,
        ),
        phase="validation",
        recovery_step=recovery_step,
        checkpoint_role="validation",
    )
    env_costs = [
        float(row["mean_attacked_scalar_cost"])
        for row in rows
        if str(row.get("attack_type")) == "environment" and np.isfinite(float(row.get("mean_attacked_scalar_cost", np.nan)))
    ]
    official_cost = float(np.mean(env_costs)) if env_costs else float("nan")
    denom = max(float(attacked_validation_cost) - float(clean_validation_cost), 1e-8)
    residual = (official_cost - float(clean_validation_cost)) / denom if np.isfinite(official_cost) else float("nan")
    closure = 1.0 - residual if np.isfinite(residual) else float("nan")
    return (
        {
            "validation_official_cost": official_cost,
            "validation_official_residual_degradation": float(residual),
            "validation_official_recovery_closure": float(closure),
        },
        rows,
    )
