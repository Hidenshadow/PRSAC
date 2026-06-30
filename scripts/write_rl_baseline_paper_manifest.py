#!/usr/bin/env python
"""Write a paper-facing manifest for the PPO/SAC baseline experiment grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_lunar_viper_staged_recovery import read_json  # noqa: E402
from run_shock_recovery_experiment import (  # noqa: E402
    map_metadata_summary,
    software_environment,
)
from train_cleanrl_sac import sac_hparams  # noqa: E402


LEVELS = ("level1", "level2", "level3")
DIFFICULTIES = ("easy", "medium", "hard")
ALGORITHMS = ("ppo", "sac")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "rl_baselines" / "paper_manifest")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--nominal-timesteps", type=int, default=50_000)
    parser.add_argument("--recovery-timesteps", type=int, default=20_480)
    parser.add_argument("--eval-interval", type=int, default=1024)
    parser.add_argument("--level1-eval-episodes", type=int, default=300)
    parser.add_argument("--real-eval-episodes", type=int, default=128)
    parser.add_argument("--train-eval-episodes", type=int, default=64)
    return parser.parse_args()


def config_path(level: str, difficulty: str) -> Path:
    return PROJECT_ROOT / "configs" / "levels" / "ppo_difficulty" / f"{level}_{difficulty}.json"


def attack_summary(level_config: dict[str, Any]) -> dict[str, Any]:
    attack = dict(level_config.get("attacks", {}).get("environment", {}))
    components = attack.get("components", [])
    component_types = []
    affected_layers: set[str] = set()
    max_error_scale = 0.0
    max_top_fraction = 0.0
    for component in components:
        if not isinstance(component, dict) or not component.get("enabled", True):
            continue
        component_types.append(str(component.get("type", "")))
        for layer in component.get("affected_layers", []):
            affected_layers.add(str(layer))
        for key in ("error_scale", "degradation_scale"):
            if component.get(key) is not None:
                max_error_scale = max(max_error_scale, float(component[key]))
        if component.get("top_fraction") is not None:
            max_top_fraction = max(max_top_fraction, float(component["top_fraction"]))
    return {
        "attack_name": attack.get("name"),
        "attack_type": attack.get("type"),
        "attack_components": "+".join(component_types),
        "attack_num_components": len(component_types),
        "attack_max_error_or_degradation_scale": max_error_scale if component_types else None,
        "attack_max_top_fraction": max_top_fraction if component_types else None,
        "attack_affected_layers": ",".join(sorted(affected_layers)),
    }


def layer_stats(level_config: dict[str, Any]) -> dict[str, Any]:
    map_source = level_config.get("map_source")
    if not map_source:
        return {}
    path = PROJECT_ROOT / str(map_source)
    if not path.exists():
        return {"map_source_exists": False}
    arrays = np.load(path)
    output: dict[str, Any] = {"map_source_exists": True}
    shape_source = next((key for key in ("layer_distance", "layer_energy", "layer_hazard") if key in arrays), None)
    if shape_source is not None:
        rows, cols = np.asarray(arrays[shape_source]).shape
        output["map_grid_rows"] = int(rows)
        output["map_grid_cols"] = int(cols)
    for layer in ("layer_energy", "layer_hazard", "layer_communication", "layer_illumination"):
        if layer not in arrays:
            continue
        data = np.asarray(arrays[layer], dtype=np.float32)
        output[f"{layer}_mean"] = float(np.nanmean(data))
        output[f"{layer}_p75"] = float(np.nanpercentile(data, 75))
        output[f"{layer}_p90"] = float(np.nanpercentile(data, 90))
    return output


def ppo_hparams() -> dict[str, Any]:
    return {
        "learning_rate": 0.0002,
        "num_envs": 8,
        "num_steps": 64,
        "batch_size": 512,
        "num_minibatches": 8,
        "update_epochs": 4,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "target_kl": 0.05,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "hidden_size": 128,
        "policy": "Beta actor-critic",
    }


def sac_manifest_hparams(nominal_timesteps: int) -> dict[str, Any]:
    class Args:
        pass

    sac_args = Args()
    sac_args.total_timesteps = nominal_timesteps
    hp = sac_hparams(sac_args)
    return {
        "learning_rate": 0.0002,
        "num_envs": 8,
        "gamma": 1.0,
        "hidden_size": 128,
        "policy": "Tanh Gaussian actor",
        **hp,
        "target_entropy": "-action_dim",
        "twin_q": True,
        "automatic_entropy_tuning": True,
    }


def experiment_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level in LEVELS:
        for difficulty in DIFFICULTIES:
            cfg = read_json(config_path(level, difficulty))
            metadata = map_metadata_summary(cfg)
            attack = attack_summary(cfg)
            layers = layer_stats(cfg)
            grid_rows = layers.get("map_grid_rows") or cfg.get("map_size")
            grid_cols = layers.get("map_grid_cols") or cfg.get("map_size")
            grid_cells_per_side = int(grid_rows) if grid_rows == grid_cols and grid_rows is not None else None
            physical_extent_m = metadata.get("requested_meters")
            meters_per_grid_cell = metadata.get("meters_per_pixel")
            for algorithm in ALGORITHMS:
                hparams = ppo_hparams() if algorithm == "ppo" else sac_manifest_hparams(args.nominal_timesteps)
                rows.append(
                    {
                        "algorithm": algorithm,
                        "level": level,
                        "difficulty": difficulty,
                        "seeds": " ".join(str(seed) for seed in args.seeds),
                        "num_training_seeds": len(args.seeds),
                        "nominal_timesteps": args.nominal_timesteps,
                        "recovery_timesteps": args.recovery_timesteps,
                        "eval_interval": args.eval_interval,
                        "num_eval_episodes": args.level1_eval_episodes if level == "level1" else args.real_eval_episodes,
                        "train_eval_episodes": args.train_eval_episodes,
                        "trainer_device_arg": "auto",
                        "scenario": cfg.get("scenario"),
                        "grid_cells_per_side": grid_cells_per_side,
                        "physical_extent_m": physical_extent_m,
                        "meters_per_grid_cell": meters_per_grid_cell,
                        "level_config_map_size": cfg.get("map_size"),
                        "map_size": cfg.get("map_size") or metadata.get("requested_meters"),
                        "map_source": cfg.get("map_source"),
                        "tile_id": cfg.get("tile_id"),
                        "task_sampling_mode": cfg.get("task_sampling_mode", cfg.get("map_sampling_mode")),
                        "min_distance_ratio": cfg.get("min_distance_ratio"),
                        "min_corridor_risk": cfg.get("min_corridor_risk"),
                        "candidate_pool_multiplier": cfg.get("candidate_pool_multiplier"),
                        "reward_mode": cfg.get("reward_mode", "relative_heuristic"),
                        "reward_scale": cfg.get("reward_scale", 10.0),
                        "reward_cost_key": cfg.get("reward_cost_key", "scalar_cost"),
                        "action_mode": cfg.get("action_mode", "preference_delta"),
                        "action_gain": cfg.get("action_gain", 3.0),
                        "max_uncertainty_lambda": cfg.get("max_uncertainty_lambda", 1.2),
                        **attack,
                        **{f"map_{key}": value for key, value in metadata.items()},
                        **layers,
                        **{f"{algorithm}_{key}": value for key, value in hparams.items()},
                    }
                )
    return rows


def write_markdown(frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# RL Baseline Paper Manifest",
        "",
        f"Algorithms: {', '.join(ALGORITHMS)}",
        f"Levels: {', '.join(LEVELS)}",
        f"Difficulties: {', '.join(DIFFICULTIES)}",
        f"Seeds: {' '.join(str(seed) for seed in args.seeds)}",
        f"Nominal timesteps: {args.nominal_timesteps}",
        f"Recovery timesteps: {args.recovery_timesteps}",
        f"Eval interval: {args.eval_interval}",
        "",
        "Paper reporting checklist:",
        "",
        "- algorithms and core hyperparameters;",
        "- map source, map size/tile, rover profile, and terrain metadata;",
        "- task sampling mode and corridor-risk thresholds;",
        "- attack type, components, affected layers, and scales;",
        "- evaluation domains, number of episodes, and training seeds;",
        "- software environment and git snapshot.",
        "",
        "Main CSV: `rl_baseline_paper_manifest.csv`.",
    ]
    (output_dir / "rl_baseline_paper_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(experiment_rows(args))
    frame.to_csv(output_dir / "rl_baseline_paper_manifest.csv", index=False)
    (output_dir / "software_environment.json").write_text(
        json.dumps(software_environment(), indent=2),
        encoding="utf-8",
    )
    write_markdown(frame, output_dir, args)
    print(f"Saved manifest CSV: {output_dir / 'rl_baseline_paper_manifest.csv'}")
    print(f"Saved manifest README: {output_dir / 'rl_baseline_paper_manifest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
