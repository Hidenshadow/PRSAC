#!/usr/bin/env python
"""Post-hoc staged recovery evaluation with belief-mismatch diagnostics.

This script reuses existing staged-recovery checkpoints. It does not train.
The output is a new CSV with the current evaluation fields, including
belief-vs-true map metrics.
"""

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

from maps.real_terrain import load_real_layers
from run_attack_recovery_finetune import evaluate_checkpoint, generate_episodes
from run_lunar_viper_staged_recovery import generate_real_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-evaluate staged recovery checkpoints with belief metrics.")
    parser.add_argument(
        "--run-dir",
        action="append",
        type=Path,
        default=[],
        help="Staged recovery run directory. Can be passed more than once. Defaults to all staged run dirs under runs/.",
    )
    parser.add_argument(
        "--mode",
        choices=("key", "all"),
        default="key",
        help="key evaluates stage boundaries plus heldout env/combined best checkpoints; all evaluates every checkpoint.",
    )
    parser.add_argument(
        "--num-eval-episodes",
        type=int,
        default=None,
        help="Override evaluation episode count. Useful for quick smoke checks.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output CSV filename. Defaults to staged_recovery_belief_metrics_<mode>.csv.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def discover_run_dirs() -> list[Path]:
    candidates = []
    for csv_path in sorted((PROJECT_ROOT / "runs").glob("**/staged_recovery_curve.csv")):
        run_dir = csv_path.parent
        if (run_dir / "run_config.json").exists():
            candidates.append(run_dir)
    return candidates


def synthetic_episodes(run_config: dict[str, Any], num_eval_episodes: int) -> dict[str, tuple[int, list[Any]]]:
    command_args = dict(run_config.get("command_args", {}))
    resolved = dict(run_config.get("resolved", {}))
    seed = int(command_args.get("seed", resolved.get("seed", 0)))
    map_size = int(resolved.get("map_size", 48))
    scenario = str(resolved.get("scenario", "lunar_rover_corridor"))
    map_pool_size = int(resolved.get("map_pool_size", 32))
    in_domain_seed = int(command_args.get("in_domain_seed", 909))
    heldout_seed = int(command_args.get("heldout_seed", 1919))
    return {
        f"in_domain_seed{in_domain_seed}": (
            in_domain_seed,
            generate_episodes(num_eval_episodes, seed + 222, map_size, scenario, in_domain_seed, map_pool_size),
        ),
        f"heldout_seed{heldout_seed}": (
            heldout_seed,
            generate_episodes(num_eval_episodes, seed + 222, map_size, scenario, heldout_seed, map_pool_size),
        ),
    }


def real_episodes(run_config: dict[str, Any], num_eval_episodes: int) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    command_args = dict(run_config.get("command_args", {}))
    level_config = dict(run_config["level_config"])
    splits = {name: resolve_path(path) for name, path in dict(run_config["splits"]).items()}
    layers_path = resolve_path(level_config["map_source"])
    seed = int(command_args.get("seed", 0))
    scenario = str(level_config.get("scenario", "real_lunar_viper"))
    mission_profile = str(level_config.get("mission_profile_scenario", "lunar_polar_shadow"))
    map_size = int(load_real_layers(layers_path)["layer_distance"].shape[0])
    episodes = {
        "train_tasks": (
            seed,
            generate_real_episodes(
                layers_path,
                splits["train"],
                scenario,
                mission_profile,
                seed + 11_000,
                num_eval_episodes,
            ),
        ),
        "heldout_tasks": (
            seed + 1,
            generate_real_episodes(
                layers_path,
                splits["heldout"],
                scenario,
                mission_profile,
                seed + 22_000,
                num_eval_episodes,
            ),
        ),
    }
    return map_size, episodes


def build_eval_context(run_config: dict[str, Any], num_eval_episodes: int) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    if "level_config" in run_config and "splits" in run_config:
        return real_episodes(run_config, num_eval_episodes)
    resolved = dict(run_config.get("resolved", {}))
    return int(resolved.get("map_size", 48)), synthetic_episodes(run_config, num_eval_episodes)


def selected_checkpoint_rows(frame: pd.DataFrame, mode: str) -> pd.DataFrame:
    checkpoints = frame.drop_duplicates("checkpoint_path").copy()
    checkpoints = checkpoints.sort_values(["cumulative_step", "stage_index", "stage_step"])
    if mode == "all":
        return checkpoints.reset_index(drop=True)

    selected_paths: set[str] = set()
    selected_paths.update(
        frame[(frame["stage_index"] == 1) & (frame["stage_step"] == 0)]["checkpoint_path"].astype(str).tolist()
    )
    for _, group in frame.groupby("stage_index"):
        max_step = group["stage_step"].max()
        selected_paths.update(group[group["stage_step"] == max_step]["checkpoint_path"].astype(str).tolist())

    heldout = frame[
        frame["eval_domain"].astype(str).str.contains("heldout", case=False, na=False)
        & frame["attack_type"].isin(["environment", "combined"])
    ]
    for _, group in heldout.groupby("attack_type"):
        if group.empty:
            continue
        selected_paths.add(str(group.loc[group["mean_attacked_scalar_cost"].astype(float).idxmin(), "checkpoint_path"]))

    return checkpoints[checkpoints["checkpoint_path"].astype(str).isin(selected_paths)].reset_index(drop=True)


def reevaluate_run(run_dir: Path, mode: str, num_eval_episodes_override: int | None, output_name: str | None) -> Path:
    run_dir = resolve_path(run_dir)
    run_config = read_json(run_dir / "run_config.json")
    old_frame = pd.read_csv(run_dir / "staged_recovery_curve.csv")
    command_args = dict(run_config.get("command_args", {}))
    num_eval_episodes = int(num_eval_episodes_override or command_args.get("num_eval_episodes", 300))
    seed = int(command_args.get("seed", dict(run_config.get("resolved", {})).get("seed", 0)))
    map_size, episodes_by_domain = build_eval_context(run_config, num_eval_episodes)
    env_attack = dict(run_config.get("environment_attack", {}))
    obs_attack = dict(run_config.get("observation_attack", {}))
    selected = selected_checkpoint_rows(old_frame, mode)

    rows: list[dict[str, Any]] = []
    for _, checkpoint_row in selected.iterrows():
        checkpoint_path = resolve_path(str(checkpoint_row["checkpoint_path"]))
        eval_rows = evaluate_checkpoint(
            checkpoint_path,
            int(checkpoint_row["cumulative_step"]),
            episodes_by_domain,
            map_size,
            env_attack,
            obs_attack,
            seed,
        )
        for row in eval_rows:
            enriched = dict(row)
            enriched["stage_index"] = int(checkpoint_row["stage_index"])
            enriched["stage"] = str(checkpoint_row["stage"])
            enriched["active_attack"] = str(checkpoint_row["active_attack"])
            enriched["stage_step"] = int(checkpoint_row["stage_step"])
            enriched["cumulative_step"] = int(checkpoint_row["cumulative_step"])
            enriched["posthoc_mode"] = mode
            enriched["posthoc_num_eval_episodes"] = num_eval_episodes
            rows.append(enriched)

    output = run_dir / (output_name or f"staged_recovery_belief_metrics_{mode}.csv")
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main() -> int:
    args = parse_args()
    run_dirs = args.run_dir or discover_run_dirs()
    if not run_dirs:
        raise SystemExit("No staged recovery run directories found.")
    for run_dir in run_dirs:
        output = reevaluate_run(run_dir, args.mode, args.num_eval_episodes, args.output_name)
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
