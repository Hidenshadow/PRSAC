#!/usr/bin/env python
"""Calibrate environment attack variants using existing PPO nominal checkpoints.

This script does not train PPO. It reuses nominal checkpoints from the
shock-recovery runs and evaluates a small menu of physically meaningful map
attacks:

- current_composite: current default config.
- stronger_spatial: stronger low-confidence/high-consequence spatial mismatch.
- route_shortcut_moderate: optimistic map error around a short route corridor.
- route_shortcut_hard: stronger route-corridor shortcut deception.
- deceptive_shortcut_moderate: short + low-confidence/high-consequence corridor.
- deceptive_shortcut_hard: stronger deceptive shortcut corridor.
- route_deception_hard: underestimate a bad shortcut and overestimate the
  conservative mission-priority corridor.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import re
import shutil
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_attack_recovery_finetune import config_value, evaluate_checkpoint
from run_lunar_viper_staged_recovery import read_json
from run_shock_recovery_experiment import (
    build_real_eval_episodes,
    build_synthetic_eval_episodes,
    disabled_attack,
    is_real_level,
    synthetic_base_args,
)
from utils.metrics import DEFAULT_MAP_SEED_POOL_SIZE


LEVELS = {
    "level1": {
        "config": PROJECT_ROOT / "configs" / "levels" / "synthetic_corridor.json",
        "run_root": PROJECT_ROOT / "runs" / "level1_synthetic_corridor_shock_recovery_3seeds",
    },
    "level2": {
        "config": PROJECT_ROOT / "configs" / "levels" / "lunar_viper.json",
        "run_root": PROJECT_ROOT / "runs" / "level2_lunar_viper_shock_recovery_3seeds",
    },
    "level3": {
        "config": PROJECT_ROOT / "configs" / "levels" / "mars_dtm.json",
        "run_root": PROJECT_ROOT / "runs" / "level3_mars_dtm_shock_recovery_3seeds",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate candidate environment attacks without retraining PPO.")
    parser.add_argument("--levels", nargs="+", choices=sorted(LEVELS), default=sorted(LEVELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Attack variants to evaluate. Defaults to the full calibration menu.",
    )
    parser.add_argument("--num-eval-episodes", type=int, default=96)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "attack_calibration_sweep")
    parser.add_argument("--in-domain-seed", type=int, default=909)
    parser.add_argument("--heldout-seed", type=int, default=1919)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    resolved = output_dir.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"refusing to clean output outside project root: {resolved}") from exc
    shutil.rmtree(resolved)


def composite_attack(
    name: str,
    selection_mode: str,
    reference_policy: str | None,
    corridor_radius: int,
    top_fraction: float,
    error_scale: float,
    background_error_scale: float,
    degradation_scale: float,
    background_degradation_scale: float,
    confidence_gamma: float = 1.25,
    correlation_sigma: float = 3.5,
) -> dict[str, Any]:
    spatial: dict[str, Any] = {
        "enabled": True,
        "type": "env_spatial_belief_mismatch",
        "mode": "risk_underestimate",
        "selection_mode": selection_mode,
        "top_fraction": float(top_fraction),
        "attack_strength": 3.0,
        "error_scale": float(error_scale),
        "background_error_scale": float(background_error_scale),
        "confidence_gamma": float(confidence_gamma),
        "correlation_sigma": float(correlation_sigma),
        "min_confidence": 0.10,
        "max_confidence": 0.95,
        "affected_layers": ["energy", "hazard", "communication", "illumination"],
        "confidence_to_uncertainty": True,
    }
    confidence: dict[str, Any] = {
        "enabled": True,
        "type": "env_confidence_degradation",
        "selection_mode": selection_mode,
        "top_fraction": float(top_fraction),
        "degradation_scale": float(degradation_scale),
        "background_degradation_scale": float(background_degradation_scale),
        "confidence_floor": 0.05,
        "confidence_gamma": 1.0,
        "correlation_sigma": float(correlation_sigma),
        "affected_layers": ["energy", "hazard", "communication", "illumination"],
    }
    if selection_mode == "path_corridor":
        spatial["corridor_radius"] = int(corridor_radius)
        confidence["corridor_radius"] = int(corridor_radius)
    if reference_policy:
        spatial["reference_policy"] = reference_policy
        confidence["reference_policy"] = reference_policy

    return {
        "enabled": True,
        "type": "env_composite",
        "name": name,
        "apply_during_training": True,
        "reward_uses_attacked_cost": True,
        "components": [spatial, confidence],
    }


def route_deception_attack(
    name: str,
    shortcut_error_scale: float,
    shortcut_background_scale: float,
    conservative_error_scale: float,
    conservative_background_scale: float,
    degradation_scale: float,
    corridor_radius: int,
) -> dict[str, Any]:
    shortcut = {
        "enabled": True,
        "type": "env_spatial_belief_mismatch",
        "mode": "risk_underestimate",
        "selection_mode": "deceptive_shortcut",
        "corridor_radius": int(corridor_radius),
        "attack_strength": 3.0,
        "error_scale": float(shortcut_error_scale),
        "background_error_scale": float(shortcut_background_scale),
        "confidence_gamma": 0.85,
        "correlation_sigma": 3.5,
        "min_confidence": 0.10,
        "max_confidence": 0.95,
        "affected_layers": ["energy", "hazard", "communication", "illumination"],
        "confidence_to_uncertainty": True,
        "shortcut_attraction": 0.90,
        "shortcut_consequence_weight": 0.05,
    }
    conservative = {
        "enabled": True,
        "type": "env_spatial_belief_mismatch",
        "mode": "risk_overestimate",
        "selection_mode": "path_corridor",
        "reference_policy": "risk_neutral_mission",
        "corridor_radius": int(max(corridor_radius - 1, 1)),
        "attack_strength": 3.0,
        "error_scale": float(conservative_error_scale),
        "background_error_scale": float(conservative_background_scale),
        "confidence_gamma": 0.90,
        "correlation_sigma": 3.5,
        "min_confidence": 0.10,
        "max_confidence": 0.95,
        "affected_layers": ["energy", "hazard", "communication", "illumination"],
        "confidence_to_uncertainty": True,
    }
    confidence = {
        "enabled": True,
        "type": "env_confidence_degradation",
        "selection_mode": "deceptive_shortcut",
        "corridor_radius": int(corridor_radius),
        "degradation_scale": float(degradation_scale),
        "background_degradation_scale": 0.08,
        "confidence_floor": 0.05,
        "confidence_gamma": 0.90,
        "correlation_sigma": 3.5,
        "affected_layers": ["energy", "hazard", "communication", "illumination"],
    }
    return {
        "enabled": True,
        "type": "env_composite",
        "name": name,
        "apply_during_training": True,
        "reward_uses_attacked_cost": True,
        "components": [shortcut, conservative, confidence],
    }


def attack_variants(level_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    current = deepcopy(level_config["attacks"]["environment"])
    variants = {
        "current_composite": current,
        "stronger_spatial": composite_attack(
            name="stronger_spatial_low_confidence_high_consequence",
            selection_mode="low_confidence_high_consequence",
            reference_policy=None,
            corridor_radius=0,
            top_fraction=0.35,
            error_scale=0.45,
            background_error_scale=0.22,
            degradation_scale=0.45,
            background_degradation_scale=0.12,
            confidence_gamma=1.10,
        ),
        "route_shortcut_moderate": composite_attack(
            name="route_shortcut_moderate",
            selection_mode="path_corridor",
            reference_policy="distance_shortcut",
            corridor_radius=3,
            top_fraction=0.25,
            error_scale=0.50,
            background_error_scale=0.08,
            degradation_scale=0.45,
            background_degradation_scale=0.05,
            confidence_gamma=1.00,
        ),
        "route_shortcut_hard": composite_attack(
            name="route_shortcut_hard",
            selection_mode="path_corridor",
            reference_policy="distance_shortcut",
            corridor_radius=4,
            top_fraction=0.25,
            error_scale=0.70,
            background_error_scale=0.12,
            degradation_scale=0.60,
            background_degradation_scale=0.08,
            confidence_gamma=0.90,
        ),
        "deceptive_shortcut_moderate": composite_attack(
            name="deceptive_shortcut_moderate",
            selection_mode="deceptive_shortcut",
            reference_policy=None,
            corridor_radius=3,
            top_fraction=0.25,
            error_scale=0.50,
            background_error_scale=0.08,
            degradation_scale=0.45,
            background_degradation_scale=0.05,
            confidence_gamma=1.00,
        ),
        "deceptive_shortcut_hard": composite_attack(
            name="deceptive_shortcut_hard",
            selection_mode="deceptive_shortcut",
            reference_policy=None,
            corridor_radius=4,
            top_fraction=0.25,
            error_scale=0.70,
            background_error_scale=0.12,
            degradation_scale=0.60,
            background_degradation_scale=0.08,
            confidence_gamma=0.90,
        ),
        "route_deception_hard": route_deception_attack(
            name="route_deception_hard",
            shortcut_error_scale=0.85,
            shortcut_background_scale=0.10,
            conservative_error_scale=0.45,
            conservative_background_scale=0.02,
            degradation_scale=0.65,
            corridor_radius=4,
        ),
        "route_deception_extreme": route_deception_attack(
            name="route_deception_extreme",
            shortcut_error_scale=1.10,
            shortcut_background_scale=0.15,
            conservative_error_scale=0.65,
            conservative_background_scale=0.03,
            degradation_scale=0.80,
            corridor_radius=5,
        ),
    }
    return variants


def seed_from_dir(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if match is None:
        raise ValueError(f"could not infer seed from {path}")
    return int(match.group(1))


def build_eval_episodes_for_seed(
    level_name: str,
    level_config: dict[str, Any],
    run_root: Path,
    seed: int,
    num_eval_episodes: int,
    in_domain_seed: int,
    heldout_seed: int,
) -> tuple[int, dict[str, tuple[int, list[Any]]]]:
    if is_real_level(level_config):
        splits = {
            "train": run_root / f"seed{seed}" / "splits" / "train_tasks.json",
            "validation": run_root / f"seed{seed}" / "splits" / "validation_tasks.json",
            "heldout": run_root / f"seed{seed}" / "splits" / "heldout_tasks.json",
        }
        missing = [str(path) for path in splits.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing task split(s) for {level_name} seed{seed}: {missing}")
        return build_real_eval_episodes(level_config, splits, seed, num_eval_episodes)

    base_args = synthetic_base_args(
        level_config,
        None,
        PROJECT_ROOT / "runs" / "_attack_calibration_tmp",
        seed,
        1,
        1,
        1,
    )
    map_pool_size = int(config_value(base_args, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE))
    namespace = SimpleNamespace(
        num_eval_episodes=int(num_eval_episodes),
        seed=int(seed),
        in_domain_seed=int(in_domain_seed),
        heldout_seed=int(heldout_seed),
    )
    return build_synthetic_eval_episodes(namespace, base_args, map_pool_size)


def summarize_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    summaries = []
    for (level, variant, training_seed, eval_domain), group in frame.groupby(
        ["level", "attack_variant", "training_seed", "eval_domain"]
    ):
        clean = group[group["attack_type"] == "none"]
        attacked = group[group["attack_type"] == "environment"]
        if clean.empty or attacked.empty:
            continue
        clean_row = clean.iloc[0]
        attacked_row = attacked.iloc[0]
        clean_cost = float(clean_row["mean_attacked_scalar_cost"])
        attacked_cost = float(attacked_row["mean_attacked_scalar_cost"])
        attack_drop = attacked_cost - clean_cost
        summaries.append(
            {
                "level": level,
                "attack_variant": variant,
                "training_seed": int(training_seed),
                "eval_domain": eval_domain,
                "clean_cost": clean_cost,
                "attacked_cost": attacked_cost,
                "attack_drop": attack_drop,
                "attack_drop_ratio": attack_drop / (abs(clean_cost) + 1e-8),
                "success_rate": float(attacked_row["success_rate"]),
                "mean_map_mismatch_penalty": float(attacked_row.get("mean_map_mismatch_penalty", np.nan)),
                "mean_path_confidence": float(attacked_row.get("mean_path_confidence", np.nan)),
                "mean_attacked_cell_exposure_ratio": float(
                    attacked_row.get("mean_attacked_cell_exposure_ratio", np.nan)
                ),
                "mean_belief_abs_error": float(attacked_row.get("mean_belief_abs_error", np.nan)),
                "mean_true_minus_belief_error": float(
                    attacked_row.get("mean_true_minus_belief_error", np.nan)
                ),
                "mean_selected_confidence": float(attacked_row.get("mean_selected_confidence", np.nan)),
                "mean_mismatched_cells": float(attacked_row.get("mean_mismatched_cells", np.nan)),
            }
        )
    return pd.DataFrame(summaries)


def main() -> int:
    args = parse_args()
    if args.clean_output:
        clean_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    for level_name in args.levels:
        spec = LEVELS[level_name]
        level_config = read_json(spec["config"])
        run_root = spec["run_root"]
        variants = attack_variants(level_config)
        if args.variants:
            missing = [name for name in args.variants if name not in variants]
            if missing:
                raise ValueError(f"unknown attack variant(s): {missing}; valid={sorted(variants)}")
            variants = {name: variants[name] for name in args.variants}
        for seed in args.seeds:
            checkpoint = run_root / f"seed{seed}" / "checkpoints" / "checkpoint_nominal.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(f"missing nominal checkpoint: {checkpoint}")
            map_size, episodes_by_domain = build_eval_episodes_for_seed(
                level_name,
                level_config,
                run_root,
                seed,
                args.num_eval_episodes,
                args.in_domain_seed,
                args.heldout_seed,
            )
            for variant_name, attack in variants.items():
                print(f"Evaluating {level_name} seed{seed} variant={variant_name}", flush=True)
                rows = evaluate_checkpoint(
                    checkpoint,
                    0,
                    episodes_by_domain,
                    map_size,
                    attack,
                    disabled_attack(),
                    seed,
                )
                for row in rows:
                    enriched = dict(row)
                    enriched["level"] = level_name
                    enriched["training_seed"] = int(seed)
                    enriched["attack_variant"] = variant_name
                    enriched["attack_config_name"] = str(attack.get("name", variant_name))
                    all_rows.append(enriched)

    curve = pd.DataFrame(all_rows)
    curve_path = args.output_dir / "attack_calibration_curve.csv"
    curve.to_csv(curve_path, index=False)

    summary = summarize_rows(all_rows)
    summary_path = args.output_dir / "attack_calibration_summary.csv"
    summary.to_csv(summary_path, index=False)

    aggregate_cols = [
        "attack_drop",
        "attack_drop_ratio",
        "success_rate",
        "mean_map_mismatch_penalty",
        "mean_path_confidence",
        "mean_attacked_cell_exposure_ratio",
        "mean_belief_abs_error",
        "mean_true_minus_belief_error",
        "mean_selected_confidence",
        "mean_mismatched_cells",
    ]
    aggregate = (
        summary.groupby(["level", "attack_variant", "eval_domain"])[aggregate_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in aggregate.columns
    ]
    aggregate_path = args.output_dir / "attack_calibration_aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    print(f"Saved curve: {curve_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved aggregate: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
