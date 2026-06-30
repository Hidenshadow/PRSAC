#!/usr/bin/env python
"""Evaluate clean-to-attacked drop on candidate real-terrain tiles.

This script does not train. It reuses existing PPO/SAC nominal checkpoints,
generates deterministic evaluation tasks on candidate tiles, and measures the
drop from clean-map evaluation to attacked-map evaluation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from maps.real_terrain import generate_task_splits, load_real_layers  # noqa: E402
from run_attack_recovery_finetune import evaluate_checkpoint  # noqa: E402
from run_lunar_viper_staged_recovery import (  # noqa: E402
    disabled_attack,
    generate_real_episodes,
    read_json,
)


CANDIDATE_RUNS: dict[str, list[dict[str, str]]] = {
    "level2_easy": [
        {"tile_id": "lunar_npd_40_tile", "label": "current lunar 40"},
        {"tile_id": "viper_200m_tile", "label": "VIPER 200m"},
    ],
    "level2_medium": [
        {"tile_id": "lunar_npd_60_tile", "label": "current lunar 60"},
        {"tile_id": "viper_200m_tile", "label": "VIPER 200m"},
        {"tile_id": "lunar_npd_80_tile", "label": "lunar 80"},
    ],
    "level3_medium": [
        {"tile_id": "marsdteed_60_tile", "label": "current Mars 60"},
        {"tile_id": "marsdteed_80_tile", "label": "Mars 80"},
        {"tile_id": "marsdteed_ridge_pgda_500_level3_tile", "label": "Mars ridge 100"},
    ],
}


LEVEL_CONFIGS = {
    "level2_easy": PROJECT_ROOT / "configs" / "levels" / "ppo_difficulty" / "level2_easy.json",
    "level2_medium": PROJECT_ROOT / "configs" / "levels" / "ppo_difficulty" / "level2_medium.json",
    "level3_medium": PROJECT_ROOT / "configs" / "levels" / "ppo_difficulty" / "level3_medium.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-scenarios", nargs="+", default=sorted(CANDIDATE_RUNS))
    parser.add_argument("--algorithms", nargs="+", choices=("ppo", "sac"), default=["ppo", "sac"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs" / "map_sensitivity_pilot")
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


def tile_paths(tile_id: str) -> tuple[Path, Path]:
    root = PROJECT_ROOT / "maps" / "real_dem_tiles" / tile_id
    layers = root / "real_map_layers.npz"
    metadata = root / "tile_metadata.json"
    if not layers.exists():
        raise FileNotFoundError(f"missing candidate tile layers: {layers}")
    if not metadata.exists():
        raise FileNotFoundError(f"missing candidate tile metadata: {metadata}")
    return layers, metadata


def candidate_level_config(source_config: dict[str, Any], tile_id: str) -> dict[str, Any]:
    layers, metadata = tile_paths(tile_id)
    merged = dict(source_config)
    merged["map_source"] = str(layers.relative_to(PROJECT_ROOT))
    merged["metadata"] = str(metadata.relative_to(PROJECT_ROOT))
    merged["tile_id"] = tile_id
    return merged


def nominal_checkpoint(algorithm: str, source_scenario: str, seed: int) -> Path:
    path = (
        PROJECT_ROOT
        / "runs"
        / "rl_baselines"
        / algorithm
        / f"{source_scenario}_shock_recovery_5seeds"
        / f"seed{seed}"
        / "checkpoints"
        / "checkpoint_nominal.pt"
    )
    if not path.exists():
        raise FileNotFoundError(f"missing nominal checkpoint: {path}")
    return path


def load_metadata(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return value if isinstance(value, dict) else {}


def task_stats(tasks_path: Path) -> dict[str, Any]:
    tasks = json.loads(tasks_path.read_text(encoding="utf-8-sig"))
    risks = []
    distances = []
    met = []
    near_slope = []
    for task in tasks:
        difficulty = task.get("difficulty", {}) if isinstance(task, dict) else {}
        if difficulty.get("corridor_risk_score") is not None:
            risks.append(float(difficulty["corridor_risk_score"]))
        if difficulty.get("euclidean_distance_cells") is not None:
            distances.append(float(difficulty["euclidean_distance_cells"]))
        if difficulty.get("met_min_corridor_risk") is not None:
            met.append(bool(difficulty["met_min_corridor_risk"]))
        if difficulty.get("corridor_near_slope_limit_fraction") is not None:
            near_slope.append(float(difficulty["corridor_near_slope_limit_fraction"]))
    return {
        "num_tasks": int(len(tasks)),
        "corridor_risk_mean": float(np.mean(risks)) if risks else np.nan,
        "corridor_risk_p90": float(np.percentile(risks, 90)) if risks else np.nan,
        "distance_cells_mean": float(np.mean(distances)) if distances else np.nan,
        "met_min_corridor_risk_rate": float(np.mean(met)) if met else np.nan,
        "near_slope_limit_mean": float(np.mean(near_slope)) if near_slope else np.nan,
    }


def prepare_eval_domains(
    level_config: dict[str, Any],
    output_dir: Path,
    seed: int,
    eval_episodes: int,
) -> tuple[int, dict[str, tuple[int, list[Any]]], dict[str, Any]]:
    layers_path = PROJECT_ROOT / str(level_config["map_source"])
    metadata_path = PROJECT_ROOT / str(level_config["metadata"])
    splits_dir = output_dir / "splits"
    generate_task_splits(
        layers_path=layers_path,
        output_dir=splits_dir,
        seed=int(seed),
        tile_id=str(level_config["tile_id"]),
        num_train_tasks=0,
        num_validation_tasks=0,
        num_heldout_tasks=int(eval_episodes),
        min_distance_ratio=float(level_config.get("min_distance_ratio", 0.62)),
        metadata_path=metadata_path,
        task_sampling_mode=str(level_config.get("task_sampling_mode", "risk_corridor")),
        min_corridor_risk=(
            float(level_config["min_corridor_risk"])
            if level_config.get("min_corridor_risk") is not None
            else None
        ),
        corridor_radius=int(level_config.get("corridor_radius", 2)),
        candidate_pool_multiplier=int(level_config.get("candidate_pool_multiplier", 60)),
        risk_weights=level_config.get("corridor_risk_weights"),
    )
    heldout_tasks = splits_dir / "heldout_tasks.json"
    raw_layers = load_real_layers(layers_path)
    map_size = int(raw_layers["layer_distance"].shape[0])
    episodes = generate_real_episodes(
        layers_path,
        heldout_tasks,
        str(level_config.get("scenario", "real_lunar_viper")),
        str(level_config.get("mission_profile_scenario", "lunar_polar_shadow")),
        int(seed) + 22_000,
        int(eval_episodes),
    )
    return map_size, {"heldout_tasks": (int(seed) + 1, episodes)}, task_stats(heldout_tasks)


def environment_attack(level_config: dict[str, Any]) -> dict[str, Any]:
    attack = dict(level_config.get("attacks", {}).get("environment", {}))
    attack.setdefault("enabled", True)
    return attack


def summarize_pair(clean: pd.Series, attacked: pd.Series) -> dict[str, float]:
    clean_cost = float(clean["mean_attacked_scalar_cost"])
    attacked_cost = float(attacked["mean_attacked_scalar_cost"])
    performance_index = 100.0 * clean_cost / attacked_cost if attacked_cost > 0 else np.nan
    return {
        "clean_cost": clean_cost,
        "attacked_cost": attacked_cost,
        "attack_drop_cost": attacked_cost - clean_cost,
        "attacked_performance_index": performance_index,
        "attack_drop_index_points": 100.0 - performance_index,
        "success_rate_attacked": float(attacked.get("success_rate", np.nan)),
        "mean_map_mismatch_penalty": float(attacked.get("mean_map_mismatch_penalty", np.nan)),
        "mean_attacked_cell_exposure_ratio": float(attacked.get("mean_attacked_cell_exposure_ratio", np.nan)),
        "mean_belief_abs_error": float(attacked.get("mean_belief_abs_error", np.nan)),
        "mean_true_minus_belief_error": float(attacked.get("mean_true_minus_belief_error", np.nan)),
        "mean_selected_confidence": float(attacked.get("mean_selected_confidence", np.nan)),
        "mean_mismatched_cells": float(attacked.get("mean_mismatched_cells", np.nan)),
    }


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "clean_cost",
        "attacked_cost",
        "attack_drop_cost",
        "attacked_performance_index",
        "attack_drop_index_points",
        "success_rate_attacked",
        "mean_map_mismatch_penalty",
        "mean_attacked_cell_exposure_ratio",
        "mean_belief_abs_error",
        "mean_true_minus_belief_error",
        "mean_selected_confidence",
        "mean_mismatched_cells",
        "corridor_risk_mean",
        "corridor_risk_p90",
        "distance_cells_mean",
        "met_min_corridor_risk_rate",
        "near_slope_limit_mean",
    ]
    grouped = (
        summary.groupby(["source_scenario", "algorithm", "candidate_tile", "candidate_label"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in col if part)
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]
    return grouped


def plot_aggregate(aggregate: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for source_scenario, group in aggregate.groupby("source_scenario"):
        labels = [
            f"{row.candidate_label}\n{row.algorithm.upper()}"
            for row in group.sort_values(["candidate_tile", "algorithm"]).itertuples()
        ]
        values = group.sort_values(["candidate_tile", "algorithm"])["attack_drop_index_points_mean"].astype(float)
        errors = group.sort_values(["candidate_tile", "algorithm"])["attack_drop_index_points_std"].astype(float).fillna(0.0)
        fig, ax = plt.subplots(figsize=(max(7.0, 0.75 * len(labels)), 4.2))
        colors = ["#4c78a8" if "ppo" == algo else "#f58518" for algo in group.sort_values(["candidate_tile", "algorithm"])["algorithm"]]
        ax.bar(np.arange(len(labels)), values, yerr=errors, capsize=3, color=colors)
        ax.axhspan(8, 12, color="#b7e1cd", alpha=0.25, label="easy target")
        ax.axhspan(12, 18, color="#fce8b2", alpha=0.25, label="medium target")
        ax.axhline(0.0, color="0.25", linewidth=1.0)
        ax.set_ylabel("attack drop (index points)")
        ax.set_title(f"Candidate tile clean-to-attacked drop: {source_scenario}")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"fig_{source_scenario}_candidate_tile_attack_drop.png", dpi=180)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    if args.clean_output:
        clean_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for source_scenario in args.source_scenarios:
        if source_scenario not in CANDIDATE_RUNS:
            raise ValueError(f"unknown source scenario {source_scenario}; valid={sorted(CANDIDATE_RUNS)}")
        source_config = read_json(LEVEL_CONFIGS[source_scenario])
        for candidate in CANDIDATE_RUNS[source_scenario]:
            candidate_tile = candidate["tile_id"]
            candidate_label = candidate["label"]
            level_config = candidate_level_config(source_config, candidate_tile)
            _, metadata_path = tile_paths(candidate_tile)
            metadata = load_metadata(metadata_path)
            for seed in args.seeds:
                eval_dir = output_dir / "eval_tasks" / source_scenario / candidate_tile / f"seed{seed}"
                map_size, episodes_by_domain, split_stats = prepare_eval_domains(
                    level_config,
                    eval_dir,
                    int(seed),
                    int(args.eval_episodes),
                )
                attack = environment_attack(level_config)
                for algorithm in args.algorithms:
                    checkpoint = nominal_checkpoint(algorithm, source_scenario, int(seed))
                    print(
                        f"Evaluating {source_scenario} {algorithm} seed{seed} on {candidate_tile} "
                        f"({args.eval_episodes} episodes)",
                        flush=True,
                    )
                    rows = evaluate_checkpoint(
                        checkpoint,
                        0,
                        episodes_by_domain,
                        map_size,
                        attack,
                        disabled_attack(),
                        int(seed),
                    )
                    for row in rows:
                        enriched = dict(row)
                        enriched.update(
                            {
                                "source_scenario": source_scenario,
                                "algorithm": algorithm,
                                "seed": int(seed),
                                "candidate_tile": candidate_tile,
                                "candidate_label": candidate_label,
                                "map_size": int(map_size),
                                "metadata_path": str(metadata_path.relative_to(PROJECT_ROOT)),
                            }
                        )
                        curve_rows.append(enriched)
                    frame = pd.DataFrame(rows)
                    clean_rows = frame[frame["attack_type"] == "none"]
                    attacked_rows = frame[frame["attack_type"] == "environment"]
                    for eval_domain in sorted(set(clean_rows["eval_domain"]).intersection(attacked_rows["eval_domain"])):
                        clean = clean_rows[clean_rows["eval_domain"] == eval_domain].iloc[0]
                        attacked = attacked_rows[attacked_rows["eval_domain"] == eval_domain].iloc[0]
                        summary = summarize_pair(clean, attacked)
                        summary.update(
                            {
                                "source_scenario": source_scenario,
                                "algorithm": algorithm,
                                "seed": int(seed),
                                "eval_domain": eval_domain,
                                "candidate_tile": candidate_tile,
                                "candidate_label": candidate_label,
                                "map_size": int(map_size),
                                "slope_p95_deg": float(metadata.get("slope_p95_deg", np.nan)),
                                "slope_max_deg": float(metadata.get("slope_max_deg", np.nan)),
                                "roughness_p95_m": float(metadata.get("roughness_p95_m", np.nan)),
                                "relief_p95_p05_m": float(metadata.get("relief_p95_p05_m", np.nan)),
                                "obstacle_cell_ratio": float(metadata.get("obstacle_cell_ratio", np.nan)),
                                **split_stats,
                            }
                        )
                        summary_rows.append(summary)

    curve = pd.DataFrame(curve_rows)
    summary = pd.DataFrame(summary_rows)
    aggregate = aggregate_summary(summary)
    curve.to_csv(output_dir / "candidate_tile_attack_drop_curve.csv", index=False)
    summary.to_csv(output_dir / "candidate_tile_attack_drop_summary.csv", index=False)
    aggregate.to_csv(output_dir / "candidate_tile_attack_drop_aggregate.csv", index=False)
    plot_aggregate(aggregate, output_dir)
    print(f"Saved curve: {output_dir / 'candidate_tile_attack_drop_curve.csv'}")
    print(f"Saved summary: {output_dir / 'candidate_tile_attack_drop_summary.csv'}")
    print(f"Saved aggregate: {output_dir / 'candidate_tile_attack_drop_aggregate.csv'}")
    print(
        aggregate[
            [
                "source_scenario",
                "algorithm",
                "candidate_tile",
                "attack_drop_index_points_mean",
                "attack_drop_index_points_std",
                "corridor_risk_mean_mean",
                "near_slope_limit_mean_mean",
            ]
        ].sort_values(["source_scenario", "algorithm", "attack_drop_index_points_mean"], ascending=[True, True, False]).to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
