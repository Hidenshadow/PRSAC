#!/usr/bin/env python
"""Post-hoc multi-attack evaluation for an existing shock-recovery run.

This is an evaluation-only script. It does not retrain the policy. It reuses
the checkpoints from a completed run and measures attack drop across a fixed
attack suite and a larger set of evaluation episodes.
"""

from __future__ import annotations

import argparse
import copy
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_attack_recovery_finetune import evaluate_checkpoint
from utils.recovery_runner_helpers import build_eval_episodes, disabled_attack, load_source, save_json


SCALE_KEYS = {
    "attack_strength",
    "error_scale",
    "background_error_scale",
    "confidence_penalty_scale",
    "slope_underestimate_scale",
    "cost_delta",
    "uncertainty_delta",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-mode",
        choices=("nominal", "nominal_final", "nominal_best_final", "all"),
        default="nominal_final",
        help="Which existing checkpoints to re-evaluate. 'nominal' is enough for attack-drop validation.",
    )
    parser.add_argument(
        "--variant-mode",
        choices=("scale", "component", "scale_component"),
        default="scale_component",
        help="Attack suite construction from the benchmark attack in run_config.json.",
    )
    parser.add_argument(
        "--strength-scales",
        nargs="+",
        type=float,
        default=[0.75, 1.0, 1.25],
        help="Multipliers for attack strength/error parameters.",
    )
    parser.add_argument("--min-meaningful-drop", type=float, default=0.05)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_recovery_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 0


def latest_recovery_checkpoint(source_run_dir: Path) -> Path | None:
    checkpoints = sorted(
        (source_run_dir / "checkpoints").glob("checkpoint_recovery_step_*.pt"),
        key=checkpoint_step,
    )
    return checkpoints[-1] if checkpoints else None


def best_recovery_checkpoint(source_run_dir: Path) -> Path | None:
    summary_path = source_run_dir / "shock_recovery_summary.csv"
    if not summary_path.exists():
        return None
    summary = pd.read_csv(summary_path)
    if "best_checkpoint_path" not in summary.columns:
        return None
    heldout = summary[summary.get("eval_domain", "") == "heldout_tasks"]
    row = heldout.iloc[0] if not heldout.empty else summary.iloc[0]
    path_text = str(row.get("best_checkpoint_path", ""))
    if not path_text or path_text.lower() == "nan":
        return None
    path = resolve(Path(path_text))
    return path if path.exists() else None


def selected_checkpoints(source_run_dir: Path, nominal_checkpoint: Path, mode: str) -> list[tuple[str, int, Path]]:
    selected: list[tuple[str, int, Path]] = [("nominal", 0, nominal_checkpoint)]
    if mode == "nominal":
        return selected

    if mode == "all":
        for checkpoint in sorted((source_run_dir / "checkpoints").glob("checkpoint_recovery_step_*.pt"), key=checkpoint_step):
            selected.append((f"recovery_step_{checkpoint_step(checkpoint)}", checkpoint_step(checkpoint), checkpoint))
        return selected

    latest = latest_recovery_checkpoint(source_run_dir)
    best = best_recovery_checkpoint(source_run_dir) if mode == "nominal_best_final" else None
    candidates: list[tuple[str, Path | None]] = [("best_recovery", best), ("final_recovery", latest)]
    seen = {nominal_checkpoint.resolve()}
    for role, checkpoint in candidates:
        if checkpoint is None:
            continue
        resolved = checkpoint.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        selected.append((role, checkpoint_step(checkpoint), checkpoint))
    return selected


def scaled_value(key: str, value: Any, scale: float) -> Any:
    if key in SCALE_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) * float(scale)
    return value


def scale_attack_config(config: dict[str, Any], scale: float) -> dict[str, Any]:
    def scale_obj(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(key): scale_obj(scaled_value(str(key), value, scale)) for key, value in obj.items()}
        if isinstance(obj, list):
            return [scale_obj(item) for item in obj]
        return obj

    scaled = scale_obj(copy.deepcopy(config))
    if isinstance(scaled, dict):
        scaled["suite_strength_scale"] = float(scale)
    return scaled


def component_attack_config(base_attack: dict[str, Any], component_index: int, component: dict[str, Any]) -> dict[str, Any]:
    variant = copy.deepcopy(base_attack)
    variant["type"] = "env_composite"
    variant["name"] = f"component_{component_index}_{component.get('type', 'attack')}"
    variant["components"] = [copy.deepcopy(component)]
    variant["suite_component_index"] = int(component_index)
    return variant


def attack_suite(base_attack: dict[str, Any], variant_mode: str, strength_scales: list[float]) -> list[tuple[str, str, dict[str, Any]]]:
    variants: list[tuple[str, str, dict[str, Any]]] = []
    if variant_mode in {"scale", "scale_component"}:
        for scale in strength_scales:
            label = "standard" if abs(scale - 1.0) < 1e-9 else f"scale_{scale:g}"
            variants.append((label, "scale", scale_attack_config(base_attack, scale)))

    components = base_attack.get("components", [])
    if variant_mode in {"component", "scale_component"} and isinstance(components, list) and components:
        for index, component in enumerate(components):
            if not isinstance(component, dict):
                continue
            label = f"component_{index}_{component.get('type', 'attack')}"
            variants.append((label, "component", component_attack_config(base_attack, index, component)))

    if not variants:
        variants.append(("standard", "base", copy.deepcopy(base_attack)))

    deduped: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for label, kind, config in variants:
        if label in seen:
            continue
        seen.add(label)
        deduped.append((label, kind, config))
    return deduped


def rows_for_checkpoint(
    checkpoint_path: Path,
    checkpoint_role: str,
    recovery_step: int,
    episodes_by_domain: dict[str, tuple[int, list[Any]]],
    map_size: int,
    variants: list[tuple[str, str, dict[str, Any]]],
    seed: int,
    eval_episodes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    clean_kept = False
    for variant_index, (variant_label, variant_kind, env_attack) in enumerate(variants):
        eval_rows = evaluate_checkpoint(
            checkpoint_path,
            int(recovery_step),
            episodes_by_domain,
            map_size,
            env_attack,
            disabled_attack(),
            int(seed) + 10_000 * int(variant_index),
        )
        for row in eval_rows:
            enriched = dict(row)
            enriched["checkpoint_role"] = checkpoint_role
            enriched["recovery_step"] = int(recovery_step)
            enriched["phase"] = "shock" if checkpoint_role == "nominal" else "recovery"
            enriched["num_eval_episodes"] = int(eval_episodes)
            if str(row.get("attack_type")) == "none":
                if clean_kept:
                    continue
                enriched["attack_variant"] = "clean_reference"
                enriched["attack_variant_kind"] = "clean"
            elif str(row.get("attack_type")) == "environment":
                enriched["attack_variant"] = variant_label
                enriched["attack_variant_kind"] = variant_kind
            else:
                continue
            rows.append(enriched)
        clean_kept = True
    return rows


def variant_summary(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame[frame["attack_type"] == "none"][
        ["eval_domain", "checkpoint_role", "recovery_step", "mean_attacked_scalar_cost"]
    ].rename(columns={"mean_attacked_scalar_cost": "clean_cost"})
    attacked = frame[frame["attack_type"] == "environment"].copy()
    merged = attacked.merge(clean, on=["eval_domain", "checkpoint_role", "recovery_step"], how="left")
    merged["attack_drop"] = merged["mean_attacked_scalar_cost"] - merged["clean_cost"]
    merged["attack_degradation"] = merged["attack_drop"] / merged["clean_cost"].abs().clip(lower=1e-12)
    merged["attack_degradation_pct"] = 100.0 * merged["attack_degradation"]
    merged["performance_index"] = 100.0 * merged["clean_cost"] / merged["mean_attacked_scalar_cost"].clip(lower=1e-12)
    merged["episode_standard_error"] = merged["std_attacked_scalar_cost"] / np.sqrt(
        merged["num_eval_episodes"].clip(lower=1)
    )
    merged["episode_ci95_half_width"] = 1.96 * merged["episode_standard_error"]
    columns = [
        "eval_domain",
        "checkpoint_role",
        "recovery_step",
        "attack_variant",
        "attack_variant_kind",
        "num_eval_episodes",
        "clean_cost",
        "mean_attacked_scalar_cost",
        "std_attacked_scalar_cost",
        "episode_standard_error",
        "episode_ci95_half_width",
        "attack_drop",
        "attack_degradation",
        "attack_degradation_pct",
        "performance_index",
        "success_rate",
        "mean_attacked_cell_exposure_ratio",
        "mean_map_mismatch_penalty",
        "mean_path_confidence",
    ]
    return merged[[column for column in columns if column in merged.columns]].sort_values(
        ["eval_domain", "checkpoint_role", "recovery_step", "attack_variant"]
    )


def suite_summary(variant_frame: pd.DataFrame, min_meaningful_drop: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["eval_domain", "checkpoint_role", "recovery_step"]
    for keys, group in variant_frame.groupby(group_cols, sort=True):
        eval_domain, checkpoint_role, recovery_step = keys
        drops = group["attack_degradation"].astype(float).to_numpy()
        attacked_costs = group["mean_attacked_scalar_cost"].astype(float).to_numpy()
        performance = group["performance_index"].astype(float).to_numpy()
        clean_cost = float(group["clean_cost"].iloc[0])
        mean_drop = float(np.nanmean(drops))
        worst_drop = float(np.nanmax(drops))
        rows.append(
            {
                "eval_domain": eval_domain,
                "checkpoint_role": checkpoint_role,
                "recovery_step": int(recovery_step),
                "num_attack_variants": int(group["attack_variant"].nunique()),
                "num_eval_episodes_per_variant": int(group["num_eval_episodes"].iloc[0]),
                "clean_cost": clean_cost,
                "mean_attacked_cost_across_variants": float(np.nanmean(attacked_costs)),
                "worst_attacked_cost_across_variants": float(np.nanmax(attacked_costs)),
                "mean_attack_degradation": mean_drop,
                "mean_attack_degradation_pct": 100.0 * mean_drop,
                "worst_attack_degradation": worst_drop,
                "worst_attack_degradation_pct": 100.0 * worst_drop,
                "mean_performance_index": float(np.nanmean(performance)),
                "worst_performance_index": float(np.nanmin(performance)),
                "std_performance_index_across_variants": float(np.nanstd(performance)),
                "mean_drop_is_meaningful": bool(mean_drop >= min_meaningful_drop),
                "worst_drop_is_meaningful": bool(worst_drop >= min_meaningful_drop),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    source_run_dir = resolve(args.source_run_dir)
    output_dir = resolve(args.output_dir) if args.output_dir else source_run_dir / "attack_suite_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    level_config, base_args, env_attack, nominal_checkpoint = load_source(source_run_dir)
    map_size, episodes_by_domain = build_eval_episodes(
        source_run_dir,
        level_config,
        base_args,
        int(args.seed),
        int(args.eval_episodes),
    )
    variants = attack_suite(env_attack, args.variant_mode, [float(value) for value in args.strength_scales])
    checkpoints = selected_checkpoints(source_run_dir, nominal_checkpoint, args.checkpoint_mode)

    all_rows: list[dict[str, Any]] = []
    for checkpoint_role, recovery_step, checkpoint_path in checkpoints:
        print(f"Evaluating {checkpoint_role} step={recovery_step} checkpoint={checkpoint_path}", flush=True)
        all_rows.extend(
            rows_for_checkpoint(
                checkpoint_path,
                checkpoint_role,
                recovery_step,
                episodes_by_domain,
                map_size,
                variants,
                int(args.seed),
                int(args.eval_episodes),
            )
        )

    frame = pd.DataFrame(all_rows)
    variants_frame = variant_summary(frame)
    suite_frame = suite_summary(variants_frame, float(args.min_meaningful_drop))

    frame.to_csv(output_dir / "attack_suite_curve.csv", index=False)
    variants_frame.to_csv(output_dir / "attack_suite_variant_summary.csv", index=False)
    suite_frame.to_csv(output_dir / "attack_suite_summary.csv", index=False)
    save_json(
        output_dir / "attack_suite_config.json",
        {
            "source_run_dir": source_run_dir,
            "eval_episodes": int(args.eval_episodes),
            "seed": int(args.seed),
            "checkpoint_mode": args.checkpoint_mode,
            "variant_mode": args.variant_mode,
            "strength_scales": [float(value) for value in args.strength_scales],
            "num_attack_variants": len(variants),
            "attack_variants": [
                {"name": name, "kind": kind, "config": config}
                for name, kind, config in variants
            ],
            "checkpoints": [
                {"role": role, "recovery_step": int(step), "path": path}
                for role, step, path in checkpoints
            ],
        },
    )

    print(f"Saved curve: {output_dir / 'attack_suite_curve.csv'}")
    print(f"Saved variant summary: {output_dir / 'attack_suite_variant_summary.csv'}")
    print(f"Saved suite summary: {output_dir / 'attack_suite_summary.csv'}")
    print(suite_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
