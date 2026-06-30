#!/usr/bin/env python3
"""Prepare recovery-only LDAC tuning jobs from completed nominal checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "sac_modified" / "ldac_tuning_pilot_20260624"

LDAC_SOURCE_ROOTS = (
    PROJECT_ROOT / "runs" / "sac_modified" / "ldac_easy_seed0_20260621",
    PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_easy_seed1_2_20260621",
    PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_easy_seed3_4_20260623",
    PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_medium_seed0_1_20260621",
    PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_medium_seed2_4_20260623",
    PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_hard_seed0_1_20260621",
    PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_hard_seed2_4_20260623",
)

DEFAULT_SCENARIOS = ("level1_easy", "level1_hard", "level3_hard")
DEFAULT_SEEDS = (0, 1)

BASELINE_PARAMS = {
    "sac_game_anchor_coef": 0.40,
    "sac_game_advantage_coef": 0.12,
    "sac_game_q_margin": 0.02,
    "sac_game_gate_temperature": 0.05,
    "sac_game_anchor_barrier_coef": 1.5,
    "sac_game_anchor_radius": 0.12,
    "sac_recovery_fixed_alpha": 0.015,
    "sac_recovery_target_entropy_scale": 0.05,
    "sac_recovery_rollout_deterministic_prob": 0.85,
    "sac_recovery_rollout_noise_std": 0.015,
    "sac_recovery_log_std_penalty_coef": 0.01,
    "sac_recovery_log_std_target": -2.0,
}

CANDIDATES: dict[str, dict[str, Any]] = {
    "anchor060": {
        "target_step": 20480,
        "params": {
            **BASELINE_PARAMS,
            "sac_game_anchor_coef": 0.60,
        },
    },
    "adv016_margin003": {
        "target_step": 20480,
        "params": {
            **BASELINE_PARAMS,
            "sac_game_advantage_coef": 0.16,
            "sac_game_q_margin": 0.03,
        },
    },
    "stable060_adv016_long30k": {
        "target_step": 30720,
        "params": {
            **BASELINE_PARAMS,
            "sac_game_anchor_coef": 0.60,
            "sac_game_advantage_coef": 0.16,
            "sac_game_q_margin": 0.03,
            "sac_game_gate_temperature": 0.04,
            "sac_game_anchor_radius": 0.10,
            "sac_recovery_fixed_alpha": 0.012,
            "sac_recovery_rollout_deterministic_prob": 0.90,
            "sac_recovery_rollout_noise_std": 0.010,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--candidates", nargs="+", default=list(CANDIDATES))
    parser.add_argument("--clean", action="store_true", help="Remove prepared candidate dirs before recreating them.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def source_seed_dir(scenario: str, seed: int) -> Path:
    rel = Path(scenario) / f"seed{seed}"
    for root in LDAC_SOURCE_ROOTS:
        candidate = root / rel
        if (candidate / "checkpoints" / "checkpoint_nominal.pt").exists():
            return candidate
    raise FileNotFoundError(f"no source nominal checkpoint for {scenario} seed{seed}")


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_seed_dir(
    source_dir: Path,
    dest_dir: Path,
    candidate_name: str,
    candidate_spec: dict[str, Any],
    clean: bool,
) -> None:
    if clean and dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = dest_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_dir / "checkpoints" / "checkpoint_nominal.pt", checkpoints_dir / "checkpoint_nominal.pt")

    if (source_dir / "splits").exists():
        copy_tree(source_dir / "splits", dest_dir / "splits")
    copy_file_if_exists(source_dir / "nominal_training_eval.csv", dest_dir / "nominal_training_eval.csv")

    run_config = json.loads((source_dir / "run_config.json").read_text(encoding="utf-8"))
    command_args = dict(run_config.get("command_args", {}))
    command_args["output_dir"] = str(dest_dir)
    command_args["clean_output"] = False
    command_args["recovery_timesteps"] = int(candidate_spec["target_step"])
    command_args["python"] = str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    command_args["device"] = "cpu"
    for key, value in dict(candidate_spec["params"]).items():
        command_args[key] = value

    splits = {}
    for name in ("train", "validation", "heldout"):
        path = dest_dir / "splits" / f"{name}_tasks.json"
        if path.exists():
            splits[name] = str(path)

    run_config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_config["command_args"] = command_args
    run_config["splits"] = splits
    run_config["tuning_candidate"] = candidate_name
    run_config["tuning_target_step"] = int(candidate_spec["target_step"])
    run_config["tuning_source_seed_dir"] = str(source_dir)
    run_config["tuning_params"] = dict(candidate_spec["params"])
    (dest_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []

    for candidate_name in args.candidates:
        if candidate_name not in CANDIDATES:
            raise ValueError(f"unknown candidate {candidate_name!r}; choices: {sorted(CANDIDATES)}")
        spec = CANDIDATES[candidate_name]
        for scenario in args.scenarios:
            for seed in args.seeds:
                source_dir = source_seed_dir(scenario, int(seed))
                dest_dir = output_root / candidate_name / scenario / f"seed{int(seed)}"
                prepare_seed_dir(source_dir, dest_dir, candidate_name, spec, clean=bool(args.clean))
                jobs.append(
                    {
                        "candidate": candidate_name,
                        "scenario": scenario,
                        "seed": int(seed),
                        "target_step": int(spec["target_step"]),
                        "seed_dir": str(dest_dir),
                        "source_seed_dir": str(source_dir),
                    }
                )

    jobs_path = output_root / "tuning_jobs.csv"
    with jobs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["candidate", "scenario", "seed", "target_step", "seed_dir", "source_seed_dir"],
        )
        writer.writeheader()
        writer.writerows(jobs)

    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_root": str(output_root),
        "scenarios": list(args.scenarios),
        "seeds": [int(seed) for seed in args.seeds],
        "candidates": {name: CANDIDATES[name] for name in args.candidates},
        "jobs_csv": str(jobs_path),
        "num_jobs": len(jobs),
    }
    (output_root / "tuning_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(jobs)} LDAC tuning jobs under {output_root}")
    print(f"Jobs CSV: {jobs_path}")


if __name__ == "__main__":
    main()
