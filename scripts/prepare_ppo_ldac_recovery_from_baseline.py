#!/usr/bin/env python
"""Prepare PPO-LDAC recovery-only runs from existing vanilla PPO nominal checkpoints."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "ppo_modified" / "ldac_ppo_from_ppo_nominal_seed0_9scenarios"
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "runs" / "rl_baselines" / "ppo"
SCENARIOS = (
    "level1_easy",
    "level2_easy",
    "level3_easy",
    "level1_medium",
    "level2_medium",
    "level3_medium",
    "level1_hard",
    "level2_hard",
    "level3_hard",
)


LDAC_RECOVERY_DEFAULTS = {
    "game_recovery_enabled": True,
    "game_attack_mixture_size": 5,
    "game_attack_jitter": 0.15,
    "game_attack_sampler": "adaptive_bandit",
    "game_attack_variant_mode": "scale",
    "game_bandit_eta": 0.8,
    "game_bandit_min_prob": 0.05,
    "game_bandit_prior_mix": 0.10,
    "game_bandit_cost_key": "policy_cost",
    "game_bandit_benchmark_floor": 0.0,
    "qre_minimax_recovery_enabled": False,
    "qre_temperature_start": 2.0,
    "qre_temperature_end": 0.25,
    "qre_temperature_schedule": "exp",
    "qre_response_rate": 0.75,
    "qre_cost_ema_beta": 0.60,
    "qre_prior_mix": 0.10,
    "qre_min_prob": 0.03,
    "qre_max_prob_cap": 0.55,
    "qre_exploration_bonus": 0.10,
    "qre_benchmark_floor": 0.20,
    "qre_cost_normalization": "std",
    "ap_cvar_enabled": False,
    "planner_regret_recovery_enabled": False,
    "game_teacher_recovery_enabled": False,
    "teacher_residual_recovery_enabled": False,
    "acbr_recovery_enabled": False,
    "bagr_recovery_enabled": False,
    "sac_game_recovery_enabled": False,
    "game_nominal_prior_coef": 0.25,
    "game_lambda_drift_coef": 0.0,
    "game_lambda_drift_margin": 0.10,
    "game_risk_drift_coef": 0.0,
    "game_risk_drift_margin": 0.10,
    "game_risk_action_indices": "energy,hazard,communication,illumination",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv" / "Scripts" / "python.exe")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--recovery-timesteps", type=int, default=20_480)
    parser.add_argument("--eval-interval", type=int, default=1024)
    parser.add_argument("--num-eval-episodes", type=int, default=300)
    parser.add_argument("--force", action="store_true", help="Remove prepared destination seed dirs before copying.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def copy_file(src: Path, dst: Path, force: bool) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def rewrite_real_terrain_paths(run_config: dict) -> None:
    """Make archived Linux real-terrain paths usable in the current workspace."""
    level_config = run_config.get("level_config")
    base_args = run_config.get("base_config_args")
    if not isinstance(level_config, dict) or not isinstance(base_args, dict):
        return
    map_source = level_config.get("map_source")
    if map_source:
        base_args["layers-path"] = str(PROJECT_ROOT / str(map_source))
    metadata = level_config.get("metadata")
    if metadata and "metadata" in base_args:
        base_args["metadata"] = str(PROJECT_ROOT / str(metadata))


def prepare_one(
    scenario: str,
    baseline_root: Path,
    output_root: Path,
    seed: int,
    python_exe: Path,
    device: str,
    recovery_timesteps: int,
    eval_interval: int,
    num_eval_episodes: int,
    force: bool,
) -> Path:
    source = baseline_root / f"{scenario}_shock_recovery_5seeds" / f"seed{seed}"
    if not source.exists():
        raise FileNotFoundError(f"missing baseline seed dir: {source}")
    source_config = source / "run_config.json"
    source_checkpoint = source / "checkpoints" / "checkpoint_nominal.pt"
    if not source_config.exists():
        raise FileNotFoundError(f"missing baseline run_config.json: {source_config}")
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"missing baseline nominal checkpoint: {source_checkpoint}")

    destination = output_root / scenario / f"seed{seed}"
    if force and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "checkpoints").mkdir(parents=True, exist_ok=True)

    copy_file(source_checkpoint, destination / "checkpoints" / "checkpoint_nominal.pt", force=force)
    nominal_eval = source / "nominal_training_eval.csv"
    if nominal_eval.exists():
        copy_file(nominal_eval, destination / "nominal_training_eval.csv", force=force)

    run_config = json.loads(source_config.read_text(encoding="utf-8"))
    rewrite_real_terrain_paths(run_config)
    command_args = dict(run_config.get("command_args", {}))
    command_args.update(LDAC_RECOVERY_DEFAULTS)
    command_args.update(
        {
            "algo": "ppo",
            "output_dir": str(destination),
            "python": str(python_exe),
            "device": device,
            "clean_output": False,
            "dry_run": False,
            "quick": False,
            "seed": int(seed),
            "recovery_timesteps": int(recovery_timesteps),
            "eval_interval": int(eval_interval),
            "num_eval_episodes": int(num_eval_episodes),
        }
    )

    run_config["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_config["protocol"] = "ppo_ldac_recovery_only_from_vanilla_ppo_nominal"
    run_config["algorithm"] = "ppo_ldac_recovery"
    run_config["command_args"] = command_args
    run_config["source_baseline_seed_dir"] = str(source)
    run_config["source_baseline_nominal_checkpoint"] = str(source_checkpoint)
    run_config["recovery_only_from_baseline_nominal"] = True
    run_config["ldac_recovery_config"] = dict(LDAC_RECOVERY_DEFAULTS)
    (destination / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    return destination


def main() -> int:
    args = parse_args()
    baseline_root = resolve(args.baseline_root)
    output_root = resolve(args.output_root)
    python_exe = resolve(args.python)
    if not python_exe.exists():
        raise FileNotFoundError(f"missing Python executable: {python_exe}")

    prepared: list[Path] = []
    for scenario in args.scenarios:
        prepared.append(
            prepare_one(
                scenario=scenario,
                baseline_root=baseline_root,
                output_root=output_root,
                seed=int(args.seed),
                python_exe=python_exe,
                device=str(args.device),
                recovery_timesteps=int(args.recovery_timesteps),
                eval_interval=int(args.eval_interval),
                num_eval_episodes=int(args.num_eval_episodes),
                force=bool(args.force),
            )
        )

    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "baseline_root": str(baseline_root),
        "output_root": str(output_root),
        "python": str(python_exe),
        "device": str(args.device),
        "recovery_timesteps": int(args.recovery_timesteps),
        "eval_interval": int(args.eval_interval),
        "num_eval_episodes": int(args.num_eval_episodes),
        "scenarios": list(args.scenarios),
        "seed_dirs": [str(path) for path in prepared],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "prepare_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(prepared)} PPO-LDAC recovery seed dirs under {output_root}")
    print(f"Manifest: {manifest_path}")
    for path in prepared:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
