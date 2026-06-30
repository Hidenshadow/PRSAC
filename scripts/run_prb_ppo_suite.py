#!/usr/bin/env python
"""Run PRB-PPO recovery across existing PPO shock-recovery seed runs."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_EXPERIMENTS = (
    "level2_hard_shock_recovery_5seeds",
    "level3_easy_shock_recovery_5seeds",
    "level3_hard_shock_recovery_5seeds",
)

PRESETS = {
    "prb_mlp_aux005": {
        "prb_encoder_type": "mlp",
        "prb_latent_dim": 64,
        "prb_hidden_dim": 64,
        "prb_aux_coef": 0.05,
        "recovery_timesteps": 3072,
        "chunk_steps": 1024,
        "eval_episodes": 32,
        "validation_episodes": 32,
    },
    "prb_mlp_aux010": {
        "prb_encoder_type": "mlp",
        "prb_latent_dim": 64,
        "prb_hidden_dim": 64,
        "prb_aux_coef": 0.10,
        "recovery_timesteps": 3072,
        "chunk_steps": 1024,
        "eval_episodes": 32,
        "validation_episodes": 32,
    },
    "prb_mlp_no_aux_control": {
        "prb_encoder_type": "mlp",
        "prb_latent_dim": 64,
        "prb_hidden_dim": 64,
        "prb_aux_coef": 0.0,
        "recovery_timesteps": 3072,
        "chunk_steps": 1024,
        "eval_episodes": 32,
        "validation_episodes": 32,
    },
    "prb_gru_aux005": {
        "prb_encoder_type": "gru",
        "prb_latent_dim": 64,
        "prb_hidden_dim": 64,
        "prb_aux_coef": 0.05,
        "recovery_timesteps": 3072,
        "chunk_steps": 1024,
        "eval_episodes": 32,
        "validation_episodes": 32,
    },
    "ppo_control_same_runner": {
        "prb_encoder_type": "mlp",
        "prb_latent_dim": 64,
        "prb_hidden_dim": 64,
        "prb_aux_coef": 0.0,
        "prb_stopgrad_residual_latent": True,
        "prb_use_random_residual_features": False,
        "recovery_timesteps": 3072,
        "chunk_steps": 1024,
        "eval_episodes": 32,
        "validation_episodes": 32,
    },
}


def explicit_cli_dests(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest
    explicit: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        dest = option_to_dest.get(option)
        if dest:
            explicit.add(dest)
    return explicit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "runs" / "rl_baselines" / "ppo")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "prb_ppo_quick")
    parser.add_argument("--suite", choices=("core", "all", "custom"), default="core")
    parser.add_argument("--experiments", type=str, default="")
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--preset", choices=tuple(PRESETS), default=None)
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--validation-episodes", type=int, default=64)
    parser.add_argument("--recovery-timesteps", type=int, default=20480)
    parser.add_argument("--chunk-steps", type=int, default=1024)
    parser.add_argument("--eval-interval", type=int, default=2048)
    parser.add_argument("--chunk-eval-episodes", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--prb-encoder-type", choices=("mlp", "gru"), default="mlp")
    parser.add_argument("--prb-latent-dim", type=int, default=64)
    parser.add_argument("--prb-hidden-dim", type=int, default=64)
    parser.add_argument("--prb-aux-coef", type=float, default=0.05)
    parser.add_argument("--prb-disable-aux-loss", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prb-use-component-costs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prb-use-scalar-cost", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prb-normalize-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prb-feature-clip", type=float, default=5.0)
    parser.add_argument("--prb-lambda-residual-total", type=float, default=1.0)
    parser.add_argument("--prb-lambda-true-total", type=float, default=0.5)
    parser.add_argument("--prb-lambda-component-residual", type=float, default=1.0)
    parser.add_argument("--prb-lambda-true-component", type=float, default=0.5)
    parser.add_argument("--prb-stopgrad-residual-latent", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prb-use-random-residual-features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    explicit_dests = explicit_cli_dests(parser, sys.argv[1:])
    if args.preset:
        for key, value in PRESETS[str(args.preset)].items():
            if key not in explicit_dests:
                setattr(args, key, value)
    return args


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def parse_seed_csv(text: str) -> list[int]:
    return [int(item) for item in parse_csv(text)]


def selected_experiments(args: argparse.Namespace) -> list[str]:
    if args.suite == "core":
        return list(CORE_EXPERIMENTS)
    if args.suite == "custom":
        experiments = parse_csv(args.experiments)
        if not experiments:
            raise SystemExit("--experiments is required when --suite custom")
        return experiments
    source_root = args.source_root if args.source_root.is_absolute() else PROJECT_ROOT / args.source_root
    return sorted(path.name for path in source_root.glob("level*_shock_recovery_5seeds") if path.is_dir())


def source_runs(args: argparse.Namespace) -> list[Path]:
    source_root = args.source_root if args.source_root.is_absolute() else PROJECT_ROOT / args.source_root
    runs = []
    for experiment in selected_experiments(args):
        for seed in parse_seed_csv(args.seeds):
            path = source_root / experiment / f"seed{seed}"
            if not (path / "run_config.json").exists():
                print(f"skip missing source: {path}", flush=True)
                continue
            runs.append(path)
    return runs


def run_command(command: list[str], args: argparse.Namespace) -> None:
    print(" ".join(str(part) for part in command), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def run_target(source_run_dir: Path, args: argparse.Namespace) -> bool:
    output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    experiment = source_run_dir.parent.name
    seed_name = source_run_dir.name
    preset_name = args.preset or "custom"
    output_dir = output_root / preset_name / experiment / seed_name
    summary_path = output_dir / "shock_recovery_summary.csv"
    if args.only_missing and summary_path.exists():
        print(f"skip existing prb_ppo: {summary_path}", flush=True)
        return False
    command = [
        args.python,
        "run_prb_ppo_recovery_experiment.py",
        "--source-run-dir", str(source_run_dir),
        "--output-dir", str(output_dir),
        "--eval-episodes", str(args.eval_episodes),
        "--validation-episodes", str(args.validation_episodes),
        "--recovery-timesteps", str(args.recovery_timesteps),
        "--chunk-steps", str(args.chunk_steps),
        "--eval-interval", str(args.eval_interval),
        "--chunk-eval-episodes", str(args.chunk_eval_episodes),
        "--learning-rate", str(args.learning_rate),
        "--update-epochs", str(args.update_epochs),
        "--num-minibatches", str(args.num_minibatches),
        "--prb-encoder-type", str(args.prb_encoder_type),
        "--prb-latent-dim", str(args.prb_latent_dim),
        "--prb-hidden-dim", str(args.prb_hidden_dim),
        "--prb-aux-coef", str(args.prb_aux_coef),
        "--prb-feature-clip", str(args.prb_feature_clip),
        "--prb-lambda-residual-total", str(args.prb_lambda_residual_total),
        "--prb-lambda-true-total", str(args.prb_lambda_true_total),
        "--prb-lambda-component-residual", str(args.prb_lambda_component_residual),
        "--prb-lambda-true-component", str(args.prb_lambda_true_component),
        "--device", str(args.device),
    ]
    command.append("--prb-disable-aux-loss" if args.prb_disable_aux_loss else "--no-prb-disable-aux-loss")
    command.append("--prb-use-component-costs" if args.prb_use_component_costs else "--no-prb-use-component-costs")
    command.append("--prb-use-scalar-cost" if args.prb_use_scalar_cost else "--no-prb-use-scalar-cost")
    command.append("--prb-normalize-features" if args.prb_normalize_features else "--no-prb-normalize-features")
    command.append("--prb-stopgrad-residual-latent" if args.prb_stopgrad_residual_latent else "--no-prb-stopgrad-residual-latent")
    command.append("--prb-use-random-residual-features" if args.prb_use_random_residual_features else "--no-prb-use-random-residual-features")
    run_command(command, args)
    return True


def main() -> int:
    args = parse_args()
    args.source_root = args.source_root if args.source_root.is_absolute() else PROJECT_ROOT / args.source_root
    args.output_root = args.output_root if args.output_root.is_absolute() else PROJECT_ROOT / args.output_root
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs_started = 0
    for source_run_dir in source_runs(args):
        jobs_started += int(run_target(source_run_dir, args))
        if args.max_jobs and jobs_started >= args.max_jobs:
            print(f"stopped after max jobs: {args.max_jobs}", flush=True)
            return 0
    print(f"PRB-PPO suite complete; jobs_started={jobs_started}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
