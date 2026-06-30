"""Continue an interrupted shock-recovery run from saved recovery checkpoints.

This helper is intentionally narrow: it does not rerun nominal training and it
does not clean the output directory. It reads the existing run_config.json,
finds the last complete checkpoint_recovery_step_*.pt, finishes the remaining
recovery chunks, then rebuilds the protocol curve/summary from checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from envs.attack_wrappers import attack_enabled
from run_attack_recovery_finetune import build_train_command, checkpoint_step, config_value
from run_shock_recovery_experiment import (
    add_protocol_columns,
    append_attack_bandit_history,
    attack_bandit_rollout_metrics_path,
    attack_probability_benchmark_floor,
    build_game_recovery_attack,
    build_real_eval_episodes,
    build_synthetic_eval_episodes,
    copy_checkpoint,
    disabled_attack,
    enforce_benchmark_floor,
    evaluate_checkpoint,
    game_attack_variants,
    game_recovery_train_args,
    initial_game_attack_probs,
    is_real_level,
    plot_outputs,
    split_paths,
    summarize_recovery,
    update_attack_bandit_probs,
    write_output_guide,
)
from run_lunar_viper_staged_recovery import load_environment_attack, read_json
from utils.metrics import DEFAULT_MAP_SEED_POOL_SIZE


CHECKPOINT_RE = re.compile(r"checkpoint_recovery_step_(\d+)\.pt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume only the tail of a shock-recovery run.")
    parser.add_argument("--seed-dir", type=Path, required=True, help="Existing seed output directory.")
    parser.add_argument("--target-step", type=int, default=None, help="Recovery step to finish; defaults to run_config.")
    parser.add_argument("--python", type=str, default=None, help="Python executable for child train chunks.")
    parser.add_argument("--device", type=str, default=None, help="Optional device override for remaining chunks.")
    parser.add_argument("--skip-rebuild-curve", action="store_true", help="Only train missing chunks.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def namespace_from_run_config(run_config: dict[str, Any], seed_dir: Path) -> argparse.Namespace:
    command_args = dict(run_config.get("command_args", {}))
    command_args.setdefault("algo", run_config.get("algorithm", "ppo"))
    command_args.setdefault("output_dir", str(seed_dir))
    command_args.setdefault("recovery_timesteps", 20_480)
    command_args.setdefault("eval_interval", 1024)
    command_args.setdefault("num_eval_episodes", 300)
    command_args.setdefault("train_eval_episodes", 64)
    command_args.setdefault("seed", 0)
    command_args.setdefault("in_domain_seed", 909)
    command_args.setdefault("heldout_seed", 1919)
    command_args.setdefault("map_pool_size", None)
    command_args.setdefault("quick", False)
    command_args.setdefault("dry_run", False)
    command_args.setdefault("game_recovery_enabled", False)
    command_args.setdefault("game_attack_sampler", "adaptive_bandit")
    command_args.setdefault("game_bandit_benchmark_floor", 0.0)
    command_args.setdefault("qre_minimax_recovery_enabled", False)
    command_args.setdefault("qre_benchmark_floor", 0.2)
    command_args.setdefault("cdr_recovery_enabled", False)
    command_args.setdefault("cdr_attack_mixture_size", 5)
    command_args.setdefault("cdr_attack_jitter_start", 0.05)
    command_args.setdefault("cdr_attack_jitter_end", 0.25)
    command_args.setdefault("cdr_attack_variant_mode", "scale")
    command_args.setdefault("cdr_benchmark_prob_start", 0.70)
    command_args.setdefault("cdr_benchmark_prob_end", 0.20)
    command_args.setdefault("cdr_schedule", "linear")
    command_args.setdefault("valt_sac_recovery_enabled", False)
    command_args.setdefault("valt_sac_eps_start", 0.0)
    command_args.setdefault("valt_sac_eps_end", 0.08)
    command_args.setdefault("valt_sac_kappa_start", 0.0)
    command_args.setdefault("valt_sac_kappa_end", 0.30)
    command_args.setdefault("valt_sac_schedule_steps", 20_480)
    command_args.setdefault("valt_sac_schedule", "linear")
    command_args.setdefault("valt_sac_bound_iters", 2)
    command_args.setdefault("valt_sac_worst_step_size", 0.0)
    command_args.setdefault("valt_sac_sgld_noise", 0.0)
    command_args.setdefault("valt_sac_policy_reg_coef", 1.0)
    command_args.setdefault("valt_sac_random_start", True)
    command_args.setdefault("valt_sac_clip_low", 0.0)
    command_args.setdefault("valt_sac_clip_high", 1.0)
    command_args.setdefault("valt_sac_attack_deterministic", False)
    command_args.setdefault("sac_recovery_deterministic_actor_update", False)
    command_args.setdefault("sac_recovery_target_entropy_scale", 1.0)
    command_args.setdefault("sac_recovery_fixed_alpha", None)
    command_args.setdefault("sac_recovery_rollout_deterministic_prob", 0.0)
    command_args.setdefault("sac_recovery_rollout_noise_std", 0.0)
    command_args.setdefault("sac_recovery_log_std_penalty_coef", 0.0)
    command_args.setdefault("sac_recovery_log_std_target", -1.5)
    args = argparse.Namespace(**command_args)
    args.output_dir = seed_dir
    args.level_config = resolve_project_path(getattr(args, "level_config", None))
    args.base_config = resolve_project_path(getattr(args, "base_config", None))
    args.clean_output = False
    args.dry_run = False
    return args


def recovery_checkpoints(checkpoints_dir: Path) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    if not checkpoints_dir.exists():
        return checkpoints
    for path in checkpoints_dir.glob("checkpoint_recovery_step_*.pt"):
        match = CHECKPOINT_RE.match(path.name)
        if match:
            checkpoints[int(match.group(1))] = path
    return checkpoints


def latest_recovery_step(checkpoints_dir: Path, target_step: int) -> int:
    steps = [step for step in recovery_checkpoints(checkpoints_dir) if 0 < step <= target_step]
    return max(steps, default=0)


def step_checkpoint(checkpoints_dir: Path, step: int) -> Path:
    return checkpoints_dir / f"checkpoint_recovery_step_{int(step):05d}.pt"


def load_existing_curve(path: Path, existing_steps: set[int]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty or "recovery_step" not in frame.columns:
        return pd.DataFrame()
    frame["recovery_step"] = pd.to_numeric(frame["recovery_step"], errors="coerce").fillna(-1).astype(int)
    valid_steps = set(existing_steps)
    valid_steps.add(0)
    return frame[frame["recovery_step"].isin(valid_steps)].copy()


def prepare_eval_context(args: argparse.Namespace, run_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, tuple[int, list[Any]]]]:
    if args.level_config is None:
        raise ValueError("run_config is missing level_config")
    level_config = read_json(args.level_config)
    env_attack = dict(run_config.get("environment_attack") or load_environment_attack(level_config))
    base_args = dict(run_config.get("base_config_args") or {})
    if not base_args:
        raise ValueError("run_config is missing base_config_args; refusing to reconstruct a possibly different run")
    if getattr(args, "device", None):
        base_args["device"] = str(args.device)

    if is_real_level(level_config):
        split_data = run_config.get("splits") or {}
        splits = {key: Path(value) for key, value in split_data.items()} if split_data else split_paths(args.output_dir)
        map_size, episodes_by_domain = build_real_eval_episodes(
            level_config,
            splits,
            int(args.seed),
            int(args.num_eval_episodes),
        )
    else:
        map_pool_size = int(
            args.map_pool_size
            if args.map_pool_size is not None
            else config_value(base_args, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE)
        )
        map_size, episodes_by_domain = build_synthetic_eval_episodes(args, base_args, map_pool_size)
    return level_config, env_attack, map_size, episodes_by_domain


def rebuild_curve(
    args: argparse.Namespace,
    env_attack: dict[str, Any],
    map_size: int,
    episodes_by_domain: dict[str, tuple[int, list[Any]]],
    checkpoints_dir: Path,
    target_step: int,
) -> pd.DataFrame:
    checkpoint_map = recovery_checkpoints(checkpoints_dir)
    wanted_steps = sorted(step for step in checkpoint_map if 0 < step <= target_step)
    curve_path = args.output_dir / "shock_recovery_curve.csv"
    existing = load_existing_curve(curve_path, set(wanted_steps))
    rows: list[dict[str, Any]] = []
    have_steps: set[int] = set()
    if not existing.empty:
        rows.extend(existing.to_dict("records"))
        have_steps = set(existing["recovery_step"].astype(int).unique().tolist())

    nominal_checkpoint = checkpoints_dir / "checkpoint_nominal.pt"
    if 0 not in have_steps:
        print(f"Evaluating shock rows from {nominal_checkpoint}", flush=True)
        rows.extend(
            add_protocol_columns(
                evaluate_checkpoint(
                    nominal_checkpoint,
                    0,
                    episodes_by_domain,
                    map_size,
                    env_attack,
                    disabled_attack(),
                    int(args.seed),
                ),
                phase="shock",
                recovery_step=0,
                checkpoint_role="nominal",
            )
        )

    for step in wanted_steps:
        if step in have_steps:
            continue
        checkpoint = checkpoint_map[step]
        print(f"Evaluating recovery step {step} from {checkpoint}", flush=True)
        rows.extend(
            add_protocol_columns(
                evaluate_checkpoint(
                    checkpoint,
                    int(step),
                    episodes_by_domain,
                    map_size,
                    env_attack,
                    disabled_attack(),
                    int(args.seed),
                ),
                phase="recovery",
                recovery_step=int(step),
                checkpoint_role="recovery",
            )
        )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_columns = [column for column in ["recovery_step", "phase", "eval_domain", "attack_type"] if column in frame.columns]
        if sort_columns:
            frame = frame.sort_values(sort_columns).reset_index(drop=True)
    frame.to_csv(curve_path, index=False)
    summarize_recovery(frame, args.output_dir)
    plot_outputs(frame, args.output_dir)
    write_output_guide(args.output_dir)
    return frame


def continue_tail(cli_args: argparse.Namespace) -> int:
    seed_dir = cli_args.seed_dir.resolve()
    run_config_path = seed_dir / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"missing run_config.json: {run_config_path}")

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    args = namespace_from_run_config(run_config, seed_dir)
    if cli_args.python:
        args.python = cli_args.python
    if cli_args.device:
        args.device = cli_args.device

    checkpoints_dir = seed_dir / "checkpoints"
    nominal_checkpoint = checkpoints_dir / "checkpoint_nominal.pt"
    if not nominal_checkpoint.exists():
        raise FileNotFoundError(f"missing nominal checkpoint: {nominal_checkpoint}")

    target_step = int(cli_args.target_step or args.recovery_timesteps)
    eval_interval = int(args.eval_interval)
    current_step = latest_recovery_step(checkpoints_dir, target_step)
    current_checkpoint = step_checkpoint(checkpoints_dir, current_step) if current_step > 0 else nominal_checkpoint
    print(f"Seed dir: {seed_dir}", flush=True)
    print(f"Current recovery step: {current_step}; target: {target_step}", flush=True)

    _level_config, env_attack, map_size, episodes_by_domain = prepare_eval_context(args, run_config)
    if not attack_enabled(env_attack):
        raise ValueError("environment attack must be enabled for recovery continuation")

    base_args = dict(run_config["base_config_args"])
    if getattr(args, "device", None):
        base_args["device"] = str(args.device)

    game_variants = game_attack_variants(env_attack, args) if bool(args.game_recovery_enabled) else []
    attack_bandit_probs = enforce_benchmark_floor(
        initial_game_attack_probs(len(game_variants)),
        attack_probability_benchmark_floor(args),
    )
    use_adaptive_attack_sampler = (
        bool(args.game_recovery_enabled)
        and str(args.game_attack_sampler) in {"adaptive_bandit", "qre_minimax"}
        and len(game_variants) > 1
    )
    qre_state: dict[str, Any] = {}
    total_recovery_chunks = int(np.ceil(max(target_step, 1) / max(eval_interval, 1)))

    recovery_step = int(current_step)
    chunk_index = recovery_step // max(eval_interval, 1)
    while recovery_step < target_step:
        chunk_index += 1
        chunk_timesteps = min(eval_interval, target_step - recovery_step)
        recovery_train_attack = build_game_recovery_attack(
            env_attack,
            args,
            int(args.seed) + 20_000 + chunk_index,
            probs=attack_bandit_probs,
            recovery_step_offset=recovery_step,
            chunk_index=chunk_index,
            total_chunks=total_recovery_chunks,
        )
        rollout_metrics_path = (
            attack_bandit_rollout_metrics_path(seed_dir, chunk_index, str(args.game_attack_sampler))
            if use_adaptive_attack_sampler
            else None
        )
        command, chunk_final = build_train_command(
            str(args.python),
            base_args,
            current_checkpoint,
            recovery_train_attack,
            disabled_attack(),
            seed_dir,
            chunk_index,
            chunk_timesteps,
            int(args.seed) + 10_000 + chunk_index,
            str(args.algo),
            extra_train_args=game_recovery_train_args(
                args,
                nominal_checkpoint,
                rollout_metrics_path,
                recovery_step_offset=recovery_step,
                chunk_index=chunk_index,
            ),
        )
        print(" ".join(str(part) for part in command), flush=True)
        if not cli_args.dry_run:
            subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
            if not chunk_final.exists():
                raise FileNotFoundError(f"expected recovery checkpoint not found: {chunk_final}")
            actual_step = checkpoint_step(chunk_final)
            recovery_step += actual_step if actual_step > 0 else chunk_timesteps
            current_checkpoint = step_checkpoint(checkpoints_dir, recovery_step)
            copy_checkpoint(chunk_final, current_checkpoint, False)
        else:
            recovery_step += chunk_timesteps
            current_checkpoint = step_checkpoint(checkpoints_dir, recovery_step)

        if use_adaptive_attack_sampler and rollout_metrics_path is not None and not cli_args.dry_run:
            attack_bandit_probs, bandit_rows = update_attack_bandit_probs(
                attack_bandit_probs,
                game_variants,
                rollout_metrics_path,
                args,
                qre_state=qre_state,
                chunk_index=chunk_index,
                total_chunks=total_recovery_chunks,
            )
            append_attack_bandit_history(seed_dir, chunk_index, bandit_rows, str(args.game_attack_sampler))
        print(f"Completed recovery step {recovery_step}", flush=True)

    if not cli_args.skip_rebuild_curve and not cli_args.dry_run:
        frame = rebuild_curve(args, env_attack, map_size, episodes_by_domain, checkpoints_dir, target_step)
        print(f"Saved rebuilt curve with {len(frame)} rows to {seed_dir / 'shock_recovery_curve.csv'}", flush=True)

    return 0


def main() -> int:
    return continue_tail(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
