"""Sequential recovery experiment across environment, observation, and combined attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.attack_wrappers import attack_enabled, load_attack_config
from run_attack_recovery_finetune import (
    build_train_command,
    checkpoint_step,
    clean_output_dir,
    config_value,
    evaluate_checkpoint,
    generate_episodes,
    load_base_args,
    load_environment_attack,
    require_existing_file,
)
from utils.metrics import DEFAULT_MAP_SEED_POOL_SIZE


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_CONFIG = PROJECT_ROOT / "configs" / "ppo_lunar_corridor_relative_reward.json"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "runs"
    / "corridor_main_strength3"
    / "lunar_rover_corridor"
    / "seed0"
    / "nominal_ppo"
    / "checkpoint.pt"
)
DEFAULT_ATTACK_CONFIG = (
    PROJECT_ROOT
    / "runs"
    / "corridor_main_strength3"
    / "lunar_rover_corridor"
    / "seed0"
    / "configs"
    / "path_corridor_attack.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "corridor_strength3_staged_recovery"
DEFAULT_OBSERVATION_ATTACK = {
    "enabled": True,
    "type": "obs_dropout",
    "dropout_prob": 0.25,
    "fill_value": 0.0,
    "clip_to_observation_space": True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential attack recovery stages.")
    parser.add_argument("--base-config", type=str, default=str(DEFAULT_BASE_CONFIG))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--attack-config", type=str, default=str(DEFAULT_ATTACK_CONFIG))
    parser.add_argument("--observation-attack-config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--stage-timesteps", type=int, default=20480)
    parser.add_argument("--eval-interval", type=int, default=1024)
    parser.add_argument("--num-eval-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--in-domain-seed", type=int, default=909)
    parser.add_argument("--heldout-seed", type=int, default=1919)
    parser.add_argument("--map-pool-size", type=int, default=None)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def disabled_attack() -> dict[str, Any]:
    return {"enabled": False}


def load_observation_attack(path_text: str | None) -> dict[str, Any]:
    if path_text is None:
        return dict(DEFAULT_OBSERVATION_ATTACK)
    data = load_attack_config(path_text)
    if "observation_attack" in data:
        data = dict(data["observation_attack"])
    data.setdefault("enabled", True)
    return data


def stage_definitions(env_attack: dict[str, Any], obs_attack: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage_index": 1,
            "stage": "environment_recovery",
            "active_attack": "environment",
            "env_attack": env_attack,
            "obs_attack": disabled_attack(),
        },
        {
            "stage_index": 2,
            "stage": "observation_recovery",
            "active_attack": "observation",
            "env_attack": disabled_attack(),
            "obs_attack": obs_attack,
        },
        {
            "stage_index": 3,
            "stage": "combined_recovery",
            "active_attack": "combined",
            "env_attack": env_attack,
            "obs_attack": obs_attack,
        },
    ]


def write_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    base_args: dict[str, Any],
    env_attack: dict[str, Any],
    obs_attack: dict[str, Any],
    seed: int,
    map_size: int,
    scenario: str,
    map_pool_size: int,
) -> None:
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command_args": vars(args),
        "resolved": {
            "seed": int(seed),
            "map_size": int(map_size),
            "scenario": scenario,
            "map_pool_size": int(map_pool_size),
            "stage_timesteps": int(args.stage_timesteps),
            "eval_interval": int(args.eval_interval),
        },
        "base_config_args": base_args,
        "environment_attack": env_attack,
        "observation_attack": obs_attack,
        "stages": [
            {
                "stage_index": stage["stage_index"],
                "stage": stage["stage"],
                "active_attack": stage["active_attack"],
            }
            for stage in stage_definitions(env_attack, obs_attack)
        ],
    }
    (output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def append_stage_columns(
    rows: list[dict[str, Any]],
    stage: dict[str, Any],
    stage_step: int,
    cumulative_step: int,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        enriched = dict(row)
        enriched["stage_index"] = int(stage["stage_index"])
        enriched["stage"] = str(stage["stage"])
        enriched["active_attack"] = str(stage["active_attack"])
        enriched["stage_step"] = int(stage_step)
        enriched["cumulative_step"] = int(cumulative_step)
        output.append(enriched)
    return output


def save_checkpoint_copy(source: Path, target: Path, dry_run: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"Would copy checkpoint {source} -> {target}")
        return
    shutil.copy2(source, target)


def plot_staged(frame: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    active = frame[frame["attack_type"] == frame["active_attack"]].copy()
    active = active.sort_values(["eval_domain", "cumulative_step"])
    no_attack_step0 = frame[(frame["attack_type"] == "none") & (frame["cumulative_step"] == 0)]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for domain, group in active.groupby("eval_domain"):
        ax.plot(
            group["cumulative_step"],
            group["mean_attacked_scalar_cost"],
            marker="o",
            linewidth=2.0,
            label=domain,
        )
    for stage_name, group in active.groupby("stage"):
        x0 = float(group["cumulative_step"].min())
        ax.axvline(x0, color="0.7", linestyle="--", linewidth=1)
        ax.text(x0, ax.get_ylim()[1], stage_name.replace("_", "\n"), fontsize=8, va="top")
    ax.set_xlabel("cumulative fine-tuning step")
    ax.set_ylabel("active attack mean cost (lower is better)")
    ax.set_title("Sequential attack recovery - active attack cost")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_staged_active_attack_cost.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for domain, group in active.groupby("eval_domain"):
        baseline = no_attack_step0[no_attack_step0["eval_domain"] == domain]
        if baseline.empty:
            continue
        base_cost = float(baseline.iloc[0]["mean_attacked_scalar_cost"])
        performance = base_cost / np.maximum(np.abs(group["mean_attacked_scalar_cost"].to_numpy()), 1e-8)
        ax.plot(group["cumulative_step"], performance, marker="o", linewidth=2.0, label=domain)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    for stage_name, group in active.groupby("stage"):
        x0 = float(group["cumulative_step"].min())
        ax.axvline(x0, color="0.7", linestyle="--", linewidth=1)
        ax.text(x0, ax.get_ylim()[1], stage_name.replace("_", "\n"), fontsize=8, va="top")
    ax.set_xlabel("cumulative fine-tuning step")
    ax.set_ylabel("normalized performance vs initial no attack")
    ax.set_title("Sequential attack recovery - normalized active performance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_staged_active_attack_performance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for (domain, attack_type), group in frame.groupby(["eval_domain", "attack_type"]):
        group = group.sort_values("cumulative_step")
        ax.plot(
            group["cumulative_step"],
            group["mean_attacked_scalar_cost"],
            marker="o",
            linewidth=1.4,
            label=f"{domain}:{attack_type}",
        )
    ax.set_xlabel("cumulative fine-tuning step")
    ax.set_ylabel("mean attacked scalar cost")
    ax.set_title("Sequential recovery - all evaluated attacks")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_staged_all_attack_cost.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for (domain, attack_type), group in frame.groupby(["eval_domain", "attack_type"]):
        group = group.sort_values("cumulative_step")
        ax.plot(
            group["cumulative_step"],
            group["relative_degradation"],
            marker="o",
            linewidth=1.4,
            label=f"{domain}:{attack_type}",
        )
    ax.set_xlabel("cumulative fine-tuning step")
    ax.set_ylabel("relative degradation")
    ax.set_title("Sequential recovery - degradation by attack")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_staged_degradation_by_attack.png", dpi=180)
    plt.close(fig)

    if "mean_attacked_cell_exposure_ratio" in frame:
        env_like = frame[frame["attack_type"].isin(["environment", "combined"])].copy()
        if not env_like.empty:
            fig, ax = plt.subplots(figsize=(11, 5.2))
            for (domain, attack_type), group in env_like.groupby(["eval_domain", "attack_type"]):
                group = group.sort_values("cumulative_step")
                ax.plot(
                    group["cumulative_step"],
                    group["mean_attacked_cell_exposure_ratio"],
                    marker="o",
                    linewidth=1.6,
                    label=f"{domain}:{attack_type}",
                )
            ax.set_xlabel("cumulative fine-tuning step")
            ax.set_ylabel("attacked cell exposure ratio")
            ax.set_title("Sequential recovery - attacked corridor exposure")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figures_dir / "fig_staged_attacked_exposure.png", dpi=180)
            plt.close(fig)


def write_output_guide(output_dir: Path) -> None:
    guide = """# Staged Attack Recovery Outputs

This experiment sequentially fine-tunes one policy under:

1. environment attack only,
2. observation attack only,
3. combined environment + observation attack.

## Main CSV

- `staged_recovery_curve.csv`: one row per checkpoint, eval domain, and evaluated attack.
- `stage`: active training stage.
- `active_attack`: attack used for training in that stage.
- `attack_type`: attack used for evaluation. Each checkpoint is evaluated on `none`, `environment`, `observation`, and `combined`.
- `stage_step`: steps within the current stage.
- `cumulative_step`: total fine-tuning steps across stages.
- `mean_attacked_scalar_cost`: main metric. Lower is better.
- `relative_degradation`: cost increase relative to the clean policy on the clean map.
- `mean_belief_scalar_cost`: planner-visible belief-map cost for the selected path.
- `mean_map_mismatch_penalty`: true-map cost minus belief-map cost. Larger positive values mean the map underestimated risk.
- `mean_map_mismatch_abs_error`: absolute true-vs-belief scalar cost gap.
- `mean_path_confidence`: average confidence along the selected path.
- `true_belief_mismatch_rate`: fraction of evaluated episodes using a separated true/belief map.
- `mean_mismatched_cells`: number of free-map cells with belief mismatch in the generated map attack.
- `mean_belief_abs_error`: average per-cell belief error magnitude from the map attack metadata.

## Figures

- `figures/fig_staged_active_attack_performance.png`: main narrative curve. It shows drop/recovery for the attack active in each stage.
- `figures/fig_staged_active_attack_cost.png`: same narrative in raw cost.
- `figures/fig_staged_all_attack_cost.png`: every evaluated attack at every checkpoint.
- `figures/fig_staged_degradation_by_attack.png`: relative degradation for every evaluated attack.
- `figures/fig_staged_attacked_exposure.png`: exposure for environment/combined attacks.
"""
    (output_dir / "OUTPUT_GUIDE.md").write_text(guide, encoding="utf-8")


def write_report(frame: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Staged Attack Recovery Report", ""]
    active = frame[frame["attack_type"] == frame["active_attack"]].copy()
    for (domain, stage), group in active.groupby(["eval_domain", "stage"]):
        group = group.sort_values("stage_step")
        first = group.iloc[0]
        final = group.iloc[-1]
        improvement = float(first["mean_attacked_scalar_cost"] - final["mean_attacked_scalar_cost"])
        rel = improvement / (abs(float(first["mean_attacked_scalar_cost"])) + 1e-8)
        lines.extend(
            [
                f"## {domain} / {stage}",
                "",
                f"- Active attack: `{first['active_attack']}`",
                f"- Stage step 0 cost: `{float(first['mean_attacked_scalar_cost']):.4f}`",
                f"- Stage final cost: `{float(final['mean_attacked_scalar_cost']):.4f}`",
                f"- Improvement: `{improvement:.4f}` (`{100.0 * rel:.2f}%`)",
                f"- Step 0 degradation: `{float(first['relative_degradation']):.4f}`",
                f"- Final degradation: `{float(final['relative_degradation']):.4f}`",
                "",
            ]
        )
    (output_dir / "staged_recovery_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.quick:
        args.stage_timesteps = 2048
        args.eval_interval = 512
        args.num_eval_episodes = 20
        if Path(args.output_dir) == DEFAULT_OUTPUT_DIR:
            args.output_dir = str(PROJECT_ROOT / "runs" / "debug_staged_attack_recovery")

    require_existing_file(args.base_config, "base config")
    require_existing_file(args.checkpoint, "checkpoint")
    require_existing_file(args.attack_config, "environment attack config")
    require_existing_file(args.observation_attack_config, "observation attack config")

    output_dir = Path(args.output_dir)
    if args.clean_output:
        clean_output_dir(output_dir, args.dry_run)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    base_args = load_base_args(args.base_config)
    env_attack = load_environment_attack(args.attack_config)
    obs_attack = load_observation_attack(args.observation_attack_config)
    if not attack_enabled(obs_attack):
        raise ValueError("observation attack must be enabled for staged recovery")

    seed = int(args.seed if args.seed is not None else config_value(base_args, "seed", 0))
    map_size = int(config_value(base_args, "map-size", 48))
    scenario = str(config_value(base_args, "scenario", "lunar_rover"))
    map_pool_size = int(
        args.map_pool_size
        if args.map_pool_size is not None
        else config_value(base_args, "map-seed-pool-size", DEFAULT_MAP_SEED_POOL_SIZE)
    )
    write_run_config(output_dir, args, base_args, env_attack, obs_attack, seed, map_size, scenario, map_pool_size)

    episodes_by_domain = {
        f"in_domain_seed{args.in_domain_seed}": (
            int(args.in_domain_seed),
            generate_episodes(args.num_eval_episodes, seed + 222, map_size, scenario, args.in_domain_seed, map_pool_size),
        ),
        f"heldout_seed{args.heldout_seed}": (
            int(args.heldout_seed),
            generate_episodes(args.num_eval_episodes, seed + 222, map_size, scenario, args.heldout_seed, map_pool_size),
        ),
    }

    rows: list[dict[str, Any]] = []
    current_checkpoint = Path(args.checkpoint)
    cumulative_step = 0
    chunk_index = 0

    for stage in stage_definitions(env_attack, obs_attack):
        stage_step = 0
        stage_checkpoint = checkpoints_dir / (
            f"checkpoint_stage{stage['stage_index']:02d}_{stage['active_attack']}_step_{stage_step:05d}.pt"
        )
        save_checkpoint_copy(current_checkpoint, stage_checkpoint, args.dry_run)
        eval_rows = evaluate_checkpoint(
            stage_checkpoint if stage_checkpoint.exists() else current_checkpoint,
            cumulative_step,
            episodes_by_domain,
            map_size,
            env_attack,
            obs_attack,
            seed,
        )
        rows.extend(append_stage_columns(eval_rows, stage, stage_step, cumulative_step))

        while stage_step < int(args.stage_timesteps):
            chunk_index += 1
            chunk_timesteps = min(int(args.eval_interval), int(args.stage_timesteps) - stage_step)
            command, chunk_final = build_train_command(
                args.python,
                base_args,
                current_checkpoint,
                stage["env_attack"],
                stage["obs_attack"],
                output_dir,
                chunk_index,
                chunk_timesteps,
                seed + 10_000 * int(stage["stage_index"]) + chunk_index,
            )
            print(" ".join(str(part) for part in command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
                if not chunk_final.exists():
                    raise FileNotFoundError(f"expected chunk checkpoint not found: {chunk_final}")
                actual_step = checkpoint_step(chunk_final)
            else:
                actual_step = chunk_timesteps

            stage_step += actual_step if actual_step > 0 else chunk_timesteps
            cumulative_step += actual_step if actual_step > 0 else chunk_timesteps
            if args.dry_run:
                continue
            current_checkpoint = checkpoints_dir / (
                f"checkpoint_stage{stage['stage_index']:02d}_{stage['active_attack']}_step_{stage_step:05d}.pt"
            )
            save_checkpoint_copy(chunk_final if not args.dry_run else stage_checkpoint, current_checkpoint, args.dry_run)
            eval_rows = evaluate_checkpoint(
                current_checkpoint if current_checkpoint.exists() else Path(args.checkpoint),
                cumulative_step,
                episodes_by_domain,
                map_size,
                env_attack,
                obs_attack,
                seed,
            )
            rows.extend(append_stage_columns(eval_rows, stage, stage_step, cumulative_step))

    frame = pd.DataFrame(rows)
    csv_path = output_dir / "staged_recovery_curve.csv"
    if not args.dry_run:
        frame.to_csv(csv_path, index=False)
        plot_staged(frame, output_dir)
        write_output_guide(output_dir)
        write_report(frame, output_dir)

    print(f"Saved staged recovery CSV to {csv_path}")
    print(f"Saved figures to {output_dir / 'figures'}")
    print(f"Saved checkpoints to {checkpoints_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
