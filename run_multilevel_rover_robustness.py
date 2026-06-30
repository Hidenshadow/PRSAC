"""Unified entrypoint for multi-level planetary rover robustness benchmarks.

This script deliberately frames synthetic, lunar DEM/VIPER, and Mars DTM as
independent benchmark levels.  It does not treat real terrain as transfer
evaluation for a synthetic policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from envs.attack_wrappers import wrap_env_with_attacks
from envs.real_terrain_env import RealTerrainPlanningEnv
from maps.real_terrain import generate_task_splits
from utils.metrics import ACTION_DIM


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LEVEL_CONFIGS = {
    "synthetic": PROJECT_ROOT / "configs" / "levels" / "synthetic_corridor.json",
    "lunar_viper": PROJECT_ROOT / "configs" / "levels" / "lunar_viper.json",
    "mars_dtm": PROJECT_ROOT / "configs" / "levels" / "mars_dtm.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a multi-level rover robustness benchmark.")
    parser.add_argument("--level", choices=("synthetic", "lunar_viper", "mars_dtm"), required=True)
    parser.add_argument("--scenario", type=str, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--stage-timesteps", type=int, default=1024)
    parser.add_argument("--eval-interval", type=int, default=256)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def parse_seeds(seed_text: str) -> list[int]:
    values: list[int] = []
    for chunk in str(seed_text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            left, right = chunk.split("-", 1)
            start, end = int(left), int(right)
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
        else:
            values.append(int(chunk))
    return values or [0]


def clean_output_dir(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    project_root = PROJECT_ROOT.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"refusing to clean output outside project root: {resolved}") from exc
    if dry_run:
        print(f"Would remove output directory: {resolved}")
        return
    shutil.rmtree(resolved)


def resolve_config(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config or DEFAULT_LEVEL_CONFIGS[args.level]
    if not config_path.exists():
        raise FileNotFoundError(f"level config not found: {config_path}")
    config = read_json(config_path)
    config["_config_path"] = str(config_path)
    if args.scenario:
        config["scenario"] = args.scenario
    return config


def default_output_dir(level: str, scenario: str) -> Path:
    return PROJECT_ROOT / "runs" / "multilevel" / f"{level}_{scenario}"


def ensure_layout(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "splits": output_dir / "splits",
        "results": output_dir / "results",
        "figures": output_dir / "results" / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def environment_attack_name(config: dict[str, Any], key: str) -> str:
    attack = config.get("attacks", {}).get(key, {})
    if not attack or not attack.get("enabled", False):
        return "none"
    return str(attack.get("type", key))


def summarize_detail_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    summaries = []
    group_cols = ["level", "scenario", "seed", "eval_domain", "attack_type", "method"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        key_data = dict(zip(group_cols, keys))
        nominal = group["scalar_cost"].astype(float)
        attacked = group["attacked_scalar_cost"].astype(float)
        summary = {
            **key_data,
            "num_episodes": int(len(group)),
            "mean_nominal_scalar_cost": float(nominal.mean()),
            "mean_attacked_scalar_cost": float(attacked.mean()),
            "relative_degradation": float(
                (attacked.mean() - nominal.mean()) / (abs(nominal.mean()) + 1e-8)
            ),
            "success_rate": float(group["success"].astype(float).mean()),
            "mean_path_length": float(group["path_length"].astype(float).mean()),
            "mean_attacked_cell_exposure_ratio": float(
                group["attacked_cell_exposure_ratio"].astype(float).mean()
            ),
            "mean_lambda_uncertainty": float(group["lambda_uncertainty"].astype(float).mean()),
            "run_mode": str(group["run_mode"].iloc[0]),
            "training_status": str(group["training_status"].iloc[0]),
        }
        summaries.append(summary)
    return pd.DataFrame(summaries)


def evaluate_real_level_smoke(
    config: dict[str, Any],
    paths: dict[str, Path],
    seeds: list[int],
    num_episodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run deterministic reset/step checks on real-terrain train and heldout splits."""

    layers_path = PROJECT_ROOT / str(config["map_source"])
    metadata_path = PROJECT_ROOT / str(config.get("metadata", ""))
    if not layers_path.exists():
        report_missing_dataset(config, paths, layers_path)
        return pd.DataFrame(), pd.DataFrame()

    quick_limit = max(int(num_episodes), 1)
    train_count = max(int(config.get("num_train_tasks", 128)), quick_limit)
    validation_count = max(int(config.get("num_validation_tasks", 64)), min(quick_limit, 32))
    heldout_count = max(int(config.get("num_heldout_tasks", 128)), quick_limit)
    if quick_limit <= 20:
        train_count = min(train_count, 32)
        validation_count = min(validation_count, 16)
        heldout_count = min(heldout_count, 32)

    split_seed = int(seeds[0])
    splits = generate_task_splits(
        layers_path=layers_path,
        output_dir=paths["splits"],
        seed=split_seed,
        tile_id=str(config.get("tile_id", config["level"])),
        num_train_tasks=train_count,
        num_validation_tasks=validation_count,
        num_heldout_tasks=heldout_count,
        min_distance_ratio=float(config.get("min_distance_ratio", 0.62)),
        metadata_path=metadata_path if metadata_path.exists() else None,
        task_sampling_mode=str(config.get("task_sampling_mode", "distance")),
        min_corridor_risk=(
            float(config["min_corridor_risk"])
            if config.get("min_corridor_risk") is not None
            else None
        ),
        corridor_radius=int(config.get("corridor_radius", 2)),
        candidate_pool_multiplier=int(config.get("candidate_pool_multiplier", 30)),
        risk_weights=config.get("corridor_risk_weights"),
    )

    rows: list[dict[str, Any]] = []
    neutral_action = np.full(ACTION_DIM, 0.5, dtype=np.float32)
    attack_cases = [
        ("none", {}, {}),
        ("environment", config.get("attacks", {}).get("environment", {}), {}),
        ("observation", {}, config.get("attacks", {}).get("observation", {})),
        (
            "combined",
            config.get("attacks", {}).get("environment", {}),
            config.get("attacks", {}).get("observation", {}),
        ),
    ]
    if "slope_risk" in config.get("attacks", {}):
        attack_cases.append(("slope_risk", config["attacks"]["slope_risk"], {}))

    for seed in seeds:
        for eval_domain, tasks in (("train", splits["train"]), ("heldout", splits["heldout"])):
            selected_tasks = tasks[: min(int(num_episodes), len(tasks))]
            for attack_type, environment_attack, observation_attack in attack_cases:
                base_env = RealTerrainPlanningEnv(
                    layers_path=layers_path,
                    tasks=selected_tasks,
                    seed=int(seed),
                    scenario=str(config["scenario"]),
                    mission_profile_scenario=config.get("mission_profile_scenario"),
                    observation_mode=str(config.get("observation_mode", "terrain")),
                    reward_mode=str(config.get("reward_mode", "relative_heuristic")),
                    reward_scale=float(config.get("reward_scale", 10.0)),
                    reward_cost_key=str(config.get("reward_cost_key", "attacked_scalar_cost")),
                    action_mode=str(config.get("action_mode", "preference_delta")),
                    action_gain=float(config.get("action_gain", 3.0)),
                    max_uncertainty_lambda=float(config.get("max_uncertainty_lambda", 1.2)),
                )
                env = wrap_env_with_attacks(
                    base_env,
                    observation_attack=observation_attack,
                    environment_attack=environment_attack,
                )
                for episode_index, task in enumerate(selected_tasks):
                    _, reset_info = env.reset(seed=int(seed) + episode_index, options={"task": task})
                    _, reward, _, _, info = env.step(neutral_action)
                    row = {
                        "level": config["level"],
                        "scenario": config["scenario"],
                        "seed": int(seed),
                        "eval_domain": eval_domain,
                        "attack_type": attack_type,
                        "attack_impl": environment_attack.get("type", observation_attack.get("type", "none")),
                        "method": "neutral_planner_parameter_policy",
                        "episode_id": episode_index,
                        "task_id": task.get("task_id", ""),
                        "tile_id": task.get("tile_id", ""),
                        "start": task.get("start", []),
                        "goal": task.get("goal", []),
                        "success": bool(info.get("success", False)),
                        "reward": float(reward),
                        "scalar_cost": float(info.get("scalar_cost", np.nan)),
                        "attacked_scalar_cost": float(info.get("attacked_scalar_cost", np.nan)),
                        "path_length": int(info.get("path_length", 0)),
                        "attacked_cell_exposure_ratio": float(info.get("attacked_cell_exposure_ratio", 0.0)),
                        "hazard_exposure": float(info.get("hazard_exposure", np.nan)),
                        "uncertainty_exposure": float(info.get("uncertainty_exposure", np.nan)),
                        "lambda_uncertainty": float(info.get("lambda_uncertainty", np.nan)),
                        "reward_cost_source": info.get("reward_cost_source", ""),
                        "environment_attack_type": reset_info.get(
                            "environment_attack_type",
                            info.get("environment_attack_type", ""),
                        ),
                        "attacked_corridor_cells": int(info.get("attacked_corridor_cells", 0)),
                        "run_mode": "reset_step_smoke",
                        "training_status": "real terrain PPO wiring pending",
                    }
                    rows.append(row)
                env.close()

    details = pd.DataFrame(rows)
    summary = summarize_detail_rows(rows)
    details.to_csv(paths["results"] / "episode_details.csv", index=False)
    summary.to_csv(paths["results"] / "summary.csv", index=False)
    plot_real_smoke_summary(summary, paths)
    write_placeholder_oracle(paths["results"] / "oracle_gap_summary.csv", config["level"])
    write_real_report(summary, details, config, paths)
    return summary, details


def plot_real_smoke_summary(summary: pd.DataFrame, paths: dict[str, Path]) -> None:
    if summary.empty:
        return
    figures_dir = paths["figures"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    frame = summary.copy()
    frame["label"] = frame["eval_domain"].astype(str) + "\n" + frame["attack_type"].astype(str)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(np.arange(len(frame)), frame["mean_attacked_scalar_cost"].astype(float).to_numpy(), color="#4C78A8")
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels(frame["label"].tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("mean attacked scalar cost")
    ax.set_title("Real-terrain reset/step smoke: attacked cost")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_smoke_attacked_cost.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(
        np.arange(len(frame)),
        100.0 * frame["relative_degradation"].astype(float).to_numpy(),
        color="#F58518",
    )
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels(frame["label"].tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("relative degradation (%)")
    ax.set_title("Real-terrain reset/step smoke: degradation metric")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_smoke_relative_degradation.png", dpi=180)
    plt.close(fig)


def write_placeholder_oracle(path: Path, level: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "level": level,
            "status": "pending",
            "note": "Oracle gap analysis should run after level-specific PPO checkpoints exist.",
        }
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def report_missing_dataset(config: dict[str, Any], paths: dict[str, Path], layers_path: Path) -> None:
    summary_path = paths["results"] / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["level", "scenario", "status", "missing_map_source", "training_status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "level": config["level"],
                "scenario": config["scenario"],
                "status": "missing_dataset",
                "missing_map_source": str(layers_path),
                "training_status": "dataset required before reset/step or PPO training",
            }
        )
    write_placeholder_oracle(paths["results"] / "oracle_gap_summary.csv", config["level"])
    report = [
        f"# {config['level']} Benchmark Report",
        "",
        "Status: dataset is not available yet.",
        "",
        f"Expected layer file: `{layers_path}`",
        "",
        "This level is structured as an independent benchmark. It should not use lunar or synthetic checkpoints as the main experiment once a Mars DTM tile is added.",
    ]
    (paths["results"] / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def write_real_report(
    summary: pd.DataFrame,
    details: pd.DataFrame,
    config: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    lines = [
        f"# {config['level']} Benchmark Report",
        "",
        "Framing: independent training and evaluation on this benchmark level.",
        "",
        "Current implementation status:",
        "",
        "- Real terrain task splits are generated deterministically.",
        "- The Gymnasium reset/step path runs with the same planner-parameter action interface as the synthetic benchmark.",
        "- Full PPO training on this real terrain environment is not wired into `train_cleanrl_ppo.py` yet.",
        "- Oracle gap output is a placeholder until level-specific PPO checkpoints exist.",
        "",
        "Layer provenance:",
        "",
        "- Terrain, slope, roughness, hazard, and obstacle layers come from the exported DEM-derived layer file.",
        "- Solar, communication, and uncertainty fields should be treated as modeled overlays unless the layer export documents direct measurement.",
        "",
    ]
    if not summary.empty:
        lines.extend(["Smoke summary:", ""])
        for _, row in summary.iterrows():
            lines.append(
                "- "
                f"{row['eval_domain']} / {row['attack_type']}: "
                f"success={float(row['success_rate']):.3f}, "
                f"cost={float(row['mean_attacked_scalar_cost']):.4f}, "
                f"degradation={100.0 * float(row['relative_degradation']):.2f}%, "
                f"exposure={float(row['mean_attacked_cell_exposure_ratio']):.3f}"
            )
    (paths["results"] / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_synthetic_level(
    config: dict[str, Any],
    paths: dict[str, Path],
    args: argparse.Namespace,
    seeds: list[int],
) -> int:
    """Delegate Level 1 to the existing staged recovery runner."""

    if len(seeds) != 1:
        print("WARNING: staged synthetic wrapper currently delegates one seed per run; using the first seed.")
    command = [
        args.python,
        str(PROJECT_ROOT / "run_staged_attack_recovery.py"),
        "--base-config",
        str(PROJECT_ROOT / str(config.get("base_config", "configs/ppo_lunar_corridor_relative_reward.json"))),
        "--checkpoint",
        str(PROJECT_ROOT / str(config.get("nominal_checkpoint", ""))),
        "--attack-config",
        str(PROJECT_ROOT / str(config.get("environment_attack_config", ""))),
        "--output-dir",
        str(paths["results"]),
        "--stage-timesteps",
        str(args.stage_timesteps),
        "--eval-interval",
        str(args.eval_interval),
        "--num-eval-episodes",
        str(args.num_episodes),
        "--seed",
        str(seeds[0]),
    ]
    if args.clean_output:
        command.append("--clean-output")
    if args.dry_run:
        command.append("--dry-run")
    if args.quick:
        command.append("--quick")
    print(" ".join(str(part) for part in command), flush=True)
    if args.dry_run:
        write_synthetic_dry_run_report(config, paths, command)
        return 0
    completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
    if completed.returncode != 0:
        return int(completed.returncode)
    staged_csv = paths["results"] / "staged_recovery_curve.csv"
    if staged_csv.exists():
        frame = pd.read_csv(staged_csv)
        frame.insert(0, "level", config["level"])
        frame.insert(1, "scenario", config["scenario"])
        frame.to_csv(paths["results"] / "summary.csv", index=False)
        write_placeholder_oracle(paths["results"] / "oracle_gap_summary.csv", config["level"])
        append_synthetic_report(config, paths)
    return 0


def write_synthetic_dry_run_report(config: dict[str, Any], paths: dict[str, Path], command: list[str]) -> None:
    write_placeholder_oracle(paths["results"] / "oracle_gap_summary.csv", config["level"])
    (paths["results"] / "summary.csv").write_text(
        "level,scenario,status\n"
        f"{config['level']},{config['scenario']},dry_run\n",
        encoding="utf-8",
    )
    lines = [
        "# Synthetic Benchmark Report",
        "",
        "Dry run only. The command below would run the independent Level 1 staged recovery benchmark:",
        "",
        "```powershell",
        " ".join(str(part) for part in command),
        "```",
    ]
    (paths["results"] / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_synthetic_report(config: dict[str, Any], paths: dict[str, Path]) -> None:
    report_path = paths["results"] / "report.md"
    existing = report_path.read_text(encoding="utf-8-sig") if report_path.exists() else ""
    prefix = [
        f"# {config['level']} Benchmark Report",
        "",
        "Framing: independent controlled synthetic benchmark for mechanism validation.",
        "",
        "This level uses synthetic route-conflict maps to test whether planner-parameter adaptation can reduce attacked corridor exposure.",
        "",
    ]
    report_path.write_text("\n".join(prefix) + "\n" + existing, encoding="utf-8")


def write_run_config(paths: dict[str, Path], args: argparse.Namespace, config: dict[str, Any], seeds: list[int]) -> None:
    value = {
        "command_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "level_config": config,
        "seeds": seeds,
        "framing": "multi-level planetary rover robustness evaluation with independent level training",
    }
    write_json(paths["root"] / "run_config.json", value)


def main() -> int:
    args = parse_args()
    config = resolve_config(args)
    seeds = parse_seeds(args.seeds)
    if args.quick:
        args.num_episodes = min(int(args.num_episodes), 20)
        args.stage_timesteps = min(int(args.stage_timesteps), 1024)
        args.eval_interval = min(int(args.eval_interval), 256)
    scenario = str(config.get("scenario", args.scenario or args.level))
    output_dir = args.output_dir or default_output_dir(args.level, scenario)
    if args.clean_output:
        clean_output_dir(output_dir, args.dry_run)
    paths = ensure_layout(output_dir)
    write_run_config(paths, args, config, seeds)

    print(f"Level: {args.level}")
    print(f"Scenario: {scenario}")
    print(f"Output: {output_dir}")

    if args.level == "synthetic":
        return run_synthetic_level(config, paths, args, seeds)
    evaluate_real_level_smoke(config, paths, seeds, int(args.num_episodes))
    split_files = sorted(paths["splits"].glob("*_tasks.json"))
    if split_files:
        print(f"Saved splits to {paths['splits']}")
    else:
        print(f"No split files written yet; see {paths['results'] / 'report.md'}")
    print(f"Saved summary to {paths['results'] / 'summary.csv'}")
    print(f"Saved report to {paths['results'] / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
