"""Evaluate PPO recovery checkpoints on the real VIPER DEM tile.

The real VIPER runner is a planning benchmark and therefore does not create PPO
checkpoints. This script bridges the two workflows: it takes the checkpoint
sequence produced by the toy recovery run and evaluates each checkpoint on the
real DEM/VIPER planning episodes so the real scenario has the same step-wise
curve points.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from envs.attack_wrappers import apply_environment_attack_to_episode
from run_real_viper_scenario import (
    DEFAULT_LAYERS,
    DEFAULT_METADATA,
    attack_config,
    build_costmap,
    load_layers,
    mission_priority,
    nearest_free_cell,
    read_json,
    sample_start_goal,
    viper_rover_state,
)
from utils.cleanrl_policy import load_cleanrl_agent
from utils.metrics import (
    DEFAULT_MAX_UNCERTAINTY_LAMBDA,
    PlanningEpisode,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    compute_observation,
    plan_with_weights,
)


DEFAULT_CHECKPOINT_DIR = Path("runs") / "corridor_strength3_recovery_curve_very_smooth" / "checkpoints"
DEFAULT_OUTPUT_DIR = Path("runs") / "real_viper_checkpoint_curve"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recovery checkpoints on the real VIPER DEM tile.")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--layers", type=Path, default=DEFAULT_LAYERS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-distance-ratio", type=float, default=0.62)
    parser.add_argument("--attack-strength", type=float, default=3.0)
    parser.add_argument("--corridor-radius", type=int, default=2)
    parser.add_argument("--observation-mode", type=str, default="terrain")
    parser.add_argument("--action-mode", type=str, default=None)
    parser.add_argument("--action-gain", type=float, default=None)
    parser.add_argument("--max-uncertainty-lambda", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--progress-interval", type=int, default=5)
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Wait for new checkpoint_step_*.pt files and evaluate them as they appear.",
    )
    parser.add_argument("--expected-final-step", type=int, default=51200)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def checkpoint_step(path: Path) -> int:
    match = re.search(r"checkpoint_step_(\d+)\.pt$", path.name)
    if match:
        return int(match.group(1))
    return -1


def list_checkpoints(path: Path, allow_empty: bool = False) -> list[Path]:
    if not path.exists():
        if allow_empty:
            return []
        raise FileNotFoundError(
            f"Checkpoint directory does not exist: {path}. Run the toy recovery command first."
        )
    checkpoints = sorted(path.glob("checkpoint_step_*.pt"), key=checkpoint_step)
    if not checkpoints and not allow_empty:
        raise FileNotFoundError(f"No checkpoint_step_*.pt files found in {path}")
    return checkpoints


def clean_output_dir(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    resolved = output_dir.resolve()
    cwd = Path.cwd().resolve()
    if cwd not in [resolved, *resolved.parents]:
        raise RuntimeError(f"Refusing to remove output outside workspace: {resolved}")
    shutil.rmtree(output_dir)


def generate_real_episodes(
    raw_layers: dict[str, np.ndarray],
    num_episodes: int,
    seed: int,
    min_distance_ratio: float,
    attack_strength: float,
    corridor_radius: int,
) -> list[tuple[PlanningEpisode, PlanningEpisode]]:
    rng = np.random.default_rng(seed)
    obstacle_mask = np.asarray(raw_layers["obstacle_mask"], dtype=bool)
    episodes = []
    for _ in range(int(num_episodes)):
        start, goal = sample_start_goal(rng, obstacle_mask, float(min_distance_ratio))
        start = nearest_free_cell(obstacle_mask, start)
        goal = nearest_free_cell(obstacle_mask, goal)
        costmap = build_costmap(raw_layers, start=start, goal=goal)
        episode = PlanningEpisode(
            costmap=costmap,
            mission_priority=mission_priority(),
            rover_state=viper_rover_state(),
            scenario="real_viper_dem",
            mission_regime="viper_hazard_energy",
            mission_severity=0.75,
        )
        attacked_episode = apply_environment_attack_to_episode(
            episode,
            attack_config(float(attack_strength), int(corridor_radius)),
            rng=rng,
        )
        episodes.append((episode, attacked_episode))
    return episodes


def checkpoint_config_value(
    checkpoint: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        return default
    return config.get(key, config.get(key.replace("-", "_"), default))


def safe_nanmean(values: list[float], default: float = float("nan")) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.isnan(array).all():
        return float(default)
    return float(np.nanmean(array))


def safe_nanstd(values: list[float], default: float = float("nan")) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.isnan(array).all():
        return float(default)
    return float(np.nanstd(array))


def evaluate_checkpoint(
    checkpoint_path: Path,
    global_step: int,
    episodes: list[tuple[PlanningEpisode, PlanningEpisode]],
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    agent, checkpoint = load_cleanrl_agent(checkpoint_path, device=device)
    action_mode = str(
        args.action_mode
        if args.action_mode is not None
        else checkpoint_config_value(checkpoint, "action_mode", "preference_delta")
    )
    action_gain = float(
        args.action_gain
        if args.action_gain is not None
        else checkpoint_config_value(checkpoint, "action_gain", 3.0)
    )
    max_uncertainty_lambda = float(
        args.max_uncertainty_lambda
        if args.max_uncertainty_lambda is not None
        else checkpoint_config_value(checkpoint, "max_uncertainty_lambda", DEFAULT_MAX_UNCERTAINTY_LAMBDA)
    )

    rows: list[dict[str, Any]] = []
    for attack_type, index in (("none", 0), ("path_corridor", 1)):
        nominal_costs = []
        attacked_costs = []
        rewards = []
        successes = []
        path_lengths = []
        exposure_ratios = []
        hazard_exposures = []
        belief_hazard_exposures = []
        uncertainty_exposures = []
        belief_uncertainty_exposures = []
        belief_costs = []
        map_mismatch_penalties = []
        map_mismatch_abs_errors = []
        path_confidences = []
        true_belief_mismatch_flags = []
        mismatched_cells = []
        mean_belief_abs_errors = []
        mean_true_minus_belief_errors = []
        mean_selected_confidences = []
        lambdas = []
        weights_list = []

        for pair in episodes:
            clean_episode = pair[0]
            episode = pair[index]
            map_size = int(clean_episode.costmap.layers["distance"].shape[0])
            clean_obs = compute_observation(
                clean_episode,
                map_size=map_size,
                observation_mode=str(args.observation_mode),
                max_uncertainty_lambda=max_uncertainty_lambda,
            )
            if clean_obs.shape[0] != int(agent.obs_dim):
                raise ValueError(
                    f"checkpoint obs_dim={agent.obs_dim} but real observation has shape {clean_obs.shape}. "
                    "Use the same observation mode as the toy recovery checkpoint."
                )
            clean_obs_tensor = torch.as_tensor(clean_obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                clean_action = agent.get_deterministic_action(clean_obs_tensor).squeeze(0).cpu().numpy()
            clean_weights = action_to_planning_weights(
                clean_episode,
                clean_action,
                action_mode=action_mode,
                action_gain=action_gain,
            )
            clean_lambda = action_to_uncertainty_lambda(
                clean_action,
                max_uncertainty_lambda=max_uncertainty_lambda,
            )
            clean_result = plan_with_weights(
                clean_episode,
                clean_weights,
                lambda_uncertainty=clean_lambda,
            )

            map_size = int(episode.costmap.layers["distance"].shape[0])
            obs = compute_observation(
                episode,
                map_size=map_size,
                observation_mode=str(args.observation_mode),
                max_uncertainty_lambda=max_uncertainty_lambda,
            )
            if obs.shape[0] != int(agent.obs_dim):
                raise ValueError(
                    f"checkpoint obs_dim={agent.obs_dim} but real observation has shape {obs.shape}. "
                    "Use the same observation mode as the toy recovery checkpoint."
                )
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.get_deterministic_action(obs_tensor).squeeze(0).cpu().numpy()
            weights = action_to_planning_weights(
                episode,
                action,
                action_mode=action_mode,
                action_gain=action_gain,
            )
            lambda_uncertainty = action_to_uncertainty_lambda(
                action,
                max_uncertainty_lambda=max_uncertainty_lambda,
            )
            result = plan_with_weights(
                episode,
                weights,
                lambda_uncertainty=lambda_uncertainty,
            )
            nominal = float(clean_result.get("scalar_cost", np.nan))
            # Match run_attack_recovery_finetune.py: no-attack uses nominal
            # scalar cost; path-corridor layer attacks mutate the map, so the
            # attacked metric is the scalar cost on the mutated map rather than
            # an additional uncertainty-attack objective.
            attacked = float(result.get("scalar_cost", np.nan))
            nominal_costs.append(nominal)
            attacked_costs.append(attacked)
            rewards.append(-attacked)
            successes.append(1.0 if bool(result.get("success", False)) else 0.0)
            path_lengths.append(float(result.get("path_length", np.nan)))
            exposure_ratios.append(float(result.get("attacked_cell_exposure_ratio", 0.0)))
            hazard_exposures.append(float(result.get("hazard_exposure", np.nan)))
            belief_hazard_exposures.append(float(result.get("belief_hazard_exposure", np.nan)))
            uncertainty_exposures.append(float(result.get("uncertainty_exposure", np.nan)))
            belief_uncertainty_exposures.append(float(result.get("belief_uncertainty_exposure", np.nan)))
            belief_costs.append(float(result.get("belief_scalar_cost", result.get("scalar_cost", np.nan))))
            map_mismatch_penalties.append(float(result.get("map_mismatch_penalty", 0.0)))
            map_mismatch_abs_errors.append(float(result.get("map_mismatch_abs_error", 0.0)))
            path_confidences.append(float(result.get("mean_path_confidence", np.nan)))
            true_belief_mismatch_flags.append(1.0 if bool(result.get("true_belief_mismatch", False)) else 0.0)
            attack_metadata = result.get("attack_metadata", {}) if isinstance(result.get("attack_metadata", {}), dict) else {}
            mismatched_cells.append(float(attack_metadata.get("mismatched_cells", np.nan)))
            mean_belief_abs_errors.append(float(attack_metadata.get("mean_belief_abs_error", np.nan)))
            mean_true_minus_belief_errors.append(float(attack_metadata.get("mean_true_minus_belief_error", np.nan)))
            mean_selected_confidences.append(float(attack_metadata.get("mean_selected_confidence", np.nan)))
            lambdas.append(float(lambda_uncertainty))
            weights_list.append(np.asarray(weights, dtype=np.float32))

        mean_nominal = safe_nanmean(nominal_costs)
        mean_attacked = safe_nanmean(attacked_costs)
        mean_weights = (
            np.nanmean(np.stack(weights_list, axis=0), axis=0)
            if weights_list
            else np.full(5, np.nan, dtype=np.float32)
        )
        rows.append(
            {
                "global_step": int(global_step),
                "attack_type": attack_type,
                "num_episodes": int(len(episodes)),
                "mean_nominal_scalar_cost": mean_nominal,
                "mean_attacked_scalar_cost": mean_attacked,
                "std_attacked_scalar_cost": safe_nanstd(attacked_costs),
                "absolute_degradation": float(mean_attacked - mean_nominal),
                "relative_degradation": float((mean_attacked - mean_nominal) / (abs(mean_nominal) + 1e-8)),
                "success_rate": safe_nanmean(successes),
                "mean_reward": safe_nanmean(rewards),
                "mean_path_length": safe_nanmean(path_lengths),
                "mean_attacked_cell_exposure_ratio": safe_nanmean(exposure_ratios),
                "mean_hazard_exposure": safe_nanmean(hazard_exposures),
                "mean_belief_hazard_exposure": safe_nanmean(belief_hazard_exposures),
                "mean_uncertainty_exposure": safe_nanmean(uncertainty_exposures),
                "mean_belief_uncertainty_exposure": safe_nanmean(belief_uncertainty_exposures),
                "mean_belief_scalar_cost": safe_nanmean(belief_costs),
                "mean_map_mismatch_penalty": safe_nanmean(map_mismatch_penalties, default=0.0),
                "mean_map_mismatch_abs_error": safe_nanmean(map_mismatch_abs_errors, default=0.0),
                "mean_path_confidence": safe_nanmean(path_confidences),
                "true_belief_mismatch_rate": safe_nanmean(true_belief_mismatch_flags, default=0.0),
                "mean_mismatched_cells": safe_nanmean(mismatched_cells),
                "mean_belief_abs_error": safe_nanmean(mean_belief_abs_errors),
                "mean_true_minus_belief_error": safe_nanmean(mean_true_minus_belief_errors),
                "mean_selected_confidence": safe_nanmean(mean_selected_confidences),
                "mean_lambda_uncertainty": safe_nanmean(lambdas),
                "mean_weight_distance": float(mean_weights[0]),
                "mean_weight_energy": float(mean_weights[1]),
                "mean_weight_hazard": float(mean_weights[2]),
                "mean_weight_communication": float(mean_weights[3]),
                "mean_weight_illumination": float(mean_weights[4]),
                "checkpoint_path": str(checkpoint_path),
                "action_mode": action_mode,
                "action_gain": action_gain,
                "max_uncertainty_lambda": max_uncertainty_lambda,
            }
        )
    return rows


def plot_curve(frame: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    def save(fig: plt.Figure, filename: str) -> None:
        fig.savefig(figures_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for attack_type, group in frame.groupby("attack_type"):
        group = group.sort_values("global_step")
        ax.plot(group["global_step"], group["mean_attacked_scalar_cost"], marker="o", label=attack_type)
    ax.set_xlabel("global_step")
    ax.set_ylabel("mean_attacked_scalar_cost")
    ax.set_title("Toy recovery checkpoints evaluated on real VIPER DEM")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_real_checkpoint_attacked_cost.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for attack_type, group in frame.groupby("attack_type"):
        group = group.sort_values("global_step")
        ax.plot(group["global_step"], group["relative_degradation"], marker="o", label=attack_type)
    ax.set_xlabel("global_step")
    ax.set_ylabel("relative_degradation")
    ax.set_title("Real VIPER degradation over toy recovery checkpoints")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_real_checkpoint_degradation.png")

    env_frame = frame[frame["attack_type"] == "path_corridor"].sort_values("global_step")
    no_attack_step0 = frame[(frame["attack_type"] == "none") & (frame["global_step"] == 0)]
    if not env_frame.empty and not no_attack_step0.empty:
        steps = env_frame["global_step"].astype(int).tolist()
        labels = ["No attack\nstep 0"] + ["Attack\nstep 0" if step == 0 else f"FT\nstep {step}" for step in steps]
        costs = [float(no_attack_step0.iloc[0]["mean_attacked_scalar_cost"])] + env_frame[
            "mean_attacked_scalar_cost"
        ].astype(float).tolist()
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(max(9.0, 0.65 * len(labels)), 4.8))
        ax.plot(x, costs, marker="o", linewidth=2.2)
        ax.axvspan(0.5, 1.5, color="tab:red", alpha=0.08)
        if len(labels) > 2:
            ax.axvspan(1.5, len(labels) - 0.5, color="tab:green", alpha=0.08)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Mean scalar cost (lower is better)")
        ax.set_title("Real VIPER continuous attack and checkpoint stages - cost")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save(fig, "fig_real_checkpoint_stage_cost.png")

        base = max(abs(costs[0]), 1e-8)
        performance = [costs[0] / max(abs(cost), 1e-8) for cost in costs]
        fig, ax = plt.subplots(figsize=(max(9.0, 0.65 * len(labels)), 4.8))
        ax.plot(x, performance, marker="o", linewidth=2.2)
        ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1)
        ax.axvspan(0.5, 1.5, color="tab:red", alpha=0.08)
        if len(labels) > 2:
            ax.axvspan(1.5, len(labels) - 0.5, color="tab:green", alpha=0.08)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Normalized performance (higher is better)")
        ax.set_title("Real VIPER continuous attack and checkpoint stages - performance")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save(fig, "fig_real_checkpoint_stage_performance.png")

    if not env_frame.empty:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(
            env_frame["global_step"],
            env_frame["mean_attacked_cell_exposure_ratio"],
            marker="o",
        )
        ax.set_xlabel("global_step")
        ax.set_ylabel("mean_attacked_cell_exposure_ratio")
        ax.set_title("Real VIPER exposure over toy recovery checkpoints")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save(fig, "fig_real_checkpoint_exposure.png")


def write_output_guide(output_dir: Path) -> None:
    guide = """# Real VIPER Checkpoint Curve Outputs

This evaluates toy-environment recovery checkpoints on the real VIPER DEM tile.
It does not train a new real-map PPO policy.

## Main CSV

- `real_checkpoint_curve.csv`: one row per `global_step x attack_type`.
- `mean_attacked_scalar_cost`: main metric. Lower is better.
- `relative_degradation`: attack cost increase relative to the same checkpoint's nominal scalar cost.
- `mean_attacked_cell_exposure_ratio`: path exposure to the attacked corridor.
- `mean_lambda_uncertainty` and `mean_weight_*`: PPO output diagnostics on real DEM observations.

## Figures

- `figures/fig_real_checkpoint_stage_performance.png`: narrative no-attack -> attack -> checkpoint curve. Higher is better.
- `figures/fig_real_checkpoint_stage_cost.png`: same narrative in raw cost. Lower is better.
- `figures/fig_real_checkpoint_attacked_cost.png`: cost by checkpoint and attack type.
- `figures/fig_real_checkpoint_degradation.png`: degradation by checkpoint.
- `figures/fig_real_checkpoint_exposure.png`: attacked-corridor exposure by checkpoint.
"""
    (output_dir / "OUTPUT_GUIDE.md").write_text(guide, encoding="utf-8")


def write_report(frame: pd.DataFrame, output_dir: Path) -> None:
    env_frame = frame[frame["attack_type"] == "path_corridor"].sort_values("global_step")
    lines = ["# Real VIPER Checkpoint Curve Report", ""]
    if not env_frame.empty:
        first = env_frame.iloc[0]
        final = env_frame.iloc[-1]
        improvement = float(first["mean_attacked_scalar_cost"] - final["mean_attacked_scalar_cost"])
        rel = improvement / (abs(float(first["mean_attacked_scalar_cost"])) + 1e-8)
        lines.extend(
            [
                f"- Step 0 attacked cost: `{float(first['mean_attacked_scalar_cost']):.4f}`",
                f"- Final attacked cost: `{float(final['mean_attacked_scalar_cost']):.4f}`",
                f"- Absolute improvement: `{improvement:.4f}`",
                f"- Relative improvement: `{100.0 * rel:.2f}%`",
                f"- Step 0 exposure: `{float(first['mean_attacked_cell_exposure_ratio']):.4f}`",
                f"- Final exposure: `{float(final['mean_attacked_cell_exposure_ratio']):.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            "Interpretation:",
            "",
            "- If the curve improves, toy attack fine-tuning transfers to the real DEM tile.",
            "- If it stays flat or worsens, the toy recovery policy is not transferring; then the real map needs its own loader/training setup.",
        ]
    )
    (output_dir / "real_checkpoint_curve_report.md").write_text("\n".join(lines), encoding="utf-8")


def save_outputs(frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    frame = frame.sort_values(["global_step", "attack_type"])
    csv_path = output_dir / "real_checkpoint_curve.csv"
    frame.to_csv(csv_path, index=False)
    plot_curve(frame, output_dir)
    write_output_guide(output_dir)
    write_report(frame, output_dir)
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "checkpoint_dir": str(args.checkpoint_dir),
                "layers": str(args.layers),
                "metadata": str(args.metadata),
                "dem_tile_row": metadata.get("row"),
                "dem_tile_col": metadata.get("col"),
                "num_rows": int(len(frame)),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.quick:
        args.num_episodes = 20
    if args.clean_output:
        clean_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_layers = load_layers(args.layers)
    metadata = read_json(args.metadata)
    checkpoints = list_checkpoints(args.checkpoint_dir, allow_empty=bool(args.watch))
    device = resolve_device(args.device)
    episodes = generate_real_episodes(
        raw_layers=raw_layers,
        num_episodes=int(args.num_episodes),
        seed=int(args.seed),
        min_distance_ratio=float(args.min_distance_ratio),
        attack_strength=float(args.attack_strength),
        corridor_radius=int(args.corridor_radius),
    )

    print("Starting real VIPER checkpoint-curve evaluation.", flush=True)
    print(f"  checkpoint_dir: {args.checkpoint_dir}", flush=True)
    print(f"  checkpoints currently available: {len(checkpoints)}", flush=True)
    print(f"  layers: {args.layers}", flush=True)
    print(f"  output_dir: {args.output_dir}", flush=True)
    print(f"  num_episodes: {args.num_episodes}", flush=True)
    print(f"  attack_strength: {args.attack_strength}", flush=True)
    if args.watch:
        print(
            f"  watch mode: waiting until checkpoint_step_{args.expected_final_step}.pt is evaluated",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    start_time = time.time()
    evaluated_steps: set[int] = set()

    while True:
        checkpoints = list_checkpoints(args.checkpoint_dir, allow_empty=bool(args.watch))
        new_checkpoints = [
            path
            for path in checkpoints
            if checkpoint_step(path) >= 0 and checkpoint_step(path) not in evaluated_steps
        ]

        for checkpoint_path in new_checkpoints:
            step = checkpoint_step(checkpoint_path)
            rows.extend(evaluate_checkpoint(checkpoint_path, step, episodes, args, device))
            evaluated_steps.add(step)

        if rows and new_checkpoints:
            frame = pd.DataFrame(rows)
            save_outputs(frame, args.output_dir, args, metadata)

        evaluated_count = len(evaluated_steps)
        total_known = len(checkpoints)
        interval = int(args.progress_interval)
        if new_checkpoints and interval > 0:
            last_step = checkpoint_step(new_checkpoints[-1])
            elapsed = max(time.time() - start_time, 1e-8)
            print(
                f"progress evaluated={evaluated_count}/{total_known} known checkpoints "
                f"latest_step={last_step} elapsed={elapsed:.1f}s",
                flush=True,
            )

        if not args.watch:
            break
        if int(args.expected_final_step) in evaluated_steps:
            break
        if not new_checkpoints:
            print(
                f"waiting for new checkpoints in {args.checkpoint_dir} "
                f"(evaluated={evaluated_count}, expected_final_step={args.expected_final_step})",
                flush=True,
            )
        time.sleep(max(float(args.poll_seconds), 1.0))

    if not rows:
        raise RuntimeError("No checkpoints were evaluated.")

    frame = pd.DataFrame(rows)
    save_outputs(frame, args.output_dir, args, metadata)

    print(f"Saved CSV: {args.output_dir / 'real_checkpoint_curve.csv'}", flush=True)
    print(f"Saved figures: {args.output_dir / 'figures'}", flush=True)
    print(f"Saved report: {args.output_dir / 'real_checkpoint_curve_report.md'}", flush=True)
    print(f"Saved guide: {args.output_dir / 'OUTPUT_GUIDE.md'}", flush=True)


if __name__ == "__main__":
    main()
