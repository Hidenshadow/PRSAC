#!/usr/bin/env python
"""Plot one clean/corrupted path-choice example for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.attack_wrappers import apply_environment_attack_to_episode  # noqa: E402
from maps.real_terrain import load_real_layers, load_task_split, make_real_planning_episode  # noqa: E402
from run_lunar_viper_staged_recovery import load_environment_attack, read_json  # noqa: E402
from utils.cleanrl_policy import load_cleanrl_agent, predict_cleanrl_action  # noqa: E402
from utils.metrics import (  # noqa: E402
    OBJECTIVE_NAMES,
    action_to_planning_weights,
    action_to_uncertainty_lambda,
    candidate_planner_configs,
    compute_observation,
    path_overlap_ratio,
    plan_with_weights,
)


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "runs"
    / "rl_baselines"
    / "ppo"
    / "level2_medium_shock_recovery_5seeds"
    / "seed0"
)
DEFAULT_LEVEL_CONFIG = PROJECT_ROOT / "configs" / "levels" / "ppo_difficulty" / "level2_medium.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "paper_figures" / "path_choice_example"
RISK_LAYERS = ("energy", "hazard", "communication", "illumination")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--level-config", type=Path, default=DEFAULT_LEVEL_CONFIG)
    parser.add_argument("--tasks", type=Path, default=None)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nonlearning-method", type=str, default="emergency_uncertainty_rule")
    parser.add_argument("--learning-checkpoint", type=Path, default=None)
    parser.add_argument("--nominal-checkpoint", type=Path, default=None)
    parser.add_argument("--learning-label", type=str, default="PPO recovery")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default="level2_medium_path_choice_example")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def latest_recovery_checkpoint(run_dir: Path) -> Path:
    checkpoints_dir = run_dir / "checkpoints"
    best_step = -1
    best_path: Path | None = None
    for path in checkpoints_dir.glob("checkpoint_recovery_step_*.pt"):
        match = re.search(r"checkpoint_recovery_step_(\d+)\.pt$", path.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > best_step:
            best_step = step
            best_path = path
    if best_path is None:
        raise FileNotFoundError(f"no recovery checkpoints found in {checkpoints_dir}")
    return best_path


def composite_risk(costmap) -> np.ndarray:
    fields = [np.asarray(costmap.layers[name], dtype=np.float32) for name in RISK_LAYERS]
    return np.mean(np.stack(fields, axis=0), axis=0).astype(np.float32)


def masked_for_plot(values: np.ndarray, obstacle_mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.array(np.asarray(values, dtype=np.float32), mask=np.asarray(obstacle_mask, dtype=bool))


def action_config(level_config: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint.get("config", {}), dict) else {}

    def pick(name: str, default: Any) -> Any:
        hyphen = name.replace("_", "-")
        return level_config.get(name, checkpoint_config.get(name, checkpoint_config.get(hyphen, default)))

    return {
        "observation_mode": str(pick("observation_mode", "terrain")),
        "action_mode": str(pick("action_mode", "preference_delta")),
        "action_gain": float(pick("action_gain", 3.0)),
        "max_uncertainty_lambda": float(pick("max_uncertainty_lambda", 1.2)),
    }


def evaluate_checkpoint_policy(
    checkpoint_path: Path,
    episode,
    level_config: dict[str, Any],
    device: str,
) -> dict[str, Any]:
    agent, checkpoint = load_cleanrl_agent(checkpoint_path, device=device)
    cfg = action_config(level_config, checkpoint)
    map_size = int(episode.costmap.layers["distance"].shape[0])
    obs = compute_observation(
        episode,
        map_size,
        observation_mode=cfg["observation_mode"],
        max_uncertainty_lambda=cfg["max_uncertainty_lambda"],
    )
    if int(obs.shape[0]) != int(agent.obs_dim):
        raise ValueError(
            f"checkpoint obs_dim={agent.obs_dim} but observation shape is {obs.shape}; "
            f"checkpoint={checkpoint_path}"
        )
    action = predict_cleanrl_action(agent, obs, device=device, deterministic=True)
    weights = action_to_planning_weights(
        episode,
        action,
        action_mode=cfg["action_mode"],
        action_gain=cfg["action_gain"],
    )
    lambda_uncertainty = action_to_uncertainty_lambda(
        action,
        max_uncertainty_lambda=cfg["max_uncertainty_lambda"],
    )
    result = plan_with_weights(
        episode,
        weights,
        lambda_uncertainty=lambda_uncertainty,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "action": np.asarray(action, dtype=np.float32),
        "weights": np.asarray(weights, dtype=np.float32),
        "lambda_uncertainty": float(lambda_uncertainty),
        "result": result,
        "config": cfg,
    }


def evaluate_nonlearning(episode, method: str, max_uncertainty_lambda: float) -> dict[str, Any]:
    configs = candidate_planner_configs(episode, max_uncertainty_lambda=max_uncertainty_lambda)
    if method not in configs:
        raise ValueError(f"unknown non-learning method {method!r}; available={sorted(configs)}")
    config = configs[method]
    result = plan_with_weights(
        episode,
        np.asarray(config["weights"], dtype=np.float32),
        lambda_uncertainty=float(config["lambda_uncertainty"]),
    )
    return {
        "method": method,
        "weights": np.asarray(config["weights"], dtype=np.float32),
        "lambda_uncertainty": float(config["lambda_uncertainty"]),
        "result": result,
    }


def path_xy(path: list[tuple[int, int]] | None) -> tuple[np.ndarray, np.ndarray]:
    if not path:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    rows = np.asarray([cell[0] for cell in path], dtype=np.float32)
    cols = np.asarray([cell[1] for cell in path], dtype=np.float32)
    return cols, rows


def add_base_map(
    ax: plt.Axes,
    costmap,
    title: str,
    attack_mask: np.ndarray | None = None,
    show_mask: bool = False,
):
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#222222")
    image = ax.imshow(masked_for_plot(composite_risk(costmap), costmap.obstacle_mask), cmap=cmap, vmin=0.0, vmax=1.0)
    obstacle = np.asarray(costmap.obstacle_mask, dtype=bool)
    if obstacle.any():
        ax.contour(obstacle.astype(float), levels=[0.5], colors="black", linewidths=0.45, alpha=0.55)
    if show_mask and attack_mask is not None and bool(np.asarray(attack_mask, dtype=bool).any()):
        mask = np.asarray(attack_mask, dtype=bool)
        overlay = np.ma.array(mask.astype(np.float32), mask=~mask)
        ax.imshow(overlay, cmap="Reds", vmin=0.0, vmax=1.0, alpha=0.18)
        ax.contour(mask.astype(float), levels=[0.5], colors="#d62728", linewidths=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return image


def add_start_goal(ax: plt.Axes, costmap) -> None:
    start_row, start_col = costmap.start
    goal_row, goal_col = costmap.goal
    stroke = [pe.withStroke(linewidth=2.6, foreground="black", alpha=0.80)]
    ax.scatter([start_col], [start_row], marker="o", s=112, c="#2ca02c", edgecolors="white", linewidths=1.3, zorder=8)
    ax.scatter([goal_col], [goal_row], marker="*", s=190, c="#1f77b4", edgecolors="white", linewidths=1.1, zorder=8)
    ax.text(start_col + 1.8, start_row + 1.8, "start", color="white", fontsize=9, weight="bold", path_effects=stroke, zorder=9)
    ax.text(goal_col + 1.8, goal_row + 1.8, "goal", color="white", fontsize=9, weight="bold", path_effects=stroke, zorder=9)


def add_path(
    ax: plt.Axes,
    path: list[tuple[int, int]] | None,
    color: str,
    label: str,
    linewidth: float = 2.4,
    zorder: int = 6,
) -> None:
    x, y = path_xy(path)
    if x.size == 0:
        return
    line = ax.plot(x, y, color=color, linewidth=linewidth, label=label, zorder=zorder)[0]
    line.set_path_effects([pe.Stroke(linewidth=linewidth + 1.7, foreground="black", alpha=0.65), pe.Normal()])


def metric_line(result: dict[str, Any], clean_cost: float) -> str:
    cost = float(result.get("scalar_cost", np.nan))
    pi = 100.0 * float(clean_cost) / max(cost, 1e-12)
    exposure = float(result.get("attacked_cell_exposure_ratio", 0.0))
    return f"true cost={cost:.3f} | PI={pi:.1f} | mask exposure={100.0 * exposure:.1f}%"


def save_figure(
    clean_episode,
    corrupted_episode,
    clean_policy: dict[str, Any],
    nonlearning: dict[str, Any],
    learning: dict[str, Any],
    output_dir: Path,
    prefix: str,
    learning_label: str,
) -> None:
    clean_map = clean_episode.costmap
    corrupted_map = corrupted_episode.costmap
    attack_mask = getattr(corrupted_map, "attack_mask", None)
    clean_result = clean_policy["result"]
    nonlearning_result = nonlearning["result"]
    learning_result = learning["result"]
    clean_cost = float(clean_result["scalar_cost"])

    panels = [
        ("Clean belief\n" + metric_line(clean_result, clean_cost), clean_map, clean_result["path"], "#ffffff", "Clean PPO", False),
        (
            "Corrupted belief + non-learning\n" + metric_line(nonlearning_result, clean_cost),
            corrupted_map,
            nonlearning_result["path"],
            "#ff9f1c",
            str(nonlearning["method"]),
            True,
        ),
        (
            "Corrupted belief + learning\n" + metric_line(learning_result, clean_cost),
            corrupted_map,
            learning_result["path"],
            "#22d3ee",
            learning_label,
            True,
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18.8, 4.9), constrained_layout=True)
    for ax, (title, costmap, path, color, label, show_mask) in zip(axes[:3], panels):
        add_base_map(ax, costmap, title, attack_mask=attack_mask, show_mask=show_mask)
        add_path(ax, path, color, label)
        add_start_goal(ax, costmap)

    add_base_map(
        axes[3],
        corrupted_map,
        "Combined overlay on corrupted belief",
        attack_mask=attack_mask,
        show_mask=True,
    )
    add_path(axes[3], clean_result["path"], "#ffffff", "clean", linewidth=2.0, zorder=5)
    add_path(axes[3], nonlearning_result["path"], "#ff9f1c", "non-learning", linewidth=2.3, zorder=6)
    add_path(axes[3], learning_result["path"], "#22d3ee", "learning", linewidth=2.3, zorder=7)
    add_start_goal(axes[3], corrupted_map)
    axes[3].legend(loc="lower left", fontsize=8, framealpha=0.86)

    fig.suptitle("Example global route choices before and after belief corruption", fontsize=13)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{prefix}.png", dpi=260)
    fig.savefig(output_dir / f"{prefix}.pdf")
    plt.close(fig)


def save_metadata(
    clean_episode,
    corrupted_episode,
    task: dict[str, Any],
    clean_policy: dict[str, Any],
    nonlearning: dict[str, Any],
    learning: dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> None:
    clean_result = clean_policy["result"]
    nonlearning_result = nonlearning["result"]
    learning_result = learning["result"]
    clean_cost = float(clean_result["scalar_cost"])

    def record(name: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = payload["result"]
        cost = float(result.get("scalar_cost", np.nan))
        path = result.get("path") or []
        return {
            "name": name,
            "true_scalar_cost": cost,
            "performance_index": 100.0 * clean_cost / max(cost, 1e-12),
            "path_length": int(result.get("path_length", 0)),
            "path": [list(map(int, cell)) for cell in path],
            "attacked_cell_exposure_ratio": float(result.get("attacked_cell_exposure_ratio", 0.0)),
            "mean_path_confidence": float(result.get("mean_path_confidence", np.nan)),
            "hazard_exposure": float(result.get("hazard_exposure", np.nan)),
            "belief_hazard_exposure": float(result.get("belief_hazard_exposure", np.nan)),
            "lambda_uncertainty": float(payload.get("lambda_uncertainty", np.nan)),
            "weights": np.asarray(payload.get("weights", []), dtype=np.float32).tolist(),
        }

    metadata = {
        "task": task,
        "start": list(clean_episode.costmap.start),
        "goal": list(clean_episode.costmap.goal),
        "clean_policy": record("clean_ppo_on_clean_belief", clean_policy),
        "nonlearning": {
            **record(str(nonlearning["method"]), nonlearning),
            "method": str(nonlearning["method"]),
        },
        "learning": {
            **record("learning_recovery_on_corrupted_belief", learning),
            "checkpoint_path": str(learning.get("checkpoint_path", "")),
        },
        "path_overlap_with_clean": {
            "nonlearning": path_overlap_ratio(clean_result.get("path"), nonlearning_result.get("path")),
            "learning": path_overlap_ratio(clean_result.get("path"), learning_result.get("path")),
        },
        "attack_metadata": getattr(corrupted_episode.costmap, "attack_metadata", {}) or {},
        "objective_names": OBJECTIVE_NAMES,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}_metadata.json").write_text(json.dumps(jsonable(metadata), indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = resolve(args.run_dir)
    level_config_path = resolve(args.level_config)
    output_dir = resolve(args.output_dir)
    level_config = read_json(level_config_path)

    run_config_path = run_dir / "run_config.json"
    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        level_config = dict(run_config.get("level_config", level_config))
        env_attack = dict(run_config.get("environment_attack", load_environment_attack(level_config)))
    else:
        run_config = {}
        env_attack = load_environment_attack(level_config)

    tasks_path = resolve(args.tasks) if args.tasks is not None else run_dir / "splits" / "validation_tasks.json"
    nominal_checkpoint = resolve(args.nominal_checkpoint) if args.nominal_checkpoint else run_dir / "checkpoints" / "checkpoint_nominal.pt"
    learning_checkpoint = resolve(args.learning_checkpoint) if args.learning_checkpoint else latest_recovery_checkpoint(run_dir)

    raw_layers = load_real_layers(resolve(Path(str(level_config["map_source"]))))
    tasks = load_task_split(tasks_path)
    if not tasks:
        raise ValueError(f"no tasks in {tasks_path}")
    task_index = int(np.clip(args.task_index, 0, len(tasks) - 1))
    task = tasks[task_index]
    rng = np.random.default_rng(int(args.seed) + int(task.get("seed", 0)) + task_index)
    clean_episode = make_real_planning_episode(
        raw_layers,
        task,
        rng,
        scenario=str(level_config.get("scenario", "real_lunar_viper")),
        mission_profile_scenario=str(level_config.get("mission_profile_scenario", "lunar_polar_shadow")),
    )
    corrupted_episode = apply_environment_attack_to_episode(
        clean_episode,
        env_attack,
        np.random.default_rng(int(args.seed) + 300_000 + task_index),
    )

    clean_policy = evaluate_checkpoint_policy(nominal_checkpoint, clean_episode, level_config, args.device)
    max_lambda = float(clean_policy["config"]["max_uncertainty_lambda"])
    nonlearning = evaluate_nonlearning(corrupted_episode, args.nonlearning_method, max_lambda)
    learning = evaluate_checkpoint_policy(learning_checkpoint, corrupted_episode, level_config, args.device)

    save_figure(
        clean_episode,
        corrupted_episode,
        clean_policy,
        nonlearning,
        learning,
        output_dir,
        args.prefix,
        args.learning_label,
    )
    save_metadata(
        clean_episode,
        corrupted_episode,
        task,
        clean_policy,
        nonlearning,
        learning,
        output_dir,
        args.prefix,
    )
    print(output_dir / f"{args.prefix}.png")
    print(output_dir / f"{args.prefix}.pdf")
    print(output_dir / f"{args.prefix}_metadata.json")


if __name__ == "__main__":
    main()
