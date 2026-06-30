#!/usr/bin/env python
"""Visualize clean vs corrupted terrain belief maps for the paper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.attack_wrappers import apply_environment_attack_to_episode  # noqa: E402
from maps.real_terrain import load_real_layers, load_task_split, make_real_planning_episode  # noqa: E402
from run_lunar_viper_staged_recovery import load_environment_attack, read_json  # noqa: E402


DEFAULT_LEVEL_CONFIG = PROJECT_ROOT / "configs" / "levels" / "ppo_difficulty" / "level2_medium.json"
DEFAULT_TASKS = (
    PROJECT_ROOT
    / "runs"
    / "rl_baselines"
    / "ppo"
    / "level2_medium_shock_recovery_5seeds"
    / "seed0"
    / "splits"
    / "validation_tasks.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "paper_figures" / "belief_map_corruption"

RISK_LAYERS = ("energy", "hazard", "communication", "illumination")
LAYER_PANELS = ("composite", "energy", "hazard", "hardmask")
LAYER_TITLES = {
    "composite": "Composite belief cost",
    "energy": "Energy cost",
    "hazard": "Hazard cost",
    "communication": "Communication cost",
    "illumination": "Illumination cost",
    "uncertainty": "Mean uncertainty",
    "hardmask": "Corruption hard mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-config", type=Path, default=DEFAULT_LEVEL_CONFIG)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default="level2_medium_clean_vs_corrupted_belief")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def mean_uncertainty(costmap) -> np.ndarray:
    fields = [np.asarray(costmap.uncertainty_layers[name], dtype=np.float32) for name in RISK_LAYERS]
    return np.mean(np.stack(fields, axis=0), axis=0).astype(np.float32)


def composite_risk(costmap) -> np.ndarray:
    fields = [np.asarray(costmap.layers[name], dtype=np.float32) for name in RISK_LAYERS]
    return np.mean(np.stack(fields, axis=0), axis=0).astype(np.float32)


def layer_array(costmap, name: str) -> np.ndarray:
    if name == "composite":
        return composite_risk(costmap)
    if name == "uncertainty":
        return mean_uncertainty(costmap)
    return np.asarray(costmap.layers[name], dtype=np.float32)


def masked_for_plot(values: np.ndarray, obstacle_mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.array(np.asarray(values, dtype=np.float32), mask=np.asarray(obstacle_mask, dtype=bool))


def add_map_overlays(ax: plt.Axes, costmap, attack_mask: np.ndarray | None, show_mask: bool = True) -> None:
    obstacle = np.asarray(costmap.obstacle_mask, dtype=bool)
    if obstacle.any():
        ax.contour(obstacle.astype(float), levels=[0.5], colors="black", linewidths=0.45, alpha=0.55)
    if show_mask and attack_mask is not None and bool(np.asarray(attack_mask, dtype=bool).any()):
        ax.contour(np.asarray(attack_mask, dtype=float), levels=[0.5], colors="#d62728", linewidths=1.1)

    start_row, start_col = costmap.start
    goal_row, goal_col = costmap.goal
    ax.scatter([start_col], [start_row], marker="o", s=96, c="#2ca02c", edgecolors="white", linewidths=1.2, zorder=5)
    ax.scatter([goal_col], [goal_row], marker="*", s=165, c="#1f77b4", edgecolors="white", linewidths=1.0, zorder=5)
    stroke = [pe.withStroke(linewidth=2.4, foreground="black", alpha=0.8)]
    ax.text(
        start_col + 1.8,
        start_row + 1.8,
        "start",
        color="white",
        fontsize=9,
        weight="bold",
        path_effects=stroke,
        zorder=6,
    )
    ax.text(
        goal_col + 1.8,
        goal_row + 1.8,
        "goal",
        color="white",
        fontsize=9,
        weight="bold",
        path_effects=stroke,
        zorder=6,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def save_side_by_side(clean_episode, corrupted_episode, output_dir: Path, prefix: str) -> None:
    clean_map = clean_episode.costmap
    corrupted_map = corrupted_episode.costmap
    attack_mask = getattr(corrupted_map, "attack_mask", None)
    clean_values = composite_risk(clean_map)
    corrupted_values = composite_risk(corrupted_map)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#222222")
    images = []
    for ax, values, title, show_mask in (
        (axes[0], clean_values, "Clean terrain belief", False),
        (axes[1], corrupted_values, "Corrupted terrain belief", True),
    ):
        image = ax.imshow(masked_for_plot(values, clean_map.obstacle_mask), cmap=cmap, vmin=0.0, vmax=1.0)
        images.append(image)
        ax.set_title(title, fontsize=12)
        add_map_overlays(ax, clean_map, attack_mask, show_mask=show_mask)

    cbar = fig.colorbar(images[-1], ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("Planner-visible composite cost", rotation=90)
    fig.suptitle("Global route-planning belief map before and after corruption", fontsize=13)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{prefix}_summary.png", dpi=260)
    fig.savefig(output_dir / f"{prefix}_summary.pdf")
    plt.close(fig)


def save_layer_breakdown(clean_episode, corrupted_episode, output_dir: Path, prefix: str) -> None:
    clean_map = clean_episode.costmap
    corrupted_map = corrupted_episode.costmap
    attack_mask = getattr(corrupted_map, "attack_mask", None)

    fig, axes = plt.subplots(2, len(LAYER_PANELS), figsize=(14.2, 6.7), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#222222")
    mask_cmap = plt.get_cmap("Reds").copy()
    mask_cmap.set_bad(color="#222222")
    for col, layer in enumerate(LAYER_PANELS):
        for row, (costmap, row_label, show_mask) in enumerate(
            (
                (clean_map, "Clean", False),
                (corrupted_map, "Corrupted", True),
            )
        ):
            ax = axes[row, col]
            if layer == "hardmask":
                values = (
                    np.zeros_like(clean_map.obstacle_mask, dtype=np.float32)
                    if row == 0 or attack_mask is None
                    else np.asarray(attack_mask, dtype=np.float32)
                )
                image = ax.imshow(
                    masked_for_plot(values, clean_map.obstacle_mask),
                    cmap=mask_cmap,
                    vmin=0.0,
                    vmax=1.0,
                )
            else:
                values = layer_array(costmap, layer)
                image = ax.imshow(masked_for_plot(values, clean_map.obstacle_mask), cmap=cmap, vmin=0.0, vmax=1.0)
            title = LAYER_TITLES[layer] if row == 0 else ""
            if title:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=11)
            add_map_overlays(ax, clean_map, attack_mask, show_mask=show_mask and layer != "hardmask")
            cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=7)
            if layer == "hardmask":
                cbar.set_ticks([0.0, 1.0])
                cbar.set_ticklabels(["0", "1"])

    fig.suptitle("Planner-visible belief layers before and after corruption", fontsize=13)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{prefix}_layers.png", dpi=260)
    fig.savefig(output_dir / f"{prefix}_layers.pdf")
    plt.close(fig)


def save_metadata(clean_episode, corrupted_episode, output_dir: Path, prefix: str, level_config: dict, task: dict) -> None:
    clean_map = clean_episode.costmap
    corrupted_map = corrupted_episode.costmap
    attack_mask = getattr(corrupted_map, "attack_mask", None)
    metadata = dict(getattr(corrupted_map, "attack_metadata", None) or {})
    if attack_mask is not None:
        metadata["visualized_corrupted_cells"] = int(np.asarray(attack_mask, dtype=bool).sum())
    metadata["level"] = level_config.get("level")
    metadata["difficulty"] = level_config.get("difficulty")
    metadata["tile_id"] = level_config.get("tile_id")
    metadata["task_id"] = task.get("task_id")
    metadata["start"] = list(clean_map.start)
    metadata["goal"] = list(clean_map.goal)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{prefix}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    level_config_path = resolve(args.level_config)
    tasks_path = resolve(args.tasks)
    output_dir = resolve(args.output_dir)

    level_config = read_json(level_config_path)
    raw_layers = load_real_layers(resolve(Path(str(level_config["map_source"]))))
    tasks = load_task_split(tasks_path)
    if not tasks:
        raise ValueError(f"no tasks found in {tasks_path}")
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
    env_attack = load_environment_attack(level_config)
    corrupted_episode = apply_environment_attack_to_episode(
        clean_episode,
        env_attack,
        np.random.default_rng(int(args.seed) + 20_000),
    )

    save_side_by_side(clean_episode, corrupted_episode, output_dir, args.prefix)
    save_layer_breakdown(clean_episode, corrupted_episode, output_dir, args.prefix)
    save_metadata(clean_episode, corrupted_episode, output_dir, args.prefix, level_config, task)
    print(output_dir / f"{args.prefix}_summary.png")
    print(output_dir / f"{args.prefix}_layers.png")
    print(output_dir / f"{args.prefix}_metadata.json")


if __name__ == "__main__":
    main()
