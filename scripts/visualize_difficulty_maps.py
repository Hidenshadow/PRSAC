"""Visualize the current 3-level x 3-difficulty map benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LightSource, LinearSegmentedColormap
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.attack_wrappers import apply_environment_attack_to_episode  # noqa: E402
from maps.map_generator import GeneratedCostMap, generate_costmap  # noqa: E402
from maps.real_terrain import (  # noqa: E402
    build_corridor_risk_field,
    load_real_layers,
    make_real_planning_episode,
    read_json as read_plain_json,
    sample_real_tasks,
)
from run_lunar_viper_staged_recovery import read_json  # noqa: E402
from utils.metrics import make_planning_episode_from_costmap  # noqa: E402


LEVEL_LABELS = {
    "level1": "Level 1 Synthetic Cost-Map",
    "level2": "Level 2 Lunar NPD DEM",
    "level3": "Level 3 Mars DTEED DTM",
}
DIFFICULTY_LABELS = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}
LEVEL_CONFIGS = {
    (level, difficulty): PROJECT_ROOT / "configs" / "levels" / "ppo_difficulty" / f"{level}_{difficulty}.json"
    for level in ("level1", "level2", "level3")
    for difficulty in ("easy", "medium", "hard")
}
DOMAIN_STYLES = {
    "level1": {
        "badge": "SIMULATED COST-MAP",
        "source": "procedural multi-layer cost map",
        "badge_color": "#0f766e",
        "spine_color": "#0f766e",
        "route_color": "#67e8f9",
        "base_cmap": LinearSegmentedColormap.from_list(
            "synthetic_costmap",
            ["#101828", "#164e63", "#0f766e", "#84cc16", "#facc15", "#fb7185"],
        ),
        "risk_cmap": "cool",
        "risk_alpha": 0.30,
        "hillshade": False,
        "grid": True,
    },
    "level2": {
        "badge": "LUNAR NPD DEM",
        "source": "real lunar polar terrain",
        "badge_color": "#64748b",
        "spine_color": "#64748b",
        "route_color": "#38bdf8",
        "base_cmap": LinearSegmentedColormap.from_list(
            "lunar_regolith",
            ["#111827", "#374151", "#6b7280", "#a3a3a3", "#d4d4d8", "#f8fafc"],
        ),
        "risk_cmap": "magma",
        "risk_alpha": 0.36,
        "hillshade": True,
        "grid": False,
    },
    "level3": {
        "badge": "MARS DTEED DTM",
        "source": "real Mars DTEED terrain",
        "badge_color": "#b45309",
        "spine_color": "#b45309",
        "route_color": "#99f6e4",
        "base_cmap": LinearSegmentedColormap.from_list(
            "mars_dust",
            ["#1f130f", "#5f2d1d", "#a8551f", "#d97706", "#c08457", "#fed7aa"],
        ),
        "risk_cmap": "plasma",
        "risk_alpha": 0.32,
        "hillshade": True,
        "grid": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create publication-style map visualizations.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "figures" / "map_visualizations",
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def unit(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    valid = array[finite]
    lo = float(np.percentile(valid, 2))
    hi = float(np.percentile(valid, 98))
    if hi - lo < 1e-8:
        out = np.zeros_like(array, dtype=np.float32)
    else:
        out = (array - lo) / (hi - lo)
    out[~finite] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def obstacle_edges(mask: np.ndarray) -> np.ndarray:
    obstacle = np.asarray(mask, dtype=bool)
    padded = np.pad(obstacle, 1, mode="constant", constant_values=False)
    neighbors = (
        padded[:-2, 1:-1]
        | padded[2:, 1:-1]
        | padded[1:-1, :-2]
        | padded[1:-1, 2:]
    )
    return obstacle & ~neighbors | (obstacle ^ neighbors)


def synthetic_map(config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["fixed_map_seed"]))
    costmap: GeneratedCostMap = generate_costmap(
        map_size=int(config["map_size"]),
        rng=rng,
        scenario=str(config["scenario"]),
        min_start_goal_distance_ratio=float(config["min_distance_ratio"]),
    )
    episode = make_planning_episode_from_costmap(costmap, np.random.default_rng(0))
    hazard = np.asarray(costmap.layers["hazard"], dtype=np.float32)
    uncertainty = np.asarray(costmap.uncertainty_layers["hazard"], dtype=np.float32)
    slope = np.asarray(costmap.slope_layer, dtype=np.float32)
    risk = np.clip(0.45 * unit(hazard) + 0.35 * unit(slope) + 0.20 * unit(uncertainty), 0.0, 1.0)
    return {
        "height": unit(costmap.height_map),
        "layers": {name: np.asarray(value, dtype=np.float32) for name, value in costmap.layers.items()},
        "uncertainty_layers": {
            name: np.asarray(value, dtype=np.float32)
            for name, value in costmap.uncertainty_layers.items()
        },
        "risk": risk,
        "obstacle": np.asarray(costmap.obstacle_mask, dtype=bool),
        "start": costmap.start,
        "goal": costmap.goal,
        "size_label": f"{int(config['map_size'])}x{int(config['map_size'])} cells",
        "metric_label": f"scenario: {config['scenario']}",
        "scale_label": "10 cells",
        "scale_cells": min(10, max(int(config["map_size"]) // 4, 1)),
        "domain_source": "procedural multi-objective cost map",
        "episode": episode,
    }


def real_map(config: dict[str, Any]) -> dict[str, Any]:
    layers_path = PROJECT_ROOT / str(config["map_source"])
    metadata_path = PROJECT_ROOT / str(config["metadata"])
    raw_layers = load_real_layers(layers_path)
    metadata = read_plain_json(metadata_path)
    risk = build_corridor_risk_field(raw_layers, risk_weights=config.get("corridor_risk_weights"))
    task = sample_real_tasks(
        raw_layers,
        count=1,
        seed=0,
        tile_id=str(config["tile_id"]),
        split="viz",
        min_distance_ratio=float(config["min_distance_ratio"]),
        meters_per_pixel=float(metadata.get("meters_per_pixel", 1.0)),
        task_sampling_mode=str(config.get("task_sampling_mode", "distance")),
        min_corridor_risk=(
            float(config["min_corridor_risk"])
            if config.get("min_corridor_risk") is not None
            else None
        ),
        corridor_radius=int(config.get("corridor_radius", 2)),
        candidate_pool_multiplier=int(config.get("candidate_pool_multiplier", 30)),
        risk_weights=config.get("corridor_risk_weights"),
    )[0]
    episode = make_real_planning_episode(
        raw_layers,
        task,
        np.random.default_rng(0),
        scenario=str(config["scenario"]),
        mission_profile_scenario=config.get("mission_profile_scenario"),
    )
    height = raw_layers.get("height_map", raw_layers.get("height_norm"))
    slope = raw_layers.get("slope_degrees", raw_layers.get("slope_layer"))
    obstacle = np.asarray(raw_layers["obstacle_mask"], dtype=bool)
    meters_per_pixel = float(metadata.get("meters_per_pixel", 1.0))
    pixels = int(metadata.get("tile_pixels", obstacle.shape[0]))
    tile_meters = float(metadata.get("tile_meters", pixels * meters_per_pixel))
    slope_p95 = float(metadata.get("slope_p95_deg", np.nan))
    relief = float(metadata.get("relief_p95_p05_m", np.nan))
    scale_meters = 100 if tile_meters <= 450 else 200
    scale_cells = max(int(round(scale_meters / max(meters_per_pixel, 1e-6))), 1)
    return {
        "height": np.asarray(height, dtype=np.float32),
        "layers": {
            "distance": np.asarray(raw_layers["layer_distance"], dtype=np.float32),
            "energy": np.asarray(raw_layers["layer_energy"], dtype=np.float32),
            "hazard": np.asarray(raw_layers["layer_hazard"], dtype=np.float32),
            "communication": np.asarray(raw_layers["layer_communication"], dtype=np.float32),
            "illumination": np.asarray(raw_layers["layer_illumination"], dtype=np.float32),
        },
        "uncertainty_layers": {
            "distance": np.asarray(raw_layers["uncertainty_distance"], dtype=np.float32),
            "energy": np.asarray(raw_layers["uncertainty_energy"], dtype=np.float32),
            "hazard": np.asarray(raw_layers["uncertainty_hazard"], dtype=np.float32),
            "communication": np.asarray(raw_layers["uncertainty_communication"], dtype=np.float32),
            "illumination": np.asarray(raw_layers["uncertainty_illumination"], dtype=np.float32),
        },
        "risk": risk,
        "obstacle": obstacle,
        "start": tuple(task["start"]),
        "goal": tuple(task["goal"]),
        "size_label": f"{pixels}x{pixels} cells, {tile_meters:.0f} m",
        "metric_label": f"slope p95 {slope_p95:.2f} deg, relief {relief:.1f} m",
        "scale_label": f"{scale_meters} m",
        "scale_cells": min(scale_cells, max(pixels - 4, 1)),
        "slope": slope,
        "domain_source": str(metadata.get("source_dem", "")),
        "episode": episode,
    }


def load_map(level: str, difficulty: str) -> dict[str, Any]:
    config = read_json(LEVEL_CONFIGS[(level, difficulty)])
    if level == "level1":
        data = synthetic_map(config)
    else:
        data = real_map(config)
    data["config"] = config
    data["level"] = level
    data["difficulty"] = difficulty
    return data


def draw_map(ax: plt.Axes, data: dict[str, Any], title: str, subtitle: str, compact: bool = False) -> None:
    height = np.asarray(data["height"], dtype=np.float32)
    risk = np.asarray(data["risk"], dtype=np.float32)
    obstacle = np.asarray(data["obstacle"], dtype=bool)
    size = int(obstacle.shape[0])
    style = DOMAIN_STYLES[str(data["level"])]

    light = LightSource(azdeg=315, altdeg=45)
    base_cmap = style["base_cmap"]
    if bool(style["hillshade"]):
        try:
            shaded = light.shade(height, cmap=base_cmap, vert_exag=0.8, blend_mode="soft")
        except Exception:
            shaded = base_cmap(unit(height))
    else:
        shaded = base_cmap(unit(0.58 * height + 0.42 * risk))
    ax.imshow(shaded, interpolation="nearest")
    ax.imshow(
        risk,
        cmap=str(style["risk_cmap"]),
        alpha=float(style["risk_alpha"]),
        interpolation="nearest",
        vmin=0.0,
        vmax=1.0,
    )

    if obstacle.any():
        ax.contour(obstacle.astype(float), levels=[0.5], colors=[(0.04, 0.04, 0.04, 0.85)], linewidths=1.0)
        ax.imshow(np.ma.masked_where(~obstacle, obstacle), cmap="gray_r", alpha=0.32, interpolation="nearest")
    if bool(style["grid"]) and size <= 80:
        grid_color = (1.0, 1.0, 1.0, 0.10)
        ax.set_xticks(np.arange(-0.5, size, 5), minor=True)
        ax.set_yticks(np.arange(-0.5, size, 5), minor=True)
        ax.grid(which="minor", color=grid_color, linewidth=0.35)

    start = tuple(map(int, data["start"]))
    goal = tuple(map(int, data["goal"]))
    ax.plot([start[1], goal[1]], [start[0], goal[0]], color="white", lw=2.8, alpha=0.92, solid_capstyle="round")
    ax.plot(
        [start[1], goal[1]],
        [start[0], goal[0]],
        color=str(style["route_color"]),
        lw=1.5,
        alpha=0.95,
        solid_capstyle="round",
    )
    ax.scatter([start[1]], [start[0]], s=42 if compact else 62, c="#22c55e", edgecolors="white", linewidths=1.0, zorder=5)
    ax.scatter([goal[1]], [goal[0]], s=50 if compact else 76, marker="*", c="#f97316", edgecolors="white", linewidths=0.9, zorder=5)

    scale_cells = int(data.get("scale_cells", max(size // 4, 1)))
    scale_cells = max(min(scale_cells, size - 5), 1)
    y = size - 4
    x0 = 3
    x1 = min(x0 + scale_cells, size - 3)
    ax.plot([x0, x1], [y, y], color="white", lw=3.0, solid_capstyle="butt")
    ax.plot([x0, x1], [y, y], color="black", lw=1.2, solid_capstyle="butt", alpha=0.75)
    ax.text(
        (x0 + x1) / 2,
        y - 1.2,
        str(data.get("scale_label", "")),
        ha="center",
        va="bottom",
        color="white",
        fontsize=7 if compact else 8,
        weight="bold",
        path_effects=[],
    )

    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.8)
        spine.set_edgecolor(str(style["spine_color"]))
    ax.set_title(
        title,
        loc="left",
        fontsize=10 if compact else 13,
        weight="bold",
        pad=8,
        color=str(style["spine_color"]),
    )
    ax.text(
        0.02,
        0.965,
        str(style["badge"]),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.9 if compact else 8.4,
        color="white",
        weight="bold",
        bbox={"facecolor": str(style["badge_color"]), "alpha": 0.88, "edgecolor": "none", "pad": 3.0},
    )
    if subtitle:
        ax.text(
            0.02,
            0.03,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.3 if compact else 9,
            color="white",
            bbox={"facecolor": "#111827", "alpha": 0.70, "edgecolor": "none", "pad": 3.0},
        )


def hard_attack_mask(data: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    config = data["config"]
    environment_attack = config.get("attacks", {}).get("environment", {})
    episode = data.get("episode")
    obstacle = np.asarray(data["obstacle"], dtype=bool)
    if episode is None or not environment_attack.get("enabled", False):
        return np.zeros_like(obstacle, dtype=bool), {}
    attacked = apply_environment_attack_to_episode(
        episode,
        environment_attack,
        np.random.default_rng(0),
    )
    mask = getattr(attacked.costmap, "attack_mask", None)
    if mask is None:
        return np.zeros_like(obstacle, dtype=bool), {}
    metadata = getattr(attacked.costmap, "attack_metadata", None) or {}
    return np.asarray(mask, dtype=bool) & ~obstacle, dict(metadata)


def redraw_route_markers(ax: plt.Axes, data: dict[str, Any], compact: bool = False) -> None:
    style = DOMAIN_STYLES[str(data["level"])]
    start = tuple(map(int, data["start"]))
    goal = tuple(map(int, data["goal"]))
    ax.plot(
        [start[1], goal[1]],
        [start[0], goal[0]],
        color="white",
        lw=3.2,
        alpha=0.94,
        solid_capstyle="round",
        zorder=8,
    )
    ax.plot(
        [start[1], goal[1]],
        [start[0], goal[0]],
        color=str(style["route_color"]),
        lw=1.7,
        alpha=0.98,
        solid_capstyle="round",
        zorder=9,
    )
    ax.scatter([start[1]], [start[0]], s=46 if compact else 68, c="#22c55e", edgecolors="white", linewidths=1.0, zorder=10)
    ax.scatter([goal[1]], [goal[0]], s=56 if compact else 82, marker="*", c="#f97316", edgecolors="white", linewidths=0.9, zorder=10)


def draw_hard_mask(ax: plt.Axes, data: dict[str, Any]) -> tuple[int, float, dict[str, Any]]:
    obstacle = np.asarray(data["obstacle"], dtype=bool)
    free_cells = int((~obstacle).sum())
    mask, metadata = hard_attack_mask(data)
    mask_cells = int(mask.sum())
    mask_fraction = 100.0 * mask_cells / max(free_cells, 1)
    draw_map(
        ax,
        data,
        f"{LEVEL_LABELS[str(data['level'])]} / Hard Mask",
        "",
        compact=True,
    )
    if mask.any():
        ax.imshow(
            np.ma.masked_where(~mask, mask),
            cmap=LinearSegmentedColormap.from_list("attack_mask_red", ["#ef4444", "#ef4444"]),
            alpha=0.58,
            interpolation="nearest",
            zorder=6,
        )
        ax.contour(mask.astype(float), levels=[0.5], colors=[(1.0, 1.0, 1.0, 0.78)], linewidths=0.9, zorder=7)
    if obstacle.any():
        ax.contour(obstacle.astype(float), levels=[0.5], colors=[(0.02, 0.02, 0.02, 0.92)], linewidths=1.0, zorder=7)
    redraw_route_markers(ax, data, compact=True)
    ax.text(
        0.5,
        -0.105,
        f"masked attack area: {mask_cells} cells ({mask_fraction:.1f}% free)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.4,
        color="#334155",
        weight="bold",
    )
    return mask_cells, mask_fraction, metadata


def make_overview(maps: dict[tuple[str, str], dict[str, Any]], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.2, 13.4), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    for row, level in enumerate(("level1", "level2", "level3")):
        for col, difficulty in enumerate(("easy", "medium", "hard")):
            data = maps[(level, difficulty)]
            title = f"{LEVEL_LABELS[level]} / {DIFFICULTY_LABELS[difficulty]}"
            subtitle = f"{data['size_label']} | {data['metric_label']}"
            draw_map(axes[row, col], data, title, subtitle, compact=True)
    fig.suptitle(
        "40/60/80 Map Benchmark Across Synthetic, Lunar, and Mars Domains",
        fontsize=20,
        weight="bold",
        x=0.02,
        ha="left",
    )
    fig.savefig(output_dir / "difficulty_maps_3x3.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "difficulty_maps_3x3.pdf", bbox_inches="tight")
    plt.close(fig)


def make_individual(maps: dict[tuple[str, str], dict[str, Any]], output_dir: Path, dpi: int) -> None:
    for (level, difficulty), data in maps.items():
        fig, ax = plt.subplots(figsize=(7.2, 7.2), constrained_layout=True)
        fig.patch.set_facecolor("#f8fafc")
        title = f"{LEVEL_LABELS[level]} / {DIFFICULTY_LABELS[difficulty]}"
        subtitle = f"{data['size_label']} | {data['metric_label']}"
        draw_map(ax, data, title, subtitle, compact=False)
        fig.savefig(output_dir / f"{level}_{difficulty}_map.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(output_dir / f"{level}_{difficulty}_map.pdf", bbox_inches="tight")
        plt.close(fig)


def make_hard_mask_overview(maps: dict[tuple[str, str], dict[str, Any]], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), constrained_layout=False)
    fig.patch.set_facecolor("#f8fafc")
    for ax, level in zip(axes, ("level1", "level2", "level3"), strict=True):
        _mask_cells, _mask_fraction, metadata = draw_hard_mask(ax, maps[(level, "hard")])
        if metadata:
            ax.text(
                0.02,
                0.885,
                str(metadata.get("composite_attack_name", "composite attack")).replace("_", " "),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.2,
                color="white",
                bbox={"facecolor": "#7f1d1d", "alpha": 0.74, "edgecolor": "none", "pad": 2.6},
            )
    fig.subplots_adjust(left=0.04, right=0.985, top=0.76, bottom=0.23, wspace=0.12)
    fig.suptitle(
        "Hard Setting Masked Areas from the Composite Map-Belief Attack",
        fontsize=17,
        weight="bold",
        x=0.02,
        y=0.96,
        ha="left",
    )
    legend_handles = [
        Patch(facecolor="#ef4444", edgecolor="white", alpha=0.58, label="attack masked area"),
        Patch(facecolor="#111827", edgecolor="#111827", alpha=0.82, label="hard obstacle"),
        Patch(facecolor="#22c55e", edgecolor="white", label="start"),
        Patch(facecolor="#f97316", edgecolor="white", label="goal"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.045),
        fontsize=9,
    )
    fig.savefig(output_dir / "hard_attack_masked_areas.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "hard_attack_masked_areas.pdf", bbox_inches="tight")
    plt.close(fig)


def make_hard_costmap_overview(maps: dict[tuple[str, str], dict[str, Any]], output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.0), constrained_layout=False)
    fig.patch.set_facecolor("#f8fafc")
    for ax, level in zip(axes, ("level1", "level2", "level3"), strict=True):
        data = maps[(level, "hard")]
        title = f"{LEVEL_LABELS[level]} / Hard"
        subtitle = f"{data['size_label']} | {data['metric_label']}"
        draw_map(ax, data, title, subtitle, compact=True)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.78, bottom=0.13, wspace=0.12)
    fig.suptitle(
        "Hard Test Costmaps Across the Three Benchmark Levels",
        fontsize=17,
        weight="bold",
        x=0.02,
        y=0.96,
        ha="left",
    )
    legend_handles = [
        Patch(facecolor="#22c55e", edgecolor="white", label="start"),
        Patch(facecolor="#f97316", edgecolor="white", label="goal"),
        Patch(facecolor="#111827", edgecolor="#111827", alpha=0.82, label="obstacle"),
        Patch(facecolor="#fb7185", edgecolor="none", alpha=0.55, label="risk/cost overlay"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=9,
    )
    fig.savefig(output_dir / "hard_costmaps_1x3.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "hard_costmaps_1x3.pdf", bbox_inches="tight")
    plt.close(fig)


def composite_uncertainty(data: dict[str, Any]) -> np.ndarray:
    uncertainty_layers = data.get("uncertainty_layers", {})
    layer_values = [
        np.asarray(uncertainty_layers[name], dtype=np.float32)
        for name in ("energy", "hazard", "communication", "illumination")
        if name in uncertainty_layers
    ]
    if not layer_values:
        return np.zeros_like(np.asarray(data["height"], dtype=np.float32), dtype=np.float32)
    return np.clip(np.mean(np.stack(layer_values, axis=0), axis=0), 0.0, 1.0).astype(np.float32)


def draw_layer_panel(
    ax: plt.Axes,
    data: dict[str, Any],
    values: np.ndarray,
    cmap: str,
    title: str,
    show_row_label: bool = False,
) -> None:
    array = np.asarray(values, dtype=np.float32)
    obstacle = np.asarray(data["obstacle"], dtype=bool)
    size = int(obstacle.shape[0])
    ax.imshow(unit(array), cmap=cmap, interpolation="nearest", vmin=0.0, vmax=1.0)
    if obstacle.any():
        ax.imshow(np.ma.masked_where(~obstacle, obstacle), cmap="gray_r", alpha=0.38, interpolation="nearest")
        ax.contour(obstacle.astype(float), levels=[0.5], colors=[(0.04, 0.04, 0.04, 0.85)], linewidths=0.75)
    start = tuple(map(int, data["start"]))
    goal = tuple(map(int, data["goal"]))
    ax.plot([start[1], goal[1]], [start[0], goal[0]], color="white", lw=1.8, alpha=0.82, solid_capstyle="round")
    ax.plot([start[1], goal[1]], [start[0], goal[0]], color="#38bdf8", lw=0.9, alpha=0.92, solid_capstyle="round")
    ax.scatter([start[1]], [start[0]], s=18, c="#22c55e", edgecolors="white", linewidths=0.6, zorder=5)
    ax.scatter([goal[1]], [goal[0]], s=24, marker="*", c="#f97316", edgecolors="white", linewidths=0.5, zorder=5)
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(size - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_edgecolor("#cbd5e1")
    ax.set_title(title, fontsize=9.2, weight="bold", pad=5, color="#0f172a")
    if show_row_label:
        style = DOMAIN_STYLES[str(data["level"])]
        ax.text(
            -0.08,
            0.5,
            f"{LEVEL_LABELS[str(data['level'])]}\nHard",
            transform=ax.transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=9.5,
            weight="bold",
            color=str(style["spine_color"]),
        )


def make_hard_layer_overview(maps: dict[tuple[str, str], dict[str, Any]], output_dir: Path, dpi: int) -> None:
    columns = [
        ("terrain", "Terrain", "gray"),
        ("energy", "Energy", "viridis"),
        ("hazard", "Hazard", "magma"),
        ("communication", "Communication", "cividis"),
        ("illumination", "Illumination", "plasma"),
        ("uncertainty", "Uncertainty", "rocket_r" if "rocket_r" in plt.colormaps() else "YlOrRd"),
    ]
    fig, axes = plt.subplots(3, len(columns), figsize=(16.0, 8.5), constrained_layout=False)
    fig.patch.set_facecolor("#f8fafc")
    for row, level in enumerate(("level1", "level2", "level3")):
        data = maps[(level, "hard")]
        layers = data.get("layers", {})
        layer_arrays = {
            "terrain": np.asarray(data["height"], dtype=np.float32),
            "energy": np.asarray(layers.get("energy", data["risk"]), dtype=np.float32),
            "hazard": np.asarray(layers.get("hazard", data["risk"]), dtype=np.float32),
            "communication": np.asarray(layers.get("communication", data["risk"]), dtype=np.float32),
            "illumination": np.asarray(layers.get("illumination", data["risk"]), dtype=np.float32),
            "uncertainty": composite_uncertainty(data),
        }
        for col, (key, title, cmap) in enumerate(columns):
            draw_layer_panel(
                axes[row, col],
                data,
                layer_arrays[key],
                cmap,
                title if row == 0 else "",
                show_row_label=(col == 0),
            )
            if col == 0:
                axes[row, col].text(
                    0.02,
                    0.03,
                    str(data["size_label"]),
                    transform=axes[row, col].transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.0,
                    color="white",
                    bbox={"facecolor": "#111827", "alpha": 0.70, "edgecolor": "none", "pad": 2.2},
                )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.08, wspace=0.055, hspace=0.12)
    fig.suptitle(
        "Hard Test Map Layers by Benchmark Level",
        fontsize=18,
        weight="bold",
        x=0.02,
        y=0.97,
        ha="left",
    )
    fig.text(
        0.02,
        0.925,
        "Rows are levels; columns show the terrain/objective cost layers used by the planner. Higher color intensity indicates larger normalized cost or uncertainty.",
        fontsize=9.5,
        color="#334155",
        ha="left",
        va="top",
    )
    fig.savefig(output_dir / "hard_map_layers_3x6.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / "hard_map_layers_3x6.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    maps = {
        (level, difficulty): load_map(level, difficulty)
        for level in ("level1", "level2", "level3")
        for difficulty in ("easy", "medium", "hard")
    }
    make_overview(maps, output_dir, args.dpi)
    make_individual(maps, output_dir, args.dpi)
    make_hard_costmap_overview(maps, output_dir, args.dpi)
    make_hard_layer_overview(maps, output_dir, args.dpi)
    make_hard_mask_overview(maps, output_dir, args.dpi)
    print(f"Wrote map visualizations to {output_dir}")
    print(f"Overview: {output_dir / 'difficulty_maps_3x3.png'}")
    print(f"Hard costmaps: {output_dir / 'hard_costmaps_1x3.png'}")
    print(f"Hard map layers: {output_dir / 'hard_map_layers_3x6.png'}")
    print(f"Hard attack masked areas: {output_dir / 'hard_attack_masked_areas.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
