#!/usr/bin/env python
"""Plot clean training -> attack drop -> recovery for PPO/SAC and PPO-anchored BVR.

BVR runs are aligned onto the recovery phase for visual comparison with PPO/SAC.
For a fixed protocol axis, BVR performance is normalized by the source PPO
nominal shock clean cost, not by BVR's anchor policy clean cost.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BVR_ROOT = PROJECT_ROOT / "runs" / "bvr" / "ppo_anchor_benefit_easy_2seed_ppo_matched_20260620"
DEFAULT_RL_ROOT = PROJECT_ROOT / "runs" / "rl_baselines"
DEFAULT_SCENARIOS = ("level1_easy", "level2_easy", "level3_easy")
DEFAULT_SEEDS = (0, 1)

LEVEL_LABELS = {
    "level1": "Level 1 synthetic",
    "level2": "Level 2 lunar",
    "level3": "Level 3 Mars",
}

STYLES = {
    "ppo": {"label": "PPO", "color": "#2f6b9a", "linestyle": "-"},
    "sac": {"label": "SAC", "color": "#2a7f62", "linestyle": "--"},
    "bvr": {"label": "PPO-anchor BVR", "color": "#7a3db8", "linestyle": "-"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvr-root", type=Path, default=DEFAULT_BVR_ROOT)
    parser.add_argument("--rl-root", type=Path, default=DEFAULT_RL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--y-limits", type=float, nargs=2, default=[70.0, 103.0])
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", str(path))
    return int(match.group(1)) if match else -1


def scenario_parts(scenario: str) -> tuple[str, str]:
    parts = scenario.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"scenario must look like level2_easy, got {scenario!r}")
    return parts[0], parts[1]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def rl_experiment_dir(rl_root: Path, algorithm: str, scenario: str) -> Path:
    return rl_root / algorithm / f"{scenario}_shock_recovery_5seeds"


def seed_dirs(experiment: Path, seeds: list[int]) -> list[Path]:
    return [experiment / f"seed{seed}" for seed in seeds if (experiment / f"seed{seed}").exists()]


def protocol_timesteps(seed_dir: Path) -> tuple[int, int]:
    config = read_json(seed_dir / "run_config.json")
    args = config.get("command_args", {}) if isinstance(config.get("command_args"), dict) else {}
    return int(args.get("nominal_timesteps", 50_000)), int(args.get("recovery_timesteps", 20_480))


def map_label(seed_dir: Path, scenario: str) -> str:
    level, difficulty = scenario_parts(scenario)
    size = {"easy": 40, "medium": 60, "hard": 80}.get(difficulty, 40)
    config = read_json(seed_dir / "run_config.json")
    level_config = config.get("level_config", {}) if isinstance(config.get("level_config"), dict) else {}
    description = str(level_config.get("description", ""))
    match = re.search(r"(\d+)x\1", description)
    if match:
        size = int(match.group(1))
    return f"{LEVEL_LABELS.get(level, level)}\n{difficulty.title()} ({size}x{size})"


def normalized_training(seed_directories: list[Path], smooth_window: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed_dir in seed_directories:
        csv_path = seed_dir / "nominal_training_eval.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        if not {"global_step", "mean_scalar_cost"}.issubset(frame.columns):
            continue
        frame = frame[["global_step", "mean_scalar_cost"]].dropna().sort_values("global_step")
        if frame.empty:
            continue
        final_cost = float(frame.iloc[-1]["mean_scalar_cost"])
        if not math.isfinite(final_cost) or final_cost <= 0:
            continue
        frame = frame.copy()
        frame["seed"] = seed_from_path(seed_dir)
        frame["performance_index"] = 100.0 * final_cost / frame["mean_scalar_cost"].clip(lower=1e-12)
        frames.append(frame[["seed", "global_step", "performance_index"]])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined.groupby("global_step")["performance_index"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "performance_mean", "std": "performance_std", "count": "num_samples"})
        .sort_values("global_step")
    )
    grouped["performance_smooth"] = grouped["performance_mean"].rolling(
        window=max(int(smooth_window), 1),
        center=True,
        min_periods=1,
    ).mean()
    return grouped


def fixed_clean_costs_from_shock(seed_dir: Path) -> dict[str, float]:
    curve_path = seed_dir / "shock_recovery_curve.csv"
    if not curve_path.exists():
        return {}
    curve = pd.read_csv(curve_path)
    required = {"eval_domain", "phase", "attack_type", "mean_attacked_scalar_cost"}
    if not required.issubset(curve.columns):
        return {}
    clean = curve[(curve["phase"] == "shock") & (curve["attack_type"] == "none")]
    return {
        str(row["eval_domain"]): float(row["mean_attacked_scalar_cost"])
        for _, row in clean.iterrows()
        if math.isfinite(float(row["mean_attacked_scalar_cost"]))
    }


def fixed_denominator_recovery(seed_directories: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed_dir in seed_directories:
        clean_by_domain = fixed_clean_costs_from_shock(seed_dir)
        curve_path = seed_dir / "shock_recovery_curve.csv"
        if not clean_by_domain or not curve_path.exists():
            continue
        curve = pd.read_csv(curve_path)
        required = {"eval_domain", "phase", "attack_type", "recovery_step", "mean_attacked_scalar_cost"}
        if not required.issubset(curve.columns):
            continue
        attacked = curve[
            (curve["attack_type"] == "environment")
            & (curve["phase"].isin(["shock", "recovery"]))
        ].copy()
        if attacked.empty:
            continue
        attacked["clean_nominal_cost"] = attacked["eval_domain"].astype(str).map(clean_by_domain)
        attacked = attacked.dropna(subset=["clean_nominal_cost", "mean_attacked_scalar_cost"])
        attacked["seed"] = seed_from_path(seed_dir)
        attacked["performance_index"] = (
            100.0
            * attacked["clean_nominal_cost"].astype(float)
            / attacked["mean_attacked_scalar_cost"].astype(float).clip(lower=1e-12)
        )
        frames.append(attacked[["seed", "eval_domain", "recovery_step", "performance_index"]])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby("recovery_step")["performance_index"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "performance_mean", "std": "performance_std", "count": "num_samples"})
        .sort_values("recovery_step")
    )


def bvr_repair_curve(bvr_root: Path, scenario: str, seeds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        seed_dir = bvr_root / scenario / f"seed{seed}"
        curve_path = seed_dir / "bvr_recovery_curve.csv"
        config = read_json(seed_dir / "bvr_run_config.json")
        source_run_dir = Path(str(config.get("source_run_dir", "")))
        if not source_run_dir.exists():
            continue
        clean_by_domain = fixed_clean_costs_from_shock(source_run_dir)
        if not clean_by_domain or not curve_path.exists():
            continue
        curve = pd.read_csv(curve_path)
        required = {"iteration", "eval_domain", "attack_type", "mean_attacked_scalar_cost"}
        if not required.issubset(curve.columns):
            continue
        attacked = curve[curve["attack_type"] == "environment"].copy()
        attacked["clean_nominal_cost"] = attacked["eval_domain"].astype(str).map(clean_by_domain)
        attacked = attacked.dropna(subset=["clean_nominal_cost", "mean_attacked_scalar_cost"])
        if attacked.empty:
            continue
        attacked["seed"] = seed
        attacked["performance_index"] = (
            100.0
            * attacked["clean_nominal_cost"].astype(float)
            / attacked["mean_attacked_scalar_cost"].astype(float).clip(lower=1e-12)
        )
        frames.append(attacked[["seed", "eval_domain", "iteration", "performance_index"]])
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return (
        combined.groupby("iteration")["performance_index"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "performance_mean", "std": "performance_std", "count": "num_samples"})
        .sort_values("iteration")
    )


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.22",
            "axes.labelcolor": "0.15",
            "xtick.color": "0.15",
            "ytick.color": "0.15",
            "font.size": 10,
        }
    )


def summary_row(algorithm: str, scenario: str, curve: pd.DataFrame, phase: str) -> dict[str, object]:
    if curve.empty:
        return {
            "algorithm": algorithm,
            "scenario": scenario,
            "phase": phase,
            "initial_index": math.nan,
            "final_index": math.nan,
            "best_index": math.nan,
        }
    return {
        "algorithm": algorithm,
        "scenario": scenario,
        "phase": phase,
        "initial_index": float(curve.iloc[0]["performance_mean"]),
        "final_index": float(curve.iloc[-1]["performance_mean"]),
        "best_index": float(curve["performance_mean"].max()),
    }


def main() -> int:
    args = parse_args()
    bvr_root = resolve(args.bvr_root)
    rl_root = resolve(args.rl_root)
    output_dir = resolve(args.output_dir) if args.output_dir else bvr_root / "protocol_figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed) for seed in args.seeds]

    setup_matplotlib()
    scenarios = list(args.scenarios)
    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.3 * len(scenarios), 4.8), squeeze=False, sharey=True)
    summary_rows: list[dict[str, object]] = []

    for index, scenario in enumerate(scenarios):
        ax = axes.ravel()[index]
        ppo_experiment = rl_experiment_dir(rl_root, "ppo", scenario)
        ppo_seed_dirs = seed_dirs(ppo_experiment, seeds)
        if not ppo_seed_dirs:
            ax.text(0.5, 0.5, "missing PPO source", transform=ax.transAxes, ha="center", va="center")
            ax.axis("off")
            continue
        nominal_end, recovery_timesteps = protocol_timesteps(ppo_seed_dirs[0])
        ax.axvspan(0, nominal_end, color="#f2f4f7", alpha=0.55, zorder=0)
        ax.axvspan(nominal_end, nominal_end + recovery_timesteps, color="#fff8ee", alpha=0.45, zorder=0)
        ax.axvline(nominal_end, color="#c23b22", linestyle="--", linewidth=1.1, alpha=0.85)
        ax.axhline(100.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.85)

        for algorithm in ("ppo", "sac"):
            experiment = rl_experiment_dir(rl_root, algorithm, scenario)
            directories = seed_dirs(experiment, seeds)
            train = normalized_training(directories, int(args.smooth_window))
            recovery = fixed_denominator_recovery(directories)
            if train.empty or recovery.empty:
                continue
            style = STYLES[algorithm]
            train_x = train["global_step"].to_numpy(dtype=float).copy()
            train_y = train["performance_smooth"].to_numpy(dtype=float).copy()
            if train_x.size and train_x[-1] < nominal_end:
                train_x = np.append(train_x, nominal_end)
                train_y = np.append(train_y, 100.0)
            elif train_x.size:
                train_x[-1] = nominal_end
                train_y[-1] = 100.0
            ax.plot(
                train_x,
                train_y,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.1,
                alpha=0.78,
                label=f"{style['label']} clean training",
            )
            shock_y = float(recovery.iloc[0]["performance_mean"])
            ax.plot([nominal_end, nominal_end], [100.0, shock_y], color="#c23b22", linewidth=1.45, alpha=0.75)
            ax.scatter([nominal_end], [shock_y], color="#c23b22", s=27, zorder=5)
            ax.plot(
                nominal_end + recovery["recovery_step"].astype(float),
                recovery["performance_mean"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.35,
                marker="o",
                markersize=3.0,
                label=f"{style['label']} recovery",
            )
            summary_rows.append(summary_row(style["label"], scenario, recovery, "recovery"))

        bvr = bvr_repair_curve(bvr_root, scenario, seeds)
        if not bvr.empty:
            style = STYLES["bvr"]
            max_iteration = max(float(bvr["iteration"].max()), 1.0)
            bvr_x = nominal_end + bvr["iteration"].astype(float) / max_iteration * recovery_timesteps
            ax.plot(
                bvr_x,
                bvr["performance_mean"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.6,
                marker="D",
                markersize=3.2,
                label=f"{style['label']} recovery",
            )
            summary_rows.append(summary_row(style["label"], scenario, bvr, "bvr_recovery"))

        ax.set_title(map_label(ppo_seed_dirs[0], scenario))
        ax.set_xlim(0, nominal_end + recovery_timesteps)
        ax.set_ylim(float(args.y_limits[0]), float(args.y_limits[1]))
        ax.set_xticks([0, nominal_end // 2, nominal_end, nominal_end + recovery_timesteps])
        ax.set_xticklabels(["0", "25k", "50k", "70k"])
        ax.grid(alpha=0.23)
        ax.set_xlabel("protocol step")
        if index == 0:
            ax.set_ylabel("performance index\n(pre-attack clean nominal = 100)")
        ax.text(nominal_end * 0.5, args.y_limits[1] - 2.0, "clean training", ha="center", va="top", color="0.32", fontsize=9)
        ax.text(nominal_end + recovery_timesteps * 0.5, args.y_limits[1] - 2.0, "attack recovery", ha="center", va="top", color="0.32", fontsize=9)

    handles: list[object] = []
    labels: list[str] = []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.03))

    fig.suptitle(
        "Clean Training, Attack Drop, and Recovery",
        y=1.10,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = output_dir / "fig_bvr_easy_protocol_recovery_phase.png"
    pdf_path = output_dir / "fig_bvr_easy_protocol_recovery_phase.pdf"
    summary_path = output_dir / "bvr_easy_protocol_recovery_phase_summary.csv"
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
