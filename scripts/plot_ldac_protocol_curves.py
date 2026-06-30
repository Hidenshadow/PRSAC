#!/usr/bin/env python
"""Plot available LDAC-SAC train -> drop -> recovery curves against PPO/SAC."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ROOT = PROJECT_ROOT / "runs" / "rl_baselines"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "sac_modified" / "protocol_figures_current"
DEFAULT_SCENARIOS = (
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

LDAC_ROOTS_BY_DIFFICULTY = {
    "easy": (
        PROJECT_ROOT / "runs" / "sac_modified" / "ldac_easy_seed0_20260621",
        PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_easy_seed1_2_20260621",
        PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_easy_seed3_4_20260623",
    ),
    "medium": (
        PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_medium_seed0_1_20260621",
        PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_medium_seed2_4_20260623",
    ),
    "hard": (
        PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_hard_seed0_1_20260621",
        PROJECT_ROOT / "runs" / "sac_modified" / "ldac_v2_hard_seed2_4_20260623",
    ),
}
CDR_SAC_ROOT = PROJECT_ROOT / "runs" / "sac_modified" / "cdr_sac_from_sac_nominal_9scenarios_2seeds_20260627"

LEVEL_LABELS = {
    "level1": "Level 1",
    "level2": "Level 2",
    "level3": "Level 3",
}

DIFFICULTY_LABELS = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}

OKABE_ITO = {
    "blue": "#0072B2",
    "bluish_green": "#009E73",
    "vermillion": "#D55E00",
    "sky_blue": "#56B4E9",
    "orange": "#E69F00",
    "reddish_purple": "#CC79A7",
}

ALGORITHM_STYLES = {
    "ppo": {
        "label": "PPO baseline",
        "color": OKABE_ITO["blue"],
        "linestyle": "-",
        "linewidth": 1.8,
    },
    "sac": {
        "label": "SAC baseline",
        "color": OKABE_ITO["bluish_green"],
        "linestyle": "-",
        "linewidth": 1.8,
    },
    "cdr_sac": {
        "label": "CDR-SAC",
        "color": OKABE_ITO["orange"],
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "ldac_sac": {
        "label": "LDAC-SAC",
        "color": OKABE_ITO["vermillion"],
        "linestyle": "-",
        "linewidth": 2.5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--baseline-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--y-limits", nargs=2, type=float, default=[70.0, 106.0])
    parser.add_argument("--filename-prefix", type=str, default="fig_ldac_protocol_current")
    parser.add_argument("--title", type=str, default="Protocol performance under map corruption")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def scenario_parts(scenario: str) -> tuple[str, str]:
    parts = scenario.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"scenario must look like level2_easy, got {scenario!r}")
    return parts[0], parts[1]


def scenario_title(scenario: str) -> str:
    level, difficulty = scenario_parts(scenario)
    return f"{LEVEL_LABELS.get(level, level)} {DIFFICULTY_LABELS.get(difficulty, difficulty.title())}"


def seed_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.fullmatch(r"seed(\d+)", part)
        if match:
            return int(match.group(1))
    return -1


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def nominal_training_csv(seed_dir: Path) -> Path | None:
    direct = seed_dir / "nominal_training_eval.csv"
    if direct.exists():
        return direct
    candidates = sorted((seed_dir / "nominal_train").glob("*/eval_metrics.csv"))
    return candidates[-1] if candidates else None


def has_protocol_outputs(seed_dir: Path) -> bool:
    return nominal_training_csv(seed_dir) is not None and (seed_dir / "shock_recovery_curve.csv").exists()


def baseline_seed_dirs(root: Path, algorithm: str, scenario: str, seeds: list[int]) -> list[Path]:
    experiment = root / algorithm / f"{scenario}_shock_recovery_5seeds"
    seed_dirs = []
    for seed in seeds:
        seed_dir = experiment / f"seed{seed}"
        if has_protocol_outputs(seed_dir):
            seed_dirs.append(seed_dir)
    return seed_dirs


def ldac_seed_dirs(scenario: str) -> list[Path]:
    _, difficulty = scenario_parts(scenario)
    by_seed: dict[int, Path] = {}
    for root in LDAC_ROOTS_BY_DIFFICULTY.get(difficulty, ()):
        scenario_dir = root / scenario
        if not scenario_dir.exists():
            continue
        for seed_dir in sorted(scenario_dir.glob("seed*")):
            if has_protocol_outputs(seed_dir):
                by_seed[seed_from_path(seed_dir)] = seed_dir
    return [by_seed[seed] for seed in sorted(by_seed)]


def cdr_sac_seed_dirs(scenario: str, seeds: list[int]) -> list[Path]:
    by_seed: dict[int, Path] = {}
    scenario_dir = CDR_SAC_ROOT / scenario
    if scenario_dir.exists():
        for seed_dir in sorted(scenario_dir.glob("seed*")):
            seed = seed_from_path(seed_dir)
            if seed in set(seeds) and has_protocol_outputs(seed_dir):
                by_seed[seed] = seed_dir
    return [by_seed[seed] for seed in sorted(by_seed)]


def protocol_timesteps(seed_dirs: list[Path]) -> tuple[int, int]:
    for seed_dir in seed_dirs:
        config = read_json(seed_dir / "run_config.json")
        command_args = config.get("command_args", {})
        if isinstance(command_args, dict):
            nominal = command_args.get("nominal_timesteps")
            recovery = command_args.get("recovery_timesteps")
            if nominal and recovery:
                return int(nominal), int(recovery)
    return 50_000, 20_480


def training_curve(seed_dirs: list[Path], smooth_window: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed_dir in seed_dirs:
        csv_path = nominal_training_csv(seed_dir)
        if csv_path is None:
            continue
        frame = pd.read_csv(csv_path)
        required = {"global_step", "mean_scalar_cost"}
        if not required.issubset(frame.columns):
            continue
        frame = frame[["global_step", "mean_scalar_cost"]].dropna().sort_values("global_step")
        if frame.empty:
            continue
        final_cost = float(frame.iloc[-1]["mean_scalar_cost"])
        if not math.isfinite(final_cost) or final_cost <= 0.0:
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
        window=max(1, int(smooth_window)),
        center=True,
        min_periods=1,
    ).mean()
    return grouped


def recovery_curve(seed_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed_dir in seed_dirs:
        curve_path = seed_dir / "shock_recovery_curve.csv"
        if not curve_path.exists():
            continue
        curve = pd.read_csv(curve_path)
        required = {"eval_domain", "phase", "attack_type", "recovery_step", "mean_attacked_scalar_cost"}
        if not required.issubset(curve.columns):
            continue
        clean = curve[(curve["phase"] == "shock") & (curve["attack_type"] == "none")][
            ["eval_domain", "mean_attacked_scalar_cost"]
        ].rename(columns={"mean_attacked_scalar_cost": "clean_nominal_cost"})
        attacked = curve[
            (curve["attack_type"] == "environment")
            & (curve["phase"].isin(["shock", "recovery"]))
        ][["eval_domain", "phase", "recovery_step", "mean_attacked_scalar_cost"]]
        if clean.empty or attacked.empty:
            continue
        merged = attacked.merge(clean, on="eval_domain", how="inner")
        merged["seed"] = seed_from_path(seed_dir)
        merged["performance_index"] = (
            100.0
            * merged["clean_nominal_cost"].astype(float)
            / merged["mean_attacked_scalar_cost"].astype(float).clip(lower=1e-12)
        )
        frames.append(merged[["seed", "eval_domain", "phase", "recovery_step", "performance_index"]])
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


def algorithm_seed_dirs(root: Path, algorithm: str, scenario: str, baseline_seeds: list[int]) -> list[Path]:
    if algorithm in {"ppo", "sac"}:
        return baseline_seed_dirs(root, algorithm, scenario, baseline_seeds)
    if algorithm == "cdr_sac":
        return cdr_sac_seed_dirs(scenario, baseline_seeds)
    if algorithm == "ldac_sac":
        return ldac_seed_dirs(scenario)
    raise ValueError(f"unknown algorithm: {algorithm}")


def collect_data(
    root: Path,
    scenarios: list[str],
    baseline_seeds: list[int],
    smooth_window: int,
) -> dict[tuple[str, str], dict[str, object]]:
    data: dict[tuple[str, str], dict[str, object]] = {}
    for scenario in scenarios:
        for algorithm in ALGORITHM_STYLES:
            seed_dirs = algorithm_seed_dirs(root, algorithm, scenario, baseline_seeds)
            if seed_dirs:
                nominal_steps, recovery_steps = protocol_timesteps(seed_dirs)
            else:
                nominal_steps, recovery_steps = 50_000, 20_480
            data[(algorithm, scenario)] = {
                "seed_dirs": seed_dirs,
                "train": training_curve(seed_dirs, smooth_window),
                "recovery": recovery_curve(seed_dirs),
                "nominal_steps": nominal_steps,
                "recovery_steps": recovery_steps,
            }
    return data


def add_curve_band(ax: plt.Axes, x: np.ndarray, y: np.ndarray, std: np.ndarray, color: str) -> None:
    if len(x) < 2:
        return
    std = np.nan_to_num(std, nan=0.0)
    ax.fill_between(x, y - std, y + std, color=color, alpha=0.10, linewidth=0)


def phase_x(values: np.ndarray | pd.Series, duration: int, left: float, right: float) -> np.ndarray:
    duration = max(1.0, float(duration))
    values_array = np.asarray(values, dtype=float)
    return left + (right - left) * values_array / duration


def plot_algorithm(
    ax: plt.Axes,
    algorithm: str,
    item: dict[str, object],
    nominal_steps: int,
    recovery_steps: int,
) -> bool:
    style = ALGORITHM_STYLES[algorithm]
    color = style["color"]
    train = item["train"]
    recovery = item["recovery"]
    seed_count = len(item["seed_dirs"])
    label = f"{style['label']} (n={seed_count})"
    plotted = False

    if isinstance(train, pd.DataFrame) and not train.empty:
        x_train = phase_x(train["global_step"], nominal_steps, 0.0, 0.5)
        y_train = train["performance_smooth"].to_numpy(dtype=float)
        ax.plot(
            x_train,
            y_train,
            color=color,
            linestyle=style["linestyle"],
            linewidth=max(1.2, float(style["linewidth"]) - 0.4),
            alpha=0.75,
            label=None,
        )
        add_curve_band(
            ax,
            x_train,
            train["performance_mean"].to_numpy(dtype=float),
            train["performance_std"].to_numpy(dtype=float),
            color,
        )
        plotted = True

    if isinstance(recovery, pd.DataFrame) and not recovery.empty:
        x_recovery = phase_x(recovery["recovery_step"], recovery_steps, 0.5, 1.0)
        y_recovery = recovery["performance_mean"].to_numpy(dtype=float)
        ax.plot(
            x_recovery,
            y_recovery,
            color=color,
            linestyle=style["linestyle"],
            linewidth=float(style["linewidth"]),
            alpha=0.95,
            label=label,
        )
        add_curve_band(
            ax,
            x_recovery,
            y_recovery,
            recovery["performance_std"].to_numpy(dtype=float),
            color,
        )
        if isinstance(train, pd.DataFrame) and not train.empty:
            y0 = float(train.iloc[-1]["performance_smooth"])
            y1 = float(recovery.iloc[0]["performance_mean"])
            ax.plot([0.5, 0.5], [y0, y1], color=color, linestyle=":", linewidth=1.5, alpha=0.9)
        plotted = True

    return plotted


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.22",
            "axes.labelcolor": "0.15",
            "xtick.color": "0.15",
            "ytick.color": "0.15",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def plot_grid(
    scenarios: list[str],
    data: dict[tuple[str, str], dict[str, object]],
    output_path: Path,
    y_limits: tuple[float, float],
    ncols: int,
    title: str,
) -> None:
    setup_matplotlib()
    nrows = int(math.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 3.15 * nrows), sharey=True, squeeze=False)
    y_min, y_max = y_limits
    clean_span_color = "#eef6fb"
    corrupted_span_color = "#fff4e6"
    boundary_color = "0.35"

    for idx, ax in enumerate(axes.flat):
        if idx >= len(scenarios):
            ax.axis("off")
            continue
        scenario = scenarios[idx]
        available_items = [data[(algo, scenario)] for algo in ALGORITHM_STYLES if data[(algo, scenario)]["seed_dirs"]]
        nominal_steps, recovery_steps = (50_000, 20_480)
        if available_items:
            nominal_steps = int(available_items[0]["nominal_steps"])
            recovery_steps = int(available_items[0]["recovery_steps"])
        protocol_end = nominal_steps + recovery_steps

        ax.axvspan(0.0, 0.5, color=clean_span_color, alpha=0.95, zorder=0)
        ax.axvspan(0.5, 1.0, color=corrupted_span_color, alpha=0.95, zorder=0)
        ax.axhline(100.0, color="0.45", linewidth=0.8, linestyle="-", alpha=0.75)
        ax.axvline(0.5, color=boundary_color, linewidth=1.1, linestyle="--", alpha=0.85)

        plotted_any = False
        for algorithm in ALGORITHM_STYLES:
            plotted_any = (
                plot_algorithm(ax, algorithm, data[(algorithm, scenario)], nominal_steps, recovery_steps)
                or plotted_any
            )

        ax.set_title(scenario_title(scenario), fontsize=12, pad=8)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, axis="y", color="0.86", linewidth=0.75)
        ax.grid(False, axis="x")
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(
            [
                "0",
                f"{nominal_steps // 2000}k",
                f"{nominal_steps // 1000}k",
                f"{int((nominal_steps + 0.5 * recovery_steps) // 1000)}k",
                f"{protocol_end // 1000}k",
            ]
        )
        ax.text(0.25, 0.965, "Clean map", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="0.25")
        ax.text(0.50, 0.965, "Corrupted", transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="0.25")
        ax.text(
            0.75,
            0.965,
            "Corrupted map",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color="0.25",
        )
        ldac_item = data[("ldac_sac", scenario)]
        if not ldac_item["seed_dirs"]:
            ax.text(
                0.52,
                0.50,
                "LDAC-SAC pending",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="0.45",
            )
        elif not plotted_any:
            ax.text(
                0.5,
                0.5,
                "No completed result",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="0.45",
            )

        if idx % ncols == 0:
            ax.set_ylabel("Performance index")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Protocol timestep (phase-normalized)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            base = re.sub(r" \(n=\d+\)$", "", label)
            if base not in labels:
                handles.append(handle)
                labels.append(base)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(4, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.004),
            handlelength=2.8,
        )
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0.0, 0.055, 1.0, 0.962])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def summary_rows(data: dict[tuple[str, str], dict[str, object]], scenarios: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for algorithm in ALGORITHM_STYLES:
            item = data[(algorithm, scenario)]
            recovery = item["recovery"]
            train = item["train"]
            row = {
                "algorithm": algorithm,
                "scenario": scenario,
                "num_seed_dirs": len(item["seed_dirs"]),
                "has_training_curve": isinstance(train, pd.DataFrame) and not train.empty,
                "has_recovery_curve": isinstance(recovery, pd.DataFrame) and not recovery.empty,
                "shock_index": math.nan,
                "final_recovery_index": math.nan,
                "best_recovery_index": math.nan,
            }
            if isinstance(recovery, pd.DataFrame) and not recovery.empty:
                row["shock_index"] = float(recovery.iloc[0]["performance_mean"])
                row["final_recovery_index"] = float(recovery.iloc[-1]["performance_mean"])
                row["best_recovery_index"] = float(recovery["performance_mean"].max())
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    baseline_root = resolve(args.baseline_root)
    output_dir = resolve(args.output_dir)
    scenarios = list(args.scenarios)
    data = collect_data(baseline_root, scenarios, args.baseline_seeds, args.smooth_window)

    all_path = output_dir / f"{args.filename_prefix}_all_available.png"
    plot_grid(
        scenarios,
        data,
        all_path,
        (float(args.y_limits[0]), float(args.y_limits[1])),
        ncols=3,
        title=args.title,
    )

    for difficulty in ("easy", "medium", "hard"):
        subset = [scenario for scenario in scenarios if scenario.endswith(f"_{difficulty}")]
        if not subset:
            continue
        plot_grid(
            subset,
            data,
            output_dir / f"{args.filename_prefix}_{difficulty}.png",
            (float(args.y_limits[0]), float(args.y_limits[1])),
            ncols=len(subset),
            title=f"{DIFFICULTY_LABELS[difficulty]} protocol performance under map corruption",
        )

    summary = pd.DataFrame(summary_rows(data, scenarios))
    summary_path = output_dir / f"{args.filename_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {all_path}")
    print(f"Saved: {all_path.with_suffix('.pdf')}")
    print(f"Saved summary: {summary_path}")
    for difficulty in ("easy", "medium", "hard"):
        png = output_dir / f"{args.filename_prefix}_{difficulty}.png"
        if png.exists():
            print(f"Saved: {png}")
            print(f"Saved: {png.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
