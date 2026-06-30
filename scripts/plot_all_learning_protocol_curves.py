#!/usr/bin/env python
"""Plot train -> corruption drop -> recovery curves for all learning methods."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_ldac_protocol_curves as protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "sac_modified" / "analysis" / "all_learning_protocol_20260629"

METHODS = {
    "ppo": {
        "label": "PPO",
        "family": "baseline",
        "color": protocol.OKABE_ITO["blue"],
        "linestyle": "-",
        "linewidth": 1.7,
    },
    "sac": {
        "label": "SAC",
        "family": "baseline",
        "color": protocol.OKABE_ITO["bluish_green"],
        "linestyle": "-",
        "linewidth": 1.7,
    },
    "cdr_sac": {
        "label": "CDR-SAC",
        "family": "variant",
        "root": PROJECT_ROOT / "runs" / "sac_modified" / "cdr_sac_from_sac_nominal_9scenarios_2seeds_20260627",
        "color": protocol.OKABE_ITO["orange"],
        "linestyle": "--",
        "linewidth": 2.0,
    },
    "stackelberg_sac": {
        "label": "Stackelberg-SAC",
        "family": "variant",
        "root": PROJECT_ROOT
        / "runs"
        / "sac_modified"
        / "stackelberg_sac_from_sac_nominal_9scenarios_2seeds_20260628",
        "color": protocol.OKABE_ITO["reddish_purple"],
        "linestyle": "-.",
        "linewidth": 2.0,
    },
    "valt_sac": {
        "label": "VALT-SAC",
        "family": "variant",
        "root": PROJECT_ROOT / "runs" / "sac_modified" / "valt_sac_from_sac_nominal_9scenarios_3seeds_20260628",
        "color": protocol.OKABE_ITO["sky_blue"],
        "linestyle": (0, (5, 1.5)),
        "linewidth": 2.0,
    },
    "ldac_sac": {
        "label": "LDAC-SAC",
        "family": "ldac",
        "color": protocol.OKABE_ITO["vermillion"],
        "linestyle": "-",
        "linewidth": 2.6,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=protocol.DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scenarios", nargs="+", default=list(protocol.DEFAULT_SCENARIOS))
    parser.add_argument("--baseline-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--y-limits", nargs=2, type=float, default=[70.0, 106.0])
    parser.add_argument("--filename-prefix", type=str, default="fig_all_learning_train_drop_recovery")
    parser.add_argument(
        "--title",
        type=str,
        default="Clean training, corruption drop, and recovery across learning methods",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def variant_seed_dirs(root: Path, scenario: str) -> list[Path]:
    scenario_dir = root / scenario
    by_seed: dict[int, Path] = {}
    if not scenario_dir.exists():
        return []
    for seed_dir in sorted(scenario_dir.glob("seed*")):
        seed = protocol.seed_from_path(seed_dir)
        if protocol.has_protocol_outputs(seed_dir):
            by_seed[seed] = seed_dir
    return [by_seed[seed] for seed in sorted(by_seed)]


def method_seed_dirs(
    baseline_root: Path,
    method: str,
    scenario: str,
    baseline_seeds: list[int],
) -> list[Path]:
    spec = METHODS[method]
    family = spec["family"]
    if family == "baseline":
        return protocol.baseline_seed_dirs(baseline_root, method, scenario, baseline_seeds)
    if family == "ldac":
        return protocol.ldac_seed_dirs(scenario)
    if family == "variant":
        return variant_seed_dirs(resolve(Path(spec["root"])), scenario)
    raise ValueError(f"unknown method family: {family}")


def collect_data(
    baseline_root: Path,
    scenarios: list[str],
    baseline_seeds: list[int],
    smooth_window: int,
) -> dict[tuple[str, str], dict[str, object]]:
    data: dict[tuple[str, str], dict[str, object]] = {}
    for scenario in scenarios:
        for method in METHODS:
            seed_dirs = method_seed_dirs(baseline_root, method, scenario, baseline_seeds)
            nominal_steps, recovery_steps = protocol.protocol_timesteps(seed_dirs) if seed_dirs else (50_000, 20_480)
            data[(method, scenario)] = {
                "seed_dirs": seed_dirs,
                "train": protocol.training_curve(seed_dirs, smooth_window),
                "recovery": protocol.recovery_curve(seed_dirs),
                "nominal_steps": nominal_steps,
                "recovery_steps": recovery_steps,
            }
    return data


def finite_or_nan(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def curve_auc(recovery: pd.DataFrame) -> float:
    if recovery.empty:
        return math.nan
    x = recovery["recovery_step"].to_numpy(dtype=float)
    y = recovery["performance_mean"].to_numpy(dtype=float)
    if len(x) == 1:
        return float(y[0])
    duration = max(1.0, float(x.max() - x.min()))
    return float(np.trapezoid(y, x) / duration)


def plot_method(
    ax: plt.Axes,
    method: str,
    item: dict[str, object],
    nominal_steps: int,
    recovery_steps: int,
) -> bool:
    spec = METHODS[method]
    color = str(spec["color"])
    linestyle = spec["linestyle"]
    linewidth = float(spec["linewidth"])
    train = item["train"]
    recovery = item["recovery"]
    plotted = False

    if isinstance(train, pd.DataFrame) and not train.empty:
        x_train = protocol.phase_x(train["global_step"], nominal_steps, 0.0, 0.5)
        y_train = train["performance_smooth"].to_numpy(dtype=float)
        ax.plot(
            x_train,
            y_train,
            color=color,
            linestyle=linestyle,
            linewidth=max(1.1, linewidth - 0.45),
            alpha=0.70,
        )
        protocol.add_curve_band(
            ax,
            x_train,
            train["performance_mean"].to_numpy(dtype=float),
            train["performance_std"].to_numpy(dtype=float),
            color,
        )
        plotted = True

    if isinstance(recovery, pd.DataFrame) and not recovery.empty:
        x_recovery = protocol.phase_x(recovery["recovery_step"], recovery_steps, 0.5, 1.0)
        y_recovery = recovery["performance_mean"].to_numpy(dtype=float)
        ax.plot(
            x_recovery,
            y_recovery,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.96,
            label=str(spec["label"]),
        )
        protocol.add_curve_band(
            ax,
            x_recovery,
            y_recovery,
            recovery["performance_std"].to_numpy(dtype=float),
            color,
        )
        if isinstance(train, pd.DataFrame) and not train.empty:
            y0 = finite_or_nan(train.iloc[-1]["performance_smooth"])
            y1 = finite_or_nan(recovery.iloc[0]["performance_mean"])
            if math.isfinite(y0) and math.isfinite(y1):
                ax.plot([0.5, 0.5], [y0, y1], color=color, linestyle=":", linewidth=1.35, alpha=0.85)
        plotted = True

    return plotted


def plot_grid(
    scenarios: list[str],
    data: dict[tuple[str, str], dict[str, object]],
    output_path: Path,
    y_limits: tuple[float, float],
    title: str,
) -> None:
    protocol.setup_matplotlib()
    ncols = 3
    nrows = int(math.ceil(len(scenarios) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.05 * ncols, 3.20 * nrows), sharey=True, squeeze=False)
    y_min, y_max = y_limits

    for idx, ax in enumerate(axes.flat):
        if idx >= len(scenarios):
            ax.axis("off")
            continue
        scenario = scenarios[idx]
        available = [data[(method, scenario)] for method in METHODS if data[(method, scenario)]["seed_dirs"]]
        nominal_steps, recovery_steps = (50_000, 20_480)
        if available:
            nominal_steps = int(available[0]["nominal_steps"])
            recovery_steps = int(available[0]["recovery_steps"])
        protocol_end = nominal_steps + recovery_steps

        ax.axvspan(0.0, 0.5, color="#edf6fb", alpha=0.96, zorder=0)
        ax.axvspan(0.5, 1.0, color="#fff3e5", alpha=0.96, zorder=0)
        ax.axhline(100.0, color="0.45", linewidth=0.8, linestyle="-", alpha=0.75)
        ax.axvline(0.5, color="0.35", linewidth=1.1, linestyle="--", alpha=0.85)

        plotted_any = False
        for method in METHODS:
            plotted_any = plot_method(ax, method, data[(method, scenario)], nominal_steps, recovery_steps) or plotted_any

        ax.set_title(protocol.scenario_title(scenario), fontsize=12, pad=8)
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
        ax.text(0.25, 0.965, "Clean training", transform=ax.transAxes, ha="center", va="top", fontsize=8.2)
        ax.text(0.50, 0.965, "Drop", transform=ax.transAxes, ha="center", va="top", fontsize=8.2)
        ax.text(0.75, 0.965, "Recovery", transform=ax.transAxes, ha="center", va="top", fontsize=8.2)

        if not plotted_any:
            ax.text(0.5, 0.5, "No completed result", transform=ax.transAxes, ha="center", va="center", color="0.45")
        if idx % ncols == 0:
            ax.set_ylabel("Performance index")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Protocol timestep")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(handles)),
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


def scenario_summary_rows(data: dict[tuple[str, str], dict[str, object]], scenarios: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for method, spec in METHODS.items():
            item = data[(method, scenario)]
            train = item["train"]
            recovery = item["recovery"]
            seed_dirs = item["seed_dirs"]
            seed_ids = [protocol.seed_from_path(path) for path in seed_dirs]
            row = {
                "method": str(spec["label"]),
                "method_key": method,
                "scenario": scenario,
                "num_seed_dirs": len(seed_dirs),
                "seed_ids": " ".join(str(seed) for seed in seed_ids),
                "train_start_index": math.nan,
                "train_final_index": math.nan,
                "shock_index": math.nan,
                "final_recovery_index": math.nan,
                "best_recovery_index": math.nan,
                "recovery_auc_index": math.nan,
                "final_gain_vs_shock": math.nan,
                "best_gain_vs_shock": math.nan,
                "final_closure_pct": math.nan,
            }
            if isinstance(train, pd.DataFrame) and not train.empty:
                row["train_start_index"] = finite_or_nan(train.iloc[0]["performance_mean"])
                row["train_final_index"] = finite_or_nan(train.iloc[-1]["performance_mean"])
            if isinstance(recovery, pd.DataFrame) and not recovery.empty:
                shock = finite_or_nan(recovery.iloc[0]["performance_mean"])
                final = finite_or_nan(recovery.iloc[-1]["performance_mean"])
                best = finite_or_nan(recovery["performance_mean"].max())
                row["shock_index"] = shock
                row["final_recovery_index"] = final
                row["best_recovery_index"] = best
                row["recovery_auc_index"] = curve_auc(recovery)
                row["final_gain_vs_shock"] = final - shock
                row["best_gain_vs_shock"] = best - shock
                denom = 100.0 - shock
                if math.isfinite(denom) and denom > 1e-9:
                    row["final_closure_pct"] = 100.0 * (final - shock) / denom
            rows.append(row)
    return rows


def fmt_mean_std(mean: float, std: float, digits: int = 2) -> str:
    if not math.isfinite(mean):
        return "--"
    std = 0.0 if not math.isfinite(std) else std
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, spec in METHODS.items():
        subset = summary[(summary["method_key"] == method) & (summary["num_seed_dirs"] > 0)].copy()
        if subset.empty:
            continue
        seed_counts = subset["num_seed_dirs"].astype(int)
        seeds_per_scenario = (
            str(int(seed_counts.iloc[0]))
            if int(seed_counts.min()) == int(seed_counts.max())
            else f"{int(seed_counts.min())}-{int(seed_counts.max())}"
        )
        out = {
            "Method": str(spec["label"]),
            "Scenario coverage": f"{len(subset)}/{len(protocol.DEFAULT_SCENARIOS)}",
            "Seeds/scenario": seeds_per_scenario,
        }
        metrics = {
            "Shock": "shock_index",
            "Final": "final_recovery_index",
            "Best": "best_recovery_index",
            "AUC": "recovery_auc_index",
            "Final gain": "final_gain_vs_shock",
            "Closure (%)": "final_closure_pct",
        }
        for label, column in metrics.items():
            values = pd.to_numeric(subset[column], errors="coerce").dropna()
            out[f"{label} mean"] = float(values.mean()) if not values.empty else math.nan
            out[f"{label} std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            out[label] = fmt_mean_std(out[f"{label} mean"], out[f"{label} std"])
        rows.append(out)
    return pd.DataFrame(rows)


def write_aggregate_latex(aggregate: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{All-learning train--drop--recovery summary across the nine scenarios. Values are scenario-level means $\\pm$ standard deviations of the performance index; higher is better. Seed coverage uses all completed runs currently available for each method.}",
        "\\label{tab:all_learning_recovery_summary}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccccccc}",
        "\\toprule",
        "Method & Coverage & Seeds & Shock & Final & Best & AUC & Gain & Closure \\\\",
        "\\midrule",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            " & ".join(
                [
                    str(row["Method"]),
                    str(row["Scenario coverage"]),
                    str(row["Seeds/scenario"]),
                    str(row["Shock"]),
                    str(row["Final"]),
                    str(row["Best"]),
                    str(row["AUC"]),
                    str(row["Final gain"]),
                    str(row["Closure (%)"]),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_root = resolve(args.baseline_root)
    output_dir = resolve(args.output_dir)
    scenarios = list(args.scenarios)

    data = collect_data(baseline_root, scenarios, args.baseline_seeds, args.smooth_window)
    output_path = output_dir / f"{args.filename_prefix}.png"
    plot_grid(scenarios, data, output_path, (float(args.y_limits[0]), float(args.y_limits[1])), args.title)

    summary = pd.DataFrame(scenario_summary_rows(data, scenarios))
    summary_path = output_dir / f"{args.filename_prefix}_scenario_summary.csv"
    summary.to_csv(summary_path, index=False)

    aggregate = aggregate_summary(summary)
    aggregate_path = output_dir / f"{args.filename_prefix}_aggregate_summary.csv"
    aggregate.to_csv(aggregate_path, index=False)
    latex_path = output_dir / f"{args.filename_prefix}_aggregate_summary.tex"
    write_aggregate_latex(aggregate, latex_path)

    print(f"Saved: {output_path}")
    print(f"Saved: {output_path.with_suffix('.pdf')}")
    print(f"Saved scenario summary: {summary_path}")
    print(f"Saved aggregate summary: {aggregate_path}")
    print(f"Saved aggregate LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
