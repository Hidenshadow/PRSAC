#!/usr/bin/env python
"""Fixed 9-scenario PPO/SAC protocol plot with oracle upper-bound overlays.

This reuses the baseline protocol plotting code from
``plot_protocol_performance_index.py`` and adds an optional per-panel oracle
reference. The oracle is a diagnostic upper bound, not a deployable algorithm:
it is expected to come from ``evaluate_recovery_oracle_upper_bound.py``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import pandas as pd

import plot_protocol_performance_index as protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "rl_baselines"
DEFAULT_ORACLE_SUMMARY = (
    PROJECT_ROOT
    / "runs"
    / "planner_oracle_protocol_analysis"
    / "seed0_best_of_candidates"
    / "oracle_panel_summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Run root containing algorithm directories.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <root>/comparison_5seed_complete_scenarios.",
    )
    parser.add_argument("--oracle-summary", type=Path, default=DEFAULT_ORACLE_SUMMARY)
    parser.add_argument("--algorithms", nargs="+", default=list(protocol.DEFAULT_ALGORITHMS))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all"],
        help="Scenario names, e.g. level2_easy. Use 'all' for the standard 3x3 order.",
    )
    parser.add_argument("--seed-count", type=int, default=5)
    parser.add_argument(
        "--require-complete-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require --seed-count completed seed directories before plotting an algorithm/scenario.",
    )
    parser.add_argument(
        "--blank-incomplete-scenarios",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Leave a scenario panel blank if any requested algorithm is missing or incomplete.",
    )
    parser.add_argument(
        "--include-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show missing panels instead of dropping incomplete scenarios.",
    )
    parser.add_argument("--smooth-window", type=int, default=3)
    parser.add_argument("--ncols", type=int, default=3)
    parser.add_argument(
        "--y-limits",
        type=float,
        nargs=2,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Shared y-axis limits for all non-empty panels. Defaults to 70-110 and expands for larger oracle values.",
    )
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument(
        "--filename-prefix",
        type=str,
        default="fig_protocol_performance_index_9scenes_complete5seeds_with_oracle",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def scenario_from_level_difficulty(level: str, difficulty: str) -> str:
    return f"{str(level)}_{str(difficulty)}"


def load_oracle_indices(path: Path) -> dict[str, float]:
    """Load oracle index by scenario.

    Supports the panel summary produced by ``evaluate_recovery_oracle_upper_bound.py``.
    If a domain-level summary is passed instead, values are averaged by panel.
    """

    path = resolve(path)
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    required = {"level", "difficulty", "oracle_best_of_candidates_index"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Oracle summary missing columns {sorted(required)}: {path}")

    grouped = (
        frame.groupby(["level", "difficulty"], as_index=False)["oracle_best_of_candidates_index"]
        .mean()
        .dropna(subset=["oracle_best_of_candidates_index"])
    )
    return {
        scenario_from_level_difficulty(row.level, row.difficulty): float(row.oracle_best_of_candidates_index)
        for row in grouped.itertuples(index=False)
    }


def default_y_limits(oracle_indices: dict[str, float]) -> tuple[float, float]:
    y_min = 70.0
    y_max = 110.0
    finite_oracles = [value for value in oracle_indices.values() if math.isfinite(value)]
    if finite_oracles:
        y_max = max(y_max, math.ceil((max(finite_oracles) + 1.0) / 5.0) * 5.0)
    return y_min, y_max


def overlay_oracle(
    ax: plt.Axes,
    scenario: str,
    data_by_algorithm: dict[str, protocol.ExperimentData],
    oracle_indices: dict[str, float],
) -> None:
    if not data_by_algorithm or scenario not in oracle_indices:
        return

    first = next(iter(data_by_algorithm.values()))
    x_min = 0
    x_max = first.nominal_timesteps + first.recovery_timesteps
    oracle_index = float(oracle_indices[scenario])
    if not math.isfinite(oracle_index):
        return

    ax.hlines(
        oracle_index,
        x_min,
        x_max,
        color="0.08",
        linestyle=(0, (4, 2)),
        linewidth=1.6,
        alpha=0.85,
        label="Oracle upper bound",
        zorder=2,
    )
    ax.scatter(
        [x_max],
        [oracle_index],
        color="0.08",
        marker="D",
        s=28,
        zorder=5,
    )
    text_transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        0.985,
        oracle_index,
        f"oracle {oracle_index:.1f}",
        transform=text_transform,
        ha="right",
        va="bottom",
        fontsize=8,
        color="0.08",
    )


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    output_dir = resolve(args.output_dir) if args.output_dir else root / "comparison_5seed_complete_scenarios"
    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = list(protocol.DEFAULT_SCENARIOS) if args.scenarios == ["all"] else args.scenarios

    oracle_summary = resolve(args.oracle_summary)
    oracle_indices = load_oracle_indices(oracle_summary)
    y_limits = tuple(float(value) for value in args.y_limits) if args.y_limits else default_y_limits(oracle_indices)

    loaded: dict[str, dict[str, protocol.ExperimentData]] = {}
    summary_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for scenario in scenarios:
        loaded[scenario] = {}
        for algorithm in args.algorithms:
            data = protocol.load_experiment(
                root,
                algorithm,
                scenario,
                args.seed_count,
                args.smooth_window,
                args.require_complete_seeds,
            )
            if data is None:
                missing.append(f"{algorithm}/{scenario}")
                continue
            loaded[scenario][algorithm] = data
            summary_rows.append(protocol.summary_row(data))

        if args.blank_incomplete_scenarios and len(loaded[scenario]) != len(args.algorithms):
            missing_algorithms = set(args.algorithms) - set(loaded[scenario])
            for algorithm in sorted(loaded[scenario]):
                missing.append(f"{algorithm}/{scenario} blanked because {sorted(missing_algorithms)} incomplete")
            loaded[scenario] = {}
            summary_rows = [row for row in summary_rows if row["scenario"] != scenario]

    plotted_scenarios = [scenario for scenario in scenarios if loaded.get(scenario) or args.include_missing]
    if not plotted_scenarios:
        details = "\n".join(f"- {item}" for item in missing[:20])
        raise SystemExit(f"No completed scenario data found under {root}\n{details}")

    protocol.setup_matplotlib()
    n_panels = len(plotted_scenarios)
    ncols = args.ncols if args.ncols > 0 else min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.2 * ncols, 4.5 * nrows),
        squeeze=False,
        sharex=False,
    )

    for idx, scenario in enumerate(plotted_scenarios):
        ax = axes.ravel()[idx]
        if not loaded.get(scenario):
            level, difficulty = protocol.scenario_parts(scenario)
            ax.text(0.5, 0.5, "missing complete 5 seeds", transform=ax.transAxes, ha="center", va="center", color="0.35")
            ax.set_title(
                f"{protocol.LEVEL_LABELS.get(level, level)}\n"
                f"{protocol.DIFFICULTY_LABELS.get(difficulty, difficulty.title())}"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        protocol.plot_panel(ax, scenario, loaded[scenario], y_limits)
        overlay_oracle(ax, scenario, loaded[scenario], oracle_indices)
        if idx % ncols == 0:
            ax.set_ylabel("performance index\n(clean nominal = 100)")
        if idx // ncols == nrows - 1:
            ax.set_xlabel("protocol step")

    protocol.hide_unused_axes(axes, len(plotted_scenarios))

    handles: list[object] = []
    labels: list[str] = []
    for ax in axes.ravel():
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        for handle, label in zip(ax_handles, ax_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(5, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.02))

    title = args.title or "PPO/SAC baseline protocol with oracle upper bound"
    fig.suptitle(title, y=1.08 if handles else 1.02, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    png_path = output_dir / f"{args.filename_prefix}.png"
    pdf_path = output_dir / f"{args.filename_prefix}.pdf"
    summary_path = output_dir / f"{args.filename_prefix}_summary.csv"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved summary: {summary_path}")
    if oracle_indices:
        print(f"Loaded oracle summary: {oracle_summary}")
    else:
        print(f"Oracle summary not found or empty: {oracle_summary}")
    if missing:
        print("Missing/incomplete runs skipped:")
        for item in missing:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
