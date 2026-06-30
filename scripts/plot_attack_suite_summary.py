#!/usr/bin/env python
"""Plot post-hoc attack-suite drop summaries.

Reads `attack_suite_summary.csv` files produced by
`scripts/evaluate_attack_suite.py` and plots heldout mean/worst attack
degradation for nominal and recovery checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "runs" / "rl_baselines_refresh_1seed" / "attack_suite_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-domain", type=str, default="heldout_tasks")
    parser.add_argument("--filename-prefix", type=str, default="fig_attack_suite_drop_summary")
    parser.add_argument("--title", type=str, default="Attack-suite degradation summary")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_case(path: Path, root: Path) -> tuple[str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 3 and parts[0] in {"ppo", "sac"}:
        algo = parts[0]
        case_name = parts[1]
    else:
        algo = "unknown"
        case_name = parts[-2]
    match = re.match(r"(.+)_seed\d+$", case_name)
    scenario = match.group(1) if match else case_name
    return algo, scenario


def load_summaries(root: Path, eval_domain: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(root.glob("**/attack_suite_summary.csv")):
        frame = pd.read_csv(csv_path)
        if frame.empty or "eval_domain" not in frame.columns:
            continue
        algo, scenario = parse_case(csv_path, root)
        frame = frame[frame["eval_domain"] == eval_domain].copy()
        if frame.empty:
            continue
        frame.insert(0, "algorithm", algo)
        frame.insert(1, "scenario", scenario)
        frame.insert(2, "source", str(csv_path))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    role_order = {"nominal": 0, "best_recovery": 1, "final_recovery": 2}
    algo_order = {"ppo": 0, "sac": 1}
    combined["_algo_order"] = combined["algorithm"].map(algo_order).fillna(99)
    combined["_role_order"] = combined["checkpoint_role"].map(role_order).fillna(99)
    combined = combined.sort_values(["scenario", "_algo_order", "_role_order"]).drop(
        columns=["_algo_order", "_role_order"]
    )
    return combined


def label_for(row: pd.Series) -> str:
    algo = str(row["algorithm"]).upper()
    scenario = str(row["scenario"]).replace("_", " ")
    role = str(row["checkpoint_role"]).replace("_", " ")
    return f"{algo}\n{scenario}\n{role}"


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


def plot_summary(frame: pd.DataFrame, output_dir: Path, filename_prefix: str, title: str, eval_domain: str) -> None:
    setup_matplotlib()
    fig_width = max(8.0, 1.25 * len(frame) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))

    x = np.arange(len(frame))
    width = 0.36
    mean_values = frame["mean_attack_degradation_pct"].astype(float).to_numpy()
    worst_values = frame["worst_attack_degradation_pct"].astype(float).to_numpy()

    bars_mean = ax.bar(x - width / 2, mean_values, width, color="#4C78A8", label="Mean drop")
    bars_worst = ax.bar(x + width / 2, worst_values, width, color="#F58518", label="Worst drop")

    for bars in (bars_mean, bars_worst):
        for bar in bars:
            value = float(bar.get_height())
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.35,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.axhline(5.0, color="0.35", linestyle="--", linewidth=1.0, alpha=0.75, label="Meaningful drop 5%")
    ax.axhline(10.0, color="#C23B22", linestyle=":", linewidth=1.2, alpha=0.75, label="Strong drop 10%")
    ax.set_xticks(x)
    ax.set_xticklabels([label_for(row) for _, row in frame.iterrows()])
    ax.set_ylabel("Attack degradation (%)")
    ax.set_title(f"{title}\n{eval_domain}, {int(frame['num_eval_episodes_per_variant'].iloc[0])} episodes per variant")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    y_max = max(float(np.nanmax(worst_values)) + 4.0, 12.0)
    ax.set_ylim(0, y_max)
    fig.tight_layout()

    png_path = output_dir / f"{filename_prefix}.png"
    pdf_path = output_dir / f"{filename_prefix}.pdf"
    csv_path = output_dir / f"{filename_prefix}.csv"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    frame.to_csv(csv_path, index=False)
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")
    print(f"Saved plotted data: {csv_path}")


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    output_dir = resolve(args.output_dir) if args.output_dir else root
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_summaries(root, args.eval_domain)
    if frame.empty:
        raise SystemExit(f"No attack-suite summaries found for eval_domain={args.eval_domain}: {root}")
    plot_summary(frame, output_dir, args.filename_prefix, args.title, args.eval_domain)
    print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
