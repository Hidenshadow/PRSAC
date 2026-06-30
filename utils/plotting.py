"""Plotting helpers for aggregate evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.metrics import OBJECTIVE_NAMES


def plot_evaluation_summary(summary: pd.DataFrame, output_dir: str | Path) -> None:
    """Save reward/scalar and objective comparison figures."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = summary["method"].tolist()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(methods, summary["mean_reward"], yerr=summary["std_reward"], capsize=4)
    ax.set_ylabel("Mean reward")
    ax.set_title("Reward comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "reward_comparison.png", dpi=180)
    plt.close(fig)

    objective_columns = [f"mean_{name}_cost" for name in OBJECTIVE_NAMES]
    x = np.arange(len(methods))
    width = 0.14

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for index, column in enumerate(objective_columns):
        offset = (index - 2) * width
        ax.bar(x + offset, summary[column], width=width, label=OBJECTIVE_NAMES[index])
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Mean objective cost")
    ax.set_title("Objective comparison")
    ax.legend(ncol=len(OBJECTIVE_NAMES), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "objective_comparison.png", dpi=180)
    plt.close(fig)
