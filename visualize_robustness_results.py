"""Visualize robustness evaluation summaries.

The script reads robustness summary CSV files produced by evaluate_robustness.py
and writes publication-friendly diagnostic figures plus a short markdown report.
It is defensive about missing columns so partial smoke-test outputs can still be
inspected.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("runs/robustness/results/robustness_summary.csv")
DEFAULT_OUTPUT_DIR = Path("runs/robustness/results/figures")
ATTACK_ORDER = ["none", "observation", "environment", "combined", "unknown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def warn(message: str, warnings: list[str]) -> None:
    warnings.append(message)
    print(f"WARNING: {message}")


def sanitize_filename(value: Any) -> str:
    text = str(value).strip() or "unknown"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unknown"


def load_results(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"robustness summary CSV does not exist: {path}")
    return pd.read_csv(path)


def _safe_text_series(df: pd.DataFrame, column: str, default: str) -> pd.Series:
    if column not in df:
        return pd.Series([default] * len(df), index=df.index, dtype="object")
    return df[column].fillna(default).astype(str)


def _infer_attack_type(row: pd.Series) -> str:
    obs = str(row.get("observation_attack_type", "none")).strip().lower()
    env = str(row.get("environment_attack_type", "none")).strip().lower()
    obs_active = obs not in {"", "none", "nan", "false"}
    env_active = env not in {"", "none", "nan", "false"}
    if obs_active and env_active:
        return "combined"
    if obs_active:
        return "observation"
    if env_active:
        return "environment"
    return "none"


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "policy_name" not in df:
        df["policy_name"] = _safe_text_series(df, "method", "unknown_policy")
    df["policy_name"] = df["policy_name"].fillna("unknown_policy").astype(str)

    if "eval_domain" not in df:
        df["eval_domain"] = "unknown"
    df["eval_domain"] = df["eval_domain"].fillna("unknown").astype(str)

    if "observation_attack_type" not in df:
        df["observation_attack_type"] = "none"
    if "environment_attack_type" not in df:
        df["environment_attack_type"] = "none"

    if "attack_type" not in df:
        df["attack_type"] = df.apply(_infer_attack_type, axis=1)
    df["attack_type"] = df["attack_type"].fillna("unknown").astype(str)

    if "mean_attacked_scalar_cost" not in df and "mean_nominal_scalar_cost" in df:
        df["mean_attacked_scalar_cost"] = df["mean_nominal_scalar_cost"]
    if "mean_nominal_scalar_cost" not in df and "mean_attacked_scalar_cost" in df:
        df["mean_nominal_scalar_cost"] = df["mean_attacked_scalar_cost"]

    for column in ("mean_nominal_scalar_cost", "mean_attacked_scalar_cost"):
        if column not in df:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ("std_nominal_scalar_cost", "std_attacked_scalar_cost"):
        if column not in df:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    if "absolute_degradation" not in df:
        df["absolute_degradation"] = df["mean_attacked_scalar_cost"] - df["mean_nominal_scalar_cost"]
    df["absolute_degradation"] = pd.to_numeric(df["absolute_degradation"], errors="coerce")

    if "relative_degradation" not in df:
        df["relative_degradation"] = df["absolute_degradation"] / (
            df["mean_nominal_scalar_cost"].abs() + 1e-8
        )
    df["relative_degradation"] = pd.to_numeric(df["relative_degradation"], errors="coerce")

    if "success_rate" not in df and "failure_rate" in df:
        df["success_rate"] = 1.0 - pd.to_numeric(df["failure_rate"], errors="coerce")
    if "failure_rate" not in df and "success_rate" in df:
        df["failure_rate"] = 1.0 - pd.to_numeric(df["success_rate"], errors="coerce")

    for column in ("success_rate", "failure_rate", "mean_path_length", "mean_planning_time"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def get_cost_column(df: pd.DataFrame) -> str:
    if "mean_attacked_scalar_cost" in df and df["mean_attacked_scalar_cost"].notna().any():
        return "mean_attacked_scalar_cost"
    return "mean_nominal_scalar_cost"


def ordered_values(series: pd.Series, preferred: list[str] | None = None) -> list[str]:
    values = [str(value) for value in series.dropna().drop_duplicates().tolist()]
    if not preferred:
        return values
    ordered = [value for value in preferred if value in values]
    ordered.extend([value for value in values if value not in ordered])
    return ordered


def _aggregate_for_plot(
    df: pd.DataFrame,
    x_col: str,
    hue_col: str,
    value_col: str,
) -> pd.DataFrame:
    return (
        df.groupby([x_col, hue_col], sort=False, as_index=False)[value_col]
        .mean()
        .dropna(subset=[value_col])
    )


def grouped_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    hue_col: str,
    value_col: str,
    output_path: Path,
    title: str,
    ylabel: str,
    hue_order: list[str] | None = None,
    percent_axis: bool = False,
) -> bool:
    plot_df = _aggregate_for_plot(df, x_col, hue_col, value_col)
    if plot_df.empty:
        return False

    x_values = ordered_values(plot_df[x_col])
    hue_values = ordered_values(plot_df[hue_col], hue_order)
    x = np.arange(len(x_values))
    width = min(0.80 / max(len(hue_values), 1), 0.28)

    fig_width = max(8.0, 1.3 * len(x_values) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    for index, hue in enumerate(hue_values):
        values = []
        subset = plot_df[plot_df[hue_col] == hue].set_index(x_col)
        for x_value in x_values:
            values.append(float(subset[value_col].get(x_value, np.nan)))
        offset = (index - (len(hue_values) - 1) / 2.0) * width
        ax.bar(x + offset, values, width=width, label=hue)

    ax.set_xticks(x)
    ax.set_xticklabels(x_values, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent_axis:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")
    return True


def plot_cost_by_policy_attack(
    df: pd.DataFrame,
    output_dir: str | Path,
    warnings: list[str] | None = None,
) -> list[Path]:
    warnings = warnings if warnings is not None else []
    output_dir = Path(output_dir)
    cost_col = get_cost_column(df)
    paths: list[Path] = []
    for domain, domain_df in df.groupby("eval_domain", sort=False):
        output_path = output_dir / f"fig_cost_by_policy_attack_{sanitize_filename(domain)}.png"
        ok = grouped_bar_chart(
            domain_df,
            x_col="policy_name",
            hue_col="attack_type",
            value_col=cost_col,
            output_path=output_path,
            title="Robustness cost by policy and attack type",
            ylabel="Mean scalar cost lower is better",
            hue_order=ATTACK_ORDER,
        )
        if ok:
            paths.append(output_path)
        else:
            warn(f"Cost chart skipped for domain={domain}: no plottable cost values.", warnings)
    return paths


def plot_relative_degradation(
    df: pd.DataFrame,
    output_dir: str | Path,
    warnings: list[str] | None = None,
) -> list[Path]:
    warnings = warnings if warnings is not None else []
    output_dir = Path(output_dir)
    paths: list[Path] = []
    for domain, domain_df in df.groupby("eval_domain", sort=False):
        plot_df = domain_df.copy()
        nonzero = plot_df["relative_degradation"].fillna(0.0).abs() > 1e-12
        plot_df = plot_df[(plot_df["attack_type"] != "none") | nonzero]
        output_path = output_dir / f"fig_relative_degradation_{sanitize_filename(domain)}.png"
        ok = grouped_bar_chart(
            plot_df,
            x_col="policy_name",
            hue_col="attack_type",
            value_col="relative_degradation",
            output_path=output_path,
            title="Relative degradation by policy and attack type",
            ylabel="Relative degradation",
            hue_order=ATTACK_ORDER,
            percent_axis=True,
        )
        if ok:
            paths.append(output_path)
        else:
            warn(f"Relative degradation chart skipped for domain={domain}: no plottable degradation.", warnings)
    return paths


def _domain_role(domain: str) -> str:
    lowered = domain.lower()
    if "held" in lowered or "out" in lowered:
        return "heldout"
    if "in" in lowered or "train" in lowered:
        return "indomain"
    return "unknown"


def plot_indomain_vs_heldout(
    df: pd.DataFrame,
    output_dir: str | Path,
    warnings: list[str] | None = None,
) -> tuple[list[Path], pd.DataFrame]:
    warnings = warnings if warnings is not None else []
    output_dir = Path(output_dir)
    cost_col = get_cost_column(df)
    domains = df["eval_domain"].dropna().unique().tolist()
    if len(domains) < 2:
        warn("In-domain vs held-out chart skipped: fewer than two eval_domain values.", warnings)
        return [], pd.DataFrame()

    paths: list[Path] = []
    for attack_type, attack_df in df.groupby("attack_type", sort=False):
        output_path = output_dir / f"fig_indomain_vs_heldout_{sanitize_filename(attack_type)}.png"
        ok = grouped_bar_chart(
            attack_df,
            x_col="policy_name",
            hue_col="eval_domain",
            value_col=cost_col,
            output_path=output_path,
            title=f"In-domain vs held-out cost: {attack_type}",
            ylabel="Mean scalar cost lower is better",
        )
        if ok:
            paths.append(output_path)

    gap_rows: list[dict[str, Any]] = []
    grouped = df.groupby(["policy_name", "attack_type", "eval_domain"], sort=False)[cost_col].mean().reset_index()
    for (policy, attack_type), group in grouped.groupby(["policy_name", "attack_type"], sort=False):
        role_map = {}
        for _, row in group.iterrows():
            role = _domain_role(str(row["eval_domain"]))
            role_map.setdefault(role, (str(row["eval_domain"]), float(row[cost_col])))
        if "indomain" in role_map and "heldout" in role_map:
            in_domain_name, in_cost = role_map["indomain"]
            heldout_name, heldout_cost = role_map["heldout"]
            gap_rows.append(
                {
                    "policy_name": policy,
                    "attack_type": attack_type,
                    "in_domain": in_domain_name,
                    "heldout_domain": heldout_name,
                    "in_domain_cost": in_cost,
                    "heldout_cost": heldout_cost,
                    "generalization_gap": heldout_cost - in_cost,
                }
            )

    gap_df = pd.DataFrame(gap_rows)
    gap_path = output_dir / "generalization_gap.csv"
    if not gap_df.empty:
        gap_df.to_csv(gap_path, index=False)
        print(f"Saved table: {gap_path}")
    else:
        warn("generalization_gap.csv skipped: could not identify both in-domain and held-out domains.", warnings)
    return paths, gap_df


def _has_stage_signal(df: pd.DataFrame) -> bool:
    if "training_stage" in df:
        return True
    stage_tokens = ("nominal", "env_ft", "obs_ft", "sequential", "mixed")
    policies = " ".join(df["policy_name"].astype(str).str.lower().tolist())
    return any(token in policies for token in stage_tokens)


def plot_forgetting_heatmap(
    df: pd.DataFrame,
    output_dir: str | Path,
    warnings: list[str] | None = None,
) -> list[Path]:
    warnings = warnings if warnings is not None else []
    output_dir = Path(output_dir)
    if not _has_stage_signal(df):
        warn("Forgetting heatmap skipped: no training_stage column or known stage names.", warnings)
        return []

    row_col = "training_stage" if "training_stage" in df else "policy_name"
    cost_col = get_cost_column(df)
    paths: list[Path] = []
    for domain, domain_df in df.groupby("eval_domain", sort=False):
        pivot = domain_df.pivot_table(
            index=row_col,
            columns="attack_type",
            values=cost_col,
            aggfunc="mean",
        )
        if pivot.shape[0] < 1 or pivot.shape[1] < 2:
            warn(f"Forgetting heatmap skipped for domain={domain}: insufficient stage/attack grid.", warnings)
            continue
        ordered_columns = [name for name in ATTACK_ORDER if name in pivot.columns]
        ordered_columns.extend([name for name in pivot.columns if name not in ordered_columns])
        pivot = pivot[ordered_columns]

        fig_width = max(7.0, 1.1 * len(pivot.columns) + 3.0)
        fig_height = max(4.0, 0.55 * len(pivot.index) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        image = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(f"Forgetting heatmap: {domain}")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label("Mean scalar cost lower is better")
        for row_index in range(pivot.shape[0]):
            for col_index in range(pivot.shape[1]):
                value = pivot.iat[row_index, col_index]
                if np.isfinite(value):
                    ax.text(col_index, row_index, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.tight_layout()
        output_path = output_dir / f"fig_forgetting_heatmap_{sanitize_filename(domain)}.png"
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"Saved figure: {output_path}")
        paths.append(output_path)
    return paths


def plot_recovery_curve(
    df: pd.DataFrame,
    output_dir: str | Path,
    warnings: list[str] | None = None,
) -> list[Path]:
    warnings = warnings if warnings is not None else []
    output_dir = Path(output_dir)
    x_col = next((column for column in ("timesteps", "global_step", "update") if column in df), None)
    y_col = next(
        (
            column
            for column in ("mean_attacked_scalar_cost", "eval_cost", "mean_nominal_scalar_cost")
            if column in df and df[column].notna().any()
        ),
        None,
    )
    if x_col is None or y_col is None:
        warn("Fine-tuning recovery curve skipped because no timesteps/global_step/update column was found.", warnings)
        return []

    line_col = "training_stage" if "training_stage" in df else "policy_name"
    plot_df = df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        warn("Fine-tuning recovery curve skipped: no finite x/y values.", warnings)
        return []

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, group in plot_df.groupby(line_col, sort=False):
        group = group.sort_values(x_col)
        ax.plot(group[x_col], group[y_col], marker="o", label=str(label))
    ax.set_xlabel(x_col)
    ax.set_ylabel("Mean scalar cost lower is better")
    ax.set_title("Fine-tuning recovery curve")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / "fig_finetune_recovery_curve.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {output_path}")
    return [output_path]


def plot_success_rate(
    df: pd.DataFrame,
    output_dir: str | Path,
    warnings: list[str] | None = None,
) -> list[Path]:
    warnings = warnings if warnings is not None else []
    output_dir = Path(output_dir)
    if "success_rate" in df and df["success_rate"].notna().any():
        metric = "success_rate"
        ylabel = "Success rate"
    elif "failure_rate" in df and df["failure_rate"].notna().any():
        metric = "failure_rate"
        ylabel = "Failure rate"
    else:
        warn("Success/failure chart skipped: success_rate and failure_rate are missing.", warnings)
        return []

    paths: list[Path] = []
    for domain, domain_df in df.groupby("eval_domain", sort=False):
        output_path = output_dir / f"fig_{metric}_{sanitize_filename(domain)}.png"
        ok = grouped_bar_chart(
            domain_df,
            x_col="policy_name",
            hue_col="attack_type",
            value_col=metric,
            output_path=output_path,
            title=f"{ylabel} by policy and attack type",
            ylabel=ylabel,
            hue_order=ATTACK_ORDER,
            percent_axis=True,
        )
        if ok:
            paths.append(output_path)
        else:
            warn(f"{ylabel} chart skipped for domain={domain}: no plottable values.", warnings)
    return paths


def plot_path_length_and_time(
    df: pd.DataFrame,
    output_dir: str | Path,
    warnings: list[str] | None = None,
) -> list[Path]:
    warnings = warnings if warnings is not None else []
    output_dir = Path(output_dir)
    paths: list[Path] = []
    metrics = [
        ("mean_path_length", "Mean path length", "fig_path_length"),
        ("mean_planning_time", "Mean planning time (seconds)", "fig_planning_time"),
    ]
    for metric, ylabel, prefix in metrics:
        if metric not in df or not df[metric].notna().any():
            warn(f"{ylabel} chart skipped: {metric} is missing.", warnings)
            continue
        for domain, domain_df in df.groupby("eval_domain", sort=False):
            output_path = output_dir / f"{prefix}_{sanitize_filename(domain)}.png"
            ok = grouped_bar_chart(
                domain_df,
                x_col="policy_name",
                hue_col="attack_type",
                value_col=metric,
                output_path=output_path,
                title=f"{ylabel} by policy and attack type",
                ylabel=ylabel,
                hue_order=ATTACK_ORDER,
            )
            if ok:
                paths.append(output_path)
            else:
                warn(f"{ylabel} chart skipped for domain={domain}: no plottable values.", warnings)
    return paths


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def data_summary_rows(df: pd.DataFrame) -> list[str]:
    metrics = [
        column
        for column in (
            "mean_nominal_scalar_cost",
            "mean_attacked_scalar_cost",
            "relative_degradation",
            "success_rate",
            "failure_rate",
            "mean_path_length",
            "mean_planning_time",
        )
        if column in df
    ]
    return [
        f"- number of rows: {len(df)}",
        f"- policy names: {', '.join(ordered_values(df['policy_name'])) or 'none'}",
        f"- eval domains: {', '.join(ordered_values(df['eval_domain'])) or 'none'}",
        f"- attack types: {', '.join(ordered_values(df['attack_type'], ATTACK_ORDER)) or 'none'}",
        f"- available metrics: {', '.join(metrics) or 'none'}",
    ]


def best_policy_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    cost_col = get_cost_column(df)
    rows: list[dict[str, Any]] = []
    for (domain, attack_type), group in df.groupby(["eval_domain", "attack_type"], sort=False):
        group = group.dropna(subset=[cost_col])
        if group.empty:
            continue
        best = group.loc[group[cost_col].idxmin()]
        rows.append(
            {
                "eval_domain": domain,
                "attack_type": attack_type,
                "best_policy": best["policy_name"],
                "cost": float(best[cost_col]),
            }
        )
    return rows


def smallest_degradation_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "relative_degradation" not in df:
        return rows
    for (domain, attack_type), group in df.groupby(["eval_domain", "attack_type"], sort=False):
        group = group[group["attack_type"] != "none"].dropna(subset=["relative_degradation"])
        if group.empty:
            continue
        best = group.loc[group["relative_degradation"].idxmin()]
        rows.append(
            {
                "eval_domain": domain,
                "attack_type": attack_type,
                "best_policy": best["policy_name"],
                "relative_degradation": float(best["relative_degradation"]),
            }
        )
    return rows


def forgetting_notes(df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    if not _has_stage_signal(df):
        return ["No sequential/mixed training-stage signal was found."]
    cost_col = get_cost_column(df)
    lower_policy = df["policy_name"].astype(str).str.lower()
    for attack in ("environment", "observation"):
        nominal = df[(lower_policy.str.contains("nominal")) & (df["attack_type"] == attack)]
        tuned = df[(lower_policy.str.contains("env_ft" if attack == "environment" else "obs_ft")) & (df["attack_type"] == attack)]
        if nominal.empty or tuned.empty:
            notes.append(f"{attack}: insufficient nominal/fine-tuned rows for recovery comparison.")
            continue
        nominal_cost = float(nominal[cost_col].mean())
        tuned_cost = float(tuned[cost_col].mean())
        direction = "lower" if tuned_cost < nominal_cost else "not lower"
        notes.append(f"{attack}: fine-tuned mean cost is {direction} than nominal ({tuned_cost:.4g} vs {nominal_cost:.4g}).")
    sequential = df[lower_policy.str.contains("sequential")]
    if sequential.empty:
        notes.append("No sequential policy rows found.")
    else:
        notes.append("Sequential policy rows found; inspect the forgetting heatmap for earlier-attack degradation.")
    return notes


def write_markdown_report(
    df: pd.DataFrame,
    output_dir: str | Path,
    warnings: list[str] | None = None,
    gap_df: pd.DataFrame | None = None,
) -> Path:
    warnings = warnings if warnings is not None else []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "robustness_visual_report.md"

    lines: list[str] = ["# Robustness Visual Report", ""]
    lines.append("## 1. Data Summary")
    lines.extend(data_summary_rows(df))
    lines.append("")

    lines.append("## 2. Best Policy by Attack Type")
    lines.append("Lower mean scalar cost is better.")
    lines.append(markdown_table(best_policy_rows(df), ["eval_domain", "attack_type", "best_policy", "cost"]))
    lines.append("")

    lines.append("## 3. Smallest Degradation")
    degradation = smallest_degradation_rows(df)
    if degradation:
        lines.append(markdown_table(degradation, ["eval_domain", "attack_type", "best_policy", "relative_degradation"]))
    else:
        lines.append("_Relative degradation was not available._")
    lines.append("")

    lines.append("## 4. Generalization Gap")
    if gap_df is not None and not gap_df.empty:
        lines.append(markdown_table(gap_df.to_dict("records"), list(gap_df.columns)))
        largest = gap_df.loc[gap_df["generalization_gap"].idxmax()]
        smallest = gap_df.loc[gap_df["generalization_gap"].idxmin()]
        lines.append("")
        lines.append(
            f"Largest gap: {largest['policy_name']} / {largest['attack_type']} = "
            f"{float(largest['generalization_gap']):.6g}."
        )
        lines.append(
            f"Smallest gap: {smallest['policy_name']} / {smallest['attack_type']} = "
            f"{float(smallest['generalization_gap']):.6g}."
        )
    else:
        lines.append("_Generalization gap was not computed because matching in-domain and held-out domains were not found._")
    lines.append("")

    lines.append("## 5. Forgetting Notes")
    for note in forgetting_notes(df):
        lines.append(f"- {note}")
    lines.append("")

    lines.append("## 6. Missing Data Warnings")
    if warnings:
        for message in warnings:
            lines.append(f"- {message}")
    else:
        lines.append("_No missing-data warnings._")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved report: {report_path}")
    return report_path


def run_step(name: str, warnings: list[str], callback: Any) -> Any:
    try:
        return callback()
    except Exception as exc:
        warn(f"{name} failed: {exc}", warnings)
        return None


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    df = ensure_columns(load_results(args.input))
    run_step("Cost by policy/attack plot", warnings, lambda: plot_cost_by_policy_attack(df, output_dir, warnings))
    run_step("Relative degradation plot", warnings, lambda: plot_relative_degradation(df, output_dir, warnings))
    gap_result = run_step("In-domain vs held-out plot", warnings, lambda: plot_indomain_vs_heldout(df, output_dir, warnings))
    gap_df = gap_result[1] if isinstance(gap_result, tuple) else pd.DataFrame()
    run_step("Forgetting heatmap", warnings, lambda: plot_forgetting_heatmap(df, output_dir, warnings))
    run_step("Fine-tuning recovery curve", warnings, lambda: plot_recovery_curve(df, output_dir, warnings))
    run_step("Success/failure plot", warnings, lambda: plot_success_rate(df, output_dir, warnings))
    run_step("Path length/planning time plots", warnings, lambda: plot_path_length_and_time(df, output_dir, warnings))
    write_markdown_report(df, output_dir, warnings, gap_df)


if __name__ == "__main__":
    main()
