#!/usr/bin/env python
"""Create paper tables that use the best recovery checkpoint for learning methods."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import update_ldac_baseline_tables as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "runs" / "baseline_tables"
ALL_LEARNING_DIR = PROJECT_ROOT / "runs" / "sac_modified" / "analysis" / "all_learning_protocol_20260629"

LEARNING_METHODS = ("PPO", "SAC", "LDAC-SAC")


def read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def baseline_summary_paths(method: str) -> list[Path]:
    algo = method.lower()
    paths: list[Path] = []
    for scenario in base.SCENARIOS:
        for seed in range(5):
            paths.append(
                PROJECT_ROOT
                / "runs"
                / "rl_baselines"
                / algo
                / f"{scenario}_shock_recovery_5seeds"
                / f"seed{seed}"
                / "shock_recovery_summary.csv"
            )
    return paths


def ldac_summary_paths() -> list[Path]:
    paths: list[Path] = []
    for difficulty, roots in base.LDAC_ROOTS_BY_DIFFICULTY.items():
        for root in roots:
            paths.extend(sorted(root.glob(f"level*_{difficulty}/seed*/shock_recovery_summary.csv")))
    return paths


def learning_summary_paths(method: str) -> list[Path]:
    if method in {"PPO", "SAC"}:
        return baseline_summary_paths(method)
    if method == "LDAC-SAC":
        return ldac_summary_paths()
    raise ValueError(method)


def scenario_from_summary_path(summary_path: Path) -> str:
    parent_name = summary_path.parent.parent.name
    suffix = "_shock_recovery_5seeds"
    if parent_name.endswith(suffix):
        return parent_name[: -len(suffix)]
    return parent_name


def recompute_learning_best_points(heldout_only: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in LEARNING_METHODS:
        for summary_path in learning_summary_paths(method):
            seed_dir = summary_path.parent
            scenario = scenario_from_summary_path(summary_path)
            if scenario not in base.SCENARIOS:
                continue
            seed = base.seed_from_path(seed_dir)
            level, difficulty = base.scenario_parts(scenario)
            frame = read_summary(summary_path)
            required = {"eval_domain", "clean_nominal_cost", "best_recovery_cost"}
            if not required.issubset(frame.columns):
                raise ValueError(f"missing columns in {summary_path}: {sorted(required - set(frame.columns))}")
            for _, row in frame.iterrows():
                eval_domain = str(row["eval_domain"])
                if heldout_only and "heldout" not in eval_domain.lower():
                    continue
                reference = base.ppo_reference_clean_cost(scenario, seed, eval_domain)
                clean_cost = float(row["clean_nominal_cost"])
                best_cost = float(row["best_recovery_cost"])
                rows.append(
                    {
                        "family": "Learning",
                        "method": method,
                        "level": level,
                        "difficulty": difficulty,
                        "seed": seed,
                        "eval_domain": eval_domain,
                        "clean_index": 100.0 * reference / clean_cost,
                        "corrupted_index": 100.0 * reference / best_cost,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no learning best-recovery rows were recomputed")
    return out.sort_values(["method", "difficulty", "level", "seed", "eval_domain"]).reset_index(drop=True)


def selected_step_cost(summary_path: Path, eval_domain: str, selected_step: int) -> float:
    curve_path = summary_path.parent / "shock_recovery_curve.csv"
    curve = read_summary(curve_path)
    required = {"eval_domain", "phase", "attack_type", "recovery_step", "mean_attacked_scalar_cost"}
    if not required.issubset(curve.columns):
        raise ValueError(f"missing columns in {curve_path}: {sorted(required - set(curve.columns))}")
    rows = curve[
        (curve["eval_domain"].astype(str) == eval_domain)
        & (curve["attack_type"].astype(str) == "environment")
        & (curve["phase"].astype(str).isin(["shock", "recovery"]))
    ].copy()
    if rows.empty:
        raise ValueError(f"missing selected-step curve rows in {curve_path} for {eval_domain}")
    rows["step_distance"] = (rows["recovery_step"].astype(float) - float(selected_step)).abs()
    return float(rows.sort_values(["step_distance", "recovery_step"]).iloc[0]["mean_attacked_scalar_cost"])


def training_domain_mask(values: pd.Series) -> pd.Series:
    lowered = values.astype(str).str.lower()
    return lowered.str.contains("train", regex=False) | lowered.str.contains("in_domain", regex=False)


def train_selected_step(frame: pd.DataFrame, summary_path: Path) -> int:
    train_rows = frame[training_domain_mask(frame["eval_domain"])]
    if not train_rows.empty and "best_recovery_step" in train_rows.columns:
        step = train_rows.iloc[0]["best_recovery_step"]
        if pd.notna(step):
            return int(step)

    curve_path = summary_path.parent / "shock_recovery_curve.csv"
    curve = read_summary(curve_path)
    train = curve[
        training_domain_mask(curve["eval_domain"])
        & (curve["attack_type"].astype(str) == "environment")
        & (curve["phase"].astype(str).isin(["shock", "recovery"]))
    ].copy()
    clean = curve[
        training_domain_mask(curve["eval_domain"])
        & (curve["attack_type"].astype(str) == "none")
        & (curve["phase"].astype(str) == "shock")
    ]
    if train.empty or clean.empty:
        raise ValueError(f"cannot select train best step from {summary_path}")
    clean_cost = float(clean.iloc[0]["mean_attacked_scalar_cost"])
    train["performance_index"] = 100.0 * clean_cost / train["mean_attacked_scalar_cost"].astype(float).clip(lower=1e-12)
    return int(train.sort_values(["performance_index", "recovery_step"], ascending=[False, True]).iloc[0]["recovery_step"])


def recompute_learning_train_selected_best_points(heldout_only: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in LEARNING_METHODS:
        for summary_path in learning_summary_paths(method):
            seed_dir = summary_path.parent
            scenario = scenario_from_summary_path(summary_path)
            if scenario not in base.SCENARIOS:
                continue
            seed = base.seed_from_path(seed_dir)
            level, difficulty = base.scenario_parts(scenario)
            frame = read_summary(summary_path)
            required = {"eval_domain", "clean_nominal_cost"}
            if not required.issubset(frame.columns):
                raise ValueError(f"missing columns in {summary_path}: {sorted(required - set(frame.columns))}")
            selected_step = train_selected_step(frame, summary_path)
            for _, row in frame.iterrows():
                eval_domain = str(row["eval_domain"])
                if heldout_only and "heldout" not in eval_domain.lower():
                    continue
                reference = base.ppo_reference_clean_cost(scenario, seed, eval_domain)
                clean_cost = float(row["clean_nominal_cost"])
                selected_cost = selected_step_cost(summary_path, eval_domain, selected_step)
                rows.append(
                    {
                        "family": "Learning",
                        "method": method,
                        "level": level,
                        "difficulty": difficulty,
                        "seed": seed,
                        "eval_domain": eval_domain,
                        "selected_recovery_step": selected_step,
                        "clean_index": 100.0 * reference / clean_cost,
                        "corrupted_index": 100.0 * reference / selected_cost,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no train-selected best-recovery rows were recomputed")
    return out.sort_values(["method", "difficulty", "level", "seed", "eval_domain"]).reset_index(drop=True)


def make_best_points(
    source_points_path: Path,
    out_path: Path,
    heldout_only: bool,
    train_selected: bool = False,
) -> pd.DataFrame:
    points = pd.read_csv(source_points_path)
    keep = ~((points["family"] == "Learning") & (points["method"].isin(LEARNING_METHODS)))
    learning = (
        recompute_learning_train_selected_best_points(heldout_only)
        if train_selected
        else recompute_learning_best_points(heldout_only)
    )
    updated = pd.concat([points.loc[keep].copy(), learning], ignore_index=True)
    updated = updated.sort_values(["family", "method", "difficulty", "level", "seed", "eval_domain"]).reset_index(
        drop=True
    )
    updated.to_csv(out_path, index=False)
    return updated


def fmt_value(mean: float, std: float, bold: bool = False) -> str:
    if not np.isfinite(mean):
        text = "--"
    else:
        std = 0.0 if not np.isfinite(std) else std
        text = f"{mean:.2f} $\\pm$ {std:.2f}"
    return f"\\textbf{{{text}}}" if bold and text != "--" else text


def write_best_latex(summary: pd.DataFrame, out_path: Path, heldout_only: bool, train_selected: bool = False) -> None:
    caption_scope = "Held-out scenario-level" if heldout_only else "Scenario-level"
    scope_phrase = "over held-out evaluations only" if heldout_only else "over completed seed/domain evaluations"
    label_prefix = "trainselected_best" if train_selected else "best"
    label = (
        f"tab:heldout_{label_prefix}_clean_corrupted_9scenario"
        if heldout_only
        else f"tab:{label_prefix}_clean_corrupted_9scenario"
    )
    selection_phrase = (
        "For learning methods, Corr. uses the recovery checkpoint selected by the best train-task recovery performance, "
        "then evaluated on the reported domain; "
        if train_selected
        else "For learning methods, Corr. uses the best checkpoint observed during corrupted-map recovery training; "
    )
    caption = (
        f"{caption_scope} clean-map and best corrupted-map performance, grouped by difficulty. "
        f"Values are common-reference performance index (mean $\\pm$ std) {scope_phrase}; higher is better. "
        "The common reference is the matching PPO clean nominal cost for the same scenario, seed, and evaluation domain. "
        f"{selection_phrase}for non-learning planners, Corr. is the static corrupted-map performance. "
        "Bold indicates the best value within each scenario/condition column."
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Family & Method & \\multicolumn{2}{c}{Level 1} & \\multicolumn{2}{c}{Level 2} & \\multicolumn{2}{c}{Level 3} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
        " & & Clean & Corr. & Clean & Corr. & Clean & Corr. \\\\",
        "\\midrule",
    ]
    for difficulty_index, difficulty in enumerate(base.DIFFICULTIES):
        if difficulty_index > 0:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{difficulty.title()}}}}} \\\\")
        best: dict[tuple[str, str], float] = {}
        for level in base.LEVELS:
            prefix = f"{level}_{difficulty}"
            best[(level, "clean")] = float(summary[f"{prefix}_clean_index_mean"].max())
            best[(level, "corrupted")] = float(summary[f"{prefix}_corrupted_index_mean"].max())
        last_family = None
        for _, row in summary.iterrows():
            family = str(row["Family"])
            method = str(row["Method"])
            if last_family == "Non-learning" and family == "Learning":
                lines.append("\\addlinespace[0.2em]")
            cells = [family, method]
            for level in base.LEVELS:
                prefix = f"{level}_{difficulty}"
                clean_mean = float(row[f"{prefix}_clean_index_mean"])
                corr_mean = float(row[f"{prefix}_corrupted_index_mean"])
                cells.append(
                    fmt_value(
                        clean_mean,
                        float(row[f"{prefix}_clean_index_std"]),
                        math.isfinite(clean_mean) and abs(clean_mean - best[(level, "clean")]) < 1e-9,
                    )
                )
                cells.append(
                    fmt_value(
                        corr_mean,
                        float(row[f"{prefix}_corrupted_index_std"]),
                        math.isfinite(corr_mean) and abs(corr_mean - best[(level, "corrupted")]) < 1e-9,
                    )
                )
            lines.append(" & ".join(cells) + " \\\\")
            last_family = family
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}"])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_paper_csv(summary: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        out = {"Family": row["Family"], "Method": row["Method"]}
        for difficulty in base.DIFFICULTIES:
            for level in base.LEVELS:
                prefix = f"{level}_{difficulty}"
                label = f"{level.replace('level', 'L')} {difficulty.title()}"
                out[f"{label} Clean"] = base.csv_fmt(
                    row[f"{prefix}_clean_index_mean"], row[f"{prefix}_clean_index_std"]
                )
                out[f"{label} Corr. Best"] = base.csv_fmt(
                    row[f"{prefix}_corrupted_index_mean"], row[f"{prefix}_corrupted_index_std"]
                )
        rows.append(out)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def fmt_compact(mean: float, std: float, bold: bool = False) -> str:
    if not math.isfinite(mean):
        text = "--"
    else:
        std = 0.0 if not math.isfinite(std) else std
        text = f"{mean:.2f} $\\pm$ {std:.2f}"
    return f"\\textbf{{{text}}}" if bold and text != "--" else text


def make_all_learning_best_table() -> None:
    source = ALL_LEARNING_DIR / "fig_all_learning_train_drop_recovery_scenario_summary.csv"
    if not source.exists():
        return
    summary = pd.read_csv(source)
    rows: list[dict[str, object]] = []
    for method, group in summary.groupby("method", sort=False):
        group = group[group["num_seed_dirs"] > 0]
        if group.empty:
            continue
        seed_counts = group["num_seed_dirs"].astype(int)
        seed_text = (
            str(int(seed_counts.iloc[0]))
            if int(seed_counts.min()) == int(seed_counts.max())
            else f"{int(seed_counts.min())}-{int(seed_counts.max())}"
        )
        out = {"Method": method, "Coverage": f"{len(group)}/9", "Seeds": seed_text}
        for label, column in (
            ("Shock", "shock_index"),
            ("Best", "best_recovery_index"),
            ("Best gain", "best_gain_vs_shock"),
            ("AUC", "recovery_auc_index"),
        ):
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            out[f"{label} mean"] = float(values.mean()) if not values.empty else math.nan
            out[f"{label} std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(out)
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(ALL_LEARNING_DIR / "all_learning_best_recovery_summary.csv", index=False)

    best_columns = ("Best", "Best gain", "AUC")
    maxima = {label: float(aggregate[f"{label} mean"].max()) for label in best_columns}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Best-checkpoint recovery performance across all learning methods and nine scenarios. Values are scenario-level means $\\pm$ standard deviations of the performance index; higher is better.}",
        "\\label{tab:all_learning_best_recovery}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Coverage & Seeds & Shock & Best & Best gain & AUC \\\\",
        "\\midrule",
    ]
    for _, row in aggregate.iterrows():
        cells = [str(row["Method"]), str(row["Coverage"]), str(row["Seeds"])]
        for label in ("Shock", "Best", "Best gain", "AUC"):
            mean = float(row[f"{label} mean"])
            std = float(row[f"{label} std"])
            cells.append(fmt_compact(mean, std, label in maxima and abs(mean - maxima[label]) < 1e-9))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    (ALL_LEARNING_DIR / "all_learning_best_recovery_summary.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def make_table_set(source_points: str, prefix: str, heldout_only: bool, train_selected: bool = False) -> None:
    points_path = TABLE_DIR / source_points
    out_points = TABLE_DIR / f"{prefix}_points.csv"
    out_summary = TABLE_DIR / f"{prefix}_summary.csv"
    out_paper_csv = TABLE_DIR / f"{prefix}_paper_table.csv"
    out_latex = TABLE_DIR / f"{prefix}_single_table_summary.tex"

    points = make_best_points(points_path, out_points, heldout_only, train_selected=train_selected)
    summary = base.aggregate(points)
    summary.to_csv(out_summary, index=False)
    write_best_paper_csv(summary, out_paper_csv)
    write_best_latex(summary, out_latex, heldout_only, train_selected=train_selected)
    print(f"Updated {out_points}")
    print(f"Updated {out_summary}")
    print(f"Updated {out_paper_csv}")
    print(f"Updated {out_latex}")


def main() -> None:
    make_table_set(
        "heldout_clean_corrupted_9scenario_points.csv",
        "heldout_best_clean_corrupted_9scenario",
        heldout_only=True,
    )
    make_table_set(
        "clean_corrupted_9scenario_points.csv",
        "best_clean_corrupted_9scenario",
        heldout_only=False,
    )
    make_table_set(
        "heldout_clean_corrupted_9scenario_points.csv",
        "heldout_trainselected_best_clean_corrupted_9scenario",
        heldout_only=True,
        train_selected=True,
    )
    make_table_set(
        "clean_corrupted_9scenario_points.csv",
        "trainselected_best_clean_corrupted_9scenario",
        heldout_only=False,
        train_selected=True,
    )
    make_all_learning_best_table()
    print(f"Updated {ALL_LEARNING_DIR / 'all_learning_best_recovery_summary.csv'}")
    print(f"Updated {ALL_LEARNING_DIR / 'all_learning_best_recovery_summary.tex'}")


if __name__ == "__main__":
    main()
