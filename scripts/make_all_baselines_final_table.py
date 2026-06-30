#!/usr/bin/env python
"""Create 9-scenario final-result tables for all current baselines."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

import update_ldac_baseline_tables as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "runs" / "baseline_tables"

LEARNING_ROOTS = {
    "PPO": PROJECT_ROOT / "runs" / "rl_baselines" / "ppo",
    "SAC": PROJECT_ROOT / "runs" / "rl_baselines" / "sac",
    "CDR-SAC": PROJECT_ROOT / "runs" / "sac_modified" / "cdr_sac_from_sac_nominal_9scenarios_2seeds_20260627",
    "Stackelberg-SAC": PROJECT_ROOT
    / "runs"
    / "sac_modified"
    / "stackelberg_sac_from_sac_nominal_9scenarios_2seeds_20260628",
    "VALT-SAC": PROJECT_ROOT / "runs" / "sac_modified" / "valt_sac_from_sac_nominal_9scenarios_3seeds_20260628",
}

METHOD_ORDER = (
    ("Non-learning", "Emergency"),
    ("Non-learning", "Val-Best"),
    ("Non-learning", "Minimax"),
    ("Non-learning", "Heuristic"),
    ("Non-learning", "Guard"),
    ("Non-learning", "Fixed"),
    ("Non-learning", "Risk-Inflated A*"),
    ("Non-learning", "Belief-CVaR A*"),
    ("Learning", "PPO"),
    ("Learning", "SAC"),
    ("Learning", "CDR-SAC"),
    ("Learning", "Stackelberg-SAC"),
    ("Learning", "VALT-SAC"),
    ("Learning", "LDAC-SAC"),
)


def seed_from_path(path: Path) -> int:
    return base.seed_from_path(path)


def scenario_from_summary_path(path: Path) -> str:
    parent = path.parent.parent.name
    suffix = "_shock_recovery_5seeds"
    return parent[: -len(suffix)] if parent.endswith(suffix) else parent


def read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def baseline_learning_summary_paths(method: str) -> list[Path]:
    root = LEARNING_ROOTS[method]
    paths: list[Path] = []
    if method in {"PPO", "SAC"}:
        for scenario in base.SCENARIOS:
            paths.extend(sorted((root / f"{scenario}_shock_recovery_5seeds").glob("seed*/shock_recovery_summary.csv")))
    else:
        for scenario in base.SCENARIOS:
            paths.extend(sorted((root / scenario).glob("seed*/shock_recovery_summary.csv")))
    return paths


def ldac_summary_paths() -> list[Path]:
    paths: list[Path] = []
    for difficulty, roots in base.LDAC_ROOTS_BY_DIFFICULTY.items():
        for root in roots:
            paths.extend(sorted(root.glob(f"level*_{difficulty}/seed*/shock_recovery_summary.csv")))
    return paths


def learning_summary_paths(method: str) -> list[Path]:
    if method == "LDAC-SAC":
        return ldac_summary_paths()
    return baseline_learning_summary_paths(method)


def recompute_learning_points(heldout_only: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in [m for family, m in METHOD_ORDER if family == "Learning"]:
        for summary_path in learning_summary_paths(method):
            seed_dir = summary_path.parent
            scenario = scenario_from_summary_path(summary_path)
            if scenario not in base.SCENARIOS:
                continue
            seed = seed_from_path(seed_dir)
            level, difficulty = base.scenario_parts(scenario)
            frame = read_summary(summary_path)
            required = {"eval_domain", "clean_nominal_cost", "final_recovery_cost"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"missing columns in {summary_path}: {sorted(missing)}")
            for _, row in frame.iterrows():
                eval_domain = str(row["eval_domain"])
                if heldout_only and "heldout" not in eval_domain.lower():
                    continue
                reference = base.ppo_reference_clean_cost(scenario, seed, eval_domain)
                clean_cost = float(row["clean_nominal_cost"])
                final_cost = float(row["final_recovery_cost"])
                rows.append(
                    {
                        "family": "Learning",
                        "method": method,
                        "level": level,
                        "difficulty": difficulty,
                        "seed": seed,
                        "eval_domain": eval_domain,
                        "clean_index": 100.0 * reference / clean_cost,
                        "corrupted_index": 100.0 * reference / final_cost,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no learning rows found")
    return out.sort_values(["method", "difficulty", "level", "seed", "eval_domain"]).reset_index(drop=True)


def make_points(source_points_path: Path, out_path: Path, heldout_only: bool) -> pd.DataFrame:
    points = pd.read_csv(source_points_path)
    nonlearning = points[points["family"].eq("Non-learning")].copy()
    updated = pd.concat([nonlearning, recompute_learning_points(heldout_only)], ignore_index=True)
    updated = updated.sort_values(["family", "method", "difficulty", "level", "seed", "eval_domain"]).reset_index(
        drop=True
    )
    updated.to_csv(out_path, index=False)
    return updated


def aggregate(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, method in METHOD_ORDER:
        row: dict[str, object] = {"Family": family, "Method": method}
        subset = points[(points["family"] == family) & (points["method"] == method)]
        for difficulty in base.DIFFICULTIES:
            for level in base.LEVELS:
                scenario_subset = subset[(subset["difficulty"] == difficulty) & (subset["level"] == level)]
                prefix = f"{level}_{difficulty}"
                if scenario_subset.empty:
                    row[f"{prefix}_clean_index_mean"] = np.nan
                    row[f"{prefix}_clean_index_std"] = np.nan
                    row[f"{prefix}_corrupted_index_mean"] = np.nan
                    row[f"{prefix}_corrupted_index_std"] = np.nan
                    row[f"{prefix}_n"] = 0
                    continue
                row[f"{prefix}_clean_index_mean"] = float(scenario_subset["clean_index"].mean())
                row[f"{prefix}_clean_index_std"] = float(scenario_subset["clean_index"].std(ddof=1))
                row[f"{prefix}_corrupted_index_mean"] = float(scenario_subset["corrupted_index"].mean())
                row[f"{prefix}_corrupted_index_std"] = float(scenario_subset["corrupted_index"].std(ddof=1))
                row[f"{prefix}_n"] = int(len(scenario_subset))
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_value(mean: float, std: float, bold: bool = False) -> str:
    if not np.isfinite(mean):
        text = "--"
    else:
        std = 0.0 if not np.isfinite(std) else std
        text = f"{mean:.2f} $\\pm$ {std:.2f}"
    return f"\\textbf{{{text}}}" if bold and text != "--" else text


def csv_fmt(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "--"
    std = 0.0 if not np.isfinite(std) else std
    return f"{mean:.2f} ± {std:.2f}"


def write_paper_csv(summary: pd.DataFrame, out_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        out = {"Family": row["Family"], "Method": row["Method"]}
        for difficulty in base.DIFFICULTIES:
            for level in base.LEVELS:
                label = f"{level.replace('level', 'L')} {difficulty.title()}"
                prefix = f"{level}_{difficulty}"
                out[f"{label} Clean"] = csv_fmt(row[f"{prefix}_clean_index_mean"], row[f"{prefix}_clean_index_std"])
                out[f"{label} Corr. Final"] = csv_fmt(
                    row[f"{prefix}_corrupted_index_mean"],
                    row[f"{prefix}_corrupted_index_std"],
                )
        rows.append(out)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_latex_single(summary: pd.DataFrame, out_path: Path, heldout_only: bool) -> None:
    caption_scope = "Held-out scenario-level" if heldout_only else "Scenario-level"
    scope_phrase = "over held-out evaluations only" if heldout_only else "over completed seed/domain evaluations"
    label = "tab:heldout_final_all_baselines_9scenario" if heldout_only else "tab:final_all_baselines_9scenario"
    caption = (
        f"{caption_scope} clean-map and final corrupted-map performance for all baselines, grouped by difficulty. "
        f"Values are common-reference performance index (mean $\\pm$ std) {scope_phrase}; higher is better. "
        "The common reference is the matching PPO clean nominal cost for the same scenario, seed, and evaluation domain. "
        "For learning methods, Corr. is the final corrupted-map recovery evaluation; for non-learning planners, "
        "Corr. is the static corrupted-map performance. Bold indicates the best value within each scenario/condition column."
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


def make_table_set(source_points: str, prefix: str, heldout_only: bool) -> None:
    points_path = TABLE_DIR / source_points
    out_points = TABLE_DIR / f"{prefix}_points.csv"
    out_summary = TABLE_DIR / f"{prefix}_summary.csv"
    out_paper = TABLE_DIR / f"{prefix}_paper_table.csv"
    out_latex = TABLE_DIR / f"{prefix}_single_table_summary.tex"

    points = make_points(points_path, out_points, heldout_only)
    summary = aggregate(points)
    summary.to_csv(out_summary, index=False)
    write_paper_csv(summary, out_paper)
    write_latex_single(summary, out_latex, heldout_only)
    print(f"Updated {out_points}")
    print(f"Updated {out_summary}")
    print(f"Updated {out_paper}")
    print(f"Updated {out_latex}")


def main() -> None:
    make_table_set(
        "heldout_clean_corrupted_9scenario_points.csv",
        "heldout_final_all_baselines_9scenario",
        heldout_only=True,
    )
    make_table_set(
        "clean_corrupted_9scenario_points.csv",
        "final_all_baselines_9scenario",
        heldout_only=False,
    )


if __name__ == "__main__":
    main()
