#!/usr/bin/env python3
"""Refresh clean/corrupted performance tables with completed LDAC seeds.

The paper tables already contain all PPO/SAC/non-learning rows. This script
replaces all LDAC-SAC rows with the completed five-seed Easy/Medium/Hard set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "runs" / "baseline_tables"

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

SCENARIOS = (
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

DIFFICULTIES = ("easy", "medium", "hard")
LEVELS = ("level1", "level2", "level3")

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
    ("Learning", "LDAC-SAC"),
)


def scenario_parts(scenario: str) -> tuple[str, str]:
    level, difficulty = scenario.split("_", 1)
    return level, difficulty


def seed_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    raise ValueError(f"could not parse seed from {path}")


def read_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def ppo_reference_clean_cost(scenario: str, seed: int, eval_domain: str) -> float:
    path = (
        PROJECT_ROOT
        / "runs"
        / "rl_baselines"
        / "ppo"
        / f"{scenario}_shock_recovery_5seeds"
        / f"seed{seed}"
        / "shock_recovery_summary.csv"
    )
    frame = read_summary(path)
    match = frame[frame["eval_domain"].astype(str) == str(eval_domain)]
    if match.empty:
        raise ValueError(f"missing PPO reference for {scenario} seed{seed} domain={eval_domain}")
    return float(match.iloc[0]["clean_nominal_cost"])


def recompute_ldac_points(heldout_only: bool) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for difficulty, roots in LDAC_ROOTS_BY_DIFFICULTY.items():
        for root in roots:
            for summary_path in sorted(root.glob(f"level*_{difficulty}/seed*/shock_recovery_summary.csv")):
                seed_dir = summary_path.parent
                scenario = seed_dir.parent.name
                seed = seed_from_path(seed_dir)
                level, parsed_difficulty = scenario_parts(scenario)
                frame = read_summary(summary_path)
                for _, row in frame.iterrows():
                    eval_domain = str(row["eval_domain"])
                    if heldout_only and "heldout" not in eval_domain.lower():
                        continue
                    reference = ppo_reference_clean_cost(scenario, seed, eval_domain)
                    clean_cost = float(row["clean_nominal_cost"])
                    corrupted_cost = float(row["final_recovery_cost"])
                    rows.append(
                        {
                            "family": "Learning",
                            "method": "LDAC-SAC",
                            "level": level,
                            "difficulty": parsed_difficulty,
                            "seed": seed,
                            "eval_domain": eval_domain,
                            "clean_index": 100.0 * reference / clean_cost,
                            "corrupted_index": 100.0 * reference / corrupted_cost,
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no LDAC rows were recomputed")
    out = out.sort_values(["level", "difficulty", "seed", "eval_domain"]).reset_index(drop=True)
    return out


def update_points(points_path: Path, heldout_only: bool) -> pd.DataFrame:
    points = pd.read_csv(points_path)
    keep = ~(
        (points["family"] == "Learning")
        & (points["method"] == "LDAC-SAC")
    )
    updated = pd.concat(
        [points.loc[keep].copy(), recompute_ldac_points(heldout_only)],
        ignore_index=True,
    )
    updated = updated.sort_values(["family", "method", "difficulty", "level", "seed", "eval_domain"]).reset_index(
        drop=True
    )
    updated.to_csv(points_path, index=False)
    return updated


def aggregate(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, method in METHOD_ORDER:
        row: dict[str, object] = {"Family": family, "Method": method}
        subset = points[(points["family"] == family) & (points["method"] == method)]
        for difficulty in DIFFICULTIES:
            for level in LEVELS:
                scenario_subset = subset[(subset["difficulty"] == difficulty) & (subset["level"] == level)]
                prefix = f"{level}_{difficulty}"
                if scenario_subset.empty:
                    row[f"{prefix}_clean_index_mean"] = np.nan
                    row[f"{prefix}_clean_index_std"] = np.nan
                    row[f"{prefix}_corrupted_index_mean"] = np.nan
                    row[f"{prefix}_corrupted_index_std"] = np.nan
                    row[f"{prefix}_n"] = 0
                else:
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
        std_value = 0.0 if not np.isfinite(std) else std
        text = f"{mean:.2f} $\\pm$ {std_value:.2f}"
    if bold and text != "--":
        return f"\\textbf{{{text}}}"
    return text


def csv_fmt(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "--"
    std_value = 0.0 if not np.isfinite(std) else std
    return f"{mean:.2f} ± {std_value:.2f}"


def display_method(method: str) -> str:
    return method


def write_paper_csv(summary: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        out = {"Family": row["Family"], "Method": row["Method"]}
        for difficulty in DIFFICULTIES:
            label_diff = difficulty.title()
            for level in LEVELS:
                label_level = level.replace("level", "L")
                prefix = f"{level}_{difficulty}"
                out[f"{label_level} {label_diff} Clean"] = csv_fmt(
                    row[f"{prefix}_clean_index_mean"], row[f"{prefix}_clean_index_std"]
                )
                out[f"{label_level} {label_diff} Corr."] = csv_fmt(
                    row[f"{prefix}_corrupted_index_mean"], row[f"{prefix}_corrupted_index_std"]
                )
        rows.append(out)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def write_latex_single(summary: pd.DataFrame, out_path: Path, heldout_only: bool) -> None:
    caption_scope = "Held-out scenario-level" if heldout_only else "Scenario-level"
    scope_phrase = "over held-out evaluations only" if heldout_only else "over completed seed/domain evaluations"
    label = "tab:heldout_clean_corrupted_9scenario" if heldout_only else "tab:clean_corrupted_9scenario_single"
    caption = (
        f"{caption_scope} clean-map and corrupted-map performance, grouped by difficulty. "
        f"Values are common-reference performance index (mean $\\pm$ std) {scope_phrase}; higher is better. "
        "The common reference is the matching PPO clean nominal cost for the same scenario, seed, and evaluation domain. "
        "Corrupted denotes final recovered corrupted-map performance for learning methods and static corrupted-map "
        "performance for non-learning planners. Bold indicates the best value within each scenario/condition column. "
        "LDAC-SAC uses five completed seeds for all Easy, Medium, and Hard scenarios."
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
    for difficulty_index, difficulty in enumerate(DIFFICULTIES):
        if difficulty_index > 0:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{difficulty.title()}}}}} \\\\")
        best: dict[tuple[str, str], float] = {}
        for level in LEVELS:
            prefix = f"{level}_{difficulty}"
            best[(level, "clean")] = float(summary[f"{prefix}_clean_index_mean"].max())
            best[(level, "corrupted")] = float(summary[f"{prefix}_corrupted_index_mean"].max())
        last_family = None
        for _, row in summary.iterrows():
            family = str(row["Family"])
            method = str(row["Method"])
            if last_family == "Non-learning" and family == "Learning":
                lines.append("\\addlinespace[0.2em]")
            cells = [family, display_method(method)]
            for level in LEVELS:
                prefix = f"{level}_{difficulty}"
                clean_mean = float(row[f"{prefix}_clean_index_mean"])
                clean_std = float(row[f"{prefix}_clean_index_std"])
                corr_mean = float(row[f"{prefix}_corrupted_index_mean"])
                corr_std = float(row[f"{prefix}_corrupted_index_std"])
                cells.append(fmt_value(clean_mean, clean_std, np.isclose(clean_mean, best[(level, "clean")], atol=0.005)))
                cells.append(
                    fmt_value(corr_mean, corr_std, np.isclose(corr_mean, best[(level, "corrupted")], atol=0.005))
                )
            lines.append(" & ".join(cells) + " \\\\")
            last_family = family
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def scenario_label(scenario: str) -> str:
    level, difficulty = scenario_parts(scenario)
    return f"{level.replace('level', 'L')} {difficulty.title()}"


def best_column_values(summary: pd.DataFrame, scenarios: tuple[str, ...]) -> dict[tuple[str, str], float]:
    best: dict[tuple[str, str], float] = {}
    for scenario in scenarios:
        best[(scenario, "clean")] = float(summary[f"{scenario}_clean_index_mean"].max())
        best[(scenario, "corrupted")] = float(summary[f"{scenario}_corrupted_index_mean"].max())
    return best


def row_cells_for_scenarios(row: pd.Series, scenarios: tuple[str, ...], best: dict[tuple[str, str], float]) -> list[str]:
    cells = [str(row["Family"]), display_method(str(row["Method"]))]
    for scenario in scenarios:
        clean_mean = float(row[f"{scenario}_clean_index_mean"])
        clean_std = float(row[f"{scenario}_clean_index_std"])
        corr_mean = float(row[f"{scenario}_corrupted_index_mean"])
        corr_std = float(row[f"{scenario}_corrupted_index_std"])
        cells.append(fmt_value(clean_mean, clean_std, np.isclose(clean_mean, best[(scenario, "clean")], atol=0.005)))
        cells.append(fmt_value(corr_mean, corr_std, np.isclose(corr_mean, best[(scenario, "corrupted")], atol=0.005)))
    return cells


def write_latex_flat(summary: pd.DataFrame, out_path: Path) -> None:
    best = best_column_values(summary, SCENARIOS)
    column_spec = "ll" + "cc" * len(SCENARIOS)
    header = ["Family & Method"]
    cmidrules = []
    subheader = [" & "]
    for idx, scenario in enumerate(SCENARIOS):
        start = 3 + idx * 2
        end = start + 1
        header.append(f"\\multicolumn{{2}}{{c}}{{{scenario_label(scenario)}}}")
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
        subheader.append("Clean & Corr.")
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Scenario-level clean-map and corrupted-map performance. Values are common-reference performance index (mean $\\pm$ std) over completed seed/domain evaluations; higher is better. The common reference is the matching PPO clean nominal cost for each scenario, seed, and evaluation domain. Corrupted denotes final recovered corrupted-map performance for learning methods and static corrupted-map performance for non-learning planners. LDAC-SAC uses five completed seeds for all scenarios.}",
        "\\label{tab:clean_corrupted_9scenario}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(header) + " \\\\",
        " ".join(cmidrules),
        " & ".join(subheader) + " \\\\",
        "\\midrule",
    ]
    last_family = None
    for _, row in summary.iterrows():
        family = str(row["Family"])
        if last_family == "Non-learning" and family == "Learning":
            lines.append("\\midrule")
        lines.append(" & ".join(row_cells_for_scenarios(row, SCENARIOS, best)) + " \\\\")
        last_family = family
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_three_panel(summary: pd.DataFrame, out_path: Path) -> None:
    best = best_column_values(summary, SCENARIOS)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Scenario-level clean-map and corrupted-map performance, split by difficulty. Values are common-reference performance index (mean $\\pm$ std) over completed seed/domain evaluations; higher is better. The common reference is the matching PPO clean nominal cost for each scenario, seed, and evaluation domain. Corrupted denotes final recovered corrupted-map performance for learning methods and static corrupted-map performance for non-learning planners. LDAC-SAC uses five completed seeds for all scenarios.}",
        "\\label{tab:clean_corrupted_9scenario_by_difficulty}",
    ]
    for difficulty_index, difficulty in enumerate(DIFFICULTIES):
        scenarios = tuple(f"{level}_{difficulty}" for level in LEVELS)
        lines.extend(
            [
                f"\\textbf{{{difficulty.title()}}}\\\\[-0.25em]",
                "\\resizebox{\\textwidth}{!}{%",
                "\\begin{tabular}{llcccccc}",
                "\\toprule",
                "Family & Method & "
                + " & ".join(f"\\multicolumn{{2}}{{c}}{{{scenario_label(scenario)}}}" for scenario in scenarios)
                + " \\\\",
                "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
                " & & Clean & Corr. & Clean & Corr. & Clean & Corr. \\\\",
                "\\midrule",
            ]
        )
        last_family = None
        for _, row in summary.iterrows():
            family = str(row["Family"])
            if last_family == "Non-learning" and family == "Learning":
                lines.append("\\midrule")
            lines.append(" & ".join(row_cells_for_scenarios(row, scenarios, best)) + " \\\\")
            last_family = family
        lines.extend(["\\bottomrule", "\\end{tabular}%", "}"])
        if difficulty_index < len(DIFFICULTIES) - 1:
            lines.append("\\vspace{0.55em}")
    lines.append("\\end{table*}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def difficulty_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        out: dict[str, object] = {"Family": row["Family"], "Method": row["Method"]}
        for difficulty in DIFFICULTIES:
            clean_values = []
            corr_values = []
            for level in LEVELS:
                prefix = f"{level}_{difficulty}"
                clean_values.append(float(row[f"{prefix}_clean_index_mean"]))
                corr_values.append(float(row[f"{prefix}_corrupted_index_mean"]))
            clean_series = pd.Series(clean_values, dtype="float64").dropna()
            corr_series = pd.Series(corr_values, dtype="float64").dropna()
            out[f"{difficulty}_clean_mean"] = float(clean_series.mean())
            out[f"{difficulty}_clean_std"] = float(clean_series.std(ddof=1))
            out[f"{difficulty}_corrupted_mean"] = float(corr_series.mean())
            out[f"{difficulty}_corrupted_std"] = float(corr_series.std(ddof=1))
        out["Scenarios"] = 9
        rows.append(out)
    return pd.DataFrame(rows)


def write_difficulty_latex(diff: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Clean-map and corrupted-map performance by difficulty. Values are common-reference performance index (mean $\\pm$ std) averaged across levels within each difficulty; higher is better. The common reference is the scenario/seed/domain PPO clean nominal cost, so Clean and Corrupted columns are directly comparable. Corrupted denotes final recovered corrupted-map performance for learning methods and static corrupted-map performance for non-learning planners. LDAC-SAC uses five completed seeds for all scenarios.}",
        "\\label{tab:clean_corrupted_by_difficulty}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccccc}",
        "\\toprule",
        "Family & Method & \\multicolumn{2}{c}{Easy} & \\multicolumn{2}{c}{Medium} & \\multicolumn{2}{c}{Hard} & Scenarios \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
        " & & Clean & Corrupted & Clean & Corrupted & Clean & Corrupted & \\\\",
        "\\midrule",
    ]
    last_family = None
    for _, row in diff.iterrows():
        family = str(row["Family"])
        if last_family == "Non-learning" and family == "Learning":
            lines.append("\\midrule")
        cells = [family, display_method(str(row["Method"]))]
        for difficulty in DIFFICULTIES:
            clean = fmt_value(float(row[f"{difficulty}_clean_mean"]), float(row[f"{difficulty}_clean_std"]))
            corr = fmt_value(float(row[f"{difficulty}_corrupted_mean"]), float(row[f"{difficulty}_corrupted_std"]))
            cells.extend([clean, corr])
        cells.append(str(int(row["Scenarios"])))
        lines.append(" & ".join(cells) + " \\\\")
        last_family = family
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_clean_map_summary(diff: pd.DataFrame, out_tex: Path, out_csv: Path) -> None:
    rows = []
    for _, row in diff.iterrows():
        out = {"Family": row["Family"], "Method": row["Method"], "Scenarios": int(row["Scenarios"])}
        for difficulty in DIFFICULTIES:
            out[difficulty.title()] = csv_fmt(float(row[f"{difficulty}_clean_mean"]), float(row[f"{difficulty}_clean_std"]))
        rows.append(out)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Clean-map performance of non-learning planners and learning policies. Values are clean-map performance index (mean $\\pm$ std) averaged across levels within each difficulty; higher is better. The index is computed as $100 \\times C_{\\mathrm{ref}} / C_{\\mathrm{clean}}$, where $C_{\\mathrm{ref}}$ is the scenario/seed/domain clean nominal PPO reference cost used by the protocol. Thus PPO is the normalization anchor; SAC and LDAC-SAC are evaluated against the same clean tasks and references. LDAC-SAC uses five completed seeds for all scenarios.}",
        "\\label{tab:clean_map_nonlearning_learning}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Family & Method & Easy & Medium & Hard & Scenarios \\\\",
        "\\midrule",
    ]
    last_family = None
    for _, row in diff.iterrows():
        family = str(row["Family"])
        if last_family == "Non-learning" and family == "Learning":
            lines.append("\\midrule")
        cells = [family, display_method(str(row["Method"]))]
        for difficulty in DIFFICULTIES:
            value = fmt_value(float(row[f"{difficulty}_clean_mean"]), float(row[f"{difficulty}_clean_std"]))
            cells.append(value)
        cells.append(str(int(row["Scenarios"])))
        lines.append(" & ".join(cells) + " \\\\")
        last_family = family
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def refresh_set(prefix: str, heldout_only: bool) -> None:
    points_path = TABLE_DIR / f"{prefix}_points.csv"
    summary_path = TABLE_DIR / f"{prefix}_summary.csv"
    paper_csv_path = TABLE_DIR / f"{prefix}_paper_table.csv"
    latex_path = TABLE_DIR / f"{prefix}_single_table_summary.tex"
    points = update_points(points_path, heldout_only)
    summary = aggregate(points)
    summary.to_csv(summary_path, index=False)
    write_paper_csv(summary, paper_csv_path)
    write_latex_single(summary, latex_path, heldout_only)
    if not heldout_only:
        write_latex_flat(summary, TABLE_DIR / "clean_corrupted_9scenario_summary.tex")
        write_latex_three_panel(summary, TABLE_DIR / "clean_corrupted_9scenario_three_panel_summary.tex")
    print(f"Updated {points_path}")
    print(f"Updated {summary_path}")
    print(f"Updated {paper_csv_path}")
    print(f"Updated {latex_path}")


def main() -> None:
    refresh_set("heldout_clean_corrupted_9scenario", heldout_only=True)
    refresh_set("clean_corrupted_9scenario", heldout_only=False)
    pooled_summary = pd.read_csv(TABLE_DIR / "clean_corrupted_9scenario_summary.csv")
    diff = difficulty_summary(pooled_summary)
    diff.to_csv(TABLE_DIR / "clean_corrupted_by_difficulty_summary.csv", index=False)
    write_difficulty_latex(diff, TABLE_DIR / "clean_corrupted_by_difficulty_summary.tex")
    write_clean_map_summary(
        diff,
        TABLE_DIR / "clean_map_nonlearning_learning_summary.tex",
        TABLE_DIR / "clean_map_nonlearning_learning_summary.csv",
    )
    print(f"Updated {TABLE_DIR / 'clean_corrupted_by_difficulty_summary.csv'}")
    print(f"Updated {TABLE_DIR / 'clean_corrupted_by_difficulty_summary.tex'}")
    print(f"Updated {TABLE_DIR / 'clean_map_nonlearning_learning_summary.csv'}")
    print(f"Updated {TABLE_DIR / 'clean_map_nonlearning_learning_summary.tex'}")


if __name__ == "__main__":
    main()
