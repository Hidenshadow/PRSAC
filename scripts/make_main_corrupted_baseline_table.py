#!/usr/bin/env python
"""Create the main-text corrupted-map baseline table."""

from __future__ import annotations

from pathlib import Path

import math
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "runs" / "baseline_tables"
SUMMARY_CSV = TABLE_DIR / "heldout_final_all_baselines_9scenario_summary.csv"

OUTPUT_CSV = TABLE_DIR / "heldout_final_corrupted_main_baselines_table.csv"
OUTPUT_TEX = TABLE_DIR / "heldout_final_corrupted_main_baselines_table.tex"

DIFFICULTIES = ("easy", "medium", "hard")
LEVELS = ("level1", "level2", "level3")

METHOD_ORDER = (
    ("Non-learning", "Minimax"),
    ("Non-learning", "Guard"),
    ("Non-learning", "Risk-Inflated A*"),
    ("Non-learning", "Belief-CVaR A*"),
    ("Learning", "PPO"),
    ("Learning", "SAC"),
    ("Learning", "CDR-SAC"),
    ("Learning", "Stackelberg-SAC"),
    ("Learning", "VALT-SAC"),
    ("Learning", "LDAC-SAC"),
)


def is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt_mean_std(mean: float, std: float) -> str:
    if not is_finite(mean):
        return "--"
    return f"{float(mean):.2f} $\\pm$ {float(std):.2f}"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )


def scenario_key(level: str, difficulty: str) -> str:
    return f"{level}_{difficulty}"


def scenario_label(level: str) -> str:
    return level.replace("level", "L")


def load_included_summary() -> pd.DataFrame:
    summary = pd.read_csv(SUMMARY_CSV)
    rows = []
    for family, method in METHOD_ORDER:
        row = summary[(summary["Family"] == family) & (summary["Method"] == method)]
        if row.empty:
            raise ValueError(f"Missing method in summary: {family} / {method}")
        rows.append(row.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def winning_methods(df: pd.DataFrame) -> dict[str, set[str]]:
    winners: dict[str, set[str]] = {}
    for difficulty in DIFFICULTIES:
        for level in LEVELS:
            key = scenario_key(level, difficulty)
            mean_col = f"{key}_corrupted_index_mean"
            values = []
            for _, row in df.iterrows():
                value = row.get(mean_col)
                if is_finite(value):
                    values.append((float(value), str(row["Method"])))
            if not values:
                winners[key] = set()
                continue
            best = max(value for value, _ in values)
            winners[key] = {method for value, method in values if abs(value - best) < 1e-9}
    return winners


def build_csv_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        out = {"Family": row["Family"], "Method": row["Method"]}
        for difficulty in DIFFICULTIES:
            for level in LEVELS:
                key = scenario_key(level, difficulty)
                out[key] = fmt_mean_std(
                    row.get(f"{key}_corrupted_index_mean"),
                    row.get(f"{key}_corrupted_index_std"),
                )
                out[f"{key}_n"] = int(row.get(f"{key}_n", 0)) if is_finite(row.get(f"{key}_n", 0)) else 0
        rows.append(out)
    return pd.DataFrame(rows)


def write_latex(df: pd.DataFrame, winners: dict[str, set[str]]) -> None:
    caption = (
        "Held-out corrupted-map performance across nine scenarios. Values are "
        "common-reference performance index, reported as mean $\\pm$ std over "
        "held-out evaluations; higher is better. The table includes only the "
        "main paper baselines: robust/risk-aware non-learning planners with "
        "clear literature support and all completed learning baselines. CDR-SAC "
        "uses two seeds; Stackelberg-SAC uses five seeds for Easy-L1 and three "
        "seeds otherwise; all other methods use five seeds."
    )

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        f"\\caption{{{caption}}}",
        "\\label{tab:heldout_final_corrupted_main_baselines}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccccccc}",
        "\\toprule",
        "Family & Method & \\multicolumn{3}{c}{Easy} & \\multicolumn{3}{c}{Medium} & \\multicolumn{3}{c}{Hard} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}",
        " & & L1 & L2 & L3 & L1 & L2 & L3 & L1 & L2 & L3 \\\\",
        "\\midrule",
    ]

    previous_family = None
    for _, row in df.iterrows():
        family = str(row["Family"])
        if previous_family is not None and previous_family != family:
            lines.append("\\addlinespace[0.2em]")
        previous_family = family

        cells = [latex_escape(family), latex_escape(str(row["Method"]))]
        for difficulty in DIFFICULTIES:
            for level in LEVELS:
                key = scenario_key(level, difficulty)
                value = str(row[key])
                if str(row["Method"]) in winners[key]:
                    value = f"\\textbf{{{value}}}"
                cells.append(value)
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_included_summary()
    table_df = build_csv_table(df)
    table_df.to_csv(OUTPUT_CSV, index=False)
    write_latex(table_df, winning_methods(df))
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
