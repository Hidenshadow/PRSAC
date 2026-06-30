#!/usr/bin/env python
"""Create the main clean/corrupted table from train-selected points."""

from __future__ import annotations

from pathlib import Path

import math
import pandas as pd

import update_ldac_baseline_tables as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "runs" / "baseline_tables"
SOURCE_POINTS = TABLE_DIR / "heldout_trainselected_corrupted_main_baselines_points.csv"

OUTPUT_CSV = TABLE_DIR / "heldout_trainselected_clean_corrupted_main_baselines_table.csv"
OUTPUT_TEX = TABLE_DIR / "heldout_trainselected_clean_corrupted_main_baselines_table.tex"

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


def fmt_value(mean: float, std: float, bold: bool = False) -> str:
    if not is_finite(mean):
        text = "--"
    else:
        std = 0.0 if not is_finite(std) else float(std)
        text = f"{float(mean):.2f} $\\pm$ {std:.2f}"
    return f"\\textbf{{{text}}}" if bold and text != "--" else text


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )


def aggregate(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, method in METHOD_ORDER:
        row: dict[str, object] = {"Family": family, "Method": method}
        subset = points[(points["family"] == family) & (points["method"] == method)]
        for difficulty in base.DIFFICULTIES:
            for level in base.LEVELS:
                scenario = subset[(subset["difficulty"] == difficulty) & (subset["level"] == level)]
                prefix = f"{level}_{difficulty}"
                if scenario.empty:
                    for metric in ("clean", "corrupted"):
                        row[f"{prefix}_{metric}_index_mean"] = math.nan
                        row[f"{prefix}_{metric}_index_std"] = math.nan
                    row[f"{prefix}_n"] = 0
                    continue
                row[f"{prefix}_clean_index_mean"] = float(scenario["clean_index"].mean())
                row[f"{prefix}_clean_index_std"] = float(scenario["clean_index"].std(ddof=1))
                row[f"{prefix}_corrupted_index_mean"] = float(scenario["corrupted_index"].mean())
                row[f"{prefix}_corrupted_index_std"] = float(scenario["corrupted_index"].std(ddof=1))
                row[f"{prefix}_n"] = int(len(scenario))
        rows.append(row)
    return pd.DataFrame(rows)


def best_values(summary: pd.DataFrame) -> dict[tuple[str, str, str], float]:
    best: dict[tuple[str, str, str], float] = {}
    for difficulty in base.DIFFICULTIES:
        for level in base.LEVELS:
            prefix = f"{level}_{difficulty}"
            for metric in ("clean", "corrupted"):
                col = f"{prefix}_{metric}_index_mean"
                values = [float(v) for v in summary[col] if is_finite(v)]
                best[(difficulty, level, metric)] = max(values) if values else math.nan
    return best


def write_csv(summary: pd.DataFrame) -> None:
    rows = []
    for _, row in summary.iterrows():
        out = {"Family": row["Family"], "Method": row["Method"]}
        for difficulty in base.DIFFICULTIES:
            for level in base.LEVELS:
                prefix = f"{level}_{difficulty}"
                out[f"{prefix}_clean"] = fmt_value(
                    row[f"{prefix}_clean_index_mean"],
                    row[f"{prefix}_clean_index_std"],
                )
                out[f"{prefix}_corrupted"] = fmt_value(
                    row[f"{prefix}_corrupted_index_mean"],
                    row[f"{prefix}_corrupted_index_std"],
                )
                out[f"{prefix}_n"] = int(row[f"{prefix}_n"])
        rows.append(out)
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)


def write_latex(summary: pd.DataFrame) -> None:
    best = best_values(summary)
    caption = (
        "Held-out clean-map and corrupted-map performance across nine scenarios. "
        "Values are common-reference performance index, reported as mean $\\pm$ std "
        "over held-out evaluations; higher is better. For learning methods, Corr. "
        "uses the recovery checkpoint selected by training-domain corrupted-map "
        "performance and then evaluated on the held-out corrupted map. Non-learning "
        "rows use static clean-map and corrupted-map planner performance. Bold "
        "indicates the best value within each scenario/condition column."
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        f"\\caption{{{caption}}}",
        "\\label{tab:heldout_trainselected_clean_corrupted_main_baselines}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Family & Method & \\multicolumn{2}{c}{L1} & \\multicolumn{2}{c}{L2} & \\multicolumn{2}{c}{L3} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
        " & & Clean & Corr. & Clean & Corr. & Clean & Corr. \\\\",
        "\\hline",
    ]

    for difficulty_index, difficulty in enumerate(base.DIFFICULTIES):
        if difficulty_index > 0:
            lines.append("\\hline")
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textbf{{{difficulty.title()}}}}} \\\\")
        previous_family = None
        for _, row in summary.iterrows():
            family = str(row["Family"])
            if previous_family == "Non-learning" and family == "Learning":
                lines.append("\\addlinespace[0.2em]")
            previous_family = family

            cells = [latex_escape(family), latex_escape(str(row["Method"]))]
            for level in base.LEVELS:
                prefix = f"{level}_{difficulty}"
                for metric in ("clean", "corrupted"):
                    mean = row[f"{prefix}_{metric}_index_mean"]
                    std = row[f"{prefix}_{metric}_index_std"]
                    winner = (
                        is_finite(mean)
                        and abs(float(mean) - best[(difficulty, level, metric)]) < 1e-9
                    )
                    cells.append(fmt_value(mean, std, bold=winner))
            lines.append(" & ".join(cells) + " \\\\")

    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not SOURCE_POINTS.exists():
        raise FileNotFoundError(
            f"{SOURCE_POINTS} does not exist. Run make_trainselected_corrupted_main_baseline_table.py first."
        )
    points = pd.read_csv(SOURCE_POINTS)
    summary = aggregate(points)
    write_csv(summary)
    write_latex(summary)
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
