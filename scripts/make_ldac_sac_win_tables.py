#!/usr/bin/env python
"""Create focused main-text tables for LDAC-SAC winning conditions."""

from __future__ import annotations

from pathlib import Path

import math
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "runs" / "baseline_tables"

DIFFICULTIES = ("easy", "medium", "hard")
LEVELS = ("level1", "level2", "level3")
LDAC_METHOD = "LDAC-SAC"

SOURCES = (
    {
        "stem": "heldout_final_ldac_sac_wins_corrupted",
        "input": TABLE_DIR / "heldout_final_all_baselines_9scenario_summary.csv",
        "caption": (
            "Held-out corrupted-map conditions where LDAC-SAC achieves the best "
            "common-reference performance index among all evaluated baselines. "
            "Values are mean $\\pm$ std over seeds; higher is better. This focused "
            "main-text table reports only LDAC-SAC winning conditions; the full "
            "all-baselines table is reported separately."
        ),
        "label": "tab:ldac_sac_wins_final_corrupted",
    },
    {
        "stem": "heldout_trainselected_best_ldac_sac_wins_corrupted",
        "input": TABLE_DIR / "heldout_trainselected_best_clean_corrupted_9scenario_summary.csv",
        "caption": (
            "Held-out corrupted-map conditions where LDAC-SAC achieves the best "
            "train-selected recovery performance among all evaluated baselines. "
            "Values are mean $\\pm$ std over seeds; higher is better. Checkpoints "
            "are selected using training-domain recovery performance. This focused "
            "main-text table reports only LDAC-SAC winning conditions; the full "
            "all-baselines table is reported separately."
        ),
        "label": "tab:ldac_sac_wins_trainselected_corrupted",
    },
)


def is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt_mean_std(mean: float, std: float) -> str:
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )


def display_difficulty(name: str) -> str:
    return name.capitalize()


def display_level(name: str) -> str:
    return name.replace("level", "Level ")


def collect_wins(summary_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_csv)
    rows: list[dict[str, object]] = []

    for difficulty in DIFFICULTIES:
        for level in LEVELS:
            prefix = f"{level}_{difficulty}_corrupted_index"
            mean_col = f"{prefix}_mean"
            std_col = f"{prefix}_std"
            n_col = f"{level}_{difficulty}_n"

            candidates = []
            for _, row in df.iterrows():
                mean = row.get(mean_col)
                if not is_finite(mean):
                    continue
                candidates.append(
                    {
                        "family": row["Family"],
                        "method": row["Method"],
                        "mean": float(mean),
                        "std": float(row.get(std_col, 0.0)) if is_finite(row.get(std_col, 0.0)) else 0.0,
                        "n": int(row.get(n_col, 0)) if is_finite(row.get(n_col, 0)) else 0,
                    }
                )

            if not candidates:
                continue

            candidates.sort(key=lambda item: item["mean"], reverse=True)
            ldac = next((item for item in candidates if item["method"] == LDAC_METHOD), None)
            if ldac is None:
                continue

            best_mean = candidates[0]["mean"]
            if ldac["mean"] + 1e-9 < best_mean:
                continue

            competitor = next(item for item in candidates if item["method"] != LDAC_METHOD)
            rows.append(
                {
                    "Difficulty": display_difficulty(difficulty),
                    "Level": display_level(level),
                    "LDAC-SAC": fmt_mean_std(ldac["mean"], ldac["std"]),
                    "Best competing baseline": competitor["method"],
                    "Competing value": fmt_mean_std(competitor["mean"], competitor["std"]),
                    "Margin": f"+{ldac['mean'] - competitor['mean']:.2f}",
                    "LDAC seeds": ldac["n"],
                    "Competitor seeds": competitor["n"],
                    "_sort_difficulty": DIFFICULTIES.index(difficulty),
                    "_sort_level": LEVELS.index(level),
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["_sort_difficulty", "_sort_level"]).drop(
            columns=["_sort_difficulty", "_sort_level"]
        )
    return out


def write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llcccccc}",
        "\\toprule",
        "Difficulty & Level & LDAC-SAC & Best competing baseline & Competing value & Margin & LDAC seeds & Comp. seeds \\\\",
        "\\midrule",
    ]

    if df.empty:
        lines.append("\\multicolumn{8}{c}{No LDAC-SAC winning conditions found.} \\\\")
    else:
        previous_difficulty = None
        for _, row in df.iterrows():
            difficulty = str(row["Difficulty"])
            if previous_difficulty is not None and previous_difficulty != difficulty:
                lines.append("\\addlinespace[0.2em]")
            previous_difficulty = difficulty
            lines.append(
                " & ".join(
                    [
                        latex_escape(difficulty),
                        latex_escape(str(row["Level"])),
                        f"\\textbf{{{row['LDAC-SAC']}}}",
                        latex_escape(str(row["Best competing baseline"])),
                        str(row["Competing value"]),
                        f"\\textbf{{{row['Margin']}}}",
                        str(row["LDAC seeds"]),
                        str(row["Competitor seeds"]),
                    ]
                )
                + " \\\\"
            )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        df = collect_wins(Path(source["input"]))
        csv_path = TABLE_DIR / f"{source['stem']}.csv"
        tex_path = TABLE_DIR / f"{source['stem']}.tex"
        df.to_csv(csv_path, index=False)
        write_latex_table(df, tex_path, str(source["caption"]), str(source["label"]))
        print(f"Wrote {csv_path} ({len(df)} rows)")
        print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
