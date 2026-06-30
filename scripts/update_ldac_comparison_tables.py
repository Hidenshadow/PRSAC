#!/usr/bin/env python3
"""Rebuild LDAC comparison tables after all five LDAC seeds complete."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "runs" / "baseline_tables"

SCENARIO_ORDER = (
    ("level1", "easy"),
    ("level1", "medium"),
    ("level1", "hard"),
    ("level2", "easy"),
    ("level2", "medium"),
    ("level2", "hard"),
    ("level3", "easy"),
    ("level3", "medium"),
    ("level3", "hard"),
)

DIFFICULTIES = ("easy", "medium", "hard")
LEVELS = ("level1", "level2", "level3")

NONLEARNING_ORDER = (
    "Emergency",
    "Risk-Inflated A*",
    "Val-Best",
    "Minimax",
    "Belief-CVaR A*",
    "Heuristic",
    "Guard",
    "Fixed",
)

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

DISPLAY_LEVEL = {
    "level1": "Level 1",
    "level2": "Level 2",
    "level3": "Level 3",
}

DISPLAY_DIFFICULTY = {
    "easy": "Easy",
    "medium": "Medium",
    "hard": "Hard",
}


def scenario_name(level: str, difficulty: str) -> str:
    return f"{level}_{difficulty}"


def seed_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    raise ValueError(f"could not parse seed from {path}")


def fmt(mean: float, std: float, bold: bool = False) -> str:
    if not math.isfinite(mean):
        text = "--"
    else:
        std_value = 0.0 if not math.isfinite(std) else std
        text = f"{mean:.2f} $\\pm$ {std_value:.2f}"
    return f"\\textbf{{{text}}}" if bold and text != "--" else text


def mean_std(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan, math.nan
    return float(numeric.mean()), float(numeric.std(ddof=1))


def load_ldac_self_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for difficulty, roots in LDAC_ROOTS_BY_DIFFICULTY.items():
        for root in roots:
            for summary_path in sorted(root.glob(f"level*_{difficulty}/seed*/shock_recovery_summary.csv")):
                seed_dir = summary_path.parent
                scenario = seed_dir.parent.name
                level, parsed_difficulty = scenario.split("_", 1)
                seed = seed_from_path(seed_dir)
                frame = pd.read_csv(summary_path)
                for row in frame.itertuples(index=False):
                    clean_cost = float(row.clean_nominal_cost)
                    final_cost = float(row.final_recovery_cost)
                    if not math.isfinite(clean_cost) or not math.isfinite(final_cost) or final_cost <= 0.0:
                        continue
                    rows.append(
                        {
                            "algorithm": "ldac_sac",
                            "scenario": scenario,
                            "level": level,
                            "difficulty": parsed_difficulty,
                            "seed": seed,
                            "eval_domain": str(row.eval_domain),
                            "final_pi": 100.0 * clean_cost / final_cost,
                        }
                    )
    points = pd.DataFrame(rows)
    if points.empty:
        raise RuntimeError("no LDAC self-normalized points found")
    grouped = []
    for (scenario, level, difficulty), group in points.groupby(["scenario", "level", "difficulty"], sort=False):
        mean, std = mean_std(group["final_pi"])
        grouped.append(
            {
                "algorithm": "ldac_sac",
                "scenario": scenario,
                "level": level,
                "difficulty": difficulty,
                "mean_final_pi": mean,
                "std_final_pi": std,
                "num_points": int(group["final_pi"].notna().sum()),
                "num_seeds": int(group["seed"].nunique()),
            }
        )
    points.to_csv(TABLE_DIR / "nonlearning_vs_learning_ldac_points.csv", index=False)
    return pd.DataFrame(grouped)


def load_learning_self_summary() -> pd.DataFrame:
    base = pd.read_csv(TABLE_DIR / "nonlearning_vs_learning_baselines_per_method_summary.csv")
    base = base[base["algorithm"].isin(["ppo", "sac"])].copy()
    base = base[
        ["algorithm", "scenario", "level", "difficulty", "mean_final_pi", "std_final_pi", "num_points", "num_seeds"]
    ]
    ldac = load_ldac_self_summary()
    return pd.concat([base, ldac], ignore_index=True)


def load_common_summary() -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / "clean_corrupted_9scenario_summary.csv")


def nonlearning_scenario_rows(common: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in NONLEARNING_ORDER:
        match = common[(common["Family"] == "Non-learning") & (common["Method"] == method)]
        if match.empty:
            continue
        row = match.iloc[0]
        for level, difficulty in SCENARIO_ORDER:
            scenario = scenario_name(level, difficulty)
            prefix = scenario
            rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "level": level,
                    "difficulty": difficulty,
                    "mean": float(row[f"{prefix}_corrupted_index_mean"]),
                    "std": float(row[f"{prefix}_corrupted_index_std"]),
                }
            )
    return pd.DataFrame(rows)


def best_nonlearning(common: pd.DataFrame) -> pd.DataFrame:
    nl = nonlearning_scenario_rows(common)
    idx = nl.groupby("scenario")["mean"].idxmax()
    return nl.loc[idx].reset_index(drop=True)


def learning_lookup(learning: pd.DataFrame) -> dict[tuple[str, str], object]:
    return {(row.algorithm, row.scenario): row for row in learning.itertuples(index=False)}


def write_compact_table(common: pd.DataFrame, learning: pd.DataFrame) -> pd.DataFrame:
    best_nl = {row.scenario: row for row in best_nonlearning(common).itertuples(index=False)}
    lookup = learning_lookup(learning)
    rows: list[dict[str, object]] = []
    for level, difficulty in SCENARIO_ORDER:
        scenario = scenario_name(level, difficulty)
        nl = best_nl[scenario]
        ppo = lookup[("ppo", scenario)]
        sac = lookup[("sac", scenario)]
        ldac = lookup[("ldac_sac", scenario)]
        winner_mean = max(float(nl.mean), float(ppo.mean_final_pi), float(sac.mean_final_pi), float(ldac.mean_final_pi))
        learning_best = max(float(ppo.mean_final_pi), float(sac.mean_final_pi), float(ldac.mean_final_pi))
        rows.append(
            {
                "level": level,
                "difficulty": difficulty,
                "scenario": scenario,
                "best_nonlearning_method": str(nl.method),
                "best_nonlearning_mean": float(nl.mean),
                "best_nonlearning_std": float(nl.std),
                "ppo_mean": float(ppo.mean_final_pi),
                "ppo_std": float(ppo.std_final_pi),
                "sac_mean": float(sac.mean_final_pi),
                "sac_std": float(sac.std_final_pi),
                "ldac_mean": float(ldac.mean_final_pi),
                "ldac_std": float(ldac.std_final_pi),
                "learning_advantage": learning_best - float(nl.mean),
                "winner_mean": winner_mean,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "nonlearning_vs_learning_ldac_baselines_compact.csv", index=False)
    return frame


def compact_latex(frame: pd.DataFrame) -> str:
    means = {
        "best_nonlearning": (
            float(frame["best_nonlearning_mean"].mean()),
            float(frame["best_nonlearning_mean"].std(ddof=1)),
        ),
        "ppo": (float(frame["ppo_mean"].mean()), float(frame["ppo_mean"].std(ddof=1))),
        "sac": (float(frame["sac_mean"].mean()), float(frame["sac_mean"].std(ddof=1))),
        "ldac": (float(frame["ldac_mean"].mean()), float(frame["ldac_mean"].std(ddof=1))),
    }
    mean_winner = max(value[0] for value in means.values())
    mean_advantage = float(frame["learning_advantage"].mean())

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Comparison of non-learning planner baselines and learning-based recovery baselines. Values are final recovery performance index (mean $\\pm$ std); higher is better. PPO, SAC, LDAC-SAC, and non-learning baselines use the completed five-seed protocol.}",
        "\\label{tab:nonlearning_vs_learning_ldac_baselines}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Level & Difficulty & Best non-learning & PPO recovery & SAC recovery & LDAC recovery & Best learning $-$ best non-learning \\\\",
        "\\midrule",
    ]
    for row in frame.itertuples(index=False):
        winner = float(row.winner_mean)
        nl_text = (
            f"{row.best_nonlearning_method} "
            f"({fmt(float(row.best_nonlearning_mean), float(row.best_nonlearning_std), np.isclose(float(row.best_nonlearning_mean), winner, atol=0.005))})"
        )
        lines.append(
            f"{DISPLAY_LEVEL[row.level]} & {DISPLAY_DIFFICULTY[row.difficulty]} & "
            f"{nl_text} & "
            f"{fmt(float(row.ppo_mean), float(row.ppo_std), np.isclose(float(row.ppo_mean), winner, atol=0.005))} & "
            f"{fmt(float(row.sac_mean), float(row.sac_std), np.isclose(float(row.sac_mean), winner, atol=0.005))} & "
            f"{fmt(float(row.ldac_mean), float(row.ldac_std), np.isclose(float(row.ldac_mean), winner, atol=0.005))} & "
            f"{float(row.learning_advantage):.2f} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            "Mean & All & "
            f"Scenario-best ({fmt(means['best_nonlearning'][0], means['best_nonlearning'][1], np.isclose(means['best_nonlearning'][0], mean_winner, atol=0.005))}) & "
            f"{fmt(means['ppo'][0], means['ppo'][1], np.isclose(means['ppo'][0], mean_winner, atol=0.005))} & "
            f"{fmt(means['sac'][0], means['sac'][1], np.isclose(means['sac'][0], mean_winner, atol=0.005))} & "
            f"{fmt(means['ldac'][0], means['ldac'][1], np.isclose(means['ldac'][0], mean_winner, atol=0.005))} & "
            f"{mean_advantage:.2f} \\\\",
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def difficulty_rows(common: pd.DataFrame, learning: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    nl = nonlearning_scenario_rows(common)
    for method in NONLEARNING_ORDER:
        group = nl[nl["method"] == method]
        if group.empty:
            continue
        out: dict[str, object] = {"family": "Non-learning", "method": method, "scenarios": 9}
        for difficulty in DIFFICULTIES:
            values = group[group["difficulty"] == difficulty]["mean"]
            mean, std = mean_std(values)
            out[f"{difficulty}_mean"] = mean
            out[f"{difficulty}_std"] = std
        rows.append(out)

    for algorithm, method in (("ppo", "PPO"), ("sac", "SAC"), ("ldac_sac", "LDAC")):
        group = learning[learning["algorithm"] == algorithm]
        out = {"family": "Learning", "method": method, "scenarios": 9}
        for difficulty in DIFFICULTIES:
            values = group[group["difficulty"] == difficulty]["mean_final_pi"]
            mean, std = mean_std(values)
            out[f"{difficulty}_mean"] = mean
            out[f"{difficulty}_std"] = std
        rows.append(out)
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLE_DIR / "baseline_by_difficulty_with_ldac.csv", index=False)
    return frame


def difficulty_latex(diff: pd.DataFrame) -> str:
    best = {difficulty: float(diff[f"{difficulty}_mean"].max()) for difficulty in DIFFICULTIES}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Difficulty-wise baseline performance. Values are scenario-level final recovery performance index averaged across levels within each difficulty; higher is better. This view preserves the Easy/Medium/Hard structure instead of averaging all nine scenarios into one number.}",
        "\\label{tab:baseline_by_difficulty_with_ldac}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Family & Method & Easy & Medium & Hard & Scenarios \\\\",
        "\\midrule",
    ]
    last_family = None
    for row in diff.itertuples(index=False):
        if last_family == "Non-learning" and row.family == "Learning":
            lines.append("\\midrule")
        cells = [str(row.family), str(row.method)]
        for difficulty in DIFFICULTIES:
            mean = float(getattr(row, f"{difficulty}_mean"))
            std = float(getattr(row, f"{difficulty}_std"))
            cells.append(fmt(mean, std, np.isclose(mean, best[difficulty], atol=0.005)))
        cells.append(str(int(row.scenarios)))
        lines.append(" & ".join(cells) + " \\\\")
        last_family = row.family
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    return "\n".join(lines)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    common = load_common_summary()
    learning = load_learning_self_summary()
    compact = write_compact_table(common, learning)
    diff = difficulty_rows(common, learning)
    latex = compact_latex(compact) + "\n" + difficulty_latex(diff)
    out_path = TABLE_DIR / "nonlearning_vs_learning_ldac_baselines.tex"
    out_path.write_text(latex, encoding="utf-8")
    print(f"Updated {out_path}")
    print(f"Updated {TABLE_DIR / 'nonlearning_vs_learning_ldac_baselines_compact.csv'}")
    print(f"Updated {TABLE_DIR / 'baseline_by_difficulty_with_ldac.csv'}")


if __name__ == "__main__":
    main()
