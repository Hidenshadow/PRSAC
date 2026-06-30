#!/usr/bin/env python
"""Build LaTeX tables comparing non-learning and learning baselines."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NONLEARNING_SUMMARY = (
    PROJECT_ROOT
    / "runs"
    / "nonlearning_planner_baselines"
    / "5seeds_incremental_full_20260619_043515"
    / "planner_baseline_domain_summary.csv"
)
DEFAULT_RL_ROOT = PROJECT_ROOT / "runs" / "rl_baselines"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "baseline_tables"

SCENARIO_ORDER = [
    ("level1", "easy"),
    ("level1", "medium"),
    ("level1", "hard"),
    ("level2", "easy"),
    ("level2", "medium"),
    ("level2", "hard"),
    ("level3", "easy"),
    ("level3", "medium"),
    ("level3", "hard"),
]

FOCUS_NONLEARNING_METHODS = [
    "fixed",
    "heuristic",
    "rover_guard",
    "emergency_uncertainty_rule",
    "validation_best_static",
    "model_minimax",
]

DISPLAY_METHOD = {
    "fixed": "Fixed",
    "heuristic": "Heuristic",
    "rover_guard": "Guard",
    "emergency_uncertainty_rule": "Emergency",
    "validation_best_static": "Val-Best",
    "model_minimax": "Minimax",
    "ppo": "PPO",
    "sac": "SAC",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nonlearning-summary", type=Path, default=DEFAULT_NONLEARNING_SUMMARY)
    parser.add_argument("--rl-root", type=Path, default=DEFAULT_RL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", type=str, default="nonlearning_vs_learning_baselines")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def scenario_name(level: str, difficulty: str) -> str:
    return f"{level}_{difficulty}"


def scenario_sort_key(scenario: str) -> int:
    parts = scenario.split("_", 1)
    key = (parts[0], parts[1]) if len(parts) == 2 else ("", "")
    try:
        return SCENARIO_ORDER.index(key)
    except ValueError:
        return 999


def mean_std(values: pd.Series) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return math.nan, math.nan
    return float(numeric.mean()), float(numeric.std(ddof=1))


def format_mean_std(mean: float, std: float, bold: bool = False) -> str:
    if not math.isfinite(mean):
        text = "--"
    elif math.isfinite(std):
        text = f"{mean:.2f} $\\pm$ {std:.2f}"
    else:
        text = f"{mean:.2f}"
    if bold and text != "--":
        return f"\\textbf{{{text}}}"
    return text


def load_nonlearning(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"level", "difficulty", "method", "seed", "eval_domain", "reference_performance_index"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing non-learning columns {sorted(missing)} in {path}")
    frame = frame[frame["method"].isin(FOCUS_NONLEARNING_METHODS)].copy()
    frame["algorithm"] = frame["method"]
    frame["scenario"] = frame["level"].astype(str) + "_" + frame["difficulty"].astype(str)
    frame["final_pi"] = pd.to_numeric(frame["reference_performance_index"], errors="coerce")
    return frame[["algorithm", "scenario", "level", "difficulty", "seed", "eval_domain", "final_pi"]]


def load_rl_algorithm(rl_root: Path, algorithm: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for level, difficulty in SCENARIO_ORDER:
        scenario = scenario_name(level, difficulty)
        run_root = rl_root / algorithm / f"{scenario}_shock_recovery_5seeds"
        for curve_path in sorted(run_root.glob("seed*/shock_recovery_curve.csv")):
            seed_text = curve_path.parent.name.replace("seed", "")
            try:
                seed = int(seed_text)
            except ValueError:
                seed = -1
            curve = pd.read_csv(curve_path)
            required = {"eval_domain", "phase", "attack_type", "recovery_step", "mean_attacked_scalar_cost"}
            if not required.issubset(curve.columns):
                continue
            clean = curve[(curve["phase"] == "shock") & (curve["attack_type"] == "none")][
                ["eval_domain", "mean_attacked_scalar_cost"]
            ].rename(columns={"mean_attacked_scalar_cost": "clean_cost"})
            recovery = curve[
                (curve["phase"] == "recovery")
                & (curve["attack_type"] == "environment")
            ][["eval_domain", "recovery_step", "mean_attacked_scalar_cost"]]
            if clean.empty or recovery.empty:
                continue
            final_step = pd.to_numeric(recovery["recovery_step"], errors="coerce").max()
            final = recovery[pd.to_numeric(recovery["recovery_step"], errors="coerce") == final_step]
            merged = final.merge(clean, on="eval_domain", how="inner")
            for row in merged.itertuples(index=False):
                clean_cost = float(row.clean_cost)
                attacked_cost = float(row.mean_attacked_scalar_cost)
                if not math.isfinite(clean_cost) or not math.isfinite(attacked_cost) or attacked_cost <= 0.0:
                    continue
                rows.append(
                    {
                        "algorithm": algorithm,
                        "scenario": scenario,
                        "level": level,
                        "difficulty": difficulty,
                        "seed": seed,
                        "eval_domain": str(row.eval_domain),
                        "final_pi": 100.0 * clean_cost / attacked_cost,
                    }
                )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (algorithm, scenario, level, difficulty), group in frame.groupby(
        ["algorithm", "scenario", "level", "difficulty"],
        sort=False,
    ):
        mean, std = mean_std(group["final_pi"])
        rows.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "level": level,
                "difficulty": difficulty,
                "mean_final_pi": mean,
                "std_final_pi": std,
                "num_points": int(group["final_pi"].notna().sum()),
                "num_seeds": int(group["seed"].nunique()),
                "num_domains": int(group[["seed", "eval_domain"]].drop_duplicates().shape[0]),
            }
        )
    summary = pd.DataFrame(rows)
    summary["scenario_order"] = summary["scenario"].map(scenario_sort_key)
    return summary.sort_values(["scenario_order", "algorithm"]).drop(columns=["scenario_order"])


def best_nonlearning_by_scenario(summary: pd.DataFrame) -> pd.DataFrame:
    nl = summary[summary["algorithm"].isin(FOCUS_NONLEARNING_METHODS)].copy()
    idx = nl.groupby("scenario")["mean_final_pi"].idxmax()
    return nl.loc[idx].copy()


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def compact_table(summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    best_nl = best_nonlearning_by_scenario(summary)
    lookup = {(row.algorithm, row.scenario): row for row in summary.itertuples(index=False)}
    best_lookup = {row.scenario: row for row in best_nl.itertuples(index=False)}

    rows: list[dict[str, object]] = []
    for level, difficulty in SCENARIO_ORDER:
        scenario = scenario_name(level, difficulty)
        nl_row = best_lookup[scenario]
        ppo = lookup[("ppo", scenario)]
        sac = lookup[("sac", scenario)]
        learning_best_mean = max(float(ppo.mean_final_pi), float(sac.mean_final_pi))
        winner_mean = max(float(nl_row.mean_final_pi), float(ppo.mean_final_pi), float(sac.mean_final_pi))
        rows.append(
            {
                "level": level,
                "difficulty": difficulty,
                "scenario": scenario,
                "best_nonlearning_method": nl_row.algorithm,
                "best_nonlearning_mean": float(nl_row.mean_final_pi),
                "best_nonlearning_std": float(nl_row.std_final_pi),
                "ppo_mean": float(ppo.mean_final_pi),
                "ppo_std": float(ppo.std_final_pi),
                "sac_mean": float(sac.mean_final_pi),
                "sac_std": float(sac.std_final_pi),
                "learning_advantage": learning_best_mean - float(nl_row.mean_final_pi),
                "winner_mean": winner_mean,
            }
        )
    table_frame = pd.DataFrame(rows)

    mean_row = {
        "level": "mean",
        "difficulty": "all",
        "scenario": "mean_all",
        "best_nonlearning_method": "scenario_best",
        "best_nonlearning_mean": float(table_frame["best_nonlearning_mean"].mean()),
        "best_nonlearning_std": float(table_frame["best_nonlearning_mean"].std(ddof=1)),
        "ppo_mean": float(table_frame["ppo_mean"].mean()),
        "ppo_std": float(table_frame["ppo_mean"].std(ddof=1)),
        "sac_mean": float(table_frame["sac_mean"].mean()),
        "sac_std": float(table_frame["sac_mean"].std(ddof=1)),
        "learning_advantage": float(table_frame["learning_advantage"].mean()),
        "winner_mean": max(
            float(table_frame["best_nonlearning_mean"].mean()),
            float(table_frame["ppo_mean"].mean()),
            float(table_frame["sac_mean"].mean()),
        ),
    }

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Comparison of non-learning planner baselines and learning-based recovery baselines. Values are final recovery performance index (mean $\\pm$ std over five seeds and two evaluation domains); higher is better.}",
        "\\label{tab:nonlearning_vs_learning_baselines}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Level & Difficulty & Best non-learning & PPO recovery & SAC recovery & Best learning $-$ best non-learning \\\\",
        "\\midrule",
    ]
    for row in rows:
        best_mean = row["winner_mean"]
        nl_text = (
            f"{DISPLAY_METHOD[row['best_nonlearning_method']]} "
            f"({format_mean_std(row['best_nonlearning_mean'], row['best_nonlearning_std'], row['best_nonlearning_mean'] == best_mean)})"
        )
        ppo_text = format_mean_std(row["ppo_mean"], row["ppo_std"], row["ppo_mean"] == best_mean)
        sac_text = format_mean_std(row["sac_mean"], row["sac_std"], row["sac_mean"] == best_mean)
        lines.append(
            f"{DISPLAY_LEVEL[row['level']]} & {DISPLAY_DIFFICULTY[row['difficulty']]} & "
            f"{nl_text} & {ppo_text} & {sac_text} & {row['learning_advantage']:.2f} \\\\"
        )
    lines.extend(
        [
            "\\midrule",
            f"Mean & All & "
            f"Scenario-best ({format_mean_std(mean_row['best_nonlearning_mean'], mean_row['best_nonlearning_std'], mean_row['best_nonlearning_mean'] == mean_row['winner_mean'])}) & "
            f"{format_mean_std(mean_row['ppo_mean'], mean_row['ppo_std'], mean_row['ppo_mean'] == mean_row['winner_mean'])} & "
            f"{format_mean_std(mean_row['sac_mean'], mean_row['sac_std'], mean_row['sac_mean'] == mean_row['winner_mean'])} & "
            f"{mean_row['learning_advantage']:.2f} \\\\",
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines), table_frame


def aggregate_table(summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    selected = summary[summary["algorithm"].isin(FOCUS_NONLEARNING_METHODS + ["ppo", "sac"])].copy()
    rows: list[dict[str, object]] = []
    for algorithm, group in selected.groupby("algorithm"):
        rows.append(
            {
                "family": "Learning" if algorithm in {"ppo", "sac"} else "Non-learning",
                "algorithm": algorithm,
                "mean_final_pi": float(group["mean_final_pi"].mean()),
                "std_across_scenarios": float(group["mean_final_pi"].std(ddof=1)),
                "num_scenarios": int(group["scenario"].nunique()),
            }
        )
    aggregate = pd.DataFrame(rows)
    aggregate["family_order"] = aggregate["family"].map({"Non-learning": 0, "Learning": 1})
    aggregate = aggregate.sort_values(["family_order", "mean_final_pi"], ascending=[True, False]).drop(
        columns=["family_order"]
    )

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Overall baseline performance averaged over the nine scenarios. Values are scenario-level final recovery performance index; higher is better.}",
        "\\label{tab:baseline_overall_mean}",
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Family & Method & Mean final PI & Scenarios \\\\",
        "\\midrule",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"{row.family} & {DISPLAY_METHOD[row.algorithm]} & "
            f"{row.mean_final_pi:.2f} $\\pm$ {row.std_across_scenarios:.2f} & {row.num_scenarios} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines), aggregate


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nonlearning = load_nonlearning(resolve(args.nonlearning_summary))
    rl = pd.concat(
        [
            load_rl_algorithm(resolve(args.rl_root), "ppo"),
            load_rl_algorithm(resolve(args.rl_root), "sac"),
        ],
        ignore_index=True,
    )
    combined = pd.concat([nonlearning, rl], ignore_index=True)
    summary = summarize(combined)

    compact_latex, compact_csv = compact_table(summary)
    aggregate_latex, aggregate_csv = aggregate_table(summary)
    latex = compact_latex + "\n" + aggregate_latex

    prefix = args.output_prefix
    summary.to_csv(output_dir / f"{prefix}_per_method_summary.csv", index=False)
    compact_csv.to_csv(output_dir / f"{prefix}_compact.csv", index=False)
    aggregate_csv.to_csv(output_dir / f"{prefix}_overall.csv", index=False)
    (output_dir / f"{prefix}.tex").write_text(latex, encoding="utf-8")

    print(output_dir / f"{prefix}.tex")
    print(output_dir / f"{prefix}_compact.csv")
    print(output_dir / f"{prefix}_overall.csv")


if __name__ == "__main__":
    main()
