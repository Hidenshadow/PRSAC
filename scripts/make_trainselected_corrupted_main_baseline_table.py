#!/usr/bin/env python
"""Create a train-selected corrupted-map main table with all learning baselines."""

from __future__ import annotations

from pathlib import Path

import math
import pandas as pd

import update_ldac_baseline_tables as base


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = PROJECT_ROOT / "runs" / "baseline_tables"
SOURCE_POINTS = TABLE_DIR / "heldout_final_all_baselines_9scenario_points.csv"

OUTPUT_POINTS = TABLE_DIR / "heldout_trainselected_corrupted_main_baselines_points.csv"
OUTPUT_CSV = TABLE_DIR / "heldout_trainselected_corrupted_main_baselines_table.csv"
OUTPUT_TEX = TABLE_DIR / "heldout_trainselected_corrupted_main_baselines_table.tex"

KEEP_NONLEARNING = ("Minimax", "Guard", "Risk-Inflated A*", "Belief-CVaR A*")
LEARNING_METHODS = ("PPO", "SAC", "CDR-SAC", "Stackelberg-SAC", "VALT-SAC", "LDAC-SAC")
METHOD_ORDER = tuple(("Non-learning", method) for method in KEEP_NONLEARNING) + tuple(
    ("Learning", method) for method in LEARNING_METHODS
)

SAC_MODIFIED_ROOTS = {
    "CDR-SAC": PROJECT_ROOT / "runs" / "sac_modified" / "cdr_sac_from_sac_nominal_9scenarios_2seeds_20260627",
    "Stackelberg-SAC": PROJECT_ROOT
    / "runs"
    / "sac_modified"
    / "stackelberg_sac_from_sac_nominal_9scenarios_2seeds_20260628",
    "VALT-SAC": PROJECT_ROOT / "runs" / "sac_modified" / "valt_sac_from_sac_nominal_9scenarios_3seeds_20260628",
}


def is_finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def training_domain_mask(values: pd.Series) -> pd.Series:
    lowered = values.astype(str).str.lower()
    return lowered.str.contains("train", regex=False) | lowered.str.contains("in_domain", regex=False)


def scenario_from_summary_path(summary_path: Path) -> str:
    parent_name = summary_path.parent.parent.name
    suffix = "_shock_recovery_5seeds"
    return parent_name[: -len(suffix)] if parent_name.endswith(suffix) else parent_name


def baseline_summary_paths(method: str) -> list[Path]:
    algo = method.lower()
    paths: list[Path] = []
    for scenario in base.SCENARIOS:
        paths.extend(
            sorted(
                (
                    PROJECT_ROOT
                    / "runs"
                    / "rl_baselines"
                    / algo
                    / f"{scenario}_shock_recovery_5seeds"
                ).glob("seed*/shock_recovery_summary.csv")
            )
        )
    return paths


def ldac_summary_paths() -> list[Path]:
    paths: list[Path] = []
    for difficulty, roots in base.LDAC_ROOTS_BY_DIFFICULTY.items():
        for root in roots:
            paths.extend(sorted(root.glob(f"level*_{difficulty}/seed*/shock_recovery_summary.csv")))
    return paths


def sac_modified_summary_paths(method: str) -> list[Path]:
    paths: list[Path] = []
    root = SAC_MODIFIED_ROOTS[method]
    for scenario in base.SCENARIOS:
        paths.extend(sorted((root / scenario).glob("seed*/shock_recovery_summary.csv")))
    return paths


def learning_summary_paths(method: str) -> list[Path]:
    if method in {"PPO", "SAC"}:
        return baseline_summary_paths(method)
    if method == "LDAC-SAC":
        return ldac_summary_paths()
    if method in SAC_MODIFIED_ROOTS:
        return sac_modified_summary_paths(method)
    raise ValueError(method)


def selected_step_cost(summary_path: Path, eval_domain: str, selected_step: int) -> float:
    curve_path = summary_path.parent / "shock_recovery_curve.csv"
    curve = read_csv(curve_path)
    required = {"eval_domain", "phase", "attack_type", "recovery_step", "mean_attacked_scalar_cost"}
    if not required.issubset(curve.columns):
        raise ValueError(f"missing columns in {curve_path}: {sorted(required - set(curve.columns))}")
    rows = curve[
        (curve["eval_domain"].astype(str) == eval_domain)
        & (curve["attack_type"].astype(str) == "environment")
        & (curve["phase"].astype(str).isin(["shock", "recovery"]))
    ].copy()
    if rows.empty:
        raise ValueError(f"missing selected-step rows in {curve_path} for {eval_domain}")
    rows["step_distance"] = (rows["recovery_step"].astype(float) - float(selected_step)).abs()
    return float(rows.sort_values(["step_distance", "recovery_step"]).iloc[0]["mean_attacked_scalar_cost"])


def train_selected_step(frame: pd.DataFrame, summary_path: Path) -> int:
    train_rows = frame[training_domain_mask(frame["eval_domain"])]
    if not train_rows.empty and "best_recovery_step" in train_rows.columns:
        step = train_rows.iloc[0]["best_recovery_step"]
        if pd.notna(step):
            return int(step)

    curve_path = summary_path.parent / "shock_recovery_curve.csv"
    curve = read_csv(curve_path)
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
    train["performance_index"] = 100.0 * clean_cost / train["mean_attacked_scalar_cost"].astype(float).clip(
        lower=1e-12
    )
    return int(train.sort_values(["performance_index", "recovery_step"], ascending=[False, True]).iloc[0]["recovery_step"])


def recompute_learning_points() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in LEARNING_METHODS:
        for summary_path in learning_summary_paths(method):
            scenario = scenario_from_summary_path(summary_path)
            if scenario not in base.SCENARIOS:
                continue
            seed = base.seed_from_path(summary_path.parent)
            level, difficulty = base.scenario_parts(scenario)
            frame = read_csv(summary_path)
            required = {"eval_domain", "clean_nominal_cost"}
            if not required.issubset(frame.columns):
                raise ValueError(f"missing columns in {summary_path}: {sorted(required - set(frame.columns))}")
            selected_step = train_selected_step(frame, summary_path)
            for _, row in frame.iterrows():
                eval_domain = str(row["eval_domain"])
                if "heldout" not in eval_domain.lower():
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
                        "clean_index": 100.0 * reference / clean_cost,
                        "corrupted_index": 100.0 * reference / selected_cost,
                        "selected_recovery_step": selected_step,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("no train-selected learning rows found")
    return out.sort_values(["method", "difficulty", "level", "seed", "eval_domain"]).reset_index(drop=True)


def make_points() -> pd.DataFrame:
    source = read_csv(SOURCE_POINTS)
    nonlearning = source[(source["family"] == "Non-learning") & (source["method"].isin(KEEP_NONLEARNING))].copy()
    nonlearning["selected_recovery_step"] = pd.NA
    points = pd.concat([nonlearning, recompute_learning_points()], ignore_index=True)
    points = points.sort_values(["family", "method", "difficulty", "level", "seed", "eval_domain"]).reset_index(
        drop=True
    )
    points.to_csv(OUTPUT_POINTS, index=False)
    return points


def aggregate(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, method in METHOD_ORDER:
        row: dict[str, object] = {"Family": family, "Method": method}
        subset = points[(points["family"] == family) & (points["method"] == method)]
        for difficulty in base.DIFFICULTIES:
            for level in base.LEVELS:
                prefix = f"{level}_{difficulty}"
                scenario_subset = subset[(subset["difficulty"] == difficulty) & (subset["level"] == level)]
                if scenario_subset.empty:
                    row[f"{prefix}_corrupted_index_mean"] = math.nan
                    row[f"{prefix}_corrupted_index_std"] = math.nan
                    row[f"{prefix}_n"] = 0
                else:
                    row[f"{prefix}_corrupted_index_mean"] = float(scenario_subset["corrupted_index"].mean())
                    row[f"{prefix}_corrupted_index_std"] = float(scenario_subset["corrupted_index"].std(ddof=1))
                    row[f"{prefix}_n"] = int(len(scenario_subset))
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_value(mean: float, std: float) -> str:
    if not is_finite(mean):
        return "--"
    std = 0.0 if not is_finite(std) else float(std)
    return f"{float(mean):.2f} $\\pm$ {std:.2f}"


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )


def winners(summary: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for difficulty in base.DIFFICULTIES:
        for level in base.LEVELS:
            prefix = f"{level}_{difficulty}"
            values = []
            for _, row in summary.iterrows():
                mean = row[f"{prefix}_corrupted_index_mean"]
                if is_finite(mean):
                    values.append((float(mean), str(row["Method"])))
            best = max((value for value, _ in values), default=math.nan)
            out[prefix] = {method for value, method in values if is_finite(best) and abs(value - best) < 1e-9}
    return out


def write_outputs(summary: pd.DataFrame) -> None:
    csv_rows = []
    for _, row in summary.iterrows():
        out = {"Family": row["Family"], "Method": row["Method"]}
        for difficulty in base.DIFFICULTIES:
            for level in base.LEVELS:
                prefix = f"{level}_{difficulty}"
                out[prefix] = fmt_value(row[f"{prefix}_corrupted_index_mean"], row[f"{prefix}_corrupted_index_std"])
                out[f"{prefix}_n"] = int(row[f"{prefix}_n"])
        csv_rows.append(out)
    pd.DataFrame(csv_rows).to_csv(OUTPUT_CSV, index=False)

    win = winners(summary)
    caption = (
        "Held-out corrupted-map performance across nine scenarios using train-selected recovery checkpoints. "
        "Values are common-reference performance index, reported as mean $\\pm$ std over held-out evaluations; "
        "higher is better. For learning methods, the recovery checkpoint is selected by training-domain "
        "corrupted-map performance and then evaluated on the held-out corrupted map. Non-learning rows use static "
        "corrupted-map planner performance. The table includes robust/risk-aware non-learning planners with clear "
        "literature support and all completed learning baselines. CDR-SAC uses two seeds; Stackelberg-SAC uses "
        "five seeds for Easy-L1 and three seeds otherwise; all other methods use five seeds."
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        f"\\caption{{{caption}}}",
        "\\label{tab:heldout_trainselected_corrupted_main_baselines}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{llccccccccc}",
        "\\toprule",
        "Family & Method & \\multicolumn{3}{c}{Easy} & \\multicolumn{3}{c}{Medium} & \\multicolumn{3}{c}{Hard} \\\\",
        "\\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}",
        " & & L1 & L2 & L3 & L1 & L2 & L3 & L1 & L2 & L3 \\\\",
        "\\midrule",
    ]
    previous_family = None
    for _, row in summary.iterrows():
        family = str(row["Family"])
        if previous_family is not None and previous_family != family:
            lines.append("\\addlinespace[0.2em]")
        previous_family = family
        cells = [latex_escape(family), latex_escape(str(row["Method"]))]
        for difficulty in base.DIFFICULTIES:
            for level in base.LEVELS:
                prefix = f"{level}_{difficulty}"
                value = fmt_value(row[f"{prefix}_corrupted_index_mean"], row[f"{prefix}_corrupted_index_std"])
                if str(row["Method"]) in win[prefix]:
                    value = f"\\textbf{{{value}}}"
                cells.append(value)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table*}", ""])
    OUTPUT_TEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    points = make_points()
    summary = aggregate(points)
    write_outputs(summary)
    print(f"Wrote {OUTPUT_POINTS}")
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
