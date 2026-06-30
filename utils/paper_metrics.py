"""Publication-facing metrics for PPO shock-recovery experiments.

Training rewards are intentionally left alone.  These helpers only convert
evaluation curves into stable, easy-to-explain tables for papers and reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_MIN_REPORTABLE_ATTACK_DEGRADATION = 0.05
EPS = 1e-8


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < EPS:
        return float("nan")
    return float(numerator) / float(denominator)


def summarize_shock_recovery_frame(
    frame: pd.DataFrame,
    min_reportable_attack_degradation: float = DEFAULT_MIN_REPORTABLE_ATTACK_DEGRADATION,
) -> pd.DataFrame:
    """Build one row per eval domain from a shock-recovery curve.

    The legacy recovery ratio becomes misleading when the attack barely changes
    cost.  We still keep raw closure columns for diagnostics, but the paper
    columns are reportable only when the initial attack degradation is large
    enough to make recovery meaningful.
    """

    rows: list[dict[str, object]] = []
    for eval_domain, group in frame.groupby("eval_domain"):
        clean = group[(group["phase"] == "shock") & (group["attack_type"] == "none")]
        shocked = group[(group["phase"] == "shock") & (group["attack_type"] == "environment")]
        recovery = group[group["attack_type"] == "environment"].copy()
        if clean.empty or shocked.empty or recovery.empty:
            continue

        clean_row = clean.iloc[0]
        shock_row = shocked.iloc[0]
        final_row = recovery.sort_values("recovery_step").iloc[-1]
        best_row = recovery.loc[recovery["mean_attacked_scalar_cost"].astype(float).idxmin()]

        clean_cost = float(clean_row["mean_attacked_scalar_cost"])
        shock_cost = float(shock_row["mean_attacked_scalar_cost"])
        final_cost = float(final_row["mean_attacked_scalar_cost"])
        best_cost = float(best_row["mean_attacked_scalar_cost"])

        attack_drop = shock_cost - clean_cost
        final_recovery = shock_cost - final_cost
        best_recovery = shock_cost - best_cost

        attack_degradation = safe_ratio(attack_drop, abs(clean_cost))
        final_residual_degradation = safe_ratio(final_cost - clean_cost, abs(clean_cost))
        best_residual_degradation = safe_ratio(best_cost - clean_cost, abs(clean_cost))

        final_closure_raw = safe_ratio(final_recovery, attack_drop) if attack_drop > EPS else float("nan")
        best_closure_raw = safe_ratio(best_recovery, attack_drop) if attack_drop > EPS else float("nan")
        reportable = bool(
            attack_drop > EPS
            and np.isfinite(attack_degradation)
            and attack_degradation >= float(min_reportable_attack_degradation)
        )
        if attack_drop <= EPS:
            attack_effect_status = "no_attack_degradation"
        elif reportable:
            attack_effect_status = "meaningful_attack"
        else:
            attack_effect_status = "weak_attack"

        final_closure = final_closure_raw if reportable else float("nan")
        best_closure = best_closure_raw if reportable else float("nan")

        rows.append(
            {
                "eval_domain": eval_domain,
                "clean_nominal_cost": clean_cost,
                "attacked_nominal_cost": shock_cost,
                "final_recovery_cost": final_cost,
                "best_recovery_cost": best_cost,
                "attack_drop": attack_drop,
                "attack_drop_ratio": attack_degradation,
                "attack_degradation": attack_degradation,
                "attack_degradation_pct": 100.0 * attack_degradation,
                "final_recovery": final_recovery,
                "best_recovery": best_recovery,
                "final_recovery_ratio_raw": final_closure_raw,
                "best_recovery_ratio_raw": best_closure_raw,
                "final_recovery_ratio": final_closure,
                "best_recovery_ratio": best_closure,
                "final_recovery_closure": final_closure,
                "best_recovery_closure": best_closure,
                "final_recovery_closure_pct": 100.0 * final_closure if reportable else float("nan"),
                "best_recovery_closure_pct": 100.0 * best_closure if reportable else float("nan"),
                "final_residual_degradation": final_residual_degradation,
                "best_residual_degradation": best_residual_degradation,
                "final_residual_degradation_pct": 100.0 * final_residual_degradation,
                "best_residual_degradation_pct": 100.0 * best_residual_degradation,
                "final_relative_degradation": float(final_row["relative_degradation"]),
                "best_relative_degradation": float(best_row["relative_degradation"]),
                "success_rate_final": float(final_row["success_rate"]),
                "mean_map_mismatch_penalty_shock": float(shock_row.get("mean_map_mismatch_penalty", np.nan)),
                "mean_path_confidence_shock": float(shock_row.get("mean_path_confidence", np.nan)),
                "mean_attacked_cell_exposure_ratio_shock": float(
                    shock_row.get("mean_attacked_cell_exposure_ratio", np.nan)
                ),
                "mean_lambda_uncertainty_shock": float(shock_row.get("mean_lambda_uncertainty", np.nan)),
                "best_checkpoint_path": str(best_row["checkpoint_path"]),
                "best_recovery_step": int(best_row["recovery_step"]),
                "min_reportable_attack_degradation": float(min_reportable_attack_degradation),
                "recovery_metric_reportable": reportable,
                "attack_effect_status": attack_effect_status,
            }
        )
    return pd.DataFrame(rows)


def numeric_summary_columns(frame: pd.DataFrame) -> list[str]:
    """Return numeric columns that are useful to aggregate across seeds."""

    excluded = {"training_seed"}
    return [
        column
        for column in frame.select_dtypes(include=[np.number, "bool"]).columns
        if column not in excluded
    ]
