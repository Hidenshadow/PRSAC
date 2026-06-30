from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_summary_metrics(run_dir: Path) -> list[dict[str, Any]]:
    summary_path = run_dir / "shock_recovery_summary.csv"
    if summary_path.exists():
        frame = pd.read_csv(summary_path)
        rows: list[dict[str, Any]] = []
        for _, row in frame.iterrows():
            clean = float(row["clean_nominal_cost"])
            attacked = float(row["attacked_nominal_cost"])
            final = float(row["final_recovery_cost"])
            best = float(row["best_recovery_cost"])
            rows.append(
                {
                    "eval_domain": str(row["eval_domain"]),
                    "shock_pi": 100.0 * clean / attacked if attacked > 0 else float("nan"),
                    "final_pi": 100.0 * clean / final if final > 0 else float("nan"),
                    "best_pi": 100.0 * clean / best if best > 0 else float("nan"),
                }
            )
        return rows

    curve_path = run_dir / "shock_recovery_curve.csv"
    if not curve_path.exists():
        raise FileNotFoundError(f"missing shock-recovery output: {run_dir}")
    frame = pd.read_csv(curve_path)
    env = frame[frame["attack_type"].astype(str) == "environment"].copy()
    rows = []
    for domain, group in env.groupby("eval_domain"):
        shock = group[group["phase"].astype(str) == "shock"]
        recovery = group[group["phase"].astype(str) == "recovery"]
        if shock.empty or recovery.empty:
            continue
        shock_row = shock.iloc[0]
        final_step = recovery["recovery_step"].max()
        final_row = recovery[recovery["recovery_step"] == final_step].iloc[-1]
        best_row = recovery.loc[recovery["mean_attacked_scalar_cost"].astype(float).idxmin()]
        clean = float(shock_row["mean_nominal_scalar_cost"])
        rows.append(
            {
                "eval_domain": str(domain),
                "shock_pi": 100.0 * clean / float(shock_row["mean_attacked_scalar_cost"]),
                "final_pi": 100.0 * clean / float(final_row["mean_attacked_scalar_cost"]),
                "best_pi": 100.0 * clean / float(best_row["mean_attacked_scalar_cost"]),
            }
        )
    return rows


def collect_algorithm(levels: list[str], seeds: list[int], root: Path, layout: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level in levels:
        for seed in seeds:
            if layout == "bvr":
                run_dir = root / level / f"seed{seed}"
            else:
                run_dir = root / f"{level}_shock_recovery_5seeds" / f"seed{seed}"
            for item in read_summary_metrics(run_dir):
                rows.append({"level": level, "seed": seed, **item})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvr-root", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-threshold", type=float, default=0.25)
    parser.add_argument("--best-threshold", type=float, default=0.50)
    parser.add_argument("--max-mean-sac-deficit", type=float, default=1.0)
    args = parser.parse_args()

    bvr_root = (PROJECT_ROOT / args.bvr_root).resolve() if not args.bvr_root.is_absolute() else args.bvr_root
    output_dir = (PROJECT_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bvr = collect_algorithm(args.levels, args.seeds, bvr_root, "bvr")
    ppo = collect_algorithm(args.levels, args.seeds, PROJECT_ROOT / "runs" / "rl_baselines" / "ppo", "baseline")
    sac = collect_algorithm(args.levels, args.seeds, PROJECT_ROOT / "runs" / "rl_baselines" / "sac", "baseline")

    for name, frame in [("bvr", bvr), ("ppo", ppo), ("sac", sac)]:
        frame["algorithm"] = name
    all_rows = pd.concat([bvr, ppo, sac], ignore_index=True)
    all_rows.to_csv(output_dir / "easy_algorithm_domain_metrics.csv", index=False)

    means = (
        all_rows.groupby(["algorithm", "level"], as_index=False)[["shock_pi", "final_pi", "best_pi"]]
        .mean()
        .sort_values(["level", "algorithm"])
    )
    means.to_csv(output_dir / "easy_level_means.csv", index=False)

    pivot = means.pivot(index="level", columns="algorithm", values=["final_pi", "best_pi"])
    decision_rows = []
    positive_levels = 0
    for level in args.levels:
        bvr_final = float(pivot.loc[level, ("final_pi", "bvr")])
        ppo_final = float(pivot.loc[level, ("final_pi", "ppo")])
        sac_final = float(pivot.loc[level, ("final_pi", "sac")])
        bvr_best = float(pivot.loc[level, ("best_pi", "bvr")])
        ppo_best = float(pivot.loc[level, ("best_pi", "ppo")])
        sac_best = float(pivot.loc[level, ("best_pi", "sac")])
        final_delta_ppo = bvr_final - ppo_final
        best_delta_ppo = bvr_best - ppo_best
        final_delta_sac = bvr_final - sac_final
        best_delta_sac = bvr_best - sac_best
        level_positive = final_delta_ppo >= args.final_threshold or best_delta_ppo >= args.best_threshold
        positive_levels += int(level_positive)
        decision_rows.append(
            {
                "level": level,
                "bvr_final_pi": bvr_final,
                "ppo_final_pi": ppo_final,
                "sac_final_pi": sac_final,
                "bvr_best_pi": bvr_best,
                "ppo_best_pi": ppo_best,
                "sac_best_pi": sac_best,
                "delta_final_vs_ppo": final_delta_ppo,
                "delta_best_vs_ppo": best_delta_ppo,
                "delta_final_vs_sac": final_delta_sac,
                "delta_best_vs_sac": best_delta_sac,
                "positive_level": bool(level_positive),
            }
        )

    decision = pd.DataFrame(decision_rows)
    decision.to_csv(output_dir / "easy_decision_summary.csv", index=False)
    mean_delta_final_ppo = float(decision["delta_final_vs_ppo"].mean())
    mean_delta_best_ppo = float(decision["delta_best_vs_ppo"].mean())
    mean_delta_final_sac = float(decision["delta_final_vs_sac"].mean())
    proceed_medium = (
        positive_levels >= 2
        and mean_delta_final_ppo >= 0.0
        and mean_delta_best_ppo >= 0.0
        and mean_delta_final_sac >= -float(args.max_mean_sac_deficit)
    )
    payload = {
        "proceed_medium": bool(proceed_medium),
        "positive_levels": int(positive_levels),
        "num_levels": int(len(args.levels)),
        "mean_delta_final_vs_ppo": mean_delta_final_ppo,
        "mean_delta_best_vs_ppo": mean_delta_best_ppo,
        "mean_delta_final_vs_sac": mean_delta_final_sac,
        "criteria": {
            "positive_levels_required": 2,
            "final_threshold_vs_ppo": float(args.final_threshold),
            "best_threshold_vs_ppo": float(args.best_threshold),
            "max_mean_final_deficit_vs_sac": float(args.max_mean_sac_deficit),
        },
    }
    (output_dir / "easy_decision.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if proceed_medium else 2


if __name__ == "__main__":
    raise SystemExit(main())
