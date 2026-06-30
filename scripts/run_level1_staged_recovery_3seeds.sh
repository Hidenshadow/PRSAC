#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
LEVEL_CONFIG="${LEVEL_CONFIG:-configs/levels/synthetic_corridor.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/level1_synthetic_corridor_staged_3seeds}"
SEEDS="${SEEDS:-0 1 2}"
NOMINAL_TIMESTEPS="${NOMINAL_TIMESTEPS:-50000}"
STAGE_TIMESTEPS="${STAGE_TIMESTEPS:-20480}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1024}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-300}"
IN_DOMAIN_SEED="${IN_DOMAIN_SEED:-909}"
HELDOUT_SEED="${HELDOUT_SEED:-1919}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"

mkdir -p "$OUTPUT_ROOT" logs

for seed in $SEEDS; do
  out="$OUTPUT_ROOT/seed${seed}"
  config_dir="$out/configs"
  base_config="$config_dir/ppo_base_lunar_rover_corridor_seed${seed}.json"
  attack_config="$config_dir/environment_attack.json"
  nominal="$out/nominal_ppo/checkpoint.pt"

  if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    rm -rf -- "$out"
  fi
  mkdir -p "$config_dir"

  "$PYTHON" - "$LEVEL_CONFIG" "$base_config" "$attack_config" "$seed" <<'PY'
import json
import sys
from pathlib import Path

level_config_path = Path(sys.argv[1])
base_config_path = Path(sys.argv[2])
attack_config_path = Path(sys.argv[3])
seed = int(sys.argv[4])

level = json.loads(level_config_path.read_text(encoding="utf-8-sig"))
base_path = Path(level["base_config"])
base = json.loads(base_path.read_text(encoding="utf-8-sig"))
args = base.setdefault("args", {})

args["seed"] = seed
args["map-size"] = int(level.get("map_size", args.get("map-size", 48)))
args["scenario"] = str(level.get("scenario", args.get("scenario", "lunar_rover_corridor")))
args["observation-mode"] = str(level.get("observation_mode", args.get("observation-mode", "terrain")))
args["reward-mode"] = str(level.get("reward_mode", args.get("reward-mode", "relative_heuristic")))
args["reward-scale"] = float(level.get("reward_scale", args.get("reward-scale", 10.0)))
args["reward-cost-key"] = str(level.get("reward_cost_key", args.get("reward-cost-key", "scalar_cost")))
args["action-mode"] = str(level.get("action_mode", args.get("action-mode", "preference_delta")))
args["action-gain"] = float(level.get("action_gain", args.get("action-gain", 3.0)))
args["max-uncertainty-lambda"] = float(
    level.get("max_uncertainty_lambda", args.get("max-uncertainty-lambda", 1.2))
)
args["map-sampling-mode"] = str(level.get("map_sampling_mode", args.get("map-sampling-mode", "map_seed_pool")))
args["fixed-map-seed"] = int(level.get("fixed_map_seed", args.get("fixed-map-seed", 909)))
args["map-seed-pool-size"] = int(level.get("map_seed_pool_size", args.get("map-seed-pool-size", 32)))

environment_attack = dict(level.get("attacks", {}).get("environment", {}))
environment_attack.setdefault("enabled", True)
environment_attack.setdefault("type", "env_belief_mismatch")
environment_attack.setdefault("mode", "risk_underestimate")
environment_attack.setdefault("selection_mode", "low_confidence_high_consequence")
environment_attack.setdefault("top_fraction", 0.25)
environment_attack.setdefault("attack_strength", 3.0)
environment_attack.setdefault("error_scale", 0.20)
environment_attack.setdefault("background_error_scale", 0.20)
environment_attack.setdefault("min_confidence", 0.15)
environment_attack.setdefault("max_confidence", 0.95)
environment_attack.setdefault("affected_layers", ["energy", "hazard", "communication", "illumination"])
environment_attack.setdefault("confidence_to_uncertainty", True)
environment_attack.setdefault("apply_during_training", True)
environment_attack.setdefault("reward_uses_attacked_cost", True)
args["environment_attack"] = environment_attack

base_config_path.write_text(json.dumps(base, indent=2), encoding="utf-8")
attack_config_path.write_text(json.dumps(environment_attack, indent=2), encoding="utf-8")
PY

  echo "=== Level 1 Synthetic PPO nominal seed=$seed ==="
  "$PYTHON" -u run_robustness_workflow.py \
    --base-config "$base_config" \
    --mode train_nominal \
    --output-root "$out" \
    --total-timesteps "$NOMINAL_TIMESTEPS"

  if [[ ! -f "$nominal" ]]; then
    echo "Missing nominal checkpoint after training: $nominal" >&2
    exit 1
  fi

  echo "=== Level 1 Synthetic staged recovery seed=$seed ==="
  "$PYTHON" -u run_staged_attack_recovery.py \
    --base-config "$base_config" \
    --checkpoint "$nominal" \
    --attack-config "$attack_config" \
    --output-dir "$out" \
    --stage-timesteps "$STAGE_TIMESTEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    --num-eval-episodes "$NUM_EVAL_EPISODES" \
    --seed "$seed" \
    --in-domain-seed "$IN_DOMAIN_SEED" \
    --heldout-seed "$HELDOUT_SEED"
done

"$PYTHON" - "$OUTPUT_ROOT" <<'PY'
from pathlib import Path
import re
import sys
import pandas as pd

output_root = Path(sys.argv[1])
frames = []
for path in sorted(output_root.glob("seed*/staged_recovery_curve.csv")):
    seed_match = re.search(r"seed(\d+)", str(path))
    frame = pd.read_csv(path)
    frame.insert(0, "training_seed", int(seed_match.group(1)))
    frames.append(frame)

if not frames:
    raise SystemExit("No staged recovery curves were produced.")

combined = pd.concat(frames, ignore_index=True)
combined_path = output_root / "staged_recovery_curve_all_seeds.csv"
combined.to_csv(combined_path, index=False)

active = combined[combined["attack_type"] == combined["active_attack"]].copy()
summary_cols = [
    "mean_attacked_scalar_cost",
    "relative_degradation",
    "mean_attacked_cell_exposure_ratio",
    "mean_lambda_uncertainty",
]
aggregate = (
    active.groupby(["eval_domain", "stage", "active_attack", "stage_step", "cumulative_step"], as_index=False)[
        summary_cols
    ]
    .agg(["mean", "std"])
)
aggregate.columns = [
    "_".join(str(part) for part in col if part)
    if isinstance(col, tuple)
    else str(col)
    for col in aggregate.columns
]
aggregate_path = output_root / "staged_active_attack_aggregate.csv"
aggregate.to_csv(aggregate_path, index=False)

print(f"Saved combined staged curve: {combined_path}")
print(f"Saved active-attack aggregate: {aggregate_path}")
PY

echo "Done. Results: $OUTPUT_ROOT"
