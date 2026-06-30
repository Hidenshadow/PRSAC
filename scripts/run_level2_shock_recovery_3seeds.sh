#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
ALGO="${ALGO:-ppo}"
DIFFICULTY="${DIFFICULTY:-medium}"
LEVEL_CONFIG="${LEVEL_CONFIG:-configs/levels/ppo_difficulty/level2_${DIFFICULTY}.json}"
BASE_CONFIG="${BASE_CONFIG:-configs/ppo_lunar_viper_relative_reward.json}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/rl_baselines}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SEED_COUNT="$(printf '%s\n' $SEEDS | wc -l | tr -d ' ')"
OUTPUT_ROOT="${OUTPUT_ROOT:-${OUTPUT_BASE}/${ALGO}/level2_${DIFFICULTY}_shock_recovery_${SEED_COUNT}seeds}"
NOMINAL_TIMESTEPS="${NOMINAL_TIMESTEPS:-50000}"
RECOVERY_TIMESTEPS="${RECOVERY_TIMESTEPS:-20480}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1024}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-128}"
TRAIN_EVAL_EPISODES="${TRAIN_EVAL_EPISODES:-64}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
DRY_RUN="${DRY_RUN:-0}"
QUICK="${QUICK:-0}"

mkdir -p logs
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$OUTPUT_ROOT"
fi

for seed in $SEEDS; do
  out="$OUTPUT_ROOT/seed${seed}"
  args=(
    "$PYTHON" -u run_shock_recovery_experiment.py
    --algo "$ALGO"
    --level-config "$LEVEL_CONFIG"
    --base-config "$BASE_CONFIG"
    --output-dir "$out"
    --seed "$seed"
    --nominal-timesteps "$NOMINAL_TIMESTEPS"
    --recovery-timesteps "$RECOVERY_TIMESTEPS"
    --eval-interval "$EVAL_INTERVAL"
    --num-eval-episodes "$NUM_EVAL_EPISODES"
    --train-eval-episodes "$TRAIN_EVAL_EPISODES"
  )
  if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    args+=(--clean-output)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    args+=(--dry-run)
  fi
  if [[ "$QUICK" == "1" ]]; then
    args+=(--quick)
  fi

  echo "=== Level 2 $ALGO shock-recovery difficulty=$DIFFICULTY seed=$seed seeds=$SEED_COUNT ==="
  "${args[@]}"
done

if [[ "$DRY_RUN" != "1" ]]; then
  "$PYTHON" scripts/aggregate_shock_recovery_3seeds.py "$OUTPUT_ROOT"
fi

echo "Done. Results: $OUTPUT_ROOT"
