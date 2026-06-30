#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-runs/lunar_viper_staged_recovery_seed}"
SEEDS="${SEEDS:-0 1 2}"
NOMINAL_TIMESTEPS="${NOMINAL_TIMESTEPS:-50000}"
STAGE_TIMESTEPS="${STAGE_TIMESTEPS:-20480}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1024}"
NUM_EVAL_EPISODES="${NUM_EVAL_EPISODES:-128}"

mkdir -p logs

for seed in $SEEDS; do
  out="${OUTPUT_PREFIX}${seed}"
  echo "=== Level 2 Lunar/VIPER PPO staged recovery seed=$seed ==="
  "$PYTHON" -u run_lunar_viper_staged_recovery.py \
    --clean-output \
    --output-dir "$out" \
    --nominal-timesteps "$NOMINAL_TIMESTEPS" \
    --stage-timesteps "$STAGE_TIMESTEPS" \
    --eval-interval "$EVAL_INTERVAL" \
    --num-eval-episodes "$NUM_EVAL_EPISODES" \
    --seed "$seed"
done

echo "Done. Results prefix: $OUTPUT_PREFIX"
