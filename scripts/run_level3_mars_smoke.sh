#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/level3_mars_dtm_smoke}"
SEEDS="${SEEDS:-0}"
NUM_EPISODES="${NUM_EPISODES:-5}"

"$PYTHON" run_multilevel_rover_robustness.py \
  --level mars_dtm \
  --quick \
  --clean-output \
  --output-dir "$OUTPUT_ROOT" \
  --num-episodes "$NUM_EPISODES" \
  --seeds "$SEEDS"

echo "Done. Results: $OUTPUT_ROOT"
