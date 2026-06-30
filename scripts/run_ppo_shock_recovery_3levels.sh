#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LEVELS="${LEVELS:-level1 level2 level3}"
DIFFICULTIES="${DIFFICULTIES:-easy medium hard}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/ppo_difficulty}"
SEEDS="${SEEDS:-0 1 2 3 4}"
SEED_COUNT="$(printf '%s\n' $SEEDS | wc -l | tr -d ' ')"

for level in $LEVELS; do
  for difficulty in $DIFFICULTIES; do
    case "$level" in
      level1)
        OUTPUT_ROOT="$OUTPUT_BASE/level1_${difficulty}_shock_recovery_${SEED_COUNT}seeds" \
          SEEDS="$SEEDS" \
          DIFFICULTY="$difficulty" scripts/run_level1_shock_recovery_5seeds.sh
        ;;
      level2)
        OUTPUT_ROOT="$OUTPUT_BASE/level2_${difficulty}_shock_recovery_${SEED_COUNT}seeds" \
          SEEDS="$SEEDS" \
          DIFFICULTY="$difficulty" scripts/run_level2_shock_recovery_5seeds.sh
        ;;
      level3)
        OUTPUT_ROOT="$OUTPUT_BASE/level3_${difficulty}_shock_recovery_${SEED_COUNT}seeds" \
          SEEDS="$SEEDS" \
          DIFFICULTY="$difficulty" scripts/run_level3_shock_recovery_5seeds.sh
        ;;
      *)
        echo "Unknown level: $level" >&2
        exit 1
        ;;
    esac
  done
done
