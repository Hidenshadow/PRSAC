#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ALGORITHMS="${ALGORITHMS:-ppo sac}"
LEVELS="${LEVELS:-level1 level2 level3}"
DIFFICULTIES="${DIFFICULTIES:-easy medium hard}"
OUTPUT_BASE="${OUTPUT_BASE:-runs/rl_baselines}"
SEEDS="${SEEDS:-0 1 2 3 4}"

for algo in $ALGORITHMS; do
  for level in $LEVELS; do
    for difficulty in $DIFFICULTIES; do
      case "$level" in
        level1)
          ALGO="$algo" SEEDS="$SEEDS" OUTPUT_BASE="$OUTPUT_BASE" \
            DIFFICULTY="$difficulty" scripts/run_level1_shock_recovery_5seeds.sh
          ;;
        level2)
          ALGO="$algo" SEEDS="$SEEDS" OUTPUT_BASE="$OUTPUT_BASE" \
            DIFFICULTY="$difficulty" scripts/run_level2_shock_recovery_5seeds.sh
          ;;
        level3)
          ALGO="$algo" SEEDS="$SEEDS" OUTPUT_BASE="$OUTPUT_BASE" \
            DIFFICULTY="$difficulty" scripts/run_level3_shock_recovery_5seeds.sh
          ;;
        *)
          echo "Unknown level: $level" >&2
          exit 1
          ;;
      esac
    done
  done
done
