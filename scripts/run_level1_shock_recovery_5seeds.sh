#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SEEDS="${SEEDS:-0 1 2 3 4}"
exec "$ROOT/scripts/run_level1_shock_recovery_3seeds.sh" "$@"
