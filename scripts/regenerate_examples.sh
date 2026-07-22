#!/usr/bin/env bash
# Regenerate reblock examples. No args => all; else pass "<metric> <city>" pairs.
# --dry-run prints the commands without running them.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0; [[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }
run() { echo "+ $*"; [[ $DRY -eq 1 ]] || "$@"; }

METRICS=(depth depth_density density_compactness)
CITIES=(capetown nairobi)

gen_multiblock() {  # <metric> <city>
  local metric="$1" city="$2"
  run pixi run python -m scripts.gen_multiblock_example "$metric" $([[ "$city" == capetown ]] || echo "$city")
}

gen_method_comparison() {
  # Single-block flagship: curves + per-method before/after renders, all reproducible (self-logs
  # run.log). See scripts/gen_method_comparison.py for the pinned block + method set.
  run pixi run python -m scripts.gen_method_comparison
}

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do gen_multiblock "$1" "$2"; shift 2; done
else
  for m in "${METRICS[@]}"; do for c in "${CITIES[@]}"; do gen_multiblock "$m" "$c"; done; done
  gen_method_comparison
fi
