#!/usr/bin/env bash
# Regenerate reblock examples. No args => all; else pass "<metric> <city>" pairs.
# --dry-run prints the commands without running them.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0; [[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }
run() { echo "+ $*"; [[ $DRY -eq 1 ]] || "$@"; }

# One entry point for every example. A variant IS a config file (conf/example/<name>.yaml); the
# multiblock ones run per city, method-comparison pins a single block so it is capetown-only.
VARIANTS=(depth depth_density density_compactness)
CITIES=(capetown nairobi)

gen() {  # <variant> <city>
  local variant="$1" city="$2"
  run pixi run python -m scripts.gen_example "$variant" $([[ "$city" == capetown ]] || echo "$city")
}

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do gen "$1" "$2"; shift 2; done
else
  for v in "${VARIANTS[@]}"; do for c in "${CITIES[@]}"; do gen "$v" "$c"; done; done
  gen method_comparison capetown
fi
