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
  local dir="examples/method-comparison"
  run pixi run python -m reblock.compare data=capetown_full \
    "block_ids=[[ZAF.9.3.1_1_40972]]" \
    "methods=[topology,clearance,greedy_arterial_repulsion,osm_footpaths]" max_blocks=1 \
    all_methods.greedy_arterial_repulsion.max_roads=8 \
    "desire_source.snapshot=$dir/desire_lines_40972.geojson" \
    "hydra.run.dir=$dir"
  # Hydra writes <job>.log (job name 'compare') into the run dir; rename to run.log.
  run bash -c "[[ -f '$dir/compare.log' ]] && mv -f '$dir/compare.log' '$dir/run.log' || true"
}

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do gen_multiblock "$1" "$2"; shift 2; done
else
  for m in "${METRICS[@]}"; do for c in "${CITIES[@]}"; do gen_multiblock "$m" "$c"; done; done
  gen_method_comparison
fi
