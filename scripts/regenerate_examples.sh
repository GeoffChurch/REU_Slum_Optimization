#!/usr/bin/env bash
# Regenerate reblock examples. No args => all; else pass "<metric> <city>" pairs.
# --dry-run prints the commands without running them.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY=0; [[ "${1:-}" == "--dry-run" ]] && { DRY=1; shift; }

# CRASH SAFETY. `gen_example` unlinks stale artifacts from the LIVE examples/<slug>/ directory
# before regenerating it, so an interruption between the unlink and the writes leaves that example
# gutted -- which is exactly what happened on 2026-08-09 (25 files deleted, recovered from git).
# Any variant that does not finish therefore has its directory restored from git, and an interrupted
# run leaves the published examples byte-identical to how it found them.
#
# Requires a clean examples/ to start, so "restore from git" cannot silently discard real work.
if [[ $DRY -eq 0 ]] && ! git diff --quiet -- examples/; then
  echo "examples/ has uncommitted changes; commit or stash them first (this script restores from" >&2
  echo "git on interruption and would discard them)." >&2
  exit 1
fi
_CURRENT_DIR=""
_CHILD=""
_restore() {
  # Stop the in-flight generator FIRST. A bash trap does not interrupt the foreground command --
  # it runs only once that command returns -- so a TERM to this script alone leaves the python
  # child running and the trap never fires (observed 2026-08-10). Kill the child's process group.
  [[ -n "$_CHILD" ]] && kill -TERM -"$_CHILD" 2>/dev/null
  [[ -n "$_CURRENT_DIR" ]] && {
    echo "!! interrupted during $_CURRENT_DIR -- restoring it from git" >&2
    git checkout -- "$_CURRENT_DIR" 2>/dev/null || true
  }
  # An EXIT trap's status becomes the script's, and the tests above are FALSE on a clean run
  # (nothing in flight, nothing to restore) -- which made a fully successful regeneration exit 1.
  return 0
}
trap _restore EXIT INT TERM

run() { echo "+ $*"; [[ $DRY -eq 1 ]] || "$@"; }

# One entry point for every example. A variant IS a config file (conf/example/<name>.yaml); the
# multiblock ones run per city, method-comparison pins a single block so it is capetown-only.
VARIANTS=(depth depth_density density_compactness)
CITIES=(capetown nairobi)

gen() {  # <variant> <city>
  local variant="$1" city="$2"
  local slug
  slug=$(pixi run python -c "
from omegaconf import OmegaConf
c = OmegaConf.load('conf/example/${variant}.yaml')
print(c.example.slug)" 2>/dev/null | tail -1)
  if [[ "$city" == capetown ]]; then _CURRENT_DIR="examples/$slug"; else _CURRENT_DIR="examples/$city/$slug"; fi
  if [[ $DRY -eq 1 ]]; then
    echo "+ pixi run python -m scripts.gen_example $variant $([[ "$city" == capetown ]] || echo "$city")"
  else
    echo "+ pixi run python -m scripts.gen_example $variant" \
         "$([[ "$city" == capetown ]] || echo "$city")"
    setsid pixi run python -m scripts.gen_example "$variant" \
        $([[ "$city" == capetown ]] || echo "$city") &
    _CHILD=$!
    wait "$_CHILD"
    _CHILD=""
  fi
  _CURRENT_DIR=""          # completed cleanly; nothing to restore
}

if [[ $# -gt 0 ]]; then
  while [[ $# -gt 0 ]]; do gen "$1" "$2"; shift 2; done
else
  for v in "${VARIANTS[@]}"; do for c in "${CITIES[@]}"; do gen "$v" "$c"; done; done
  gen method_comparison capetown
  # The screen bake-off is not a gen_example variant: it grades SCREENS against ground truth rather
  # than methods against a region, so it has its own entry point. Cape Town only -- the ground truth
  # is the City's own structure survey and no equivalent exists for Nairobi.
  run pixi run python -m scripts.gen_screen_bakeoff
  # The graph figure set is likewise its own entry point, and for the same kind of reason: it is a
  # figure set for one site page rather than a graded example, and it deliberately does NOT re-run
  # the ten-method comparison whose block and roads it borrows.
  run pixi run python -m scripts.gen_perm_graph
  # The web bundle shares gen_perm_graph's block, method and pinned config, and is the directory's
  # largest committed file -- it belongs in the same regeneration path as the PNGs it must stay in
  # sync with (fix wave, I7), not off it where a re-bake is easy to forget.
  run pixi run python -m scripts.gen_web_bundle
  # The frontier bundle shares the same pinned block (via scripts/_example_block.py) but is a
  # separate file for a separate page -- see gen_frontier_bundle's module docstring.
  run pixi run python -m scripts.gen_frontier_bundle
  # The displacement field: the same pinned block once more, its own bundle AND its own fallback
  # PNG (deliberately not shared with the perm-graph bundle -- see gen_displacement_field's module
  # docstring). Both are committed, so the same I7 reasoning applies: on this path, not off it.
  run pixi run python -m scripts.gen_displacement_field
fi
