# Clearance Flagship — Design

**Status:** approved design (config confirmed 2026-07-14) · **Date:** 2026-07-14

**Goal:** A massive-region flagship example — reblock an **entire ~175-block Cape Town informal
settlement** (~17,400 homes, up to 13 parcels deep) with the clearance reblocker (chord_diag
default) so every home is within **3 parcels** of a road, in ~11 seconds. It's the headline
demonstration that the substrate work makes whole-settlement reblocking tractable. Complements
`examples/capetown-flagship/` (the dijkstra/mesh/arterial four-lens method comparison), which stays.

Feasibility was measured (`scratchpad/bigregion_spike.py`): the deepest screen seed grown to
`max_buildings=16000` yields **175 blocks / 17,403 parcels**, reblocked to depth 3 in **~11 s with
282 roads** (depth 4 = 113 roads/8 s; depth 2 = 1,089 roads/20 s — every target fully reached, the
road cap never binds). The greedy is fast at this scale — no perf pass needed.

## The region (auto-detected)

Screen `capetown_full` (`DenseCompactScreen`, default gates), take the **deepest seed**
(`ZAF.9.3.1_1_38528`), grow it with `DenseClusterRegionBuilder(max_buildings=16000)` into a
contiguous ~175-block neighborhood (~17,403 parcels). Auto-detected end-to-end — the generator
prints the actual seed + member count + parcel count; no hand-picked block list.

## Deliverables

`examples/clearance-flagship/` — a generator, committed PNGs, and a README, reproducing from
`capetown_full` (auto-downloaded to `~/.cache/reblock`, never committed), mirroring the flagship
pattern.

### 1. Hero — screen → grow → reblock (bespoke generator)

`examples/clearance-flagship/generate.py` (runnable `PYTHONPATH=. pixi run python …`):
- **`region_map.png`** — the two-panel map (`emit.region_map`): the city-wide locator (flagged
  blocks + a rectangle on the settlement) beside the ~175-block neighborhood.
- **`before.png`** — the settlement's status-quo access depth (starts at depth 13, dark interiors).
- **`after.png`** — reblocked with `ClearanceReblocker(depth_target=3)` (chord_diag): ~282 roads as
  buffered corridors + red displaced sites, the whole settlement brought to depth ≤ 3.

**Filename note (required):** the 175-block region's `block_id` is `"region:" + "+".join(ids)` ≈
3,000+ chars, so `reblock.run`'s `region:{id}_before.png` would exceed the 255-char filename limit.
The generator therefore renders via `reblock.render.{render_before, render_after}` and writes
**explicit** filenames (`before.png` etc.), exactly as the `clearance-repulsion` generator does —
NOT the raw `run` CLI.

### 2. Compare panels — both sweeps (via `method_sweep`, CLI)

Two `reblock.compare` runs on the same region (`max_blocks=1`), each emitting 4-lens AUC tables +
curves (compare truncates+hashes the long region label, so no filename issue):
- **depth_target sweep** `{2, 3, 4}`: `method_sweep={base: clearance, param: depth_target, values:
  [2, 3, 4]}` — the roads-vs-coverage tradeoff (the reason depth 3 is the sweet spot).
- **repulsion sweep** `{-3, 0, 3}`: with the base pinned to depth 3
  (`all_methods.clearance.depth_target=3`), `method_sweep={base: clearance, param: repulsion,
  values: [-3, 0, 3]}` — the directness↔displacement knob on massive real fabric.

The README documents both reproduce commands and commits the resulting `curve_{metric}_*.png` (or a
selected subset — access + directness at minimum) + the AUC numbers.

### 3. Scaling payoff (the thesis)

A short table/paragraph making the tractability concrete: 17,403 parcels reblocked in ~11 s on the
chord substrate's O(parcels) node count, contrasted with a fixed `res=1.5` grid's ≈ area/res² node
count over the settlement's bounding area (state the estimate) — i.e. *why* a whole settlement is
reblockable at all. Sourced from the sizing/big-region spikes.

## README

Follows `capetown-flagship/README.md`'s structure and tone: (1) screen the metro; (2) grow the
settlement (name the actual seed + member/parcel counts); (3) reblock to depth 3 (the hero
before/after + road count); (4) the depth_target tradeoff (from the sweep — why 3); (5) the
repulsion knob (from the sweep); (6) the scaling payoff. Every number from a real run — no
placeholders. Reproduce commands for the generator + both compares.

## Correctness / validation

- The hero reblock reaches `max_depth ≤ 3` on the region (assert in the generator; the spike
  confirms 282 roads / depth 3).
- `pixi run check` stays green (the generator is ruff-clean; `examples/` is outside the mypy/pytest
  scope — validate by running).
- Render legibility is the real risk at 17k parcels + 282 roads: the generator uses the current
  buffered-corridor render at dpi 300; eyeball the outputs and, only if needed, tune figure
  size / line widths in the generator (not the shared `render.py`).
- Numbers in the README come from the committed run, re-checked against a fresh run.

## Out of scope (still deferred)

- The broader examples/README rework (de-embedding the root README's galleries, pruning
  `convex-hull`, etc.) — its own follow-up, unchanged.
- Any clearance/greedy perf work — the spike showed it is unnecessary at this scale.
- Weighted footprints; the funnel/portal navmesh (separate projects).

## Decisions (confirmed)

- **Region:** deepest seed grown to `max_buildings=16000` (~175 blocks / ~17,403 parcels).
- **Hero depth_target = 3** (~282 roads, ~11 s; legible).
- **Both compare sweeps:** depth_target `{2,3,4}` and repulsion `{-3,0,3}`, 4-lens AUC via
  `method_sweep`.
- **Bespoke generator** for the hero (explicit filenames); compares via CLI. New
  `examples/clearance-flagship/`, complementing `capetown-flagship`.
