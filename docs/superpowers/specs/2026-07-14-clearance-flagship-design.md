# Clearance Flagship — Design

**Status:** realized 2026-07-14 (design evolved during build — see note) · **Date:** 2026-07-14

**Goal:** A whole-settlement flagship example — reblock the metro's **deepest informal settlement
core** (**23 blocks, ~10,700 homes, up to 24 parcels deep**) with the clearance reblocker
(chord_diag default) so every home is within **3 parcels** of a road, in ~11 seconds — delivered as
a single `reblock.run` **CLI one-liner**. The headline demonstration that the substrate + screen
work make whole-settlement reblocking tractable. Complements `examples/capetown-flagship/` (the
dijkstra/mesh/arterial four-lens method comparison), which stays.

**Design note (what changed during the build):** the original plan targeted a ~175-block region
around seed `ZAF.9.3.1_1_38528` (depth 13). Two changes reshaped it: (1) the screen's cheap gate
migrated from building density to the depth proxy `√(n·A)/P`, which surfaced a genuinely *deeper*
seed, `ZAF.9.3.1_1_5810` (depth **24**), that the old density gate excluded; (2) the region builder
now grows by that same proxy, so it follows the deep informal fabric. Growing `5810` to
`max_buildings=3000` isolates a clean 23-block / 10,706-parcel deep core (the full `16000` budget
spilled into shallow adjacent housing). Realized: **depth 24 → 3, ~13,700 m of road, 10.6 s**.

**Delivery — a CLI one-liner (no bespoke generator):** the original plan used a bespoke `generate.py`
purely to dodge the ~3,000-char region `block_id` overflowing output filenames (255-char limit) and
stretching plot titles. That plumbing is now fixed centrally (`render.short_label` / `title_label`,
reused by `run` + `compare`), so the flagship reproduces from a plain `reblock.run` command and the
generator was deleted. Screen locator + region map come from `region_map.enabled=true`
(`screen.png` + `region.png`); before/after from `render.enabled=true`.

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

## Decisions (as realized)

- **Region:** deepest seed `ZAF.9.3.1_1_5810` (depth 24) grown by the depth proxy to
  `max_buildings=3000` — a **23-block / 10,706-parcel deep core**.
- **Hero depth_target = 3** (304 roads, 13,699 m, 959 displaced, depth 24 → 3, ~11 s).
- **Both compare sweeps:** depth_target `{2,3,4}` and repulsion `{-3,0,3}`, 4-lens AUC via
  `method_sweep` (`max_roads` raised so the target binds, not the road cap).
- **CLI one-liner** for the hero (no bespoke generator — the filename/title plumbing fix removed the
  need); compares via CLI. New `examples/clearance-flagship/`, complementing `capetown-flagship`.
