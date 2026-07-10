# Multi-block (region) reblocking — Design

**Status:** built + reviewed; model revised to **existing-egress** (2026-07-10, owner decision) · **Date:** 2026-07-10

Reblock a *region* — several adjacent blocks — **jointly**, so roads (especially arterials) can
span old block boundaries instead of stopping at them. The insight (from the roadmap
discussion): this is mostly *plumbing* — build one region-level `Block` and run the existing
methods on it. No new reblocking algorithm.

## The model (RESOLVED — existing egress)

**The existing inter-block streets are existing egress, not part of the intervention.** A region
is just a single `Block` whose streets are the full existing road network (outer perimeter +
inter-block streets), reblocked exactly like any single block. (Owner decision, 2026-07-10: this
*replaced* an earlier "seed / count existing roads as first-added" model, which produced a
misleading 'before' — a deep-access core measured only to the outer perimeter, as if the real
inter-block streets didn't exist. No back-compat: the seed/perimeter/eval-swap machinery was
deleted.) Concretely:

- **Egress = every existing street** (outer perimeter AND inter-block). A parcel already served by
  an inter-block street is shallow in the 'before' — the physical status quo.
- **The intervention = only the method's added roads.** The cost-benefit grades those new roads
  against the existing-network baseline, so a method is credited for the reachability it *adds*,
  not for pre-existing pavement.
- **The method extends the network:** it routes parcels on the existing streets and adds
  *complementary* roads (arterials for navigability, spurs for any still-deep pockets).

## API (`src/reblock/region.py`)

Common pieces: **parcels** = every block's parcels concatenated, re-`parcel_id`ed to a unique
running range, same CRS; **boundary** = `unary_union([b.boundary for b in blocks])` (a Polygon,
or its convex hull if the union is a MultiPolygon); **crs** = the shared CRS (asserted equal);
**source_content_hash** = a deterministic hash of the sorted constituent identities (or `""` if
any is uncacheable).

**No method changes needed** — the region is a single `Block` and the existing single-block
reblock path handles it:

- **`region_block(blocks) -> Block`** — the block a method reblocks. `streets` = **union of every
  block's existing streets** (outer perimeter + inter-block = the full existing road network).
  Routing on this means already-served parcels stay served, so the method adds only *complementary*
  roads; and it is the egress the evals score the method's added roads against.
- **`region_reblock(blocks, method, evals) -> Result`** — reblock the region-block exactly like a
  single block: `rb = region_block(blocks)`; `proposal = method.propose(rb)`; score `proposal` on
  `rb` with each eval → `Result`. No seed/perimeter split and no eval-swap: the existing
  inter-block streets are egress in `rb.streets`, and only the method's added roads are the
  intervention on the cost-benefit curve.

## Region selection

A region is a set of block_ids grouped from a `Source`. v1: caller supplies the group (a list
of block_ids known to be adjacent). A `region_reblock(source, block_ids, method, evals) ->
Result` helper builds the region-block from those and reblocks it. (A spatial "auto-group
adjacent blocks" selector and a Hydra CLI entrypoint are v2 polish — not needed to test the
hypothesis.)

## The hypothesis this exists to test

*Greedy arterial will do well for multi-block — likely better than a fancy OT/transfer reblocker.*
So the deliverable includes a **head-to-head on a real region**: arterial vs dijkstra vs mesh,
graded on the three lenses. Expected (and to be reported): arterial leads directness/efficiency
by an even wider margin than single-block, because a region has more room for long
cross-block through-roads; dijkstra stays access/speed-strong. This result *informs* whether the
transfer idea (thread 3) is worth building at all.

## Scope / testing

- `region_block` builder — unit tests: N blocks → one valid `Block` (parcels = Σ parcels,
  unique ids; streets = full existing network, incl. the inter-block shared edges; CRS preserved).
- **Cross-block roads** — on a region of ≥2 adjacent blocks, a method produces at least one road
  that crosses an original block-boundary line (proves *joint*, not per-block).
- **Hypothesis** — on a real adjacent-block region, `greedy_arterial` directness/efficiency AUC
  > dijkstra's (record the numbers).
- **Determinism / street-connectivity** — buildable region roads are street-connected; runs are
  deterministic.

Adjacency: the tests need genuinely adjacent blocks; the implementer finds ≥2 touching DJI
blocks (share a boundary segment) or constructs a synthetic 2×1 block region.

## Out of scope (v2+)

- Auto-selection of adjacent block groups (spatial clustering) + a Hydra `reblock.region` CLI.
- Keeping a *classified* subset of existing roads (needs a major/minor road classification).
- Region-scale performance (arterial is already slow per block; a large region is slower — cap
  region size in v1, and the north-star resistance-marginal greedy is the real fix).
- The segment-group egress model (which perimeter segments are live vs inert) — a whole region's
  perimeter is where this matters most; cross-references docs/metrics-north-star.md.
