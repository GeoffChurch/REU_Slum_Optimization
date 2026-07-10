# Multi-block (region) reblocking — Design

**Status:** autonomous draft (built overnight; **decisions flagged for review**) · **Date:** 2026-07-10

Reblock a *region* — several adjacent blocks — **jointly**, so roads (especially arterials) can
span old block boundaries instead of stopping at them. The insight (from the roadmap
discussion): this is mostly *plumbing* — build one region-level `Block` and run the existing
methods on it. No new reblocking algorithm.

## The one load-bearing decision (⚠️ REVIEW THIS)

**Interior block-boundary roads are treated as removable / re-plannable; only the region's
outer perimeter is kept as `streets` (egress).** This is what makes it *joint* reblocking: if
we kept every existing inter-block road as a street, each block's interior would just route to
its own boundary independently and no cross-block road could ever be justified. Removing the
interior roads reframes the question as "if we re-planned this whole area's circulation from its
outer edges, what road network is best?" — which is the multi-block question worth asking, and
the one where region-spanning arterials earn their keep. It is an aggressive assumption (it
implies demolishing minor interior roads); a softer future variant would keep a *classified*
subset of major roads, but we have no major/minor classification today. **If you'd rather keep
all existing roads (independent per-block, no joint value), that's a one-line change.**

## `region_block(blocks) -> Block`

Given a list of `Block`s (a region):
- **parcels** — concatenate every block's parcels, re-`parcel_id`ed to be globally unique
  (`f"{block_id}:{parcel_id}"` or a running index), same CRS.
- **streets** — `unary_union([b.boundary for b in blocks]).boundary` — the perimeter of the
  unioned region land (outer ring + any holes). Interior shared edges vanish in the union, so
  this is exactly "outer perimeter only" (the decision above).
- **boundary** — `unary_union([b.boundary for b in blocks])` (region extent; a Polygon, or the
  convex hull if the union is a MultiPolygon, so the `Block.boundary: Polygon` contract holds).
- **crs** — the shared block CRS (assert all equal).
- **source_content_hash** — hash of the constituent blocks' `(source_content_hash, block_id)`
  identities, so region reblocks are derivation-cacheable like single blocks.

The result is an ordinary `Block`, so **every existing `Method` (`dijkstra`, `mesh`,
`greedy_arterial`, …) runs on it unchanged** and emits region-spanning roads.

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
  unique ids; streets = outer perimeter, interior edges dropped; CRS preserved; identity set).
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
