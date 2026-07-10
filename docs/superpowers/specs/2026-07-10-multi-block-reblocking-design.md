# Multi-block (region) reblocking — Design

**Status:** autonomous draft (built overnight; **decisions flagged for review**) · **Date:** 2026-07-10

Reblock a *region* — several adjacent blocks — **jointly**, so roads (especially arterials) can
span old block boundaries instead of stopping at them. The insight (from the roadmap
discussion): this is mostly *plumbing* — build one region-level `Block` and run the existing
methods on it. No new reblocking algorithm.

## The model (RESOLVED — seed / augment)

**Existing inter-block roads are kept as a pre-added "seed" road network that the methods
*extend* — treated the same as roads we add, as if added first — not demolished.** (Owner
decision, 2026-07-10; the clean-slate "drop interior roads and re-plan from the perimeter"
alternative was rejected.) Concretely:

- **True egress = the region's outer perimeter** (its link to the wider city). This is what the
  cost-benefit measures reachability *to*.
- **The inter-block roads are the first roads of the proposal** — already built, but **counted
  as road** in the cost-benefit (they are pavement), placed first in budget/drainage order.
- **The method extends the seed:** it routes parcels on the existing network (perimeter + seed)
  and adds *complementary* roads (arterials for navigability, spurs for any still-deep pockets),
  rather than re-deriving what is already there.
- **Access/directness are measured on the full network (seed + added) against the perimeter
  egress** — so a parcel reaches the city *through* the whole network.

This is the honest model (build on what exists; count existing pavement fairly), and its
egress-vs-internal-network split is the *same* distinction the north-star / segment-group egress
model needs — the two converge, a good sign it's the right cut.

## API (`src/reblock/region.py`)

Common pieces: **parcels** = every block's parcels concatenated, re-`parcel_id`ed to a unique
running range, same CRS; **boundary** = `unary_union([b.boundary for b in blocks])` (a Polygon,
or its convex hull if the union is a MultiPolygon); **crs** = the shared CRS (asserted equal);
**source_content_hash** = a deterministic hash of the sorted constituent identities (or `""` if
any is uncacheable).

The seed model is realized with three geometries and one orchestrator — **no method changes
needed** (the trick is that the method routes on the *full existing network* while the
evaluation counts the interior roads against the *perimeter* egress):

- **`region_block(blocks) -> Block`** — the block a method reblocks. `streets` = **union of every
  block's existing streets** (perimeter + inter-block = the full existing road network). Routing
  on this means seed-adjacent parcels are already served, so the method adds only *complementary*
  roads. *(This supersedes Task 1's perimeter-only `region_block`; Task 1's perimeter computation
  moves to `region_perimeter`.)*
- **`region_perimeter(blocks) -> GeoDataFrame`** — the outer perimeter lines
  (`unary_union([b.boundary for b in blocks]).boundary`), used as the **eval egress**.
- **`region_seed_roads(blocks) -> GeoDataFrame`** — the interior existing roads = all existing
  streets minus the perimeter (`difference` within a small buffer). These are the **counted
  seed**, emitted first.
- **`region_reblock(blocks, method, evals) -> Result`** —
  1. `rb = region_block(blocks)` (streets = full existing network);
  2. `added = method.propose(rb).roads` (the method extends the seed);
  3. `full = concat([seed, added])` with `seed = region_seed_roads(blocks)` marked highest-drain
     (added first);
  4. `eval_block = replace(rb, streets=region_perimeter(blocks))` (egress = perimeter only);
  5. score `full` on `eval_block` with each eval → `Result`.

So the method sees the seed as existing (routes/extends), and the cost-benefit counts the seed
as the first-added road against the true perimeter egress — exactly "treated as roads we add,
added first."

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
