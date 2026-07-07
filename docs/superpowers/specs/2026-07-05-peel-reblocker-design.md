# reblock — Peel-reblocker (core): connected descent spine + connectivity-aware access

**Status:** draft for review · **Date:** 2026-07-05 · **Branch:** `peel-reblocker` (to be cut)

## Why this exists (and why it was redesigned)

The peel-reblocker is the second, deliberately different road-builder the multi-method design was
built for: it reuses the BFS access-depth peel to find targets and routes roads by **steepest
descent down the peel gradient** — deterministic and graph-free, unlike `topology`.

An earlier draft materialized roads as the **shared boundary each parcel shares with its descent
parent** ("party-wall seams"). A four-lens red-team plus a ground-truth spike **killed that
approach**: a party wall is shared between two *interior* parcels, so it can never touch the
street (spike: **0 of 329** road pieces reached the street on the real `cape_0` block), the seams
form disconnected iso-depth arcs rather than corridors, and the access metric only scored them as
"access" because it is **connectivity-blind**. Fixing the metric to be honest makes the seam
method score *zero improvement*; leaving it blind makes the peel-vs-topology comparison an artifact
of the instrument. Either way, seams are dead.

**This redesign, chosen after the red-team:**
1. Roads are a **connected descent spine** — centerline corridors that flow down the gradient and
   reach the street (spike: **k 6→1**, all roots street-connected on `cape_0`). They cross plot
   interiors, which is exactly what carving a reblocking road through a dense settlement is.
2. The peel metric becomes **connectivity-aware** — a road only grants access if it connects to the
   street — applied to *both* methods, so the head-to-head is fair and the metric is no longer
   gameable. This fixes a latent bug in the existing eval independent of this method.

**Core-first scope.** This slice is the spine method (**`full` only, a single `Proposal`, no
contract change**) + the connectivity-aware eval + diagnostics, validated end-to-end against
`topology` on a real block. The budget **sweep** (limits / depth / disjunctions) and the
`Method.propose → Iterable[Proposal]` migration are deferred to Slice 2 (§8), since the red-team
showed the DSL and a contract-wide migration should not be built on an unproven core.

---

## 1. Connectivity-aware access metric (`reblock.derive.access`)

Today `parcel_access_layers` seeds its BFS frontier from **any** parcel within `tol` of
`streets ∪ roads` (access.py, the `g.distance(street) <= tol` seed) with **no check that the road
connects to the street**. So a road segment floating in the block interior confers depth-1 on its
neighbours for free. That is a latent, gameable defect: *any* method could "win" access with
disconnected segments.

**Fix — seed only from the street-connected road network:**
- Let `street_geom = union(block.streets)` (always kept — it *is* the street).
- If `roads` is non-empty, split all street+road segments into **touch-connected components**
  (two segments joined when within `tol`; STRtree `dwithin` + union-find, `O(S log S)`), keep the
  components that contain a street segment, and set `seed_geom = union(kept segments)`. Roads
  transitively connected to a street are included; disconnected roads are excluded.
- Seed the peel frontier from parcels within `tol` of `seed_geom`, then BFS outward as today.
- With `roads=None`, `seed_geom == street_geom`, so **`access_before` is unchanged**, and any
  method whose roads *are* street-connected (topology by construction; the spine by construction)
  scores exactly as before. Only disconnected roads lose their unearned credit.

**Risk to verify early (shared-code change).** `topology`'s `proposal.roads` are the *newly added
interior* roads (it excludes the initial street edges as `initial`), so the connectivity-aware seed
credits them only if they geometrically touch `block.streets` within `tol` at their boundary
attachment. This is expected (build_all_roads grows the network outward from the marked street
edges), but it must be **confirmed before committing to the metric change** — the "topology
unchanged under the new metric" test (§6) is the gate. If topology's interior roads do *not* reach
`block.streets` geometrically, the metric must seed from roads whose component reaches a
street-*adjacent parcel* (not the street line itself), and both methods rescored — resolve this in
the first implementation task, not late.

**Diagnostics.** `KComplexityEval` additionally emits, per proposal:
`n_road_components` and `connected_road_frac` = (road length in street-connected components) /
(total road length). This surfaces disconnection in the scorecard so it can never hide again
(≈1.0 for topology and the spine; ≈0 for the dead seam approach).

`tol` is threaded explicitly: `parcel_access_layers(block, roads, *, tol=STREET_TOL)`; callers that
use a non-default `tol` (the reblocker) pass the same value used to build adjacency (§3).

## 2. The connected descent spine (`reblock.methods.peel`)

Work on parcels (nodes) and shared-boundary adjacency (edges); never build topology's planar
edge-graph.

**Setup.** `layers = parcel_access_layers(block, None, tol=self.tol)` — depth per parcel, indexed by
`parcel_id`; `neighbors = parcel_adjacency(geoms, self.tol)` (§4).

**Descent forest.** For each interior parcel `P` (`layer ≥ 2`), its **descent parent** is the
adjacent parcel `Q` with `layer[Q] == layer[P] − 1`, chosen as **min `parcel_id`** among such `Q`
(all id-keyed — see §3). Such a `Q` always exists: the BFS peel gave `P` its layer from an `L−1`
neighbour. Parents form a forest rooted at the street-adjacent (`layer 1`) parcels.

**Spine geometry (connected, reaches the street).**
- For each interior `P`: a **descent link** `LineString(centroid(P), centroid(parent(P)))` — it
  crosses the shared boundary at the "doorway", chaining P→parent→…→root into a connected corridor.
- For each **root** (a `layer-1` parcel that is some parcel's descent parent): a **street connector**
  `LineString(centroid(root), nearest_point_on(block.streets))`, so the corridor physically reaches
  the public edge.
- `roads = GeoDataFrame` of all links + connectors — clean 2-point `LineString`s (the spine sidesteps
  the degraded polygon/`MultiLineString` "shared boundaries" the seam approach hit on real data).

Serving every interior parcel yields a spine that reaches k=1 under the connectivity-aware metric
(spike-verified on `cape_0`). The naive spine is length-inefficient (spike: ~3.3 km vs topology's
~1.6 km at equal k=1) because each root gets its own street stub; **trunk-merging is an explicit
Slice-2/roadmap optimization (§8)** — this slice proves *correct + honest + connected*, not *shortest*.

**Determinism.** All tie-breaks are by `parcel_id`; **no RNG** (topology seeds numpy's global RNG).
Identical block → identical roads.

**Unreachable parcels.** Parcels in a component disconnected from every street share one flat `far`
layer with no gradient, so no descent parent exists; they are skipped and counted in
`Proposal.params["unreachable"]`, honestly leaving their depth in `k_after`.

**Output.** A single `Proposal(block_id, crs, roads=<spine>, edges=None, proposal_id="peel",
method="peel", params={"unreachable": n})`. `propose` returns one `Proposal` (unchanged `Method`
contract — the migration lands in Slice 2). `render_after` and the connectivity-aware
`KComplexityEval` consume `.roads` unchanged.

## 3. Correctness fixes (from the red-team)

- **Single `tol`.** The peel's internal adjacency, `parcel_adjacency`, and every wave/after recompute
  must use one `tol`; the parent-existence invariant holds only when the reblocker's adjacency
  matches the peel's. Thread `self.tol` through all three.
- **Id-space, not row position.** `parcel_adjacency` returns neighbour sets keyed by **row
  position**, while `layers` is keyed by **`parcel_id`**. The descent must map position→`parcel_id`
  for both the layer lookup and the `min` tie-break; using a raw position as a min key makes the
  chosen parent depend on row order and silently breaks determinism. A determinism test uses a
  **shuffled-rows / `parcel_id ≠ position`** fixture and asserts identical road WKT.
- **Degenerate guards.** `parcel_id` uniqueness is assumed (documented as a precondition, asserted in
  the reblocker); an all-streetless block (every layer collapses to a false depth-1) is out of scope
  but guarded with a clear error rather than silent no-op.

## 4. Refactor: shared parcel adjacency (`reblock.derive.adjacency`)

`_adjacency`/`_shared_len` live privately in `derive/access.py`; the reblocker needs the same
adjacency. Move them (one-way, no duplication) to `src/reblock/derive/adjacency.py`:

```python
def parcel_adjacency(geoms: list[BaseGeometry], tol: float) -> list[set[int]]:
    """Neighbour sets (positional) — adjacent iff a positive-length shared boundary
    within tol, via the GEOS-robust _shared_len (snap-then-intersect with a
    make_valid direct-intersection fallback on a side-location conflict)."""
```

The spine uses **centroids**, so it needs only neighbour sets, **not** the shared-boundary geometry —
so `parcel_adjacency` returns just `list[set[int]]` (no geometry dict; the "free to compute but not
free to hold" tension the red-team flagged does not arise). `parcel_access_layers` imports it; the
snap-crash regression test moves to `tests/derive/test_adjacency.py`. `_adjacency`/`_shared_len` are
deleted from `access.py`.

## 5. Files & wiring

- **New** `src/reblock/methods/peel.py` — `PeelReblocker(tol=STREET_TOL)`, `propose(block,
  prior=None) -> Proposal` (`prior` accepted, unused).
- **New** `src/reblock/derive/adjacency.py` — §4.
- **New** `conf/method/peel.yaml` — `_target_: reblock.methods.peel.PeelReblocker`.
- **Modify** `src/reblock/derive/access.py` — connectivity-aware seeding (§1); import
  `parcel_adjacency`; delete moved helpers.
- **Modify** `src/reblock/eval/kcomplexity.py` — emit `n_road_components`, `connected_road_frac`.
- Contract, `run.py`, `topology.py` **unchanged** this slice.

## 6. Testing highlights

- **Connectivity-aware metric (ground-truth guard):** a hand-built **disconnected** road set scores
  **no better** than no roads (k unchanged), while a **street-connected** road set of equal length
  reduces k — proving the metric rewards connectivity and is not gameable. `access_before` (roads
  `None`) is byte-identical before/after the change.
- **Spine reaches k=1 connected:** 5×5 grid (centre `layer 3`) → spine → `k_after == 1` and every
  road component touches the street; `connected_road_frac == 1.0`.
- **Determinism:** a **shuffled-rows** fixture (`parcel_id ≠ position`) yields identical road WKT.
- **Unreachable island:** skipped, counted in `params["unreachable"]`, left at depth in `k_after`.
- **topology unchanged under the new metric:** topology's k and `connected_road_frac ≈ 1.0` on a
  fixture block match the pre-change numbers (its network is street-connected by construction).
- **adjacency refactor:** migrated snap-crash GEOS-robustness regression (fixture still raises under
  raw `snap`, adjacency recovered) against `parcel_adjacency`.
- **integration / head-to-head:** `topology` vs `peel` on a real block (`cape_0`) both score under the
  connectivity-aware eval and render; peel reaches k=1 connected (honestly longer than topology —
  the fair result).

## 7. Complexity

`n` parcels, planar tiling so `m = O(n)` adjacency edges, `S = O(n)` spine segments. Adjacency
(STRtree) + peel `O(n log n)`; parent forest `O(n)`; spine materialization `O(n)`; connectivity-aware
seeding (segment components) `O(S log S) = O(n log n)`. **Total `O(n log n)`**, no planar noding —
the graph-free substrate that is the method's whole point.

## 8. Roadmap — Slice 2 and beyond

- **Slice 2 — the budget sweep + contract migration.** `Method.propose → Iterable[Proposal]`
  (unifying with `RegionMethod`; `topology` yields one). Atomic limits `length`/`depth`/`full`;
  `any_of` disjunctions (earliest stop on the monotonic deepest-first trajectory); one descent
  truncated per limit → a family of proposals, one after-map + scorecard row each. Red-team
  requirements to fold in then: `depth` uses **downward-closed / covering** selection (serve whole
  root-anchored subtrees, not a horizontal layer cut — a layer cut over-serves and lands below the
  target); `Metrics` gains `proposal_id` (family members must be labelable); OmegaConf-safe limit
  parsing (`DictConfig` is not a `dict`); `proposal_id` includes `tol` + a duplicate-limit guard;
  note that `WeakDualKEval` cannot score peel roads (endpoint matching finds no edges → silent
  `delta_k=0`).
- **Trunk-merging optimization** — merge shared descent corridors and reuse street stubs so the spine
  is length-competitive with topology (the naive spine over-connects). This is where peel can *win*
  the head-to-head rather than just tie on access.
- **Boundary-hugging spine** (optional) — route links through shared-boundary midpoints instead of
  centroids to reduce plot-crossing, if realism warrants.
- Unchanged: kblock source (Slice 2 of the broader project), displacement eval, water method,
  regional screening.

## Migration note (owner's "migrate, don't accommodate")

`_adjacency`/`_shared_len` **move** to `derive.adjacency` (not duplicated) and are deleted from
`access.py`; the regression test moves with them. `parcel_access_layers` seeding changes in place to
be connectivity-aware — the old blind-seed path is replaced, not kept alongside. No back-compat
branches. The `Method.propose` signature is left untouched this slice and migrated wholesale in
Slice 2 (no interim dual return type).
