# Parametric Routing Substrate for the Clearance Reblocker — Design

**Status:** approved design (decisions confirmed 2026-07-12) · **Date:** 2026-07-12

**Goal:** Make the clearance reblocker's routing substrate a **pluggable Strategy** (grid | chord_diag |
theta_spanner | cdt_gap, or a provided graph), with **`chord_diag` the new default**. Three spike rounds
(`docs/superpowers/notes/2026-07-12-substrate-headtohead-findings.md`) concluded chord_diag
Pareto-dominates the shipped grid: better directness + lower displacement, **8.9–47.7× fewer nodes**
(node count ∝ parcels vs the grid's ∝ area/res²), robust (0 failures across connected substrates), and
the sparsity/speed gap *widens* with region size. This refactor makes that win the default and exposes
the substrate as a first-class knob, reusing the already-substrate-agnostic cost field + greedy loop.

## Architecture

The cost field and greedy loop only need node coords, weighted edges, and a network-seed rule — so we
lift the grid out behind a protocol.

```python
@dataclass(frozen=True)
class RoutingGraph:            # a built substrate
    pts:   NDArray[np.float64]    # (M, 2) node coords
    rows:  NDArray[np.int64]      # symmetric COO edge endpoints (both directions)
    cols:  NDArray[np.int64]
    edist: NDArray[np.float64]    # edge lengths
    net_tol: float                # a node within this of a street seeds the network + gates street-snap

class Substrate(Protocol):
    def build(self, block: Block) -> RoutingGraph: ...
    @property
    def identity(self) -> Hashable: ...   # folds into ClearanceReblocker.identity + the derive cache key
```

**New module `reblock/methods/substrates.py`** holds the protocol, `RoutingGraph`, a shared
`_boundary_vertices(block)` helper, and the implementations:

- **`GridSubstrate(res: float = 1.5)`** — wraps the existing `_build_grid(block.boundary, res)`;
  `net_tol = res * 1.5`. `identity = ("grid", res)`.
- **`ChordSubstrate`** (default) — nodes = parcel-boundary vertices (`_boundary_vertices`, reusing
  `reblock.methods.dijkstra._boundary_graph`); edges = the boundary segments **plus all within-cell
  diagonals** (every non-adjacent vertex pair in each parcel's exterior ring; parcels are ~convex so
  every diagonal is interior/valid). `net_tol = STREET_TOL`. `identity = ("chord_diag",)`.
- **`SpannerSubstrate(cones: int = 6)`** — a Θ-graph on the same boundary vertices (per node, connect the
  nearest node in each of `cones` angular cones). `identity = ("theta_spanner", cones)`.
- **`CdtSubstrate`** — the Delaunay triangulation of the boundary vertices
  (`scipy.spatial.Delaunay`), edges clipped to `block.boundary` (drop any edge whose segment leaves the
  block). `identity = ("cdt_gap",)`.
- **`PrebuiltSubstrate(graph: RoutingGraph)`** — the "provided graph" escape hatch; `build` returns it
  verbatim. `identity` = a content hash of the graph, or `None` (uncacheable) for an ad-hoc graph.

The three tessellation substrates share `_boundary_vertices` and differ only in edge selection.
`cdt_bldg` (nodes on building points) is **excluded** — the spike proved it degenerate: every
building-anchored node's nearest building is itself (distance 0), so `_node_clearance` floors to `ε`
everywhere and the repulsion knob becomes a no-op. Routing nodes must live in the free space.

`reblock/methods/clearance.py` keeps the cost field, the incremental relax, the greedy loop (now taking a
`RoutingGraph`), and `ClearanceReblocker`:

```python
@dataclass
class ClearanceReblocker:
    substrate: Substrate = field(default_factory=ChordSubstrate)   # chord_diag default
    repulsion: float = 0.0
    depth_target: int = 2
    max_roads: int = 400
    # `res` is REMOVED as a top-level param — it now lives on GridSubstrate.
```

## Cost field: 3-point edge sampling (the new standard)

`_edge_weights` changes from endpoint-average to **endpoint + midpoint + endpoint**:
`weight = edist · mean([cost(u), cost(mid), cost(v)])`, `cost(x) = (1 − t) + t / clearance(x)`, with
midpoint clearance from a `_node_clearance` call on the edge midpoints. Applied to **every** substrate —
one rule, no dual path. This is the ~5.4%-at-`s=+6` change to the grid; the **new grid+3-point output is
the golden reference** for tests (the old endpoint-only rule is deleted, not kept as a fallback). It
makes long chords honest (a chord skimming a building reads as expensive, not cheap-because-its-endpoints
-are-clear). A future refinement (subdivide very long edges into >3 samples) is out of scope; parcels are
small so 3-point suffices.

## Interface, identity, cache

- `identity = ("clearance", self.substrate.identity, repulsion, depth_target, max_roads)`.
- `proposal_id = f"clearance:{tag}:r{repulsion:g}:d{depth_target}:mr{max_roads}"`, where `tag` is the
  substrate's short name (`chord_diag`/`grid`/`theta_spanner`/`cdt_gap`) — so distinct substrates/configs
  get distinct `Proposal.identity` and never collide in the `access_after`/`geometric_after` caches.
- `propose` is side-effect-free (no global RNG); two calls → WKT-identical roads. The grid-unroutable
  (`-np.inf` pin + `grid_unreachable` count) and honest final-recompute logic are substrate-agnostic and
  carry over unchanged. The initial depth still seeds `unreached_depth = len(parcels) + 1` (the
  `_relax_depth` precondition).

## Config — a `conf/substrate/` group

Matching the repo's config-group pattern: `conf/substrate/{chord_diag,grid,theta_spanner,cdt_gap}.yaml`,
each a `_target_` for its `Substrate`. `conf/method/clearance.yaml` defaults `substrate` to `chord_diag`
(nested default); `method/substrate=grid` swaps it. In `conf/compare_config.yaml` `all_methods`, keep a
`clearance` entry (chord_diag) plus a `clearance_grid` reference entry so `compare` grades substrates
side-by-side on the four lenses; the rest are reachable via override.

## Examples

- **Regenerate `examples/clearance-repulsion/`** (the repulsion-knob demo) under the new default
  (chord_diag + 3-point) — new PNGs + refreshed README numbers, same auto-detected region + 5-repulsion
  sweep. This is a **migration necessity** (the committed output changes with the new default), not new
  example work. No other example uses `ClearanceReblocker`, so nothing else regenerates.

The broader examples/README rework — a massive-region **flagship**, de-embedding the root `README.md`'s
example galleries/recipes down to links into the already-links-only `examples/README.md`, and pruning the
set for redundancy — is a **separate follow-up** (see Out of scope), deliberately kept out of this
refactor.

## Migration (no-legacy — change, don't dual-path)

- Remove the top-level `res` param (migrates onto `GridSubstrate`); update `conf/method/clearance.yaml`.
- Re-pin the test goldens to the chord_diag default + 3-point: `proposal_id` format (now with a substrate
  tag), the column-block roads, and the `identity` tuple. The determinism / achieves-target / relax-
  precondition / weighted-`radii` tests are substrate-agnostic and stay.
- Add `reblock/methods/substrates.py` to `reblock.derive_graph._DERIVATION_MODULES` (busts the memoized
  `propose` cache on substrate-algorithm changes, like the method modules).
- Regenerate the `clearance-repulsion` gallery (the new default changes its committed output).

## Correctness strategy / testing

- **New `tests/methods/test_substrates.py`** — per substrate (grid, chord_diag, theta_spanner, cdt_gap):
  `build` returns a `RoutingGraph` with symmetric COO edges and a non-empty net seed on a small block
  with building points; the graph is connected enough that `_greedy_reblock` hits `depth_target` with
  `grid_unreachable = 0` on the real 1808 block **and** a synthetic deep block; `identity` is the expected
  tuple. `PrebuiltSubstrate` round-trips a hand-built graph.
- **3-point edge weights** — a unit test where a long edge crossing a building gets a higher weight under
  3-point than under endpoint-only (midpoint sees the building).
- **Substrate-agnostic greedy** — determinism (two `propose` → WKT-identical) and target achievement
  hold for each substrate; the golden column-block roads pin the chord_diag default.
- `mypy --strict`, ruff, `pixi run check` green.

## Out of scope (deferred)

- **Funnel / portal navmesh** — the any-angle substrate. The spike proved naive string-pulling breaks the
  greedy's vertex-*coincidence* connectivity (roads disconnect, coverage fails); doing it right needs a
  corridor/portal connectivity model — a separate project.
- **Examples & README rework (its own follow-up spec):** a massive-region **flagship**
  (`examples/clearance-flagship`, chord_diag on a ~3-block `capetown_full` region, with the reproduce
  recipe + the "chord makes this tractable; grid's node count ≈ area/res²" scaling payoff); de-embedding
  the root `README.md`'s ~150 lines of example galleries/recipes down to links into the already-links-only
  `examples/README.md`; and auditing the set for redundancy (e.g. `convex-hull` is a narrow region-builder
  demo overlapping `multi-block`; `single-block`/`detect-reblock`/`multi-block` share a pattern).
- **`cdt_bldg`** (building-point nodes) — degenerate (knob no-op), excluded.
- Subdividing very long edges for finer-than-3-point field sampling.

## Decisions (confirmed)

- **`chord_diag` is the default** substrate; grid retained as an opt-in fallback + field-fidelity
  reference.
- **3-point edge sampling everywhere** (one rule; the grid's high-repulsion output changes; the
  repulsion gallery regenerates).
- **Ship four connected substrates** (grid, chord_diag, theta_spanner, cdt_gap); exclude cdt_bldg + funnel.
- **Examples/README rework is deferred** to its own follow-up spec (the massive-region flagship, root-README
  de-embedding, and set audit) — this refactor only regenerates the `clearance-repulsion` gallery as a
  migration step.
