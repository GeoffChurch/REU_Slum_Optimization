# reblock — Layered architecture + Slice 1 (walking skeleton)

**Status:** draft for review · **Date:** 2026-07-03

## Goal

reblock "adds roads to maps via pluggable methods, then evaluates them." The system is a
set of statically-typed layers wired by **Hydra structured configs**, checked under
`mypy --strict`. Two submodules are fixed points:

- `ext/topology` — the classic Brelsford/Mansueto parcel-graph reblocker (**owned** — we
  co-develop it). Provides **the first method** *and* the canonical **k-complexity** metric
  (the weak-dual nesting depth the Brelsford work defines; `stacked_duals`).
- `ext/kblock` — the Mansueto block-delineation pipeline. A **data source** (blocks, streets,
  population); **not** used for eval algorithms (its `compute_k` is a reimplementation of
  topology's original — we use topology's).

This spec records the durable architecture (shared by all slices) and then scopes **Slice 1**,
a walking skeleton that proves every interface end-to-end on data already in the repo, with
zero external downloads.

---

## Durable architecture decisions

Hold across all slices; not re-litigated per slice.

1. **Canonical waist.** One extensible `Block` type sits at the center. Sources adapt *into*
   it; methods and evals consume *from* it. This gives **N+M+K** adapters (one per component),
   not **N×M** pairwise transforms — a new source works with every method for one adapter.
   (The hub-and-spoke that pandoc/LLVM converge on; an "adapter mesh" re-derives the same
   waist once you compose transforms.)

2. **Two-level unit: `Region` ⊃ `Block`.** The atomic unit is a single street-bounded
   `Block`. A `Region` is a collection of `Block`s plus **regional-scale layers** (amenity POIs,
   a routable network) and metadata. A `Source` yields a `Region`; iterating it yields
   `Block`s (lazily, at continental scale). Any computation that **spans blocks** (e.g.
   shortest-path-to-amenities) runs at the `Region` level and **collapses to per-block
   scalars** written onto each `Block`; everything downstream stays per-block and
   embarrassingly parallel.

3. **Three component roles, one direction of flow.**
   - **`Screen`** (targeting): `Region -> per-block priority`. Scores the *status quo* to decide
     *where* to intervene. No `Proposal`.
   - **`Method`**: `Block -> Proposal`. Adds interventions (roads, water, …).
   - **`Eval`**: `(Block, Proposal) -> Metrics`. Scores an intervention.

4. **Symmetric, explicit, named layers on every waist object.** `Region` (amenity layers),
   `Block` (input layers), and `Proposal` (intervention layers) each carry a small set of
   **explicit named optional** geometry layers — *no open layer map*. New layer types (e.g.
   topography) are added as named fields when introduced (migrate, don't accommodate). Only
   `attrs` (scalar metadata: population totals, accessibility, admin codes, provenance)
   remains an open map.

5. **Extensibility = new named layers, not new formats.** Future inputs (water, topography,
   land use) and interventions (water points/mains) are named layers on the relevant waist
   object, not new top-level formats and not an N×M explosion.

6. **Method-specific shapes = typed one-way derivations off the waist.** `Block ->
   PlanarParcelGraph` (topology's view), `Region -> AccessNetwork` (routing view). Spokes, not
   a mesh: we only ever write `Block -> X` / `Region -> X`, never `SourceY -> X`.

7. **Data-side tessellation.** `Block.parcels` is always space-filling — real cadastre when a
   source has it (topology's datasets do), else a momepy enclosed tessellation of footprints
   (as kblock computes internally). Methods never tessellate.

8. **Consume-first for kblock.** The data layer *reads kblock's published GeoParquet* via one
   adapter. kblock's batch/legacy stack (pygeos, dask, SLURM, broken `__init__`) stays out of
   reblock. (Vendoring specific delineation functions is deferred until on-demand delineation
   is needed.)

9. **CRS policy.** In-memory `Block`s are reprojected to a **local UTM zone** (accurate
   areas/lengths; topology requires metric regardless). Adapters reproject on load and record
   `Block.crs`; anything reblock writes records its own CRS.

10. **topology is a co-developed, typed dependency (owned).** We own topology, so rather than
    quarantine it we make it a properly-packaged, `py.typed` dependency and **add** a
    `Block`/`Proposal`-native API *alongside* its existing shapefile-in / GeoJSON-out
    interfaces. Those stay — they're legitimate standalone options on the same `MyGraph`
    engine, not compat shims (the shapefile loader even powers the regression oracle). Typing
    scope (owner choice): the **public surface reblock imports** (`MyGraph`/`MyNode`/`MyEdge`/
    `MyFace`, `graphFromMyFaces`, `define_roads`, `define_interior_parcels`, `build_all_roads`,
    `road_length`, `myedges`, and a new `k_complexity`); internal algorithm typing deferred.
    The `Block`→graph adapter maps **`Block.streets` → the graph's initial road edges** (the
    data layer owns "what a street is"; the method needs only *an* initial road set). For
    Phule Nagar `Block.streets` = boundary, matching topology's own baseline (and preserving
    the regression oracle). Known wart, deferred: topology's GeoJSON emits `"true"`/`"false"`
    as strings, not JSON booleans.

11. **Compute, don't consume — and reuse the algorithm on both ends.** Metrics like
    k-complexity and amenity-accessibility must be *computed* (kblock's parquet only holds
    *status-quo* values; evals need *post-intervention* values). The same algorithm is reused
    by a `Screen` (status quo → targeting) and by an `Eval` (post-intervention → improvement).
    k-complexity is **topology's own** metric — the weak-dual nesting depth
    (`stacked_duals`/`form_equivalence_classes`), which the dual excludes road edges from, so
    it is road-relative and drops as roads are added. We expose it as a typed
    `k_complexity(graph) -> int` and import it in the eval; we do **not** port kblock's
    `compute_k`. The amenity travel-time model is net-new (kblock has none).

---

## Core contracts (`reblock.contracts`) — the spine

The most consequential deliverable of Slice 1; every future slice depends on these types.

```python
@dataclass(frozen=True)
class Region:
    region_id: str
    crs: CRS
    blocks: Iterable[Block]                   # may be lazy at continental scale
    # regional context / amenity layers (explicit, named, optional):
    roads: GeoDataFrame | None = None
    water: GeoDataFrame | None = None
    food: GeoDataFrame | None = None
    healthcare: GeoDataFrame | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class Block:
    block_id: str
    crs: CRS                                  # projected, metric (local UTM)
    boundary: Polygon                         # the enclosing street loop
    parcels: GeoDataFrame                     # space-filling; cols: [parcel_id, population?, geometry]
    streets: GeoDataFrame                     # existing roads; cols: [geometry, highway?]
    buildings: GeoDataFrame | None = None     # raw footprints; cols: [population?, geometry]
    water: GeoDataFrame | None = None         # existing water features (rivers/standpipes)
    barriers: GeoDataFrame | None = None      # impassable edges
    attrs: Mapping[str, object] = field(default_factory=dict)  # block scalars: population total, accessibility, admin, provenance

@dataclass(frozen=True)
class Proposal:
    block_id: str
    crs: CRS
    roads: GeoDataFrame | None = None         # LineStrings  (topology fills this)
    water_points: GeoDataFrame | None = None  # Points: wells/standpipes
    water_mains: GeoDataFrame | None = None    # LineStrings: pipe corridors
    edges: GeoDataFrame | None = None         # method's tagged debug graph (road/interior/barrier)
    method: str = ""
    params: Mapping[str, object] = field(default_factory=dict)

@dataclass(frozen=True)
class Metrics:
    block_id: str
    method: str
    eval: str
    values: Mapping[str, float]               # e.g. k_before, k_after, delta_k, displaced_people, added_road_length_m

class Source(Protocol):
    def region(self) -> Region: ...

class Screen(Protocol):                        # targeting: status-quo → where to intervene
    def rank(self, region: Region) -> Mapping[str, float]: ...   # block_id -> priority

class Method(Protocol):
    def propose(self, block: Block) -> Proposal: ...

class Eval(Protocol):
    def score(self, block: Block, proposal: Proposal) -> Metrics: ...
```

**Typing strategy.** Geometry layers are `GeoDataFrame`s (the ecosystem's native currency;
kblock wants `GeoSeries`, tessellation wants `GeoDataFrame`, so a hand-rolled container would
convert at every boundary). geopandas' type coverage is partial, so:

- Scalar fields fully typed; layer *column* contracts validated at construction in
  `__post_init__` (required columns present, CRS present and projected, geometry non-empty) —
  runtime validation compensating for static gaps, and closing the implicit-column-contract
  hole that makes kblock hard to consume.
- geopandas typing gaps are absorbed with localized ignores *inside adapters*, never in
  contract signatures.
- (Candidate later: `pandera` GeoDataFrame schemas for statically- and runtime-checked layer
  schemas. Deferred to keep Slice 1 lean.)

**Derivations** (`reblock.derive`): `to_parcel_graph(block) -> PlanarParcelGraph` (Slice 1),
`to_access_network(region) -> AccessNetwork` (screening slice). A component calls the
derivation it needs directly; a registry/resolver is deferred (YAGNI until multiple exist).

---

## Slice 1 — walking skeleton

**Thesis:** prove the whole spine + method + eval on **topology's in-repo Phule Nagar
parcels** before any external data. Phule Nagar is already space-filling cadastre — no
tessellation, no downloads.

### Components

| Module | Responsibility |
|---|---|
| `reblock.contracts` | the types above |
| `reblock.data.shapefile` | `ShapefileSource`: `gpd.read_file` topology's `examples/data/phule_nagar_v6.shp`, group parcels into blocks by **adjacency (connected components)**, and return a **`Region`** whose `blocks` are one `Block` per component (`parcels` = the component's polygons, `boundary` = its dissolved outer ring, reprojected to local UTM). Component-splitting is a small geopandas-native step the Source owns. **Not** topology's pyshp path. |
| `reblock.derive.parcel_graph` | `to_parcel_graph(block) -> PlanarParcelGraph` — the view topology needs |
| `reblock.methods.topology` | port topology's I/O to consume `PlanarParcelGraph` / emit `Proposal` (fills `roads` + tagged `edges`); wrap as a `Method` |
| `reblock.eval.kcomplexity` | `Eval` wrapping topology's exposed `k_complexity(graph)`; computes `k_before` / `k_after` / `delta_k` + `added_road_length_m` |
| `reblock.run` | Hydra entrypoint: `data=phule method=topology eval=kcomplexity` |

### Data flow

```
ShapefileSource.region()                              # a Region; blocks = one per connected component
  → for each Block(parcels, boundary, streets=[boundary], crs=local UTM):
      → to_parcel_graph → PlanarParcelGraph
        → topology.propose → Proposal(roads, edges tagged road/interior/barrier)
          → kcomplexity.score(block, proposal) → Metrics{k_before, k_after, delta_k, added_road_length_m}
            → run: print/persist per-block Metrics (+ optional GeoJSON/PNG debug render)
```

For the very first skeleton the runner may restrict to a single chosen component, then
generalize to all components once the interfaces are proven.

### Eval definition

- k-complexity = topology's **weak-dual nesting depth** (`stacked_duals`), exposed as a typed
  `k_complexity(graph) -> int`. The weak dual excludes road edges, so k is road-relative.
- `to_parcel_graph(block)` builds the graph; the eval marks `Block.streets` as the initial
  road edges → reads `k_before`; then marks the `Proposal.roads` edges too → reads `k_after`.
  For Phule Nagar `Block.streets` = boundary (topology's baseline).
- Everything runs in the **Block's local UTM CRS** — no kblock, no pygeos, no `srid=3395`.

### Out of scope for Slice 1

kblock source · population/displacement · water · regional screening · footprint tessellation
· batch/scale + a real result store · derivation registry. (See roadmap.)

---

## Error handling

- Adapters validate at the boundary and **fail fast** with typed errors: CRS present and
  projected, required columns present, geometries non-empty/valid.
- topology's 2-decimal rounding and re-zero stay *inside* its adapter; `Block` coordinates are
  never rounded and remain in the Block's CRS.

## Testing

- **Contract tests:** construction + `__post_init__` validation (missing column / geographic
  CRS / empty geometry rejected).
- **Port oracle:** the `Block`→graph derivation reproduces topology's *native* planar graph
  (same inner-face count) built from the same shapefile — deterministic, so it isolates the
  I/O port from the stochastic road-builder. Method correctness is covered separately by an
  "all interior parcels resolved" invariant.
- **Eval sanity:** `k_after ≤ k_before`, `delta_k ≥ 0`, `added_road_length_m > 0` when roads added.
- **End-to-end:** `reblock.run data=phule method=topology eval=kcomplexity` emits `Metrics`;
  passes `pixi run check` (ruff + mypy --strict + pytest).

---

## Roadmap (each its own spec → plan → implementation; order flexible)

- **Slice 2 — kblock Source + population.** `kblock_parquet -> Region` of `Block`s: footprint
  → parcel tessellation, per-block population (area-allocated to parcels), local-UTM
  reprojection. Unlocks real-world blocks and the population attribute.
- **Slice 3 — `DisplacementEval` (cross-cutting).** `displaced_people` = area-allocated
  population of parcels/buildings under the **buffered** intervention footprint (a
  `corridor_width` param). Applies to *any* destructive layer — roads today, water mains
  later — so it's one eval across method types.
- **Slice 4 — Water.** A `water-siting` `Method` that fills `Proposal.water_points` /
  `water_mains`; a `WaterAccessEval`; existing water as a `Block.water` / `Region.water` input
  layer. Methods compose (roads then water into one `Proposal`).
- **Slice 5 — Regional screening / targeting.** Amenity ingest (OSM `amenity`/`healthcare`/
  `shop` POIs — *not* in kblock's outputs) into `Region.{roads,water,food,healthcare}`;
  `to_access_network(region)`; a `Screen` computing shortest-path-to-amenity → per-block
  priority (slum/benefit detection) at continental scale; selection feeds the per-block loop.
  The same travel-time model powers a post-intervention `AccessEval`.
- **Cross-cutting — batch/scale + result store.** Threaded in as slices need it; the
  per-block collapse (decision 2) keeps it embarrassingly parallel.

---

## Resolved decisions (from review)

1. **Streets are a data-layer decision.** `Block.streets` is the single source of truth;
   the method consumes it uniformly. kblock supplies real OSM streets (Slice 2); the
   `ShapefileSource` supplies the boundary (Slice 1). The topology port maps
   `Block.streets` → initial road edges (see decision 10), so topology requires *an* initial
   road set, not the boundary specifically. No explicit frontage exists for Phule Nagar, so
   the boundary is the correct fill there — and it matches topology's own baseline.
2. **Typing: `__post_init__` now, pandera at Slice 2.** pandera's advantage is real but
   *runtime/declarative* (enforced, self-documenting layer schemas + CRS/geometry checks —
   directly fixing kblock's implicit-column-contract problem), **not** a `mypy --strict` win.
   Slice 1's trivial layers don't need it; adopt it when Slice 2 ingests the first messy
   external source, replacing `__post_init__` validation wholesale (no dual path).
3. **`Screen` emits authoritative per-block (leaf) priorities.** The Screen owns the ranking
   (`Mapping[block_id, float]`), keeping control and staying directly consumable by the
   per-block loop; a detector working in native geometry does the geometry→block reduction
   *itself* and may expose flagged geometry as an auxiliary artifact. Flat per-block scores
   are the **leaf level of a region tree** (internal nodes ≈ the existing GADM/GHSL
   hierarchy; an antichain = a cut through it), so upgrading to a full tree — with a
   Screen-authored rule inducing leaf priorities — is **additive, not breaking**. Deferred to
   the screening slice (build the tree only when a Screen's detection is genuinely
   multi-scale; YAGNI).
