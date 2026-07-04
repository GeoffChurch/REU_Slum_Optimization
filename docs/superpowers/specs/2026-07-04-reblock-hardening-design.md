# reblock — Architecture hardening (post red-team) + render + peel-reblocker roadmap

**Status:** draft for review · **Date:** 2026-07-04 · **Branch:** `hardening`

## Why this exists

After Slice 1 merged, a 4-lens adversarial red-team (architecture, domain-correctness,
data-model, YAGNI) stress-tested the render design and the overall architecture. It surfaced
real warts — load-bearing contract shapes and silent-wrong-output risks — cheap to fix while
the project is young. This doc consolidates the resulting changes and **supersedes decisions
2, 4, 10, 11 and the contracts** of `2026-07-03-reblock-architecture-and-slice-1-design.md`.

**Confirmed sound by the red-team — keep unchanged:** the canonical `Block` waist (N+M+K, not
N×M); `to_parcel_graph` as a one-way derivation; `__post_init__` boundary validation;
CRS→local-UTM at load; the render "before = data / after = method" split; keeping topology's
native shapefile/GeoJSON I/O; and Slice 1's behavior on the real Phule Nagar data.

---

## 1. Contracts v2 (`reblock.contracts`)

The root wart: `Metrics` is a per-block *scalar bag*, but access data is *per-parcel*; every
render workaround (kwargs on `Eval.score`, a bare `ndarray`, reaching into topology internals)
patched that one missing shape. Fix the shape once.

- **`Metrics` gains a per-parcel channel.**
  ```python
  @dataclass(frozen=True)
  class Metrics:
      block_id: str
      method: str
      eval: str
      values: Mapping[str, float]                       # scalar reductions: k_before, delta_k, ...
      fields: Mapping[str, "pd.Series[float]"] = field(default_factory=dict)  # per-parcel, indexed by parcel_id
  ```
  The eval **emits** per-parcel arrays into `fields` (e.g. `access_before`, `access_after`);
  render **reads** them. No injection kwargs, no dual path.

- **`Result` bundles the scorecard so methods are comparable and results persist/render later.**
  ```python
  @dataclass(frozen=True)
  class Result:
      block: Block
      proposal: Proposal
      metrics: tuple[Metrics, ...]     # one per eval
  def run(cfg) -> list[Result]: ...
  ```
  Cross-method selection = `max(results_for_block, key=lambda r: r.metric("kcomplexity","delta_k"))`.

- **`Method` role split for composition + cross-block** (the specs already promise both — water
  along a prior method's roads; arterials that cross block boundaries):
  ```python
  class Method(Protocol):        # per-block, composable
      def propose(self, block: Block, prior: Proposal | None = None) -> Proposal: ...
  class RegionMethod(Protocol):  # cross-block
      def propose(self, region: Region) -> Iterable[Proposal]: ...
  ```
  (topology stays a `Method`; `prior` defaulted so it's a no-op for it today.)

- **Trim speculative surface (zero consumers today):** delete `Region.{water,food,healthcare}`,
  `Block.{buildings,water,barriers}`, `Proposal.{water_points,water_mains}`, and the `Screen`
  Protocol. When amenities/interventions actually land (Slice 4/5) they return as **one layer
  with a controlled-vocabulary `category`/`kind` column** (`Region.pois[category]`,
  `Proposal.interventions[kind]`) — a typed layer validated in `__post_init__`, *not* the open
  map the spec rightly rejects, and *not* a churn-the-core sibling field per amenity type.
  Keep the `Region`/`Block`/`Proposal` cores and `Source`/`Method`/`Eval`.

- **`Source` yields a lazy `Region`;** `Region.blocks` is a real generator (see §4). Regional
  per-block enrichment is by `dataclasses.replace` (blocks stay `frozen`) — the arch spec's
  "write scalars onto each Block" is corrected to "derive an enriched Block."

- **`proposal_id`** on `Proposal` (method name + salient params, or a run counter) so parameter
  sweeps (`alpha=2` vs `alpha=4`) don't collide in filenames or result keys.

## 2. Access metric — BFS peel (primary) + weak-dual (optional)

The red-team **verified** three real (currently-latent) bugs in topology's weak-dual `k`: a
single-file corridor scores k=1 for any length; `k` silently caps at 8 (`stacked_duals
maxdepth=15`); and a concave "wraparound" parcel can vanish (`trace_faces`). All live in the
topology-graph path.

- **Primary metric = BFS peel on parcels** (kblock's definition). `parcel_access_layers(block,
  roads) -> pd.Series[int]` (**indexed by `parcel_id`**, not row position): build parcel
  adjacency (shared-edge, with a small tolerance for pinch-point gaps), seed = parcels touching
  `streets ∪ roads`, BFS outward; each parcel's layer = steps to the nearest street-adjacent
  parcel; block `k = max(layers)`. This is **robust** (no strip/cap/`trace_faces` bug — it never
  builds topology's graph), **fast** (STRtree adjacency + linear BFS; skips the ~48 s
  `clean_up_geometry` trace), **per-parcel native** (no face→parcel nearest-centroid mapping —
  finding 4 gone), and it makes eval + render **independent of topology** (topology becomes
  purely the road-building method).
- **Weak-dual `k` retained as an optional secondary eval** for comparability with published
  Brelsford k-complexity / kblock numbers, and as a cross-check. On well-behaved blocks the two
  agree exactly (2×2→1, 3×3→2, 5×5→3); they diverge only on the pathological shapes where the
  peel is the correct one.
- `KComplexityEval` refactored to compute on the peel and emit per-parcel `fields`; the
  weak-dual becomes a distinct optional eval.

## 3. Render (`reblock.render`)

- Two functions, **separate files**, shared colour scale: `render_before(block, layers, *,
  vmax) -> Figure` (block-only, method-independent → `{block_id}_before.png`) and
  `render_after(block, proposal, layers, *, vmax, metrics=None) -> Figure` (per proposal;
  overlays roads → `{block_id}_{proposal_id}_after.png`). `YlOrRd` heatmap, `vmin=1`,
  `vmax=k_before`, colourbar; geopandas/matplotlib `Agg`. Reads layers from `Metrics.fields`.
- `run` wiring: `RunConfig.render_dir: str | None`; resolved under
  `HydraConfig.get().runtime.output_dir` (per-run isolation); `run()` keeps a Hydra-agnostic
  `render_base` param for testability.

## 4. Robustness — fail loud, not silently wrong

- **CRS:** `ShapefileSource(path, *, assumed_crs: CRS | ... )` — when the shapefile has no
  `.prj`, require an explicit assumed CRS (no silent EPSG:3857 guess that can land data on Null
  Island). Honour the arch spec's own fail-fast policy.
- **Lazy + row-validated loading:** `Region.blocks` is a genuine generator; validate/normalise
  each raw record (explode or reject native `MultiPolygon` records at the *row*, with an error
  naming the feature) so one malformed record can't crash a whole dataset (the Epworth load
  failure) — critical at kblock/continental scale.
- **Consume `Block.streets`:** the method + eval map `Block.streets → initial road set` instead
  of the boundary-only shortcut, with a `streets ⊊ boundary` test. (Peel seeds from
  `streets ∪ roads`; the topology method marks graph edges coincident with `Block.streets`.)

## 5. Topology fixes — for the *method* (the peel doesn't rescue the road-builder)

The peel fixes the *metric*, but topology's `build_all_roads` still runs on the traced graph,
so these are fixed in the submodule for the method's correctness:

- **`trace_faces`:** pick the outer face by **winding order / centroid-containment**, not "most
  edges" — so a concave wraparound parcel isn't dropped and its boundary isn't mislabelled as a
  street. Add the U-shaped-parcel synthetic case as a permanent regression fixture.
- **`stacked_duals` `maxdepth`:** raise/parametrise it and make truncation **loud** (warn/raise
  when the cap is actually hit) so deep blocks never silently report k=8. (Relevant to the
  method because `build_all_roads` picks paths via `form_equivalence_classes`.)

## 6. Hydra pluggability — make the thesis real

Today `run.py` hardcodes one Source/Method/Eval; there are no config groups. Scaffold
`conf/{data,method,eval}/*.yaml` with `_target_` + `hydra.utils.instantiate`, a `defaults`
list, and **`methods: list` / `evals: list`** in the loop — so `method=… eval=…` (and multirun
`method=topology,peel`) actually work, and adding a component is config, not a `run.py` edit.

## 7. Perf / memoization

`build_all_roads` + eval currently re-trace the same block ~3×. Memoize `Block →
PlanarParcelGraph` (keyed by a stable block identity) and re-mark roads via `define_roads_on`
instead of rebuilding, for the topology-method path. The peel path needs no cache (it's cheap
and traces nothing). Note: CapeTown's ~21 min is `build_all_roads` itself (it recomputes the
weak-dual every greedy iteration) — **not** addressed here; see roadmap.

## Roadmap (next, after this pass)

- **Peel-reblocker — the first distinct *second* method.** Reuses `parcel_access_layers` to
  find the deepest interior parcels, then routes roads by **steepest-descent down the peel
  gradient** (connect each deep parcel to a shallower neighbour, step to a street) → `Proposal`.
  Fast (no weak-dual), robust (no topology graph), and it validates the whole multi-method
  design: two methods → two afters → compared via `Result`/`Metrics`. Genuine research value
  (weak-dual-greedy vs peel-gradient-greedy, head to head). The *routing* is the real work; the
  target-finding is free from the peel derivation this pass already builds.
- **Optional — `build_all_roads` internal perf:** have topology's road-builder use the peel to
  find targets instead of rebuilding the weak-dual each iteration (10×+ on big blocks). Deferred:
  it's surgery on the method's core algorithm; the peel-reblocker gives a fast path without it.
- Earlier roadmap items unchanged: kblock source (Slice 2), displacement eval, water method,
  regional screening.

## Testing highlights

- `parcel_access_layers`: 2×2→all layer 1; 3×3→centre layer 2; **a strip of N→layer N** (the
  case the weak-dual gets wrong — proves the peel is honest); nonzero-origin; a `parcel_id`
  reorder (proves id-keying survives reindexing).
- Eval: peel-`k` matches weak-dual on the grids; `Metrics.fields` populated per parcel.
- Render: `render_before`/`render_after` return figures; separate PNGs; shared `vmax`.
- Robustness: `.prj`-less shapefile without `assumed_crs` raises; Epworth loads (bad record
  isolated, not fatal); `streets ⊊ boundary` changes the initial road set.
- topology: U-parcel fixture (outer face correct, parcel not dropped); deep grid past the old
  cap reports the true k (loud, not silently 8).
- Hydra: `method=topology eval=kcomplexity` and a two-eval run compose via config groups.

## Migration note (per owner's "migrate, don't accommodate")

This reworks merged Slice-1 contracts (`Metrics`, `Eval`, `Method`, `Source`) and the eval.
There is no dual-path/back-compat: the old `KComplexityEval._k`/`Eval.score` weak-dual path is
replaced by the peel; the deleted contract fields are removed outright. Existing tests migrate
to the new shapes.
