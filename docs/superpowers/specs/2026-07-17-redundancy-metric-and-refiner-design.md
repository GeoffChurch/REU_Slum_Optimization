# Redundancy metric (commute-ratio) + loop-closure refiner — Design

**Status:** design approved (2026-07-17), pending spec review
**Author:** owner + Claude
**Supersedes on the internal axis:** the `cycle_density` metric shipped in PR #4
(`docs/superpowers/specs/2026-07-16-metric-basis-reporting-design.md`)

## 1. Goal

Two coupled changes that let us **fill the empty internal-connectivity axis** — the corner of the
frontier no reblocking method reaches (every current method is a tree; internal connectivity is flat
~0.003 at region scale):

1. **Replace the internal-connectivity metric** `cycle_density` with the **commute-ratio ρ**, a
   continuous, gaming-resistant, dwelling-based redundancy measure.
2. **Add a composable loop-closure refiner** that takes any method's proposal (in practice
   clearance, the access champion) and greedily adds gap-following loops, moving it into the
   high-external + high-internal corner.

The metric change is foundational: it must land **first**, because the shipped `cycle_density` is
*perverse* (it rewards tiny "Bermuda-triangle" loops more than big useful ones), so it cannot
faithfully credit what the refiner builds. The two ship as **two sequenced implementation plans**.

## 2. Why ρ — the investigation in one paragraph

We stress-tested every candidate for the internal axis against a corpus gate (clearance
repulsion-sweep + both arterials + random road subsets, two blocks; loading on the internal axis,
orthogonality to access, anti-gaming BIG-vs-TINY loop test, continuity). Falsified: **Fiedler λ₂**
(the road∪street graph is disconnected → λ₂≡0; largest-component surgery inflates for fragmented
nets and lands it on the *external* axis); **effective-resistance duality on a walkable-base graph**
(the external/internal integrals collapse, corr→1.00); **cycle_density** (size-blind and perverse);
**log-spanning-tree count** (loads internal beautifully but is gameable by tiny loops — any
*cycle-counting* measure is); **soft load-weighted bypassability Φ** (still gameable). The lesson:
**anti-gaming comes from measuring per-dwelling redundancy extent, not counting loops.** The one
candidate that is simultaneously continuous, anti-gaming, internal-loading, and (after whitening)
exactly access-orthogonal is the **commute-ratio ρ**. See [[road-structure-metric-basis]] and the
scratchpad spikes `spike_gate*.py`, `spike_decorrelate.py`.

Key empirical facts the design relies on:
- ρ loads on the internal axis: corr(ρ, 2ec) ≈ +0.94, corr(ρ, cycle) ≈ +0.74.
- ρ is anti-gaming: on the BIG-vs-TINY loop test, ρ scores BIG ≈ 3× TINY (cycle and logtree score
  TINY > BIG).
- ρ's raw access coupling is +0.33..+0.42. Conditioning the mean on *reachable* dwellings, and/or
  residualizing against access, drives it to ≈0 **without changing the anti-gaming outcome**
  (whitening fixes orthogonality, never gameability — demonstrated in `spike_decorrelate.py`).
- Duality: `R(dwelling→street) = R_geo · (1 − ρ)` — external = the single-best-route distance
  `R_geo` (|corr| 0.84 with access), internal = the parallel-redundancy factor ρ. Series–parallel
  (equivalently cut-space ⊥ cycle-space) decomposition of one operator.

## 3. Part 1 — Commute-ratio internal metric

### 3.1 Definition

For a `Block` and a road set `roads`, on the **planarized road∪street graph**
`G = _noded_graph(roads, block.streets)` (nodes are `_rnd`-snapped coordinate tuples, edges carry
Euclidean length `len(e)`, conductance `c(e) = 1/len(e)`):

- **Street nodes** `S` = graph nodes that appear in `_explode_segments(block.streets.geometry)`.
- **Grounded effective resistance** `R(v)` = effective resistance from interior node `v` to the
  whole grounded street `S`. Computed per connected component: for each component containing ≥1
  street node, form the grounded Laplacian `L_g` over that component's interior nodes (off-diagonal
  `-c(e)` for interior–interior edges; each street-incident edge adds `c(e)` to its interior
  endpoint's diagonal), and `R(v) = (L_g⁻¹)_vv`. A component with **no** street node →
  `R(v) = ∞` for its interior nodes (genuinely stranded — no egress).
- **Geodesic resistance** `R_geo(v)` = shortest-path resistance from `v` to the nearest street node
  = shortest-path *length* (multi-source Dijkstra from `S`, weight `len`). `∞` if unreachable.
- **Per-dwelling redundancy** `ρ(v) = 1 − R(v)/R_geo(v) ∈ [0, 1)` where both are finite; `ρ = 0`
  where the dwelling is unreachable. (`R ≤ R_geo` always, since parallel paths only lower
  resistance, so `ρ ≥ 0`; `ρ = 0` for a single-path/tree route, `ρ → 1` as parallel routes thicken.)
- **Block metric** `commute_ratio(block, roads)` = **mean of `ρ(v)` over parcels**, where each
  parcel maps to its nearest interior graph node (a `cKDTree` query from parcel centroids to the
  non-street graph nodes).

`0.0` for no roads / no parcels / an empty graph.

### 3.2 Aggregation & orthogonality (Task-1 decision, gated before wiring)

The raw mean-over-all-parcels ρ carries a +0.33..+0.42 correlation with access. **Task 1 selects the
aggregation that minimizes this coupling while preserving the anti-gaming property**, evaluated on
the same two-block corpus, target: **corr(internal, access) ≤ 0.25** (2ec's level) and BIG > TINY
on the loop test. Candidate aggregations, in preference order (prefer an *intrinsic* fix over a
baked constant):
1. **Reachable-conditioned mean** — average ρ only over parcels whose nearest node has finite
   `R_geo` (exclude stranded parcels, which are an *access* failure already counted on the external
   axis). The detour and random-walk analyses predict this is the primary decoupler.
2. If (1) is insufficient, **parcel-count normalization variants** (e.g. weight by nothing vs by
   1/parcels-served) explored in the same gate.

We do **not** bake a whitening coefficient into the reported metric (it would be a magic constant
tied to a reference corpus). Whitening stays an *analysis* tool used only to demonstrate achievable
orthogonality; the reported value is raw (reachable-conditioned) ρ, whose coupling is already ≤
shipped `cycle_density`'s (+0.49). Task 1 records the chosen aggregation and the measured
loading/orthogonality/anti-gaming numbers in the metric's module docstring.

### 3.3 Interfaces (match the existing `BenefitFactory` seam)

In `src/reblock/budget.py`, mirroring `cycle_density` / `cycle_benefit`:

```python
def commute_ratio(block: Block, roads: GeoDataFrame | None) -> float:
    """Internal connectivity: mean over parcels of 1 - R(dwelling->street)/R_geodesic, on the
    noded road∪street graph (grounded effective resistance vs single-best-route resistance). 0 for
    a tree/single-path route, ->1 as parallel backup routes thicken. Continuous and gaming-resistant
    (rewards big loops that shorten many dwellings' routes, not loop count). 0.0 with no roads / no
    parcels / an empty graph."""

def commute_ratio_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                          tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]:
    """Internal-connectivity benefit factory (shares the access_benefit signature so it plugs into
    cost_benefit_curve(..., benefit_fn=commute_ratio_benefit) and the _sweep frontier). roads_full/
    tol are unused (commute_ratio is self-contained), kept for the shared BenefitFactory signature."""
    del roads_full, tol
    def f(roads: GeoDataFrame | None) -> float:
        return commute_ratio(block, roads)
    return f
```

`commute_ratio` returns a value in `[0, 1)`, so it drops straight into `cost_benefit_curve` /
`_sweep` (like `access_benefit`), and the frontier/curve machinery is unchanged.

Reuses existing helpers: `_noded_graph`, `_explode_segments` (kept — now consumed by
`commute_ratio` as well). New internal helper(s) for the grounded-resistance solve live in
`budget.py` next to `commute_ratio`.

### 3.4 Reporting wiring

- `src/reblock/compare.py:127` — swap `benefit_fn=cycle_benefit` → `benefit_fn=commute_ratio_benefit`
  (and the import on line 24). The metric key stays `"internal_connectivity"` (the axis is
  unchanged; only its operationalization changes), so `emit.py` frontier/curve filenames and
  `test_compare.py`'s file-existence assertions are untouched.
- `src/reblock/emit.py:244` — update `_METRIC_YLABELS["internal_connectivity"]` from
  `"internal connectivity (independent cycles per parcel)"` to
  `"internal connectivity (backup-route redundancy, mean 1 − R/R_geo)"`.

### 3.5 Migration (migrate, don't accommodate)

Per the standing directive — **no dual path, no back-compat shim**:
- **Delete** `cycle_density` and `cycle_benefit` from `budget.py`.
- **Delete** `tests/test_cycle_density.py`.
- **Update** `tests/test_compare.py`: the
  `test_compare_two_adjacent_block_region_arterial_beats_clearance_internal_connectivity` test
  encodes a property of the *old* metric (a loopier method out-scores a tree). Plan 1 must **measure
  the true ρ ordering** for that fixture's methods (`greedy_arterial_buildable` vs `clearance`) and
  re-author the test to assert whatever ρ actually does — including renaming it if the ordering
  differs — rather than assuming the old ordering survives. Do not tune ρ to preserve the old
  assertion; the test follows the metric, not the reverse.
- Confirm no other consumer: `cycle_density`/`cycle_benefit` are used only by reporting
  (`compare.py`, `emit.py` label) — arterial's objective is `directness`/`efficiency`, not cycles —
  so deletion is contained.

### 3.6 Tests (`tests/test_commute_ratio.py`)

- Tree / single-path roads → `ρ = 0` (every route is a bridge).
- A single loop closing a route → `ρ > 0`; a **big** loop (spanning many parcels' routes) scores
  strictly higher than a **tiny** loop enclosing a few — the anti-gaming property, asserted as an
  inequality on constructed fixtures.
- Disconnection: a road spur that reaches no street → its parcels contribute `ρ = 0`, no blow-up;
  the reachable-conditioned mean ignores them.
- Empty roads / `None` / no parcels → `0.0`.
- Monotone-ish sanity: adding a genuinely parallel route to an existing tree does not *decrease* ρ.

## 4. Part 2 — Loop-closure refiner

### 4.1 Component

A new `Method` at `src/reblock/methods/loop_closure.py`:

```python
@dataclass(frozen=True)
class LoopClosureRefiner:
    base: Method                     # the method whose proposal we refine (e.g. clearance)
    budget_m: float | None = None    # max total added road length; None -> until diminishing returns
    max_loops: int = 20              # hard cap on loops added
    min_graph_dist: int = 8          # only close loops spanning >= this many hops (skip Bermuda loops)
    search_radius_m: float = 60.0    # candidate endpoint pairs within this spatial distance
    snap_lam: float = 2.0            # gap-follow repulsion for _snap

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        base_prop = prior if prior is not None else self.base.propose(block)
        # ... refine base_prop.roads, return Proposal(roads=base_prop.roads + added loops)
```

This is the first real use of the `Method.propose(block, prior=…)` seam (every other method does
`del prior`). It works two ways: standalone via config (`self.base.propose`) and, if a pipeline ever
supplies a `prior`, it refines that directly.

### 4.2 Candidate generation

- Explode `base_prop.roads` into road-graph node coordinates; build the noded graph
  `G = _noded_graph(base_prop.roads, block.streets)` for graph-distance and bridge queries.
- Build the gap router once: `sg = _snap_graph(_boundary_graph(block.parcels))` (reused from
  `methods/arterial.py` and `methods/dijkstra.py`).
- Candidate loop edges = road-node pairs within `search_radius_m` (a `cKDTree.query_pairs`) whose
  graph distance in `G` is `≥ min_graph_dist` (so the closed loop is *big* — spans many parcels —
  not a tiny triangle). For each, `_snap(LineString([a, b]), sg, snap_lam)` produces a
  **gap-following** connecting road (threads existing gaps → ~zero building displacement). Drop
  `None`/zero-length snaps.

### 4.3 Greedy objective & stop rule

- **Objective (candidate ranking):** marginal **bridges-removed per metre** —
  `(bridges(roads) − bridges(roads + seg)) / seg.length`, where `bridges(·) = len(list(nx.bridges(
  _noded_graph(·, block.streets))))`. Bridges-removed is the linear-time, size-aware surrogate for
  ρ-gain (the gate showed bridges-removed correlates +0.92 with loop size, and computing ρ per
  candidate would need a Laplacian solve each). A big loop turns a long chain of bridges into
  2-edge-connected edges → high score; a tiny loop removes few → low score.
- **Greedy loop:** repeatedly add the highest-scoring remaining candidate; recompute marginal scores
  against the growing road set (a candidate's value changes as neighbours are added). **Stop** when
  any of: `budget_m` of added length is reached, `max_loops` loops added, or the best marginal
  bridges-removed ≤ 0 (diminishing returns).
- Return `Proposal(roads = concat(base_prop.roads, added_segments))` (preserve any other
  `Proposal` fields from `base_prop`).

### 4.4 Config

Add to `conf/compare_config.yaml` `all_methods` (and, if a standalone file is wanted,
`conf/method/loop_closure.yaml`):

```yaml
clearance_looped:
  _target_: reblock.methods.loop_closure.LoopClosureRefiner
  base: {_target_: reblock.methods.clearance.ClearanceReblocker, substrate: "${substrate}",
         repulsion: 0.0, depth_target: 2, max_roads: 400}
  budget_m: null
  max_loops: 20
  min_graph_dist: 8
```

### 4.5 Tests (`tests/test_loop_closure.py`)

- Refining a tree proposal on a fixture with an obvious gap adds ≥1 loop: `bridges` strictly
  decreases and `commute_ratio` strictly increases versus the base proposal.
- Budget respected: `budget_m` caps total added length; `max_loops` caps loop count.
- No candidates (already 2-edge-connected, or no admissible gap) → returns the base proposal
  unchanged (roads equal by geometry).
- `prior` pass-through: `propose(block, prior=p)` refines `p` and does **not** call `self.base`.
- Composition: the returned roads are a superset of `base_prop.roads` (existing roads preserved).

### 4.6 Efficacy validation (example, not a unit test)

On block `ZAF.9.3.1_1_40972` and one region, `clearance_looped` must **raise internal connectivity
(ρ) substantially above `clearance` while holding external (access) within tolerance and keeping
displacement bounded** — i.e. visibly move clearance up the internal axis toward the empty corner
(the loop-closure spike showed ~7× internal at ~flat displacement, access held). Captured as a
regenerated comparison figure/curve, reviewed by eye, in the refiner plan's final task.

## 5. Global constraints (bind every task)

- **Migrate, don't accommodate:** delete `cycle_density`, `cycle_benefit`, `tests/test_cycle_density.py`;
  no dual path, no deprecated alias.
- `pixi run check` (ruff lint + mypy --strict + pytest) stays green.
- ruff: no semicolons (E702), ≤100-char lines (E501), `zip(..., strict=…)` (B905).
- New public functions typed for mypy --strict; match surrounding budget.py/methods style.
- Commit trailers and PR-body footer per repo convention.

## 6. Implementation phasing

Two sequenced plans (writing-plans produces them one at a time):
1. **Metric migration** — Task 1 gates the aggregation/orthogonality on the corpus **before** wiring;
   then add `commute_ratio`/`commute_ratio_benefit` + tests, swap the reporting `benefit_fn`, update
   the ylabel, delete the `cycle_density` path, re-baseline `test_compare.py`. Ends green.
2. **Loop-closure refiner** — `LoopClosureRefiner` + candidate/greedy machinery + tests + config
   entry + the efficacy example regeneration. Depends on Plan 1 (uses `commute_ratio` in tests and
   the efficacy check).

## 7. Out of scope / future

- The other de-risked continuous candidates (detour-ratio via Suurballe; dwelling-weighted
  reliability; min-cut split Φ) — ρ won the gate; they are recorded in the scratchpad brainstorm if
  ρ ever needs a cheaper or more interpretable substitute.
- Making the *external* metric literally `1/R_geo` for an exactly-matched one-operator dual (kept as
  `access_benefit` for now; ρ is its validated internal partner).
- Narrow one-way corridors for loop-roads (would let the refiner's added length count as reduced
  cost) — a displacement-side follow-on.
- The refiner optimizing ρ directly (vs the bridges surrogate) behind a config flag, if the surrogate
  ever proves too loose.
