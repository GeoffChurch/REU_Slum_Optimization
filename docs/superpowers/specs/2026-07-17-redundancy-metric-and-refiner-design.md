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

### 2.1 Alternatives considered and rejected (so we don't re-litigate)

All measured on the same two-block corpus gate. Numbers are from `spike_gate*.py` /
`spike_decorrelate.py` (scratchpad). "BIG/TINY" = the anti-gaming test (add 3 big vs 3 tiny
gap-snapped loops to clearance; a good internal metric scores BIG > TINY).

| candidate | verdict — why rejected |
|---|---|
| **`cycle_density`** (shipped) | **Perverse**: size-blind cycle count; TINY 0.046 > BIG 0.019. Rewards Bermuda loops. |
| **Fiedler λ₂** | Road∪street graph is disconnected → λ₂≡0; largest-component surgery *inflates* for fragmented nets → loads on the **external** axis (corr(access) −0.5). Closing the full boundary into one ring revives it on *one* block (size-aware corr +0.94) but it still fails the corpus. |
| **Effective-resistance duality** (Rext/Rint on a walkable base) | The two "dual" axes **collapse** (corr 0.97→1.00 at every road/walk contrast); both track first-order access; redundancy is a swamped second-order term. Plain resistance/commute-time is a dead end. |
| **log-spanning-tree count** (Matrix-Tree) | Loads internal cleanly (corr +0.98 w/ 2ec, ⊥ access) but **gameable** (TINY 0.107 > BIG 0.089). Any *cycle-counting* measure is. |
| **soft load-weighted bypassability Φ** (UST marginal × drainage load) | Continuous and the most access-orthogonal (−0.13), but **still gameable** (TINY 0.122 > BIG 0.092). Softening a count ≠ robustness. |
| **2-edge-connectivity (2ec)** | Robust (BIG > TINY) + cheap + interpretable; the **runner-up**. Rejected only for the hard bridge/not-bridge boolean the owner wanted to avoid; kept as the documented fallback if ρ's cost or coupling disappoints. |

Cross-cutting lesson, load-bearing for the whole design: **robustness to loop-count gaming comes from
measuring per-dwelling redundancy *extent*, not counting loops; and whitening fixes orthogonality but
*never* gameability** (residualizing a gameable metric against access leaves its BIG-vs-TINY ordering
intact — demonstrated). So the metric had to be robust by construction, not decorrelated into it.

**The limit of that robustness (red-team, verified numerically).** ρ resists loop-*count* gaming but
NOT corridor-*duplication*: effective resistance rewards electrical parallelism, which does not
require geographic separation, so k near-parallel duplicate stubs drive a parcel's ρ to `1 − 1/k`
without any real backup value. ρ is therefore **not** "intrinsically anti-gaming" in isolation. The
design accepts this because the metric lives in a **suite**: duplicating a corridor costs road
*length* and *displacement*, so a duplicator raises internal connectivity only by worsening the cost
axes — and at the block-mean level a genuine big loop (serving many parcels at once) is *more*
ρ-per-metre efficient than bundling parcels one at a time. The acceptance criterion is **"no single
move games the whole suite {external, internal, displacement-vs-length}"**, which §3.2's gate
verifies directly (a bundler must be Pareto-dominated). Near-parallel edge-collapse is the committed
fallback if that gate ever shows otherwise.

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
- **Block metric** `commute_ratio(block, roads)` = **mean of `ρ(v)` over reachable parcels**, where
  each parcel's entry node is its frozen `_line_entries` entry (nearest point on an edge — the same
  entry `access_benefit` uses), NOT a centroid→nearest-vertex query (see §3.3.1).

`0.0` for no roads / no reachable parcels / no interior nodes / an empty graph.

**Semantics note (what ρ actually credits).** Because R grounds the *entire* street set as one node,
ρ credits **all egress redundancy**, which is broader than "internal loops": a parcel that reaches
the street at two distinct points via diverging paths gets ρ>0 **even on a graph-theoretic tree**
(circuit rank 0). So the invariant is "**single-egress** tree route → ρ=0", not "any tree → 0". This
is arguably *correct* (two ways to a street is genuine redundancy — if one street access floods, use
the other), but it means (a) tests must use single-egress fixtures for the ρ=0 case, and (b)
multiple-egress credit is a structural contributor to ρ's access coupling that reachable-conditioning
does not remove (§3.2). If a *pure internal-loop* semantics is ever wanted, ground each street
*component* separately instead of shorting all of S — deferred unless the §3.2 gate demands it.

### 3.2 Aggregation & orthogonality (Task-1 decision, gated before wiring)

The raw mean-over-all-parcels ρ carries a +0.33..+0.42 correlation with access. Task 1 selects the
aggregation on the two-block corpus, running these checks and **committing this exit branch** (no
open-ended "hunt for variants"):

- **Aggregation candidates, preferred first:** (1) reachable-conditioned mean (average ρ only over
  parcels with finite `R_geo`; stranded parcels are an access failure already on the external axis);
  (2) if (1) disappoints, one parcel-weighting variant. We do **not** bake a whitening constant into
  the reported value (it would be a magic constant tied to a reference corpus).
- **Hard gate (the committed rule):** ship reachable-conditioned ρ **iff** it clears BIG>TINY on the
  loop test **and** `corr(internal, access) ≤ 0.49` (strictly beats shipped `cycle_density`).
  `≤ 0.25` (2ec's level) is the *aspiration*; among aggregations that pass, pick the lowest-coupling
  one. **If no aggregation clears 0.49, fall back to 2ec** (§2.1) — a named, committed branch, not an
  open question. An implementer measuring 0.35 therefore has an unambiguous instruction: ship it.
- **Gaming vectors the gate MUST run (not just BIG/TINY):** (a) the **corridor-duplication** vector —
  add k near-parallel stubs and confirm that although ρ rises, the network is **dominated on the
  joint suite** (higher displacement and/or road length at equal or worse external — i.e. a bundler
  is not Pareto-competitive with a genuine looper on {external, internal, displacement-vs-length});
  (b) the **monotonicity** check — a drainage-ordered `commute_ratio_benefit` sweep on a
  spur+stranded-pocket fixture must be non-decreasing (validates the §3.3.1 freezing). If the
  duplication vector shows a bundler IS competitive on the suite, add the near-parallel edge-collapse
  fallback (merge edges parallel within ε≈3–5 m before the solve) and re-run.

Task 1 records the chosen aggregation and the measured loading / orthogonality / BIG-vs-TINY /
duplication-suite / monotonicity numbers in the metric's module docstring.

### 3.3 Interfaces (match the existing `BenefitFactory` seam)

In `src/reblock/budget.py`, mirroring `cycle_density` / `cycle_benefit`:

```python
def commute_ratio(block: Block, roads: GeoDataFrame | None) -> float:
    """Internal connectivity: mean over reachable parcels of 1 - R(dwelling->street)/R_geodesic, on
    the noded road∪street graph (grounded effective resistance vs single-best-route resistance).
    0 for a single-egress tree route, ->1 as parallel backup routes thicken. Clipped to [0, 1).
    Resists loop-COUNT gaming (a big loop that shortens many dwellings' routes beats many tiny loops);
    NOT robust in isolation to corridor-DUPLICATION (see §3.2), which the suite's displacement/length
    axes penalize. 0.0 with no roads / no reachable parcels / no interior nodes / an empty graph."""

def commute_ratio_benefit(block: Block, roads_full: GeoDataFrame | None, *,
                          tol: float = STREET_TOL) -> Callable[[GeoDataFrame | None], float]:
    """Internal-connectivity benefit factory (shares the access_benefit signature so it plugs into
    cost_benefit_curve(..., benefit_fn=commute_ratio_benefit) and the _sweep frontier). Unlike a bare
    per-prefix call, this FREEZES the parcel->entry map and the reachable/averaged set against
    `roads_full` and only grows the edge set across prefixes -- required for a non-jagged frontier
    curve (see §3.3.1)."""
```

`commute_ratio` returns a value in `[0, 1)`. **It does NOT "drop straight in" to `_sweep` unchanged**
— see §3.3.1 (monotonicity) below; the benefit factory must freeze entries.

#### 3.3.1 Implementation: restore the deleted sparse+frozen resistance engine — do NOT write new dense code

The grounded resistance-to-street `R(v)` is *exactly* what the deleted egress-resistance engine
computed (`resistance_benefit`/`_resistance_core`/`resistance_frozen`/`_ground_indices`, removed in
PR #4, recoverable from commit `fe4180c`; design at
`docs/superpowers/specs/2026-07-11-resistance-eval-design.md`). **Plan 1 restores and adapts that
machinery rather than writing a fresh dense inverse.** This is not a back-compat resurrection of the
deleted *metric* (egress-resistance loaded external and stays deleted) — only its computational core,
which ρ legitimately needs. Restoring it fixes five red-team findings at once:

- **Cost (mandatory — see §5.1 gate).** `_resistance_core` factorizes `L_g` ONCE via
  `scipy.sparse.linalg.factorized` and back-substitutes per distinct entry node — the spike note
  clocks a 3425-parcel region at 55 s doing hundreds of solves. A dense per-prefix inverse (what the
  original spec implied) is ~1–2 orders slower and walls at region scale, where `cost_benefit_curve`
  calls the benefit ~20× per method per block.
- **Ground set S — geometric, not combinatorial.** Define street nodes by proximity,
  `Point(node).distance(street_geom) <= tol` (as `_ground_indices`/`road_drainage` already do), NOT
  "nodes appearing in raw `block.streets`". Planarization *creates* on-street nodes at road/street
  crossings that are absent from the raw street vertices; the combinatorial definition mis-classifies
  them as interior and yields silently wrong ρ.
- **Parcel→entry map — line-proximity, not centroid→nearest-vertex.** Reuse `_line_entries` /
  `_BlockScoringContext` frozen entries (nearest point *on an edge*, matching `access_benefit`), not
  a `cKDTree` from parcel centroids to graph vertices — the latter is the exact mapping `_line_entries`
  documents as wrong (undercounts chords; a wide parcel's centroid is far from every edge), and using
  it would also compute ρ and access on *different* parcel maps, muddying the §3.2 orthogonality read.
- **Frozen entries → non-jagged sweep (§3.3.1 monotonicity).** Freeze the parcel→entry map and the
  reachable/averaged set against `roads_full`, then only grow the edge set across prefixes (mirror
  `_efficiency_factory`, budget.py:559–585). Without this, a re-mapped parcel or a newly-reachable
  stranded parcel makes the block-mean *drop* as roads are added → the internal curve regresses.
  **Caveat that must be stated in code:** even frozen, ρ = 1 − R/R_geo is a ratio of two co-decreasing
  quantities, so monotonicity is NOT structural (unlike `directness`) — Plan 1 asserts it empirically
  on a drainage-ordered prefix sweep over a spur+stranded-pocket fixture, and never assumes it.
- **`[0,1)` clip + range guards.** R(v) (matrix solve) and R_geo(v) (Dijkstra sum) reach the same
  number by different float paths, so on bridge routes ρ can compute to −ε; ill-conditioning
  (0.01 m edges @ conductance 100 vs 200 m depths) can push it past 1. Clip
  `ρ = min(max(ρ, 0.0), 1 - 1e-12)`; guard "no interior (non-street) nodes → 0.0" and
  "empty reachable set → 0.0" alongside the empty-graph guard.

Reuses (kept): `_noded_graph`, `_explode_segments`, `_line_entries`, `_BlockScoringContext`. Restored
(adapted from `fe4180c`): the sparse grounded solve + geometric ground set + frozen-entry context.

### 3.4 Reporting wiring

- `src/reblock/compare.py:127` — swap `benefit_fn=cycle_benefit` → `benefit_fn=commute_ratio_benefit`
  (and the import on line 24, **and the stale module comment on line 38** that names `cycle_benefit`).
  The metric key stays `"internal_connectivity"` (the axis is unchanged; only its operationalization
  changes), so `emit.py` filenames and `test_compare.py`'s file-existence assertions are untouched.
  (Verified by the red-team: there is **no** scalar AUC/rank scale-sensitivity — that logic was
  removed in PR #6 — so the metric-key swap is safe for ranking.)
- `src/reblock/emit.py:244` — update `_METRIC_YLABELS["internal_connectivity"]` to
  `"internal connectivity (backup-route redundancy, mean 1 − R/R_geo)"`.
- `src/reblock/emit.py:309` — the frontier CSV writes benefit as `%.4f`; ρ at region scale is small,
  so widen the internal-connectivity benefit precision to `%.6g` (else tiny-but-real ρ rounds to
  `0.0000`, breaking both the sweep read and the arterial-vs-clearance test — see §3.5).

### 3.5 Migration (migrate, don't accommodate)

Per the standing directive — **no dual path, no back-compat shim**:
- **Delete** `cycle_density` and `cycle_benefit` from `budget.py`. (Verified contained in *code*:
  arterial's objective is `directness`/`efficiency`, not cycles; no `__all__` export; only
  `compare.py`/`emit.py` consume them.)
- **Delete** `tests/test_cycle_density.py` — but first **re-home its `_noded_graph` tests**
  (`test_crossing_is_noded_into_a_shared_vertex`, `test_subdivision_invariance`) into
  `tests/test_commute_ratio.py` or a `test_budget.py` section. They test `_noded_graph`, which is
  *kept* and goes from 1 caller to 3 (metric + refiner candidate-gen + refiner objective) — deleting
  its only coverage exactly as it becomes load-bearing is the opposite of what we want. Order the
  deletion so `_noded_graph` is never transiently caller-less.
- **Update `tests/test_compare.py`** — the arterial-beats-clearance test encodes an *old-metric*
  property. Plan 1 must **measure the true ρ ordering from the metric function** (`commute_ratio` /
  the un-rounded `Curve.benefit[-1]`, **NOT** the `%.4f` CSV — which can read a real sub-0.0001
  arterial win as a tie and mis-author the test) for `greedy_arterial_buildable` vs `clearance`, and
  re-author (rename if needed) to assert whatever ρ actually does. Do not tune ρ to preserve the old
  assertion. **Flag the narrative risk:** if buildable arterial does *not* beat clearance under ρ,
  the repo's "arterial owns internal connectivity" story is falsified.
- **Reporting surface is larger than code.** §3.4's swap regenerates `curve_internal_connectivity*.png`
  under a ~30–100× different y-scale, but the following still say "independent cycles/parcel" with
  cycle-era numbers and would become self-contradictory shipped docs — Plan 1 must reconcile or
  explicitly mark them "not yet regenerated": `examples/method-comparison/README.md`,
  `examples/multiblock/README.md` (both hard-code internal-connectivity tables/prose + embed the
  PNGs), and the `road-structure-metric-basis` memory note.

### 3.6 Tests (`tests/test_commute_ratio.py`)

- **Single-egress** tree route → `ρ = 0` (fixture must have one street-egress path; a *multi*-egress
  tree gives ρ>0 by design — see §3.1 semantics — so do not assert 0 there).
- A single loop → `ρ > 0`; a **big** loop (spanning many parcels' routes) scores strictly higher than
  a **tiny** loop — the loop-count anti-gaming property, as an inequality on constructed fixtures.
- Range: `ρ ∈ [0, 1)` on every fixture (asserts the clip); no NaN.
- Disconnection: a spur reaching no street → those parcels excluded from the reachable-conditioned
  mean, no blow-up. Empty roads / `None` / no parcels / **no interior nodes** / **all-stranded
  (empty reachable set)** → `0.0` (each guarded explicitly).
- **Frontier monotonicity (the real path, not a hand-picked case):** a drainage-ordered
  `commute_ratio_benefit` sweep (via `cost_benefit_curve`) over a fixture containing dead-end spurs
  **and** a stranded pocket must return a **non-decreasing** benefit list — this is what validates the
  §3.3.1 freezing; the old "monotone-ish, adding a parallel route" check is insufficient.

## 4. Part 2 — Loop-closure refiner

### 4.1 Component

A new `Method` at `src/reblock/methods/loop_closure.py` — a plain `@dataclass` (NOT `frozen=True`;
siblings are non-frozen, and a frozen dataclass auto-hashes its `base` field, which is unhashable):

```python
@dataclass
class LoopClosureRefiner:
    base: Method                     # the method whose proposal we refine (e.g. clearance)
    budget_m: float | None = None    # max total ADDED road length (not total); None -> diminishing-returns stop
    max_loops: int = 20
    min_loop_len_m: float = 40.0     # geometric loop-perimeter floor (NOT a hop count -- see below)
    search_radius_m: float = 60.0    # candidate endpoint pairs within this spatial distance
    snap_lam: float = 2.0

    @property
    def identity(self) -> tuple | None:              # REQUIRED for the content-addressed cache
        bid = getattr(self.base, "identity", None)
        if bid is None:
            return None                              # uncacheable base -> uncacheable refiner
        return ("loop_closure", bid, self.budget_m, self.max_loops, self.min_loop_len_m,
                self.search_radius_m, self.snap_lam)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        base_prop = prior if prior is not None else propose(self.base, block)  # via derivations, shares cache
        # ... refine base_prop.roads, return Proposal(roads = base_prop.roads + added loops)
```

First real use of the `Method.propose(block, prior=…)` seam. **Cache wiring (a correctness
requirement, not perf-only):**
- The `identity` property above is mandatory — without it `derive` reads `getattr(method, "identity",
  None) is None` → silent cache bypass (base clearance + refiner recompute every run). It must fold in
  `base.identity` and propagate `None` when the base is uncacheable (e.g. a prebuilt-substrate
  clearance whose `identity is None`).
- **Register `reblock.methods.loop_closure` in `_DERIVATION_MODULES`** (`derive_graph.py:34–50`), or
  edits to the refiner return **stale cached roads** — a real correctness bug.
- Compute the base via the caching `propose(self.base, block)` entrypoint (not a bare
  `self.base.propose`), so a standalone `clearance` in the same sweep shares its L1/L2 cache.

Hydra recursive `instantiate` builds the nested `base` correctly (verified), so the config
composition itself is sound.

### 4.2 Candidate generation

- Build `G = _noded_graph(base_prop.roads, block.streets)`; get road-graph node coordinates from
  `_explode_segments(base_prop.roads)`; build the gap router once,
  `sg = _snap_graph(_boundary_graph(block.parcels))`.
- Candidate loop edges = road-node pairs within `search_radius_m` (`cKDTree.query_pairs`) whose
  **gap-snapped connector would close a loop of geometric perimeter ≥ `min_loop_len_m`**. Use a
  **geometric** loop-size floor, NOT `min_graph_dist` hops: `_noded_graph` edges are unweighted, so a
  hop count means metres-per-hop that varies wildly with subdivision (8 hops ≈ a few metres on dense
  frontage — lets Bermuda loops through — and can span a whole block on a coarse arterial base —
  admits nothing). `_snap(LineString([a,b]), sg, snap_lam)` produces a gap-following connector (low —
  **not zero** — building displacement: `_snap`'s cost has no building-repulsion term). Drop
  `None`/zero-length snaps.

### 4.3 Greedy objective & stop rule — lazy + incremental (mandatory, see §5.2)

- **Objective (candidate ranking):** marginal **bridges-removed per metre**. Bridges-removed is a
  size-aware, dwelling-*blind* surrogate for ρ-gain — a known looseness: it credits a loop by the
  number of graph edges made 2-edge-connected, not by how many *dwellings* route through them, and it
  is subdivision-sensitive where ρ is invariant. So the surrogate can prefer a long lightly-used spur
  loop over a shorter dense-core loop that raises ρ more. Mitigations, in order: (1) the efficacy
  gate (§4.6) measures the *reported ρ* delta, not bridges, so a surrogate that fails to move ρ is
  caught; (2) if §4.6 shows poor ρ-per-metre, weight bridges-removed by dwellings-served, or optimize
  ρ directly on the small surviving candidate set (a handful of solves, affordable) behind a flag.
- **Cost — do NOT recompute `nx.bridges` per candidate per step.** On a tree base the candidate set
  is tens of thousands, and a full `unary_union`+`nx.bridges` per candidate per step is hours at
  region scale (benchmarked). Instead: (a) compute the **bridge-tree / 2-edge-connected components
  once per step** (O(V+E)); each candidate's bridges-removed is then the bridge-tree path length
  between its endpoints' components (O(path) or O(log V) with LCA); (b) bridges-removed is
  **submodular** (a later loop finds no more bridges on an already-covered path), so use **CELF /
  lazy-greedy** — re-score only the stale top-of-heap candidate — exact and ~1–2 orders faster. This
  is the same pattern the "arterial-too-slow-on-regions" work already validated.
- **Stop** at any of `budget_m` *added* length, `max_loops`, or best marginal bridges-removed ≤ 0.
  Note the default `budget_m: null` + `max_loops: 20` can over-pave a tree (every admissible loop
  removes ≥1 bridge, so the ≤0 stop rarely fires) — set a sane default `budget_m` or lower `max_loops`.
- Return `Proposal(roads = concat(base_prop.roads, added))` preserving other `base_prop` fields.

### 4.4 Config

Add to `conf/compare_config.yaml` `all_methods` (and, if a standalone file is wanted,
`conf/method/loop_closure.yaml`):

```yaml
clearance_looped:
  _target_: reblock.methods.loop_closure.LoopClosureRefiner
  base: {_target_: reblock.methods.clearance.ClearanceReblocker, substrate: "${substrate}",
         repulsion: 0.0, depth_target: 2, max_roads: 400}
  budget_m: 200.0        # added-length cap (a sane default; null risks over-paving via max_loops)
  max_loops: 20
  min_loop_len_m: 40.0
```

### 4.5 Tests (`tests/test_loop_closure.py`)

- Refining a tree proposal on a fixture with an obvious gap adds ≥1 loop: `bridges` strictly
  decreases and `commute_ratio` strictly increases versus the base proposal.
- Budget respected: `budget_m` caps total added length; `max_loops` caps loop count.
- No candidates (already 2-edge-connected, or no admissible gap) → returns the base proposal
  unchanged (roads equal by geometry).
- `prior` pass-through: `propose(block, prior=p)` refines `p` and does **not** call `self.base`.
- Composition: the returned roads are a superset of `base_prop.roads` (existing roads preserved).
- Cache identity: `identity` folds in `base.identity` and is `None` when the base's is `None`;
  changing `budget_m`/`min_loop_len_m` changes `identity` (distinct cache keys).

### 4.6 Efficacy validation (example, not a unit test)

On block `ZAF.9.3.1_1_40972` and one region, `clearance_looped` must **raise internal connectivity
(ρ) substantially above `clearance` while holding external (access) within tolerance and keeping
displacement bounded** — i.e. visibly move clearance up the internal axis toward the empty corner
(the loop-closure spike showed ~7× internal at ~flat displacement, access held). Captured as a
regenerated comparison figure/curve, reviewed by eye, in the refiner plan's final task.

## 5. Global constraints (bind every task)

- **Scalability is a first-class, blocking requirement** (not an optimization to defer). ρ runs ~20×
  per method per block during *reporting*, so a slow ρ slows the whole compare pipeline for every
  method at region scale — it MUST use the restored sparse+frozen engine (§3.3.1). The refiner MUST
  use lazy-greedy + an incremental bridge-tree (§4.3). Each plan opens with a benchmark gate (§6)
  that blocks the rest of the plan until met. (The flagship *methods'* own propose cost is unchanged
  by this work; ρ is a metric and the refiner is opt-in.)
- **Migrate, don't accommodate:** delete `cycle_density`, `cycle_benefit`, `tests/test_cycle_density.py`;
  no dual path, no deprecated alias.
- `pixi run check` (ruff lint + mypy --strict + pytest) stays green.
- ruff: no semicolons (E702), ≤100-char lines (E501), `zip(..., strict=…)` (B905).
- New public functions typed for mypy --strict; match surrounding budget.py/methods style.
- Commit trailers and PR-body footer per repo convention.

## 6. Implementation phasing

Two sequenced plans (writing-plans produces them one at a time):

1. **Metric migration.**
   - **Task 0 — scalability gate (blocking):** implement `commute_ratio` on the restored sparse+frozen
     engine (§3.3.1) and benchmark it vs `cycle_density` on a real region block (~2000–3400 parcels,
     per the spike note's `ZAF.9.3.1_1_38528` / 6-block region). Gate: a full `commute_ratio_benefit`
     frontier sweep stays within a small factor of `cycle_density`'s reporting cost (seconds, not
     minutes). If it doesn't, stop and fix before wiring.
   - **Task 1 — metric gate (blocking):** run the §3.2 gate (aggregation, orthogonality exit branch,
     BIG/TINY, corridor-duplication-suite, monotonicity) on the corpus; commit the chosen aggregation
     or fall back to 2ec per the committed rule.
   - **Tasks 2+:** wire `benefit_fn`, ylabel, CSV precision, comment; delete the `cycle_density` path
     (re-homing the `_noded_graph` tests first); re-author `test_compare.py` from the metric; and
     **reconcile the example READMEs + memory** (§3.5) so nothing ships self-contradictory. Ends green.
2. **Loop-closure refiner.**
   - **Task 0 — scalability gate (blocking):** implement the lazy-greedy + incremental-bridge-tree
     core (§4.3) and benchmark one region reblock finishes in seconds, not hours (the naive
     recompute-all is benchmarked at ~hours).
   - **Tasks 1+:** `LoopClosureRefiner` (identity + `_DERIVATION_MODULES` registration + cache-aware
     base) + candidate gen + tests + config + the §4.6 ρ-delta efficacy example. Depends on Plan 1.

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
