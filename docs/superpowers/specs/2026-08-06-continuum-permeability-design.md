# Continuum permeability: conduction on the free space (2026-08-06)

**Status: BLOCKED 2026-08-06 by its own acceptance criterion A5.** `eps` reorders methods at the
sample A5 specifies (21 blocks x 6 methods x 4 eps): 119 rank flips, 3 winner changes, min Kendall
tau +0.600, with flips even inside the 0.5-1.0 band. The "eps is a nuisance parameter" measurement
this spec rests on was an n=10 x 5-method x 1-eps-pair artifact. Real building footprints are now a
PREREQUISITE rather than an improvement: with measured polygons there is no `eps` to choose. See
`notes/2026-08-05-continuum-metric-spike.md`, "A5 GATE 2026-08-06".

**Original status: SPEC'D, not built.** Replaces the parcel-centroid graph metric outright — migrate and
delete, no toggle. Evidence base is six rounds of measurement in
`notes/2026-08-05-continuum-metric-spike.md`; every number quoted here was measured, not assumed.

**Goal in one line:** stop scoring a graph of parcel centroids and score the space people actually
walk through.

## The model

    -div(sigma grad u) = f    in Omega_free,    u = 0 on the street,    P = f^T u

    permeability = 1 - P(roads) / P(no roads)          -- unchanged definition

`Omega_free` is the block minus the building disks. `sigma` is `sigma_walk` in free space and
`sigma_road` inside a road corridor. `f` is the escape demand. Buildings are HOLES, not low-`sigma`
regions.

Discretized on a road-independent grid of spacing `h`: cells whose centre lies in the block and
outside every disk become nodes; 5-point neighbours become edges; the interface conductance is the
harmonic mean of the two cells' `sigma`. For uniform `sigma` a grid edge has conductance
`sigma*h/h = sigma`, independent of `h` — which is why the discretization can converge at all.

### Demand is injected on each building's PERIMETER

One unit per building, spread over the free cells ringing that building's disk. Not at a point: a 2D
point source has log-divergent self-energy, so `P` would grow without bound as `h -> 0`. Perimeter
injection is also what a person does — you leave through a wall. Measured to work: `P0` is stable
across a 4x refinement.

### Ground is a Dirichlet condition, not a shunt

`u = 0` where free space meets the street. "Reaching the street means escaped" is exactly the
metric's semantics, and it **deletes `g_street`** rather than requiring a value for it.

## What this subsumes

Every item below is a current wart that disappears, not a feature to re-implement:

- **D1** (`specs/2026-07-30-road-first-mesh-design.md`) — a zigzag and a straight road covering the
  same parcels score bit-identically at detour ratios to 3.07x. Gone: a zigzag road is a zigzag
  high-`sigma` region, and travel distance is intrinsic to the domain.
- **D2** — travel/crow-flies median 1.395, conductance overstated ~40%. Gone for the same reason.
- **The road-coverage boolean** — no more "does `roads.buffer(w/2)` intersect the centroid-to-centroid
  segment". Roads are regions with higher `sigma`.
- **The node-placement question entirely** — demand is a density. Centroid vs building point vs
  Voronoi vertex stops being a question. (A Voronoi cell's centroid is not even its own generator,
  so today's node is not where the building is.)
- **`_footpath_conductance`'s `(d - r_i - r_j)/d` clearance factor AND its fair-normalization** — a
  channel of width `w` and length `L` conducts `sigma*w/L` automatically in 2D.
- **`FOOTPATH_EPS`**, an arbitrary dimensionless floor on a conductance ratio.
- **`radius_frac`** and **`parcel_radii`'s containment join**.
- **`g_street`** — becomes a boundary condition.
- **The `distance <= STREET_TOL` grounding test.**

Net parameter change: `g_street`, `radius_frac`, `FOOTPATH_EPS` deleted; `eps` (a physical length)
added; `h` added as a numerical resolution. `g_walk -> sigma_walk` and
`g_road_per_m -> sigma_road` carry over unchanged in role.

### Monotonicity stops being a proof

`sigma' >= sigma` pointwise implies `P' <= P` by the Dirichlet principle. Adding a road only raises
`sigma`. That is the whole argument.

This matters more than elegance. Three prior attempts at a better mesh died on monotonicity — a
nearest-road access edge MOVES when roads are added, breaking Rayleigh's nested-edge-set
requirement (`3a8dd25`, values falling ~9%). Here there is no edge set to nest: the grid is a
function of the block and the buildings alone, and roads enter only through `sigma`.

## The parameters, and what is actually known about them

### `eps` — minimum building separation

Building radii are `max(NN/2 - eps, 0.25)`. Without it the model fails outright: `NN/2` makes a
mutual-nearest-neighbour pair have `r_i + r_j = d` EXACTLY, so free space is pinched shut across
**10.0% of adjacent pairs** (18.0% within 0.5 m), the topology fragments without limit (components
climbing 7 -> 21 and 8 -> 24 under refinement), and permeability oscillates +/- 0.013-0.024 with no
plateau. With `eps ~ 0.5-1.0 m` free space collapses to ONE component and stays there.

> **~~`eps` is a NUISANCE parameter, and this is the load-bearing measurement.`~~ FALSIFIED
> 2026-08-06 by A5.** The claim below rested on 10 blocks x 5 methods x ONE eps pair — 50
> comparisons against the 756 A5 asks for. At 21 blocks x 6 methods x 4 eps values: **119 rank
> flips, 3 winner changes, min Kendall tau +0.600**, with flips inside the 0.5-1.0 band too. `eps`
> DOES reorder methods, so it is a free modelling constant, not a nuisance. The original text is
> kept below only to show what was believed and on what evidence.

~~It moves absolute permeability — median 0.0057, max 0.0285 over 20 blocks — and it moves methods
DIFFERENTIALLY (within-block spread of the shift up to 0.0309; `clearance` falls while `flow_paths`
rises on the same block). But over 10 blocks x 5 methods it never reorders a single pair: 0 of 50
rank changes, winner unchanged on 10 of 10, per-block Kendall tau median AND minimum `+1.000`.~~

~~Permeability is used comparatively — both lenses compare methods on one block — so the requirement
is only that `eps` be held FIXED across a comparison.~~

`eps` is a regularization standing in for a measurement. Real building footprints would let it be
measured; `data/provision.py:57` records that Open Buildings polygons exist and were declined on
size (*"points; the polygon variants are 14.09 GB"*).

### `h` — grid resolution, with a per-block check

`h` is numerical, not physical, so the metric must be insensitive to it. Measured: on the common
population every block converges cleanly, `|perm(h=0.35) - perm(h=0.25)| <= 0.0018`, with flat
progressions.

**A rare block does not.** `ZAF.9.3.1_1_41829` still moves 0.045 between the two finest grids at
`eps = 0.5`; raising `eps` to 1.0 improves that 6.7x. Rare enough that a 40-block sample contained
none. Two things follow, and both must be built:

1. **The under-resolved case is SELF-DETECTING.** Its own `h` sweep fails to converge. So the
   implementation must expose a convergence check — solve at `h` and `h/1.4`, and flag the block if
   the two differ by more than a stated tolerance — rather than silently returning a
   resolution-dependent number.
2. That check is a **correctness criterion, not a tuning knob**: a mesh must resolve the geometry it
   claims to represent.

Do NOT reach for a body-fitted triangulation on the strength of this. It would fix the rare
under-resolved case and not the common `eps` sensitivity, which is a modelling effect rather than a
discretization one — a distinction round 5 got wrong and round 6 corrected.

## Displacement must use the SAME radii

If circulation treats a building as `NN/2 - eps` while `displacement` charges `NN/2`, the two axes
disagree about the same geometry. Measured, that disagreement is **not** a uniform level shift:

    method                    disp@eps=0   ratio@0.5   ratio@1.0
    flow_paths_noreinforce        0.2070      0.918       0.823   <- shrinks most
    topology                      0.4184      0.920       0.826
    clearance_looped              0.6419      0.964       0.922   <- shrinks least

    eps=1.0: ratio spans 0.823-0.922 (11.3% of mean); 10% of per-block method ranks flip

The direction matches the mechanism — `topology` and `flow_paths` thread tight gaps, so their
displacement is most sensitive to assumed building size. Decoupling would therefore distort the
Pareto frontier method-differentially. **One radius, both axes.** Published displacement moves
8-18%, which is arguably toward reality since `NN/2` is a packing UPPER BOUND rather than an
estimate.

## Cost

Calibrated from real geometry (median 80.2 m^2 of block per parcel), the 11,006-parcel depth region
implies **3.53M cells at `h = 0.5`**. No `pyamg` / `sksparse` / `petsc4py` is installed and none is
needed:

    measured at 3.42M cells:   spsolve 60.4 s,  peak RSS 6.35 GB    (227 GB available)
    peak RSS scaling:          1.65 / 3.60 / 6.35 GB at 1M / 2M / 3.4M

    spsolve scales as N^1.33;  CG+Jacobi is ~20x worse and its iteration count grows as sqrt(N)

So ~40 min per 40-solve lens curve at region scale, against 3-13 s per solve today — a batch-job
cost, not an interactive one. **Use the DIRECT solver**; reaching for an iterative one is the wrong
instinct here.

## Scope

**In scope:** the metric — `permeability` / `egress_power` and everything they own. The old
parcel-graph path is DELETED, not retained behind a flag.

**Out of scope, and deliberately:** `resistance_greedy` and `resistance_lp` keep optimizing today's
constant-gain parcel-edge model as a first-order PROXY. Their assumption — that a road's benefit to
an edge is knowable before the road set is — has no analogue here, and reformulating the LP is
research. This must be MEASURED (A6) and documented, because a silent proxy is the 2026-07-30 bug
where methods optimized a different Laplacian than the evaluator graded.

**Follow-on spec:** unify `clearance.py`'s cost field with the metric's `sigma`. They become the
same object under this model, which closes the proxy gap properly rather than measuring it.

## Acceptance

Mechanical gate. Agreement with the CURRENT metric is explicitly NOT a criterion: perfect agreement
would prove the change pointless, and disagreement cannot say which is right. Report it, do not gate
on it — the same reason the road-first spec's S1 was called circular.

- **A1 (D1)** A zigzag and a straight road covering identical parcels must score differently.
  Today: bit-identical to 3.07x detour.
- **A2 (D2)** Travel distance is intrinsic; verify on hand-built geometry that a road of known
  detour scores as its true length, not its chord.
- **A3 (monotonicity)** `sigma' >= sigma` implies `P' <= P` on real blocks under incremental road
  addition. FAULT-INJECTED: perturb one `sigma` assignment downward and confirm the test fails.
- **A4 (resolution)** The per-block convergence check exists, fires on `ZAF.9.3.1_1_41829`, and does
  not fire on the 40-block common population.
- **A5 (`eps` is a nuisance)** Extend the ranking test to >= 20 blocks, >= 6 methods, and
  `eps` in {0.25, 0.5, 1.0, 1.5}, including a known under-resolved block. Kendall tau must stay at
  `+1.000`; ANY rank flip means `eps` must be pinned by measurement, not convention, and this spec
  needs revisiting.
- **A6 (proxy gap)** Quantify what `resistance_greedy` loses by searching the old model, on >= 10
  blocks at matched displacement. A recorded number, not a pass/fail.
- **A7 (cost)** Region-scale solve within 2x the measured 60.4 s, and total regeneration time stated
  before any example is regenerated.
- **A8 (lens survival)** Every method's reachability of Lens B's `P* = 0.60` is re-checked. If
  methods stop reaching it, `P*` is re-chosen BEFORE regeneration, not after.

## Blast radius

Every published permeability number changes, and every displacement number moves 8-18%. `P0` is
recomputed under the new model, so unlike the parked road-geometry spec there is no "numerator only"
consolation. Example regeneration is required and should be treated as the expensive, once step it
is.

## What was already falsified — do not re-derive

Six rounds of spike, four of my own hypotheses dead. Recorded so the next attempt starts further on:

- **The medial-axis mesh is not the fix for D1/D2.** The route through the gate between two
  buildings is EXACTLY the straight line (median 1.000 over 4,358 pairs) — geometrically forced,
  since the Voronoi edge is the perpendicular bisector of the two generators. Crow-flies is already
  correct for footpath edges; D1/D2 are about roads.
- **Corners alone are not the continuum.** Connecting a building only to its cell's corners routes
  neighbour trips through chambers: median 1.069 but p90 **1.618**, p99 2.619 against the gate route.
- **The lobe problem is a phantom.** 2,618 parcels, exactly 1:1 with building points, zero
  point-less, zero multipart. Concavity negligible (0.4% exceed 5% concave).
- **Per-cell convex hulls would break the tessellation** — adjacent hulls overlap and spill outside
  the block, and parcels are a PARTITION that `parcel_adjacency` and `displacement` depend on.
- **Proposed roads must NEVER enter the tessellation.** The mesh would change with the road set and
  Rayleigh would give nothing — the moving-edge-set failure by a new door.
- **`pinched_frac` does not predict `eps` sensitivity** (r = +0.104; it is 0.09-0.10 on every block).
- **Street frontage does not either** — convincing at n=6 (r = -0.640) and dead at n=20
  (Spearman +0.027, p = 0.91). Nor does block size (+0.039, p = 0.87).
- **`eps` and `h` sensitivity are NOT the same error** (Spearman +0.214, p = 0.61; tail/control
  h_sens ratio 0.9x), so a triangulation does not address the common case.
- **4-neighbour "anisotropy" is not a concern for conduction** — the 5-point stencil is a consistent
  isotropic discretization of the Laplacian. Anisotropy bites shortest-PATH problems.

## Known gaps in the evidence

Stated so acceptance is not mistaken for coverage. All spike work used `clearance` except the
ranking test (5 methods) and the displacement test (6). Blocks were Cape Town, <= 250 parcels. The
`eps` ranking result spans only {0.5, 1.0} and drew no known outlier — which is exactly what A5
exists to close.
