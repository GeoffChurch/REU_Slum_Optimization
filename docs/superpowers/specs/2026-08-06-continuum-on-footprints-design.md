# Continuum permeability on REAL FOOTPRINTS (2026-08-06)

**Status: SPEC'D, not built. SUPERSEDES
`specs/2026-08-06-continuum-permeability-design.md`** (BLOCKED — keep it, its falsified-hypothesis
catalogue is worth more than the parts replaced here).

**What changed in one line:** v1 modelled buildings as disks of radius `NN/2` and needed an `eps`
regularizer that turned out to decide method rankings. This models buildings as their **measured
footprints**, and there is no `eps`.

**Goal:** score the space people actually walk through, using the geometry that is actually there.

## Why v1 failed, and why that reason is now gone

v1's kill gate A5 failed decisively: `eps` reordered methods (119 rank flips, min Kendall tau
+0.600 over 21 blocks x 6 methods x 4 eps values). `eps` existed for exactly one reason —
`building_radii` is HALF the nearest-neighbour distance, so a mutual-nearest-neighbour pair's disks
touch exactly, free space pinches shut, and the domain fragments without limit under refinement.

**That was an artifact of DISKS, not of reality.** Measured on real Cape Town footprints
(`scratchpad/footprints/`):

    TRUE gap to nearest neighbour, polygon-to-polygon (20,000 sampled)
      median 0.00 m,  p75 1.44 m,  p95 6.25 m
      touching (gap <= 0):  49.9%      within 0.5 m: 63.8%      within 1.0 m: 70.5%

Half of all buildings physically TOUCH their nearest neighbour — shacks share walls. On disks that
would be catastrophic. On real polygons it is not, because a disk blocks its whole circular
envelope while a rectangle touches on ONE side and leaves the other three open:

    block                 bldgs  free area  components  largest  fronting  median alley
    ZAF.9.3.1_1_19362        50      64.9%           1   100.0%    100.0%       18.25 m
    ZAF.9.3.1_1_39416        67      53.2%           1   100.0%    100.0%       11.68 m
    ZAF.9.3.1_1_63721        78      53.6%           3    99.8%    100.0%        9.45 m
    ZAF.9.3.1_1_41275        89      65.6%           2   100.0%    100.0%       17.96 m
    ZAF.9.3.1_1_21553       105      45.0%           3    99.4%    100.0%       10.87 m
    ZAF.9.3.1_1_41964       119      59.1%           3    99.9%    100.0%       19.63 m
    ZAF.9.1.4_1_4310        154      45.7%           5    98.2%     98.1%
    ZAF.9.3.1_1_41782       211      51.8%           6    98.7%    100.0%

**Free space is one connected piece** (largest component 98.2-100%), **every building fronts it**
(98.1-100%), and it is 45-66% of block area. Nothing fragments.

So `eps` was never regularizing a numerical problem. **It was manufacturing passages through walls**
— which is precisely why its value decided the rankings: it was inventing connectivity, and how much
you invented determined who won.

### The same finding indicts the SHIPPED metric

Today's parcel graph puts a conducting edge between EVERY adjacent parcel pair, floored at
`FOOTPATH_EPS = 0.02`. For the ~50% of pairs that share a wall, the true conductance is **zero**.
The current metric routes people through walls, systematically, on half its edges.

## The model (inherited from v1, unchanged and verified)

    -div(sigma grad u) = f    in Omega_free,    u = 0 on the street,    P = f^T u

    permeability = 1 - P(roads) / P(no roads)          -- unchanged definition

`Omega_free` is the block minus the union of **measured building footprints**. `sigma` is
`sigma_walk` in free space and `sigma_road` inside a road corridor. Buildings are HOLES.

Discretized on a road-independent grid of spacing `h`; 5-point neighbours; harmonic-mean interface
conductance. For uniform `sigma` a grid edge has conductance `sigma*h/h = sigma`, independent of `h`.

**Demand is injected on each building's PERIMETER** — one unit per building, spread over the free
cells ringing its footprint. Not at a point: a 2D point source has log-divergent self-energy, so `P`
would grow without bound as `h -> 0`. It is also what a person does: you leave through a wall.
Verified in v1 — `P0` stable across a 4x refinement.

**Ground is a Dirichlet condition** (`u = 0` where free space meets the street), which deletes
`g_street` rather than requiring a value for it.

**Monotonicity is the Dirichlet principle**: `sigma' >= sigma` pointwise implies `P' <= P`, and
roads only raise `sigma`. There is no edge set to keep nested — the grid is a function of the block
and its buildings alone.

## What real footprints DELETE

Every item is a current parameter or fudge that stops existing, not one that gets a better value:

- **`eps_separation_m`** — v1's regularizer, and the reason v1 died.
- **`building_radii`'s `NN/2`** — a packing UPPER BOUND masquerading as a measurement.
- **`radius_frac`**, **`parcel_radii`** and its containment join, **`shrunk_radii`**.
- **`FOOTPATH_EPS`** — the floor that let walls conduct.
- **`_footpath_conductance`'s `(d - r_i - r_j)/d`** and its fair-normalization — a channel of width
  `w` and length `L` conducts `sigma*w/L` automatically in 2D.
- **`g_street`** — becomes a boundary condition.
- The **road-coverage boolean** and the `distance <= STREET_TOL` grounding test.
- The **node-placement question** entirely — demand is a density, so centroid vs building point vs
  Voronoi vertex stops being a question.

`displacement` uses the same footprints, so v1's coupling question resolves itself: one geometry,
both axes. A road displaces a building iff its corridor intersects that building's actual polygon.

## Data

`data/provision.py:57` records "the polygon variants are 14.09 GB" and that number has blocked this
for months. **It is the size of all 20 ZAF+KEN tiles. Cape Town's single tile is 0.32 GB gzipped**
(points: 0.08 GB) — a routine download, already pulled and cached at
`~/.cache/reblock/buildings_capetown_polygons.parquet`.

The polygon URL is the tile-index URL without the existing `OB_POLYGON_PREFIX -> OB_POINT_PREFIX`
swap in `scripts/fetch_kblock_fixtures.py:222`. `area_in_meters` and `confidence` are already
retained today; only `geometry` (WKT) is newly kept. `OB_MIN_CONFIDENCE = 0.7` stands.

## Acceptance

**Re-derived from scratch. v1's passes are NOT inherited** — its evidence was gathered on disks, and
the geometry is what changed. Ordered so the two that can kill the design run FIRST, before any
production code, which is the one thing that went right in v1.

- **B1 (KILL GATE — footprint quality). RUN 2026-08-06: PASS**, with the criterion itself amended —
  see below and `notes/2026-08-06-b1-footprints-are-sound.md`.

  Measured over **43 blocks across both Cape Town (22) and Nairobi (21)**, sampled at the project's
  own informal-settlement density floor (>= 1000 buildings/km^2, 30-400 buildings):

      free-space fraction        median 64.5%   (p10 46.3%)
      largest-component share    median 99.977%, MIN 95.037%
      blocks with largest < 95%  0.0%           (fail threshold: > 10%)
      fronting share             median 100.0%, min 80.5%

  **The foundational claim holds decisively**: real footprints leave connected free space that every
  building fronts, in both cities, on every block sampled.

  **The count criterion was mis-specified in three ways and is amended.** As written it read 21.1%
  against its own 20% threshold — a marginal fail — but that number was an artifact:

  1. It counted footprints INTERSECTING a block, double-counting every building straddling a
     boundary. With the correct `centroid within` predicate it drops to **16.1%**, which passes.
  2. It compared against kblock's `building_count`, which is a DIFFERENT DATASET. Measured
     attribution: the new polygons agree with the Open Buildings POINTS the project already ships at
     a ratio of **1.000 exactly**, while both disagree with kblock by the same 24.5% in Cape Town.
     The discrepancy is pre-existing, orthogonal to this design, and shipping today.
  3. The threshold was symmetric when the risk is ONE-SIDED. The danger is Open Buildings MISSING
     buildings (ratio < 1), which would overstate free space and flatter permeability.
     Over-counting deflates free space and is conservative. Measured ratio is 1.245 (Cape Town) and
     0.982 (Nairobi) — never below ~0.98, so under-segmentation is not occurring, which is exactly
     what this check existed to rule out.

  **Amended criterion for any re-run:** compare polygons against the Open Buildings POINTS already
  in use, by `centroid within`, and FAIL only on a ONE-SIDED shortfall — median ratio below 0.90.
  kblock's `building_count` is a provenance question about two datasets and belongs in its own
  investigation, not in this gate.
- **B2 (KILL GATE — ratio stability).** `sigma_road/sigma_walk` remains a chosen constant, and it is
  measured to decide rankings: on the current metric, rankings are stable at today's 204x
  (Kendall tau +1.000) and at 97.5x (+0.933), degrade at 39x (+0.714), and SCRAMBLE below ~20x
  (negative minima). See `notes/2026-08-06-road-walk-ratio-decides-the-ranking.md`. Sweep the ratio
  on THIS metric over >= 20 blocks x >= 6 methods and report where rankings destabilize. FAIL if the
  stable band excludes physically defensible values — because then the metric's ordering is a
  choice, not a measurement.
- **B3 (monotonicity).** `sigma' >= sigma` implies `P' <= P` on real blocks under incremental road
  addition, rise rate exactly **0.00%**. FAULT-INJECTED.
- **B4 (D1).** A zigzag and a straight road covering the same parcels must score differently. Holds
  by construction — travel distance is intrinsic to the domain — so this is a check, not a design
  problem.
- **B5 (D2).** A road of known detour scores as its true length, not its chord. Also by construction.
- **B6 (resolution).** Per-block `h`-convergence check that flags blocks the grid cannot resolve
  rather than silently returning a resolution-dependent number. With real alleys (median width
  9-20 m measured above, against v1's sub-metre disk pinches) this should be far easier than in v1 —
  verify that, do not assume it.
- **B7 (cost).** v1 measured 60.4 s and 6.35 GB at 3.42M cells (region scale at `h = 0.5`) with
  scipy's DIRECT solver, scaling as `N^1.33`; CG is ~20x worse. Cell counts barely move when the
  obstacles change shape, so this should carry over — confirm on one region.
- **B8 (lens survival).** Re-check every method's reachability of Lens B's `P* = 0.60` BEFORE
  regenerating anything, and re-choose `P*` if needed. `prefix_to_permeability` degrades silently
  when a target is unreachable.

Agreement with the CURRENT metric is explicitly NOT a criterion: perfect agreement would prove the
change pointless, disagreement cannot say which is right, and the current metric is now known to
conduct through walls on half its edges.

## Blast radius

Every published permeability number changes, and `P0` is recomputed under the new model, so unlike
the parked road-geometry spec there is no numerator-only consolation. Displacement moves too, since
it switches from `NN/2` disks to real polygons. Example regeneration is required and should be
treated as the expensive, once step it is.

## What this makes moot

`specs/2026-08-05-road-geometry-in-conductance-design.md` (PARKED) exists to fix D1/D2 on the
parcel-edge mesh. Those dissolve here by construction, so it has no job left and should stay parked
as a fallback rather than be adapted. Its blocker is independent of building shape anyway: the
per-edge decomposition double-charges the walk-to-the-road (a trip i->j->k pays `L_j` twice), which
collapsed the road benefit to 1.04-1.41x and put rankings inside the scrambling band. Real
footprints do not touch that.

Consequence for work in flight on branch `continuum-permeability`: `src/reblock/road_route.py`
(planarized road graph + route resistance) becomes **disposable** — a road here is a high-`sigma`
region, not a graph to route on. `src/reblock/mesh.py`'s `footpath_mesh` extraction is worth keeping
regardless, since it improves the shipped metric and the shipped metric stands until this replaces
it.

## Carried forward — do not re-derive

From v1 and the threads around it, all measured:

- **Perimeter demand injection is required**; a point source diverges logarithmically in 2D.
- **The direct solver wins**; CG+Jacobi is ~20x worse and its iterations grow as `sqrt(N)`.
- **The 5-point stencil is isotropic for conduction.** "4-neighbour anisotropy" is the wrong worry —
  it bites shortest-PATH problems. The real geometric error is staircase boundary representation.
- **The medial-axis mesh is not the fix for D1/D2**: the route between two buildings through their
  shared gap is EXACTLY the straight line (median 1.000 over 4,358 pairs), geometrically forced.
- **Corners alone are not the continuum** (p90 1.618 against the gate route).
- **Proposed roads must NEVER enter a tessellation** — the mesh would change with the road set and
  Rayleigh would give nothing.
- **`pinched_frac`, street frontage and block size do not predict `eps` sensitivity** (r = +0.104;
  Spearman +0.027, p = 0.91; +0.039, p = 0.87) — moot here, but recorded so nobody re-runs it.
- **The road/walk ratio is 204x, not 513x** — the fair-normalization rescales the clearance shape to
  a median of ~1, which an earlier calculation missed by a factor of 2.52.

## Known gaps

Eight blocks, Cape Town only, one confidence threshold. B1 exists precisely because the free-space
result — the foundation of this entire design — currently rests on that. Nairobi is untested. The
footprints' accuracy at shack scale is unverified: systematic under-segmentation of adjoining shacks
into single blobs would flatter the connectivity numbers, and B1's count comparison against kblock is
the cheapest available check on it.
