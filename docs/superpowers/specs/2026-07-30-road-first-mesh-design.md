# Road-first mesh: making the metric represent roads as objects

**Status: DO NOT BUILD AS WRITTEN.** Red-teamed 2026-07-30 by four independent reviewers. The four
DEFECT claims in section 2 were all verified true and quantified -- but the PROPOSAL in section 3 is
a third attempt at a model this repo already built and retired twice, it breaks the monotonicity
permeability depends on, and two of its three acceptance criteria are unachievable. See section 7
before reading anything below it as a plan.

> **D3 LARGELY FALSIFIED 2026-08-04 -- and it was OUR bug, not the metric's.** The floating fragments
> D3 counted were manufactured by `osm_footpaths`' own street-corridor subtraction, which cut every
> imported path back to *exactly* the distance `street_connectivity` tests against. Fixed at source.
> **D1 and D2 are now the whole case for touching the mesh.** See
> [D3 was our own artifact](#d3-was-our-own-artifact-2026-08-04).

Written after the one-way thread (`notes/2026-07-30-oneway-half-width.md`) hit a wall that turned
out to be about the mesh rather than about one-way.

**Motivating claim in one line:** `permeability` does not model roads. It models *which parcel pairs
are near one*.

## 1. What the current mesh actually is

From `permeability.egress_power` / `_mesh`:

- **Nodes** are parcel centroids. There are no road nodes.
- **Edges** are parcel-adjacency pairs. There are no road edges.
- An edge's length is `d = hypot(cx_i - cx_j, cy_i - cy_j)` -- the **straight-line centroid-to-
  centroid distance**.
- A road enters through exactly one boolean: does `roads.buffer(corridor_m)` intersect that
  centroid-to-centroid segment. If yes the edge's conductance becomes `g_road / d`, otherwise it
  keeps `_footpath_conductance(d, r0, g_walk)`.
- Ground is a shunt on the diagonal of any parcel within `STREET_TOL` of `block.streets` -- the
  PRE-EXISTING street only. A method's own roads never become ground.

## 2. The four defects that follow, none of which is about one-way

1. **Road length and shape never enter the metric.** They appear only inside a boolean. Two very
   different roads covering the same parcel pairs score *identically* on permeability; only
   `displacement` distinguishes them. A method is therefore free to draw a long, winding, expensive
   road and be scored as though it drew the short straight one.
2. **Travel distance is wrong by construction.** Conductance is `g_road / d` with `d` a crow-flies
   centroid distance. Actual travel is along the road, which may be far longer. The metric prices a
   trip that nobody can take.
3. **Road-network connectivity is invisible.** A floating fragment still upgrades the adjacency edges
   it covers, so it still raises permeability while granting no access at all. This is the
   permeability-vs-access-depth inconsistency measured in
   `notes/2026-07-30-egress-vs-circulation.md` (`osm_footpaths` on one region: 14 road components,
   70.8% of length street-connected, and all 136 adjacent-but-deep parcels touching only floating
   fragments). Under a road-first mesh this needs no special rule -- a disconnected road connects
   nothing, so it conducts nothing.

   **LARGELY FALSIFIED 2026-08-04.** That 70.8% was our own artifact: `osm_footpaths` subtracted a
   `STREET_TOL` street corridor and `street_connectivity` then tested against `STREET_TOL`, so a
   footpath drawn meeting the street was cut back to sit exactly on the threshold. Fixed at source
   (`osm_footpaths._reach_street`); connected length is now 87.5% and, decisively, is **flat from
   0.5 m to 5 m** where it used to jump at 0.51. The mechanism in this bullet is still real -- a
   genuinely floating road really does conduct -- but it was firing mostly on damage we inflicted.
4. **No capacity or hierarchy.** Every covered edge gets the same `g_road / d`. An arterial and an
   alley are indistinguishable. There is no way to express width, and hence no way to express the
   one-way/half-width idea that started this.

Defect 4 is the one-way motivation. **Defects 1-3 stand entirely on their own**, and are the reason
to do this. One-way should be treated as a possible follow-on, not the justification.

## 3. Proposed model

Nodes: parcel centroids (as now) PLUS road-network nodes (junctions and endpoints of the proposed
roads, planarized against the existing street).

Edges:
- **road edges** between road nodes, conductance `g_road / L` with `L` the road's own length --
  the first time actual road geometry enters the metric;
- **access edges** joining each parcel to the nearest point on a road (or street) it fronts,
  conductance over the real access distance;
- **footpath edges** between adjacent parcels, unchanged (`_footpath_conductance`), so the
  no-roads baseline and the r0-corridor calibration survive;
- **ground** as now: a shunt on parcels within `STREET_TOL` of the pre-existing street. The street
  itself becomes part of the road graph, so a proposed road that reaches it is connected to ground
  through road edges rather than by a coincidence of adjacency.

Everything downstream (`b = ones`, `P = b^T L^-1 b`, `permeability = 1 - P1/P0`, monotonicity by
Rayleigh) is unchanged -- this replaces the graph, not the metric.

## 4. Acceptance criteria, stated BEFORE building

The all-pairs probe (`notes/2026-07-30-egress-vs-circulation.md`) closed a metric change because it
reproduced the same ranking at real cost -- Kendall tau +0.800, same winner 10/12. The same bar
applies here, and this section exists so it is checked early rather than after the work.

**Necessary (correctness -- these are the reasons to build it):**

- **C1.** Two road sets covering the same parcel pairs but with different total length must score
  differently. Today they score identically. Test: take a method's output, replace a straight road
  with a winding road between the same endpoints, confirm permeability falls.
- **C2.** A road component not connected to the street must contribute nothing. Test: the
  `osm_footpaths` case above -- its permeability must fall once floating components stop conducting,
  and the drop should be commensurate with the 29.2% of its length that is disconnected.
- **C3.** The no-roads baseline `P0` must be unchanged, so `permeability` remains comparable in
  meaning (footpath mesh untouched).

**Sufficient (is it worth shipping):**

- **S1.** The method RANKING must change on at least one lens, or this is the all-pairs situation
  again -- real cost, same answer. Measure Kendall tau against today's ranking on >= 20 blocks; a
  tau near +1.0 with no winner changes is a stop signal, not a success.
- **S2.** The two-lens comparison must be re-run end to end and the examples regenerated, since
  every published number moves.

**Recalibration required before any of the above means anything:** `g_road`, `g_walk`, `r0_frac`
and `corridor_m` were tuned for the parcel-adjacency mesh (the r0-corridor work that took method
spread from ~0.9pts to ~11.8pts). Road edges have different length scales, so the road/footpath
balance must be re-derived, not inherited. **Comparing a recalibrated new mesh against an
un-recalibrated old one would measure the calibration, not the model.**

## 5. Risks

- **Cost.** More nodes and edges per block, plus the planarization. Region scale (11,006 parcels)
  needs checking before this can sit inside a lens.
- **Access-edge definition is a modelling choice with teeth.** "Nearest point on a road it fronts"
  needs a rule for parcels fronting several roads, and for parcels fronting none. Getting this wrong
  reintroduces defect 3 in a new form.
- **Methods emit only `roads`.** Nothing in `Proposal` carries width or direction, so defect 4 stays
  unaddressed until the contract changes. That is deliberate -- it keeps this spec about 1-3.
- **Sample sizes.** Several probes in the one-way and prefix-order threads reversed between n=6 and
  n=20 (see `backlog.md`). Nothing here should be concluded below n=20 blocks.

## 6. Explicitly out of scope

One-way streets, width, capacity, and directed ingress+egress. The prerequisite for all of them is
this mesh; bundling them would make the correctness work (defects 1-3) hostage to an idea whose own
payoff is still unvalidated -- the width half is gameable alone and the directed half showed no
interior optimum on the disc fixture.


## 7. Red-team verdict (2026-07-30) -- what survives and what does not

Four independent reviewers with repo access, each required to cite file:line or a measured number.

### The defect claims (section 2) all hold, and are now quantified

Measured over 24 real blocks:

- **D1 length/shape invisible -- TRUE.** A straight road and a zigzag covering an identical
  covered-edge set score **bit-identically**, at detour ratios up to 3.07x. Nit: the published curve's
  x-axis IS cumulative road length (`budget.py:692`), so length is not invisible in *reporting*.
- **D2 crow-flies distance -- TRUE.** Travel/crow-flies on covered edges: **median 1.395**, per-edge
  max 2.63. Conductance is overstated ~40%.
- **D3 floating fragments conduct -- TRUE as a mechanism, but LARGELY FALSIFIED as a defect
  (2026-08-04).** Trimming a road's street end gives `street_connectivity` 0.000 and improves access
  depth for **0 parcels** (vs 1,426 connected), yet retains a median **99.3%** of permeability. That
  experiment still stands: permeability *is* blind to street-connectedness. What does not stand is
  the population it was extrapolated to -- see below.
- **D4 no capacity/hierarchy -- TRUE.** One scalar `g_road`, one `corridor_m`, no width or direction
  on `Proposal`.

### Why the PROPOSAL is nonetheless rejected

1. **It is the third attempt at a retired model.** `budget._road_street_graph` -- road segments as
   edges, parcels attached by line-proximity, i.e. section 3's model -- shipped and carried
   `network_efficiency` / `directness` / `resistance_benefit`. Deleted in `180bbf6`. It returned as
   `commute_ratio` on the planarized road-street graph and was retired again by permeability
   (`specs/2026-07-22-permeability-metric-design.md`). This spec cited none of it.
2. **It breaks monotonicity, which section 3 claims it preserves.** Rayleigh needs a NESTED edge set;
   a nearest-road access edge MOVES when roads are added. That is the documented bug fixed by
   `3a8dd25 fix: network_efficiency monotone via fixed entry mapping` (values could FALL, ~9% drops).
   Freezing entries fixed the efficiency form; for the RESISTANCE form it did not
   (memory `commute-ratio-monotonicity-fundamental`: every fully monotone variant became gameable,
   multi-frontage worst). Permeability's monotonicity is load-bearing, stated in its module docstring.
3. **C2 is false** -- three reviewers independently. Footpath edges survive, so parcels fronting a
   floating fragment give it access edges and it becomes a parcel->road->parcel bypass. Measured:
   `osm_footpaths` scores 0.1040 keeping floating components vs 0.0074 dropping them -- **93% of its
   score comes from road that reaches no street.** Enforcing C2 needs exactly the explicit strict rule
   `notes/2026-07-30-egress-vs-circulation.md` argues is backwards.
   **The 93% is mostly artifact (2026-08-04)** -- see below. The reasoning is untouched; the number
   it was measured on is not.
4. **C3 is unachievable.** P0 is highly sensitive to the parameters section 4 mandates re-deriving
   (g_walk 0.1->0.2 halves P0; r0_frac 0.55->0.80 cuts it 3.4x), and making the street part of the
   road graph replaces the `g_street` shunt with a near-short access edge, moving P0 again.
5. **S1 is circular.** `scripts/calibrate_permeability.py:383` selects by "widest mean cross-method
   permeability spread". Calibrating for method spread and then accepting on ranking change is
   nearly the same operation twice. Spread is also not a correctness proxy, and the calibration used
   the same example regions the comparison publishes.

### What was RIGHT that the spec got wrong in the other direction

**Cost is a reason to build, not a risk.** Section 5 warned about scale; measured on the real
`multiblock_depth` region (11,006 parcels), the proposed mesh is 1.1-1.8x the nodes and 1.07-1.53x
the edges, and a solve is **0.15-0.5 s against 3-13 s today** -- 5-40x FASTER, because today's cost is
a Python loop calling `corridor.intersects` 31,395 times per solve. That is a genuine finding about
the CURRENT metric independent of any redesign.

### Concrete traps for any future attempt

- **Ungrounded components make L singular.** Measured planarized components: `euclidean_grid` 62,
  `osm_footpaths` 65, LP 35; `spsolve` returned NaN for grid and osm. Pruning is mandatory.
- **"Fronts" is a free parameter that moves a third of the region.** Parcels fronting a road:
  30.1% at 0.5 m, 55.1% at 10 m, 74.1% at 25 m. Parcels touching >=2 roads at 3 m: LP **58.8%**.
  Multi-frontage is the common case -- and it is the documented gameable branch.
- **Two incompatible graph conventions already coexist**: `_road_net` (raw `_rnd` keys) vs
  `_noded_graph` (`unary_union`). Measured disagreement on the LP: **521 raw components vs 35
  planarized**. `street_first_ordered` and `road_drainage` -- which generate every scored prefix --
  use the raw one.
- **Slivers.** Planarized minimum segment is 0.0100 m (the `_rnd` floor), giving `g_road/L = 2000`
  against a median 3.2 -- six orders of conditioning. The 2026-07-17 redundancy spec already hit this.

## D3 was our own artifact (2026-08-04)

The prompt was a hypothesis about the DATA: *maybe OSM footpaths are simply drawn stopping a couple
of metres short of the street they obviously meet, so the "floating" fragments physically reach it
and permeability's tolerance is a feature.* The hypothesis is right about the conclusion and wrong
about the culprit. The gap is not OSM's. It is ours.

`osm_footpaths.interior_desire_lines` subtracts `streets.buffer(tol)` so an imported path does not
retrace the perimeter street it already runs along -- legitimate, and `tol` defaults to `STREET_TOL`.
`derive/access.street_connectivity` then asks whether a component comes within `STREET_TOL` of a
street. **Same constant on both sides.** A footpath drawn meeting the street is cut back to lie
exactly on the corridor boundary, and whether it then reads as connected is decided by whether
`0.5000000000000001 <= 0.5`.

Measured over 10 Cape Town blocks carrying interior footpaths (`scratchpad/width/`):

    footpath-to-street gap        touching street (<=0.5 m)   median gap
      RAW, as mapped                      71.4%                 0.00 m
      after interior_desire_lines         42.4%                 0.50 m

    street-connected fraction of footpath LENGTH, by connectivity tolerance
      0.50 m   0.700   (4/10 blocks fully connected)   <- the subtraction width
      0.51 m   0.875   (8/10)
      5.00 m   0.875   (8/10)

A **one-centimetre** move takes connected length from 70.0% to 87.5%, and then nothing changes out
to 5 m. A real tolerance question produces a curve; this is a step at one value, which is the
signature of a threshold collision.

**Fixed at source.** `_reach_street` extends any end the subtraction cut back to the street. The
connector is at most `tol` long and lies inside the street corridor, so it displaces nothing; a path
running ALONG the street still loses its whole overlapping length. Post-fix the connected fraction is
0.875 **flat from 0.5 m to 5 m** -- the result no longer depends on where the seam is drawn. Three
guards pin it (`tests/methods/test_osm_footpaths.py`), fault-injected: removing the reconnection
fails two of them at `0.5 == 0.0`.

Shrinking `tol` below the access seam would NOT have worked, which is why the fix is a reconnection
rather than a constant. `tol` also sets how far off the kblock outline an OSM way may run and still
count as the same feature; the census sweeps it *upward* for that reason. The two quantities want to
move in opposite directions, so they must not be coupled through a shared threshold at all.

### What this changes

- **Do not build a street-connectivity gate on the road upgrade.** It was the cheap fix for D3 that
  kept the existing mesh. Its motivating measurement is ~85% artifact, and the gate would have
  converted a floating-point coin flip into a 93% score swing on the one method that imports real
  as-built circulation.
- **Permeability's tolerance to the gap was protecting us.** A metric that shrugs when a pipeline
  severs a road from its street by half a metre is robust. D3 read that robustness as blindness.
- **D1 and D2 are now the entire case for touching the mesh** -- and they are structural, not
  threshold artifacts: a zigzag scoring bit-identically to a straight road at 3.07x detour, and
  travel/crow-flies median 1.395.
- **The residual is 12.5% of footpath length**, genuinely disconnected at any tolerance out to 5 m
  (component gaps of 13.4 m and 20.0 m). Small enough that it does not by itself justify a mechanism.
- n = 10 blocks, Cape Town, one method. Small -- but a 1 cm threshold flipping 4 of 10 blocks is not
  a sample-size question, and the fix is verified by the step disappearing rather than by the level.

### Where to look next: the nodes, not the road model

The mesh's other half is untouched by all of this. Nodes are parcel CENTROIDS -- and a Voronoi cell's
centroid is not even its own generator, so today's node is not at the building. People walk in the
gaps between buildings, not through their centres, which is exactly why edge length is crow-flies
centre-to-centre and `_footpath_conductance`'s `(d - r_sum)/d` has to exist to compensate. D2's
median 1.395 IS that compensation failing.

The candidate is the medial axis of the building points: **Voronoi vertices as junctions** (locally
widest clearance), **Voronoi edges as passages** (length = the real walk), and the **osculation point
of the two guessed circle footprints as each passage's gate** (aperture `d - r_i - r_j`). Demand
stays injected at building points, spurred to their own cell's corners.

Critically, this is NOT a fourth attempt at the retired road-first graph, and it fails differently
from the three attempts in "Why the PROPOSAL is rejected". Those died on monotonicity: a nearest-road
access edge MOVES when roads are added, so Rayleigh's nested-edge-set requirement breaks. The Voronoi
skeleton is a function of the building points ALONE -- roads never touch it. Road upgrades stay
weight-only through the same `max(footpath, road)` in `edge_conductances`, so the monotonicity
argument survives intact. That is the specific property the rejected proposals lacked.

### Bonus: a real bug in shipped code, found and fixed

`resistance_greedy.py:200` and `resistance_lp.py:247` built their mesh with
`parcel_adjacency(geoms, corridor_m)` (3.0) while `egress_power` scores at `STREET_TOL` (0.5) -- a 6x
looser adjacency, so both methods optimized a **different Laplacian than the evaluator grades**,
contradicting `_mesh`'s own docstring. Fixed 2026-07-30.
