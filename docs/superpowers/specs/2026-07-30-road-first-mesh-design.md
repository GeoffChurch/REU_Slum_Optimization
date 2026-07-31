# Road-first mesh: making the metric represent roads as objects

**Status:** spec, nothing built. Written after the one-way thread
(`notes/2026-07-30-oneway-half-width.md`) hit a wall that turned out to be about the mesh rather
than about one-way.

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
