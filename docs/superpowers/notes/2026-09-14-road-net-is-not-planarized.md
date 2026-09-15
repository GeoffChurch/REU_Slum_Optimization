# `_road_net` never nodes road-road crossings, and that decides the Grid's published rows (2026-09-14)

`budget._road_net` builds the road graph by keying nodes on `_rnd`-snapped segment ENDPOINTS. Two
roads that cross mid-segment therefore share no node, and **the junction does not exist** for
anything downstream. That is deliberate and documented:

> Nodes are `_rnd`-snapped segment endpoints -- deliberately NOT `unary_union`-noded, because that
> re-nodes geometry into vertices no `_rnd` key matches.
> -- `src/reblock/budget.py:157`

The stated reason is real (unioning re-nodes the geometry out from under the `_rnd` keys), but the
conclusion does not follow: a graph can be planarized *without* touching the road rows, by splitting
each segment at its crossings and keying the split points the same way. Nothing required the
junctions to be dropped.

**What consumes that graph:** `road_drainage` -> `street_first_ordered` -- THE canonical prefix
order that both published lenses truncate on, that `_sweep` builds every frontier curve from, and
that `animate` grows the GIFs in. The backlog already records prefix ordering as the single largest
lever measured in this project ("Lens prefix selection", a 12x swing on `resistance_lp`). This is
that lever being pulled by an artifact.

## How it surfaced

From a question about whether the reblockers build loops. The recorded bridge fractions -- quoted
in `src/reblock/methods/cycle_native.py:22` and in the backlog's orientation section -- run through
the same graph, so they inherit the same blindness. Re-measured on the pinned block
`ZAF.9.3.1_1_40972` (263 parcels), with the roads left alone and only the graph planarized
(`scratchpad/noding_check.py`):

| method | bridge frac, as recorded | properly noded | unnoded crossings |
|---|---|---|---|
| Direct Objective (LP) | 1.0000 | **0.8349** | 10 |
| Grid | 1.0000 | **0.3885** | 10 |
| Frontage (street-priced) | 0.1899 | 0.1899 | 0 |
| Looped Tree | 0.5253 | 0.5253 | 0 |
| Loop Network | 0.0168 | 0.0168 | 0 |
| Least-Cost Tree | 1.0000 | 1.0000 | 0 |

So two of the recorded figures are wrong in the same direction: **the Grid is ~61% loops, not 0%,
and the LP is ~17% loops, not 0%.** `cycle_native`'s distinguishing claim survives -- it is still
the lowest by a wide margin -- but "`resistance_lp` has the HIGHEST bridge fraction" and the
backlog's "`cycle_native` 0.000 / `clearance_looped` 0.577 / `resistance_lp` 1.000" should be read
as artifacts of the instrument, not properties of the methods.

## The mechanism: drainage collapses, and the order it drives goes with it

`scratchpad/grid_order_diag.py`, same block:

    euclidean_grid (9 roads, 741 m)
      graph nodes/edges          shipped  17/9      planar  27/29
      connected components       shipped  8         planar  1
      roads with no path to the street   shipped  4         planar  0
      per-road drainage          shipped  [2,0,0,1,0,2,0,1,0]
                                 planar   [2,12,2,6,4,9,3,3,3]
      total drainage             shipped  6         planar  44

The shipped graph sees the Grid as **eight disconnected sticks**, four of which cannot reach the
street at all. Drainage -- parcel routes counted per road -- collapses from 44 to 6, so six of the
nine roads score 0 and the priority order among them is effectively arbitrary.

**This is a drainage-QUALITY failure, not a connectivity failure.** Both arms' scored prefixes
measure 1.000 street-connected under the truthful graph, so `street_first_ordered`'s stated
guarantee ("every prefix is connected") is not violated. What breaks is the part that decides
*which* roads come first.

## The lens impact

Both published lenses, same roads, same rows, only the graph planarized
(`scratchpad/lens_noding_impact.py`; D = 0.10, P* = 0.60 from `conf/permeability.yaml`):

| method | Lens A perm (shipped -> planar) | Lens A road m | Lens B homes (shipped -> planar) | Lens B road m |
|---|---|---|---|---|
| **Grid** | **0.7071 -> 0.8158** (+0.109) | 270.0 -> 188.6 | **0.1485 -> 0.0973** (-34%) | **270.0 -> 130.1** (-52%) |
| Direct Objective (LP) | 0.8876 -> 0.8876 | 400.9 -> 406.7 | 0.0477 -> 0.0477 | 146.6 -> 146.6 |
| Least-Cost Tree | 0.6913 -> 0.6913 | 143.0 -> 143.0 | 0.0792 -> 0.0792 | 89.4 -> 89.4 |
| Looped Tree | 0.6642 -> 0.6642 | 109.7 -> 109.7 | 0.0720 -> 0.0720 | 82.6 -> 82.6 |
| Loop Network | 0.7697 -> 0.7697 | 243.1 -> 243.1 | 0.0440 -> 0.0440 | 92.7 -> 92.7 |
| Frontage (street-priced) | 0.8154 -> 0.8154 | 339.3 -> 339.3 | 0.0361 -> 0.0361 | 170.0 -> 170.0 |

**On this block the Grid is under-reported and everything else is intact** (the region section
below finds the same method affected and the same others clean, but NOT the same sign). At the
matched-permeability standard
it is charged 270 m and 14.9% of homes where 130 m and 9.7% reach the same P* -- **2.1x the road**.
At matched displacement it reads 10.9 permeability points low. The LP's Lens A prefix lengthens by
5.8 m (+1.4%) with no change to either scored value; every other method is bit-identical.

Why the Grid and not the others: it is the one published method that emits a few long lines meant to
cross each other. The LP's 227 short path segments already share `_rnd` endpoints densely, so most
of its junctions are noded by accident; the arterial family's chords run between anchors that are
already network vertices; the trees have nothing to cross.

## At region scale, and against the published tables

`multiblock_density_compactness` (18 blocks, 4,615 parcels), the variant's own tuned method configs,
all six published methods. **The shipped arm reproduces every cell of the committed lens tables in
`examples/multiblock_density_compactness/README.md` exactly** -- 6 methods x 2 lenses x road/homes/
permeability -- so this is the published pipeline, not a re-implementation of it:

| method | Lens A perm | Lens A road m | Lens B homes | Lens B road m |
|---|---|---|---|---|
| **Grid** | **0.6175 -> 0.6301** | **3554 -> 3867** | **0.0974 -> 0.1063** | **3465 -> 3867** |
| Loop Network | 0.7279 -> 0.7282 | 2846 -> 2887 | 0.0589 -> 0.0589 | 1547 -> 1547 |
| Looped Tree | 0.7405 -> 0.7405 | 2284 -> 2284 | 0.0565 -> 0.0565 | 1301 -> 1301 |
| Direct Objective (LP) | 0.8075 -> 0.8075 | 4454 -> 4454 | 0.0485 -> 0.0485 | 2026 -> 2026 |
| Frontage (street-priced) | 0.7470 -> 0.7470 | 5895 -> 5895 | 0.0567 -> 0.0567 | 2651 -> 2651 |
| OSM Footpaths | 0.6304 -> 0.6304 | 2874 -> 2874 | 0.0901 -> 0.0901 | 2506 -> 2506 |

**Which methods move is the durable result here: the Grid materially, the Loop Network marginally,
the other four not at all.**

> **The planar-arm NUMBERS above are superseded.** They were measured with a probe monkeypatch that
> planarized the graph but still carried the OLD street predicate -- before the tolerance defect
> below was found. They are kept because the *shape* of the finding survives (which methods are
> affected, and that the sign differs between lenses); the actual post-fix figures are whatever the
> regenerated artifacts hold, and those went through both fixes.

## The sign is not consistent, and that is the interesting part

At block scale the planarized arm improved the Grid on **both** lenses. At region scale it improves
Lens A (+0.013 permeability) and **worsens** Lens B (+0.0089 homes, +402 m to reach the same P*).

That is not a contradiction, and it is not an argument for leaving the graph wrong. Both arms emit
legitimate, fully street-connected prefixes; what differs is which roads the heuristic ranks first.
Drainage counts **traffic**, and the lens question is **permeability per home displaced** -- so a
better-informed drainage order is not automatically a better lens number. Feeding the heuristic the
correct graph exposes that its objective is not the lens's objective.

This is independent evidence for the backlog's "Lens prefix selection" item, from a direction that
entry did not anticipate: not that a fixed order is a weak heuristic for the optimization, but that
even *correcting its input* can move a reported cost the wrong way.

## The fix exposed a SECOND, older defect: the two street tests disagree

Planarizing changed which road leads the Grid's order, and the regenerated artifact came back with
`access_burden_reduction` of exactly **0** from a 189 m prefix. That is not a rounding artifact
(`%g` prints exact zero as `0`) -- it means the prefix granted no access at all.

**Two predicates, two answers.** `_road_net` decided street-adjacency on `_rnd`-ROUNDED nodes;
`derive.access.street_connectivity` -- what the peel, access depth and burden use -- decides on RAW
segments. `_rnd` snaps to the centimetre. `euclidean_grid` trims to `street_buffer: 0.5`, which is
EXACTLY `STREET_TOL`, so on the pinned block six of nine grid roads sit within 5e-10 m of the
boundary and a centimetre of snapping flips the verdict:

    road   raw seg dist        <= tol   rounded-node dist   <= tol
       0   0.4999999996893671   True    0.49797              True     agree
       1   0.5000000000153105   False   0.49504              True     DISAGREE
       2   0.5000000000522269   False   0.50085              False    agree
       3   0.49999999989363597  True    0.49792              True     agree
       4   0.49999999999151523  True    0.50184              False    DISAGREE
       5   0.5000000004982021   False   0.49792              True     DISAGREE

Three of nine, decided by float noise in the 16th significant digit.

### It is NOT caused by the planarization -- checked, not assumed

Reconstructing the pre-fix builder verbatim and re-running the order
(`scratchpad/old_vs_new_order.py`):

    pre-fix (vertex-keyed)   order [0, 5, 3, 7, 1, 2, 4, 6, 8]
      prefix[:1]   67.1 m   street_connectivity frac 1.000
      prefix[:2]  125.5 m   street_connectivity frac 0.534   <- ALREADY BROKEN
      prefix[:3]  270.0 m   street_connectivity frac 1.000

    post-fix (planar)        order [1, 5, 3, 4, 6, 7, 8, 0, 2]
      prefix[:1]  130.1 m   street_connectivity frac 0.000
      prefix[:2]  188.6 m   street_connectivity frac 0.000

**`street_first_ordered`'s documented guarantee -- "every prefix is connected" -- was already
violated for the Grid before any of this.** Planarizing changed which road leads and made a partial
failure a total one. So the Grid's published lens rows were never sound under either code path, and
`tests/test_every_scored_prefix_reaches_the_street` did not catch it because its fixture tests the
SORT KEY, not the street predicate.

### The fix: decide on raw positions, never snapped ones

`street_nodes` now comes from each node's raw coordinate, with the nearest raw point winning when
several snap to one node -- so the verdict cannot depend on which segment created the node first.
Guarded by `test_every_prefix_is_connected_by_the_same_test_the_peel_uses`, whose fixture straddles
the tolerance deliberately (y = 0.504 snaps to 0.500 and is accepted; raw 0.504 is refused) and
which is asserted non-vacuous before it asserts anything else. On the real grid block every prefix
is now 1.000 street-connected, with burden reduction +0.236 / +0.406 / +0.789.

### What is deliberately NOT fixed

`euclidean_grid`'s `street_buffer: 0.5` still equals `STREET_TOL` exactly, so the Grid's roads still
sit on the knife edge -- both predicates now agree, but they agree about a tie broken by the 16th
digit. Moving the buffer off the boundary changes a published method's emitted GEOMETRY, which is a
different kind of change from fixing an inconsistency; owner's call, taken as consistency-only.
Tracked in the backlog.

## Where else the same mistake lives -- surveyed

Every other place this project turns road geometry into a graph, checked:

| builder | nodes crossings? | blast radius |
|---|---|---|
| `budget._road_net` | **no -> FIXED 2026-09-14** | `road_drainage`, `street_first_ordered`, both lenses, every frontier curve, the GIFs |
| `derive.access.street_connectivity` | **yes** -- whole segments joined by an STRtree `dwithin` query, so two crossing segments are within `tol` and get an edge | clean; access depth, the peel and every `k` metric are unaffected |
| `budget._explode_segments` -> `_build_csr` | **no** -- same `_rnd`-endpoint explosion, same blindness | `efficiency` / `directness` |

**The third row is a real second instance and it is deliberately NOT fixed here.** Two roads that
cross without a turn possible at the crossing overstate door-to-door travel, so `directness` and
`efficiency` read low by the same mechanism. It is left alone because its blast radius is entirely
unpublished: those two quantities are consumed only by `eval/structure.py`, which is an optional
eval (the default is `kcomplexity`), and by the arterial variants configured `objective: directness`
-- none of which appear in any published example lineup (every one runs
`greedy_arterial_access_displacement`, whose objective is `access`, a peel quantity).

Fixing it is also more invasive than it looks: `_edges_in_nx_order`'s ordering is load-bearing for
`_line_entries`'s edge-index tie-break, so changing the segment set changes tie-breaks and therefore
the metric on exact-distance ties, which grid geometry produces. It needs its own red-green cycle
and its own measurement of whether the correction moves anything. Tracked, not done.

## What this does NOT establish

- **Two regions, not a survey.** One 263-parcel block and one 18-block region, both Cape Town. The
  drainage collapse is structural and will reproduce wherever roads cross unnoded; the *size* and
  the *sign* of the lens deltas are two data points, and they already disagree on sign.
- **It says nothing about permeability or displacement themselves.** Neither metric uses
  `_road_net` -- permeability upgrades mesh edges by corridor intersection, displacement buffers
  road geometry. Only the prefix ORDER is affected, so full-build-out numbers (the "one deep block,
  six methods" terminal table) are untouched.
- **It is not the "Lens prefix selection" optimization item.** That entry asks whether a fixed order
  is the right heuristic at all. This is narrower and prior to it: the heuristic is being fed a
  wrong graph. Fixing the graph does not settle the heuristic question.
- **The published numbers below are the PRE-fix ones.** They are what the committed artifacts held
  when this was measured; the fix landed the same day and the examples were regenerated after it.
  The deltas stated here are therefore the record of what moved, not a description of the current
  artifacts.

- **It does not say the Grid is better than reported.** It says the Grid's reported numbers are an
  artifact of a graph missing its junctions. Which way they move depends on the lens and the region.

## Scope

One Cape Town block (263 parcels) and one Cape Town region (18 blocks, 4,615 parcels), six methods
each, the two shipped lens thresholds, one planarization rule (split each segment at its crossings
with segments of other rows; same-row pairs skipped, since consecutive segments of one road already
share `_rnd` endpoints). Probes: `scratchpad/planar_roadnet.py`, `scratchpad/lens_noding_impact.py`,
`scratchpad/grid_order_diag.py`, `scratchpad/noding_check.py`.
