# Vehicle access: line-sampled gates are a knob, the widest path fixes it, and the objective says build nothing

2026-09-21. Prompted by asking why `clearance_looped` (Looped Tree) looks like the best plan and
scores worst, while `resistance_lp` scores best and is a mesh of 6.2 m stubs. FIVE things were
measured. Results 1-4 are the first pass. **Result 5 rebuilds the measure properly, and it
OVERTURNS Result 2 on the only block that can actually test it** -- read it before quoting anything
above it. Results 2 and 3 were measured with a clearance computation Result 5 shows over-reads.

## The question

Permeability is a smooth flow over a mesh whose footpath conductance is non-zero everywhere, so it
never saturates: more road always buys more. Displacement charges homes, not metres, so a method
that threads gaps gets more road for the same price. Hence the hypothesis: make the benefit
**binary and saturating** — a building is served once a corridor of vehicle width reaches it, and
further road buys nothing — and restraint should start to pay.

## Result 1 (POSITIVE): the raster direct-access measure works

Free space rasterized at 0.5 m, Euclidean distance transform, a cell admits a `W`-wide corridor iff
its clearance is `>= W/2`, components touching the block edge are grounded (a kblock face is
street-bounded, so its boundary IS the street). Same primitive `gap_widths.py` used.

Non-degenerate and width-sensitive. On `ZAF.9.3.1_1_40972` (263 buildings) with no intervention:

    W = 3.0 m   0.806 served
    W = 3.5 m   0.741
    W = 4.0 m   0.513

It discriminates hardest right at the ambulance width, which is the property `eps` lacked in the
continuum spike: `W` is a policy input you report a sweep over, not a constant you tune.

## Result 2 (NEGATIVE): saturation does not flip the ranking

Scored at Lens A (matched displacement) on the same block, `W = 3.5`:

    method                road_m   served
    clearance_looped         110    0.814
    osm_footpaths            215    0.829
    euclidean_grid           218    0.875
    cycle_native             243    0.848
    greedy_arterial          339    0.913
    (no roads)                 0    0.741

Vehicle access tracks road length near-monotonically and **Looped Tree stays last**. Saturation is
never reached: the baseline is 0.741 and the ceiling 0.913, so more road keeps buying access the
whole way.

On reflection that is correct rather than a defect of the measure. At matched displacement
`greedy_arterial` gets 339 m for the same 10% that buys Looped Tree 110 m, and if displacement is
the whole cost then three times the road for the same price genuinely is better. **Saturation
cannot make restraint pay under a cost model that does not charge for what restraint saves.** The
lever is a cost on road length, which is where this started.

## Result 3 (NEGATIVE): line-sampled gate widths are a knob, not a detail

"Gate depth" is a clean generalisation of the existing BFS peel: between adjacent parcels the gate
is the channel width, an edge costs 0 if the gate admits `W` and 1 otherwise, and depth is a 0-1
BFS from street-fronting parcels. `W -> 0` gives depth 0 everywhere; `W -> inf` recovers the
current ring count exactly. `sum_i depth_i` is `access_burden`.

The idea is sound. **Sampling the gate along the centre-to-centre segment is not**, and the choice
of reduction reorders everything:

    variant                median gate   do-nothing served   verdict
    min along segment          1.00 m          0.190         Looped Tree wins, +61.0
    max (ridge) along segment  6.32 m          1.000         do nothing wins, all methods negative

Calibration settles which is wrong, and the answer is BOTH. `gap_widths.py` measured a 7.65 m
median channel for this block's density band and 2.5-3.0 m at 8k+ bldg/km². `min` is pinned at
`2 x RES`, the resolution floor. `max` reads **8.00 m on the spine block** — 3x too wide on the
dense fabric where the question actually lives, because the segment between two adjacent parcels'
building centres can cross open ground and `max` finds that instead of the passage.

This is the continuum spike's failure mode again (`eps` reordering methods, 119 rank flips): a
sampling choice that looks like an implementation detail turning out to be the thing that decides
the answer. The pairwise `dist_ij - r_i - r_j` fails the same way for the same reason — it is the
clearance between two discs, not the width of the passage, and a third disc pinching the channel
never appears in it.

**What it needs is a genuine bottleneck (widest-path) computation on the distance transform**: the
gate between two parcels is the maximin clearance over paths between them, not a statistic of a
line. That is real work, not a parameter.

## Result 4 (NEGATIVE, and a retraction): demolition/displacement divergence does not generalise

At equal 10% displacement on the 263-parcel block, `euclidean_grid` demolishes 17.5% of buildings
against `clearance_looped`'s 10.6% -- 65% more homes destroyed for the same score, because
fractional disc overlap charges grazing many buildings the same as flattening fewer. That looked
like a strong critique of the displacement metric.

**It does not hold on the spine block.** There every method lands at ~10% demolished, and
`greedy_arterial` demolishes the FEWEST (6.7%) while laying the MOST road (9,713 m). Claimed from
one block and withdrawn on the second.

## Result 5: the widest path is real; it needed FOUR bugs fixed, and it overturns Result 2

Result 3 asked for "a genuine bottleneck (widest-path) computation ... real work, not a parameter".
Built (`widest.py`): sweep channel widths DOWNWARD over the clearance grid; at each `w` the passable
set is `clearance >= w/2`, label it, ground the components touching the block edge, and building `i`
takes the highest `w` at which a grounded passable cell comes within `r_i + w/2 + RES` of its centre.
One pass yields every `w_i` AND the whole `W`-sweep. True maximin, no reduction to pick: the knob is
gone.

**Four bugs, and every one produced a plausible-looking table rather than an error.** This is the
reason to read this section:

* Grounding on the GRID edge rather than the BLOCK edge.
* Testing clearance at the building's nearest free cell -- which sits against its own wall, where
  clearance is `~RES` by construction. Every building read exactly 1.00 m.
* `distance_transform_edt(inside & ~obstacle)` -- **treating the block boundary as a wall.** The
  boundary IS the street, the widest open space present. Including it capped every edge cell at
  `~RES`, so nothing could ground above `W = 1.0 m` and again every building read exactly 1.00 m.
* **Raster EDT of a RASTERIZED disc mask instead of analytic clearance.** The one that mattered,
  because it produced numbers that looked entirely reasonable. Rasterizing a disc by
  centre-inclusion drops every cell whose centre lies outside `r` but which the disc still
  overlaps, so the obstacle is systematically SMALLER than the disc and clearance reads
  systematically WIDER. Worse, `building_radii` is NN/2, so a building with a close neighbour has a
  radius below the cell size and VANISHES, opening corridors that do not exist. Use
  `vehicle_access.py`'s analytic `min_i(|p - c_i| - r_i)`. Cost of the bug: no-roads median
  5.00 -> 4.25 m and served at `W = 4.0` **0.760 -> 0.513**.

**Calibration is an EQUALITY, not an ordering.** Grounding and endpoint are `vehicle_access.py`'s
verbatim and 3.0/3.5/4.0 lie on the sweep grid with nested passable sets, so the no-roads served
fractions must reproduce Result 1 exactly: 0.810/0.741/0.513 against 0.806/0.741/0.513. Two exact,
`W = 3.0` off by one building in 263 -- the endpoint test is not strictly monotone in `w` (shrinking
`w` shrinks the RHS while growing `reach` shrinks the LHS), so `max{w : hit(w)} >= 3.0` can differ
from `hit(3.0)` for a building on the boundary. **Two earlier "validations" were withdrawn to get
here**: `gap_widths.py` is the WRONG BAND (7.65 m is 1-2k bldg/km2; both blocks are 8k+, median
2.50 m) and in any case compares a RIDGE-SAMPLE distribution to a per-building BEST-ROUTE one, which
forces no inequality; and "Result 1 dominates in the admissible direction" was wrong because Result
1 is also a connectivity measure, so it should be EQUAL, and the gap was the bug above.

### Corrected results, both blocks

    ZAF.9.3.1_1_5810 (6619)        D      p50   >=3.0   >=3.5   >=4.0      OBJ   per-unit
    (no roads)                 0.000     4.00   0.770   0.638   0.533   6022.4      0.910
    clearance_looped (5377m)   0.100     5.75   0.912   0.848   0.787   5706.8      0.958
    euclidean_grid   (5544m)   0.103     6.00   0.882   0.823   0.765   5630.0      0.948
    cycle_native     (7279m)   0.101     6.00   0.889   0.830   0.773   5644.2      0.949
    greedy_arterial  (9713m)   0.070     5.50   0.881   0.819   0.739   5822.0      0.946
    osm_footpaths    (3094m)   0.025     4.25   0.787   0.661   0.559   5905.6      0.915

`OBJ = sum_i (1 - c_i) * min(w_i / 3.5, 1)`; per-unit is `OBJ / sum_i (1 - c_i)`, i.e. access
quality among surviving mass.

**Vehicle access does NOT saturate.** The earlier "it saturates on both blocks" was the raster
over-read: with no roads only 63.8% are served at 3.5 m and 53.3% at 4.0 m. Retract "vehicle width
is not the binding constraint" -- there is real headroom, and the measure discriminates hardest at
4.0 m (0.533 -> 0.787).

**Result 2 is overturned, on the block that can actually test it.** Lens A only MATCHES for three
methods here: `greedy_arterial` cannot spend the 10% budget even with 9,713 m (it threads gaps,
reaching D = 0.070) and `osm_footpaths` reaches 0.025. Among the three comparable ones,
`clearance_looped` is FIRST at 0.958 and does it with the SHORTEST road (5,377 m) -- where Result 2,
on the small block with the old measure, had Looped Tree last. The ranking is block-dependent; the
small block is not a matched comparison at all (achieved D ranges 0.101-0.171 there).

### The objective prefers do-nothing -- and the D-sweep turns that into an exact condition

`dsweep.py`, `clearance_looped` on the spine block, sweeping the Lens A target:

    target D   got D   road_m    p50   >=3.5m       OBJ  per-unit    vs D=0
       0.000   0.000        0   4.00    0.638    6022.4     0.910      +0.0
       0.005   0.008      587   4.25    0.685    6001.5     0.914     -20.8
       0.010   0.010      683   4.25    0.688    5992.1     0.915     -30.2
       0.030   0.031     1655   4.50    0.712    5916.9     0.922    -105.4
       0.050   0.051     2931   5.00    0.747    5864.7     0.933    -157.6
       0.100   0.100     5377   5.75    0.848    5706.8     0.958    -315.6
       0.150   0.150     7928   6.25    0.896    5444.2     0.968    -578.2

Monotone decline from the first 587 m -- **for this method**. Sweeping all five (`msweep.py`,
`verify.py`) shows that conclusion does NOT generalise: `cycle_native` HAS an interior optimum.

    method                     D  road_m  gone  near       OBJ   vs D=0  u_prime
    cycle_native          0.0037     177    25     1    6031.6     +9.2    1.291
    cycle_native          0.0066     318    46     5    6029.9     +7.6    1.090
    cycle_native          0.0107     623    73    11    6032.9    +10.6    1.071
    cycle_native          0.0118     809    77    13    6025.9     +3.5    0.966
    cycle_native          0.0145    1230    96    14    6010.4    -12.0    0.797
    cycle_native          0.0207    1802   137    17    5971.3    -51.1    0.549
    clearance_looped      0.0035     214    23     3    6002.4    -19.9    0.046
    clearance_looped      0.0216    1189   146    22    5949.6    -72.8    0.410
    euclidean_grid        0.0127     748     -     -    5960.8    -61.6    0.182
    greedy_arterial       0.0063     993     -     -    5994.6    -27.8    0.242
    osm_footpaths         0.0058    1254     -     -    5984.7    -37.6   -0.065

**So pricing displacement DOES tune the road budget** -- to 177-623 m on a 57.7 ha block, not the
5,377 m Lens A buys. The most efficient point is the smallest (177 m, `u' = 1.291`).

Checked for the obvious artefact and it is not one. `widths()` deletes a building from the OBSTACLE
field on a hard `c > 0.5` while OBJ weights by the SOFT `c`, so a method that concentrates
displacement just over the threshold could buy corridor cheaply. `near` (buildings with
`c` in (0.4, 0.6)) is **1** at the +9.2 point -- a single boundary building cannot make that swing --
and `clearance_looped` carries equal or larger `near` counts (3, 12, 13, 16, 22) while staying
negative throughout. The win does not track the threshold population.

**The criterion predicts the sign in 14 of 14 rows.** Exactly, `OBJ - OBJ_0 = n D (u' - u(D))`, so
the sign is `u'` against `u(D)`: every `cycle_native` row above ~0.91 is positive INCLUDING the
marginal 0.966 (+3.5), every row below is negative, and all seven `clearance_looped` rows sit at
`u' <= 0.53` and are negative. The analysis is predictive, not a post-hoc description.

The sweep gives more still. Write `OBJ(D) = n(1-D) u(D)`
with `u` the per-unit access (`OBJ / sum_i (1 - c_i)`):

* **A hard bound, independent of method.** `f = min(w/3.5, 1) <= 1` gives `OBJ(D) <= n(1-D)`, so
  beating do-nothing REQUIRES `D < 1 - u(0)`. That is **9.0% on the spine block** and **7.2% on the
  small block** (`u(0)` = 0.910 and 0.928). No plan by any method can justify more displacement than
  the do-nothing access DEFICIT. **Lens A's pinned 0.100 is already above the bound on both
  blocks**, so at Lens A the comparison against do-nothing is settled before a method is chosen.
  This is a property of a SATURATING benefit (`f <= 1`); an unbounded benefit has no such bound.
* **A marginal condition, which is a target.** `dOBJ/dD` at 0 is `n[u'(0) - u(0)]`, positive iff
  `u'(0) > u(0)`. Measured `u'(0) ~ (0.914 - 0.910)/0.008 = 0.50` against a required 0.91 -- short
  by a factor of ~1.8, not by orders of magnitude. **A method that roughly DOUBLES the access opened
  per home destroyed would show a genuine interior optimum.**

This is the answer to "displaced with probability p means zero reachability": no singularity, it
simply prices the target and finds 10% unjustified by vehicle access alone -- and it says by how
much, and what would have to change.

Scope it carefully. The sweep walks a drainage-ordered PREFIX of a plan designed for D = 0.10, so a
small-D point is the first few segments of that plan, not the best plan at small D. **An interior
optimum found this way would be real; a declining curve is weaker evidence**, because none of these
five methods optimizes the objective it is being scored on. The open test is a method that maximises
`sum_i (1 - c_i) f(w_i)` directly, and it now has a number to beat: `u'(0) > u(0) ~ 0.91`.

### Not the disc model -- hypothesis formed and killed

`building_radii` is HALF THE NEAREST-NEIGHBOUR DISTANCE, so a "building" is as big as its spacing.
That predicts the disc model invents the corridors. **Measured, and it is wrong.** Open Buildings
ships `area_in_meters`, which `KBlockSource.building_points` DROPS at the reader
(`columns=["geometry"]`):

    block                   disc cov (r=NN/2)   TRUE footprint cov   disc r    true r
    ZAF.9.3.1_1_40972             0.325               0.374          2.19 m    2.32 m
    ZAF.9.3.1_1_5810              0.291               0.253          2.61 m    2.30 m

On the dense block true coverage is LOWER than the disc model's, so real footprints would open the
fabric FURTHER. Median true radius is 2.30 m on BOTH blocks -- actual shacks are the same ~17 m^2
everywhere and only spacing varies -- while the disc radius moves 2.19 -> 2.61, so the disc model
conflates size with spacing and OVER-states building size by 13% on the dense block.

### Lens A does not always match, and nothing says so

Achieved D against a 0.100 target: 0.101/0.171/0.113/0.144/0.107 on the small block, and
0.100/0.103/0.101/0.070/0.025 on the spine block. Both are `prefix_to_displacement` behaving as
documented -- MINIMAL prefix at or ABOVE the target, all roads when the target is unreachable -- so
this is not a bug. But the harness does not REPORT achieved D, so an unmatched comparison looks
matched, and a ranking read off raw OBJ assuming equal mass is wrong. **Report achieved D beside any
Lens A comparison**, and treat a method that cannot reach the target as absent from it.

One more trap in the same family: `served = mean((w >= t) | gone)` counts a DEMOLISHED building as
served, which makes demolition look free. `vehicle_access.py` returns `np.mean(served | gone)`, so
**Result 2's table above does share it.** Checked, and Result 2's conclusion survives on its own
block -- dividing out the demolished gives 0.792/0.809/0.848/0.829/0.899 against 0.741 for no roads,
still near-monotone in road length. `OBJ` weights by `(1 - c_i)` and avoids the trap; prefer it.

## Result 6: "reblocking does not pay" was a statement about W = 3.5, and the ranking ROTATES with W

`w_i` does not depend on the standard, so one widest-path pass per method scores every `W` for free
(`wscore.py`). Lens A D = 0.100, spine block, OBJ relative to do-nothing:

    OBJ vs do-nothing            D  road_m    W=3.0   W=3.5   W=4.0   W=4.5   W=5.0   W=6.0   W=8.0
    (no roads) ABSOLUTE      0.000       0   6277.4  6022.4  5729.6  5425.9  5129.6  4599.1  3745.3
    clearance_looped         0.100    5377   -463.1  -315.6  -155.0    -4.0  +132.1  +325.2  +408.2
    euclidean_grid           0.103    5544   -521.3  -392.3  -241.8   -89.8   +51.1  +272.6  +442.6
    cycle_native             0.101    7279   -511.3  -378.2  -221.8   -61.8   +91.4  +325.7  +427.6
    greedy_arterial          0.070    9713   -327.0  -200.4   -64.9   +60.4  +168.6  +303.2  +290.8
    osm_footpaths            0.025    3094   -132.8  -116.7   -96.6   -83.1   -72.1   -54.8   -25.4

**The sign flips at W ~ 4.5 m.** Below it do-nothing wins; above it reblocking pays, and by 6 m the
full plans are +300 or better. Result 5's "the objective prefers do-nothing" is therefore NOT a fact
about reblocking -- it is a fact about pinning the standard at 3.5 m, where this fabric is already
64% compliant. The crossover is the reportable quantity: **4.5 m is the access standard at which
reblocking begins to pay on this block.** It follows directly from the bound `D < 1 - u(0)`, which
widens from 5.2% at W = 3.0 to 9.0% at 3.5, 13.4% at 4.0, 22.5% at 5.0 and 30.5% at 6.0 -- the
do-nothing access DEFICIT is what there is to buy, and at 3.5 m there is almost nothing to buy.

**The ranking ROTATES with W, so nothing here is dominated:**

* `greedy_arterial` first at W = 4.5-5.0 (+60.4, +168.6), helped by spending only D = 0.070.
* `cycle_native` (+325.7) and `clearance_looped` (+325.2) tied at W = 6.0 -- on 7,279 m against
  5,377 m, so Looped Tree reaches the same place on 26% less road.
* `euclidean_grid` first at W = 8.0 (+442.6); the grid's regularity pays once the standard is wide.
* `osm_footpaths` NEGATIVE at every W. Existing footpaths never repay their displacement under a
  vehicle objective, which is what paths not built for vehicles should do.

This is the frontier case from the working rules, not a dominance case: each of four methods is best
at some standard a user could reasonably pick, so `W` is an operating point to ship presets for, and
none of them should be deleted. It also vindicates Result 1 -- `W` is a policy input you sweep, not
a constant you tune.

## Result 7: a method that OPTIMIZES the objective, and the min-cut dual that aims it

Every method in Results 1-6 optimizes something else -- depth, cycles, resistance, directness --
and is then scored on `OBJ`. `direct.py` optimizes it: greedy over least-cost paths plus
street-to-street through-roads, each candidate scored by a FULL Delta-OBJ recompute, stopping when
nothing gains. The road budget is an OUTPUT -- no `depth_target`, no `max_roads`, no pinned Lens A.

### The min-cut dual (`dual.py`): use it to AIM, not to measure

    w_i = max over paths of (min clearance)  =  min over encircling chains of (max gap)

the max-min/min-max form of max-flow/min-cut. Buildings are sources, the block boundary is the
sink, and the cut is a CHAIN OF GAPS. Delaunay on the surviving centres, each edge a gate of width
`dist - r_a - r_b`; the dual has triangles as nodes joined by their shared gate; every triangle the
boundary passes through -- or outside the hull, where by definition there are no buildings -- is
grounded. Sweeping w downward is one union-find pass in DESCENDING gap order, so every `w_i` falls
out of a single O(E log E) sweep.

**As the METRIC it is NOT adopted.** It over-reads against the validated raster: p50 4.62 vs 4.25,
`r = 0.917`, and 0.730 vs 0.513 served at W = 4.0. Cause: Delaunay-on-centres is exact only for
EQUAL radii, and `building_radii` is NN/2 and varies widely, so the true blocking pair is often not
a Delaunay edge of the centres and the dual misses blocking gates. The fix is an additively-weighted
(Apollonius) diagram -- NOT a power-diagram lifting, which takes subtractive `r^2` -- and that is
real work. `clearance.py` already flags the same approximation in `_node_clearance`.

**As the CANDIDATE GENERATOR it is a clear win**, and approximation error is FREE there: the dual
only ranks, and every candidate is still scored by the exact raster. The min-cut NAMES the cut --
123 gates cap 225 of 263 buildings, so the binding constraints are a small enumerable set.

### Measured

    small block, W = 6.0       roads  road_m       D  OBJ gain   1st-road u'
    gate-targeted (min-cut)        4     176   0.076     +11.2         2.035
    building-targeted              4     212   0.115      +9.6         1.138
    osm_footpaths (best incumbent) -     215   0.107      +4.3             -

    spine block, W = 6.0       roads  road_m       D  OBJ gain
    gate-targeted (min-cut)       16   4,949   0.051    +383.7
    cycle_native                   -   7,279   0.101    +325.7
    clearance_looped               -   5,377   0.100    +325.2
    building-targeted              3   1,631   0.018    +165.2

**A Pareto win on all three axes: +18% access, 32% less road, HALF the displacement.** And the
mechanism is legible: gate-targeting's first road is WORSE (+129.5 vs +160.3) yet the trajectory
climbs monotonically to +383.7 instead of stalling at +165. The greedier opening move is what
trapped the building-targeted run. **The candidate family was never the ceiling -- the aiming was.**

### Three things this settles

* **Non-submodularity is real.** Building-targeted at W = 6: road 2 LOSES 0.3 and road 3 then gains
  5.1. A strict `Delta-OBJ > 0` stop leaves value on the table; gate-targeting mostly sidesteps this
  by not walking into the trap in the first place.
* **The stub pathology is a LOW-W symptom, not the objective's.** At W = 3.5 the optimizer emits
  5-25 m stubs; at W = 6 road 1 is 1,294-1,527 m. Stubs appear only once the worthwhile roads are
  taken. A road-length cost is still the direct fix, but the objective is not inherently stub-loving.
* **At W = 3.5 on the spine block it beats the best incumbent 2x** -- 273 m / D = 0.0041 / +21.7
  against `cycle_native`'s 623 m / D = 0.0107 / +10.6 -- on 44% of the road and 38% of the
  displacement.

**CAVEAT, and it is not small: this is an ORACLE comparison.** The optimizer maximises exactly what
it is then scored on, so these numbers BOUND what is achievable on this objective and measure how
far the incumbents sit from that bound. They are NOT evidence it is a better reblocker. It targets
16 roads chosen purely for vehicle width and ignores network structure entirely, so expect it to
score badly on permeability. Score it there before calling it a method. **Scored in Result 10:
dominated by `cycle_native` on both blocks, and not built.**

## Result 8: we never had footprints, and the disc model OVER-READS access by 34% at W = 3.5

`scripts/fetch_kblock_fixtures.py` resolves the Open Buildings tile manifest -- whose `tile_url` is
ALREADY the polygon URL -- and then rewrites it to points:

    return str(feat["properties"]["tile_url"]).replace(OB_POLYGON_PREFIX, OB_POINT_PREFIX)

So real footprints were one `.replace` away the whole time. Fetched (320 MB against the point
tile's 83 MB) and measured on `ZAF.9.3.1_1_40972`. **263/263 footprints match the pipeline's
building points within 3 m, median 0.00 m** -- the "building points" ARE the footprint centroids, so
the tiers line up one-to-one and the comparison is exact. Clearance is exact in both arms
(`STRtree` distance to the true polygon vs `min_i(|p - c_i| - r_i)`), so the ONLY difference is the
building model.

                    p10     p25     p50     p75
    FOOTPRINT      1.55    2.75    3.75    6.25
    disc           2.50    3.25    4.25    8.00

        W   footprint     disc   disc over-read
      3.0       0.631    0.802           +0.171
      3.5       0.548    0.734           +0.186
      4.0       0.460    0.517           +0.057
      5.0       0.395    0.437           +0.042
      6.0       0.300    0.346           +0.046

**Served at 3.5 m is 0.734 on discs against 0.548 on footprints -- a 34% relative over-read.** The
error is strongly WIDTH-DEPENDENT, large at 3.0-3.5 m and small at 5-6 m, which pins the mechanism:
a narrow channel squeezes through the corner gap between two circles inscribed in abutting
rectangles, and that gap does not physically exist. At 5-6 m the channel is too wide for corner
gaps to matter either way. Note it is NOT an area effect -- true median footprint area (16.9 m^2)
slightly EXCEEDS the disc area (15.0 m^2). It is shape.

**What this invalidates.** Everything at W = 3.5 is materially affected: `u(0)` falls, so there is
MORE headroom, the bound `D < 1 - u(0)` widens, and **the W ~ 4.5 crossover moves DOWN** -- reblocking
pays at a lower standard than Result 6 concluded. Same direction as the raster fix in Result 5:
every correction so far has found the fabric LESS passable than we thought. The W = 6 results
(Result 7's +383.7 vs +325.7) survive better at +0.046 but still need recomputing.

**A fifth plausible-looking degenerate table**, same family as Result 5's four. `STRtree.query`
returns `(INPUT indices, TREE indices)`; naming them the other way round wrote cell ids into
polygon-id slots, every building fell through to a centroid fallback, and since a centroid sits
INSIDE its own building that test is unsatisfiable -- so every `w_i` read exactly 0.00. The tell was
the same as before: a column of identical values is a bug, not a measurement.

### The tier design

Three tiers now exist -- points, points+area, polygons -- and the machinery should span all three
with unsupported combinations rejected by the CHECKER, not at runtime. A subtyping ladder of
capability Protocols does that:

    Positions  (xy)  <-  Extents  (radii, clearance)  <-  Shapes  (polygons)

with three concrete `Extents`: `SpacingDiscs` (today's NN/2), `AreaDiscs` (`r = sqrt(area/pi)`,
free -- `area_in_meters` is already in the parquet and dropped at the reader) and `Footprints`.
A function needing extent takes `Extents` and CANNOT be passed points; one needing true outlines
takes `Shapes` and rejects discs -- a type error at every site rather than a crash. The tier is
resolved ONCE where config and data are read, injected downstream, and validated at LOAD: a
configured tier the data cannot supply raises there, loudly, never degrading silently to discs.
Side benefit: it makes the NN/2 wart VISIBLE as a named strategy someone opts into, rather than the
invisible default that cannot tell dense-small from sparse-large.

## Result 9: cycle_native's "double roads" are waste on the block where they matter

Rendered, `cycle_native` shows many near-parallel road pairs a few metres apart -- two roads where
one wider road would seem to do. Displacement charges per BUILDING, not per land taken, so the
sliver between the two is free and nothing pushes the method to merge them.

`double_roads.py` finds near-parallel pairs (within 14 m, within 20 degrees), merges each into ONE
road carrying the pair's combined width, then re-tunes that extra width by bisection until
displacement MATCHES the original exactly, with every road held at the 7 m two-way floor that
`buildable_widths` enforces. Two earlier attempts were confounded: conserving WIDTH inflated
land take (corridor 1607 -> 2174 m^2), and conserving analytic `length x width` did not conserve
the buffered polygon either, because end caps grow with width and the pair's buffers overlapped.

    block                  pairs  share of road   D (both)   perm orig -> merged   road
    ZAF.9.3.1_1_40972          2           69%     0.1126     0.7697 -> 0.7580    -31%
    ZAF.9.3.1_1_5810          18           61%     0.1008     0.8933 -> 0.9124    -30%

**On the spine block merging is a Pareto improvement on every axis** -- +2.1% permeability, 30% less
road, identical displacement, land take within 1.2%. On the small block it costs 1.5%. Weight the
spine: 9x the pairs, the realistic scale, and there the doubling is 61% of the whole network. The
sign is NOT universal, so do not claim it is.

**The non-planarization artifact is not what sustains it.** Hypothesis was that `_road_net` never
noding road-road crossings (`2026-09-14-road-net-is-not-planarized.md`) lets the solver credit two
parallel roads as independent routes. If so, merging would LOSE permeability. On the spine it
GAINS. Only 9 of 18 pairs have both members touching the street, so half are not even real cycles.

**The cause is the cost model**: nothing prices road length or land take. The cheap fix is a
post-pass merging near-parallel pairs on the `Method.prior` refiner seam -- free on the spine block,
but one block is thin evidence for shipping a method change.

## Result 10 (NEGATIVE): on the reported axes the direct optimizer is dominated by `cycle_native`

2026-09-23. Result 7's caveat, closed. The saved W = 6 road sets (spine: 16 roads, 4,949 m; small:
4 roads, 176 m) scored on the CURRENT metric -- footprint-overlap displacement at the explore
variants' footprint tier, 7 m roads, permeability -- against each incumbent's own proposal
truncated to the direct's displacement (`prefix_to_displacement`, the minimal prefix at or above
it). Lens A's 10% cannot be applied: the direct stops by itself at 2.9% and 4.3%.

    spine ZAF.9.3.1_1_5810        road_m        D       P    Lens B: D to reach P* = 0.60
    direct (W = 6)                 4,949   0.0290   0.774    0.0187
    cycle_native                   3,457   0.0297   0.806    0.0168
    greedy_arterial (access_disp)  8,554   0.0291   0.802    0.0082
    euclidean_grid                 2,113   0.0296   0.656    0.0296
    clearance_looped               2,485   0.0292   0.624    0.0280
    clearance                      3,020   0.0298   0.516    0.0472

    small ZAF.9.3.1_1_40972       road_m        D       P    Lens B
    direct (W = 6)                   176   0.0426   0.561    never reaches 0.60
    greedy_arterial (access_disp)    270   0.0472   0.777    0.0182
    euclidean_grid                   130   0.0730   0.716    0.0730
    cycle_native                     119   0.0465   0.714    0.0465
    clearance                         89   0.0491   0.625    0.0491
    clearance_looped                  83   0.0446   0.619    0.0446

* **`cycle_native` dominates it on both blocks**: more permeability at matched displacement, less
  displacement to reach P*, and on the spine block 30% less road. The incumbents' prefixes sit
  up to 0.0008 above the direct's D; at `cycle_native`'s measured slope (0.206 permeability per
  0.0129 D) that is worth at most 0.011, a third of its 0.032 lead.
* It is NOT the worst method on the spine block -- it beats `clearance`, `clearance_looped` and
  `euclidean_grid` there -- and it IS the worst on the small one. Aiming at vehicle-width gates
  buys some permeability as a side effect; not as much as methods that aim at network structure.
* Caveats: two blocks, one run each; the roads were optimized under the old disc displacement,
  before footprints; and the objective it wins on, vehicle access at W = 6, is not a reported
  axis and was measured on discs, which Result 8 shows over-read access.

**Decision: not built as a method** -- it would ship dominated on every reported axis. What would
put it back on the list: vehicle access becoming a REPORTED axis, where it bounds the incumbents
(and would first need footprint-tier clearance, per Result 8). *Untested:* aiming with the min-cut
dual while scoring candidates on permeability instead of vehicle access.

## What to take from this

* **Compute clearance ANALYTICALLY** (`min_i(|p - c_i| - r_i)`, `vehicle_access.py`), never as a
  raster EDT of a rasterized disc mask. The mask is systematically smaller than the discs, so
  clearance over-reads, and sub-cell radii vanish entirely. It moved served at `W = 4.0` from 0.760
  to 0.513 while looking perfectly reasonable throughout.
* **Do not quote Result 2's ranking as settled.** At matched displacement on the spine block with
  the corrected measure, `clearance_looped` is FIRST and on the shortest road. Result 2 ran on the
  small block, where Lens A does not match at all (achieved D 0.101-0.171).
* **Report achieved D beside every Lens A comparison**, and treat a method that cannot reach the
  target as absent from that comparison rather than as a competitor. `greedy_arterial` cannot spend
  10% even with 9,713 m; `osm_footpaths` reaches 0.025.
* **A saturating benefit bounds the justifiable displacement at `1 - u(0)`** -- 9.0% on the spine
  block, 7.2% on the small one -- for ANY method. Lens A's pinned 0.100 exceeds both, so that
  comparison is decided before a method is chosen. Pick the displacement target against this bound,
  not by convention.
* **The margin criterion is predictive.** `OBJ - OBJ_0 = n D (u' - u(D))` called the sign in 14/14
  measured rows. Use it to screen a method cheaply before scoring it: `u' > u(0)` or it loses.
* **`cycle_native` HAS an interior optimum** at D = 0.4-1.1% (177-623 m on 57.7 ha, +9 to +11),
  verified not to be the hard/soft displacement-threshold artefact. So pricing displacement does
  tune the road budget -- to a tiny intervention at W = 3.5, and to full plans at W >= 5 (Result 6).
  Do not repeat "the objective says build nothing"; it was measured on `clearance_looped` alone.
* Do not sample a channel width along a line (Result 3). Both reductions are wrong and they
  disagree. The widest-path sweep is the replacement and it is cheap: one pass gives every `w_i`
  and the whole `W`-sweep.
* Saturation alone does not reward restraint (Result 2's reason still stands): access genuinely
  increases with road, so a cost on road LENGTH remains the untested lever.
* `W` is the parameter this family gets right -- physical, sweepable, reportable -- where `g_walk`
  is not. See `2026-08-08-the-road-walk-ratio-has-no-physical-support.md`, whose gap-width table is
  banded by density: check the band before using it as a comparator, and remember it samples RIDGES
  while this samples BUILDINGS.
* `KBlockSource.building_points` drops `area_in_meters` at the reader. Real footprints are on disk
  and one `columns=` change away, if a measure ever needs true building size rather than NN/2.
