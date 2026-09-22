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

### The objective says BUILD NOTHING, and that is the answer to probabilistic displacement

On raw OBJ, which prices displacement: do-nothing 6022.4 > osm 5905.6 > arterial 5822.0 >
clearance 5706.8 > cycle 5644.2 > euclidean 5630.0. Monotone decreasing in achieved D, on BOTH
blocks. Laying 5,377 m moves mass down 10% and access per unit up 5.3%, so it loses.

This IS the "displaced with probability p means zero reachability" idea, and the answer is: it does
not blow up via singularity, it simply prices the 10% target and finds it **unjustified by vehicle
access alone**. Under a utilitarian sum where a demolished home contributes 0, reblocking pays only
where access gained across everyone else exceeds the value destroyed.

Scope the claim carefully: **none of these five methods beats do-nothing, at any displacement any of
them achieves.** That is not a proof that no method could -- none of them OPTIMIZES this objective.
The test is a method that maximises `sum_i (1 - c_i) f(w_i)` directly, and a sweep over D rather
than a pinned 0.100, since the optimum over the tested points sits at D = 0.

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
* **The objective prefers do-nothing on both blocks**, monotonically in D. That is the answer to
  "displaced with probability p means zero reachability": no singularity, it just prices the 10%
  target and finds it unjustified by vehicle access alone. The open test is a method that OPTIMIZES
  `sum_i (1 - c_i) f(w_i)` directly, plus a sweep over D instead of a pinned 0.100 -- none of the
  five methods here optimizes it, so "none of these beats do-nothing" is not "nothing could".
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
