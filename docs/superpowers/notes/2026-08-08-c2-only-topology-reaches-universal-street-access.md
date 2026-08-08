# C2: only `topology` reaches universal street access, and the 7 m floor is why it costs 45% (2026-08-08)

> **SUPERSEDED 2026-08-08 by [C4](2026-08-08-c4-the-bakeoff-was-missing-two-thirds-of-the-field.md).**
> This ran 5 methods. The bakeoff configures 16, and `peel` -- the method built for this very
> objective -- was not registered in `all_methods` at all. With the full 17-method field the
> headline below does not hold. The measurements are correct; the FIELD was wrong.

> Specifically: 7 of 17 methods reach universal access, not 1. The 7 m width finding survives.

Kill gate on the block-complexity idea (`scratchpad/complexity/c2_gate.py`): measure the COST to
reach k = 1, universal street access, which is the objective the Brelsford/Bettencourt line of work
treats as the definition of reblocking. Chosen after two parameter-driven failures because **it has
no free parameter** -- k = 1 is a standard, not a knob.

**The gate FAILS as a ranking metric, and the way it fails is the finding.**

## Four of five methods can never reach k = 1, at any budget

Minimum k achievable with each method's FULL road set, 10 density_compactness blocks:

    method                    min   median   max
    topology                    1      1.0     1
    clearance                   2      2.0     2
    clearance_grid              2      2.0     2
    euclidean_grid              2      2.0     3
    flow_paths_noreinforce      2      3.0     3

Only 20% of (block, method) pairs reach the standard, all of them `topology`, which reaches it on
10/10. Everything else tops out at k = 2 or 3.

This is why G1 came back undefined on 0/10 blocks: four methods tie at "never", so there is no
ranking for Kendall tau to compare. **The metric does not rank this method suite -- it partitions
it.** By the literature's own definition of the deficit (k >= 2 means parcels without street
access), four of five methods leave every block still deficient no matter how much road you build.

That is not a defect in the metric. It says the shipped methods, except `topology`, are not
optimizing universal access at all -- which is worth knowing, because it is the objective the
research literature considers definitive.

## The 7 m floor makes the objective cost 4.8x what it needs to

Reachability is width-INDEPENDENT -- `parcel_access_layers` seeds from road CENTRELINES within
`STREET_TOL`, and never reads `width_m`. So narrow lanes would not help any of the four methods
reach k = 1. But width dominates the COST, and there the effect is large. `topology`'s k = 1
network, priced at three widths:

    width      displacement to k = 1
    7 m (today's floor)      0.449
    3 m                      0.156
    2 m                      0.094

**Universal street access costs 9.4% displacement with 2 m lanes and 44.9% with the mandated 7 m
street.** Lens A operates at 10%. So the literature's objective is affordable at the project's own
comparison budget, and unaffordable under the project's own width floor.

`min_road_width_m = 7.0` is justified on fire/ambulance/refuse access grounds, which is correct for
a road that carries vehicles. Applying it to every road forbids the pedestrian lane that actually
delivers universal frontage.

## Correction: road types were a better idea than I judged

Asked on 2026-08-07 whether "different types of roads with constant width -- narrow streets for
foot/bicycle, wide streets for EMS" would help, I said it was a choice-set change that does not
change the conductance law, and that the narrow type would look nearly worthless.

**That was right in the flow framing and wrong here.** Under permeability a 3 m paved lane is barely
better than the 3 m dirt gap already there, so it does look worthless. Under access it is the whole
game: it is what buys k = 1 at 9% displacement instead of 45%. The two road types serve two
different objectives and the model needs both -- narrow lanes for universal access, wide streets for
vehicle reach. Judging the idea inside a single objective is what made it look empty.

## The depth-greedy ordering did not pay

Tested as the owner suggested: same connector-chain machinery as `street_first_ordered`, but roads
sorted by the deepest parcel they serve rather than by drainage, on the intuition that one road
driven deep beats many spurs trickling in from the boundary.

    cost to k = 1, median displacement    drainage 0.4451    depth-greedy 0.4488   (+0.8%)
    depth-greedy cheaper on 10% of (block, method)

Slightly WORSE, and it changed no ranking. Not a refutation of the idea -- the heuristic here is
deliberately crude (one no-roads peel, each road scored by the static maximum depth it fronts, no
re-peel after commits). A true greedy that re-peels per commit is O(R^2) peels and was not
attempted. What is established is that the cheap version does not beat drainage ordering, so the
ordering is not where the cost is hiding.

## What this leaves

C2 fails its own gate and should not ship as a ranking metric on this method suite. But it produced
the first result in this line that is neither parameter-dependent nor saturated: a clean, discrete
separation between a method that achieves universal access and four that cannot.

The obvious next move is a second road type below the width floor -- an access lane -- and a re-run
asking what each method costs to reach k = 1 when it is allowed to build one. That is a model
change, not a measurement, so it is the owner's call.

Open caveat, unchanged from before: parcels here are Voronoi cells of building points (median
56-143 m², ~7.5-12 m across), not a cadastre. k over a modelled tessellation is a model of tenure,
not a measurement of it, and the whole quantity is about LEGAL access. Testing k over
footprint-seeded cells is the other half of this experiment and has not been run.
