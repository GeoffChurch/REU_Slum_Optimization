# C3: the access curve crosses, and Σ(d−1)² is the statistic that sees it (2026-08-08)

> **SUPERSEDED 2026-08-08 by [C4](2026-08-08-c4-the-bakeoff-was-missing-two-thirds-of-the-field.md).**
> This ran 5 methods. The bakeoff configures 16, and `peel` -- the method built for this very
> objective -- was not registered in `all_methods` at all. With the full 17-method field the
> headline below does not hold. The measurements are correct; the FIELD was wrong.

> Specifically: neither `flow_paths_noreinforce` nor `topology` wins anywhere in the full field --
> `resistance_lp` dominates at 7 m and `greedy_arterial_repulsion` at 2 m. The width-dependence and
> the case for the zero-indexed burden statistic both survive.

C2 asked only what universal access COSTS, and concluded `topology` is the only method that ever
reaches it. That is a question about the end of the curve. The owner's objection was that the
interesting part may be the start — another method could be far more efficient at low displacement
and simply stop short, which a cost-to-k=1 number cannot see.

**Confirmed.** `scratchpad/complexity/c3_depth_curves.py`, 10 density_compactness blocks, full
prefix sweep, three road widths.

## The winner changes along the curve

Lowest median burden at each budget, roads priced at today's 7 m floor:

    2%: flow_paths   5%: flow_paths   10%: flow_paths   15%: flow_paths   25%: topology

So C2's verdict — "only `topology` is viable" — is right about the endpoint and wrong about the
whole regime this project actually compares in. Lens A operates at 10% displacement, where
`flow_paths_noreinforce` is the most access-efficient method and `topology` is not.

## Max depth is too coarse; the zero-indexed burden is not

At 7 m and a 10% budget, four of five methods report **the same** max depth:

    method            k0    burden
    clearance        2.0      0.99
    clearance_grid   2.0      0.95
    euclidean_grid   3.0      1.46
    topology         2.0      1.00
    flow_paths       2.0      0.89

`k0` ties three ways; `burden` separates them cleanly. Max depth throws away the distribution, and
on a corpus whose blocks start at k = 3–5 there are only a few integers to move through. The owner's
zero-indexed `Σ(depth−1)² / n` is the statistic worth reporting — it is 0 exactly when every parcel
fronts a street, so it is a deficit rather than an offset, unlike `budget.access_burden`'s shipped
form which scored a perfect block at n.

## Width changes the ranking, not just the cost

    roads at 7 m    2%: flow_paths   5%: flow_paths   10%: flow_paths   15%: flow_paths   25%: topology
    roads at 2 m    2%: topology     5%: topology     10%: topology     15%: topology     25%: topology

At 2 m `topology` dominates throughout and reaches burden 0.01 at 10% displacement and **k0 = 0
(universal access) at 15%**. At 7 m nothing reaches universal access inside 25%.

This is a real dependence and it has to be stated, having just criticised `sigma` and `D` for
exactly this. The difference is that width is an **observable design decision with a directly
measurable cost**, not a constant fitted to produce method spread. Two agencies building different
things — a pedestrian lane network versus a vehicle street grid — genuinely should choose different
methods, and a metric that said otherwise would be hiding something true. But any published ranking
has to name the width it was computed at.

## `euclidean_grid` is uniformly poor on access

Highest burden at essentially every budget and every width, still at k0 ≥ 2 past 25% displacement.
It builds a grid that never reaches the interior. Under permeability it is mid-pack; under access it
is last by a wide margin. That divergence is the clearest evidence so far that this axis carries
information permeability does not.

## Why this one is worth keeping

It is the first candidate in this line that clears every bar the previous two failed:

* **No free parameter.** `k` and `burden` have no calibration constant. Compare `sigma_road/sigma_walk`
  (no defensible value in its stable band) and `D` (informative only where unstable).
* **No saturation.** Unlike C1's coverage, which was 1.000 for every method at D ≥ 50 m.
* **It discriminates**, and with genuine crossovers rather than a fixed order.
* **It matches the literature's own objective** rather than a quantity invented here.

## Caveats

* **Parcels are Voronoi cells of building points, not a cadastre** (median 56–143 m², ~7.5–12 m
  across). k is about LEGAL access, so this substitution does more work here than it does for
  permeability. The footprint-seeded variant has not been run.
* 10 blocks, Cape Town only.
* The max-depth panels of the plot are jumpy because a median over displacement bins draws on
  different subsets of blocks per bin; the burden panels are the readable ones.
* Prefix order is `street_first_ordered` throughout. The depth-greedy alternative was measured in C2
  and was 0.8% worse, so this is not sensitive to that choice — but every cost here is still an
  upper bound on the true minimum over orderings.
