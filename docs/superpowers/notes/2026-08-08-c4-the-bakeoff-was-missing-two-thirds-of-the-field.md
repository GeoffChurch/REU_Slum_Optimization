# C4: the bakeoff was missing two thirds of the field, and both earlier conclusions were artifacts of that (2026-08-08)

C2 and C3 ran **5 methods**. `compare_config` configures **16**, and `PeelReblocker` is configured
in `conf/method/peel.yaml` but absent from `all_methods` entirely, so it had never been a bakeoff
participant at all. Re-running the access-deficit curve over all 17
(`scratchpad/complexity/c4_full_roster.py`) overturns both notes.

## C2's headline was wrong

C2: *"only `topology` reaches k = 1."* With the full field, **7 of 17 methods reach universal access
on at least one block, and 4 reach it on the median block**:

    method                        k0_med  reaches univ.   disp@7m   disp@2m
    greedy_arterial_repulsion      0.000          90%       0.529     0.090
    topology                       0.000         100%       0.449     0.094
    clearance_looped               0.000         100%       0.651     0.313
    peel                           0.000         100%       0.906     0.728
    ... 13 others                 >= 1.0

## …but not for the reason I predicted

I expected `peel` to dominate, since its docstring says it builds "a connected centerline network
reaching the street (full access)". **It reaches universal access on 10/10 blocks and is the most
expensive way to do it**: 90.6% displacement at 7 m and 72.8% at 2 m, against `topology` at
44.9%/9.4% and `greedy_arterial_repulsion` at 52.9%/**9.0%**. Peel achieves the standard by paving
the block, and its matched-displacement curve is among the worst in the field.

Built for the objective is not the same as efficient at it.

## C3's headline was also wrong

C3 (5 methods) concluded `flow_paths_noreinforce` was most access-efficient at low budgets and
`topology` took over at 25%. **Neither wins anywhere in the full field.**

Burden (mean (depth−1)², 0 = universal access) at matched displacement, best per column in bold:

    roads at 7 m                    2%     5%    10%    15%    25%
    resistance_lp                 1.33   0.98   0.43   0.23   0.21   <- best at EVERY budget
    greedy_arterial_repulsion     1.41   1.05   0.77   0.51   0.23
    cycle_native                  1.46   1.19   0.53   0.33   0.29
    topology                      1.50   1.33   1.00   0.77   0.33
    flow_paths_noreinforce        1.40   1.21   0.89   0.73   0.46
    peel                          1.52   1.46   1.25   1.11   0.74
    osm_footpaths                 1.71   1.67   1.61   1.61   1.61   <- worst, and flat

    roads at 2 m                    2%     5%    10%    15%    25%
    greedy_arterial_repulsion     0.48   0.07   0.00   0.00   0.00   <- best at EVERY budget
    greedy_arterial_buildable     0.68   0.25   0.02   0.02   0.02
    resistance_lp                 0.52   0.22   0.21   0.21   0.21
    topology                      0.99   0.40   0.01   0.00   0.00
    peel                          1.46   1.40   1.11   0.81   0.53

**`greedy_arterial_repulsion` reaches universal street access at 10% displacement with 2 m lanes** —
exactly Lens A's operating budget.

## Two findings that survive and one that reverses

**Survives — width changes the ranking, and more strongly than C3 showed.** The winner is
`resistance_lp` at 7 m and `greedy_arterial_repulsion` at 2 m, at *every* budget. Not a crossover
within one width but a different champion per width.

**Survives — the axis carries information permeability does not.** `euclidean_grid` is near-last on
access at both widths while being mid-pack under permeability. `greedy_arterial_displacement` builds
almost nothing (0.008 displacement) and stays flat at burden 1.42 — nearly free and nearly useless.

**Reverses — my "negative control" framing.** I expected the permeability optimizers to do badly on
access, as evidence the two axes diverge. `resistance_lp` optimizes permeability directly and is the
**best** access method at 7 m at every budget. So optimizing flow does deliver access at street
width; the axes agree there and diverge at lane width. That is a more interesting result than the
one I was expecting and it weakens the case for access as a *separate reported axis* at 7 m, while
strengthening it at 2 m.

## The methodological lesson

Twice now a headline conclusion has been an artifact of a restricted method set, and nobody had
checked what the set was. The bakeoff was 5 of 17 with the objective-native method not even
registered. **Before any further metric work, the roster is part of the experiment and has to be
stated.** Adding the missing 12 cost one background run.

## Caveats

* `osm_footpaths` produced roads on only **5 of 10** blocks, and `demand_greedy` failed with an
  HTTPError on 2 (network flakiness in its desire-line source). Their rows are n = 5 and n = 8.
  `osm_footpaths` being worst on access is therefore suggestive, not established.
* 10 blocks, Cape Town, Voronoi cells rather than a cadastre — unchanged from C2/C3.
* `greedy_arterial_repulsion` is the slow one (~57 s worst block); it did not truncate.
* Prefix order is `street_first_ordered` throughout, so every cost is an upper bound on the true
  minimum over orderings.
