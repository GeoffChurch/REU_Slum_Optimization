# What separates real footpath networks from synthesized ones

**Date:** 2026-07-28
**Status:** measured, 25 Cape Town blocks, paired within block, matched on displacement.
`scripts/real_vs_synthetic.py`.

The standing intuition — real networks carry qualities our metrics do not reward — is testable now
that the census knows 16,497 blocks' real networks. Score the real network and several synthesized
ones on candidate statistics, paired within block, and see what separates.

## Results (paired Wilcoxon vs the real network, same blocks)

| statistic | real | clearance | p |
|---|---|---|---|
| frontage_frac | **0.016** | 0.065 | <0.0001 |
| road_per_parcel_m | **1.99** | 1.47 | <0.0001 |
| mean_leg_m | **13.9** | 16.1 | 0.023 |
| cycle_ratio (mean) | **0.024** | 0.000 | 0.018 |
| deadend_frac | 0.682 | 0.750 | 0.063 |
| straightness | 0.921 | 0.894 | 0.80 |

## The finding: real networks spread access evenly rather than concentrating it

Two of these point opposite ways and that is the interesting part.

Real networks give **four times fewer parcels direct frontage** (0.019 vs 0.097 mean) yet leave a
**shorter mean walk** to a road (17.4 m vs 19.0 m). Our methods put more parcels *on* a path while
leaving others further away; real paths put fewer parcels on a path but nobody far from one.

That is a claim about the **distribution** of access, not its total — and no current metric measures
it. Permeability is a flow aggregate, displacement counts homes destroyed, and neither notices that
one network's access is lumpy and another's is even. **The concrete metric candidate this suggests
is distributional: p90 walk-leg, or a Gini over per-parcel legs, rather than the mean.**

Real networks also use **~35% more road per parcel at matched displacement** (1.99 m vs 1.47 m),
which is consistent: threading a longer, gentler route through gaps buys evenness without demolition.

## Loops: real, small, and completely absent from our output

`cycle_ratio` is 0 for every synthesized network on every block — all of them are strict drainage
trees, by construction. **7 of 25 real networks contain at least one cycle** (mean ratio 0.024,
p=0.018). The effect is small but it is one-sided, and it is exactly what `LoopClosureRefiner`
exists to address.

## Two methodological cautions from this run

**`looped_tree` scored identically to `clearance` on every statistic, twice.** `LoopClosureRefiner`
appends its connectors *after* the base tree, so any prefix-based budget match — length or
displacement — truncates the loops away entirely and measures the base. Prefix matching can never
show loop closure's benefit. Anything comparing that method on a budget needs a different
convention.

**The first version of this measurement was broken and its numbers should be ignored.** Keying the
network graph on raw segment endpoints found no shared nodes at all in real OSM data (every endpoint
degree 1), so `cycle_ratio` read 0.000 everywhere including for real networks — the statistic
measured nothing. Real footpaths cross mid-segment and are not drawn to share endpoints;
`unary_union` must node the collection first. That fix is what turned cycle_ratio from a flat zero
into the one-sided signal above, and it also flipped the sign of `mean_leg_m`.
