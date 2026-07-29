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

## CORRECTION: measured with the repo's own walk-distance primitive, this reverses

The table above uses a crow-flies distance from each parcel to the nearest road line. That is the
wrong measure — it ignores that a parcel behind three others must walk *through* them — and the
repo already ships the right one: `derive.geometric_access.geometric_access_distances` walks the
parcel-adjacency graph, and `StructureEval` already exposes its p95 as "B equity".

Re-measured with it, on the same 25 blocks at matched displacement:

| statistic | real | clearance | demand_greedy | p (real vs clearance) |
|---|---|---|---|---|
| walk mean (m) | 7.85 | **3.83** | 4.42 | 0.0008 |
| walk p95 (m) | 21.16 | **13.70** | 13.70 | 0.0083 |
| walk max (m) | 25.46 | **18.16** | 17.03 | 0.0250 |
| walk Gini | **0.563** | 0.591 | 0.646 | 0.092 |
| meshedness | 0.011 | 0.009 | 0.009 | 0.151 |
| frontage_frac | 0.421 | **0.597** | 0.556 | 0.0032 |
| road per parcel (m) | 1.99 | **1.47** | 1.52 | <0.0001 |

**The synthesized networks are simply better on access.** Real footpaths leave people walking twice
as far on average, and further at p95 and max, while using ~35% more road for the same
displacement. The earlier "real networks leave shorter walks" claim was an artefact of the
crow-flies measure and does not survive.

## So the "missing metric" hypothesis is not supported

Nine candidate statistics were tried. Only two point the real networks' way, and both are weak:

- **walk Gini 0.563 vs 0.591** (p=0.092) — real access is marginally more *even*, which is the one
  surviving trace of the distributional intuition.
- **cheap loops** — see below.

Everything else says our methods are better, or says nothing. The honest conclusion is that real
networks' advantage is confined to **displacement**, which is already a primary metric and which we
already know we lose on. There is no evidence here for a quality the metric set is blind to.

Worth stating plainly: this measurement was wrong three times in a row — an endpoint-keyed graph
that found no nodes, a length prefix that deleted the loops, and a crow-flies distance standing in
for a walk. Each time the fix was to use a primitive the repo already had.

## The one actionable finding: real networks close loops CHEAPLY

## (superseded) The finding: real networks spread access evenly rather than concentrating it

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

## Loops — and a correction

An earlier version of this note said "`cycle_ratio` is 0 for every synthesized network on every
block — all of them are strict drainage trees, by construction." **That was wrong**, and wrong in a
way that matters: `LoopClosureRefiner` produces genuinely looped networks, and the zeros were an
artefact of my own budget matching.

Measured directly on the FULL (untruncated) output:

| block | base segs | looped segs | cycles, full | cycles, truncated | segs kept |
|---|---|---|---|---|---|
| ZAF.9.3.1_1_38143 | 13 | 14 | 0 | 0 | 13/14 |
| ZAF.9.3.1_1_55726 | 21 | 24 | **2** | 0 | 3/24 |
| ZAF.9.3.1_1_41223 | 33 | 37 | **2** | 0 | 1/37 |
| ZAF.9.3.1_1_23788 | 57 | 64 | **6** | 0 | 4/64 |
| ZAF.9.3.1_1_40968 | 83 | 92 | **6** | 0 | 33/92 |
| ZAF.9.1.4_1_4399 | 149 | 166 | **17** | 0 | 16/166 |

The Looped Tree has loops on 5 of 6 blocks. The displacement-matched prefix removes 80–97% of its
segments, and every loop with them.

What survives from the original claim is only the comparison against the real networks: **7 of 25
real networks contain a cycle within the same displacement budget** (mean ratio 0.024, p=0.018)
where the truncated synthetic ones contain none. Real footpaths close a loop *cheaply*, early in
the budget; loop closure as implemented spends its connectors last, so under a tight budget they
are the first thing cut.

That is a statement about **where in the budget a method spends its loops**, not about whether it
can make them — and it is arguably a more useful finding than the one it replaces.

## Two methodological cautions from this run

**`looped_tree` scored identically to `clearance` on every statistic, twice.** `LoopClosureRefiner`
appends its connectors *after* the base tree, so any prefix-based budget match — length or
displacement — truncates the loops away entirely and measures the base. Prefix matching can never
show loop closure's benefit. Anything comparing that method on a budget needs a different
convention, and the table above quantifies how severe the truncation is (down to 1 of 37 segments
on one block).

**The first version of this measurement was broken and its numbers should be ignored.** Keying the
network graph on raw segment endpoints found no shared nodes at all in real OSM data (every endpoint
degree 1), so `cycle_ratio` read 0.000 everywhere including for real networks — the statistic
measured nothing. Real footpaths cross mid-segment and are not drawn to share endpoints;
`unary_union` must node the collection first. That fix is what turned cycle_ratio from a flat zero
into the one-sided signal above, and it also flipped the sign of `mean_leg_m`.


## Acting on it: cheap loops improve the flagship

`clearance_looped` sets `min_loop_len_m: 40.0`, which excludes exactly the short connectors that
would score highest on bridges-removed-per-metre and survive a budget prefix. Dropping it to 5 m
(`clearance_looped_cheap`) improves the Looped Tree on **both** lenses of the density_compactness
Cape Town example (18 blocks, 4,615 parcels):

| | Lens A permeability | Lens A road | Lens B road | Lens B displacement |
|---|---|---|---|---|
| clearance_looped | 0.7426 | 2750 m | 1474 m | 0.0564 |
| **clearance_looped_cheap** | **0.7463** | **2698 m** | **1434 m** | **0.0549** |

Better permeability with less road on Lens A; less road and less displacement on Lens B. Small, but
a strict improvement on every axis, from a one-parameter change motivated directly by the property
the real networks demonstrably have.
