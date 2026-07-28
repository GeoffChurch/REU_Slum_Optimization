# Barycenter consensus at n=20: functionally strong, geometrically not a reconstruction

**Date:** 2026-07-28
**Status: HEADLINE CORRECTED by [`consensus-k-sweep-and-displacement`](2026-07-28-consensus-k-sweep-and-displacement.md).**
Two claims below do not survive. (a) "beats the best single donor in 100% of blocks, +0.412" was
measuring the EXTRACTION method, not consensus: at k=1, where averaging cannot operate,
`demand_greedy_reblock` beats `gap_snap_routed` by +0.303 in 95% of blocks, and going from k=1 to
k=30 is worth −0.009 (p=0.064). Averaging donors adds nothing. (b) the reblocker parity was an
artefact of length-matching: at matched DISPLACEMENT, clearance wins (ratio 0.87–0.99, consensus
ahead in only 30–35% of blocks). The leakage result and the IoU result below both stand.

**Measured:** 20 Cape Town recipients × 15 donors, both holdout arms.
`scripts/consensus_matrix.py`, artifact `data/benchmarks/consensus_matrix.parquet`.

The single-donor question closed as a null ([`no-detectable-distance-effect`](2026-07-28-no-detectable-distance-effect.md)).
This tests the mechanism the 2026-07-23 study actually found promising — a weighted consensus of
several similar blocks' real OSM footpaths — which had never been run beyond n=1.

## Result (held-out arm: donors > 2 km away)

| | median | IQR |
|---|---|---|
| consensus permeability / block's **own OSM** | **1.093** | [1.015, 1.446] |
| consensus permeability / **direct clearance** | **1.041** | [0.977, 1.113] |
| IoU@10 m vs own OSM | 0.266 | [0.139, 0.453] |
| chamfer recall (real paths missed) | 13.5 m | [8.6, 17.0] |
| chamfer precision (paths drawn that aren't there) | 11.9 m | [7.1, 21.9] |

- beats the block's **own OSM** on permeability in **85%** of blocks
- beats **direct clearance** in **65%**
- beats the **best single donor** in **100%**, median gain **+0.412** permeability

## The single-donor → consensus jump is the solid finding

100% of blocks, median +0.412. That is not marginal and it is not noise-limited. It confirms the
2026-07-23 diagnosis — a single donor copies whichever donor's idiosyncratic coverage gaps it
happened to get, and averaging several washes them out — and it is the one claim here that needs
no hedging.

## Leakage is not inflating it

The 2 km exclusion costs **nothing**, paired across the same 20 recipients:

```
perm_ratio_own     held_out 1.093   leaky 1.104   median delta +0.000   wilcoxon p=0.96
iou_10m            held_out 0.266   leaky 0.286   median delta +0.000   wilcoxon p=0.09
chamfer_recall_m   held_out 13.45   leaky 12.45   median delta +0.000   wilcoxon p=0.14
```

This was a live concern, not a formality: a median 26.7% of a recipient's nearest 15 donors sit
inside 2 km and for 24.5% of recipients all 15 do, and the original 94% figure was measured with no
distance constraint at all. It turns out not to matter — consistent with the separate finding that
geographic distance is uncorrelated with GW distance, so nearby donors are not privileged donors.

## But it is NOT reconstructing the real network

IoU@10 m of **0.266** and chamfer recall of **13.5 m** say the consensus network is not where the
real footpaths are. One recipient makes the point starkly: permeability ratio 2.08 with IoU 0.001 —
a network **twice as permeable** as the block's own, sharing essentially no geometry with it.

So the two branches get different answers from the same run:

- **As a reblocker**: consensus is competitive, and beats single-donor transplant outright.
- **As a prediction** of a block's real footpaths — the framing the 94% figure was reported under —
  it fails. It finds *a* good network, not *the* network. "Predict the unmapped block's footpaths"
  is not supported by these numbers.

## And the reblocker win is Pareto-ambiguous

Displacement, at matched length: consensus **0.248**, the block's own OSM **0.160**, direct
clearance **0.205**. Consensus buys its extra permeability by displacing more homes — ~21% more
than clearance and ~55% more than the real network. Since permeability and displacement are the
repo's paired primary metrics, "1.04× permeability" is only half a sentence: consensus is **more
permeable and more destructive**, on both comparisons, and neither dominates.

The honest reblocker claim is therefore *competitive with clearance on a different point of the
tradeoff*, not *better than clearance*.

## Caveats

- One city, 20 recipients, k=15. Nairobi's pool is too homogeneous to replicate (GW range 1.71×)
  and Gauteng is 3% mapped.
- Not numerically comparable to the 2026-07-23 "94%": that ran on the pre-fix gradient, so its
  ε=0.01 was an effective 0.02. Same mechanism, half the regularization here.
- Length-matched, not displacement-matched. A displacement-matched comparison would be the
  fairer test of the reblocker claim and is not run here.
- `k=15` is inherited, not calibrated. The k-sweep is the obvious next measurement, and the
  single-donor→consensus gain of +0.412 suggests the curve is steep at the low end.
