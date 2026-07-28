# The consensus gain is the extraction, not the consensus — and it loses on displacement

**Date:** 2026-07-28
**Status:** measured, 20 recipients × 8 rungs of k, both matching conventions.
`scripts/consensus_sweep.py`, data `scratchpad/ot/consensus_sweep.parquet`.
**Corrects [`consensus-at-scale`](2026-07-28-consensus-at-scale.md)'s headline reading.**

## 1. Averaging donors adds nothing

| k | perm/own | perm/direct (length-matched) | IoU@10 m | perm/direct (**displacement-matched**) |
|---|---|---|---|---|
| 1 | 1.128 | 1.031 | 0.275 | 0.989 |
| 2 | 1.111 | 1.034 | 0.272 | 0.884 |
| 3 | 1.085 | 1.020 | 0.284 | 0.872 |
| 5 | 1.087 | 1.033 | 0.287 | 0.884 |
| 8 | 1.097 | 1.030 | 0.276 | 0.963 |
| 12 | 1.083 | 1.041 | 0.278 | 0.928 |
| 20 | 1.101 | 1.040 | 0.287 | 0.982 |
| 30 | 1.080 | 1.032 | 0.307 | 0.899 |

The curve is **flat**. k=30 over k=1: median **−0.0086** permeability, Wilcoxon p = 0.064 — if
anything slightly negative. Thirty donors are no better than one.

## 2. So what was the "+0.412 over the best single donor"?

Not consensus. **The extraction method.**

At k=1 there is exactly one donor, so averaging cannot be operating. Comparing the two ways of
turning that single donor into a network:

```
k=1 demand-field extraction (demand_greedy_reblock)  perm median 0.915
same donor, gap-snapped     (gap_snap_routed)        perm median 0.541
gain +0.303, extraction wins in 95% of blocks, wilcoxon p < 0.0001
```

The gain reported yesterday as "consensus beats the best single donor in 100% of blocks" was
measuring `demand_greedy_reblock` — routing along the **recipient's own ChordSubstrate** guided by
a demand field — against `gap_snap_routed`, which warps the donor's geometry and snaps it. The
comparison was extraction-vs-snapping and I read it as consensus-vs-single.

This also explains the [null distance effect](2026-07-28-no-detectable-distance-effect.md), and
the two measurements now corroborate each other: if the donor only supplies a coarse demand hint
that the recipient's own substrate then re-routes, then neither *which* donor nor *how many*
should matter much. Neither does.

## 3. At matched displacement, clearance wins

Everything until now was length-matched. That convention flattered consensus:

```
same blocks, k=8:
  LENGTH-matched        consensus / clearance = 1.030
  DISPLACEMENT-matched  consensus / clearance = 0.963   (consensus wins in 35% of blocks)
```

Across the ladder the displacement-matched ratio runs **0.87–0.99**, never above parity. And
consensus needs **more** road to reach the same displacement (193 m vs clearance's 144 m at k=8) —
it spends length in cheaper places without converting it into permeability as well.

Displacement is the harder constraint: it counts homes destroyed. A method that only wins when the
comparison is made on length is not winning.

## What this settles

**The OT transplant arc does not beat a direct clearance solve.** That was the 2026-07-23
conclusion, reopened this session on the grounds that it was underpowered at n=1. It is now
measured at 20 recipients, with a leakage holdout, a corrected GW solver, a calibrated screen and a
57×-larger donor pool — and it reinstates the original verdict on much better evidence.

Nor is it a reconstruction: IoU@10 m stays at **0.27–0.31** at every k, so the prediction framing
("recover an unmapped block's real footpaths") is not supported either.

## The keeper

`demand_greedy_reblock` is a **good extraction method**: +0.303 permeability over gap-snapping, in
95% of blocks, p < 0.0001. That result does not depend on the donor material at all — it is about
routing along the recipient's own substrate under a demand prior.

That suggests the obvious follow-up, and it no longer involves OT: **where else could the demand
prior come from?** If one donor is as good as thirty, the field is carrying very little
information, and a cheap heuristic prior — or none — may do as well. At k=1 the
displacement-matched ratio against clearance is 0.989, i.e. parity. A demand-prior variant of
clearance is a small experiment and does not need a donor, a GW fit, or a retrieval index.
