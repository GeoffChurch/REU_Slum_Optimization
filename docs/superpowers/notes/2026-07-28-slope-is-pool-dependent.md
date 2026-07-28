# The fidelity-vs-distance slope is a property of the pool, not a general result

**Date:** 2026-07-28
**Status:** measured, three cells of a 2×2. Supersedes the headline reading of
`2026-07-27-gw-pair-matrix-findings.md` without contradicting its arithmetic.

## What happened

`scripts/pair_matrix.py` defined its own pool — `building_count in [60,300] AND k_complexity >= 4`
— rather than selecting through the repo's `Screen`. Repointing it at `density_compactness`
(n/P² at its calibrated absolute floor) changed the answer:

| pool | donor source | β | p | permutation p |
|---|---|---|---|---|
| hand-rolled band, k≥4 | Overpass | −9.582 | 0.0141 | 0.0110 |
| hand-rolled band, k≥4 | local PBF | **−9.592** | 0.0139 | 0.0106 |
| screen (n/P² ≥ 3.55e-4) | local PBF | **+5.374** | 0.4259 | 0.4921 |

Two things changed at once, so the third cell was filled to separate them.

**The donor source is exonerated.** The PBF reproduces the Overpass result to three significant
figures. That is also a free validation of `PbfDesireLines` against the live API on the same 100
pairs — worth having, since the census depends on the same reader.

**The pool is entirely responsible.**

## It is not range restriction

The obvious explanation is that the screen selects a morphologically tighter population, so there
is less variation in the independent variable. That is true as far as it goes — the screened
pool's within-recipient GW spread is **64%** of the legacy pool's (sd 0.00299 vs 0.00469), and
the range ratio drops from 6.96× to 4.17×.

But it does not explain the result. Trimming the **legacy** pool per recipient to an even narrower
spread (sd 0.00206, tighter than the screened pool) leaves the slope intact and larger:

```
legacy full                          n=100  beta= -9.592  p=0.0139
legacy TRIMMED to a narrower x-spread n= 81  beta=-26.861  p=0.0028
screen                                n= 89  beta= +5.374  p=0.4259
```

Selection on x attenuates a *correlation* but does not bias an OLS *slope*, and that is exactly
what this shows. Narrowing the legacy pool's range does not reproduce the screened pool's null, so
the two populations genuinely differ.

## What this costs the programme

**β = −9.58 is not a general fact about transplant fidelity.** It is a property of the k≥4
population it was measured on. The 2026-07-27 note's arithmetic stands — the Simpson's-paradox
correction, the statsmodels cross-check, the POT solver cross-validation all hold — but its
headline should be read as scoped to that pool, and the effect does not survive the change.

Nor is the new number evidence of a positive effect: n=89, p=0.43, one city. The honest statement
is that **there is no detectable relationship in the screened pool**, and that the relationship
found in the legacy pool does not transfer.

Routing through the repo's `Screen` is what exposed this. The private pool had been hiding a
dependence on the pool definition for the whole arc, and no shipped method's numbers were
comparable to it either.

## The outline hypothesis — proposed here, then MEASURED AND REFUTED (same day)

The section below proposed that GW distance in the legacy pool was substantially encoding
**outline** difference, and that outline similarity was what predicted transplant success. It was
tested immediately (`scratchpad/ot/outline_vs_fabric.py`) on the same 100 legacy pairs, with two
descriptors that are both rotation/translation/scale invariant but see different things: an
outline EDM spectrum over 64 uniformly-sampled boundary points, and a fabric descriptor (the k-NN
distance profile of parcel centroids, k=1..6, normalized by the mean 1-NN distance).

**It is wrong.**

```
within-recipient correlation with real_gw_dist:  outline +0.444   fabric +0.237

within-recipient slope on perm_gap, each alone:
  real_gw_dist   beta=  -9.592  se= 3.811  t=-2.52  p=0.0139
  outline_dist   beta=  -0.012  se= 0.011  t=-1.13  p=0.2630
  fabric_dist    beta=  +0.074  se= 0.170  t=+0.43  p=0.6658

real_gw_dist controlling for outline_dist:
  real_gw_dist   beta=  -9.498  se= 4.281  t=-2.22
  outline_dist   beta=  -0.001  se= 0.012  t=-0.05
```

GW distance **keeps essentially its whole slope** when outline is controlled for (−9.592 →
−9.498), and outline collapses to nothing (t = −0.05). Outline is genuinely *inside* GW distance —
it correlates at r = +0.444 — but it is not the part that predicts fidelity. Neither is local
packing.

So GW's predictive content is in the full joint correspondence structure, and does not decompose
into "global outline shape" plus "local packing" — at least not as measured by these two
descriptors. That is a real, if inconvenient, argument for keeping the expensive GW fit rather
than replacing it with cheap shape features.

**Limit of this test.** Controlling for a *noisily measured* covariate only partially controls for
it, so a better outline descriptor could still claim some of the slope. The 64-point boundary
spectrum is crude. What the result rules out is outline being the *dominant* carrier, which is
what the hypothesis claimed.

**The mechanism behind the pool dependence is therefore still unknown.** Depth is ruled out (mean
recipient depth 4.40 vs 4.24), donor source is ruled out, range restriction is ruled out, and now
outline is ruled out. That the effect is pool-dependent stands; why, does not.

## The hypothesis worth testing next (SUPERSEDED — see above)

Depth does not explain the difference: mean recipient depth is 4.40 (legacy) vs 4.24 (screen).
The pools are equally deep.

What differs is **outline variety**. The legacy band constrains `k_complexity` and building count
but nothing about shape, so its blocks have uncontrolled outlines. `density_compactness` selects
on n/P², which is *directly* a shape term — so it homogenizes outlines by construction.

If GW distance in the legacy pool was substantially encoding **outline** difference, and outline
similarity is what predicted transplant success, then β = −9.58 was measuring something much
weaker than "morphologically similar donors transplant better": it was closer to "donors with
similar block outlines transplant better." Homogenize the outlines and the signal has nothing left
to track — which is what the screened pool shows.

This is testable and cheap: decompose GW distance into an outline component and an interior-fabric
component on the legacy pairs, and check which one carries the correlation with `perm_gap`.

It also raises the stakes on the shape-standardizing `RegionBuilder` that was specified and never
built (see the Phase 1 spec's "Retrieval unit" section). That builder was motivated as removing
outline as a *confound*. If outline is instead carrying the *signal*, the same measurement is
still the right one to run — the outline's share of inter-region GW distance variance — but the
conclusion it feeds is different, and worth knowing before Phase 3's donor-material test is
designed around it.
