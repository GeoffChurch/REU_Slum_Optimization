---
title: "The arithmetic of access"
---

# The arithmetic of access

How many homes sit more than two houses' walk from a street? That question decides who an ambulance
cannot reach, and answering it exactly costs a Voronoi tessellation and a breadth-first peel of
every block in the city. This page derives a closed form that answers it from three numbers a
street map already gives you — the building count `n`, the block area `A`, and its perimeter `P` —
and shows that the screen this project ships is, algebraically, one member of the family that falls
out.

## Rings

Parcels tile the block, so the mean parcel is `s = √(A/n)` across. Peeling works inward one ring at
a time: ring 1 is everything touching a street, ring 2 everything touching ring 1, and so on. A
ring hugging the boundary therefore holds about `P/s` parcels, and the ring count that fits is the
inradius divided by `s`.

For a square of side `L` the inradius is `L/2 = 2A/P`, which gives

```
K = (2A/P) / sqrt(A/n) = 2·sqrt(nA)/P = 2d        where   d = sqrt(nA)/P
```

where `d` is the **depth proxy** this project already uses to rank blocks. So the depth proxy is
*half the ring count* — a fact worth stating plainly, because it is easy to assume otherwise and
the factor of two propagates into everything below.

## How many are deep

`k` rings consume a fraction `k/K` of each linear dimension, and the surviving interior shrinks in
**both** dimensions. So the count beyond depth `k` is

```
D(depth > k)  ~=  n·(1 − k/K)²  =  n·(1 − k·P / (2·sqrt(nA)))²
```

It is zero exactly when `√(nA)/P ≤ k/2` — the block is too thin or too sparse to have an interior
at all. That boundary is not imposed; it falls out.

*Check.* A 10×10 grid of parcels has `d = 2.5`, so `D(>2) = 100(1 − 1/2.5)² = 36`. Peel a 10×10
grid twice and 36 remain.

## How many are at depth exactly k

The difference of consecutive terms is **linear**, which makes it the simpler object of the two:

```
N(k)  =  D(>k−1) − D(>k)  =  P·sqrt(n/A) − (2k−1)·P²/(4A)
```

An arithmetic progression. The first ring holds `P√(n/A) − P²/4A` parcels — the perimeter divided
by parcel width, less a corner term — and every ring inward loses a further `P²/2A`. The corner
term audits itself: for a square `P²/4A = 4`, exactly the four corners the outer ring does not get.

## Weighting by depth

Because `N(k)` is linear in `k`, every polynomial weight collapses to a closed form:

```
sum over k of  k^m · N(k)   =   n · 2^(m+1) · d^m / ((m+1)(m+2))
```

| weight | quantity | value |
|---|---|---|
| `m = 0` | every parcel | `n` |
| `m = 1` | **total access burden** — every home's depth, summed | `⅔·n·d` |
| `m = 2` | burden with depth penalised quadratically | `⅔·n·d²` |
| `m = 3` | | `⅘·n·d³` |

Any polynomial weighting of depth gives `n` times a power of the depth proxy. Two corollaries drop
out of the same algebra: **mean depth `≈ ⅔·d`**, and mean depth is one third of maximum depth,
which is what a linearly-thinning ring structure must give.

## The shipped screen is burden per unit area

Take the `m = 1` row — the total access burden — and divide by the block's area:

```
(2/3)·n·d / A   =   (2/3) · n^1.5 / (P·sqrt(A))   =   (2/3) · depth_density_proxy
```

The metric this project screens on was derived as *depth × density*. It turns out to be **total
access burden per unit area**, up to a constant. That identity explains a behaviour we measured
long before we understood it: the proxy ranks blocks superbly and counts people badly, because a
count wants the burden *un*normalised — `n·d`, which is the same quantity multiplied back by `A`.

## Does it work

Against true Voronoi peels, with nothing fitted:

| | blocks | rank correlation with true deep count | total estimate / truth |
|---|---|---|---|
| Cape Town | 18,309 | 0.782 | 1.29× |
| Nairobi | 3,839 | **0.897** | **1.08×** |

It was derived on Cape Town and tested on Nairobi without refitting, where it does better. For
comparison, on the same task the depth proxy alone scores 0.760 and the raw building count 0.725.

The practical consequence is that a gate which needed a full tessellation no longer does. Requiring
*at least 50 homes more than two houses from a street, in fabric above 100 buildings per hectare*
selects 310 Cape Town blocks when evaluated on real peels; evaluated on the closed form it selects
333, agreeing with the exact answer at 90.7% precision and 97.4% recall — from columns a street map
already provides.

## What is not settled

The ring-count constant is not stable across cities. Measured against true peel depth, `max depth /
d` is 1.98 in Nairobi — essentially the predicted 2.00 — but 1.56 in Cape Town.

The obvious explanation is elongation, since `2A/P` is the inradius only for a square. **It is not
the explanation.** The elongation index `P²/16A` is 1.00 in Cape Town and 1.32 in Nairobi, so the
city that deviates more is the *less* elongated one, and an elongation-aware ring count derived
from the rectangle inradius makes the rank correlation worse in both cities. Candidates that remain
untested: parcel-size heterogeneity, interior open space, and disagreement between the building
count and the parcel count after clipping.

Rank correlation is unaffected by a constant factor, which is why the estimator transfers while its
calibration does not. Absolute counts from this formula should be read as order-of-magnitude until
that constant is understood.

The thresholds are a separate and weaker matter. "Two houses' walk" and "100 buildings per hectare"
are stated in physical units, but they were chosen against Cape Town: only three Nairobi blocks
clear both, despite that city holding 137,570 parcels beyond depth 2. Physical units bought
transferability for the *formula* and not for the *cut-points*.
