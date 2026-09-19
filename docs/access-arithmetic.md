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

$$K \;=\; \frac{2A/P}{\sqrt{A/n}} \;=\; \frac{2\sqrt{nA}}{P} \;=\; 2d,
\qquad d \;\equiv\; \frac{\sqrt{nA}}{P}$$

where `d` is the **depth proxy** this project already uses to rank blocks. So the depth proxy is
*half the ring count* — a fact worth stating plainly, because it is easy to assume otherwise and
the factor of two propagates into everything below.

## How many are deep

`k` rings consume a fraction `k/K` of each linear dimension, and the surviving interior shrinks in
**both** dimensions. So the count beyond depth `k` is

$$D(\text{depth} > k) \;\approx\; n\left(1 - \frac{k}{K}\right)^{2}
\;=\; n\left(1 - \frac{kP}{2\sqrt{nA}}\right)^{2}$$

It is zero exactly when `√(nA)/P ≤ k/2` — the block is too thin or too sparse to have an interior
at all. That boundary is not imposed; it falls out.

*Check.* A 10×10 grid of parcels has `d = 2.5`, so `D(>2) = 100(1 − 1/2.5)² = 36`. Peel a 10×10
grid twice and 36 remain.

## How many are at depth exactly k

The difference of consecutive terms is **linear**, which makes it the simpler object of the two:

$$N(k) \;=\; D(>k-1) - D(>k) \;=\; P\sqrt{\frac{n}{A}} \;-\; (2k-1)\,\frac{P^{2}}{4A}$$

An arithmetic progression. The first ring holds `P√(n/A) − P²/4A` parcels — the perimeter divided
by parcel width, less a corner term — and every ring inward loses a further `P²/2A`. The corner
term audits itself: for a square `P²/4A = 4`, exactly the four corners the outer ring does not get.

## Weighting by depth

Because `N(k)` is linear in `k`, every polynomial weight collapses to a closed form:

$$\sum_{k} k^{m}\,N(k) \;=\; n\,\frac{2^{\,m+1}\,d^{\,m}}{(m+1)(m+2)}$$

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

$$\frac{\frac{2}{3}\,n\,d}{A} \;=\; \frac{2}{3}\cdot\frac{n^{1.5}}{P\sqrt{A}}
\;=\; \frac{2}{3}\,\texttt{depth\_density\_proxy}$$

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

## Is a tessellation the right assumption?

The derivation assumes parcels tile the block on a characteristic scale `s`. The real parcels are a
Poisson–Voronoi tessellation of building points, so an alternative model — place `n` points
uniformly at random and ask for the depth distribution — is arguably more faithful, and would smear
the sharp rings into something closer to geometric decay.

Measured, the tessellation assumption wins. Taking blocks whose maximum depth is *exactly* 8, so
that no block drops out of the tally as `k` grows:

| k | N(k) | ratio to previous | difference |
|---|---|---|---|
| 3 | 4,657 | 0.81 | −1,102 |
| 4 | 3,565 | 0.77 | −1,092 |
| 5 | 2,555 | 0.72 | −1,010 |
| 6 | 1,678 | 0.66 | −877 |
| 7 | 835 | 0.50 | −843 |

The successive **differences** hold within about 25%; the **ratios** span 0.81 down to 0.50. That is
an arithmetic progression, which is what the tessellation model predicts and what a smeared random
model does not.

Conditioning on each block's own maximum depth matters. Aggregating rings across blocks of
different depths produces a clean-looking geometric decay with ratio ≈ 0.55, which is not a ring
structure at all — it is the distribution of maximum depths across blocks, with blocks dropping out
of the tally as `k` passes their deepest parcel.

*Caveat:* `access_before` may index street-fronting parcels at depth 0, in which case the series
above is offset by one. The linear-versus-geometric conclusion does not depend on the offset; the
absolute first-ring count does.

## The ring constant: a latitude bug, not geometry

Measured against true peel depth, `max depth / d` was 1.98 in Nairobi — the predicted 2.00 — but
1.56 in Cape Town. Four explanations were proposed and eliminated (elongation, aspect ratio,
boundary digitisation detail, parcel-size heterogeneity — the last one's within-city correlation
runs the *wrong way*, +0.34). The cause turned out to be neither geometry nor the model:

| | `block_area_m2` ÷ true geometry area |
|---|---|
| Cape Town | **1.4491** (IQR 1.4466–1.4518) |
| Nairobi | 0.9999 |

The `block_area_m2` column shipped with the block data is 45% too large for Cape Town. The factor
is not arbitrary: \(1/\cos^{2}(33.9^\circ) = 1.452\). The column was computed from WGS84 degrees
without the `cos(latitude)` correction, so it is inflated by \(1/\cos^{2}\phi\) — which at
Nairobi's latitude of 1.3° is 1.0005 and invisible. **The same defect is in both files; only Cape
Town is far enough from the equator to show it.**

`reblock.metric._cols` prefers that column over the geometry, so Cape Town's `A` enters every
metric inflated by 1.449, and `d = √(nA)/P` by `√1.449 = 1.204`. Substituting the true area:

| | with the column | with true geometry area |
|---|---|---|
| Cape Town | 0.782 | **0.942** |
| Nairobi | 0.988 | 0.987 |

which closes the gap and leaves a residual comfortably inside what a square-based approximation
should be expected to give.

**What this does and does not affect.** Rankings *within* Cape Town are unharmed: the inflation is
very nearly constant (IQR 1.4466–1.4518), and a constant factor cannot reorder blocks. Everything
absolute is affected — the `0.0128` floor, the "100 buildings per hectare" threshold above, and any
comparison *between* cities, since one city's areas are right and the other's are not. Densities
quoted for Cape Town from this column are understated by the same 1.449.

## What is not settled

The thresholds are a separate and weaker matter. "Two houses' walk" and "100 buildings per hectare"
are stated in physical units, but they were chosen against Cape Town: only three Nairobi blocks
clear both, despite that city holding 137,570 parcels beyond depth 2. Physical units bought
transferability for the *formula* and not for the *cut-points*.
