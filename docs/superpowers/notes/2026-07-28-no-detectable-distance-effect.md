# Decoupling the roles: no detectable donor-distance effect

**Date:** 2026-07-28
**Status:** the best-designed run available. `data/benchmarks/gw_pair_matrix.parquet` is now this
matrix. Supersedes every earlier β in this arc.

## The design error

`load_pools` returned **one** pool, so recipients and donors were selected by the same criterion.
They should not be. A recipient needs building points and parcels — it is a reblocking target, so
the screen should decide it. A donor needs interior footpaths — it is material to transplant, and
nothing about being dense-and-compact itself is required.

Holding both roles to `density_compactness` made every donor morphologically near-identical to its
recipient, which starves the only independent variable the experiment has. Decoupling widens the
corpus-wide donor pool **57×**, from 247 to 14,189 blocks (Cape Town 147 → 1,415; Gauteng 18 →
3,966).

Donor eligibility now comes from the census (`interior_length_m_0.5 >= 100`) rather than being
discovered by fetching, which is also what makes it cheap: the Gauteng run had spent 509
`empty_interior` skips to find 68 usable pairs.

## Result

```
within-recipient beta = -0.1817   SE = 1.7234   p = 0.916
donor bootstrap 95%   [-4.094, +3.854]   median -0.324   negative in 56.1%
real_gw_dist range    10.11x  (widest of any run)
```

Four properly-powered 500-pair runs, ordered by how much range the design achieved:

| run | n | GW range | β | bootstrap 95% | negative in |
|---|---|---|---|---|---|
| Nairobi, screened both roles | 500 | 1.71× | +7.22 | [−19.16, +31.69] | 29% |
| Cape Town, screened both roles | 89 | 4.17× | +5.37 | [−11.97, +16.94] | 27% |
| Cape Town, legacy band | 500 | 8.96× | −2.89 | [−6.33, +0.59] | 95% |
| **Cape Town, decoupled** | **500** | **10.11×** | **−0.18** | **[−4.09, +3.85]** | **56%** |

**The conclusion is a null.** The run with the widest donor-distance range and the tightest
interval puts β at −0.18, dead on zero, with donor resamples negative 56% of the time — a coin
flip.

This also kills the "wider range → more negative" pattern I read off the first three runs. That
was three points and a story; the fourth point breaks it. Range restriction was a real limitation
of those designs, but it was not hiding an effect.

## What is and is not concluded

**Concluded:** for *single-donor* transplant, the fidelity of the result does not measurably depend
on the donor's GW distance to the recipient, within ±4 units at 95% confidence, in Cape Town at
n=500 over 20 recipients.

**Not concluded:** that donor choice never matters, or that OT transplant is worthless. In
particular this says nothing about **barycenter consensus over many donors**, which is a different
mechanism and which the 2026-07-23 study found to be the promising one — reaching ~94% of a
block's own OSM where single-donor transplant was Pareto-dominated.

That consistency is worth noting rather than glossing: the 2026-07-23 note already concluded
**single-donor transplant is dead**. This result is what that conclusion predicts. The pair matrix
has been measuring single-donor transplant all along, so a null here corroborates the earlier
finding rather than contradicting it.

## The strategic consequence

**The retrieval index (Phase 2) is premature, and possibly unnecessary.** Its purpose is to find
the nearest donor quickly. If nearest-donor distance does not predict transplant quality, a faster
nearest-donor lookup buys nothing. Building a masked-NCC FFT index on top of this would be
optimizing a step whose output does not matter.

What the evidence supports instead is testing the **consensus** mechanism at scale — many donors
averaged, which is where the one strong result in this arc came from, and which needs a donor pool
rather than a donor ranking. The decoupled donor pool (14,189 blocks) is exactly the right input
for that, and it now exists.

## Caveats

- One city, 20 recipients. Nairobi's own pool is too morphologically homogeneous (range 1.71×) to
  replicate this, and Gauteng is unmapped (3% of screened blocks can donate).
- `perm_gap` is a length-matched comparison against a direct clearance solve; a different outcome
  measure could behave differently.
- An effect smaller than roughly ±4 units would not be detected here.
