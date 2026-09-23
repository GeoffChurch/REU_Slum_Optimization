# Consensus re-measured at n=220 under the shared lenses: clearance wins, donors add nothing

**Date:** 2026-09-23
**Status:** measured. Confirms and strengthens
[`consensus-k-sweep-and-displacement`](2026-07-28-consensus-k-sweep-and-displacement.md), now at
11x the sample, on the current metric and a fair comparison.
Data: `data/benchmarks/consensus_matrix.parquet` (2,640 rows: 220 recipients x 12 arms, 0 skipped),
from `pixi run python -m scripts.consensus_matrix recipients=220 "k_sweep.arms=[consensus_held_out]"
"k_sweep.ks=[1,3,5,8,15,25]" out=data/benchmarks/consensus_matrix.parquet` at main 0947e66.

## What changed since July
The July study matched budgets with its own code, and the copies disagreed:
- consensus overshot the length budget while clearance undershot it;
- consensus and clearance were built to different depth targets;
- the donor-count sweep did not length-match the single donor.

It also ran on the old metric. Now:
- consensus is `DemandGreedyReblocker` fed by a `ConsensusDesireSource`, and the single-donor
  transplant is `DonorTransplantReblocker`, both bit-identical to the old code;
- every arm is truncated by the SAME code, `compare.lens_prefixes`: Lens A at D = 10% displacement,
  Lens B at P* = 0.60 permeability;
- the prediction lens is `budget.prefix_to_displacement` at each recipient's own network's
  displacement;
- the metric is the current one: per-road widths, the 7 m floor, footprint-overlap displacement,
  Open Buildings screening.

## Results
Paired per-recipient differences, n = 220 Cape Town recipients, as the median with a 95% bootstrap
interval (10,000 resamples of the median), plus the share of recipients where the first arm is better.

**Clearance beats consensus as a reblocker, on both reported axes.**

| | clearance - consensus (held-out) | clearance better in |
|---|---|---|
| Lens A permeability at D = 10% | +0.073 [+0.058, +0.100] | 75% |
| Lens B displacement to reach P* = 0.60 (both reached, n = 217) | -0.023 [-0.032, -0.017] | 80% |

**Donors add no predictive value.** Consensus reproduces each block's real footpaths no better than
clearance, which uses no donor information:

| | clearance - consensus |
|---|---|
| IoU at 10 m | -0.003 [-0.012, 0.000] |
| permeability at the prediction lens | +0.004 [-0.011, +0.021] |

**Averaging donors adds nothing.** Holding the extraction fixed, one donor is at least as good as
25:

| | k = 1 - k = 25 |
|---|---|
| Lens A permeability | +0.016 [+0.005, +0.031], k = 1 better in 61% |
| IoU at 10 m | +0.000 [0.000, +0.000] |

Consensus beats the raw single-donor transplant (+0.169 Lens A permeability [+0.121, +0.217]), but
that is the demand-greedy extraction beating snap-and-transplant. July found the same (+0.303 at
k = 1).

**No leakage.** Leaky minus held-out donors (inside vs beyond 2 km) has a median difference of
exactly 0 on both Lens A permeability and IoU.

Medians per arm:

| arm | Lens A perm | Lens B disp | Lens B reached | pred. perm | IoU@10m |
|---|---|---|---|---|---|
| clearance | 0.777 | 0.065 | 99% | 0.770 | 0.223 |
| consensus held-out | 0.694 | 0.094 | 100% | 0.706 | 0.215 |
| consensus leaky | 0.689 | 0.090 | 100% | 0.690 | 0.226 |
| single donor held-out | 0.547 | 0.088 | 44% | 0.496 | 0.133 |
| own OSM footpaths | 0.648 | 0.065 | 61% | 0.764 | 1.000 |

## What this settles, and what it does not
- **As a reblocker, consensus is dominated by clearance** on permeability at matched displacement
  and on displacement at matched permeability, and it costs far more (donor fetches plus GW fits).
- **As a prediction of real footpaths it ties with clearance** (IoU 0.215 vs 0.223, interval
  touching 0). So it is not strictly dominated on every axis. Clearance matching real footpaths
  about as well suggests the agreement comes from the block's own geometry, not from the donors.
- **The single-donor transplant is dominated by clearance on every measured axis.**
- **Untested:** consensus with donors drawn from other cities or wider pools. The pool here is
  Cape Town's 2,500 donors, and the no-distance-effect result makes wider pools unpromising.

## Decision
Both donor methods (`method=consensus`, `method=donor_transplant`) are KEPT as research methods
(owner, 2026-09-23), even though the single-donor transplant is dominated on every measured axis.
They stay off every published example lineup and out of `compare_config`'s `all_methods`, so they
cost the site nothing and are available if the donor line is reopened.

