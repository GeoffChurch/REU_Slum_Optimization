# `max_anchors` is a region-scale win, and a block-scale mistake

**Date:** 2026-08-11
**Branch:** `continuum-permeability`
**Harnesses:** `scripts/perf/anchor_cap.py` (block), `scripts/perf/region_anchor_cap.py` (region)
**Data:** `scripts/perf/anchor_cap.json`, `scripts/perf/region_anchor_cap.json`

The handoff left `max_anchors` "unevaluated, not rejected", expecting it to *cost* access quality:
it drops per-vertex anchors and biases toward long chords, "which for an access objective is
precisely the wrong bias." Measured, the bias does not appear at either scale, and at region scale
the cap is **8.2× faster and better on both reported metrics**.

---

## The headline

Region block, 11,006 parcels, tier-2 shortlist 512, 15 roads, matched displacement budget 0.10:

| `max_anchors` | burden_red | perm | road_m | minutes | candidate growth |
|---|---|---|---|---|---|
| 128 | 0.8705 | **0.6369** | 3,779 | **9.7** | 1.62× |
| 256 | 0.8632 | **0.6476** | 3,643 | **13.1** | 1.37× |
| uncapped (shipped) | 0.8610 | 0.5485 | 3,635 | 79.7 | 2.52× |

`128` vs uncapped: burden **+0.0095**, perm **+0.0884**, **8.2×** faster.
`256` vs uncapped: burden **+0.0021**, perm **+0.0991**, **6.1×** faster.

The uncapped arm reproduced the previously recorded run exactly — identical candidate counts at
every one of the 15 steps (468,968 → 1,180,388), 29 road rows, 3,635 m, 79.7 min against the
recorded 79.6 — so this is the same measurement, not a differently-configured one.

**Why the permeability number is the one that matters.** Burden is what the greedy optimizes;
permeability is co-reported and nothing selects on it. A burden-only difference is tie-break
scatter. Here burden is flat (both deltas far inside the block-scale noise band) and permeability
moves **+0.09 in the same direction under two independent caps**. That is the pre-registered
signature of a real structural difference rather than noise — stated in the harness docstring
before the run, not chosen afterwards.

---

## `max_anchors` is a mode switch, not a tuning knob

This is the fact that reframes the parameter, and it is verifiable in three lines:

```
max_anchors=16   anchors=17    committed endpoint present: False
max_anchors=32   anchors=33    committed endpoint present: False
max_anchors=64   anchors=65    committed endpoint present: False
max_anchors=128  anchors=129   committed endpoint present: False
uncapped         anchors=35    committed endpoint present: True
```

`_anchor_points` (`src/reblock/methods/arterial.py:53`) takes arc-length samples on the capped
branch and `return sorted(pts)` **early**, before the vertex loop. Its own docstring says
vertices-as-anchors is what makes "committed-segment endpoints always anchors → continuations come
for free."

So the cap does not thin the anchor set — it **replaces one family with another**. At `cap=128` you
get 129 anchors where uncapped gives 35, nearly four times as many, and the committed endpoint is
*still* gone. There is **no setting that preserves continuations**, and "set it larger to be safe"
buys only cost. Any future discussion of tuning this parameter should start here.

---

## Block scale: the cap is dominated — do not use it there

12 blocks, paired bootstrap against uncapped, same tier-2 selector in every arm, matched
displacement:

| cap | Δburden | 95% CI | Δperm | 95% CI | speed |
|---|---|---|---|---|---|
| 32 | −0.0067 | [−0.0351, +0.0240] | −0.0195 | [−0.0538, +0.0214] | 1.25× |
| 64 | −0.0358 | [−0.0810, −0.0009] | −0.0287 | [−0.0897, +0.0185] | 1.00× |
| 128 | −0.0063 | [−0.0259, +0.0113] | +0.0023 | [−0.0207, +0.0238] | **0.59×** |
| 256 | −0.0281 | [−0.0866, +0.0108] | −0.0220 | [−0.0700, +0.0150] | **0.24×** |

Seven of eight CIs span zero. `cap=64`'s burden CI clears zero by 0.0009 — across eight comparisons
that is what chance produces, and its own permeability CI spans zero, so the independent check does
not corroborate it. **No quality effect.**

But every useful setting is *slower*: uncapped needs only 1,272 candidates on these blocks while
`cap=256` enumerates 34,688 — 27× more work for nothing. Neutral on quality, worse on speed, so the
cap is **Pareto-dominated at block scale** and should never be offered there.

This is the unusual shape worth remembering: the same parameter is dominated at one scale and
strongly Pareto-optimal at another. It is not a setting with a good default — it is a setting whose
correct value is a function of the input size.

---

## Why it wins at region scale

Not the reason I first guessed. The structure of the output networks:

| arm | segs | total_m | mean_m | median_m | junctions | deg ≥ 3 |
|---|---|---|---|---|---|---|
| 128 | 57 | 3,779 | 66.3 | 11.7 | 64 | 6 |
| 256 | 63 | 3,643 | 57.8 | 12.4 | 70 | 5 |
| uncapped | 29 | 3,635 | 125.3 | 113.8 | 34 | 9 |

Uncapped builds **fewer, longer, more-branching** roads — it has *more* true intersections (9
against 5–6), which is what having continuations available should produce. The caps build many
shorter segments across more junctions, at the same total length.

The mechanism is **stratification, not connectivity**. Arc-length anchors are evenly spaced along
the network by construction; vertex anchors pile up wherever the parcel-boundary graph is
geometrically dense. With the shortlist budget fixed at 512, the capped arm samples **1.7%** of a
well-spread candidate set (30,353 at the last step) while uncapped samples **0.04%** of a clustered
one (1,180,388). Even coverage of the block beats more branching, on the metric nobody optimizes.

The corollary is that the win is partly an artifact of the *fixed* shortlist budget interacting with
candidate-set size — which is a testable prediction: raising the shortlist for the uncapped arm
should close some of the gap, at proportionally more cost. Untested.

---

## Cost: the handoff's diagnosis of the growth was right

Uncapped candidates grow 2.52× across 15 region steps and 3.27× across 8 block steps, and every cap
flattens it (block scale: 1.34× / 1.19× / 1.05× / 1.02×). The residual growth is **linear, not
quadratic** — the capped branch yields `max_anchors + n_lines` anchors, so it gains about one per
committed road. Committed-road vertices really are what drives the growth, as diagnosed.

---

## Retired: the "unexplained" 66-minute observation

The handoff carries this as an open puzzle: "the 66-minute `max_anchors=24` observation is still
unexplained — 276 candidates at 242 ms is ~67 s per step. Something else was slow in that run and
nobody knows what."

There is nothing to explain. `scratchpad/perf/anchors.log` is 77 bytes: the region line and the
column header, and **zero data rows**. The harness prints one row *after* each `propose` returns, so
no row means the first `propose` never returned. 66 minutes is wall-clock-until-killed with
`ma=24` still running — not a completed timing. It also drove the **exact** greedy, which is the
cost tier 2 exists to remove.

---

## Caveats, stated plainly

- **The region result is n=1 block.** Two independent caps agreeing in direction and magnitude on
  permeability, plus a large structural difference at equal length, is real evidence — but it is one
  region block, and no interval can be put on it. Replication is the obvious next move; the capped
  arms cost 10–13 min each, so the expense is the 80-minute uncapped baseline per block, not the
  caps.
- **`cap=128` and `cap=256` are not separated** by this data. Both beat uncapped; which is better is
  inside the noise.
- **Only one displacement budget (0.10) and one shortlist (512)** were tested, and the stratification
  mechanism predicts the shortlist interacts with the result.
- The block-scale null was measured with the *same* tier-2 selector in every arm, so it says nothing
  about capping under the exact greedy — which is moot, since the exact greedy is what tier 2
  replaced at this scale.

---

## What this changes

1. `max_anchors` moves from "unevaluated" to **the region-scale lever**, and from a suspected
   quality cost to a measured quality *gain* on permeability.
2. It must be selected by input scale, not set globally. A single default is wrong at one of the two
   scales, and at block scale the wrong default costs up to 4× wall clock for nothing.
3. The productionization decision (handoff §4) now has a second dimension: an `ArterialEngine`
   choice, and an anchor policy that is scale-dependent. Both resolve upstream, in config.
4. Region-scale access is now **9.7 minutes**, against 79.6 for tier 2 alone and "not finished after
   11.6 h" before it. Combined with tier 2's 40×, that is roughly **330×** on the original problem.
