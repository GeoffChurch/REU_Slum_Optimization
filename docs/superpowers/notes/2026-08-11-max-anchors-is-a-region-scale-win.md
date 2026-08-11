# `max_anchors` is a region-scale COST win, and a block-scale mistake

**Date:** 2026-08-11
**Branch:** `continuum-permeability`
**Harnesses:** `scripts/perf/anchor_cap.py` (block), `scripts/perf/region_anchor_cap.py` (region)
**Data:** `scripts/perf/anchor_cap.json`, `scripts/perf/region_anchor_cap.json`

The handoff left `max_anchors` "unevaluated, not rejected", expecting it to *cost* access quality:
it drops per-vertex anchors and biases toward long chords, "which for an access objective is
precisely the wrong bias." Measured, that bias does not appear. The cap is **8.2× faster at
roughly comparable quality**, which is what makes region-scale access affordable.

> ## CORRECTION (same day, before this note was acted on)
>
> **The quality half of this note's first version was wrong, and the error is instructive.** It
> claimed the cap was "better on both metrics" on the strength of **+0.0884 permeability**. That
> comparison was not displacement-matched, despite saying it was.
>
> At region scale the 15-road networks displace only **0.0115–0.0193** of buildings, so
> `prefix_to_displacement(..., 0.10)` never truncated anything — its documented behaviour when a
> budget is unreachable is to return *all* roads. Every "matched budget 0.10" region figure below
> is therefore **road-count-matched, not displacement-matched**, and the arms displace very
> different amounts: `cap=128` displaces **68% more** than uncapped (0.0193 vs 0.0115) for 4% more
> road length.
>
> Re-evaluated at budgets all three arms can actually reach:
>
> | budget | arm | roads | road_m | burden_red | perm |
> |---|---|---|---|---|---|
> | 0.005 | 128 | 8 | 1,303 | 0.6294 | 0.3656 |
> | 0.005 | 256 | 26 | 1,217 | 0.6346 | 0.3885 |
> | 0.005 | **uncapped** | 10 | 1,788 | **0.7137** | **0.4250** |
> | 0.010 | 128 | 13 | 2,166 | 0.7098 | 0.4970 |
> | 0.010 | 256 | 34 | 2,353 | 0.7936 | **0.5766** |
> | 0.010 | **uncapped** | 19 | 2,948 | **0.8260** | 0.5012 |
>
> At equal displacement uncapped wins outright at 0.005 and wins on burden at 0.010; only
> `cap=256`'s permeability at 0.010 beats it. **The permeability win was displacement the capped
> arm spent and was not charged for.**
>
> What survives untouched: the **8.2× / 6.1× speedup**, the mode-switch finding, the block-scale
> dominance result, the candidate-growth diagnosis, and the retirement of the 66-minute
> observation. What does not: "better on both metrics".
>
> A coherent mechanism sits in the road lengths — uncapped needs **more metres for the same
> displacement** (1,788 m vs 1,303 m at 0.005), i.e. its roads run through emptier corridors.
> Vertex anchors sit on parcel-boundary-graph vertices, which are the gaps *between* buildings;
> arc-length samples land anywhere, including mid-cluster.
>
> **Lesson worth keeping:** a budget-matching helper that silently degrades to "no truncation" when
> the budget is unreachable will not error, and the resulting comparison looks matched. Check that
> the budget actually binds before claiming it did.

---

## The headline (road-count-matched — see the correction above)

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

**Why the permeability number looked like it mattered — and why that reasoning failed.** Burden is
what the greedy optimizes; permeability is co-reported and nothing selects on it, so a metric that
moves without being selected on normally indicates a real structural difference rather than
tie-break scatter. That signature was pre-registered in the harness docstring before the run, and
it fired: permeability moved +0.09 under two independent caps.

The inference was still wrong, because it assumed the arms were otherwise comparable. They were
not — the capped arms spent 68% more displacement, which lifts *both* metrics at once. A
pre-registered signature protects against choosing your evidence after the fact; it does not
protect against an uncontrolled variable. See the correction above.

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

## How the networks actually differ

The structural difference is real and survives the correction — it is only its *interpretation* as
a quality win that does not:

| arm | segs | total_m | mean_m | median_m | junctions | deg ≥ 3 |
|---|---|---|---|---|---|---|
| 128 | 57 | 3,779 | 66.3 | 11.7 | 64 | 6 |
| 256 | 63 | 3,643 | 57.8 | 12.4 | 70 | 5 |
| uncapped | 29 | 3,635 | 125.3 | 113.8 | 34 | 9 |

Uncapped builds **fewer, longer, more-branching** roads — it has *more* true intersections (9
against 5–6), which is what having continuations available should produce. The caps build many
shorter segments across more junctions, at the same total length.

Arc-length anchors are evenly spaced along the network by construction; vertex anchors pile up
wherever the parcel-boundary graph is geometrically dense. That is a genuine difference in what the
two families propose, and it shows up twice:

* **Spatially** — the capped networks spread more evenly, the uncapped ones concentrate.
* **In displacement per metre** — uncapped needs **1,788 m** to displace 0.005 where `cap=128`
  needs **1,303 m**. Boundary-graph vertices *are* the gaps between buildings, so vertex-anchored
  roads thread through empty space; arc-length samples land wherever the arc length falls,
  including through building clusters.

The second point is the whole correction. Spreading evenly means displacing more, and displacement
buys burden and permeability. Once displacement is charged for, the apparent quality gain is mostly
gone. **This is a cost win with a structural side-effect, not a quality win.**

Two things remain genuinely open. The **shortlist confound**: at a fixed 512, the capped arm scores
1.7% of its candidates and uncapped 0.04%, so the arms differ in sampled fraction as well as anchor
family (`region_shortlist_confound.py` climbs the uncapped ladder to separate them). And whether
the even-spread family is *preferable* at matched displacement is now an open question rather than a
settled one — at 0.010 `cap=256` did post the best permeability of the three.

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

- **The region result is n=1 block**, and the matched-displacement re-analysis is n=1 as well.
  Replication across six regions of 3.4k–12k parcels is running (`region_cap_replicate.py`), which
  also gives a **size gradient** — the mechanism predicts any anchor-family effect grows with
  region size and vanishes toward block scale, where it measured ~0.
- **`cap=128` and `cap=256` are not separated.** At matched displacement they do not even agree:
  256 posted the best permeability at 0.010 and 128 the worst of the three.
- **The shortlist confound is untested.** At a fixed 512 the arms differ in sampled *fraction*
  (1.7% vs 0.04%) as well as anchor family. `region_shortlist_confound.py` holds the family at
  uncapped and climbs the shortlist to separate them; if the gap closes, the honest lever is the
  shortlist rather than the cap.
- **Matched-displacement budgets must be checked for reachability.** 0.05–0.20 are all unreachable
  at region scale with 8–15 roads; the reachable band is roughly 0.005–0.019. Absolute budgets
  carried over from block scale silently do nothing here.
- The block-scale null was measured with the *same* tier-2 selector in every arm, so it says nothing
  about capping under the exact greedy — which is moot, since the exact greedy is what tier 2
  replaced at this scale.

---

## What this changes

1. `max_anchors` moves from "unevaluated" to **the region-scale affordability lever** — a measured
   **8.2× cost win at roughly comparable quality**. It is *not* a quality win; the first version of
   this note said so and was wrong.
2. It must be selected by input scale, not set globally. A single default is wrong at one of the two
   scales, and at block scale the wrong default costs up to 4× wall clock for nothing.
3. The productionization decision (handoff §4) now has a second dimension: an `ArterialEngine`
   choice, and an anchor policy that is scale-dependent. Both resolve upstream, in config.
4. Region-scale access is now **9.7 minutes**, against 79.6 for tier 2 alone and "not finished after
   11.6 h" before it. Combined with tier 2's 40×, that is roughly **330×** on the original problem.
   This is the claim that carries the work, and it is untouched by the correction.
5. **Method note for this repo:** `prefix_to_displacement` returns all roads when the budget is
   unreachable. Any comparison claiming to be displacement-matched should assert the budget binds —
   otherwise it silently becomes road-count-matched and the arms are free to spend different
   amounts of the thing that buys the metrics.
