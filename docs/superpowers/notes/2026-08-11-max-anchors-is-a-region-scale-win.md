# `max_anchors`: a 7.6× region-scale cost win, a block-scale mistake, and three wrong turns

**Date:** 2026-08-11 (rewritten same day; supersedes three earlier versions of this note)
**Branch:** `continuum-permeability`
**Harnesses:** `scripts/perf/{anchor_cap,region_cap_replicate,region_cap_matched,region_shortlist_confound}.py`
**Analysis:** `pixi run python -m scripts.perf.region_cap_report`

The handoff left `max_anchors` "unevaluated, not rejected", expecting it to *cost* access quality:
it drops per-vertex anchors and biases toward long chords, "which for an access objective is
precisely the wrong bias." That bias does not appear. What the cap actually buys is **speed**, and
the quality question is **unresolvable at the n this work can afford** — which is itself the useful
finding, because it says the decision should be made on cost.

---

## The claim

Six independent region blocks (3.4k–12k parcels), tier-2 shortlist 512, 8 roads:

| | cap=128 | cap=256 |
|---|---|---|
| speedup vs uncapped | **7.6× median** (2.5–12.2×), 6/6 regions | **5.5× median** (1.4–8.7×), 6/6 |
| quality at matched displacement | no detectable difference | no detectable difference |

Every quality interval against uncapped spans zero; win-counts are 2/6–3/6 with sign-test p ≥ 0.69.
Region-scale access goes from "not finished after 11.6 h" to **~10 minutes** — roughly **330×**
combined with tier 2, and that is the number carrying this work.

**"No detectable difference" means exactly that, not "no difference."** The between-region sd of
the burden delta is **0.0984**, and this same greedy moves by up to **0.1356** under a 1e-10
perturbation of its own gains (2026-08-09 tie-sensitivity note). The scatter between regions is the
same order as the method's own arbitrariness. n=6 cannot resolve a quality effect against that, and
no affordable n would: each uncapped region costs 30–53 minutes.

---

## Three wrong turns, and why each happened

Recorded because the *pattern* is the transferable part, and because two of them were published
before being caught.

### 1. "Better on both metrics" — the budget never bound

The first version headlined **+0.0884 permeability** at a "matched displacement budget of 0.10."
Region networks displace **0.0115–0.0193**. `prefix_to_displacement` returns *all* roads when a
budget is unreachable — documented, silent, no error — so nothing was ever truncated. The
comparison was road-count-matched, and the capped arm had quietly spent **68% more displacement**,
which buys burden and permeability alike.

*Transferable:* a budget-matching helper that degrades to "no truncation" rather than raising will
produce a comparison that looks matched and is not. **Assert the budget binds.** Absolute budgets
carried from block scale (0.05–0.20) are all unreachable at region scale, where the band is
0.005–0.02 and differs per region — which is why matching is now computed *from the arms* in
`region_cap_matched.py` rather than configured.

### 2. "Costs quality" — a biased interim sample

The correction over-shot. At n=3 matched, capping looked *worse* (perm negative at all four budget
fractions), and that was reported. It does not survive n=6.

The cause is mechanical and worth remembering: the replication runs regions **ascending by size**,
because uncapped cost grows steeply and cheapest-first makes an interrupted run maximally useful.
That same ordering makes any interim read the **three smallest regions** — and it happened to
include region 3, the one large negative outlier (−0.16 burden). One outlier in a biased
sub-sample.

*Transferable:* an ordering chosen for kill-resilience is not a random sample. **Interim results
from a deliberately-ordered run should not be given a direction.**

### 3. "The stratification mechanism" — an n=1 story

Also claimed: capped roads displace more because arc-length samples land mid-cluster while
boundary-graph vertices thread the gaps between buildings. Coherent, and true on region 0. Across
six regions the direction **flips** — uncapped displaces most in regions 5, 4 and 1 and least in 3
and 0. Retracted; the per-region displacement column is now reported so it cannot be assumed again.

**A pre-registered signature did not save turn 1.** The harness declared in advance that
"permeability moving without being selected on indicates real structure", and it fired. Declaring
your evidence in advance guards against choosing it after the fact; it does nothing about an
uncontrolled variable lifting both metrics together.

---

## `max_anchors` is a mode switch, not a tuning knob

The most durable structural finding, verifiable in three lines:

```
max_anchors=16   anchors=17    committed endpoint present: False
max_anchors=128  anchors=129   committed endpoint present: False
uncapped         anchors=35    committed endpoint present: True
```

`_anchor_points` (`src/reblock/methods/arterial.py:53`) takes arc-length samples on the capped
branch and `return sorted(pts)` **early**, before the vertex loop. Its own docstring says
vertices-as-anchors is what makes "committed-segment endpoints always anchors → continuations come
for free."

So the cap does not thin the anchor set — it **replaces one family with another**. At `cap=128` you
get 129 anchors where uncapped gives 35, nearly four times as many, and the committed endpoint is
*still* gone. **No setting preserves continuations**, and "set it larger to be safe" buys only cost.

---

## Block scale: dominated — do not use it there

12 blocks, paired bootstrap, same tier-2 selector in every arm:

| cap | Δburden | 95% CI | Δperm | 95% CI | speed |
|---|---|---|---|---|---|
| 32 | −0.0067 | [−0.0351, +0.0240] | −0.0195 | [−0.0538, +0.0214] | 1.25× |
| 64 | −0.0358 | [−0.0810, −0.0009] | −0.0287 | [−0.0897, +0.0185] | 1.00× |
| 128 | −0.0063 | [−0.0259, +0.0113] | +0.0023 | [−0.0207, +0.0238] | **0.59×** |
| 256 | −0.0281 | [−0.0866, +0.0108] | −0.0220 | [−0.0700, +0.0150] | **0.24×** |

Seven of eight CIs span zero (cap=64's burden clears it by 0.0009 — across eight comparisons that is
chance, and its own permeability CI spans zero). No quality effect, and every useful setting is
*slower*: uncapped needs 1,272 candidates on these blocks while `cap=256` enumerates 34,688.

**Pareto-dominated at block scale.** The same parameter is dominated at one scale and strongly
favourable at another, so its correct value is a function of input size — it resolves upstream in
config, never as a global default.

---

## What the speedup scales with

Not parcels. Regions 5 and 2 have near-identical parcel counts (3,404 / 3,427) and differ **7×** in
uncapped runtime (2.5 vs 18.6 min), because anchors follow street and boundary-graph vertices.

| | Spearman vs speedup | exact permutation p |
|---|---|---|
| candidate count | **+0.829** | 0.058 |
| parcel count | +0.371 | 0.497 |

Candidates is what the cap acts on, so this is the mechanically coherent reading — but at n=6 it is
suggestive, not established. Candidate growth flattens as designed: uncapped 1.58–2.51× across 8
steps, capped 1.05–1.53×.

---

## Frontier: neither cap dominates the other

The comparison that *does* resolve, because both caps share the arc-length anchor family and so shed
the between-family scatter that swamps the cap-vs-uncapped tests:

| cap=128 minus cap=256 | mean | 95% CI |
|---|---|---|
| burden | −0.0341 | [−0.0634, −0.0011] |
| permeability | −0.0459 | [−0.0678, −0.0200] |

`cap=128` is faster in **6/6** regions; `cap=256` is better on both metrics with intervals excluding
zero. **Neither is dominated** — 128 buys speed, 256 buys quality — so under the frontier directive
both stay, selectable, alongside uncapped.

---

## Retired: the "unexplained" 66-minute observation

The handoff carried this as an open puzzle. There is nothing to explain:
`scratchpad/perf/anchors.log` is 77 bytes — the region line and a column header, **zero data rows**.
The harness prints one row *after* each `propose` returns, so the first `propose` never returned. 66
minutes is wall-clock-until-killed, not a timing. It also drove the **exact** greedy, the cost tier 2
exists to remove.

---

## What this changes

1. `max_anchors` is **the region-scale affordability lever**: 7.6× at no detectable quality cost,
   with the honest caveat that n=6 cannot resolve a cost of the size that would matter.
2. It must be selected **by input scale**. Dominated at block scale, favourable at region scale.
3. Productionization (handoff §4) gains a second dimension: an `ArterialEngine` choice *and* a
   scale-dependent anchor policy, both resolved upstream in config.
4. **Method rules for this repo**, each paid for above: assert a displacement budget actually binds
   before calling a comparison matched; do not read a direction off a deliberately-ordered interim;
   persist full road lists so budget questions are re-analysis, not re-runs.

---

## Addendum: the shortlist confound is closed — uncapped is saturated at 512

At a fixed shortlist of 512 the arms differ in two ways, not one: anchor **family**, and the
**fraction** of their own candidates they score exactly — 2.40% for `cap=128` against **0.06%** for
uncapped, a 40× difference. If the fraction were doing the work, the honest lever would be the
shortlist and not the cap, and every comparison above would be unfair to uncapped.

It is not. Holding the anchor family at uncapped on region 0 and climbing the shortlist:

| arm | scored | permeability |
|---|---|---|
| uncapped, shortlist 512 | 0.06% | 0.4536 |
| uncapped, shortlist 1024 | 0.12% | 0.4536 |
| uncapped, shortlist 2048 | 0.23% | 0.4536 |
| `cap=128`, shortlist 512 | 2.40% | 0.5275 |

Quadrupling the shortlist produces a **bit-identical network**. The first-order ranking already
puts the step's winner inside the top 512, so candidates below that rank never change an argmax.
**Uncapped is saturated at 512, the comparison was fair, and the shortlist is not a lever here.**
(The 4096 rung was still running at the time of writing; three identical rungs settle the shape.)

This also retires the "sampling density" half of the stratification story in wrong turn 3: whatever
separates the two anchor families, it is not how much of each family gets scored.

---

## Still open

- **The quality question is not closed, and cannot be closed cheaply.** Resolving a ~0.03 effect
  against sd 0.0984 needs roughly n≈40 regions, i.e. 20–35 hours of uncapped baselines. Whether that
  is worth buying is a judgement call, not a measurement.
- Only one road budget (8, plus 15 on region 0) was swept, and — given saturation — one shortlist is
  now known to be sufficient rather than merely untested.
