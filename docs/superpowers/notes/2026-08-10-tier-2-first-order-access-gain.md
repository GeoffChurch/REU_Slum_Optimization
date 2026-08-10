# Tier 2: the first-order access gain works, and it exposes something worse

**Date:** 2026-08-10
**Scripts:** `scratchpad/perf/{snap_vs_peel,rank_throughput,first_order_rank,rank_decompose,shortlist_greedy,shortlist_ab,control_check,region_shortlist}.py`

Tier 2 is the backlog's "build this first" fix for the access objective's region-scale cost:
replace a full BFS peel per candidate with a local estimate, shortlist on it, and score only the
survivors exactly. It works — but establishing that required discarding the acceptance test it was
specified with, and the reason it had to be discarded is a more important result than the speedup.

## 1. The backlog blamed the wrong term

The entry recorded `_snap` as the likely dominant per-candidate cost, inferred from a
`max_anchors=24` run that ran far longer than the peel count alone predicted. Measured directly on
the real 11,006-parcel region block, at step 0:

| per candidate | mean | median | share |
|---|---|---|---|
| `_snap` (Dijkstra over the boundary graph) | 28.3 ms | 28.1 ms | **12%** |
| peel (`_score`, full BFS over 11,006 parcels) | 214.2 ms | 217.8 ms | **88%** |

The peel dominates, so tier 2 aims at the right term. The `max_anchors` inference was wrong, and
**why that run was slow is still unexplained** — 276 candidates at 242 ms is ~67 s serial per step,
nowhere near the 66 minutes observed. Left open deliberately; it is not on tier 2's path.

## 2. The real problem was never per-candidate cost

Step 0 enumerates **468,968 candidates** — 961 street vertices, so ~C(961,2). At 28.3 ms of
snapping each, 15 steps costs **3.5 h with the peel made entirely free**. So the shortlist cannot be
formed after snapping. It has to rank the *unsnapped* chord, and the exact pass snaps only the
survivors.

The backlog anticipated this ("the shortlist has to be formed BEFORE snapping") but as a consequence
of snapping being dominant. It follows from snapping being *subdominant* too, and would have been
missed by fixing only what the timing pointed at.

## 3. The estimate is good; the cost proxy was the weak half

`rank_decompose.py` scores every candidate both ways in the same step and correlates the pieces
separately (Spearman, median over 24 steps on 6 blocks):

| | ρ |
|---|---|
| **numerator** — first-order estimate ~ exact benefit | **+0.937** |
| denominator — chord length ~ exact displacement | +0.649 |
| denominator — **buildings in corridor** ~ exact displacement | **+0.922** |
| denominator — snapped length ~ exact displacement | +0.732 |
| geometry — chord length ~ snapped length | +0.975 |
| end to end — est / chord length ~ exact gain | +0.785 |
| end to end — **est / buildings in corridor** ~ exact gain | **+0.829** |
| ceiling — *exact* benefit / chord length ~ exact gain | +0.875 |

Three things follow. The estimate itself is sound, so the ripple it ignores is not what limits it.
The denominator was the weak half, and swapping chord length for a second bulk `dwithin` over the
building tree closes most of the gap to the ceiling at identical cost. And ranking pre-snap is safe:
chord and snapped length correlate at +0.975, so snapping barely reorders anything.

Two corrections to the specification, both load-bearing:

- The weights are **`d² − 1`, not `(d−1)²`.** The greedy optimizes `budget.access_burden` = Σd²;
  the *reported* metric is `eval.access_burden.burden` = Σ(d−1)²/n. Two functions, similar names,
  different linearizations. The estimate must linearize the greedy's own objective.
- **12.1% of live candidates have zero exact displacement**, hence infinite gain — the "take the
  free navigability first" rule. A ranking that floors the denominator at 1 cannot express that
  class. Not yet addressed; see open questions.

## 4. The per-step acceptance test is unanswerable, and that is the finding

The natural gate — "is the exact winner in the top k?" — fails badly. Over 80 steps on 10 blocks,
ranking by gain per metre at r=0.5:

| | median | p90 | max | top-8 | top-32 | top-128 | top-512 |
|---|---|---|---|---|---|---|---|
| rank of an optimal candidate | 384 | 1,822 | 3,979 | 10% | 16% | 32% | 57% |

with a median of 3,172 candidates per step. A shortlist of 128 preserves the exact greedy's move
about a third of the time.

The rank scored here is the *shallowest shortlist containing any candidate that achieves the winning
gain*, not the rank of the one chord `_best_candidate` happened to return — thousands of chords snap
to the same road, and this greedy has exact ties between genuinely different roads, so scoring a
single representative counts its own equals as competitors. A first version did exactly that and
reported a median of 594; the correction matters and still does not rescue the gate.

A +0.937 ranking that cannot find the argmax is a contradiction only until you add the third
measurement, which was already on record: **the exact greedy's own argmax flips under a 1e-10
perturbation of the gains**, moving burden reduction by up to 13 points
(`2026-08-09-greedy-arterial-is-near-tie-sensitive.md`). The gains are densely near-tied. "The exact
winner" is one arbitrary draw, and no approximation can reproduce an arbitrary draw.

So the gate was asking the shortlist to meet a standard the method does not meet against itself. The
answerable question is whether the *outcome* matches.

## 5. End to end: exactness buys nothing measurable

`shortlist_ab.py`, 8 blocks, `max_roads=8`, D=0.10 prefix — the same blocks, arms and metrics as the
tie-sensitivity run, so the two tables read side by side. The `shortlist=0` control arm re-states
`_greedy_arterials`' step loop, so it was checked against the shipped function rather than assumed
equal to it: identical road geometry WKT-for-WKT on 3 blocks at `max_roads=5` (`control_check.py`).
Every deviation below is therefore the shortlist's.

| arm | burden_red | perm | road_m | secs | speedup | median \|Δburden\| | max \|Δ\| | beat exact |
|---|---|---|---|---|---|---|---|---|
| exact | 0.7414 | 0.7305 | 201.9 | 13.1 | 1.0× | — | — | — |
| k=512 | 0.7418 | 0.7410 | 221.5 | 4.6 | 2.8× | 0.032 | 0.246 | 2/8 |
| k=128 | 0.7451 | 0.7741 | 181.6 | 2.8 | 4.7× | 0.025 | 0.161 | **6/8** |
| k=32 | 0.7571 | 0.7547 | 178.4 | 1.8 | 7.3× | 0.047 | 0.322 | 3/8 |

**Every shortlist arm's median burden reduction and median permeability match or exceed the
exhaustive search's.** Per-block deviations are large (up to 0.25) and exceed the 1e-10 tie band —
a shortlist is a far bigger perturbation than 1e-10, so that is expected — but they are
*bidirectional*: k=128 lands above exact on 6 of 8 blocks.

An approximation cannot beat an exhaustive search on a step they share. When it does, the exact
argmax was not buying a better final network — the greedy is myopic, and its per-step optimum is
not on the path to a better whole. On this evidence **the exhaustive per-candidate scoring has no
demonstrated outcome value for the access objective**, which is a stronger statement than "the
shortlist is an acceptable approximation".

Read this with the sample size in mind: n=8. The near-identical medians support "no systematic
loss" well; 6/8 does not establish that k=128 genuinely beats exact (binomial p ≈ 0.15). The claim
is the absence of a penalty, not the presence of a gain.

## 6. Region scale

On the real 11,006-parcel / 10,998-building region block, step 0 has 468,968 candidates. Ranking all
of them costs **355 s single-threaded** — versus 468,968 × 242.5 ms ≈ 31.6 h to score them exactly.
That is the tier-2 speedup at region scale: **~320× per step**, against 2.8–7.3× at block scale,
because block peels are cheap and an 11k-parcel peel is not.

`STRtree.query` does release the GIL, but scaling saturates early — the query is memory-bandwidth
bound, not compute bound:

| threads | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| rank all 468,968 | 354.9 s | 120.8 s | **104.3 s** | 134.0 s |
| speedup | 1.00× | 2.94× | **3.40×** | 2.65× |

So 8 threads, not 16, and ~104 s per step. Fifteen steps ≈ **26 min of ranking**, plus a shortlist
tail of 512 × 242.5 ms ≈ 124 s serial (≈ 8 s across the 16-worker fork pool) per step. Against
`depth`'s 1,115 s without this method and 11.6 h of not finishing with it, that is the difference
between impossible and roughly a half-hour surcharge.

### Measured: 79.6 minutes, and the projection above was wrong in an instructive way

A full 15-step run at `shortlist=512` with the ranking threaded at 8: **79.6 min, all 15 roads
placed, 29 road rows, 3,635 m.** Against "not finished after 11.6 h", that is the difference between
impossible and an overnight-free afternoon. But it is more than double the ~35 min the table above
implies, and the gap is the whole point.

**The projection held step 0's candidate count fixed. It is not fixed — it grows 2.52×.**

| step | 1 | 4 | 8 | 12 | 15 |
|---|---|---|---|---|---|
| candidates | 468,968 | 644,652 | 879,773 | 1,027,433 | **1,180,388** |
| secs | 139.5 | 200.5 | 390.5 | 394.9 | 466.9 |

Uncapped `_anchor_points` takes every network vertex, and each committed road is a boundary-graph
path contributing tens of new ones, so the candidate set compounds with every commit and step cost
tracks it (139.5 s → 466.9 s, 3.3×). The threading itself behaved exactly as measured — step 1's
139.5 s is 104 s of ranking at 8 threads plus ~35 s of enumeration, peel and the exact tail — so the
miss is entirely the growth term.

Scored against what actually happened rather than the fixed-count assumption: **12,675,441
candidates** were ranked across the run. Exhaustively scoring that many at 242.5 ms each is 854 h
serial, or **53.4 h on 16 workers**. Tier 2 did it in 1.33 h — a **40× end-to-end speedup at region
scale**, at 377 µs per candidate all-in.

The standing cost is now unambiguous: **candidate enumeration, not candidate scoring.** Tier 2
removed the term the backlog was written about and promoted the one it only flagged in passing.

*(The first attempt at this run was killed at 73 minutes having printed nothing — the fourth long
background run in this repo stopped mid-flight, after C9 at 2/10 and C20 at 2/12 and 5/12. Two
faults were mine: the script reported only totals, so a kill destroyed every minute of it, and it
did not go through `scratchpad/complexity/instrumented.py`, which exists to name the signal's
sender. Both fixed; the rerun logged each step as it landed and finished normally under the
wrapper.)*

## 7. What to do with it

Tier 2 is worth shipping, but the shape matters. It is not a faster way to compute the same answer —
it is a different, equally defensible answer computed much faster, and the honest framing has to say
so.

- **Where it goes.** `arterial_lazy` already occupies this exact seam ("reuses arterial's exact
  scoring machinery unchanged; only changes which candidates get scored each step"), and CELF is
  invalid for this objective because burden reduction is not submodular. Tier 2 is the access
  objective's counterpart to what CELF does for the submodular ones — a sibling engine, not another
  flag on the same one.
- **How it is selected.** `GreedyArterialReblocker` already carries `lazy` + `candidate_policy` +
  `rescore_every`, three fields that jointly pick an engine, and `propose` dispatches on them.
  Adding a fourth makes that worse. An injected `ArterialEngine` (exact / lazy / shortlist),
  resolved once in config, is what the standing directive asks for. That refactor also changes
  `ArterialIdentity`, so it invalidates the derive cache and forces an examples regeneration —
  a real cost that belongs in the decision, not hidden inside it.
- **What it must not do.** It cannot quietly replace the exact path for the published single-block
  examples: the same block moves by up to 0.25 burden reduction depending on the engine. Either the
  examples move with an announced before/after, or the shortlist engine is used only where exact is
  infeasible (region scale) and the block-scale figures stay on exact.

## Open questions

- **The free class.** 12.1% of candidates have zero exact displacement and infinite gain. The cheap
  ranking floors the denominator at one building and so cannot prioritize them the way the exact
  scorer does. Mirroring the `denom <= 0 and raw > 0 → inf` rule is cheap; whether it matters is not
  measured, and at region scale a free class of that size would swamp any shortlist.
- **Candidate enumeration is now the floor — measured, not suspected.** The set grows 2.52× across
  15 steps (469k → 1.18M) and step cost tracks it 3.3×; two thirds of the run's 79.6 min is growth
  the projection did not anticipate. `max_anchors` caps exactly this, and is the obvious next lever
  — with the caveat that it drops per-vertex anchors and so biases toward long chords over short
  local connectors, which for an ACCESS objective is precisely the wrong bias and needs measuring
  rather than assuming. Its earlier dismissal rested on an inference now known to be wrong (§1), so
  it is unevaluated rather than rejected.
- **Why the `max_anchors=24` run took 66 minutes** is still unexplained (§1).
