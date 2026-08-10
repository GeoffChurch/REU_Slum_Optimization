# The ranking earns its place; tier 3 does not

**Date:** 2026-08-10
**Follows:** `2026-08-10-tier-2-first-order-access-gain.md`
**Scripts:** `scratchpad/perf/{selectors,null_model,null_analyze,rank_decompose,stochastic_restarts}.py`

Three follow-ups to the tier-2 result, chosen because tier 2 changed which questions were worth
asking. Two of them are gates the original cascade did not specify, and one of them kills tier 3.

Everything runs through one seam: `selectors.py` defines a `CandidateSelector` Protocol and the
greedy takes an injected instance, so `ScoreAll` (the shipped exhaustive path), `FirstOrder`,
`RandomSample` and `StochasticFirstOrder` differ in *how they choose* and in nothing else. The
`ScoreAll` arm is verified bit-identical to `_greedy_arterials` on real blocks, so it remains an
honest control.

## 1. The control tier 2 never had: is the ranking better than a coin flip?

Tier 2's case rested on the first-order estimate correlating with the exact benefit at ρ = +0.937.
That was never tested against the obvious null. The candidate gains are densely near-tied — the
argmax flips under a 1e-10 perturbation — and on a near-tied distribution, best-of-k over a *random*
k is close to best-of-everything by order statistics alone, with no geometry involved. If random-k
matched ranked-k, the estimate would be decoration and the real finding would be "this greedy only
ever needed a subsample".

Identical machinery, identical k, identical per-step peel (the random arm pays for depths it ignores
so the timing is not flattered). 8 blocks, 5 seeds per random arm:

| arm | burden_red | perm | secs | vs exact | beats exact |
|---|---|---|---|---|---|
| exact | 0.7414 | 0.7305 | 13.7 | — | — |
| **fo-128** | **0.7451** | 0.7741 | 3.0 | **+0.0037** | 6/8 |
| rand-128 (×5) | 0.6910 | 0.7265 | 2.6 | **−0.0503** | 12/40 |
| **fo-32** | **0.7571** | 0.7547 | 1.9 | **+0.0157** | 3/8 |
| rand-32 (×5) | 0.6154 | 0.6646 | 1.5 | **−0.1260** | 7/40 |

**The ranking earns its place.** `FirstOrder(128)` beats the random *mean* on **8 of 8** blocks and
sits at the 80th percentile of the random spread; random-k loses 0.05 burden reduction where
ranked-k gains 0.004.

The internal consistency check is the more convincing part: the ranking's advantage **widens as k
shrinks** — +0.054 at k=128, +0.142 at k=32. That is what a ranking carrying real signal must do.
The tighter the budget, the more it matters which candidates you spend it on; a decorative ranking
would show no such trend.

Note this does not contradict the earlier finding. Exhaustive scoring buys nothing *over a good
shortlist*; it buys a great deal over a bad one. Both are true and they are about different
comparisons.

## 2. The tier-3 gate: headroom exists and does not cash out

Tier 3 (ALT/landmark depths) improves only the **numerator** — a better estimate of each candidate's
benefit. Its ceiling is therefore a *perfect* numerator over the best cheap denominator, which is
directly measurable by substituting the exact benefit into the ranking:

| ranking | ρ vs exact gain |
|---|---|
| est / buildings in corridor (what tier 2 ships) | +0.829 |
| **EXACT benefit / buildings in corridor** (tier 3's ceiling) | **+0.919** |
| est / EXACT displacement (the reverse control) | +0.846 |

So a perfect numerator buys **+0.090** and a perfect denominator only **+0.017**. After the
denominator fix, the numerator *is* the binding term for ranking fidelity — which reads like a green
light for tier 3, and is why the gate is worth stating carefully.

**It is a red light, because ranking fidelity is not the thing that pays.** The exact greedy already
*is* the perfect ranking — it scores every candidate exactly and takes the true argmax, ρ = 1 by
construction. It scores 0.7414 against fo-128's 0.7451, and loses on 6 of 8 blocks. There is no
outcome headroom above fo-128 to climb toward; the +0.090 of ρ leads somewhere already measured to
be no better.

Tier 3 would buy ranking accuracy over a range where ranking accuracy has been shown not to matter.
**Not worth building** — and, unlike a null result from failure, this one is a null result about
*value*, so it would not be rescued by a better landmark scheme.

The honest limits: n=8 blocks and the fo-128-vs-exact difference (+0.0037) is well inside the
noise. The claim is that no gain is detectable above fo-128, not that fo-128 is superior.

## 3. Restarts: the scatter is a resource, not just a defect

The tier-2 note treated this method's wide, bidirectional outcome scatter as a reliability problem.
It is also unexploited range. If one greedy run is a draw from a wide distribution, R draws and keep
the best is a better estimator of what the method can do — and tier 2 made a draw cheap enough to
take several.

Deterministic top-k cannot do this: every restart returns the same network. `StochasticFirstOrder(k,
pool, seed)` draws k from the top `pool` by score, keeping the ranking's signal (§1 says it pays)
while making runs independent. 8 blocks, k=128, R=4:

| arm | burden_red | perm | secs | vs exact: burden | perm | beats exact |
|---|---|---|---|---|---|---|
| exact | 0.7414 | 0.7305 | 13.2 | — | — | — |
| fo-128 (deterministic) | 0.7451 | 0.7741 | 3.0 | +0.0037 | +0.0435 | 6/8 |
| best-of-1 pool=256 | 0.7418 | 0.7750 | 3.0 | +0.0005 | +0.0444 | 3/8 |
| best-of-4 pool=256 | 0.7789 | 0.7774 | 12.4 | +0.0375 | +0.0469 | 4/8 |
| best-of-1 pool=1024 | 0.7163 | 0.7611 | 3.1 | −0.0251 | +0.0306 | 2/8 |
| **best-of-4 pool=1024** | **0.7867** | 0.7712 | **12.5** | **+0.0453** | +0.0407 | 5/8 |

**One exact run costs the wall clock of 4.6 restarts**, so best-of-4 fits inside exact's budget and
beats it on burden by +0.045 *and* on permeability by +0.041. Cost-matched, it dominates.

Two reasons to believe it rather than treat it as selection artefact:

- **Permeability is not selected on and rises anyway.** Best-of-4 minus the mean of its own draws is
  +0.087 burden (which is forced — it is the max) and **+0.060 permeability** (which is not). If the
  selection were harvesting noise on burden, an unselected metric would stay flat.
- **The `pool` sweep behaves exactly as the mechanism predicts.** At R=1 the tighter pool wins
  (0.7418 vs 0.7163) because diversity costs per-run quality; at R=4 the wider pool wins (0.7867 vs
  0.7789) because diversity is what best-of-R spends. A spurious effect would not reverse with R in
  the direction the theory requires.

Caveats: n=8, single displacement budget (D=0.10), and `pool` was swept over two values, not tuned.
The comparison is also block-scale — at region scale the ranking dominates a restart's cost, so R
restarts cost nearly R× rather than fitting inside exact's budget (exact is not runnable there at
all, so the relevant comparison is different).

## The through-line

The three results are one argument. A greedy whose per-step argmax is arbitrary to within 1e-10
cannot be improved by computing that argmax more exactly — which is why exhaustive scoring buys
nothing (tier-2 note §5) and why tier 3 is not worth building (§2). What it *can* be improved by is
choosing a good candidate pool (§1: rankings pay, and pay more the tighter the budget) and by
sampling the outcome distribution instead of taking one draw from it (§3).

Put the other way: effort spent on **fidelity to a single step** is wasted, and effort spent on
**coverage of the space of runs** is not. The original cascade — tiers 1, 2, 3 — is entirely a
fidelity ladder, and only its cheapest rung was ever needed.

## Where this leaves the cascade

- **Tier 2 — ship.** Validated against the null, ~5× at block scale and ~320× per step at region
  scale, no outcome penalty.
- **Tier 3 (ALT/landmark) — drop.** §2. Improves a quantity measured to be outcome-irrelevant over
  the range that matters. Recorded here rather than left on the list, because the idea is seductive
  and would otherwise be re-proposed: it is genuinely the more principled approximation, and that is
  not the same as being worth having.
- **Tier 1 (uniform-density raster proxy) — still open, and now differently motivated.** It was
  specified as a cheaper *ranking*; §1 says rankings pay, and §2 says extra ranking fidelity does
  not. So tier 1's value is not accuracy but whether it can cut the cost of ranking 469k candidates
  further. That is a throughput question, and it now competes with capping candidate ENUMERATION —
  the cost tier 2 does not touch and which grows quadratically across steps.
- **Stochastic restarts — the new item, and the one with the best measured return.** Cost-matched it
  dominates exact on both reported metrics. Before it could ship it needs: a wider block sample, a
  `pool`/R sweep rather than two points, confirmation across displacement budgets (only D=0.10 was
  tested), and a decision about what a restart selects on when a method reports two metrics — burden
  is the greedy's objective, but permeability is co-reported and nothing currently arbitrates.
- **The free class stays open** (12.1% of candidates have zero exact displacement and infinite gain;
  the cheap denominator floors at one building and cannot express it). Untouched by any of this.
