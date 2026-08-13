# Pareto dominance across the lineup: the grid is dominated, the looped tree is not, and two budgets were hiding both answers

**Date:** 2026-08-13
**Status:** measured, 7 regions (3 metrics × 2 cities + the pinned flagship block), from the
committed example frontiers. Reproduce with `pixi run python -m scripts.pareto_dominance`
(add `--overlap` for the cross-check in §1).

## Why this was needed

The frontier plot is supposed to make dominance readable — "which method buys more permeability for
less displacement reads straight off it". Nobody had actually read it. The lineup had grown to nine
methods on the strength of each one being interesting, and the owner's frontier rule says an
implemented alternative earns its place by being best at *something*, with dominated variants
deleted. That requires the pairwise answer, and the pairwise answer had never been computed.

## 0. How dominance is tested, and the two artifacts that corrupt it

A dominates B iff at every displacement B samples, A can buy at least as much permeability for no
more displacement: `env_A(d) >= perm_B(d)` for every sampled `d`, where
`env_A(d) = max{perm_A(d') : d' <= d}`. The running max IS the achievable set, because each curve is
a sequence of prefixes of one road set in drainage order.

**Both of the following changed answers in this analysis. Neither is visible in the plot.**

**LOW END — subsampling resolution.** `compare_report` subsamples each method's road list to 21
points, so a method with more roads gets finer resolution near zero. Scoring B's early samples
against an A that has no sample there yet reads A as permeability 0 and manufactures a loss. It
manufactured a **−0.237 "loss"** for `cycle_native` against `euclidean_grid` on
nairobi/density_compactness — a region where `cycle_native`'s curve is *above* the grid's at every
displacement it actually reports. Fix: start at `max(first positive sample of A, of B)`.

**HIGH END — truncation.** A method whose budget truncates it early cannot cover a rival's tail, and
that reads as non-dominance even when the truncation is a config artifact. This was not hypothetical
either: it is §3.

There is no analysis-side fix for the high end. The only fix is to not truncate — which is why the
first run of this analysis produced a *provisional* verdict, and why §1 and §2 report the numbers
from after the budgets were unbound (`e2ede5d`, `794d52e`).

The origin `(0, 0)` is dropped throughout. Every method passes through it, so including it makes the
worst-case margin read as an exact tie for every pair — the first version of this analysis reported
several spurious "dominates with margin +0.0000" for exactly that reason.

## 1. `euclidean_grid` IS dominated — by two independent methods, 6 of 7 regions

Covering the grid's **entire** curve, worst-case margin (the permeability the dominator has to spare
at the grid's best point):

| region | Direct Objective (LP) | Loop Network |
|---|---|---|
| capetown/depth | +0.039 | +0.008 |
| capetown/depth_density | +0.109 | +0.006 |
| capetown/density_compactness | +0.092 | +0.087 |
| nairobi/depth | +0.073 | +0.018 |
| nairobi/depth_density | +0.109 | +0.023 |
| nairobi/density_compactness | +0.130 | +0.066 |
| one-block | −0.025 | −0.057 |

The lone exception is the same for both and is not about the grid: on the pinned block the grid
builds out to 47% displacement while both dominators stop at their configured
`max_displacement: 0.20`. Restricted to where all three actually operate, it is **7 of 7 for both**.

It is not close, and not confined to the curve. At the Lens B operating point — matched benefit,
compare costs — the grid loses on **both** costs in 7 of 7, paying 2.0–7.0× the homes and 1.3–4.0×
the metres for the same permeability:

| region | Grid | Direct Objective (LP) | Loop Network |
|---|---|---|---|
| capetown/depth | 2.6% / 2,295 m | **1.0% / 1,728 m** | 1.3% / 1,643 m |
| capetown/depth_density | 8.8% / 2,068 m | 2.2% / 1,089 m | **3.8% / 899 m** |
| capetown/density_compactness | 9.7% / 3,466 m | 4.9% / 2,026 m | **5.9% / 1,547 m** |
| nairobi/depth | 6.3% / 6,040 m | 0.9% / 1,645 m | **1.7% / 1,512 m** |
| nairobi/depth_density | 9.0% / 8,625 m | 1.3% / 2,169 m | **2.8% / 2,087 m** |
| nairobi/density_compactness | 5.0% / 4,713 m | 1.4% / 1,615 m | **2.0% / 1,165 m** |
| one-block | 14.8% / 270 m | 4.8% / 147 m | **4.4% / 93 m** |

(The grid overshoots P\* because its curve is coarse at `spacing: 250`, which inflates its apparent
Lens B cost. The full-curve test above compares at every displacement and is immune to that. Both
say the same thing.)

**Do not delete it on this evidence alone.** Three things the numbers do not capture:

- It is a **baseline**, like `osm_footpaths` — the conventional-planning yardstick every other
  method is measured against. That is a different job from being a candidate, and the frontier rule
  is about candidates.
- It is **Pareto-optimal on runtime by three orders of magnitude**: 0.5 s on the 11k-parcel flagship
  against 388 s for the LP and 4,549 s for the Loop Network. Over the 1.8M-block ZAF+KEN pool that is
  the difference between an afternoon and a decade, so it is a real axis, not a technicality.
- **`spacing` has never been swept at region scale.** The examples override it to 250 m. Dominance
  is therefore measured at exactly one setting, and the burden-of-proof half of the frontier rule
  says that is not enough to delete.

## 2. `clearance_looped` is NOT dominated by `cycle_native` (the seductive one)

The Loop Network is the better idea on paper — the cycle is the primitive, the output is bridgeless
by construction, it beats the Looped Tree on Lens B. It is the natural thing to propose replacing the
flagship with. It does not dominate it.

Covering the Looped Tree's whole curve: **2 of 7** regions (capetown/depth +0.037; nairobi/depth
+0.000, an exact tie). The Looped Tree wins the other five, by up to **0.223** on
capetown/density_compactness. The reverse holds in **0 of 7**. Both stay.

At Lens B the two genuinely trade rather than rank:

| region | Looped Tree | Loop Network | |
|---|---|---|---|
| capetown/depth | 2.9% / 2,635 m | **1.3% / 1,643 m** | Loop Network on both |
| nairobi/depth | 3.3% / 2,215 m | **1.7% / 1,512 m** | Loop Network on both |
| capetown/density_compactness | **5.7% / 1,301 m** | 5.9% / 1,547 m | Looped Tree on both |
| capetown/depth_density | 5.1% / **874 m** | **3.8%** / 899 m | trade |
| nairobi/depth_density | 3.3% / **1,803 m** | **2.8%** / 2,087 m | trade |
| nairobi/density_compactness | 2.5% / **942 m** | **2.0%** / 1,165 m | trade |
| one-block | 7.2% / **83 m** | **4.4%** / 93 m | trade |

Loop Network wins outright in exactly the two `depth` regions — the same
depth-is-what-decides-whether-loops-pay pattern
[the six-region grid found](2026-07-29-tree-grid-six-regions.md). Elsewhere it buys lower
displacement with more road.

And it is **~90× slower** on the flagship region (4,549 s against 49 s). Note the committed `run.log`
timings understate this: `clearance`/`loop_closure` register cached derivations with
`reblock.derive_graph` and `cycle_native` does not, so a warm `clearance_looped` line can read 0.0 s
against a `cycle_native` that always pays cold.

## 3. Both verdicts above were hidden by budgets, in opposite directions

`cycle_native.py` contained a bare `for _ in range(60)` — its real stopping rule: unconfigurable,
undocumented, untested, and binding in every settlement region (each emitted exactly 120 segments =
60 cycles × 2 legs) at 6.9–16.0% displacement, while its config said `max_displacement: 0.20`. The
published docs rendered that as *"it converges below the shared budget"*. It did not converge; it ran
out of iterations.

Unbinding it moved both verdicts, in opposite directions:

| | truncated | unbound |
|---|---|---|
| Loop Network covers Looped Tree | 1/7 | 2/7 |
| Loop Network covers Grid | 2/7 | **6/7** |

So the truncation had been *understating* the grid's dominance — the cycle method could not reach
the grid's tail, so it read as a non-dominator of a method it in fact covers everywhere. And it had
been marginally overstating the Looped Tree's position. **A truncated curve is not a conservative
estimate; it is an estimate with an unknown sign.**

The access arterials' `max_roads: 15` was the same shape but visible and deliberate — a scope
choice never swept. At 15 the pair terminated at 0.5–2.6% displacement, was graded at its own cap by
Lens A everywhere, and missed P\*=0.60 in 3 of 6 regions. Raised to 60 it reaches P\* in 7 of 7 —
but is **still** short of the Lens A budget in 4 of 6 (terminals 4.3–10.8%), so the lift only partly
achieved its purpose.

## 4. `Frontage`'s command of the low-displacement corner was partly the curve's length

At `max_roads: 15` it looked like the strongest method below ~2% displacement, beating everything but
the LP. Extended to 60 roads it covers `euclidean_grid` in 3 of 7 and `clearance_looped` in 1 of 7,
and at Lens B it buys its lower displacement with **2–4× the road** of every alternative (3,460 m
against the Looped Tree's 942 m on nairobi/density_compactness).

It is still the specialist at the very low end, and that is the regime that matters most in
practice. But the short curve was doing part of the work, and two structural facts should be stated
beside any result: `cost: displacement` **optimises the reported x-axis directly**, and
`SnapToBoundary` places roads along parcel boundaries, i.e. **where displacement is cheapest by
construction**. It converts metres into permeability at an unusually low home-cost and pays in road
length.

## Consequences

- `cycle_native`'s cap is a `max_cycles` field defaulting to 400 — a safety valve, not a stopping
  rule. Measured: regions use 80/118/214/248/251/300 cycles and every one now terminates on
  `max_displacement`.
- Access arterials at `max_roads: 60`, both variants (they are a matched pair differing only in
  `cost`).
- `frontier_xmax: 0.40` on the frontier plot — display only, since terminals now span 0.015 to 0.828.
- `euclidean_grid`'s status is an open decision, not an action: see the backlog entry.
- Nothing else in the lineup is dominated. Next-closest: the LP covers `clearance_looped` in 4 of 7
  and `osm_footpaths` in 4 of 6. Six shipped methods (`peel`, `flow_paths`, `demand_greedy`,
  `resistance_greedy`, and the two directness arterials) appear on **no** committed frontier, so
  under "dominated means measured" nothing can be said about them either way.

## The lesson worth keeping

**A hardcoded budget is a measurement instrument.** `range(60)` was not a performance knob — it
silently decided what the frontier plot showed for one method in every published example, and it
biased two dominance verdicts in opposite directions. Every other method in the lineup already had
its budget as a field; the one that did not is the one whose published number was its own cap.

The corollary is the rule `conf/compare_config.yaml` already states for `resistance_lp` and
`cycle_native` and which should be checked for any new method: **a method's own cap must sit clear
of the budget the lenses grade at**, or the lens compares a terminal network against everyone else's
truncated prefix.

**And: sweep the region with the most headroom, not the most convenient one.** The `max_cycles` cost
was estimated on capetown/depth_density at 1.4× and was 4.8–6.8× everywhere else (the flagship went
666 s → 4,549 s), because that region needed 80 cycles where the others needed 214–300. Same failure
as [the six-region grid note](2026-07-29-tree-grid-six-regions.md) records for method claims, now in
cost estimation.
