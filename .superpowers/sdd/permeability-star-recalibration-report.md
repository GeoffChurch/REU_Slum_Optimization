# P* (matched_permeability) recalibration under the r0-corridor metric — REPORT ONLY

No conf/permeability.yaml change made. No commit made (the temporary widened-PERMEABILITY_LEVELS
edit to `scripts/calibrate_permeability.py` used to run this probe was reverted afterward via
`git checkout` — working tree is clean, matching commit `b7c82d7`). No examples regenerated.

Full probe run: `pixi run python -m scripts.calibrate_permeability` (all 7 regions, natural config
per generator, r0-corridor metric with `g_walk=0.1`/`r0_frac=0.55` from the committed
`conf/permeability.yaml`), with Lens-B candidates widened to (0.3, 0.4, 0.5, 0.6, 0.7, 0.8) for
this run only (the old 0.3–0.5 saturates too fast to resolve a bar). Raw output:
`/tmp/claude-1641171234/-home-gchurchill-src-reblock/27c82570-a74d-47e6-9e87-e53987507f6d/
scratchpad/calib_full.log`.

## Per-region frontier summary (terminal permeability @ terminal displacement)

| region | greedy_arterial_repulsion | clearance_looped | euclidean_grid | osm_footpaths |
|---|---|---|---|---|
| depth/capetown | 0.723 @ 5.7% | 0.842 @ 17.5% | 0.888 @ 11.0% | 0.205 @ 3.5% |
| depth/nairobi | 0.606 @ 5.0% | 0.799 @ 11.9% | 0.877 @ 11.9% | 0.403 @ 1.6% |
| depth_density/capetown | 0.798 @ 16.0% | 0.855 @ 21.7% | 0.787 @ 12.5% | 0.841 @ 18.1% |
| depth_density/nairobi | 0.402 @ 2.3% | 0.846 @ 11.9% | 0.824 @ 11.8% | 0.308 @ 1.3% |
| **density_compactness/capetown** | **0.277 @ 4.0%** | **0.845 @ 18.0%** | **0.639 @ 10.9%** | **0.915 @ 32.2%** |
| density_compactness/nairobi | 0.206 @ 2.0% | 0.289 @ 0.9% | 0.503 @ 11.2% | (no snapshot) |
| method_comparison (flagship) | 0.935 @ 28.4% | 0.991 @ 77.2% | 0.962 @ 42.4% | 0.948 @ 28.4% |

Terminal permeability is now materially higher and wider-ranging than the old metric across the
board (many methods reach 0.8–0.99; `osm_footpaths` in depth/capetown and depth_density/nairobi,
and both weak methods in density_compactness/nairobi, stay genuinely low — the metric is not just
uniformly inflated, it still separates strong fabric from weak).

## Lens B: candidate P* coverage + discrimination (3 synthetic methods x 7 regions = 21 pairs)

| P* | (method,region) pairs reaching it | displacement-cost spread | density_compactness/capetown: clearance / euclidean / osm displacement-cost |
|---|---|---|---|
| 0.3 | 18/21 | 0.0741 | 1.93% / 3.72% / 3.86% |
| 0.4 (OLD) | 18/21 | 0.0823 | 2.70% / 4.79% / 3.86% |
| 0.5 | 17/21 | 0.0999 | 3.76% / 7.26% / 5.24% |
| **0.6 (PROPOSED)** | **16/21** | **0.1021** | **5.90% / 9.90% / 8.94%** |
| 0.7 | 14/21 | 0.1291 | 9.70% / unreached (terminal 63.9%<70%) / 12.06% |
| 0.8 | 10/21 | 0.1841 | 14.22% / unreached / 18.66% |

`greedy_arterial_repulsion` never reaches ANY of 0.3–0.8 in density_compactness/capetown (terminal
27.7%) — correctly, honestly flagged weak at every candidate level, exactly the old P*=0.40
rationale's "honestly missed by the sparse/weak ones" clause.

## Proposed P* = 0.6

At the OLD P*=0.40, every method that reaches it in density_compactness/capetown does so almost
immediately (2.7–4.8% displacement, a ~2pp spread) — confirms the coordinator's concern directly:
0.40 is now reached "too cheaply" to discriminate Lens B (displacement-cost-to-reach-P*) at all.

**0.6** is where the picture changes qualitatively:
- Coverage (16/21) matches the OLD P*=0.40's own cited coverage figure (16/21, per the current
  `conf/permeability.yaml` comment) almost exactly — same selectivity, same "most pairs but not
  all" character the original threshold was calibrated to.
- On density_compactness/capetown specifically, the three reaching methods spread out
  meaningfully: clearance_looped 5.9%, osm_footpaths 8.9%, euclidean_grid 9.9% (near euclidean's
  own terminal displacement of 10.9% — it just barely clears 0.6, an honest "weakest of the
  three" signal) — a ~4pp spread vs 0.40's ~2pp, and ordered sensibly (clearance_looped, this
  repo's strongest internal method, cheapest; euclidean_grid, the weakest synthetic method here,
  priciest).
- Aggregate displacement-cost spread (0.1021) is the first candidate meaningfully above 0.40's
  0.0823, without yet paying 0.7's price (euclidean_grid dropping out ENTIRELY in our priority
  region — 0.7 exceeds its 63.9% terminal, so it stops being comparable there at all).
- `greedy_arterial_repulsion` is excluded almost everywhere at 0.6 (terminal permeability is
  routinely 0.2–0.4 outside the flagship/depth_density regions) — an honest "this method's roads
  aren't doing internal-circulation work" signal, consistent with existing findings about
  `greedy_arterial_repulsion` being the weaker synthetic method (see `arterial-too-slow-on-regions`
  memory note).

**Fallback: P*=0.5** (17/21 coverage, spread 0.0999) is a more conservative alternative if 0.6 is
judged too aggressive — density_compactness/nairobi's euclidean_grid clears 0.5 only by using its
ENTIRE displacement budget (11.17% disp == its own terminal, i.e. barely, not with headroom); at
0.6 it fails outright (terminal 50.3%<60%). Whether that specific case should count as "reached"
or "should honestly miss" is really the swing vote between 0.5 and 0.6 — I lean 0.6 because a
threshold a method can only touch by spending its ENTIRE network isn't meaningfully "reached" in
the spirit of Lens B (cost to reach a bar with some headroom left), but this is a judgment call,
not a hard fact.

## Lens A sanity check: matched_displacement=0.10 still behaves sensibly

Lens A is purely a displacement threshold (unrelated to the permeability scale), so its
*definition* is mechanically unaffected by the conductance change — the question is whether the
permeability values it reports at D=10% are still meaningful. They are:
- Real, non-degenerate spread at D=10% within every region that has ≥2 methods reaching it (e.g.
  density_compactness/capetown: clearance_looped 75.1%, osm_footpaths 68.5%, euclidean_grid
  62.2% — a 12.9pt spread, consistent with the metric-change validation run's measured 11.72pt
  D=10% spread across the same three methods).
- Correctly reports "converged below budget" (not a fabricated D=10% value) for methods whose full
  network falls short of 10% displacement — `greedy_arterial_repulsion` and (region-dependently)
  `osm_footpaths`/`clearance_looped` — an honest signal Lens A's `run_permeability_lenses` already
  surfaces via `at_budget=False`/"converged at X%".
- The probe's own D-candidate comparison still clearly favors 10%: mean cross-method spread
  0.0829 with 6/7 regions populated at D=10%, versus only 1/7 regions having ≥2 methods reach
  D=20/30/40% at all (most methods' terminal displacement sits under 20%). **D=10% remains the
  right Lens-A level; no change indicated.**

## Concern: does a single global P* still work?

Real, and probably sharper than before: terminal permeability for the SAME method now spans nearly
the full [0,1] range across regions (`clearance_looped`: 0.29 in density_compactness/nairobi up to
0.99 in the small `method_comparison` flagship). Any single global P* will therefore be
near-trivial on easy/small regions (method_comparison's methods clear even P*=0.8 by ~15–24%
displacement — plenty of headroom) while being fundamentally unreachable by weak fabric
(density_compactness/nairobi: two of three synthetic methods top out at 0.21–0.29 terminal, below
even P*=0.3). This isn't a bug in the r0-corridor change — the OLD metric had the same structural
issue (a single absolute threshold across heterogeneous regions) — but the NEW metric's wider
dynamic range makes it more visible. I don't have a same-region old-vs-new comparison to quantify
whether it got meaningfully worse, just flagging it as worth the coordinator's attention before
finalizing: a per-region-relative Lens B (e.g. "displacement to reach X% of THIS region's terminal
permeability") would sidestep this, but that is a bigger design change than picking a new global
P*, and out of scope for this report-only step.

## Bottom line

- **Proposed P* = 0.6** (fallback 0.5). Do not bake in yet per instructions.
- Lens A (D=10%) needs no change; confirmed sensible under the new metric.
- Single global P* still "works" in the sense the old one did (majority coverage, honest misses)
  but the region-to-region heterogeneity in reachable ceilings is a standing concern independent
  of this recalibration.
