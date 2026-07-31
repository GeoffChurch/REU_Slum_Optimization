# One-way streets: CLOSED, negative — dominated by simply building narrower two-way roads (2026-07-31)

The one-way idea is finished. Both halves were finally priced together and the answer is that the
direction constraint is **pure cost**: everything one-way appeared to buy is available without
giving up a direction.

## What finally made the question answerable

[2026-07-30-oneway-half-width.md](2026-07-30-oneway-half-width.md) got as far as "the cheap half is
only meaningful as part of the expensive half" — the width discount is a claim that a road can
function one-way, and that claim is only cashable if the scoring enforces one-wayness. Otherwise it
is free money and the cheapest way to collect is one enormous loop. Both halves now land:

* **benefit** — `edge_conductances` gives the permitted direction the full width and the other only
  footpath conductance; `egress_power` scores a directed set as egress **and ingress** halved, so an
  out-tree gains nothing.
* **cost** — a one-way road is narrower, and how much narrower is now *derived*: `one_way_width` =
  `(W + margin)/2` = 3.5 m, not the naive 3.0 m, because both roads pay the margin once.

`reblock/orient.py` supplies the missing primitive: Robbins' theorem (bridgeless ⟺ strongly
orientable), with the street contracted to one super-node so a street-to-street loop reads as the
cycle it is rather than a path of bridges.

## The measurement

`cycle_native` proposes once per block; every arm is that same road set through
`strong_orientation`, so geometry and row granularity are identical and only direction and width
vary. 14 blocks, P\* = 0.60.

| arm | perm | disp | disp at matched P\* | vs 6.0 m two-way |
|---|---|---|---|---|
| two_way_6.0 | 0.8855 | 0.1967 | 0.0588 | — (control) |
| one_way_3.5 | 0.8170 | 0.1185 | 0.0546 | cheaper 5/14, **p=0.81**, +81 m, dominates **0/14** |
| one_way_3.0 | 0.7990 | 0.1043 | 0.0536 | cheaper 6/14, p=0.90, +98 m, dominates 0/14 |
| one_way_6.0 | 0.8713 | 0.1967 | 0.0871 | cheaper 0/14, p=0.0015 |
| **two_way_3.5** | 0.8326 | 0.1185 | **0.0401** | **cheaper 10/14, median −0.0132, p=0.020** |

**Narrowing works; orienting does not.** One-way saves 0.0757 of displacement (14/14) and costs
0.0787 of permeability (14/14) — a ~1:1 trade that vanishes at a matched budget.

**The decomposition is the finding.** `one_way_6.0` (oriented, *not* narrowed) costs only 0.0197 of
permeability. So three quarters of one-way's permeability loss comes from the NARROWING, not the
direction — and narrowing needs no orientation at all.

**The decisive control.** `two_way_3.5` has the identical paved footprint to `one_way_3.5`
(displacement differs by exactly 0.00e+00), so whichever scores higher permeability wins outright.
Two-way wins: one-way is **−0.0149, better on only 5/14, p=0.030**. At equal footprint, two slow
directions beat one fast one. There is no residual advantage for the direction constraint to have.

## Why this is not an artifact

* `one_way_6.0` is the guard on the instrument: orienting at unchanged width must cost permeability
  and save *exactly* zero displacement, and it does (14/14 worse, 0.00e+00 saved). The directed
  solve is charging direction, so the rest of the table means something.
* Orientation coverage is 92.4% of road length against a 95.7% Robbins ceiling. The first attempt
  managed only 39% — a row that crosses several 2-edge-connected components does not orient
  consistently, and refusing it outright discarded 56 points of the discount. Rows are split at
  orientation boundaries now, and the two-way control is split identically so granularity cannot
  favour either arm.
* The emitted orientations are genuinely strongly connected on real blocks: 0/10 blocks strand
  anything, every node reaches the street and returns (`scratchpad/width/orient_valid.py`).
* Orientation *choice* cannot rescue it: the score is (egress + ingress)/2, so reversing an entire
  orientation gives the identical value. A cleverer orientation (real one-way systems use paired
  couplets) is the one unexplored move, but it would have to find 0.0149 — and even at parity
  one-way gains nothing, because its only benefit is a narrowing that two-way roads get for free.
  The sole mechanism that could favour one-way is a score weighting egress above ingress, and a
  residential block's traffic is symmetric over a day.

## The sleeper result: 6 m may simply be too wide

`two_way_3.5` is the only arm that BEATS the flagship at matched permeability — cheaper on 10/14,
median −0.0132, p=0.020, for +41 m of road. It is a Pareto trade rather than dominance (2/14 win on
both axes), but road width was a hardcoded global until [the width
refactor](2026-07-31-width-is-per-road.md) and has never been swept. **Sweeping `road_width_m` is
the obvious next probe**, and it is a question the one-way work only surfaced by accident.

## What was kept and what was deleted

Kept: `reblock/orient.py` (`strong_orientation`, `one_way_width`, `bridge_fraction`), with 7 tests,
two of them fault-injected. `bridge_fraction` is a real structural measure — `cycle_native`'s claim
to be the only bridgeless method is stated in its docstring and now has an instrument.

Deleted: the `CycleNativeReblocker.oneway` flag. It was a measured loser that nothing selected, and
an option nobody sets is exactly the wart the no-legacy rule bans. The probe calls
`strong_orientation` directly, so the result stays reproducible without it.
