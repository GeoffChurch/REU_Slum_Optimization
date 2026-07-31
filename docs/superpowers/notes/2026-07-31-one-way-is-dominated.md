# One-way streets: CLOSED, negative — dominated at equal legal footprint (2026-07-31)

> **CORRECTED same day.** The first version of this note called one-way "dominated by simply
> building narrower two-way roads" and recommended sweeping `road_width_m`. Both claims rested on a
> `two_way_3.5` arm that **cannot physically exist**, and both are retracted — see [Retraction:
> continuous width was fiction](#retraction-continuous-width-was-fiction) below. The verdict on
> one-way survives on legal widths and is restated there; the road-width recommendation does not.

The one-way idea is finished. Both halves were finally priced together and the answer is that the
direction constraint is **pure cost**: at the same buildable footprint, spending width on two slow
directions beats spending it on one fast one.

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
| ~~two_way_3.5~~ | ~~0.8326~~ | ~~0.1185~~ | ~~0.0401~~ | **RETRACTED — unbuildable, see below** |

**Narrowing works; orienting does not.** One-way saves 0.0757 of displacement (14/14) and costs
0.0787 of permeability (14/14) — a ~1:1 trade that vanishes at a matched budget.

**The decomposition is the finding.** `one_way_6.0` (oriented, *not* narrowed) costs only 0.0197 of
permeability. So three quarters of one-way's permeability loss comes from the NARROWING, not the
direction — and narrowing needs no orientation at all.

**The (retracted) decisive control.** `two_way_3.5` has the identical paved footprint to
`one_way_3.5`, and two-way won by −0.0149, p=0.030 — but a 3.5 m two-way road cannot be built. The
legal version of this test is `one_way_6.0` vs `two_way_6.0`, which reaches the same conclusion at
−0.0165, p=4.9e-07. See the retraction below.

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

## ~~The sleeper result: 6 m may simply be too wide~~ — RETRACTED

~~`two_way_3.5` beats the flagship at matched permeability, so sweep `road_width_m`.~~ See below.

## Retraction: continuous width was fiction

Owner's objection, and it is correct: **width is continuous in the model but real roads are a margin
plus an integer number of lanes.** A 3.5 m road has 2.5 m of usable width — one car — and the affine
model scored it as two 1.25 m lanes carrying traffic simultaneously at half speed. `two_way_3.5` is
not a narrow road; it is an impossible one, and it was the arm that produced both the "decisive"
−0.0149 and the entire sleeper result.

The integer reading is not an arbitrary overlay — it reproduces the shipped defaults exactly:

    W = margin + k * LANE_M,   k >= 1;  two-way needs k >= 2
    LANE_M = 2.5, margin = 1.0  ->  k=1 gives 3.5 m  (= one_way_width(6.0))
                                    k=2 gives 6.0 m  (= DEFAULT_ROAD_WIDTH_M)

So the *parameters* were already integer-lane consistent; only the continuum between them was
fictional. Two consequences:

1. **Two-way has a floor of 6.0 m.** There is nothing narrower to sweep, so the sleeper result is
   dead, not merely unproven. The recommendation to sweep `road_width_m` is withdrawn.
2. **One-way's real advantage is a lower floor**, not a cheaper equal-function road — it can build
   where a two-way road will not fit inside the budget. That is a Lens A question, and the original
   probe only asked Lens B at a generous budget, so it could not have seen it.

### Re-run on legal widths only (`scratchpad/width/integer_lanes.py`, n=36)

| | perm | disp | permA @ D=0.02 | permA @ D=0.10 | dispB |
|---|---|---|---|---|---|
| two_way_6.0 (1 lane each way) | 0.8815 | 0.1961 | 0.4142 | 0.7591 | 0.0751 |
| one_way_3.5 (1 lane, one way) | 0.7736 | 0.1243 | 0.2362 | 0.7058 | 0.0792 |
| one_way_6.0 (2 lanes, one way) | 0.8690 | 0.1961 | 0.2026 | 0.6455 | 0.1096 |

* **Equal legal footprint (6.0 m both, displacement identical to 0.00e+00):** two-way wins,
  −0.0165, better on 8/36, **p=4.9e-07**. This is the legal replacement for the retracted
  `two_way_3.5` control and it reaches the same conclusion with correct physics.
* **Lens A, every budget including the tightest:** one_way_3.5 worse — D=0.02 p=4.4e-06, D=0.04
  p=4.7e-05, D=0.06 p=1.3e-06, D=0.10 p=7.1e-04. The lower floor is real (one-way fits +16 to
  +118 m of road under the same budget) and it still does not compensate for losing a direction.
* **Lens B:** one_way_3.5 is a wash — 13/35, median +0.0066, **p=0.081**, not significant.

**So the verdict survives, on better evidence.** What changes is the reason: one-way is not beaten
by a narrower two-way road (there is no such thing), it is beaten by a two-way road of the same
buildable width.

### Also learned: the earlier Lens B result was sample-noise

The original probe put one_way_3.5 vs two_way_6.0 on Lens B at p=0.81 (n=14); at n=36 the same
comparison is p=0.081 with the opposite sign. Both are "no significant difference", but the
instability at n=14 is a caution about every small-n paired result in this line of work. Lens A was
stable across n=16 and n=36; Lens B was not.

### Open: LANE_M is a domain parameter nobody has pinned down

2.5 m is the value that makes the shipped defaults consistent, and it is narrow — service access
(fire, ambulance, refuse) is usually the binding constraint in these settlements and is commonly
cited at 3.0–4.0 m clear. **At LANE_M = 3.0 the shipped 6.0 m default would itself be an illegal
two-way road** (floor 7.0 m). That is worth settling with a domain source before any width work,
because it moves the floor everything else is measured against.

## What was kept and what was deleted

Kept: `reblock/orient.py` (`strong_orientation`, `one_way_width`, `bridge_fraction`), with 7 tests,
two of them fault-injected. `bridge_fraction` is a real structural measure — `cycle_native`'s claim
to be the only bridgeless method is stated in its docstring and now has an instrument.

Deleted: the `CycleNativeReblocker.oneway` flag. It was a measured loser that nothing selected, and
an option nobody sets is exactly the wart the no-legacy rule bans. The probe calls
`strong_orientation` directly, so the result stays reproducible without it.
