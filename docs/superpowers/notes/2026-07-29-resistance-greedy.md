# One objective instead of tree-then-loops: built, and it does not dominate

**Date:** 2026-07-29
**Status:** built and measured. `src/reblock/methods/resistance_greedy.py`,
frontier in `scratchpad/ot/resistance_frontier.py`.

## The objective already existed

`specs/2026-07-21-unified-resistance-objective-notes.md` proposed grounding the network at the
street and minimizing each parcel's effective resistance to ground — one quantity capturing access
(electrically far from the street) *and* redundancy (a spur has higher resistance than a loop) — and
called for writing a grounded-resistance scorer.

**That scorer shipped the next day under another name.** `permeability` solves `P = bᵀL⁻¹b` on the
parcel graph with the street eliminated as ground: the collective, contention-aware form of exactly
that quantity. The note predates it — `commute_ratio` was the only resistance proxy at the time,
and has since been retired — so it asks for something that now exists. No new solver was needed.

That also means the note's frontier verdict ("no fusion Pareto-dominates clearance+loops") was
measured in **(external connectivity, ρ) space, both of which were retired the following day**. It
had never been re-tested under the current metric, which is why this was worth building.

## What was built

Route (B) of the note: stochastic greedy. Resistance-reduction is submodular, so sampling
~(N/k)·log(1/ε) candidates per step gives (1−1/e−ε) at O(N log 1/ε) evaluations. Candidate
generation is deliberately **identical to clearance's** — one multi-source Dijkstra from the current
network per round — so the only difference from the shipped method is *which candidate is chosen*:
best permeability-gain-per-metre, versus deepest-parcel-first.

Route (A), the convex program, was not attempted. Total effective resistance is convex in edge
conductances, but the note's blocker stands: *"the ROUNDING is the hard part, because a road is a
combinatorial frontage PATH, not a free edge."*

## Result: better on the median, not dominant

Lens A per block (matched displacement 10%), 10 Cape Town blocks:

| method | permeability at D | road m | displacement | seconds |
|---|---|---|---|---|
| clearance | 0.6638 | 89.6 | 0.0907 | 0.03 |
| clearance_looped | 0.6638 | 89.6 | 0.0907 | 0.38 |
| **resistance_greedy** | **0.6752** | **88.3** | **0.0873** | 7.46 |

Median advantage **+0.0125** permeability, with slightly *less* road and *less* displacement — but
it wins on only **6 of 10** blocks, and costs ~20× the wall clock.

The note's stopping rule is explicit: *"If it does not dominate, stop — clearance+loops stands.
Only if it does, consider route (A)'s convex solve."* **6 of 10 is not domination, so: stop.**

## Two caveats that keep it alive as a direction

**The comparison is degenerate for the flagship.** `clearance` and `clearance_looped` score
identically (0.6638) because at a 10% displacement budget on single blocks, the prefix truncates
every loop connector away. So this really compares resistance-greedy against *plain* clearance, and
the flagship's distinguishing feature never appears.

**The regime where loops matter is untested here.** The six-region grid showed loops are worth
+0.084 and +0.054 permeability on *deep* regions and nothing on compact ones. Deep regions are
exactly where a single objective should have most to gain — and exactly where this method is
currently too slow to run: each candidate is one sparse solve, so it is `sample_size` solves per
road, fine per block and impractical at 11k parcels.

So the honest verdict is *not* "the single objective is worse". It is **"on small blocks, where the
flagship's loops are truncated away anyway, it wins narrowly on the median; the test that matters
needs the incremental scorer the note sketches."**

## Testing note

Two versions of the selection-rule test were written and both were **vacuous**, caught by fault
injection:

- *"permeability rises as roads are added"* — monotone **by construction** for any road set
  whatsoever, since roads only add conductance. Passes for a method choosing at random.
- *"beats what clearance would have picked"* — clearance's deepest-parcel road is usually the long
  one, so a method picking the longest candidate passes too.

The surviving test enumerates the candidate set the method chose from and requires its pick to be
the **argmax by gain-per-metre**. It fails under both injected faults (select-by-length, and
select-by-raw-gain-instead-of-per-metre).
