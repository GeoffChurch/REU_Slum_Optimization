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

## The incremental scorer: built, and it changes the picture

The note's cost concern was real — one sparse solve per candidate meant sampling candidates rather
than considering them. The fix is the first-order sensitivity. With `P = bᵀL⁻¹b` and `v = L⁻¹b`
already solved, upgrading edge (i,j) by `dg` changes `L` by `dg·(eᵢ−eⱼ)(eᵢ−eⱼ)ᵀ`, so

    ΔP ≈ −dg · (vᵢ − vⱼ)²

which costs **one solve per round** and then O(1) per edge. The exact rank-1 value divides by
`1 + dg·(eᵢ−eⱼ)ᵀL⁻¹(eᵢ−eⱼ) ≥ 1`, so the linearization **overstates** the gain — it is a ranking
heuristic, not a score. So it shortlists (all candidates, free) and the exact metric decides the
shortlist. `linearized_gain` in the module.

Effect: **every** candidate is now considered instead of a random 12, and the method went from
~20× clearance's wall clock to ~8×.

## Against `topology`, on the small blocks where it can run

n=8 blocks ≤120 parcels, Lens A at matched displacement 10%:

| method | permeability at D | road m | displacement | seconds |
|---|---|---|---|---|
| clearance / clearance_looped | 0.5829 | 67.2 | 0.0695 | 0.03 / 0.39 |
| **resistance_greedy** | 0.5583 | **39.4** | 0.0792 | 3.23 |
| topology | 0.3399 | 76.8 | 0.0966 | 3.12 |

- beats **topology on 8/8** blocks, median paired delta **+0.273**
- beats **clearance on 5/8**, median paired delta **+0.105** (up from +0.0125 with the sampled
  scorer) — and does it with **41% less road**

The topology result deserves a caveat rather than a victory lap: `topology` optimizes universal
street access / k-complexity, not permeability, so losing on permeability is close to what its own
objective predicts. It is a meaningful reference, not a like-for-like competitor.

The per-block spread against clearance is wide — −0.136, −0.103, −0.058, +0.069, +0.140, +0.283,
+0.324, +0.473 — so this is a method that wins big and loses moderately, not one that edges ahead
uniformly. **Still 5/8, so still not domination, and the note's stopping rule still applies.** But
the margin grew eightfold when the scorer stopped sampling, which suggests the remaining gap is at
least partly search quality rather than the objective being wrong.

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


## Pushing further: both levers failed, informatively

The +0.0125 → +0.105 jump came from the scorer, so the hypothesis was that the remaining gap is
search quality. Two levers, same 10 blocks, Lens A at D=10%:

| variant | permeability | road m | secs |
|---|---|---|---|
| clearance / clearance_looped | 0.6321 | 82.9 | 0.03 / 0.39 |
| rg, shortlist 6, no loops | **0.6752** | 66.3 | 4.5 |
| rg, shortlist 6, **with loop candidates** | **0.6752** | 66.3 | 9.6 |
| rg, shortlist 20, with loops | 0.6622 | 62.2 | 18.2 |

All three beat clearance on **6/10** blocks. Neither lever helped.

**Loop candidates are generated and never chosen.** Not dead code — 90–275 connectors per round on
50–160-parcel blocks — and the output is bit-for-bit identical with them enabled. The reason is
structural: an access road moves a parcel from footpath-only to road-adjacent, a large first-order
gain, while a connector only adds redundancy among already-served parcels, a second-order one. Per
metre, **access dominates at every step** until the gain floor stops the greedy.

That is the honest answer to "does one objective subsume access and redundancy?" — it does, and its
verdict inside this budget is *access first, redundancy never*. It also explains why
`clearance_looped` scores identically to `clearance` at a 10% displacement budget: the two-stage
method's first stage terminates on **depth**, not on gain, leaving budget the loop stage then
spends — and the objective says that budget is better spent on more access.

Loop candidacy is now default-off (it doubled wall clock for a bit-identical result), with the
capability kept because the measurement is the point.

**More exact search makes it WORSE.** Shortlist 20 scores 0.6622 against shortlist 6's 0.6752, at
4× the wall clock. That falsifies the search-quality hypothesis: this is **greedy myopia**, not
search deficiency. A purer per-metre argmax picks a locally better road that leads to a worse
trajectory; the linearization's bias at shortlist 6 is acting as an accidental regularizer.

## Where that leaves it

The gap to clearance is not the candidate set and not the search budget — it is the greedy itself.
Which is precisely what route (A), the convex relaxation, would fix: it optimizes the whole road set
at once instead of one road at a time. The obstacle is still the one the design note named, and
nothing here has moved it: **a road is a combinatorial frontage path, not a free edge**, so the
rounding remains unsolved.

So the direction is coherent and the near-term verdict is unchanged: 6/10 is not domination,
clearance+loops stands, and the next real move is rounding — not more greedy.
