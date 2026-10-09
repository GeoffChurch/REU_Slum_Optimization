# The add/remove search as one Strategy

Owner, 2026-10-08: prioritized ("merging the add/remove searches into one interface").
BACKLOG, New methods, "Add/remove subset search".
Revised after a design review the same day; the owner approved the revised shape ("Step + state + combinators").

## Why

Four searches in research/roadless each implement part of one family, plus-l take-away-r (wiki `pages/methods/plus-l-take-away-r.md`):

| search | where | moves | ranked by | scored by | accepted |
|---|---|---|---|---|---|
| the greedy | clear.grow + a Picker | add batches | tension (finite difference, eps world, search conductance) | exact, or not at all | always |
| exchange | relax.exchange | add 1, swap 1-1; 2-1 / 1-2 when no single improves | eps-world gradient dJ/ds | eps world 1e-6, metric conductance | if strictly better |
| grow then prune | search.grow_prune | the greedy to 3D, then restore batches | restore tension, optional eps-screened shortlist | exact at D <= d_max (recorded only) | always |
| SIMP's polish | relax.Incumbent.polish | exchange over the undecided buildings | eps-world gradient | the incumbent's eps scorer | if strictly better |

Each owns its loop, and the choices are resolved in several places: `picker_of`'s string switch, a `score` flag each call site sets, exchange's `pairs` escalation, Incumbent's exact-or-eps scorer, and picker strings carried in GrowPrune and SIMP plans and resolved downstream.
A new schedule would be a fifth loop.

## Why not a finer split

The first draft split every round into Ranker, Mover, Scorer and Acceptor.
The review found that the members do not come apart that way.
- Choosing, scoring and their order are woven into each member: Screened chooses by exact gain per cost, Spread scores only when more than one distinct batch survives, exchange scores all of a tier and escalates to pairs only when no single improves, and restore_batch re-ranks its shortlist by eps-world solves inside the batch.
- Holding a round's ranking (its Tension: system, AMG hierarchy) in the engine through the scoring solves would undo Spread's `del t, H` (clear.py:586), a GPU memory regression a CPU check cannot see.

So the unit is a member's whole round, moved over nearly unchanged.

## The Strategy

**A Step is one round of one member.**

```python
class Score(NamedTuple):
    J: float                        # J_power
    P: float                        # P (J_1); NaN where the world gives none

class Outcome(NamedTuple):
    cleared: list[int]              # buildings the round clears, in pick order
    restored: list[int]             # buildings it puts back
    score: Score | None             # the new state's score in the step's world, if it has one

class Step(Protocol):
    def step(self, s: SearchState, wants_score: bool) -> Outcome | None: ...
        # None: no move (exchange found nothing better; a conditional add was refused)
```

**The state is explicit and per block.**

```python
@dataclass
class SearchState:
    c: Clearing                     # c.removed is the clearing; only the engine changes it
    power: float
    score: Score                    # the current state's score (NaN: not scored) ...
    world: World | None             # ... and the world it is in (None: not scored)
    order: list[int]                # every building cleared, in pick order (cut_to_budget's)
    scorings: int                   # candidate scorings spent (exchange's and floating's caps)
    movable: NDArray[np.bool_]      # the buildings a move may touch (the polish's undecided set)
```

The engine applies an Outcome: it clears, restores, appends to `order` and carries the score.
If the Record wants the new state scored and the step did not score it, the engine scores it exactly through `clear._exact`, looked up at call time so memprobe_greedy's instrumentation still applies.
At a phase boundary where the next step works in another world than `s.world`, the engine re-scores the state in that world first, in today's call order (polish_greedy scores `value(r)` before the first `value_sgrad(r)`).

**Schedules are combinators**: `Seq(*parts)` and `Until(part, stop)`, a part being a Step or a schedule; a Step returning None ends its `Until`.
Stops are a Protocol, `done(s) -> bool`: `Reached(d)` (D at or past d, or nothing left), `Emptied()` (nothing cleared), `Spent(n)` (scorings at or past n).
relax.one's `.G` (two tracks, the lower eps J kept) stays a comparison of two results in relax.one: input, not configuration.

**The Record says what the run owes**: `wants(s) -> bool` (score this state exactly), `on_step(s, outcome)`, `rows() -> DataFrame | None`.
It replaces the `score` flag at call sites: the greedy's step passes `wants_score` to its picker.
- `GreedyRows` (clear.py): today's per-step format, every state scored (greedy_block).
- `Silent()`: nothing, nothing scored (a seed, a track, polish_greedy's greedy phase).
- `PruneRows(d_max)` (search.py): grow_prune's states at or below d_max, cleared in index order, written in increasing D.
- `FloatRows(archive)` (search.py): floating search's best state per budget level (below).

**Two layers.**
The presets parse into frozen config objects (the run's identity, built once where the run is configured), whose `bind(c, power)` makes the per-block parts.
A `World(eps, rtol, conductance)` binds to one Relaxation per block, shared by every part in that world, so a scorer and a gradient share a warm start as polish_greedy and the incumbent do now.
`conductance` (the metric's or the search's) is part of the world and its name: the polish scores under the metric, the restore screen and the tension under the search conductance.

## The steps

- **`GreedyAdd(picker)`** (clear.py): the tension, then the picker's pick, unchanged; `wants_score` is the picker's `score`.
- **`Exchange(world, width, pairs, pool)`** (relax.py): one round of relax.exchange over `s.movable`, its tiers (singles; pairs when no single improves) handed to a `Chooser`.
- **`RestoreBatch(step, shortlist, screen)`** (search.py): restore_batch's body.
  The screen is a `Valuer` Protocol (`value(x)`), so search.py does not import relax.
- **`Chooser(scorer, width)`** (search.py), shared: score the top `width` of each tier against the scorings left, take the best if it beats the threshold, else try the next tier, else return None.
  Moves come as arrays, materialised only for the top `width` (exchange's singles under `P` are about 10^6 swaps).
- **`ConditionalAdd(picker, archive)`** (search.py), floating search's: a greedy add from the current state, exactly scored, kept only if it beats the archive's best at its level.

## The presets

Every current spec string stays and is parsed once, at the edge, into config objects; picker strings inside GrowPrune and SIMP plans are resolved there, not downstream.
No row directory, plan name or result moves.

| spec | schedule |
|---|---|
| `M4`, `B0.01g3`, `S0.01cat`, `S0.01catw4` | Until(GreedyAdd(picker), Reached(d_max)), GreedyRows |
| polish_greedy `<picker> P<t>w<w>[x2]` | Until(GreedyAdd, Reached(D)), Silent; cut_to_budget(s.order); Until(Exchange(eps 1e-6, metric), Spent(t)) |
| `GP<grow>x<picker>r<step>[m<k>]` | Seq(Until(GreedyAdd, Reached(grow D)), Until(RestoreBatch, Emptied())), PruneRows(D) |
| SIMP `.g` / `.G` seed and track | Until(GreedyAdd, Reached(budget)), Silent; cut_to_budget(s.order) |
| SIMP `.p` / `.P` polish | Until(Exchange(the incumbent's world), Spent(t)) over the incumbent's movable set |
| **new**: `FL<grow>x<picker>r<step>[m<k>]` | grow-then-prune, each restore followed by Until(ConditionalAdd, refused or Spent(cap)), FloatRows |

Result-changing constants become required, keyword-only fields of the config objects, set where presets are parsed: Spread's `reach` and `width` (today defaulted, clear.py:566-567), exchange's PAIR_POOL, search's EPS_SCREEN, and polish_greedy's RTOL and SCORE_EPS.

## Floating search

This is the schedule NOTES ("Warm SIMP path and grow-then-prune") and BACKLOG name: backward floating on grow-then-prune, re-adding after a restore that hurts.
It is the fix for the prune's collapse on gated blocks, where substitutes (each cheap to restore alone) are restored in one batch and the gate's pocket shuts.

- After each restore batch come conditional adds: a greedy add from the current state, exactly scored, kept only if its J beats the best J archived at its budget level (Pudil's rule).
- The archive keeps the best exact Score and clearing per level, the levels being the restore step's lattice (ceil(D / step)); every exact score updates it.
- Termination: a kept add strictly improves a level's best, and there are finitely many states; a scorings cap (`Spent`) bounds the time.
  Re-adding exactly what was just restored reproduces an archived state, does not beat it, and is refused: no cycle.
- It is not the forward form ("restore while J falls").
  Clearing only ever adds conductance (common.py:186), so a restore with everything else fixed raises exact J, and that form would never fire.
- Rows: the archive's best per level, in increasing D, each with its full clearing (column `clearing`).
  Its states are not nested, so it writes no `cleared` deltas: renderers reading `cleared_through` (clear.py:660) do not read it; picker_compare and simp_compare read `D` and `perm`, which it has.
- First run: grow-then-prune's blocks in NOTES (9712, 22422 and the small blocks), against GP3xS0.01catr0.005m8 and the SIMP path, at D 0.05.

## Placement

- search.py: the engine (SearchState, Score, Outcome, Step, Record, the stops, Seq, Until, Chooser), RestoreBatch, ConditionalAdd and the archive, PruneRows, FloatRows, and the GP and FL presets.
  It imports clear, not relax.
- clear.py: Tension, the pickers, GreedyAdd, Reached, GreedyRows, rows_dir, cleared_through.
- relax.py: World binding, Exchange, and the Incumbent's use of the engine.
- Deleted once their presets agree: clear.grow, relax.exchange (its body becomes Exchange and the Chooser), search.grow_prune, and Incumbent.polish's own loop.
  Their callers move to the engine: greedy_block, polish_greedy, relax.one, gramprobe, profile_round, releaseprobe, memprobe_greedy.

## What has to hold

- **Agreement checked, deviations flagged (owner: "we don't strictly need bit-identical, just to flag any deviations for investigation").**
  - Baseline first: the current code run twice on the CPU (pyamg, one core), which should agree with itself bit for bit; on the GPU, the noise floor of two runs (atomic sums).
  - Compare the engine against fresh runs of the current code, not the rows on disk (GPU, older code).
  - Blocks: three small ones on the CPU.
    Columns are compared exactly except times (`t`, `t_greedy`): `moves` and `scorings` exactly, `cleared` in pick order (index order for grow-then-prune).
  - Presets: the four greedy pickers, polish_greedy P64w8x2, GP3xS0.01catr0.005m8, and a SIMP plan with `.k1s1e-06.p256w8x2` (a polish needs `.k<n>s<eps>`).
  - Report the first step that diverges (rows after it carry no information), with which building and which number, and investigate it before the old path is deleted.
    An explained difference can stand, its explanation in the plan's ledger.
  - Coverage: count how often each branch fires on the check blocks (the pairs tier, Spread's deduplication, a partial step in cut_to_budget, the shortlist), adding a block where one never does.
  - Fault injection: perturb the engine once (a tie broken the other way) and show the check reports it.
- **GPU memory.** A memprobe_greedy run on a large block under the translucent search, old against new, before the old path is deleted.
- **Checkable.** Steps, stops and records are Protocols, never strings switched on; picker strings are resolved at the edge.

## Not in scope

- Experiments beyond floating search's first run (a substitute-aware restore score; a Spread-discounted restore batch).
- Changes to Clearing, Tension or the solves, including Clearing's defaulted `rtol`.
