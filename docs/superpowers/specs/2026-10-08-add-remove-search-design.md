# The add/remove search as one Strategy

Owner, 2026-10-08: prioritized ("merging the add/remove searches into one interface").
BACKLOG, New methods, "Add/remove subset search".

## Why

Four searches in research/roadless each implement part of one family, plus-l take-away-r (wiki `pages/methods/plus-l-take-away-r.md`):

| search | where | moves | linearization | scored by | accepted |
|---|---|---|---|---|---|
| the greedy | clear.grow + a Picker | add batches | tension (finite difference, eps world, search conductance) | exact, or not at all | always |
| exchange | relax.exchange | add 1, swap 1-1; 2-1 / 1-2 when singles fail | eps-world gradient dJ/ds | eps world 1e-6 | if strictly better |
| grow then prune | search.grow_prune | the greedy to 3D, then restore batches | restore tension, optional eps-screened shortlist | exact at D <= d_max (recorded only) | always |
| SIMP's polish | relax.Incumbent.polish | exchange over the undecided buildings | eps-world gradient | the incumbent's eps scorer | if strictly better |

Each new schedule (floating search; a substitute-aware restore score) would be a fifth copy of the loop.
And the choices are resolved in several places: `picker_of`'s string switch, `tension(restore=...)`, `score` flags read inside pickers, exchange's `pairs` escalation, Incumbent's exact-vs-eps scorer.

## The Strategy

One engine, `search.run(c, schedule, ...)`, and four injected parts per phase, built once from the preset's spec where the run is configured.

```python
class Move(NamedTuple):
    add: NDArray[np.int64]          # buildings the move clears
    restore: NDArray[np.int64]      # buildings it puts back
    est: float                      # the linearized change in J (lower is better), for ordering

class Ranking(NamedTuple):
    add: NDArray[np.float64]        # per building: linearized dJ if cleared (inf: not addable)
    restore: NDArray[np.float64]    # per building: linearized dJ if put back (inf: not cleared)
    tension: Tension | None         # the round's system, for movers that sweep it (Catchment)

class Ranker(Protocol):             # the linearization at the current state
    def rank(self, c: Clearing, power: float) -> Ranking: ...

class Mover(Protocol):              # candidate moves, best first
    def moves(self, c: Clearing, rk: Ranking, budget: Budget) -> list[Move]: ...

class Scorer(Protocol):             # J after a move (or NaN: not scored)
    def J(self, c: Clearing, m: Move) -> float: ...

class Acceptor(Protocol):           # which scored move to make, or None: the phase ends
    def choose(self, J: float, scored: list[tuple[Move, float]], cost) -> Move | None: ...

class Phase(NamedTuple):
    ranker: Ranker
    mover: Mover
    scorer: Scorer
    acceptor: Acceptor
    stop: Stop                      # D reached / D left / no move / tries spent
```

A phase is: rank, propose, score, choose, apply; repeated until `stop`.
A schedule is a tuple of phases, with a `Record` observer that writes the rows a preset owes.
Those are the greedy's per-step format, polish_greedy's one row, and grow-then-prune's states at or below d_max.

The parts and the members they come from:

- **Rankers.**
  - `Tension(restore)`: Clearing.tension, both directions from one class (an add phase uses restore=False, a remove phase restore=True).
  - `EpsGradient(eps, rtol)`: Relaxation.value_sgrad at the 0/1 state.
    Adding a is g_a and closing b is -g_b, so one gradient serves both.
- **Movers.**
  - The three pickers as add movers, unchanged in what they propose: `Screened(M)`, `Batched(delta, gap)`, `Spread(delta, source, width)`.
  - `Exchange(width, pairs, pool)`: relax._singles and _pairs.
    The pairs come only when no single improves; that escalation becomes the mover offering a second proposal when the acceptor declines the first.
  - `RestoreBatch(step, shortlist, screen)`: search.restore_batch.
- **Scorers.**
  - `Exact()`: _exact (real geometry, RTOL_SCORE).
  - `Eps(eps, rtol)`: Relaxation.value.
    It shares its warm start with an `EpsGradient` of the same world, as polish_greedy and the incumbent do now.
  - `Unscored()`: NaN, for a single proposal under a greedy that does not score its steps.
- **Acceptors.**
  - `Best(key)`: always make one move (lowest J, or Screened's best gain per cost).
  - `IfBetter()`: strictly lower J, else end the phase.

## The presets

Every current spec string stays, parsed once into a schedule (the edge).
So no row directory, plan name or result moves:

| spec | schedule |
|---|---|
| `M4`, `B0.01g3`, `S0.01cat`, `S0.01catw4` (the greedy) | [add: Tension, picker, Exact or Unscored, Best, until d_max] |
| polish_greedy `<picker> P<t>w<w>[x2]` | greedy to D; cut_to_budget; [exchange: EpsGradient(1e-6), Exchange(w, x2), Eps(1e-6), IfBetter, t scorings] |
| `GP<grow>x<picker>r<step>[m<k>]` | greedy to grow x D; [remove: Tension(restore), RestoreBatch, Exact at D <= d_max, Best, until empty] |
| SIMP `.p<t>w<w>[x2]` / `.P...` | [exchange over Incumbent's movable set, the incumbent's scorer] |

And the first new one, to show the engine pays: **floating search** `FL<picker>`.
It adds a batch, then restores one building at a time while that lowers exact J, and repeats to D.
BACKLOG names it as the next schedule to try.

## What has to hold

- **Bit identity on the CPU.**
  Each preset, run by the engine, reproduces the current code's rows exactly on three small blocks (CPU solver and scans).
  The presets checked: the four greedy pickers, polish_greedy P64w8x2, GP3xS0.01catr0.005m8, and a SIMP plan with `.p256w8x2`.
  GPU runs are compared to 1e-9 (summation order already varies there).
- **No legacy path.**
  clear.grow, relax.exchange, search.grow_prune and Incumbent.polish's loop are deleted once their presets match, and their callers move to the engine.
  Row formats are unchanged, so nothing needs migrating.
- **Checkable.**
  Pickers stop reading a `score` flag: the scorer is injected.
  `picker_of` stays the one place a spec string becomes objects.
  Result-changing parameters (width, tries, delta, eps, rtol) are required and keyword-only below the edge.

## Not in scope

New experiments beyond floating search's first run, such as a substitute-aware restore score or mixing rankers per phase.
Any change to Clearing, Tension or the solves.

## Risks

- **Subtle order dependence.**
  Screened scores its M even when unscored; Spread's width alternatives are deduplicated by set; exchange scores all `k` before choosing.
  Grow-then-prune's rows list each step's cleared buildings in index order, not pick order.
  Each is a line in the identity tests, not a behaviour to improve in passing.
- **Size.**
  About 400 lines move.
  The identity tests are the guard, run before each member's old path is deleted.
