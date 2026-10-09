# The add/remove search as one Strategy

Owner, 2026-10-08: prioritized ("merging the add/remove searches into one interface").
BACKLOG, New methods, "Add/remove subset search".
Third version.
The first split every round into four injected parts; a review found the members did not come apart that way as written.
The second made a member's whole round the unit (Step + state + combinators); the owner then asked for the best long-term design, and chose this one: the outer layer of the second, with each round factored the way every member already works.

## Why

Four searches in research/roadless each implement part of one family, plus-l take-away-r (wiki `pages/methods/plus-l-take-away-r.md`), each with its own loop: the greedy (clear.grow and three pickers), exchange (relax.exchange), grow-then-prune (search.grow_prune) and SIMP's polish (relax.Incumbent.polish).
Every one of their rounds does the same four things:

1. **rank** the buildings by a cheap first-order estimate of what a move does (the tension, or the eps-world gradient);
2. **build** candidate moves from the ranking (one building, a spaced or a diverse batch, swaps), optionally after **refining** the ranking of a shortlist in a costlier world;
3. **evaluate** some candidates in a world (exactly, in an eps world, or not at all);
4. **accept** one (always the best by some key, or only if it beats a threshold).

| member | rank | refine | build | evaluate | accept |
|---|---|---|---|---|---|
| Screened `M<m>` | add tension | | the top m singles | exact, all | best gain per cost |
| Batched `B<d>g<gap>` | add tension | | a spaced batch to the next multiple of d | | the one |
| Spread `S<d>cat[w<w>]` | add tension | | w diverse batches (catchment discount) | exact, all, if more than one | lowest J |
| exchange | eps gradient | | singles and swaps; then a pair tier | eps world, the top width | lowest J, if below J now |
| restore batch | restore tension | the top max(m, 2k) in the eps world | a batch to the next multiple below | | the one |

That table is the design: the members become configurations of one `Round`, and a new search is a new row.
Two experiments BACKLOG names are further rows: a **substitute-aware restore** (the restore batch built with Spread's diversity discount) and **screened adds** (an add ranking refined in the eps world).
Each needs parts of its own ("What the two new rows need", below).

## The parts

All in search.py, typed so the checker catches a bad pairing (a builder that needs the tension's system under a gradient ranking).

```python
class Score(NamedTuple):
    J: float                        # J_power
    P: float                        # P (J_1); NaN in an eps world

class World(Protocol):              # where a clearing is scored
    def score(self, s: SearchState, removed: NDArray[np.bool_]) -> Score: ...
```
- `EXACT`, the exact world: real geometry, the metric conductance, RTOL_SCORE (`clear.exact(c, removed, power)`, the one exact entry point; memprobe_greedy patches it there).
- An eps world is a `relax.Relaxation` bound to the block, which gains `score(s, removed)`; its eps, rtol and conductance (the metric's `c.p` or the search's `c.ps`) are its constructor's, set where the run is configured.

```python
class Ranker(Protocol[R]):          # R: the ranking type it makes
    def rank(self, s: SearchState) -> R: ...

class Refiner(Protocol[R]):         # re-rank a shortlist in a costlier world
    def refine(self, s: SearchState, r: R, first_k: int) -> R: ...

class Builder(Protocol[R]):         # candidate moves, as tiers (a later tier only if no move of
    def tiers(self, s: SearchState, r: R) -> Iterator[Tier]: ...   # an earlier one is accepted)
    def size(self, s: SearchState, r: R) -> int: ...               # first-order batch size (the refiner's k)

class Move(NamedTuple):
    add: list[int]                  # in pick order
    restore: list[int]

class Tier(Protocol):               # one tier's moves, materialised only when looked at
    est: NDArray[np.float64]        # per move: linearized change in J, lower better; inf infeasible
    def move(self, i: int) -> Move: ...
    # Batches: a list of Moves. Swaps: (4, M) index arrays (exchange's singles under P are ~10^6)

class Candidate(NamedTuple):
    move: Move
    est: float
    score: Score | None             # in the evaluator's world, if scored

class Evaluator(Protocol):          # score some of a tier, counting s.scorings
    def evaluate(self, s: SearchState, t: Tier, needs_score: bool) -> list[Candidate]: ...

class Acceptor(Protocol):           # the move to make from the evaluated candidates, or None
    needs_score: bool
    def accept(self, s: SearchState, cands: list[Candidate]) -> Candidate | None: ...

@dataclass(frozen=True)
class Round(Generic[R]):            # a Step
    rank: Ranker[R]
    refine: Refiner[R] | None
    build: Builder[R]
    evaluate: Evaluator
    accept: Acceptor
```
`Round.step` ranks, refines, builds its tiers and drops the ranking (the tension's system and AMG hierarchy) before any scoring solve, as Spread's `del t, H` does now; then it evaluates and accepts tier by tier.
The refiner runs while the ranking is alive, so a refined round's ranking must not hold the tension's system.
An evaluator scores a tier only when the acceptor needs a score or the tier holds more than one move, which is Spread's "score only if there is a choice" and Screened's always, as one rule.

The parts the members need:
- Rankers: `AddTension()`, `RestoreTension()` (Clearing.tension, ranking per unit of population as now; its ranking, `Order`, holds the indices only, so the Screen's solves never run beside the tension's system), `Gradient(world)` (value_sgrad at the 0/1 clearing; adding a is g_a, closing b is -g_b).
- Refiner: `Screen(world, m)`: the top max(m, 2k) by first-order loss re-ranked by their own change in `world` (restore_batch's shortlist).
- Builders: `TopSingles(m)`, `Spaced(delta, gap)`, `Diverse(delta, source, *, reach, width)`, `Swaps(budget, pairs, pool)` (singles and swaps, then the pair tier), `ToLevel(step)` (a restore batch to the next multiple below).
- Evaluators: `NoScore()`, `All(world)`, `Top(world, width, tries)`.
- Acceptors: `Only()`, `Best(key)` (lowest J, or best gain per cost), `IfBetter()` (strictly below s.score.J, in the evaluator's world), `BeatsArchive(archive, d_max)`.

## The outer layer

```python
class Outcome(NamedTuple):
    cleared: list[int]              # in pick order
    restored: list[int]
    score: Score | None             # the new state's, in `world`
    world: World | None

@dataclass
class SearchState:
    c: Clearing                     # c.removed is the clearing; only `apply` changes it
    power: float
    score: Score                    # in `world` (UNSCORED, None: not scored)
    world: World | None
    order: list[int]                # every building cleared, in pick order (cut_to_budget's)
    scorings: int                   # candidate scorings spent
    movable: NDArray[np.bool_]      # the buildings a move may touch (the polish's undecided set)
```
- **Schedules**: `Do(step)`, `Until(part, stop)`, `Seq(parts)`, `With(record, part)` (a phase under its own record) and `Rescore(world)` (the state scored in a world before a phase that compares in it: a phase boundary written in the schedule, not hidden in a step).
- **Stops**: `Reached(d)`, `Emptied()`, `Spent(n)`, `Above(d)`, `AnyOf(stops)`.
- **Records** say what the run owes: `wants(s)` (score this state exactly) and `on_step(s, outcome)`; `GreedyRows`, `Silent`, `PruneRows`, `FloatRows`. They replace the pickers' `score` flag: `apply` scores a new state exactly when its record wants it and the round did not, the same solve the flag made.

## The presets

Every spec string stays, parsed once at the edge; no row directory, plan name or result moves.

| spec | schedule |
|---|---|
| `M4`, `B0.01g3`, `S0.01cat`, `S0.01catw4` | Until(Do(greedy round), Reached(d_max)), GreedyRows |
| polish_greedy `<picker> P<t>w<w>[x2]` | the greedy to D, Silent; cut_to_budget(s.order); Seq(Rescore(eps 1e-6, metric), Until(Do(exchange round), Spent(t))) |
| SIMP `.g` / `.G` seed and track | the greedy to the budget, Silent; cut_to_budget(s.order) |
| SIMP `.p` / `.P` polish | Seq(Rescore(the incumbent's world), Until(Do(exchange round), Spent(t))) over its movable set |
| `GP<grow>x<picker>r<step>[m<k>]` | Seq(With(Silent, the greedy to grow D), Until(Do(restore round), Emptied())), PruneRows |
| **new** `FL<grow>x<picker>r<step>[m<k>]c<cap>` | grow-then-prune, each restore followed by Until(Do(conditional add), AnyOf(Spent(cap), Above(d_max))), FloatRows |

`greedy_round(spec, sweep)` replaces `picker_of` (the one place a picker string becomes objects); the Picker protocol and its three classes go, their logic now in the builders and acceptors.
Result-changing constants become required, keyword-only fields set where presets are parsed: Diverse's `reach` and `width`, Swaps' `pool` (PAIR_POOL), the screen's eps (EPS_SCREEN), polish_greedy's RTOL and SCORE_EPS.

## Floating search

Backward floating on grow-then-prune, the schedule NOTES ("Warm SIMP path and grow-then-prune") and BACKLOG name: re-add after a restore that hurts, the fix for the prune's collapse on gated blocks (substitutes restored one by one until the gate's pocket shuts, 22422).
- After each restore, conditional adds: the greedy's add round from the current state, scored exactly, kept only if its J beats the best archived at its budget level (Pudil's rule) and its D is within d_max.
- The archive keeps the best exact state per level (the restore step's lattice), as NamedTuples; every exact state at or below d_max updates it.
- Termination: a kept add strictly improves its level's best and there are finitely many states; re-adding exactly what was restored reproduces an archived J and is refused; a scorings cap bounds the time.
- The forward form ("restore while J falls") would never fire: clearing only adds conductance (common.py:186).
- Rows: the archive's best per level, in increasing D, each with its full clearing (`clearing`); no `cleared` deltas (the states are not nested), so `cleared_through` readers do not read them; picker_compare and simp_compare need only D, perm, perm1, t.
- First run: grow-then-prune's five blocks (19421, 19510, 19537, 9712, 22422) at d_max 0.15, against GP3xS0.01catr0.005m8 and the greedy.

## Placement

- search.py: everything above, the GP and FL presets and their CLI; it imports clear, never relax (eps worlds are built by the caller and passed in).
- clear.py: Clearing, Tension, `exact`, Catchment and the sweeps, `_spread`, rows_dir, cleared_through, and the greedy CLI (`clear.py run|some`, unchanged; greedy_block imports search inside the function, as search imports clear).
- relax.py: Relaxation (gaining `score`), Incumbent and relax.one using the engine.
- Deleted once their presets agree: clear.grow, the Picker protocol, Screened, Batched, Spread, picker_of, `_exact`, relax.exchange, search.grow_prune and restore_batch.
  Their callers move: greedy_block, polish_greedy, relax.one, Incumbent.polish, gramprobe, memprobe_greedy, profile_round, releaseprobe.

## What has to hold

- **Agreement checked, deviations flagged** (owner: "we don't strictly need bit-identical, just to flag any deviations for investigation").
  - The current code run twice on the CPU first, concurrently: its noise floor (expected none).
  - The engine compared against fresh runs of the current code on three small blocks (19421, 19510, 38138), on the CPU, every column but times, the first diverging row reported with its columns and investigated; an explained difference can stand.
  - Presets: the four greedy pickers, polish_greedy P64w8x2, GP3xS0.01catr0.005m8, SIMP `.k1s1e-06.p256w8x2.GS0.01cate1`.
  - Coverage: which branches fire on the check blocks (the pair tier, Spread's deduplication, a partial step in cut_to_budget, the shortlist reordering), from counters in the code paths themselves; a block added where one never fires.
  - Fault injection: one perturbation shown to be reported.
- **GPU memory**: memprobe_greedy's own peak lines on a large block under the translucent search, old twice (its noise) and new.
- **Each task leaves every script runnable**: a member's callers move in the task that deletes its old loop.

## What the two new rows need

The final review found that neither is one configuration of the parts as built; this spec had said each was.
- **Substitute-aware restore**: a restore ranking that keeps the Tension for the catchment's gram and sheds its system before the Screen solves, and a restore-direction diverse builder (`Diverse` shortlists unremoved buildings and fills to the next multiple above).
- **Screened adds**: an add-direction `Refiner[Gains]` (`Screen` re-ranks restores, closing each building in its world), and a builder that reads the refined order (`Diverse` re-derives its gains from the tension).
- Both: the ranking cannot hold the tension's system through the refiner's solves, so a ranking that needs the Tension later splits into the arrays the refiner reads and the system it does not.

## Not in scope

- The two new rows (substitute-aware restore, screened adds).
- Changes to Clearing, Tension or the solves.
