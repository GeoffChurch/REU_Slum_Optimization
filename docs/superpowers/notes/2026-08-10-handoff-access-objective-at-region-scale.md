# Handoff: the access objective at region scale

**Date:** 2026-08-10
**Branch:** `continuum-permeability` @ `72f8126`, 72 commits ahead of `main`, working tree clean
**State:** 583 tests pass, `mypy --strict` clean. No production code was changed today.

Written for a session starting cold. Read §0 first — there is one decision that has to be made
before the work can be picked up at all.

---

## §0. RESOLVED 2026-08-10 — option (b), tracked under `scripts/perf/`

> The owner chose **(b)**: the live code moved to `scripts/perf/` unchanged and is now tracked.
> `scripts/` was already the home for exactly this kind of harness (`consensus_sweep.py`,
> `crossblock_probe.py`, `pair_matrix.py`), so this is the existing convention, not a new one.
> **§4's productionization decision is untouched and still open.**
>
> Moved: the 16 live modules + their JSON results. Deliberately **not** moved, and still
> destroyable: `arterial_incremental.py` / `arterial_lazy_incremental.py` (patched forks of
> `src/reblock/methods/arterial{,_lazy}.py`) with `run_patched.sh` and `bias.py`. Tracking a stale
> 531-line fork that a script `cp`s over `src/` is a drift hazard and a footgun; the incremental
> reformulation is a closed experiment whose finding is recorded in the 2026-08-09 tie-sensitivity
> note. Logs and `region_block.pkl` also stay behind — the pkl self-regenerates in ~67 s.
>
> The original problem statement is kept below as the record of why.

Everything built today — the tier-2 shortlist greedy, the selector Protocol, and eight measurement
harnesses, about **1,670 lines** — lives in `scratchpad/perf/`, which is **git-ignored**
(`.gitignore:38`). None of it is committed. Only the *findings* are (three notes plus the backlog).

That follows the repo's existing convention (the tie-sensitivity note cites `scratchpad/perf/`
paths the same way) and it was fine when scratchpad held throwaway probes. It is no longer fine:
this is working, measured, reusable code, and `git clean -fdx` or a fresh clone destroys it.

**Decide before doing anything else:**

- **(a) Promote it.** Move `selectors.py` + `shortlist_greedy.py` into `src/reblock/methods/` as the
  real thing (see §4), leaving the harnesses in scratchpad. This is the productionization step
  anyway.
- **(b) Track the scratchpad files as-is** under `scripts/` or an `experiments/` tree, unchanged, so
  they survive while the productionization decision waits.
- **(c) Accept the loss.** Legitimate — the notes carry every number, and the code is
  re-derivable — but it is a choice, not a default, and it costs about a day to rebuild.

Files that matter, in dependency order:

| file | lines | what it is |
|---|---|---|
| `selectors.py` | ~165 | `CandidateSelector` Protocol + `ScoreAll` / `FirstOrder` / `StochasticFirstOrder` / `RandomSample` |
| `shortlist_greedy.py` | ~180 | `_greedy_arterials` with an injected selector; `ScoreAll` arm verified bit-identical |
| `control_check.py` | ~55 | proves the control arm IS the shipped greedy — **run this after any edit to the above** |
| `null_model.py` / `null_analyze.py` | ~200 | ranked-k vs random-k |
| `rank_decompose.py` | ~230 | numerator/denominator decomposition + the tier-3 gate |
| `stochastic_restarts.py` | ~160 | best-of-R |
| `region_shortlist.py` | ~110 | the region-scale run |
| `first_order_rank.py`, `snap_vs_peel.py`, `rank_throughput.py` | ~450 | the diagnostics behind §1–§2 |

`scratchpad/perf/region_block.pkl` (1.7 MB) caches the 11,006-parcel region block; regenerating it
costs ~67 s and it is what makes region experiments re-runnable in minutes.

---

## §1. What the day established

Full detail in the three committed notes. Condensed, so nothing is re-derived:

- `notes/2026-08-10-tier-2-first-order-access-gain.md`
- `notes/2026-08-10-the-ranking-earns-its-place-tier-3-does-not.md`
- `docs/superpowers/backlog.md` — "Making the access objective affordable at region scale"

**Tier 2 works and is the only rung of the cascade that was needed.**

| | |
|---|---|
| region scale, measured | **79.6 min** for 15 roads (was: not finished after 11.6 h) — **40×** |
| block scale | ~5× at k=128 |
| per-candidate cost split | peel **88%**, `_snap` 12% — the backlog blamed the wrong term |
| ranking quality | first-order estimate ~ exact benefit, ρ = **+0.937** |
| outcome vs exhaustive | k=128 median 0.7451 vs exact 0.7414; beats it on **6/8** blocks |

**Three corrections to the original spec**, each measured:

1. Weights are **`d² − 1`**, not `(d−1)²`. The greedy optimizes `budget.access_burden` = Σd²; the
   *reported* metric is `eval.access_burden.burden` = Σ(d−1)²/n. Similar names, different functions.
2. The cost proxy was the weak half. Chord length ~ exact displacement is only +0.649;
   **buildings-in-corridor** is **+0.922**, for the same single bulk `dwithin`.
3. Ranking pre-snap is safe: chord length ~ snapped length is +0.975.

**Two results that redirect the work:**

- **Exhaustive per-candidate scoring buys no measurable outcome quality.** The exact greedy's argmax
  flips under a 1e-10 perturbation, so it is one arbitrary draw. A shortlist matches or beats it.
- **Therefore fidelity is the wrong axis.** Effort on computing a single step's argmax more exactly
  is wasted; effort on covering the space of *runs* is not.

**Dead — do not re-propose.** Tier 3 (ALT/landmark distances). It improves only the numerator, whose
ceiling measures +0.090 of ρ — which reads as a green light and is not one, because the exact greedy
already *is* that perfect ranking and loses to tier 2 on 6/8 blocks. A null result about **value**,
so a better landmark scheme does not rescue it. Recorded at length in the backlog precisely because
it is the most principled-looking rung and will otherwise be suggested again.

---

## §2. The binding constraint has moved

Tier 2 removed per-candidate scoring cost. **Candidate enumeration is now the cost**, and this is
measured, not suspected — across the 15-step region run:

| step | 1 | 8 | 15 |
|---|---|---|---|
| candidates | 468,968 | 879,773 | **1,180,388** |
| secs | 139.5 | 390.5 | 466.9 |

The set grows **2.52×** because uncapped `_anchor_points` takes every network vertex and each
committed road is a boundary-graph path adding tens more. Two thirds of the 79.6 min is that growth.

**`max_anchors` caps exactly this and is the obvious next lever** — with a caveat that must be
measured, not assumed: it drops per-vertex anchors and biases toward long chords over short local
connectors, which for an *access* objective is precisely the wrong bias. Its earlier dismissal
("does not rescue it") rested on an inference now known to be wrong, so it is **unevaluated, not
rejected**. Note also that the 66-minute `max_anchors=24` observation behind that dismissal is
**still unexplained** — 276 candidates at 242 ms is ~67 s per step. Something else was slow in that
run and nobody knows what.

---

## §3. The most promising open thread: stochastic restarts

The best measured return of anything here, and it is *not* from the original cascade.

The scatter between arbitrary choices is wide and **bidirectional** — so it is unexploited range,
not only a reliability problem. `StochasticFirstOrder(k, pool, seed)` draws k from the top `pool` by
score, keeping the ranking's signal while making runs independent, so best-of-R becomes possible.

| arm | burden_red | perm | secs |
|---|---|---|---|
| exact | 0.7414 | 0.7305 | 13.2 |
| **best-of-4, pool=1024** | **0.7867** | **0.7712** | **12.5** |

One exact run costs the wall clock of **4.6 restarts**, so best-of-4 fits *inside* exact's budget
and beats it on burden by +0.045 and on permeability by +0.041. Two reasons to believe it: perm is
not selected on and rises +0.060 alongside the selected burden's +0.087; and the `pool` sweep
reverses with R exactly as the mechanism requires (tight pool wins at R=1, wide at R=4).

**Before it could ship:** a wider block sample (n=8 today), a real `pool`/R sweep rather than two
points, confirmation across displacement budgets (only D=0.10 tested), and — the genuinely
unresolved design question — **what a restart selects on when the method reports two metrics.**
Burden is the greedy's objective; permeability is co-reported; nothing arbitrates. Picking one
silently would be a methodological choice smuggled in as an implementation detail.

---

## §4. The productionization decision (not made — deliberately)

Tier 2 is not wired into the shipped method. Doing so is a real decision with a real cost, and it
was left for the owner.

**Where it belongs.** `arterial_lazy` already occupies this exact seam ("reuses arterial's exact
scoring machinery unchanged; only changes which candidates get scored each step"), and CELF is
invalid for this objective because burden reduction is not submodular. Tier 2 is the access
objective's counterpart — a sibling engine, not another flag.

**How it should be selected.** `GreedyArterialReblocker` already carries `lazy` +
`candidate_policy` + `rescore_every` — three fields that jointly pick an engine, dispatched on in
`propose`. A fourth makes it worse. An injected `ArterialEngine` (exact / lazy / shortlist),
resolved once in config, is what the standing directive asks for and `selectors.py` is already
shaped that way.

**What it costs.** That refactor changes `ArterialIdentity`, which **invalidates the derive cache**
and forces an examples regeneration. It also touches `conf/method/greedy_arterial*.yaml` and the two
`lazy: true` lines in `conf/compare_config.yaml`. Real, bounded, and it belongs in the decision
rather than hidden inside it.

**What it must not do.** It cannot quietly replace the exact path for published single-block
examples — the same block moves up to 0.25 burden reduction depending on engine. Either the examples
move with an announced before/after, or the shortlist engine is used only where exact is infeasible
(region scale) and block-scale figures stay on exact.

---

## §5. Operational traps this work actually hit

Cheap to avoid, expensive to rediscover.

- **`pixi run lint` does not check `scratchpad/`.** Ruff respects `.gitignore`. Use
  `pixi run ruff check --no-cache <path>` explicitly. I briefly mistook this for a cache bug; it is
  not, it is the ignore file.
- **Long background runs get killed.** Four so far (C9 at 2/10, C20 at 2/12 and 5/12, and the first
  region run at 73 min). Cause still unknown; not OOM (cgroup reports `oom_kill 0`, pressure-stall
  flat). Run anything long via
  `pixi run python -m scratchpad.complexity.instrumented <module> <logfile>` — it catches signals
  with `sigwaitinfo` so the `siginfo` names the **sending pid**, which an ordinary handler cannot.
- **Print progress incrementally, with `flush=True`.** The first region run burned 73 minutes and
  produced *zero* rows because it only reported totals. `greedy_shortlist` now takes an `on_step`
  callback for this reason.
- **`| tail -N` on a background run buffers everything** until exit — redirect to a log file instead.
- **`pkill -f <pattern>` self-matches** when the pattern appears in the same command line; it kills
  its own shell. Filter with `grep -v grep` and act on explicit PIDs.
- **Run scripts as `pixi run python -m scratchpad.perf.<name>`.** The pythonpath is pytest-only.
- **Hooking `_best_candidate` cannot see `_STEP_STATE`** — `_greedy_arterials` clears it in a
  `finally` *before* the reduce is called. My first instrumentation silently recorded zero steps
  because of this. Track what you need yourself.
- **The permeability width floor is 7 m** (`DEFAULT_ROAD_WIDTH_M`). Hardcoding a 3.0 half-width
  raises `ValueError` from `buildable_widths`. Use `DEFAULT_ROAD_WIDTH_M / 2.0`.

---

## §6. Suggested order of work

1. **§0 decision** — the code is unprotected until this is made.
2. **`max_anchors`, measured properly** (§2). Highest value: it attacks the now-binding cost, it is
   cheap, and it is currently unevaluated on a bad inference. Measure the long-chord bias explicitly
   against the access objective rather than assuming it away.
3. **Widen the restart evidence** (§3) — more blocks, a real `pool`/R sweep, more than one
   displacement budget. Then raise the two-metric selection question with the owner.
4. **Productionize tier 2** (§4) once the owner has weighed the cache-invalidation cost.
5. Leave alone: tier 3 (dead, §1), tier 1 (open but now only a throughput play, and it competes with
   item 2).

## §7. Also still open, from before today

- **Task #21 — ShapeStandardizingRegionBuilder**, never implemented, gates Phase 3 of the OT work.
- **The "gravity grid" / coherence-term build.** The owner asked for it; the rectilinearity
  measurement came back discouraging (ρ −0.036 across methods) but confounded by block, and the
  owner explicitly said "I'm not sure we should skip the build even if coherence is emergent... I
  was curious if the converse would hold". **Never built.** This is the oldest outstanding ask.
- **The free class** — 12.1% of candidates have zero exact displacement and infinite gain; the cheap
  denominator floors at one building and cannot express it. Untouched.
- **PR #51** is open on a *different* branch (`escape-time-and-spectrum`), not this one.
