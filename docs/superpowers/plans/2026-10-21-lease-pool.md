# The lease pool: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** reblock's cluster waves run as leases of ai's GPU/CPU market: a queue per wave on the shared home, pools of short identical workers per (node, memory class), a steward per wave and a long lane, submitted by a `wave` verb; built and smoke-tested beside today's `blocks` and `cpu` kinds, which go in one commit once the cycle_desire g3 runs have settled.

**Architecture:** A wave's jobs are rows of a SQLite queue in `$HOME/reblock-waves/<w>/` (journal_mode DELETE, one IMMEDIATE transaction per operation, behind a `Queue` Protocol).
Pool leases (kind `pool`, one per (node, class), continued and widened by the market through `cs.launches.width`) run `pool.py work`, which claims the longest job that fits its walltime, runs it as a child, counts the rows it writes, and promotes it a rung on an OOM; a steward lease per wave (kind `steward`) reaps dead claims on squeue's evidence, asks workers to step aside, writes `summary.json` and ends the wave; jobs no lease can hold run as `long` runs sized per job.
This machine's `Waves` Campaigns says which runs are whose leases and what the market starts for a class on a node (`Waves.template`: the wave with the most pending jobs there, at its own commit), `Pool.more` continues a lease from the steward's summary, and `cluster_demand` states the one `reblock` campaign's demand with every wave's lines.
Every pooled script keeps the rows that stand (`clearings.Standing`, `Keep`), so a double claim or a killed job wastes a run and nothing else.

**Tech Stack:** Python 3.12 (uv), sqlite3, pandas and pyarrow (the estimators' stored rows), GeoffChurch/cluster_submit pinned at ec79a0eb (`Kind`, `Campaigns` with `template` and `starts`, `Template`, `launches`, `gpu`, `gpuboard`, `remote`), Slurm 23.11 on ai.

**Spec:** docs/superpowers/specs/2026-10-21-lease-pool-design.md (approved by the owner 2026-10-21, "Lease-pool spec LGTM"; its section 9 records the decisions on its twelve questions), with its brief and GeoffChurch/mycooc's lessons in .superpowers/sdd/2026-10-21-lease-pool/.
Read the spec whole before any task; section numbers below (3.2, 3.7, ...) are its.
What landed after the spec, and the rulings that reconcile it, are in Global Constraints.

## Global Constraints

Every task's requirements include this section.

### Where the work happens, and when

- **Tasks 1-9 run in the worktree** `/home/gchurchill/src/reblock/.worktrees/lease-pool` on branch research/lease-pool; push it after each commit (`git push origin research/lease-pool`).
  Why not research/roadless in the main checkout: the project's one watch timer (`cluster_submit-reblock-sync`) runs the main checkout's working-tree `research/roadless/cluster.py sync` on that checkout's registry, so a half-built cluster.py there would stop the syncs of cp-roads4-desire-g3a and -g3b, which must never be cancelled.
- **Tasks 10 and 11 run on research/roadless in the main checkout** `/home/gchurchill/src/reblock`, after the controller merges research/lease-pool into it (Task 10 Step 1); push research/roadless after each commit.
  Waves are only ever submitted from the main checkout: the timer syncs no other registry, so a wave submitted elsewhere would never be continued.
- **Against the epistemetry move** (docs/superpowers/plans/2026-10-17-epistemetry-move.md, executing now on research/roadless in the main checkout):
  - Task 5 (`Standing`, which edits consolidated.py) starts only once that plan's Task 7 commit is on origin/research/roadless (its subject begins `research: roadless -- consolidated screen-report decides through epistemetry's advance`); merge origin/research/roadless into research/lease-pool first.
    Until then do Tasks 6-8, which touch no file of that plan's.
  - Its Task 8 deletes seqscreen.py and epireplay.py; neither calls clearings, so the order of that task and Task 5 does not matter.
  - Locate every edit to an existing file by function, class or heading name, never by a line number: the move's executor is editing consolidated.py, NOTES.md and BACKLOG.md now.
- **Against the g3 runs:** Task 11 (the migration) waits for cp-roads4-desire-g3a and -g3b to finish and sync (their records are of kind `cpu`); its precondition step checks it.
- **cluster_submit is pinned at ec79a0eb** (research/roadless 335a735d, the owner's "Yes, bump to ec79a0e"): `Campaigns.template(ctx, campaign, node, bundle) -> cs.Template | cs.Done` and `Campaigns.starts(market)`, and no `Campaigns.commit`.
  Read its README "GPU market" and docs/migrating/reblock.md's last "Also" section (`git -C ~/src/cluster_submit show ec79a0eb:<path>`) before Task 6: that section is what `Waves` answers.
- The plan ends with the pool built, guarded and smoke-tested; the held re-run under the paved-only boost (.superpowers/sdd/2026-10-20-paved-boost/brief.md) is its first user and is not part of it.

### Rulings (the spec reconciled with what landed after it)

1. **One campaign, one writer of its demand.**
   reblock states one demand, campaign `reblock` (the owner's ruling via the controller, 2026-10-21: its spend accumulates over the waves, so a new wave never ranks first in fair share).
   Its file has one owner: this machine's demand step (`cluster_demand.state`, after every verb of `cluster.STATES_DEMAND`, the watch timer's 150-s sync among them); a steward never writes a demand.
   Each steward publishes its wave's lines in its `summary.json`, and the step reads the live waves' summaries (one ssh call) beside squeue.
   Why: one campaign is one file, and two writers would each rewrite the other's whole file with what it cannot see (a steward sees neither the registry's throttles of today's kinds, which the step states while they last, nor which runs are whose leases); the spec's reason for the steward (an unfinished run, so the timer keeps syncing while a wave has work: its 3.7) still holds, so the demand is restated every 150 s while any wave is live, well inside the market's 30-minute stale_after; and every continuation and start is that same sync's, so a fresher demand would change no decision reblock makes.
   What it gives up: while this machine does not sync, the demand goes stale and the keeper treats what reblock holds as free, as it does today for every reblock run.
   This overrules the spec's 3.6 step 4, 3.7's "one file per wave" and section 5's "the market-demand writer goes with them": cluster_demand.py stays, generalized.
2. **A pool line's units** are, for each wave and (node, class), the larger of its workers running there and its jobs there pending or claimed, summed over the waves sharing the (node, bundle).
   The controller's ruling on the market-demand branch (cluster_demand's docstring: the keeper treats what a campaign holds and does not demand as free) makes held workers units; the spec's guard 7 ("units are the jobs pending or claimed; a finished wave states 0") holds once its workers have exited.
   `minutes` is the mean of the line's jobs' estimates, or of its running workers' time limits when it has none.
   Long runs and stewards are stated as today's runs are: running tasks held, waiting ones by their throttle.
3. **A CPU class holds a whole core: 2 CPUs.**
   `cluster.py gpu` at 2026-10-21T19:32Z shows g3a and g3b, submitted with 1 CPU, held at 0/2/850 and 0/2/540 (ai's CR_CORE_MEMORY holds both of a core's threads), so a 1-CPU class's demand line would never match what its workers hold, and the market's `held` and step-aside would miss them.
   Jobs stay single-threaded (cluster_env.sh).
   This overrules the spec's "cpu-score (1 CPU)".
4. **Starts from the campaign's template.**
   The market starts every pool lease (ec79a0eb's `Campaigns.template`): for a (node, bundle) line of the campaign's demand that the allocation gives more units than its unfinished leases there hold, `Waves.template` answers the pool of the live wave with the most pending jobs of that class on that node (from the stewards' summaries; ties by wave name), at that wave's commit, named `<w>-<node>-<class>-s<N>` (N the first the registry lacks, so never a run it has nor a continuation's `-l<N>`), or `cs.Done(why)` when no live wave has work of that class there.
   So waves of the one campaign share a node and class freely and run at different commits, and a class reached by promotion is started with nothing submitted by hand (the spec's 3.4 and 3.7, as cluster_submit's docs/migrating/reblock.md says).
   `wave submit` therefore submits no pool: the steward's first summary puts the classes in the next sync's demand, the keeper allocates, and the sync after starts them; `wave <w> pools` hand-submits a first lease per class for a GPU file in shadow mode, where the market starts nothing.
   `Waves.starts(market)` is `tuple(market())`.

5. **Stepping aside across waves.**
   The allocation's `yields` are the campaign's per (node, bundle); each steward asks of its own live workers there its largest-remainder share of them, in proportion to every live wave's workers there as their latest summaries count them (ties by wave name), counting only its own asks since that allocation.
6. **Progress and strikes.**
   `clearings.write_once` prints `ROW_WRITTEN` followed by the row's path for every row it writes, and the worker counts those lines of its child's output into the attempt.
   A strike is an attempt that ends failed, or is reaped before its walltime, with no row written; a walltime kill is no strike (it raises the time estimate), nor is an OOM (it promotes), nor an exit 0 that kept every row.
   So a job that kills its worker (a node fault, a cgroup OOM that takes the worker too) fails for a person after three tries instead of looping (mycooc's lesson 7).
7. **Any process of a wave makes its queue; inputs are shipped once.**
   `wave submit` ships the seed (`jobs.json`: the header and every job), the bank, the tars and the `--with` files by rsync to `$HOME/reblock-waves/<w>/inputs/` before it submits anything.
   The first steward or worker to open the queue makes it from the seed (idempotent: unique on block and item), and under `Fixed` closes it; no worker waits for the steward's first round, and no continuation ships inputs.
   Every run of a wave is submitted `--at` the wave's commit (HEAD when `wave submit` ran), so another agent's uncommitted edits in the main checkout neither block a wave nor reach it.
8. **`Standing` replaces `replace: bool`.**
   `clearings.Standing` (`Refuse`, `Replace`, `Keep`) is resolved once per script by `clearings.standing_of(argv)` from `--replace` or `--keep-standing`; `check_new` and `replace_flag` go, and every caller moves to it in the same commit.
9. **Leases are the preset's length** (`Ladder.lease_minutes`: 120 in both production presets, the active tier's 2 h), the plan's `--time`; the market caps a launched lease at its node's tier's, and a quiet node's 3 h is not used.
10. **Throttles.** A CPU pool lease runs `--gpus CPU_NODES[node]` and a GPU one `--gpus PROJECT.max_gpus` (4, decision 5): `Waves.template` puts it in its args and continuations keep it, so a widened lease runs whole.
11. **The steward runs on orion with 2 CPUs and 3 GB**, a bundle no ladder's rung has (`pool_ladder.STEWARD`, a `cs.Bundle`), so its unit never joins a pool's demand line; its memory is measured in the smoke wave (the spec's question 1).
12. **Requests go through the steward.** `wave <w> retry` and `wave <w> cancel` write a request file in the wave's directory, which the steward applies on its next round; `wave cancel` also cancels every pool run of the wave, and the steward then cancels the wave's pending and claimed jobs, writes the final snapshot and exits 0.
13. **`summary.json` is read exactly** (format 1, every key by name); its format changes only while no wave is live.
14. **Records.** `research/roadless/waves/<w>/` (summary.json, ledger.jsonl, final.db) is git-ignored, as rows are; the next wave's estimators read its `final.db`.
15. **Estimates.**
    A job's stored basis is the most demanding stored run of the same work on the same block (rows carry no finish time to tell the last), after an earlier wave's final queue, which comes first.
    The fit with nothing stored is desire_fit.out's: over the family's jobs on blocks of at least 1 km² (`FIT_MIN_KM2`, where a job's fixed overhead is a small share), the worst per-km² rate times the block's area; on a smaller block, the family's largest peak on blocks under 1 km².
    Taken over every block, the worst rate is a small block's overhead (20 GB/km² on 0.08 km² against 8.8 on the large ones), which would class every large block past every node.

### Code rules (~/.claude/CLAUDE.md)

- No legacy shims: when a format or interface changes, migrate the data and delete the old path.
  The old kinds stay only while their runs are live, and go in one commit (Task 11).
- Strategy resolved upstream: a Protocol and an instance built where the command line is read, passed down; no `if mode == ...` at call sites.
  The name lookups (`LADDERS[name]`, `FAMILIES[name]`) are the command line's, resolved once, with no default (an unknown name is a SystemExit listing the names).
- No `getattr` or `.get` defaults on closed sets; frozen dataclasses (`frozen=True, kw_only=True`) and NamedTuples; every JSON file (`jobs.json`, `summary.json`, a request) read by key at one boundary function, exactly.
- Every parameter that can change a result is required and keyword-only below the edge.
  Nothing in this plan changes a row: ladders, margins, estimates, placements of jobs on nodes within a group and lease lengths change cost only, and are recorded in the queue's header.
  A block's group is its node (results can differ by machine), fixed at submit and recorded.
- Guards need fault injection: a guard returns its failures, each fault is a patch inside the guard's process (gap_clear_check.py's `_patched`), and `both-ways` shows it passes as it is and fails under each fault.
  A guard's inputs are ones production can construct.
- Keep the frontier, delete the dominated.

### Guards, lint and types

- `ENV="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=."`; every guard runs from the repo root as `env $ENV uv run python research/roadless/<check>.py ...`.
- pool_check.py follows screening_check.py's command line: `<guard> [fault]`, `both-ways <guards,>`, a `GUARDS` and a `FAULTS` dict.
  Task 6 gives cluster_check.py the same `both-ways` (its `cpu` check stays a print until Task 11).
- Guards that read stored rows take the rows root as an argument (`<rows>`), because parquet rows are git-ignored and live in the main checkout only: from the worktree pass `/home/gchurchill/src/reblock/research/roadless`.
- Ruff skips research/ (pyproject's `extend-exclude`), so name the paths: every new module passes `uv run ruff check <paths>`; a touched existing module passes `uv run ruff check --select F,E9 <path>` (check it passes before editing).
- Types: `MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent <paths>` reports no error in a new module, nor in cluster.py, cluster_demand.py or cluster_check.py (clean at 335a735d); for another touched module the count of errors does not rise (record it before editing).

### Processes and the cluster

- At most 8 local processes in all; the executor's share is the controller's to give, and pool_check's `claim` guard uses 4.
  Single-threaded numeric pools; long runs through the Bash tool's `run_in_background`; never `pkill -f` (kill by PID); no polling loops (`cluster.py wait <run>`, exit 0 passed, 3 failed, 4 timed out, 5 blind).
- Cluster: never `keeper install`, never `gc --drop-unpulled`, never clean the cluster's uv cache, never cancel cp-roads4-desire-g3a or -g3b, commit before submitting.
  Every submit to the cluster is the controller's to confirm first: the smoke waves (Task 10) and nothing else in this plan.
- Never touch research/roadless/stopping_review/, research/group_steiner/, research/roadless/road_rows/lineup-f8fa4e53/ (beyond reading), research/roadless/road_rows/superseded/, research/roadless/screening_migration.py, or research/roadless/{ids58,ids59,ov16,ov19,ov_a,ov_b,ss_miss}.txt.
  Don't edit src/reblock's Block producers (kblock, region*, derivations*, derive/, buildings.py, budget.py): that colds the derive cache.
- NOTES.md and BACKLOG.md are column-wrapped at 99; new Markdown is one sentence per line.
  Name repositories by their GitHub names (GeoffChurch/cluster_submit, GeoffChurch/mycooc), not local paths.

### Commits

- `git -c core.hooksPath=.githooks commit -m "<subject>" -m "<body>" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RoNpoEMZ1kz2hogCsAsmXJ" -- <paths>`; a new file is `git add`ed first, a deletion `git rm`ed; the hook needs the submodules (`git submodule update --init` if missing).
- Subjects: `research: roadless -- <what, with its numbers>`, the branch's style.
- Commit only your own paths; never push main.

## Review Focus

1. **A long run, or a lease started by hand, starts before anything has made its wave's queue, or before its steward has written a summary**: a worker makes the queue from the seed and runs; `Pool.more` raises (asked again next sync), never Done, with no summary.
   Pinned by Task 3's `worker` guard (a worker on an absent queue.db) and Task 6's `more` guard (no summary).
2. **Two waves sharing a (node, bundle), at different commits**: the demand sums their lines, each steward asks only its share to step aside, and the market's start there goes to the wave with the most pending jobs of the class, at that wave's commit.
   Pinned by Task 8's `demand` guard, Task 4's `aside` guard and Task 6's `template` guard.
3. **A summary of another format, or one older than a lease's end** (the steward between its own leases, or a wave at an older commit after a format change): `Pool.more` raises, and the demand states that wave's running workers alone and says so.
   Pinned by Task 6's `more` guard and Task 8's `demand` guard.
4. **An OOM that takes the worker with its job** (a cgroup that kills the whole step): the child starts at `oom_score_adj` 1000, so the kernel takes the job first; if the worker dies anyway its attempt is reaped as a strike, never looped; the smoke wave forces an OOM on ai.
   Pinned by Task 3's `oom` guard (reads the child's `oom_score_adj`), Task 4's `dead-claim` guard, Task 10 Step 6.
5. **A job that exits 0 having kept every row** (a double claim, or a job run again after a full write): done, no strike, no `Exists`.
   Pinned by Task 9's `double-claim` guard.

## Files

- Create `research/roadless/pool_ladder.py` (stdlib and cluster_submit's `Bundle`): `Ladder`, the presets `LADDERS`, `class_name`, `rung_of`, `STEWARD`.
- Create `research/roadless/pool_queue.py` (stdlib only): the `Queue` Protocol, `SqliteQueue`, the seed's format (`Seed`, `seed_json`, `read_seed`), job states and outcomes.
- Create `research/roadless/pool_plan.py`: estimates (`Estimate`, `Observed`, the `Family` Strategy and `FAMILIES`, `Estimator`, `history_of`, `areas_of`) and `plan_wave`.
- Create `research/roadless/pool_summary.py` (stdlib only): `Summary` and its exact reader and writer, `lines_of`, `aside_share`.
- Create `research/roadless/pool_worker.py`: `run_job`, the `OomEvidence` Strategy, `work`, `work_one`.
- Create `research/roadless/pool_steward.py`: the `Liveness` Strategy, the `Feed` Strategy (`Fixed`), `round_`, `steward`.
- Create `research/roadless/pool.py`: the cluster entry (`work`, `steward`), every Strategy built there.
- Create `research/roadless/cluster_wave.py`: the `wave` verb.
- Create `research/roadless/pool_check.py`: the guards `claim`, `transitions`, `class`, `walltime`, `oom`, `identity`, `worker`, `dead-claim`, `aside`, `end`, `keep`, `double-claim`, and `smoke-job`.
- Modify `research/roadless/cluster.py`: kinds `Pool`, `Steward`, `Long`, the pull `WaveRecord`, `Waves` (Campaigns: `lease`, `template`, `starts`), `PROJECT`, the `wave` dispatch, `state_demand`, the docstring; Task 11 deletes `Blocks` and `Cpu`.
- Modify `research/roadless/cluster_demand.py`: pool lines (Task 8); Task 11 drops `Stated.unplaced`.
- Modify `research/roadless/cluster_check.py`: `both-ways`, guards `kinds`, `more`, `template`, `wave`, `demand`; Task 11 rewrites `cpu`, `ids` and `script` for waves.
- Modify `research/roadless/clearings.py` (`ROW_WRITTEN`, Task 3; `Standing`, Task 5) and clearings_check.py, and every caller of `check_new`, `write_once` or `replace_flag`: consolidated.py, road_rescore.py, arterial_study.py, uniform.py, gap_calib.py, polish_greedy.py, search.py, relax.py, pweight.py, region_leaks.py.
- Modify `research/roadless/.gitignore`: `waves/`.
- Modify `research/roadless/NOTES.md` (Task 10), `research/roadless/BACKLOG.md` and .superpowers/sdd/2026-10-16-consolidated-pass/followup.md (Task 11).

---

### Task 1: The queue

The spec's 3.2 and 3.4's promotion, as one module, after mycooc's `scheduler/jobs.py` (GeoffChurch/mycooc; `git -C ~/src/mycooc show HEAD:scheduler/jobs.py`).

**Files:**
- Create: `research/roadless/pool_ladder.py`, `research/roadless/pool_queue.py`
- Create: `research/roadless/pool_check.py` (its harness, guards `claim` and `transitions`)

**Interfaces:**
- Produces, in pool_ladder.py:
  - `@dataclass(frozen=True, kw_only=True) class Ladder`: `name: str`, `cards: int` (0 or 1), `cpus: int`, `rungs: tuple[int, ...]` (GB, ascending), `lease_minutes: float`, `margin_minutes: float`, `strikes: int`, `idle_minutes: float`, `poll_seconds: float`, `stored_mem: float`, `fit_mem: float`, `stored_time: float`, `about: str`; `bundle(mem_gb: int) -> Bundle` (a worker of that rung: `Bundle(cards=cards, cpus=cpus, mem_gb=mem_gb)`); `__post_init__` refuses (ValueError) rungs not ascending and positive, cards not 0 or 1, a margin not below the lease, and a rung whose bundle is `STEWARD`.
  - `LADDERS: dict[str, Ladder]`, each `about` saying what it optimizes and gives up:
    - `cpu-score`: cards 0, cpus 2, rungs (4, 12, 24, 56, 88), lease 120, margin 10, strikes 3, idle 10, poll 60, stored_mem 1.2, fit_mem 1.5, stored_time 1.25 (Task 7's classes and a 4-GB rung, which 174 of the 565 lineup networks fit; past 88 GB the long lane).
    - `gpu-presets`: cards 1, cpus 4, rungs (16, 32, 64, 96), the rest as cpu-score.
    - `smoke`: cards 0, cpus 2, rungs (2, 4), lease 20, margin 2, strikes 3, idle 3, poll 15, the same memory and time margins (Task 10's waves: a walltime kill in 20 minutes).
  - `STEWARD = Bundle(cards=0, cpus=2, mem_gb=3)` (Ruling 11); `class_name(cards: int, mem_gb: int) -> str` (`g16`, `m12`); `rung_of(gb: float, ladder: Ladder) -> int | None` (the smallest rung holding `gb`, None past the top).
- Produces, in pool_queue.py:
  - `class JobState(StrEnum)`: `PENDING`, `CLAIMED`, `DONE`, `FAILED`, `LONG`, `CANCELLED`; `class Lane(StrEnum)`: `POOL`, `LONG`; `class Outcome(StrEnum)`: `DONE`, `FAILED`, `OOM`, `REAPED`, `WALLTIME`, `RETIRED`, `CANCELLED`.
  - `@dataclass(frozen=True, kw_only=True) class Header`: `wave`, `commit`, `family`, `ladder: Ladder`, `created: str` (cluster_submit's STAMP), `feed: str` (`"fixed"`).
  - `@dataclass(frozen=True, kw_only=True) class JobSeed`: `id: int`, `block`, `item`, `group`, `line`, `node`, `cls`, `gb: float` (needed: basis times margin), `minutes: float` (needed), `peak_gb: float` (the basis), `source: str`, `lane: Lane`.
  - `@dataclass(frozen=True, kw_only=True) class Seed`: `header: Header`, `jobs: tuple[JobSeed, ...]`, `closed: bool`; `seed_json(seed: Seed) -> str`, `read_seed(text: str, where: str) -> Seed` (exact; `format` 1).
  - `class Job(NamedTuple)`: the seed's fields plus `state: JobState`, `strikes: int`, `worker: str | None`; `class Claim(NamedTuple)`: `job: Job`, `attempt: int`; `class Worker(NamedTuple)`: `id`, `run`, `node`, `cls`, `ends_at: float`, `retired: bool`; `class Reaped(NamedTuple)`: `worker`, `run`, `job: int | None`, `outcome: Outcome | None`.
  - `class Queue(Protocol)` with: `header() -> Header`; `closed() -> bool`; `close() -> None`; `metric() -> str | None`; `name_metric(name: str) -> None`; `register(*, worker: str, run: str, node: str, cls: str, ends_at: float) -> None`; `retire(worker: str) -> int`; `claim(*, worker: str, minutes_left: float) -> Claim | None`; `claim_job(*, worker: str, job: int) -> Claim | None`; `wrote(attempt: int) -> None`; `finish(attempt: int, *, worker: str, outcome: Outcome, peak_gb: float, minutes: float, rc: int) -> JobState`; `reap(worker: str, *, now: float) -> Reaped`; `ask_aside(*, node: str, cls: str, units: int, since: float) -> list[str]`; `asked_aside(worker: str) -> bool`; `retry(ids: Sequence[int]) -> int`; `cancel() -> int`; `jobs() -> list[Job]`; `workers() -> list[Worker]`; `backup(path: Path) -> None`.
  - `class SqliteQueue` implementing it, `SqliteQueue.open(path: Path, seed: Seed) -> SqliteQueue`: makes the tables if absent and grants the seed's jobs once (unique on block and item; a seed of another wave or commit than the header already there is refused, ValueError), closing the queue when `seed.closed`.

- [ ] **Step 1: Write the harness and the guard `claim`**

pool_check.py's command line as screening_check.py's (`main`, `both_ways`, `GUARDS`, `FAULTS`), with gap_clear_check.py's trailing arguments: `<guard> [fault] [args ...]` and `both-ways <guards,> [args ...]`, the args passed to every guard named (Task 2's `class` takes the rows root).
`claim`: a 200-job queue (one node, one class, estimates 1-200 minutes) in a temporary directory on local disk; 4 processes (`multiprocessing`, fork) each register a worker and claim (`minutes_left` 1000: every job fits; the fit is `walltime`'s), finish `DONE` and claim again until nothing is left, with a 0-10 ms random sleep between operations.
Failures: a job with two attempts, a job not `DONE`, an attempt finished by a worker other than its claimer, and a worker whose claims' estimates ever rise (each claim the longest pending at its time, so each worker's sequence never rises).
Fault `split-tx`: `SqliteQueue.claim` patched to select in one transaction and update in a second, sleeping 5 ms between: some job is claimed twice.

- [ ] **Step 2: Write the guard `transitions`**

On a local queue under `cpu-score` (lease 120, margin 10, strikes 3), each case a fresh queue, the job's state, class, strikes, `gb` and `minutes` after each event as listed:
- `finish DONE` -> DONE.
- `finish FAILED` with two `wrote` -> PENDING, strikes 0; with none, three times -> PENDING (1), PENDING (2), FAILED (3).
- `finish OOM` with peak 10 from class `m4` -> PENDING in `m12`, `peak_gb` 10; from `m88` with peak 90 -> LONG, lane LONG, `gb` 135 (1.5 x 90).
- `reap` of a worker holding a job, `now` before its `ends_at`: with a row written -> PENDING, strikes 0; without -> strikes 1 (Ruling 6); `now` after `ends_at`, the attempt having run 50 minutes on a job estimated 30 -> PENDING with `minutes` 50, outcome WALLTIME, strikes 0; having run 110 -> LONG (past `fresh_minutes`, 120 - 10 - 5).
- a reaped worker's late `finish DONE` -> the attempt recorded, the job left as the reaper left it (Spec 3.2, "Done").
- `retry` of a FAILED job -> PENDING, strikes 0; `cancel` -> every PENDING and LONG job CANCELLED, CLAIMED ones untouched.
- `claim_job` of a LONG job -> CLAIMED; its `finish FAILED` without rows -> LONG, strikes 1.
- `ask_aside(units=2)` on a node with three live workers of the class, one idle -> the idle one first, then the one whose job ends soonest; asked again with the same `since` -> none.
Faults: `late-finish` (`finish` ignores whose claim it is: the reaped worker's late DONE marks the job done), `drop-on-top` (past the top rung the job is FAILED, not LONG: decision 4 and the spec's "OOM promotes; it never drops").

- [ ] **Step 3: Run both guards to see them fail**

Run: `env $ENV uv run python research/roadless/pool_check.py claim`
Expected: an ImportError or AttributeError naming pool_queue (nothing exists yet).

- [ ] **Step 4: Implement pool_ladder.py and pool_queue.py**

As mycooc's `Queue` (read it first): SQLite in `PRAGMA journal_mode=DELETE`, `sqlite3.connect(path, timeout=120, isolation_level=None)`, each operation its own connection in one `BEGIN IMMEDIATE`, timestamps for reading only.
Tables: `header` (one row: the header's JSON and the metric), `jobs`, `attempts` (job, worker, node, cls, start, end, outcome, peak_gb, minutes, rc, rows), `workers`, `aside`, `closed`.
`claim`: the worker's node and class from `workers` (registered, not retired; else ValueError), the PENDING job of that node and class with `minutes <= minutes_left`, by `minutes DESC, id`; it becomes CLAIMED with a new attempt.
`ask_aside` and `asked_aside` are mycooc's, restricted to the class.

- [ ] **Step 5: Run both ways**

Run: `env $ENV uv run python research/roadless/pool_check.py both-ways claim,transitions`
Expected: `claim: passes`, `transitions: passes`; `claim under the fault split-tx: FAILS on ...`, `transitions` under `late-finish` and `drop-on-top` each `FAILS on ...`; `both ways, claim, transitions: as expected`.

- [ ] **Step 6: Lint, types, commit, push**

Run the ruff and mypy gates on pool_ladder.py, pool_queue.py and pool_check.py.
Subject: `research: roadless -- the lease pool's queue (pool_queue.py, behind a Queue Protocol): SQLite in DELETE journal mode, one IMMEDIATE transaction per operation (mycooc's lesson 1), claims of the longest job that fits the walltime left, done only by the claim's holder, OOM promotion a rung up and past the top to the long lane, strikes by rows written, the walltime kill raising the estimate; the ladders cpu-score, gpu-presets and smoke (pool_ladder.py, a CPU class a whole core); guards claim (4 processes, 200 jobs; fault split-tx) and transitions (faults late-finish, drop-on-top)`
Paths: `research/roadless/pool_ladder.py research/roadless/pool_queue.py research/roadless/pool_check.py`

---

### Task 2: Classes and estimates

The spec's 3.4, with Ruling 15.

**Files:**
- Create: `research/roadless/pool_plan.py`
- Modify: `research/roadless/pool_check.py` (guard `class`)

**Interfaces:**
- Consumes: `Ladder`, `rung_of`, `class_name` (Task 1); `JobSeed`, `Seed`, `Header`, `Lane`, `read_seed` and the queue's tables (Task 1, for `history_of`).
- Produces:
  - `class Observed(NamedTuple)`: `block`, `item`, `peak_gb: float`, `minutes: float`, `source: str` (`rows:<path>` or `wave:<w>`).
  - `class Estimate(NamedTuple)`: `peak_gb: float`, `minutes: float` (the bases), `mem_margin: float`, `time_margin: float`, `source: str`; methods `gb() -> float` and `needed_minutes() -> float` (basis times margin).
  - `class Family(Protocol)`: `name: str` property; `stored(block: str, item: str) -> Observed | None` (the most demanding stored run of this work on that block: Ruling 15); `observed() -> Sequence[Observed]` (every finished job of this work on record, for the fit).
  - The families, each a frozen dataclass with a required `root: Path` (research/roadless of the checkout whose rows it reads):
    - `Baselines`: `consolidated_rows/baselines/*/<id>.parquet`, item the row's `part`, peak `peak_rss_gb`, minutes `(t_build + t_mesh + t_model) / 60`.
    - `Onecard(device: str)`: `consolidated_rows/onecard/*/<device>/<id>.parquet`, item `""`, `peak_rss_gb`, `wall / 60`.
    - `Lenses`: the block's `runtime` baseline as its basis (the same mesh and model: the spec's section 7), item `""`.
    - `LineupScore`: every `road_rows/**/area_*/<id>__<item>.parquet` (the superseded included), item the method, `peak_rss_gb`, `t_total / 60`.
    - `Proposals`: every `road_rows/**/proposals/<id>__<item>.parquet` with `status == "ok"`, item the method, `peak_rss_gb`, `total_wall_s / 60`.
    - `Smoke`: a table by item for Task 10, the same for every block and taken as stored (`quick` 0.5 GB 1 min, `big` 3 GB 1 min, `oom` 1 GB 1 min, `slow` 0.5 GB 3 min, `gpu` 2 GB 2 min), source `smoke`; `observed()` is empty (no fit).
  - `FAMILIES: dict[str, Callable[[Path], Family]]`: `baselines`, `presets-gpu`, `presets-cpu`, `lenses`, `lineup-score`, `propose`, `smoke`.
  - `FIT_MIN_KM2 = 1.0` (Ruling 15; desire_fit.out's choice).
  - `@dataclass(frozen=True, kw_only=True) class Estimator`: `family: Family`, `ladder: Ladder`, `areas: Mapping[str, float]` (km²), `history: Sequence[Observed]`, `fit_min_km2: float`; `estimate(block: str, item: str) -> Estimate`: the newest wave's observation of (block, item) in `history`, else `family.stored(block, item)` (margins `stored_mem`, `stored_time`), else the fit (margins `fit_mem`, 1.0); a block with no area and nothing stored is a SystemExit naming it.
  - `history_of(waves: Path, family: str) -> list[Observed]`: each `waves/<w>/final.db` whose header's family is `family`, its jobs' last DONE attempts, newest wave first by `created`.
  - `areas_of(features_csv: Path, bank: Mapping[str, object]) -> dict[str, float]`: `summary_features.csv`'s `area_m2` by block, then each bank block's `boundary.area` for blocks the CSV lacks (5810@major), in km².
  - `plan_wave(*, wave: str, commit: str, family: str, ladder: Ladder, estimator: Estimator, nodes: Sequence[str], ids: Sequence[str], items: Sequence[str], line: Callable[[str, str], str], created: str) -> Seed`.

- [ ] **Step 1: Write the guard `class`**

`pool_check.py class <rows>`:
- On `LineupScore(rows)` under `cpu-score`: for every stored (block, method), the planned `cls` is `class_name(0, r)` for `r` the smallest rung holding 1.2 times the most demanding stored peak, or the job is LONG past 88 GB; its needed minutes are 1.25 times the stored wall.
- On `Proposals(rows)` items `cycle_native_betweenness_contrast`, leave-one-out over its jobs on blocks of at least 1 km²: the estimate with the block's own rows removed equals 1.5 x (the worst peak per km² among the others) x its area, recomputed here (rel 1e-9), and holds the block's measured peak.
  The tightest is 62803 (164.7 GB on 11.90 km², 13.8 GB/km², the worst rate on record since the spec: it finished after desire_fit.out), held by 53556's 10.9 GB/km².
- `plan_wave` on 12 stored lineup blocks over `nodes=("orion", "quadro2")`: every job of a block on one node; the nodes' summed needed minutes within the largest group's of each other (longest-processing-time first); a job whose needed minutes pass `lease_minutes - margin_minutes` has lane LONG.
Faults: `median-rate` (the fit's rate the family's median per-km² rate: the recomputation fails) and `no-fit-margin` (the fit without its 1.5: 62803 classed below its measured peak).

- [ ] **Step 2: Run it to see it fail**

Run: `env $ENV uv run python research/roadless/pool_check.py class /home/gchurchill/src/reblock/research/roadless`
Expected: ImportError naming pool_plan.

- [ ] **Step 3: Implement pool_plan.py**

`plan_wave`: one job per (id, item) (`items == [""]` for a command without `{item}`), its id its index in that order from 1; `group` the block; groups to nodes by longest-processing-time first on needed minutes (ties by node order); `cls` from `rung_of(estimate.gb(), ladder)`; lane LONG when the needed minutes pass `pool_queue.fresh_minutes(ladder)` (lease less margin less startup: 105 in the production presets) or the gb passes the top rung, with `gb` then `1.5 * peak_gb`; `closed=True` (the `Fixed` feed); `source` the estimate's.
pool_plan.py reads rows with pandas; nothing on the cluster imports it.

- [ ] **Step 4: Run both ways**

Run: `env $ENV uv run python research/roadless/pool_check.py both-ways class /home/gchurchill/src/reblock/research/roadless` (give `both-ways` the guard's arguments after the guard list).
Expected: `class: passes`; under `median-rate` and `no-fit-margin`, `FAILS on ...`; `as expected`.

- [ ] **Step 5: Lint, types, commit, push**

Subject: `research: roadless -- the pool's classes and estimates (pool_plan.py): a job's class the smallest rung holding 1.2 times its most demanding stored peak (an earlier wave's final queue first), or with none stored the family's worst per-km² rate over blocks of at least 1 km² times 1.5 (desire_fit.out's fit; a small block's overhead would class every large block past every node), time 1.25 times the stored wall; the families baselines, presets-gpu/cpu, lenses, lineup-score, propose and smoke; a block's jobs on one node, groups balanced by needed minutes; guard class (faults median-rate, no-fit-margin; 62803's cycle_desire the tightest leave-one-out)`
Paths: `research/roadless/pool_plan.py research/roadless/pool_check.py`

---

### Task 3: The worker

The spec's 3.5, Rulings 6 and 7, and decision 9 (the OOM evidence probed: here resolved from `/proc/self/cgroup` at each worker's start, confirmed on ai in Task 10).

**Files:**
- Create: `research/roadless/pool_worker.py`, `research/roadless/pool.py` (the `work` command)
- Modify: `research/roadless/clearings.py` (`ROW_WRITTEN`, printed by `write_once`)
- Modify: `research/roadless/pool_check.py` (guards `walltime`, `oom`, `identity`, `worker`)

**Interfaces:**
- Consumes: `Queue`, `SqliteQueue.open`, `read_seed`, `Outcome`, `JobState` (Task 1).
- Produces:
  - `clearings.ROW_WRITTEN = "[row] wrote "`; `write_once` prints it and the path, flushed, after each row it writes.
  - `class OomEvidence(Protocol)`: `kills() -> int | None` (the cgroup's OOM-kill count now; None when it cannot be read) and `about: str`; `CgroupV2(path: Path)` (`memory.events`'s `oom_kill`), `CgroupV1(path: Path)` (`memory.oom_control`'s `oom_kill`), `Unreadable(why: str)`; `oom_evidence(proc_self_cgroup: str, root: Path) -> OomEvidence` (v2: the `0::` line; v1: the `memory` controller's line; else `Unreadable`).
  - `class Ran(NamedTuple)`: `rc: int` (the shell's code, `-signal` when killed), `peak_gb: float` (`os.wait4`'s `ru_maxrss`), `minutes: float`, `rows: int`.
  - `run_job(line: str, *, cwd: Path, env: Mapping[str, str], log: Callable[[str], None], wrote: Callable[[], None]) -> Ran`: `bash -c <line>` with stdout and stderr merged and streamed to `log` line by line between job markers, the child's `oom_score_adj` set to 1000 before it execs (an unprivileged process may raise its own), `wrote()` called for each line starting with `clearings.ROW_WRITTEN`.
  - `class Exit(IntEnum)`: `DRAINED = 0`, `OWN_FAULT = 1`, `RETIRED = 75`.
  - `work(*, queue: Queue, worker: str, run: str, node: str, cls: str, commit: str, ends_at: float, now: Callable[[], float], wait: Callable[[], None], run_line: Callable[[Claim], Ran], oom: OomEvidence, log: Callable[[str], None]) -> Exit`.
  - `work_one(*, queue: Queue, worker: str, run: str, job: int, commit: str, ends_at: float, now: Callable[[], float], run_line: Callable[[Claim], Ran], oom: OomEvidence, log: Callable[[str], None]) -> Exit` (the long lane: claim that job, run it, record it; 0 when done, 1 otherwise).
  - pool.py `work --wave <w> --node <n> --class <c>` and `work --wave <w> --job <id>`: the queue `SqliteQueue.open(<home>/reblock-waves/<w>/queue.db, read_seed(<inputs>/jobs.json))`; the worker `<SLURM_ARRAY_JOB_ID>_<SLURM_ARRAY_TASK_ID>`, its run `$CLUSTER_SUBMIT_RUN`, its end `$SLURM_JOB_END_TIME`, its commit the checkout manifest's (`$CLUSTER_SUBMIT_CHECKOUT/.cluster_submit_manifest.json`, key `commit`); the child's env the worker's with `CLUSTER_SUBMIT_INPUTS=<wave>/inputs` and `REBLOCK_BLOCK_BANK=<wave>/inputs/bank.pkl`.

- [ ] **Step 1: Write the guards**

Each on a local queue in a temporary directory, with real child processes (`bash -c` lines running `python -c ...` or `sleep`), a fake clock where the walltime matters:
- `walltime` (the spec's guard 4): with 40 minutes left and margin 10, a queue of jobs estimated 20, 29, 31 and 60 minutes: it runs 29 then 20, never 31 or 60, and exits 75 with them pending.
  Fault `no-fit`: `SqliteQueue.claim` patched to ignore `minutes_left`.
- `oom` (the spec's guard 6, its local half): a child that kills itself with SIGKILL while a fake `OomEvidence` reads one more kill than before: the job is promoted (Task 1's transitions), the worker goes on; the same with the count unchanged: a FAILED attempt, a strike; an `Unreadable` evidence: a strike, the log saying why; the child's `/proc/self/oom_score_adj` (printed by the line) is 1000.
  Fault `fail-on-oom`: the worker records an OOM as FAILED.
- `identity` (the spec's guard 9): a header naming another commit than the worker's: exit 1 before any claim, no attempt in the queue.
  Fault `no-identity`: the check patched out: it runs the jobs.
- `worker`: on a queue.db that does not exist yet the worker makes it from the seed and runs every job (Review Focus 1); a job that fails does not end the worker (the next job runs; exit 0 drained); three different jobs failing in a row: exit 1; a job that prints two `ROW_WRITTEN` lines and exits 1: no strike; an open queue with nothing pending: exit 75 after `idle_minutes`; asked aside between jobs: exit 75 before the next claim.
  Fault `fail-worker`: a failed job ends the worker with exit 1.

- [ ] **Step 2: Run them to see them fail**

Run: `env $ENV uv run python research/roadless/pool_check.py walltime`
Expected: ImportError naming pool_worker.

- [ ] **Step 3: Implement**

The loop is the spec's 3.5 steps 1-5 with Ruling 7 (the queue made from the seed by whoever opens it first), after mycooc's `scheduler/worker.py`'s `work`.
`run_job`: `subprocess.Popen(["bash", "-c", line], stdout=PIPE, stderr=STDOUT, text=True, bufsize=1, preexec_fn=<write "1000" to /proc/self/oom_score_adj>)`, read lines to EOF, then `os.wait4(pid, 0)` for the status and `ru_maxrss` (KiB; set the Popen's `returncode` so it never waits again).
OOM classification: the evidence's count read before and after the child; an increase is OOM; a SIGKILL without one is FAILED (decision 9).
Job markers in the log: `=== job <id> <block> <item> start` and `=== job <id> <outcome> rc <rc> <minutes> min peak <gb> GB rows <n>`.
No SIGTERM handler: a self-retirement on SIGTERM needs all three legs of the wiki's "Self-retiring claim on walltime" or is worse than none (the spec's 3.2); the walltime fit and the reaper cover it.
The worker logs, once at start, the evidence it resolved (`about`).
pool.py builds every Strategy where it reads its command line and imports nothing that needs a GPU.

- [ ] **Step 4: Run both ways**

Run: `env $ENV uv run python research/roadless/pool_check.py both-ways walltime,oom,identity,worker`, and again `both-ways claim,transitions` (the queue unchanged).
Expected: each passes, each fault `FAILS on ...`, `as expected`.

- [ ] **Step 5: Lint, types, commit, push**

clearings.py takes the `--select F,E9` gate and the mypy count rule.
Subject: `research: roadless -- the pool's worker (pool_worker.py, pool.py work): claims the longest job that fits its walltime less the margin, runs it as a child at oom_score_adj 1000 (the cgroup's OOM killer takes the job, not the worker), counts its rows from clearings.ROW_WRITTEN lines, promotes it on the cgroup's OOM count (v2 memory.events, v1 memory.oom_control; unreadable: a strike), never fails on a failed job, retires with 75 when nothing fits, when idle or when asked aside, refuses a queue of another commit; the long lane's work --job; guards walltime (no-fit), oom (fail-on-oom), identity (no-identity), worker (fail-worker)`
Paths: `research/roadless/pool_worker.py research/roadless/pool.py research/roadless/clearings.py research/roadless/pool_check.py`

---

### Task 4: The steward and the summary

The spec's 3.6 (with Ruling 1: no demand written), Rulings 5, 6 and 12, and mycooc's `scheduler/liveness.py`.

**Files:**
- Create: `research/roadless/pool_summary.py`, `research/roadless/pool_steward.py`
- Modify: `research/roadless/pool.py` (the `steward` command)
- Modify: `research/roadless/pool_check.py` (guards `dead-claim`, `aside`, `end`)

**Interfaces:**
- Consumes: the queue (Task 1), `Exit` (Task 3); `cluster_submit.gpu.read_allocation`, `mode_named`, `Allocation.yields`; `cluster_submit.context.STAMP`.
- Produces, in pool_summary.py:
  - `SUMMARY_FORMAT = 1`; `class WaveState(StrEnum)`: `OPEN`, `CLOSED`, `FINISHED`, `CANCELLED`.
  - `class ClassLine(NamedTuple)`: `node`, `cls`, `cards: int`, `cpus: int`, `mem_gb: int`, `pending: int`, `claimed: int`, `minutes: float` (mean needed minutes of the pending and claimed jobs, 0 with none), `long: int`, `done: int`, `failed: int`, `workers: int` (live).
  - `class ReapedWorker(NamedTuple)`: `task: str`, `run: str`, `held_job: bool`, `walltime: bool`.
  - `class LongJob(NamedTuple)`: `job: int`, `block`, `item`, `node`, `mem_gb: int`, `minutes: float`.
  - `@dataclass(frozen=True, kw_only=True) class Summary`: `wave`, `commit`, `at: str` (STAMP), `round: int`, `state: WaveState`, `metric: str | None`, `lines: tuple[ClassLine, ...]`, `reaped: tuple[ReapedWorker, ...]` (every worker reaped in the wave), `long: tuple[LongJob, ...]` (the LONG jobs, waiting for a long run).
  - `summary_json(s: Summary) -> str` (one line); `read_summary(text: str, where: str) -> Summary` (exact: another format or a missing key is a ValueError naming `where`).
  - `lines_of(jobs: Sequence[Job], workers: Sequence[Worker], ladder: Ladder) -> tuple[ClassLine, ...]`: one line per (node, class) with a job or a live worker, the bundle `(ladder.cards, ladder.cpus, rung)`.
  - `aside_share(*, yields: int, workers: Mapping[str, int], wave: str) -> int` (Ruling 5).
- Produces, in pool_steward.py:
  - `class Liveness(Protocol)`: `dead(workers: Sequence[str]) -> frozenset[str]`; `Squeue(run, user, own)`: mycooc's `SlurmWorkers` (`squeue -h -r -u <user> -o %i`, a minute's timeout; a failed squeue, or an answer without `own`, says nothing).
  - `class Feed(Protocol)`: `round(queue: Queue) -> None`; `Fixed` (nothing after the seed: the queue is closed from the start; `Screened` waits for the re-run: decision 7).
  - `round_(*, queue: Queue, wave_dir: Path, waves: Path, liveness: Liveness, feed: Feed, allocation: Callable[[], Allocation | None], stale_after: timedelta, now: Callable[[], datetime], log: Callable[[str], None]) -> Summary`: requests, reap, aside, feed, summary (written whole: a temporary name, then a rename), the ledger's line (`ledger.jsonl`, appended).
  - `steward(*, queue: Queue, wave_dir: Path, waves: Path, liveness: Liveness, feed: Feed, allocation: Callable[[], Allocation | None], stale_after: timedelta, now: Callable[[], datetime], log: Callable[[str], None], ends_at: float, poll_seconds: float) -> Exit`: rounds every `poll_seconds` (60); exit 75 when two rounds no longer fit before `ends_at`; when every job is DONE, FAILED or CANCELLED: the queue's backup to `final.db`, the bank and tars deleted from `inputs/`, a last summary (FINISHED, or CANCELLED after a cancel), exit 0.
  - pool.py `steward --wave <w> --board <path> --stale-after-minutes <m>`: the allocation read from `<board>/gpu/allocation.json` each round (a relative board is below `$HOME`, as the login node reads it) (None when absent or unreadable: it asks nobody); the header's metric named once at the first round (`import metrics; metrics.RUNTIME` at the commit: the spec's 3.11).
  - Requests: `<wave>/requests/<stamp>.json`, `{"retry": [<job id>, ...]}` or `{"cancel": true}`, applied in name order and deleted, each recorded in the ledger.

- [ ] **Step 1: Write the guards**

- `dead-claim` (the spec's guard 3): two live workers, one holding a job and SIGKILLed (a real child process), squeue stubbed to omit its task and to list the steward's own: after a round the job is PENDING with one strike (it wrote no row: Ruling 6, Review Focus 4), the summary's `reaped` names it with `held_job` True; a worker past its `ends_at` holding a job: `walltime` True, the job's minutes raised (Task 1); a squeue that fails, or answers without the steward's own task: nothing reaped.
  Faults: `by-age` (claims older than an hour reaped: a live worker's long job reaped), `blind` (a failed squeue read as an empty answer: every worker reaped).
- `aside` (Ruling 5, Review Focus 2): two waves' queues and directories under one temporary `waves`, sharing (quadro2, `m4` of cpu-score), with 4 and 2 live workers, a hand-made active allocation of `yields` 3 there: wave a asks 2, wave b 1; a second round under the same allocation asks none; an allocation older than `stale_after`, or one in shadow mode, asks none.
  Fault `whole-yields` (each steward asks all the yields of its own workers: 3 and 2).
- `end`: a wave whose last job finishes: `final.db` holds the queue's jobs and attempts, `inputs/bank.pkl` and the tars are gone, the summary FINISHED, exit 0; a `retry` request takes a FAILED job back to PENDING; a `cancel` request: every pending and claimed job CANCELLED, the final snapshot written, exit 0; a LONG job waiting: no end.
  Fault `early-end` (the steward ends with a LONG job waiting).

- [ ] **Step 2: Run them to see them fail**

Run: `env $ENV uv run python research/roadless/pool_check.py dead-claim`
Expected: ImportError naming pool_steward.

- [ ] **Step 3: Implement pool_summary.py and pool_steward.py, and pool.py's `steward`**

Ask aside as mycooc's executor does: `cs.gpu.mode_named(a.mode).aside(lambda: a.yields("reblock/reblock"))`, each (node, bundle) mapped to this wave's class there, the share from `aside_share` over every live wave's latest summary under `waves` (`*/summary.json`, `state` OPEN or CLOSED), then `queue.ask_aside(node=..., cls=..., units=share, since=<the allocation's at>)`.
A summary of another wave that does not read is left out of the shares and said in the ledger.

- [ ] **Step 4: Run both ways**

Run: `env $ENV uv run python research/roadless/pool_check.py both-ways dead-claim,aside,end`, and `both-ways claim,transitions,walltime,oom,identity,worker`.
Expected: all as expected.

- [ ] **Step 5: Lint, types, commit, push**

Subject: `research: roadless -- the pool's steward (pool_steward.py, pool.py steward): every minute it reaps on squeue's evidence alone (abstaining when squeue fails or does not list it: mycooc's lesson 15), asks workers to step aside for its share of the campaign's yields among the waves holding there, applies retry and cancel requests, writes summary.json (pool_summary.py, read exactly) and its ledger; at the wave's end the final queue, the bank and tars deleted, exit 0; it writes no demand (this machine's step does: one campaign); guards dead-claim (by-age, blind), aside (whole-yields), end (early-end)`
Paths: `research/roadless/pool_summary.py research/roadless/pool_steward.py research/roadless/pool.py research/roadless/pool_check.py`

---

### Task 5: `Standing`: refuse, replace or keep a row that stands

The spec's 3.3 and decision 4, as Ruling 8.
**Precondition:** the epistemetry move's Task 7 commit is on origin/research/roadless: `git fetch` and `git log --format=%s origin/research/roadless | grep -c "consolidated screen-report decides through epistemetry"` prints 1 or more; then `git merge origin/research/roadless` into research/lease-pool.

**Files:**
- Modify: `research/roadless/clearings.py` (`Standing`, `Refuse`, `Replace`, `Keep`, `standing_of`, `KEEP_STANDING`; `check_new` and `replace_flag` deleted), `research/roadless/clearings_check.py` (guard `overwrite`)
- Modify every caller of `check_new`, `write_once` or `replace_flag` (`git grep -n -E 'check_new|write_once|replace_flag|replace=' -- research/roadless/*.py` lists them): consolidated.py (`features`, `baselines`, `presets`, `lenses`, `screen`, `screen_lenses`, `main`), road_rescore.py (`cx_network`, `cx_plain`, `_cut_and_score`, `lineup_one`, `lineup_score`, `cx`, the `__main__` block), arterial_study.py, uniform.py, gap_calib.py, polish_greedy.py, search.py, relax.py, pweight.py, region_leaks.py.

**Interfaces:**
- Produces, in clearings.py:
  - `class Standing(Protocol)`:
    - `todo(path: Path) -> bool`: whether the identity at `path` is to be run (Refuse: True, or `Exists` when a row stands; Replace: True; Keep: True unless a row stands);
    - `reuse(path: Path) -> bool`: whether a standing row is read rather than made again where a caller may reuse one (road_rescore.cx_plain's presets): Refuse and Keep: a row stands; Replace: False;
    - `write(df: pd.DataFrame, path: Path) -> None`: Refuse: `write_once` (an `Exists` raised); Replace: a temporary name, then `os.replace`, then `ROW_WRITTEN`; Keep: `write_once`, an `Exists` (a double claim's) printed as `[row] kept <path>` and not raised.
  - `Refuse()`, `Replace()`, `Keep()`, frozen dataclasses; `KEEP_STANDING = "--keep-standing"`; `standing_of(argv: list[str]) -> tuple[list[str], Standing]`: the argv less `--replace` and `--keep-standing`, and the Strategy (neither: `Refuse()`; both: SystemExit).
  - Every function that took `replace: bool` takes `standing: Standing`, keyword-only and required, and passes it down; consolidated `features` writes through `Replace()` as it did with `replace=True`.

- [ ] **Step 1: Extend the guard `overwrite` first**

clearings_check.py `overwrite` keeps its cases (each optimizer's second run refuses before any work and leaves the first row's bytes; with `Replace()` writes a new row in its place), each now run through `standing_of`.
New cases: with `Keep()`, a row that stands is not run again and keeps its bytes, an identity without one is written; a `Keep().write` racing a row written meanwhile (written between `todo` and `write`) leaves the first row's bytes and raises nothing; `standing_of(["--replace", "--keep-standing"])` is a SystemExit.
Its fault `no-check` stays (`Refuse.todo` a no-op and the exclusive link an overwrite); new fault `keep-raises` (`Keep.write` raises the `Exists`).

- [ ] **Step 2: Run it to see it fail**

Run: `env $ENV uv run python research/roadless/clearings_check.py overwrite`
Expected: ImportError or AttributeError naming `standing_of`.

- [ ] **Step 3: Implement `Standing` and move every caller**

Each script's `main` (or `__main__` block) calls `clearings.standing_of` once where `replace_flag` was; each function threads `standing` where it threaded `replace`.
Where a caller checks every identity before any work (consolidated `presets`: each preset and the one-card row), it runs only those `todo` answers True for, and returns at once when none is; `presets`' one-card row names in `presets` only the presets this run ran (the timing is the run's own: Task 9 guards it).
No behaviour of `Refuse` or `Replace` changes: a person's run is as before.

- [ ] **Step 4: Run the guards that cover the callers**

Run: `env $ENV uv run python research/roadless/clearings_check.py overwrite`, then with each fault (`no-check`, `keep-raises`); `env $ENV uv run python -c "import sys; sys.path.insert(0, 'research/roadless'); import consolidated, road_rescore, arterial_study, uniform, gap_calib, polish_greedy, search, relax, pweight, region_leaks"`; `git grep -n -E 'check_new|replace_flag|replace: bool' -- research/roadless` prints nothing.
Expected: `overwrite: passes`, each fault `FAILS on ...`; the imports succeed.

- [ ] **Step 5: Lint, types, commit, push**

Every touched module: `ruff check --select F,E9` and the mypy count rule.
Subject: `research: roadless -- clearings.Standing replaces replace: bool (Refuse, a person's run as before; Replace, --replace; Keep, --keep-standing, the pool's: a row that stands is that identity done, at the check and at the write, where a double claim's Exists means the same), resolved once per script by standing_of; check_new and replace_flag deleted, every caller moved (consolidated, road_rescore, arterial_study, uniform, gap_calib, polish_greedy, search, relax, pweight, region_leaks); guard overwrite gains Keep (fault keep-raises)`
Paths: `research/roadless/clearings.py research/roadless/clearings_check.py` and every module the grep named.

---

### Task 6: The kinds, the campaign's leases and starts, and `more`

The spec's 3.7 (its `more` and starts), 3.10, 3.11, with Rulings 2, 4, 9, 10, 11 and 13, on cluster_submit ec79a0eb's `Campaigns` (read its docs/migrating/reblock.md's last section first).

**Files:**
- Modify: `research/roadless/cluster.py` (kinds `Pool`, `Steward`, `Long`; pull `WaveRecord`; `Waves`; `pool_more`; `PROJECT`; constants `WAVES`, `CAMPAIGN`, `STEWARD_NODE`; the docstring's new paragraphs)
- Modify: `research/roadless/cluster_check.py` (the `both-ways` harness; guards `kinds`, `more` and `template`)
- Modify: `research/roadless/.gitignore` (`waves/`)

**Interfaces:**
- Consumes: `LADDERS`, `class_name`, `STEWARD` (Task 1); `read_summary`, `Summary`, `WaveState` (Task 4); `cs.Template`, `cs.Done`, `cs.launches.Start`, `cs.launches.line_checked`, `cs.gpu.chain_base` (ec79a0eb).
- Produces, in cluster.py:
  - `WAVES = "reblock-waves"` (below the cluster home), `CAMPAIGN = "reblock"` (= `PROJECT.name`), `STEWARD_NODE = "orion"`.
  - `Pool` (name `pool`, routine exits {75}, pull `IntoTree("research/roadless", ("*.parquet",))`): arguments `--wave`, `--preset` (a name of `LADDERS`), `--node`, `--mem` (a rung of the preset); its plan one task `python -u research/roadless/pool.py work --wave <w> --node <n> --class <class_name(cards, mem)>`, CPU only for cards 0, else a card (`gpu_gb` 8: the largest GPU peak stored is 7.6 GB), `Resources(cpus=ladder.cpus, mem_gb=mem, time=<lease_minutes as H:MM:SS>, node=node)`, no inputs, prelude `source research/roadless/cluster_env.sh`.
  - `Steward` (name `steward`, routine exits {75}, pull `WaveRecord`): arguments `--wave`, `--board`; one task `python -u research/roadless/pool.py steward --wave <w> --board <b> --stale-after-minutes <PROJECT.gpu_market's>`, `Resources(cpus=2, mem_gb=3, time="2:00:00", node=STEWARD_NODE)`.
  - `Long` (name `long`, no routine exits, pull as `Pool`'s): arguments `--wave`, `--preset`, `--job`, `--node`, `--mem`, `--minutes`; one task `pool.py work --wave <w> --job <id>`, `--time` the minutes as H:MM:SS, at most 7 days.
  - `WaveRecord` (a `Pull`): `pull` mirrors `<WAVES>/<w>/{summary.json,ledger.jsonl,final.db}` (those present) into `research/roadless/waves/<w>/`, replacing what is there (a mirror, not write-once results); `unpulled` and `retained` are `()`.
  - `Waves` (a `cs.Campaigns`):
    - `lease(ctx, rec)`: for a `pool` record, `cs.Lease(CAMPAIGN, args.node, ladder.bundle(args.mem))`, else None (a steward, a long run, today's kinds);
    - `template(ctx, campaign, node, bundle) -> cs.Template | cs.Done` (Ruling 4): the live waves (the registry's unfinished stewards), their summaries read with one `ctx.remote().probe`; among those whose ladder's `bundle(rung)` is `bundle` for some rung and whose summary has a line of that class on `node`, the one with the most pending jobs (ties by wave name): `cs.Template(kind="pool", run=f"{w}-{node}-{class_name(cards, rung)}-s{n}", commit=<its steward's commit>, args=("--wave", w, "--preset", p, "--node", node, "--mem", str(rung), "--gpus", str(<Ruling 10>)))`, `n` the first the registry lacks; `cs.Done(why)` when none has pending jobs there, the why naming the bundle and node; a live wave whose summary does not read is left out and named in a Done's why, and SystemExit when no live wave's reads (the next sync asks again);
    - `starts(market)`: `tuple(market())`.
  - `pool_more(ctx: cs.Context, rec: cs.RunRecord, *, killed: tuple[str, ...]) -> cs.Continue | cs.Done` (`Pool.more`), in this order: no unfinished `steward` run of the wave in `ctx.registry`: Done; its summary (one `ctx.remote().probe("cat reblock-waves/<w>/summary.json")`) absent, unreadable or of another format: SystemExit; its `at` before `rec.gone_at`, or a task of `killed` not among its `reaped`: SystemExit; a task of `killed` reaped with `held_job` False: Done; no pending job of the lease's class on its node: Done; else `c = cs.launches.width(ctx, rec, today=lambda: cs.Continue(workers=min(cs.launches.width_of(rec), pending)))`, and a Continue capped at `pending`.
  - `Steward.more`: Continue(workers=1) unless the summary is FINISHED or CANCELLED (Done); SystemExit when it does not read.
  - `PROJECT`: kinds `(Blocks(), Cpu(), Pool(), Steward(), Long())`, `campaigns=Waves()`; the rest as at 335a735d.

- [ ] **Step 1: Give cluster_check.py the `both-ways` harness, then write the guards**

cluster_check.py's command line becomes screening_check.py's (`GUARDS`, `FAULTS`, `main`, `both_ways`) for `ids`, `script` and `demand` with their faults as they are; `cpu` stays a print.
- `kinds`: each kind planned from a command line at HEAD (`_plan`): a `pool` of cpu-score at 12 GB on quadro2 is one CPU array pinned there, 2 CPUs, 12 GB, 2:00:00, its line `pool.py work ... --class m12`; one of gpu-presets at 32 GB on tesla1 a card's array; a gpu-presets pool on orion is refused (its cards unusable); a `--mem` that is no rung of the preset is refused; a `steward` lands on orion with 2 CPUs and 3 GB; a `long` of 90 GB and 3000 minutes asks 90 GB and 2-02:00:00; `Waves.lease` of each record: a pool's `Lease("reblock", node, Bundle(0, 2, 12))`, None for the rest.
  Fault `lease-bundle`: `Waves.lease` gives a CPU class 1 CPU, so the lease's bundle is not what its plan asks Slurm for (Ruling 3): the guard compares each pool's lease bundle with its plan's resources.
- `more` (the spec's guard 8, Review Focus 1 and 3): a real `cs.Context` over a temporary registry, its runner answering the summary's `cat`, `cs.launches.width` patched (the market's arithmetic is cluster_submit's, not this guard's).
  Cases: three pending and a killed task reaped holding a job: Continue(2) for a lease two wide with `width` calling `today`, and Continue(3) with `width` patched to answer the market's Continue(10) (capped by the jobs pending); a killed task reaped idle: Done; a killed task not yet reaped: SystemExit; a summary older than the lease's `gone_at`: SystemExit; no summary: SystemExit; one of format 2: SystemExit; no unfinished steward: Done; nothing pending in the class: Done; the steward's own `more`: Continue(1) while CLOSED, Done when FINISHED.
  Fault `continue-idle`: the idle-kill rule removed: the idle kill is continued.
- `template` (Ruling 4, Review Focus 2): a real `cs.Context` over a temporary registry with two live waves `a` and `b` (their stewards at two different commits), both with a summary line of (quadro2, `m12` of cpu-score) answering 3 and 7 pending jobs, and a pool lease of `b` there already named `b-quadro2-m12-s1`.
  The template for (quadro2, Bundle(0, 2, 12)) is `b`'s: kind `pool`, run `b-quadro2-m12-s2`, commit `b`'s steward's, args `--wave b --preset cpu-score --node quadro2 --mem 12 --gpus 40`.
  It passes every refusal of ec79a0eb's `launches.start`, checked with launches' own code: `ctx.parse(("submit", kind, run, "--at", commit, *args))` reads `at == commit` and not `dry`; `chain_base(run) == run`; `not ctx.registry.exists(run)`; the pool kind's plan of it passes `cs.launches.line_checked(plan, run, cs.launches.Start(...))` for that node and bundle.
  A bundle no live wave has a line of, a node no wave serves, and lines with nothing pending: Done, its why naming the bundle and node; `Waves.starts(lambda: (s,))` is `(s,)`.
  Faults, one per refusal the template must not trip and one for its choice: `other-mem` (the args' `--mem` the next rung: `line_checked` refuses the bundle), `other-node` (the args' `--node` another node), `continuation-name` (`<w>-<node>-<class>-l<N>`), `reused-name` (`n` not advanced past a run the registry has), `with-at` (the args carry `--at` HEAD's commit: the parse reads another commit than the template's), `first-wave` (the first live wave by name, whatever its pending jobs).

- [ ] **Step 2: Run them to see them fail**

Run: `env $ENV uv run python research/roadless/cluster_check.py kinds`
Expected: argparse's refusal of the unknown kind `pool`.

- [ ] **Step 3: Implement the kinds, the pull, `Waves` (`lease`, `template`, `starts`) and `pool_more`**

The docstring gains the pool's paragraphs (what a wave is, the three kinds, the `wave` verb Task 7 adds, the market's widths and starts, and where a wave's state lives).
`.gitignore` gains `waves/`.

- [ ] **Step 4: Run both ways**

Run: `env $ENV uv run python research/roadless/cluster_check.py both-ways ids,script,demand,kinds,more,template`
Expected: each passes; every fault `FAILS on ...`; `as expected`.

- [ ] **Step 5: Lint, types, commit, push**

Subject: `research: roadless -- the pool's kinds in cluster.py (pool: a lease of identical workers per wave, node and class, continued while its class has jobs on its node and every walltime kill held one, at the market's width capped by the jobs pending; steward: one 2-CPU 3-GB lease a wave on orion, continued until its wave ends; long: a run sized per job) and Waves, the one campaign's Campaigns on cluster_submit ec79a0eb (a pool run is reblock's lease of its class's bundle; the market starts a class's lease from Waves.template: the live wave with the most pending jobs of that class on that node, at its own commit, a fresh chain <w>-<node>-<class>-s<N>); the steward's pull mirrors the wave's summary, ledger and final queue into waves/<w>/; cluster_check gains both-ways and guards kinds (fault lease-bundle), more (continue-idle) and template (other-mem, other-node, continuation-name, reused-name, with-at, first-wave)`
Paths: `research/roadless/cluster.py research/roadless/cluster_check.py research/roadless/.gitignore`

---

### Task 7: The `wave` verb

The spec's 3.8 and 3.9 with Rulings 4, 7 and 12.

**Files:**
- Create: `research/roadless/cluster_wave.py`
- Modify: `research/roadless/cluster.py` (`main` dispatches `wave` before `cs.main`; `wave submit` and `wave <w> pools|long|retry|cancel` state the demand afterwards, as `STATES_DEMAND`'s verbs do)
- Modify: `research/roadless/cluster_check.py` (guard `wave`)

**Interfaces:**
- Consumes: `plan_wave`, `Estimator`, `FAMILIES`, `history_of`, `areas_of` (Task 2); `seed_json` (Task 1); `LADDERS`, `class_name` (Task 1); the kinds and `Waves` (Task 6); `read_summary` (Task 4); `common.write_bank`, `common.ids_of`; `cluster._listed`, `cluster._unpacks`, `cluster._script_at`, `cluster._command`.
- Produces:
  - `cluster.py wave submit <w> --preset <p> --family <f> --nodes <n,...> [--items <items,|@file>] [--unpack TAR] [--with FILE ...] [--board PATH] [--dry] <ids,|@file> -- <script> <args with {id} [and {item}]>`:
    1. the wave's commit is HEAD (`git rev-parse HEAD`); `<w>` is `project.NAME`, not `reblock`, and no record of the registry nor directory under `reblock-waves/` has it; the script is a file at the commit (`_script_at`); `{id}` in the command, and `{item}` exactly when `--items` is given;
    2. the bank written (`common.write_bank`, every id checked), areas (`areas_of`), the estimator (`FAMILIES[f](HERE)`, `history_of(HERE / "waves", f)`, `FIT_MIN_KM2`), the seed (`plan_wave`), each job's line `python -u <command with {id} and {item}> --keep-standing`, prefixed `python -u research/roadless/cluster_unpack.py "$CLUSTER_SUBMIT_INPUTS/<tar>" && ` with `--unpack`;
    3. printed: jobs by node and class, the long ones, the estimates' sources counted (the spec's 3.4), the cards and core-hours estimated;
    4. without `--dry`: the inputs (`jobs.json`, `bank.pkl`, the tars, the `--with` files) rsynced to `reblock-waves/<w>/inputs/` (Ruling 7); `submit steward <w>-steward --at <commit> --wave <w> --board <board>` (`--board` defaulting to `PROJECT.cluster.board()`: where the steward reads the allocation, which changes no row: Task 10 points a smoke wave at a hand-made one); for each LONG job `submit long <w>-long-<id> --at <commit> --wave <w> --preset <p> --job <id> --node <n> --mem <ceil gb> --minutes <min(1.5 x needed minutes, 7 days)>`. No pool is submitted: the market starts each class's leases from `Waves.template` (Ruling 4).
  - `cluster.py wave <w> status`: the summary (via ssh) by node and class, the reaped workers, the LONG jobs without a live long run, the classes with pending jobs and no unfinished pool lease of the wave (the market starts them as the allocation gives them units; in shadow mode, `wave <w> pools`), and the wave's runs in the registry with their states.
  - `cluster.py wave <w> pools`: by hand, a first lease for each (node, class) with pending jobs and no unfinished pool lease of the wave there, with `Waves.template`'s arguments: for a GPU file in shadow mode, where the market starts nothing (Ruling 4).
  - `cluster.py wave <w> long`: a long run for each LONG job without an unfinished long run of it, sized from the summary's `LongJob`.
  - `cluster.py wave <w> retry [<id>,...|failed]` and `cluster.py wave <w> cancel`: a request file (Ruling 12) written over ssh as cluster_demand's `board_script` writes (a temporary name, then a rename); `cancel` then `cs.main(PROJECT, ["cancel", run])` for every unfinished pool run of the wave (long runs only by name: `cluster.py cancel <run>`).

- [ ] **Step 1: Write the guard `wave`**

All under `--dry` (nothing leaves this machine), from command lines at HEAD:
- `wave submit g-1 --preset cpu-score --family lineup-score --nodes orion,quadro2 --items clearance,resistance_lp <two large block ids> -- research/roadless/road_rescore.py lineup-score <scratch> {id} {item}`: the seed's jobs one per (block, item), each block's on one node, each line ending `--keep-standing`, the classes per node printed and no pool run planned (the market starts them), every planned run name (the steward, each long run) a `project.NAME` (no block id in a name: ids have dots);
- the misspelt id of the `ids` guard (`MISTYPED`) is refused by the bank's writer before anything is planned;
- a script the commit lacks is refused naming the commit (as `script` does for today's kinds);
- `{item}` without `--items`, and `--items` without `{item}`: refused;
- a job the estimator puts past the lease (a stored wall whose estimate passes `fresh_minutes`, 105 minutes): a LONG job and a planned `long` run of `--mem` 1.5 times its peak.
Faults: `no-keep` (the job lines without `--keep-standing`), `block-names` (long runs named `<w>-long-<block>`: no `project.NAME`).

- [ ] **Step 2: Run it to see it fail**

Run: `env $ENV uv run python research/roadless/cluster_check.py wave`
Expected: `cluster.py` refuses the unknown verb `wave` (argparse).

- [ ] **Step 3: Implement cluster_wave.py and the dispatch**

Each submit goes through `cs.main(PROJECT, [...])`, its exit code checked; a failed submit stops the verb and says which runs it submitted.
The inputs go up with `Remote(PROJECT.cluster.host, SubprocessRunner()).rsync(...)` before the first submit.

- [ ] **Step 4: Run both ways**

Run: `env $ENV uv run python research/roadless/cluster_check.py both-ways ids,script,demand,kinds,more,template,wave`
Expected: `as expected`.

- [ ] **Step 5: Lint, types, commit, push**

Subject: `research: roadless -- cluster.py wave: submit (the wave's commit HEAD, every run --at it; the bank, seed, tars and --with files rsynced to reblock-waves/<w>/inputs once; a steward and a long run per job no lease holds, no pool: the market starts each class's leases from Waves.template), status, pools (by hand, for shadow mode), long, retry and cancel (requests the steward applies); guard wave (faults no-keep, block-names)`
Paths: `research/roadless/cluster_wave.py research/roadless/cluster.py research/roadless/cluster_check.py`

---

### Task 8: The demand with the waves' lines

The spec's 3.7 ("Demand"), with Rulings 1 and 2.

**Files:**
- Modify: `research/roadless/cluster_demand.py` (`demand_of`, `state`, the docstring)
- Modify: `research/roadless/cluster.py` (`state_demand`: the pool runs of the registry by `Waves.lease`, and the live waves' summaries read in one ssh call)
- Modify: `research/roadless/cluster_check.py` (guard `demand` extended)

**Interfaces:**
- Consumes: `Waves.lease`, `cs.submitted` (Task 6); `read_summary`, `Summary`, `lines_of` (Task 4).
- Produces:
  - `class PoolRun(NamedTuple)`: `wave: str`, `node: str`, `bundle: Bundle`.
  - `demand_of(recs, seen, limits, *, pools: Mapping[str, PoolRun], waves: Sequence[Summary], project, campaign, weight, at, nodes) -> Stated`: `pools` each unfinished pool run of the registry by name; a running task whose run is in `pools` counts on its wave's (node, bundle) and never at its tres-alloc; a waiting task of a pool run is no unit (its wave's pending jobs are); each (node, bundle) of a wave's summary line contributes `max(running, pending + claimed)` (Ruling 2), summed with the other waves' and with the steward's and long runs' units on that key; `minutes` as Ruling 2.
  - `Stated` gains `waves: tuple[str, ...]` (the waves whose lines are stated) and `unread: tuple[str, ...]` (live waves whose summary was absent or did not read: their running workers stated alone, and said).
  - `state(...)` reads the live waves' summaries (the registry's unfinished stewards' waves) with one `login.sh` of `for w in ...; do printf '%s\t' "$w"; cat reblock-waves/$w/summary.json 2>/dev/null; echo; done`; a wave whose summary does not read is in `unread` and makes `state` return 1 (said; the rest stated).

- [ ] **Step 1: Extend the guard `demand` first**

To its made-up squeue and seven runs add two waves sharing (quadro2, Bundle(0, 2, 4)), each with a local queue whose summary comes from `lines_of`: wave a with 5 pending, 2 claimed and 3 running workers there; wave b with 0 pending, 0 claimed and 2 running workers (idle, about to retire); a finished wave c whose summary has nothing pending and whose worker has exited; a pool lease of wave a waiting in Slurm on that key.
Expected: (quadro2, (0, 2, 4)) units 9 (max(3, 7) + max(2, 0)), its minutes the jobs' mean weighted as Ruling 2; the waiting pool task no unit; wave c no line; the steward on orion a unit at its held bundle; runs name every pool, steward and long run; a summary that does not read: `unread` names it, its two workers stated alone.
New faults: `count-done` (`lines_of` counts DONE jobs as pending: wave c states units), `every-node` (`lines_of` counts a group's jobs on each of the wave's nodes), `queue-only` (a pool line's units the jobs alone: wave b's idle workers dropped, Ruling 2).

- [ ] **Step 2: Run it to see it fail**

Run: `env $ENV uv run python research/roadless/cluster_check.py demand`
Expected: a TypeError naming `demand_of`'s missing `pools`.

- [ ] **Step 3: Implement**

`state_demand` builds `pools` from `Registry(REPO).all()` through `Waves.lease` and `cs.submitted` (no `Context`: the record's own arguments), and passes the reader of summaries.
The docstring's paragraphs on the one campaign and on units say how pool lines are made, and that the step, not the stewards, writes the file (Ruling 1).

- [ ] **Step 4: Run both ways**

Run: `env $ENV uv run python research/roadless/cluster_check.py both-ways demand,more,template,wave`
Expected: `demand: passes`; each of `finished-named`, `no-throttle`, `any-reason`, `no-running`, `count-done`, `every-node`, `queue-only` `FAILS on ...`; `as expected`.

- [ ] **Step 5: Lint, types, commit, push**

Subject: `research: roadless -- reblock's one demand states its waves' pool lines (Ruling 1: this machine's step, after every sync, is the file's one writer; the stewards publish their lines in summary.json): per wave and node and class the larger of its running workers and its jobs pending or claimed (held workers are units: the keeper treats held and undemanded as free), summed over waves sharing a bundle; a pool run's waiting tasks no units; a summary that does not read leaves its wave's running workers stated alone, and says so; guard demand gains count-done, every-node and queue-only`
Paths: `research/roadless/cluster_demand.py research/roadless/cluster.py research/roadless/cluster_check.py`

---

### Task 9: Every pooled command keeps what stands

The spec's 3.3 and its guard 2, decision 4.
**Precondition:** Task 5 committed.

**Files:**
- Modify: `research/roadless/pool_check.py` (guards `keep` and `double-claim`)
- Modify, where the guard finds a command that does not keep what stands: consolidated.py, road_rescore.py, arterial_study.py, uniform.py

**Interfaces:**
- Consumes: `Keep`, `standing_of`, `ROW_WRITTEN` (Tasks 3 and 5); `run_job`, `work` (Task 3).
- Produces: every command of the spec's 3.3 list keeps what stands under `--keep-standing`: consolidated `baselines`, `presets`, `lenses`, `screen`, `screen-lenses`; road_rescore `lineup-score`, `lineup-one` and `propose`; arterial_study `run` and `propose`; uniform `block`.
  road_rescore `propose` writes its proposal row through `Standing` (a failed proposal's row is a row: "a failure is a finding"), so a person's second `propose` of one (block, method) now refuses as every optimizer's second run does, and `--replace` proposes again; `propose-all` proposes only pairs without a row, as before.

- [ ] **Step 1: Write the guard `keep`**

`pool_check.py keep`: each command run in this process on one block in a temporary rows root (each script's rows directory patched to it), its heavy call patched to return a canned row (`relax.one`, `polish_greedy.run`, `metric.fixed` and `metric.model`, `road_rescore._cut_and_score`'s scoring, the proposal method, uniform's arms and scorer: each patch named in the guard), with one of its identities' rows standing beforehand.
For each: under `Keep()`, the standing row's bytes unchanged, every other identity written, a `ROW_WRITTEN` line per row written and none for the standing one, exit without raising; with every row standing, nothing runs (the patched call never called).
consolidated `presets`: its one-card row names in `presets` only the presets run.
Fault `refuse`: `standing_of` resolves `--keep-standing` to `Refuse()` (each command raises `Exists`).

- [ ] **Step 2: Write the guard `double-claim`**

The spec's guard 2: two workers (Task 3's `work`, two processes) on one queue whose one job's two claims are forced (the second worker's claim patched to take the CLAIMED job), its line `python -u research/roadless/pool_check.py smoke-job <id> quick --keep-standing` writing its two rows through `Keep`: one set of rows; both attempts end DONE, the second's `[row] kept` lines in its log; no strike.
Fault `refuse` as above: the second attempt fails with `Exists` and is a strike.

- [ ] **Step 3: Run them to see them fail**

Run: `env $ENV uv run python research/roadless/pool_check.py keep`
Expected: `keep: FAILS on` at least road_rescore `propose` (it writes its proposal row with `write_parquet`, outside `Standing`); any other command named is one Task 5's threading left rerunning or refusing, and Step 4's to fix.

- [ ] **Step 4: Make each failing command keep what stands, and write `smoke-job`**

Only the commands the guard names change, each so that `todo` decides per identity and a command with nothing to do returns before building the block.
`pool_check.py smoke-job <id> <item> [--keep-standing]` (Task 10's job): per item, allocate (`big`, `oom`: 3 GB touched page by page), sleep (`slow`: 25 minutes; others 20 s), import cupy and touch the card (`gpu`), then write two rows `research/roadless/pool_smoke_rows/<id>__<item>__<k>.parquet` through the resolved `Standing`.

- [ ] **Step 5: Run both ways**

Run: `env $ENV uv run python research/roadless/pool_check.py both-ways keep,double-claim`, and every earlier guard of pool_check.py and cluster_check.py both ways.
Expected: `as expected` throughout.

- [ ] **Step 6: Lint, types, commit, push**

Subject: `research: roadless -- every pooled command keeps what stands under --keep-standing (consolidated baselines, presets, lenses, screen, screen-lenses; road_rescore lineup-score, lineup-one, propose; arterial_study run, propose; uniform block): a standing row is that identity done, a command with nothing left returns before building its block, presets' one-card row names the presets it ran; pool_check smoke-job (Task 10's job); guards keep and double-claim (fault refuse: the second claim's Exists a strike)`
Paths: `research/roadless/pool_check.py` and the modules changed.

---

### Task 10: Merged, and the smoke waves on ai

The spec's section 6's smoke wave, and the guards' live halves (the claim on ai's home, OOM evidence in a Slurm cgroup).
Every submit here is the controller's to confirm first.

**Files:**
- Modify: `research/roadless/NOTES.md` (a section at the end)
- Git-ignored, removed at the end: `research/roadless/pool_smoke_rows/`, `research/roadless/waves/smoke-*/`

**Interfaces:**
- Consumes: everything above, on research/roadless in the main checkout.

- [ ] **Step 1: The merge (the controller's)**

In the main checkout, once `git status` shows no uncommitted edit of the epistemetry executor's to a file research/lease-pool changed: `git merge origin/research/lease-pool` into research/roadless, push.
Then check the timer's next sync still settles the live runs: `uv run python research/roadless/cluster.py status` lists cp-roads4-desire-g3a and -g3b as before, and `journalctl --user -u cluster_submit-reblock-sync -n 20` shows no traceback.

- [ ] **Step 2: The smoke waves, planned dry**

From the main checkout, `--dry`:
- `smoke-claim`: `wave submit smoke-claim --preset smoke --family smoke --nodes orion,quadro2 --items quick @<200 small blocks> -- research/roadless/pool_check.py smoke-job {id} {item}` (the 200 first small blocks of `consolidated_rows/summary_features.csv` by id, in a file under the scratchpad).
- `smoke-cases`: 3 small blocks, `--items big,oom,slow`, `--nodes quadro2`, `--board reblock-waves/smoke-cases/board` (the hand-made allocation's: Step 6).
- `smoke-gpu`: 2 small blocks, `--items gpu`, `--preset gpu-presets`, on the GPU node whose capacity has a card (`cluster.py gpu`).
Expected: smoke-claim's 200 jobs on two nodes, class `m2`; smoke-cases' `big` in `m4`, `oom` and `slow` in `m2`, none LONG; smoke-gpu's two jobs in `g16`.
smoke-cases shares (quadro2, `m2`) with smoke-claim, and the two run at once (Step 3): the market's starts there go to the wave with more pending jobs (Review Focus 2).

- [ ] **Step 3: smoke-claim and smoke-cases (the controller confirms)**

Submit both; `cluster.py wait smoke-claim-steward` and `cluster.py wait smoke-cases-steward` in the background.
While it runs: `cluster.py gpu` shows reblock's lines (orion, 0/2/2) and (quadro2, 0/2/2) with `held` equal to the workers running there (Ruling 3: if `held` is 0 while they run, the bundle Slurm holds is not the class's; stop and report), the market starts each class's pools (`<w>-<node>-<class>-s<N>`: `wave submit` submits none) and their continuations (`-l<N>`) take the market's widths, and each start on (quadro2, 0/2/2), which both waves use, goes to the wave with more pending jobs there (the start's log line and the run's `--wave`).
After: the final queue in `research/roadless/waves/smoke-claim/final.db`: 200 jobs DONE, each with one DONE attempt and no two attempts overlapping in time (the claim guard on ai's home, across two nodes); every row of `pool_smoke_rows` for its ids; `reblock-waves/smoke-claim/inputs/bank.pkl` gone; the steward's peak memory from its log against its 3 GB (Ruling 11); the `reblock` demand with no line of the wave once its workers exited.

- [ ] **Step 4: smoke-gpu (the controller confirms)**

Submit it; wait; both jobs DONE on a card (the job's log names the device), the lease's record a GPU array of `gpu_gb` 8 on that node.

- [ ] **Step 5: smoke-cases' cases**

Once its steward has finished (Step 3):
- `big`: DONE in `m4` from its first attempt.
- `oom`: its first attempt's outcome OOM with the worker's log naming its evidence (`cgroup v2 ...` or `v1`), the job promoted to `m4` and DONE there (the live half of guard 6; decision 9: an `Unreadable` evidence makes it a strike instead: report which ai gives, and stop if the worker itself was killed: Review Focus 4).
- `slow`: killed at its 20-minute walltime, reaped with `walltime` True and its estimate raised past the lease, LONG; `wave smoke-cases long` submits `smoke-cases-long-<id>`, which runs it to DONE.

- [ ] **Step 6: The step-aside, through a hand-made allocation**

While smoke-cases' `m2` workers run, write `reblock-waves/smoke-cases/board/gpu/allocation.json` on ai: the board's current `allocation.json`, its `at` now, mode `active`, and on quadro2 for `reblock/reblock` a place of bundle (0, 2, 2) with `yields` 1 (the steward reads it within a minute: write it whole, a temporary name then a rename).
Expected: the steward's ledger records one ask; that worker's log says it was asked to step aside and it exits 75 at its next job boundary; the lease is continued while its class has jobs.

- [ ] **Step 7: The records**

NOTES: a section at the end, `### The lease pool (<the date of the smoke waves>)`, wrapped at 99, current numbers only: what a wave is now (the queue, pools by class, the steward, the long lane, the `wave` verb), the rulings that departed from the spec and why (one campaign with this machine's step as its writer; a whole core per CPU class with g3's held bundles; starts from the campaign's template, by the most pending jobs; strikes on reaped attempts), the smoke waves' numbers (jobs, attempts, widths and starts seen, the OOM evidence ai gives, the walltime kill, the step-aside, the steward's peak), and the guards with their faults.
Then remove `research/roadless/pool_smoke_rows/` and `research/roadless/waves/smoke-*/` (git-ignored test rows) and `rm -r` the smoke waves' directories under `reblock-waves/` on ai (`ssh ai rm -r reblock-waves/smoke-claim reblock-waves/smoke-cases reblock-waves/smoke-gpu`).
Subject: `research: roadless -- NOTES "The lease pool": built and smoke-tested on ai (` then Steps 3-6's numbers: smoke-claim's jobs, attempts, workers and starts per node; the OOM evidence ai gives; the walltime kill to the long lane; the step-aside; the steward's peak `)`
Paths: `research/roadless/NOTES.md`

---

### Task 11: The migration: the pool replaces `blocks` and `cpu`

The spec's section 5, decision 6, with Ruling 1 (cluster_demand.py stays).

**Precondition**, each checked and printed before any edit; any that fails stops the task:
- cp-roads4-desire-g3a and -g3b finished and synced: `cluster.py status` lists no unfinished run, and their records have `finished_at` and `settled`;
- no record of kind `blocks` or `cpu` is unfinished, unsettled, pinned (`cluster.py pins`) or has unpulled results;
- followup.md's g3 steps that need no cluster run are done (the five networks migrated into the lineup's root), and its lineup-score of them is written as the wave `pb-lineup-g3` (the spec's section 7), not run here;
- Task 10 passed.

**Files:**
- Modify: `research/roadless/cluster.py` (`Blocks`, `Cpu`, whatever only they used and the docstring's paragraphs on them deleted; `PROJECT.kinds = (Pool(), Steward(), Long())`; `CPU_NODES` kept: Ruling 10's throttles)
- Modify: `research/roadless/cluster_check.py` (`cpu` rewritten as a guard of a `wave submit --dry` on each CPU node; `ids` and `script` on `wave submit --dry` instead of `submit blocks` and `submit cpu`)
- Modify: `research/roadless/cluster_demand.py` (`Stated.unplaced` and its paragraph deleted: no run is unpinned now)
- Modify: `research/roadless/BACKLOG.md`, `.superpowers/sdd/2026-10-16-consolidated-pass/followup.md`
- Move (not tracked): every `blocks` and `cpu` record, `.cluster_submit/runs/<run>.json` and `.cluster_submit/runs/<run>/`, to `.cluster_submit/archive/` (the spec's open question 6, decided: archived)

- [ ] **Step 1: Check the preconditions**

A one-off `python - <<'EOF'` over `Registry(REPO).all()` and `cluster.py pins`, printing each condition and the records failing it.

- [ ] **Step 2: Rewrite the guards first**

`cpu`: a wave of cpu-score on each node of `CPU_NODES` plans pools pinned there with no GPU request, 2 CPUs and each rung's memory, its lines unpacking each block's tar through cluster_unpack.py; a GPU preset on a CPU node refused.
`ids` and `script` as before, through `wave submit --dry` (fault names kept: `no-check`, `no-script-check`).
Run `cpu` to see it fail: it also checks that `PROJECT.kinds` names exactly `pool`, `steward` and `long`, and `blocks` and `cpu` are still there.

- [ ] **Step 3: Delete the old kinds; archive their records**

Delete as listed; then move every `blocks` and `cpu` record (json and directory) to `.cluster_submit/archive/`, printing the count (about 330 settled runs: the spec's section 5).
Archived records stay JSON as they were (their kind is what ran); desire_fit.py and compute.py under .superpowers read `.cluster_submit/archive/` from now on (their path constants edited if they name runs/).

- [ ] **Step 4: Run every guard both ways, and a sync**

Run: `env $ENV uv run python research/roadless/cluster_check.py both-ways cpu,ids,script,demand,kinds,more,template,wave`, `env $ENV uv run python research/roadless/pool_check.py both-ways <every guard>`, then `uv run python research/roadless/cluster.py sync` and `status`.
Expected: `as expected`; the sync reads every record (no "unreadable records", no "has no kind").

- [ ] **Step 5: The records**

BACKLOG (wrapped at 99, current state only): the head's "how we run on the cluster" lines name waves and the `wave` verb; the re-run's Next names its waves as the spec's section 7 does, with `pb-lineup-g3` after followup.md's migration; under "Not pursued, with what would change that": checkpointing cycle_desire and the arterial (decision 8: only if an oversized re-proposal is planned), mycooc's `reroute` (decision 11: if a wave's nodes are allocated unevenly), `Screened` (decision 7: after the re-run), the 4-card cap (decision 5: once the first waves show the market's widths behaving), and moving the pool to GeoffChurch/cluster_submit (decision 2: once mycooc and reblock both run on it).
followup.md: its g3 commands as waves.

- [ ] **Step 6: Commit and push**

Subject: `research: roadless -- the lease pool replaces the blocks and cpu kinds (the spec's section 5; after cp-roads4-desire-g3a and -g3b settled): cluster.py's kinds are pool, steward and long, no old kind kept as an option (a wave reproduces any array); cluster_check's cpu, ids and script guard waves; cluster_demand's unplaced count goes (every run is pinned now); N blocks and cpu records archived to .cluster_submit/archive/; BACKLOG and followup.md run the re-run as waves`, N the count Step 3 printed
Paths: `research/roadless/cluster.py research/roadless/cluster_check.py research/roadless/cluster_demand.py research/roadless/BACKLOG.md .superpowers/sdd/2026-10-16-consolidated-pass/followup.md`

---

## Execution notes for the controller

- Subagent-driven, implementers and reviewers on opus; each task's reviewer runs its `both-ways` lines and the gates before approving.
- Order: 1, 2, 3, 4, then 6, 7, 8 (and 5 as soon as the epistemetry move's Task 7 is on origin/research/roadless), 9 after 5, 10 after 1-9, 11 when its precondition holds (the g3 runs settled).
  Tasks 1-4 depend on each other's Interfaces in order; 6-8 consume 1-4; 9 consumes 3 and 5.
- Stop points that need the owner or the controller: a smoke-wave submit (each); Task 10 Step 3's `held` at 0 while workers run (the bundle Slurm holds is not the class's); Task 10 Step 5's OOM killing the worker, or ai's OOM count unreadable.
- If the epistemetry move's Task 7 is far off when Tasks 6-8 are done, Task 5 waits; nothing else does, but Task 10 needs Task 9, which needs Task 5.
