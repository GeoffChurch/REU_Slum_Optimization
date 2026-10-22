"""Guards of the lease pool (docs/superpowers/specs/2026-10-21-lease-pool-design.md), each with its
faults (a patch in one process, no file edited, as gap_clear_check.py's): a guard passes on the
code as it is and must fail under each of its faults.

    S=research/roadless/pool_check.py      # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S claim [--fault <name>]         # fault: split-tx
    uv run python $S transitions [--fault <name>]   # faults: late-finish, drop-on-top,
                                                    # any-worker, second-claim, no-exclude,
                                                    # claimed-runs-on
    uv run python $S class [--fault <name>] <rows>  # faults: median-rate, no-fit-margin,
                                                    # family-rate, all-long, no-capacity,
                                                    # with-proposed, one-run
    uv run python $S walltime [--fault <name>]      # faults: no-fit, left-once
    uv run python $S oom [--fault <name>]           # faults: fail-on-oom, unified-first
    uv run python $S identity [--fault <name>]      # fault: no-identity
    uv run python $S worker [--fault <name>]        # faults: fail-worker, no-exclude,
                                                    # no-retire, no-kill, pid-only, no-rebirth,
                                                    # no-checks
    uv run python $S dead-claim [--fault <name>]    # faults: by-age, blind
    uv run python $S aside [--fault <name>]         # faults: whole-yields, stale-kept,
                                                    # unread-fatal
    uv run python $S end [--fault <name>]           # faults: early-end, claimed-runs-on,
                                                    # open-ended, request-fatal, open-fixed
    uv run python $S both-ways <guards,> [args ...]   # each guard, then under each of its faults

A guard runs as `<guard> [--fault <name>] [args ...]`: the fault is named by its flag, so a
mistyped one is never read as an argument, and an unknown one exits naming the guard's faults.
Each guard declares its arguments (GUARDS), and run alone it refuses any other count of them;
`both-ways` passes its arguments to every guard it names. `class` takes the rows root (`<rows>`:
research/roadless of the checkout holding the stored rows, which are git-ignored: from a worktree,
the main checkout's); the others take none.

`claim`: CLAIM_JOBS jobs of one node and one class under cpu-score, estimated evenly over (0, 105]
minutes (pool_queue.fresh_minutes, a lease less its margin and a worker's startup: every job a
plan would pool), in a queue in a temporary directory on local disk; CLAIM_WORKERS processes
(fork), started together, each opens the queue (the first makes it from the seed), registers a
worker (all of one run) and claims with a fresh worker's 105 minutes left (every job fits: the
fit is `walltime`'s; the run's attempted jobs, excluded, are all done), finishes the job done and
claims again until nothing is left, sleeping 0 to CLAIM_SLEEP between operations. Fails on a job
with two attempts, a job not done, an attempt finished by a worker other than its claimer (the
queue's attempts against what each process says it finished), and a worker whose claims'
estimates ever rise (each claim is the longest pending at its time, so a worker's never rise).
Fault `split-tx`: SqliteQueue.claim selects in one transaction and updates in a second,
SPLIT_SLEEP apart: some job is claimed twice.

`transitions`: on a queue under the preset cpu-score (lease 120, margin 10, startup 5, strikes 3),
each case a fresh queue, its workers claiming with a fresh worker's 105 minutes left, the job as
it stands after each event against the job it should be (every field: its state, class, lane,
strikes, gb, peak_gb and minutes among them):
- `done`: finish DONE -> DONE.
- `failed-progress`: finish FAILED having written two rows -> PENDING, strikes 0.
- `failed-strikes`: finish FAILED with none, three times -> PENDING (1), PENDING (2), FAILED (3).
- `oom-promote`: finish OOM with peak 4 from m4 (a peak at the class's rung, as a cgroup measures
  it) -> PENDING in m12, peak_gb 4, gb 6 (fit_mem times the peak: the peak that killed it is a
  floor of its need).
- `oom-top`: finish OOM with peak 88 from m88 -> LONG, lane LONG, peak_gb 88, gb 132 (1.5 x 88).
- `reap-progress`, `reap-strike`: reap of a worker holding a job, `now` before its end: having
  written a row -> PENDING, strikes 0; none -> PENDING, strikes 1 (the plan's Ruling 6).
- `walltime`: reap after its end, the attempt having run 50 minutes on a job estimated 30 ->
  PENDING, minutes 50, outcome WALLTIME, strikes 0; `walltime-long`: having run 115 -> LONG
  (past 120 - 10 - 5).
- `late-finish`: a reaped worker's late finish DONE -> the attempt recorded (its rc and peak), the
  job as the reaper left it (the spec's 3.2, "Done").
- `retry`: retry of a FAILED job -> PENDING, strikes 0.
- `cancel`: every PENDING, LONG and CLAIMED job CANCELLED (3), the claimed one's attempt
  CANCELLED; its worker's later finish FAILED with no row -> still CANCELLED, no strike, the
  attempt's code recorded; its worker's retire -> settles nothing, still CANCELLED.
- `claim-job`: a pool's claim leaves a LONG job; claim_job takes it (CLAIMED); its finish FAILED
  without rows -> LONG, strikes 1.
- `aside`: ask_aside(units=2) of class m4 on a node with three live workers of it, one idle, and
  an idle worker of m12 -> the idle m4 worker, then the one whose job ends soonest; asked again
  with the same `since` -> none.
- `other-worker`: a worker's finish DONE of another worker's attempt is refused (ValueError), the
  job still the claimer's and the attempt unrecorded.
- `one-claim`: a worker holding a claim is refused another, by claim, by claim_job of another job
  and by claim_job of the job it holds (ValueError: refused before the job's state is read), the
  other job left PENDING.
- `exclude`: a job a lease's worker attempted (FAILED, a strike: PENDING) is no claim of the
  lease's: its other worker takes the shorter pending job instead, and with only that one
  pending neither gets any (the job as it was); the next lease's worker takes it. (Each case's
  workers register in leases, `Case.worker`; a job struck three times is struck by three
  successive leases' workers, as no lease claims a job it attempted again.)
Faults: `late-finish` (finish ignores whose claim it is: the reaped worker's late DONE marks the
job done), `drop-on-top` (past the top rung the job is FAILED, not LONG: the spec's "OOM promotes;
it never drops"), `any-worker` (finish takes any worker's word for an attempt: the other worker's
DONE marks the job done), `second-claim` (a worker holding a claim may claim again), `no-exclude`
(no lease's attempts excluded, pool_queue._attempted_by answering none: the lease's other worker
takes back the job it attempted), `claimed-runs-on` (cancel leaves a claimed job CLAIMED: its
later failure takes it back to PENDING after its wave was cancelled).

`class` (pool_plan's classes and estimates, the spec's 3.4 with the plan's Ruling 15), on the
stored rows under `<rows>`, every block's area from its consolidated_rows/summary_features.csv:
- `stored`: the lineup-score family's every stored (block, method) (scoring-only runs: no root of
  pool_plan.PROPOSED_AND_SCORED), planned under cpu-score (a plan of every stored block and method
  over orion and quadro2), its basis recomputed here from the family's runs (its largest stored
  peak and, apart from it, its longest stored wall), lanes both ways: needing 1.2 times that
  peak, a job a rung holds whose 1.25 times that wall fits a fresh lease (fresh_minutes, 105) is
  a pool's in that rung's class, any other the long lane's needing 1.5 times its peak, in the
  class of the rung holding it (the top rung's past every rung); its needed minutes are 1.25
  times the wall (rel 1e-9). And two facts of the rows (2026-10-21) no reader
  supplies, pinning the scoring-only ruling: no stored network is past the top rung, and 62803's
  cycle_desire is a pool's in m24 (SCORED_ALONE_*).
- `leave-one-out`: for each item of the propose family, its jobs on blocks of at least
  FIT_MIN_KM2: each block's estimate with its own rows removed (a family without them) is 1.5
  times the worst per-km2 peak of the others on such blocks, the item's own when they are on
  FIT_MIN_BLOCKS blocks, else the family's, times its area (recomputed here, rel 1e-9), its source
  naming that fit; cycle_desire's (CLASS_ITEM) holds each block's measured peak (another item's
  misfit, which costs an OOM's promotion, is counted). cycle_desire's tightest is 62803 (164.7 GB
  on 11.90 km2, 13.8 GB/km2, the worst rate on record: it finished after desire_fit.out), held by
  53556's 10.9 GB/km2. The other items tell a family-wide fit from an item's: cycle_desire's own
  rate is the family's.
- `plan`: plan_wave of the CLASS_BLOCKS stored lineup blocks with the most stored work (4287 among
  them: its arterial network's 3.0 h, the long lane's) and the seven methods, over CLASS_NODES
  with pool_plan.USABLE_GB: each job's lane both ways against its estimate, as in `stored`;
  every job of a block on one node, its group the block, and that node's usable memory holding
  each of them (a pool job's rung, a long job's need: the g3 blocks' fitted cycle_desire scoring
  is past quadro2's 248 GB); the placement replayed, the groups longest-processing-time first on
  their pool jobs' minutes, each to the least loaded of the nodes that hold it.
Faults: `median-rate` (the fit's rate the median per-km2 rate, not the worst: the recomputation
fails), `no-fit-margin` (the fit without its 1.5: 62803 is classed below its measured peak),
`family-rate` (every item fitted at the family's rate: the cheap methods' recomputation fails),
`all-long` (every job sent to the long lane: the pool's jobs fail the lanes), `no-capacity`
(groups placed on any node: a long job on quadro2 past its 248 GB), `with-proposed` (lineup-score
reads the rows that proposed as they scored: the facts fail) and `one-run` (the stored wall the
largest peak's run's, not the longest: the needed minutes fail).

The worker's guards (pool_worker, pool.py `work`; the spec's 3.5 and guards 4, 6 and 9) each run
the worker on a queue in a temporary directory under cpu-score (margin 10, strikes 3, idle 10
minutes, polls of 60 s), its jobs real children (`bash -c` lines: `true`, `exit <n>`, a shell
that kills itself with SIGKILL, a python writing rows), pool_worker.work's on a fake clock that
moves only a poll at a time unless a case moves it, the workers of one lease (RUN) unless a case
says another:
- `walltime` (guard 4): 40 minutes left, jobs estimated 20, 29, 31 and 60: `still`, it runs 29
  then 20, never 31 or 60, and retires (75) with them pending; `moving`, each job spending 15
  minutes of the clock, it runs 29 and retires, the 20-minute job no longer fitting the 15 left.
  Faults: `no-fit` (SqliteQueue.claim ignores the minutes left: it runs all four) and `left-once`
  (the minutes left computed once a worker: moving runs 20 after 29).
- `oom` (guard 6, its local half): `promote`, a child SIGKILLed while a fake count of OOM kills
  rises by one: the job promoted (PENDING in m12, an attempt `oom` rc -9), the worker going on to
  the next job and draining (0); the child's oom_score_adj, which it prints, 1000, the worker's own
  unchanged (and below 1000, so the child's is the worker's doing); `unchanged`, the count not
  rising, and `unreadable`, no count (pool_worker.Unreadable): the attempt FAILED, a strike, the
  job PENDING for the next lease's worker and this one retired (75: the only job pending is one
  it attempted), the log saying why (unreadable). `readers`: pool_worker.oom_evidence on
  fixtures under a cgroup root of their own: v2 (memory.events's oom_kill, read afresh at each
  call), v1 (memory.oom_control's), hybrid (v1's memory line beside a v2 one: v1), and Unreadable
  for a v2 cgroup with no memory.events, a v1 one with no oom_kill line (before kernel 4.13) and no
  line at all; this machine's is printed, unchecked. Faults: `fail-on-oom` (an OOM recorded
  FAILED: promote fails) and `unified-first` (the v2 line taken first: hybrid fails).
- `identity` (guard 9): a queue at COMMIT and a worker's checkout at another: work exits 1 before
  registering, no attempt; work_one (the long lane's) the same. Fault `no-identity` (the check
  patched out): each runs its job.
- `worker`: `absent-queue` (the plan's Review Focus 1), pool.py `work` run in this process as a
  cluster_submit task would be (its environment set: HOME, the Slurm array ids and end time, the
  run, the checkout and its manifest) on a wave whose shipped seed no process has opened: it makes
  the queue, its job 1 failing (a strike) without ending it, then job 2, which checks its own
  environment and directory (CLUSTER_SUBMIT_INPUTS the wave's inputs, common.BANK_ENV the wave's
  bank, the checkout), done; exit 75 (job 1, pending, is one it attempted), the worker registered
  and retired. `bad-node`: jobs 1-6 failing (a bad node) and job 7 not, three workers of one
  lease in turn: the first fails jobs 1-3 and exits 1 (three different jobs in a row), the second,
  excluding what its lease attempted, fails 4-6 and exits 1, the third runs job 7 and retires
  (75): every failing job one strike, none FAILED; `one-job`, its only job failing: one attempt,
  one strike, retired (75). `progress`: a job writing two rows by clearings.write_once and exiting
  1, run by a worker of one lease and then of the next: the first's attempt counts 2 rows
  (row_marker.ROW_WRITTEN lines) and is no strike, and it retires (75) rather than claim it again;
  the second's is refused at its first row (Exists), writes none and strikes. `raises`: the
  worker's log raising at the child's first line (`echo started; exec sleep 5`): the log's own
  error out of work within 2 s (the child killed), the job PENDING with no strike, its attempt
  RETIRED, the worker retired. `group`: the same of a compound line whose shell forks a
  grandchild (GRANDCHILD, `true && sh -c '...; exec sleep 5'; echo after`, the grandchild
  printing its pid): the grandchild gone (or a zombie) within GONE_WAIT of the kill, the job's
  whole process group killed. `requeued`: task 4242_0 of a lease, registered and holding job 1
  and never retired (killed with its node), run again by pool.py at SLURM_RESTART_COUNT 1 (Slurm
  keeps a requeued task's array id): registered again, its earlier self reaped (job 1 PENDING,
  a strike, the reap recorded with its job), job 2 run, retired (75: job 1 is its lease's
  attempted); at the same count again, refused. `refusals`: pool.py `work --class m5` (no class
  of cpu-score), `--node orion` on quadro2 ($SLURMD_NODENAME) and `--job 1` (orion's) on
  quadro2: each a SystemExit naming what it is not, no worker registered.
  `idle`: an open queue with nothing pending: retired (75) after the ladder's idle minutes, 10
  polls; `idle-at-end`, the same with less than the margin left: at once. `aside`: asked aside
  during job 1: retired (75) after it, job 2 never claimed. `long-lane`: pool.py `work --job` of a
  LONG job, twice: done (0), then not waiting (1), each worker registered on the job's node in the
  class `long`. Faults: `fail-worker` (one failed job ends the worker, exit 1) and `no-exclude`
  (no lease's attempts excluded: a lease spends a failed job's every strike), each failing
  absent-queue, bad-node, one-job and progress; `no-retire` (SqliteQueue.retire a no-op: raises
  and the cases reading the workers fail), `no-kill` (pool_worker._kill a no-op: raises and group
  wait out the child), `pid-only` (pool_worker._kill kills the shell alone, not its group: group's
  grandchild runs on), `no-rebirth` (a requeued task's registration refused as any second one:
  requeued's restart exits refused, its claim held) and `no-checks` (pool.py's checks of --node,
  --job's node and --class patched out: refusals runs).

The steward's guards (pool_steward, pool_summary; the spec's 3.6 and guard 3), each on waves under
a temporary `waves`, squeue stubbed (pool_steward.Squeue's runner: an answer listing given tasks,
the steward's own STEWARD_TASK among them, or a failure), no allocation unless a case makes one:
- `dead-claim` (guard 3, Review Focus 4): `round`, jobs 1 (30 min), 2 (100) and 3 (20) of m4 held
  by three workers of three leases registered at t0: one live (job 2), one a real child process
  SIGKILLed holding job 1, one whose walltime ended WALLTIME_RAN minutes after t0 (job 3); a
  round at t0 + 2 h, squeue listing the live one and the steward's own: job 1 PENDING with one
  strike (it wrote no row), job 2 still the live worker's, job 3 PENDING with no strike and its
  minutes raised to the 50 it ran; the summary's reaped the killed worker (held a job, not past
  its walltime) and the ended one (held a job, past it), its registered the live one alone,
  summary.json reading back as the summary, one ledger line. `fails`, `hangs` and
  `without-own`: the same workers (all registered here), squeue failing (exit 1), outliving its
  timeout, and answering the live worker without the steward's own: nobody reaped, every job
  still its worker's. Faults: `by-age` (a claim older than an hour reaped, squeue unasked: the
  live worker's long job reaped) and `blind` (a failed squeue, or one without the steward's own,
  read as an empty answer: every worker reaped).
- `aside` (Ruling 5, Review Focus 2): two waves (ASIDE_WORKERS: 4 and 2 live workers on quadro2
  in m4 of cpu-score) under one `waves`, each steward's first round with no allocation (their
  summaries), then two rounds each under a hand-made allocation (the keeper's JSON, read back)
  in which the campaign yields ASIDE_YIELDS (3) units of m4's bundle on quadro2: `active`, the
  workers each wave has asked after each round 2, 1, then 2, 1 (none more); `stale`, the
  allocation older than STALE_AFTER, and `shadow`, its mode shadow: none; `ghost`, beside them a
  dead steward's CLOSED summary written STALE_AFTER and a minute before, counting GHOST_WORKERS
  (6) workers there: it gets no share (2, 1, 2, 1) and the first wave's ledger names it as stale;
  `unreadable`, beside them a summary of another format: left out (2, 1, 2, 1) and named as
  refused. Faults: `whole-yields` (each steward asks all the yields its own workers can give: 3
  and 2), `stale-kept` (no summary too old: the ghost's 6 workers take a share, 1, 1) and
  `unread-fatal` (a summary that does not read stops the round).
- `end`: pool_steward.steward with a lease of three END_POLL polls: `finish`, a wave both of whose
  jobs are done: exit 0, final.db holding the queue's jobs and attempts, the bank and the tars
  deleted from its inputs (its seed and --with file kept), the summary FINISHED, the ledger's
  last line the end; `retry`, a FAILED job and requests to retry it, to retry job 99 (none such)
  and a malformed one (not JSON): job 1 PENDING with no strike, the other two refused, all three
  deleted and in the ledger, the malformed one's refusal logged, exit 75 (two rounds no longer
  fit), no final.db; `cancel`, a cancel request with a job pending, one claimed (its
  worker live), one LONG and one done: the three CANCELLED, exit 0, final.db, the bank and tars
  deleted, the summary CANCELLED, and the claimed one's later failure leaving it CANCELLED;
  `long`, a done job and a LONG one waiting: exit 75, no final.db, the bank kept, the summary
  CLOSED listing the long job; `open`, a wave whose one job is done but whose queue is open (a
  feed that never closes it, as Screened's would between grants): exit 75, no final.db, the
  summary OPEN; `fixed-open`, a fixed feed's seed leaving its queue open (nothing would close it):
  refused by Seed and by read_seed of a jobs.json edited so, each naming the wave; `command`,
  pool.py `steward` run in this process as a cluster_submit task would be (HOME, the Slurm array
  ids and end time, USER; a relative --board with no allocation) on a shipped wave whose one job
  is done: metrics.RUNTIME named in the queue's header, exit 0, the summary FINISHED naming it,
  final.db, the bank deleted. Faults: `early-end` (a job waiting for the long lane counted as
  ended: long ends),
  `claimed-runs-on` (transitions': cancel leaves the claimed job running, so the wave never
  ends), `open-ended` (an open queue's wave ended once its jobs have: open ends) and
  `request-fatal` (a request that does not read, or that the queue refuses, stops the round)
  and `open-fixed` (pool_queue._open_fixed answering no: an open fixed seed accepted).
"""
from __future__ import annotations

import io
import json
import math
import multiprocessing as mp
import os
import queue
import random
import shlex
import signal
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, redirect_stdout
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Barrier
from pathlib import Path
from typing import NamedTuple

from cluster_submit import Bundle
from cluster_submit.context import STAMP
from cluster_submit.gpu import (
    Allocation,
    Fit,
    FittedTier,
    NodeShare,
    Place,
    Standing,
    allocation_json,
    read_allocation,
)
from cluster_submit.remote_build import MANIFEST

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import pool as pl  # noqa: E402
import pool_plan as pp  # noqa: E402
import pool_queue as pq  # noqa: E402
import pool_steward as ps  # noqa: E402
import pool_summary as psum  # noqa: E402
import pool_worker as pw  # noqa: E402
import row_marker  # noqa: E402
from pool_ladder import LADDERS, Ladder, class_name, rung_of  # noqa: E402

NODE = "orion"
COMMIT = "0123456789abcdef0123456789abcdef01234567"   # a wave's commit (any, the same for all)
CLAIM_JOBS = 200         # claim: jobs, one class, one node, estimated over (0, fresh_minutes]
CLAIM_WORKERS = 4        # ... processes, each one worker
CLAIM_SLEEP = 0.010      # ... the most it sleeps between operations (s)
CLAIM_WAIT = 600.0       # ... the most the guard waits for a process (s)
SPLIT_SLEEP = 0.005      # split-tx: between the claim's select and its update (s)
# An open queue's feed (Screened's, decision 7: not built yet, so no steward resolves it; the
# cases of an open queue inject _Unclosed in its place): a fixed feed's seed is closed.
OPEN_FEED = "screened"


# ---- the queues the guards make, as a wave's plan makes them --------------------------------

def _header(ladder: Ladder) -> pq.Header:
    return pq.Header(wave="pool-check", commit=COMMIT, family="pool-check", ladder=ladder,
                     created=datetime.now(UTC).strftime(STAMP), feed=pq.FIXED)


def _seed_job(i: int, *, minutes: float, peak_gb: float, ladder: Ladder) -> pq.JobSeed:
    """Job `i` as a plan makes it from a stored peak and a time estimate of `minutes`: needing
    the peak times stored_mem, of the class of the smallest rung holding that; the long lane's
    past the top rung (then of the top class) or past pool_queue.fresh_minutes (keeping its
    rung's class)."""
    gb = peak_gb * ladder.stored_mem
    rung = rung_of(gb, ladder)
    pooled = rung is not None and minutes <= pq.fresh_minutes(ladder)
    block = f"KEN.1.1_1_{i}"
    return pq.JobSeed(id=i, block=block, item="lineup-score", group=block,
                      line=f"echo {block}", node=NODE,
                      cls=class_name(ladder.cards, ladder.rungs[-1] if rung is None else rung),
                      gb=gb, minutes=minutes, peak_gb=peak_gb, source="stored",
                      lane=pq.Lane.POOL if pooled else pq.Lane.LONG)


def _seed(ladder: Ladder, jobs: Sequence[pq.JobSeed]) -> pq.Seed:
    return pq.Seed(header=_header(ladder), jobs=tuple(jobs), closed=True)


def _as_job(s: pq.JobSeed, *, state: pq.JobState, worker: str | None = None) -> pq.Job:
    """The job `s` grants, as the queue should hold it with no strikes yet."""
    return pq.Job(id=s.id, block=s.block, item=s.item, group=s.group, line=s.line, node=s.node,
                  cls=s.cls, gb=s.gb, minutes=s.minutes, peak_gb=s.peak_gb, source=s.source,
                  lane=s.lane, state=state, strikes=0, worker=worker)


class Attempt(NamedTuple):
    """An attempt as the queue's table holds it (read directly: the Queue has no reader of it)."""
    id: int
    job: int
    worker: str
    outcome: str | None
    peak_gb: float | None
    rc: int | None
    rows: int


def _attempts(path: Path) -> list[Attempt]:
    db = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        return [Attempt(id=r["id"], job=r["job"], worker=r["worker"], outcome=r["outcome"],
                        peak_gb=r["peak_gb"], rc=r["rc"], rows=r["rows"])
                for r in db.execute("SELECT * FROM attempts ORDER BY id")]
    finally:
        db.close()


def _some(ids: Sequence[object], k: int = 5) -> str:
    return ", ".join(map(str, ids[:k])) + (f" ({len(ids)} in all)" if len(ids) > k else "")


# ---- claim ----------------------------------------------------------------------------------

class Claimed(NamedTuple):
    """One claim a process made: the attempt, its job and the job's estimate."""
    attempt: int
    job: int
    minutes: float


class Report(NamedTuple):
    """What one process did: its worker, its claims in order, and the error it died of."""
    worker: str
    claims: list[Claimed]
    error: str | None


def _claimer(path: Path, seed: pq.Seed, k: int, start: Barrier, out: MpQueue[Report]) -> None:
    left = pq.fresh_minutes(seed.header.ladder)     # a fresh worker's
    rng = random.Random(k)
    worker = f"pool-check-{k}"
    claims: list[Claimed] = []
    try:
        start.wait()
        q: pq.Queue = pq.SqliteQueue.open(path, seed)
        time.sleep(rng.uniform(0.0, CLAIM_SLEEP))
        q.register(worker=worker, run="pool-check", node=NODE, cls=seed.jobs[0].cls,
                   ends_at=time.time() + 3600.0, restarts=0)
        while True:
            time.sleep(rng.uniform(0.0, CLAIM_SLEEP))
            c = q.claim(worker=worker, minutes_left=left)
            if c is None:
                break
            claims.append(Claimed(attempt=c.attempt, job=c.job.id, minutes=c.job.minutes))
            time.sleep(rng.uniform(0.0, CLAIM_SLEEP))
            q.finish(c.attempt, worker=worker, outcome=pq.Outcome.DONE, peak_gb=1.0,
                     minutes=c.job.minutes, rc=0)
        q.retire(worker)
    except BaseException as e:
        out.put(Report(worker=worker, claims=claims, error=f"{type(e).__name__}: {e}"))
        raise
    out.put(Report(worker=worker, claims=claims, error=None))


def claim(_args: list[str]) -> list[str]:
    ladder = LADDERS["cpu-score"]
    fresh = pq.fresh_minutes(ladder)
    seed = _seed(ladder, [_seed_job(i, minutes=fresh * i / CLAIM_JOBS, peak_gb=3.0, ladder=ladder)
                          for i in range(1, CLAIM_JOBS + 1)])
    if {(j.node, j.cls, j.lane) for j in seed.jobs} != {(NODE, "m4", pq.Lane.POOL)}:
        return ["the jobs are not of one node and one pool class"]
    ctx = mp.get_context("fork")
    with tempfile.TemporaryDirectory(prefix="pool_check-") as tmp:
        path = Path(tmp) / "queue.db"
        start, out = ctx.Barrier(CLAIM_WORKERS), ctx.Queue()
        procs = [ctx.Process(target=_claimer, args=(path, seed, k, start, out))
                 for k in range(CLAIM_WORKERS)]
        for p in procs:
            p.start()
        reports: list[Report] = []
        try:
            for _ in procs:
                reports.append(out.get(timeout=CLAIM_WAIT))
        except queue.Empty:
            pass
        for p in procs:
            p.join(timeout=CLAIM_WAIT)
            if p.is_alive():       # hung: killed, so no claimer outlives the guard
                p.kill()
                p.join()
        bad = [f"{r.worker} died: {r.error}" for r in reports if r.error is not None]
        bad += [f"{CLAIM_WORKERS - len(reports)} processes reported nothing"] if len(
            reports) < CLAIM_WORKERS else []
        bad += [f"a process exited {p.exitcode}" for p in procs if p.exitcode != 0]
        if bad:
            print("claim: " + "; ".join(bad) + " -> FAILS", flush=True)
            return bad
        jobs = pq.SqliteQueue.open(path, seed).jobs()
        attempts = _attempts(path)
    tries: dict[int, int] = {}
    for a in attempts:
        tries[a.job] = tries.get(a.job, 0) + 1
    twice = sorted(j for j, n in tries.items() if n > 1)
    undone = [j.id for j in jobs if j.state != pq.JobState.DONE]
    held = {a.id: a.worker for a in attempts}
    other = [c.attempt for r in reports for c in r.claims
             if c.attempt not in held or held[c.attempt] != r.worker]
    rising = [r.worker for r in reports
              if any(b.minutes > a.minutes for a, b in pairwise(r.claims))]
    bad = ([f"jobs with more than one attempt: {_some(twice)}"] if twice else []) + (
        [f"jobs not done: {_some(undone)}"] if undone else []) + (
        [f"attempts finished by a worker other than their claimer: {_some(other)}"]
        if other else []) + ([f"workers whose estimates rose: {_some(rising)}"] if rising else [])
    if len(jobs) != CLAIM_JOBS:
        bad.append(f"{len(jobs)} jobs in the queue, not {CLAIM_JOBS}")
    print(f"claim: {CLAIM_WORKERS} workers, {len(jobs)} jobs, {len(attempts)} attempts (each "
          f"worker's claims: {', '.join(str(len(r.claims)) for r in reports)})"
          + (" -> ok" if not bad else " -> FAILS: " + "; ".join(bad)), flush=True)
    return bad


# ---- transitions ----------------------------------------------------------------------------

def _differs(got: pq.Job, want: pq.Job) -> list[str]:
    """The fields in which `got` is not `want`: floats within 0.01 (a time estimate raised to an
    attempt's minutes, read off the clock), the rest equal."""
    return [f"{name} {g!r}, want {w!r}"
            for name, g, w in zip(pq.Job._fields, got, want, strict=True)
            if not (math.isclose(g, w, abs_tol=0.01)
                    if isinstance(g, float) and isinstance(w, float) else g == w)]


class Registered(NamedTuple):
    """A worker a case registered, and its Slurm end time."""
    name: str
    ends: float


class Case:
    """One case of `transitions`: a fresh queue of `jobs` in its own file, and what it saw."""

    def __init__(self, path: Path, ladder: Ladder, jobs: Sequence[pq.JobSeed]) -> None:
        self.q: pq.Queue = pq.SqliteQueue.open(path, _seed(ladder, jobs))
        self.fresh = pq.fresh_minutes(ladder)     # a fresh worker's minutes left
        self.path = path
        self.notes: list[str] = []

    def job(self, i: int) -> pq.Job:
        return next(j for j in self.q.jobs() if j.id == i)

    def worker(self, name: str, *, cls: str, minutes: float, lease: str) -> Registered:
        """Register `name`, a worker of `cls` with `minutes` of its walltime left, of the run
        (a lease) `pool-check-<NODE>-<cls>-<lease>`."""
        ends = time.time() + 60.0 * minutes
        self.q.register(worker=name, run=f"pool-check-{NODE}-{cls}-{lease}", node=NODE, cls=cls,
                        ends_at=ends, restarts=0)
        return Registered(name=name, ends=ends)

    def claim(self, worker: str, *, left: float, want: pq.Job) -> pq.Claim | None:
        c = self.q.claim(worker=worker, minutes_left=left)
        if c is None:
            self.notes.append(f"{worker}'s claim: none, want job {want.id}")
        else:
            self.expect(f"{worker}'s claim", c.job, want)
        return c

    def expect(self, step: str, got: pq.Job, want: pq.Job) -> None:
        wrong = _differs(got, want)
        if wrong:
            self.notes.append(f"{step}: " + ", ".join(wrong))

    def after(self, step: str, state: pq.JobState | None, want: pq.Job) -> None:
        """The job as it stands after `step`, and the state the step returned (None: none)."""
        if state is not None and state != want.state:
            self.notes.append(f"{step} returned {state}, want {want.state}")
        self.expect(step, self.job(want.id), want)

    def check(self, step: str, ok: bool, what: str) -> None:
        if not ok:
            self.notes.append(f"{step}: {what}")


def _small(lad: Ladder, i: int = 1, minutes: float = 30.0) -> pq.JobSeed:
    """A job of class m4 under cpu-score (peak 3 GB, needing 3.6)."""
    return _seed_job(i, minutes=minutes, peak_gb=3.0, ladder=lad)


def _done(path: Path, lad: Ladder) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    w = c.worker("w1", cls=s.cls, minutes=120, lease="s1").name
    got = c.claim(w, left=c.fresh, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.DONE, peak_gb=3.1,
                        minutes=25.0, rc=0)
        c.after("finish DONE", st, _as_job(s, state=pq.JobState.DONE))
    return c.notes


def _failed_progress(path: Path, lad: Ladder) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    w = c.worker("w1", cls=s.cls, minutes=120, lease="s1").name
    got = c.claim(w, left=c.fresh, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        c.q.wrote(got.attempt)
        c.q.wrote(got.attempt)
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.FAILED, peak_gb=3.1,
                        minutes=25.0, rc=1)
        c.after("finish FAILED, two rows written", st, _as_job(s, state=pq.JobState.PENDING))
    return c.notes


def _strike_thrice(c: Case, s: pq.JobSeed) -> None:
    """Three claims of `s`, each by a worker of the next lease (no lease claims a job it has
    attempted again), each finished FAILED with no row written."""
    for n, state in enumerate((pq.JobState.PENDING, pq.JobState.PENDING, pq.JobState.FAILED), 1):
        w = c.worker(f"w{n}", cls=s.cls, minutes=120, lease=f"l{n}").name
        got = c.claim(w, left=c.fresh, want=_as_job(
            s, state=pq.JobState.CLAIMED, worker=w)._replace(strikes=n - 1))
        if got is None:
            return
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.FAILED, peak_gb=3.1,
                        minutes=25.0, rc=1)
        c.after(f"finish FAILED {n}, no row", st, _as_job(s, state=state)._replace(strikes=n))


def _failed_strikes(path: Path, lad: Ladder) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    _strike_thrice(c, s)
    return c.notes


def _oom_promote(path: Path, lad: Ladder) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    w = c.worker("w1", cls=s.cls, minutes=120, lease="s1").name
    got = c.claim(w, left=c.fresh, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.OOM, peak_gb=4.0,
                        minutes=5.0, rc=-9)
        c.after("finish OOM, peak 4 in m4", st, _as_job(s, state=pq.JobState.PENDING)._replace(
            cls="m12", peak_gb=4.0, gb=6.0))
    return c.notes


def _oom_top(path: Path, lad: Ladder) -> list[str]:
    s = _seed_job(1, minutes=30.0, peak_gb=70.0, ladder=lad)
    c = Case(path, lad, [s])
    c.check("the seed", s.cls == "m88" and s.lane == pq.Lane.POOL, f"{s.cls}, {s.lane}")
    w = c.worker("w1", cls=s.cls, minutes=120, lease="s1").name
    got = c.claim(w, left=c.fresh, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.OOM, peak_gb=88.0,
                        minutes=5.0, rc=-9)
        c.after("finish OOM, peak 88 in m88", st, _as_job(s, state=pq.JobState.LONG)._replace(
            lane=pq.Lane.LONG, peak_gb=88.0, gb=132.0))
    return c.notes


def _reaped(path: Path, lad: Ladder, *, rows: int) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    w = c.worker("w1", cls=s.cls, minutes=120, lease="s1").name
    got = c.claim(w, left=c.fresh, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        for _ in range(rows):
            c.q.wrote(got.attempt)
        r = c.q.reap(w, now=time.time())
        c.check("reap", r == pq.Reaped(worker=w, run=f"pool-check-{NODE}-m4-s1", job=s.id,
                                       outcome=pq.Outcome.REAPED, walltime=False), f"{r}")
        c.after(f"reap before its end, {rows} rows", None, _as_job(
            s, state=pq.JobState.PENDING)._replace(strikes=0 if rows else 1))
    return c.notes


def _walltime(path: Path, lad: Ladder, *, ran: float, want: pq.JobState) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    reg = c.worker("w1", cls=s.cls, minutes=ran, lease="s1")
    w = reg.name
    got = c.claim(w, left=ran - lad.margin_minutes, want=_as_job(
        s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        r = c.q.reap(w, now=reg.ends + 60.0)
        c.check("reap", r is not None and (r.job, r.outcome, r.walltime) == (
            s.id, pq.Outcome.WALLTIME, True), f"{r}")
        c.after(f"reap after its end, {ran:g} minutes run", None, _as_job(s, state=want)._replace(
            minutes=ran, lane=pq.Lane.LONG if want == pq.JobState.LONG else pq.Lane.POOL))
    return c.notes


def _late_finish(path: Path, lad: Ladder) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    w = c.worker("w1", cls=s.cls, minutes=120, lease="s1").name
    got = c.claim(w, left=c.fresh, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        c.q.reap(w, now=time.time())
        reaped = _as_job(s, state=pq.JobState.PENDING)._replace(strikes=1)
        c.after("reap", None, reaped)
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.DONE, peak_gb=3.1,
                        minutes=25.0, rc=0)
        c.after("the reaped worker's finish DONE", st, reaped)
        a = next((a for a in _attempts(c.path) if a.id == got.attempt), None)
        c.check("its attempt", a is not None and a.rc == 0 and a.peak_gb == 3.1,
                f"{a}, want rc 0 and peak 3.1 recorded")
    return c.notes


def _retry(path: Path, lad: Ladder) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    _strike_thrice(c, s)
    n = c.q.retry([s.id])
    c.check("retry", n == 1, f"{n} retried, want 1")
    c.after("retry", None, _as_job(s, state=pq.JobState.PENDING))
    return c.notes


def _cancel(path: Path, lad: Ladder) -> list[str]:
    held, waits = _small(lad, 1, minutes=30.0), _small(lad, 3, minutes=20.0)
    long = _seed_job(2, minutes=30.0, peak_gb=100.0, ladder=lad)
    c = Case(path, lad, [held, long, waits])
    c.check("the seed", long.lane == pq.Lane.LONG, f"job 2's lane {long.lane}")
    w = c.worker("w1", cls=held.cls, minutes=120, lease="s1").name
    got = c.claim(w, left=c.fresh, want=_as_job(held, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        n = c.q.cancel()
        c.check("cancel", n == 3, f"{n} cancelled, want 3")
        cancelled = _as_job(held, state=pq.JobState.CANCELLED)
        c.after("cancel", None, cancelled)
        c.after("cancel", None, _as_job(long, state=pq.JobState.CANCELLED))
        c.after("cancel", None, _as_job(waits, state=pq.JobState.CANCELLED))
        # the claimed job's worker ends it later, failed with no row: no strike, still cancelled
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.FAILED, peak_gb=3.1,
                        minutes=25.0, rc=1)
        c.after("its worker's later finish FAILED", st, cancelled)
        a = next((a for a in _attempts(c.path) if a.id == got.attempt), None)
        c.check("its attempt", a is not None and (a.outcome, a.rc) == (
            pq.Outcome.CANCELLED.value, 1), f"{a}, want cancelled with rc 1 recorded")
        c.check("its worker's retire", c.q.retire(w) == 0, "settled a job")
        c.after("its worker's retire", None, cancelled)
    return c.notes


def _claim_job(path: Path, lad: Ladder) -> list[str]:
    s = _seed_job(1, minutes=30.0, peak_gb=100.0, ladder=lad)
    c = Case(path, lad, [s])
    pool = c.worker("pool", cls=s.cls, minutes=120, lease="s1").name
    c.check("a pool's claim", c.q.claim(worker=pool, minutes_left=c.fresh) is None,
            "took the LONG job")
    w = c.worker("long", cls=s.cls, minutes=3 * 24 * 60, lease="s1").name
    got = c.q.claim_job(worker=w, job=s.id)
    if got is None:
        c.notes.append("claim_job: none, want the LONG job")
    else:
        c.expect("claim_job", got.job, _as_job(s, state=pq.JobState.CLAIMED, worker=w))
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.FAILED, peak_gb=50.0,
                        minutes=25.0, rc=1)
        c.after("finish FAILED, no row", st, _as_job(s, state=pq.JobState.LONG)._replace(
            strikes=1))
    return c.notes


def _aside(path: Path, lad: Ladder) -> list[str]:
    jobs = [_small(lad, 1, minutes=90.0), _small(lad, 2, minutes=30.0)]
    c = Case(path, lad, jobs)
    since = time.time()
    c.worker("w-m12", cls="m12", minutes=120, lease="s1")
    for name, s in (("w-a", jobs[0]), ("w-b", jobs[1])):
        w = c.worker(name, cls=s.cls, minutes=120, lease="s1").name
        c.claim(w, left=c.fresh, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    c.worker("w-c", cls="m4", minutes=120, lease="s1")
    first = c.q.ask_aside(node=NODE, cls="m4", units=2, since=since)
    c.check("ask_aside", first == ["w-c", "w-b"], f"{first}, want ['w-c', 'w-b']")
    again = c.q.ask_aside(node=NODE, cls="m4", units=2, since=since)
    c.check("ask_aside again", again == [], f"{again}, want none")
    asked = {w: c.q.asked_aside(w) for w in ("w-m12", "w-a", "w-b", "w-c")}
    c.check("asked_aside", asked == {"w-m12": False, "w-a": False, "w-b": True, "w-c": True},
            f"{asked}")
    return c.notes


def _other_worker(path: Path, lad: Ladder) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    w = c.worker("w1", cls=s.cls, minutes=120, lease="s1").name
    other = c.worker("w2", cls=s.cls, minutes=120, lease="s1").name
    claimed = _as_job(s, state=pq.JobState.CLAIMED, worker=w)
    got = c.claim(w, left=c.fresh, want=claimed)
    if got is not None:
        try:
            st = c.q.finish(got.attempt, worker=other, outcome=pq.Outcome.DONE, peak_gb=3.1,
                            minutes=25.0, rc=0)
            c.notes.append(f"{other}'s finish DONE of {w}'s attempt returned {st}, want refused")
        except ValueError:
            pass
        c.after(f"{other}'s finish", None, claimed)
        a = next((a for a in _attempts(c.path) if a.id == got.attempt), None)
        c.check("its attempt", a is not None and a.rc is None and a.worker == w,
                f"{a}, want {w}'s and unrecorded")
    return c.notes


def _one_claim(path: Path, lad: Ladder) -> list[str]:
    first, second = _small(lad, 1, minutes=30.0), _small(lad, 2, minutes=20.0)
    c = Case(path, lad, [first, second])
    w = c.worker("w1", cls=first.cls, minutes=120, lease="s1").name
    if c.claim(w, left=c.fresh, want=_as_job(first, state=pq.JobState.CLAIMED, worker=w)) is None:
        return c.notes
    tries: tuple[tuple[str, Callable[[], pq.Claim | None]], ...] = (
        ("claim", lambda: c.q.claim(worker=w, minutes_left=c.fresh)),
        ("claim_job", lambda: c.q.claim_job(worker=w, job=second.id)),
        # of the job it holds: refused, not None (the one-claim check comes before the job's
        # state is read, so a check moved back after it would answer None here)
        ("claim_job of its own job", lambda: c.q.claim_job(worker=w, job=first.id)))
    for how, again in tries:
        try:
            got = again()
            c.notes.append(f"a second {how} by {w}: {got and got.job.id}, want refused")
        except ValueError:
            pass
    c.after("the second claims", None, _as_job(second, state=pq.JobState.PENDING))
    return c.notes


def _excluded(path: Path, lad: Ladder) -> list[str]:
    longer, shorter = _small(lad, 1, minutes=60.0), _small(lad, 2, minutes=30.0)
    c = Case(path, lad, [longer, shorter])
    w1 = c.worker("w1", cls=longer.cls, minutes=120, lease="s1").name
    got = c.claim(w1, left=c.fresh, want=_as_job(longer, state=pq.JobState.CLAIMED, worker=w1))
    if got is None:
        return c.notes
    c.q.finish(got.attempt, worker=w1, outcome=pq.Outcome.FAILED, peak_gb=3.1, minutes=25.0,
               rc=1)
    struck = _as_job(longer, state=pq.JobState.PENDING)._replace(strikes=1)
    c.after("finish FAILED, no row", None, struck)
    # another task of the same lease: the shorter job, not the longer one its lease attempted
    w2 = c.worker("w2", cls=longer.cls, minutes=120, lease="s1").name
    nxt = c.claim(w2, left=c.fresh, want=_as_job(shorter, state=pq.JobState.CLAIMED, worker=w2))
    if nxt is None:
        return c.notes
    c.q.finish(nxt.attempt, worker=w2, outcome=pq.Outcome.DONE, peak_gb=3.1, minutes=25.0, rc=0)
    # only the job the lease attempted pending: none for any of its workers, the job as it was
    for w in (w1, w2):
        none = c.q.claim(worker=w, minutes_left=c.fresh)
        c.check(f"{w}'s claim", none is None, f"job {none and none.job.id}, want none")
    c.after("the lease's claims", None, struck)
    # the next lease's worker takes it
    w3 = c.worker("w3", cls=longer.cls, minutes=120, lease="l2").name
    c.claim(w3, left=c.fresh, want=struck._replace(state=pq.JobState.CLAIMED, worker=w3))
    return c.notes


#: The cases of `transitions`, by name, each on a fresh queue under cpu-score.
TRANSITIONS: tuple[tuple[str, Callable[[Path, Ladder], list[str]]], ...] = (
    ("done", _done),
    ("failed-progress", _failed_progress),
    ("failed-strikes", _failed_strikes),
    ("oom-promote", _oom_promote),
    ("oom-top", _oom_top),
    ("reap-progress", lambda p, lad: _reaped(p, lad, rows=1)),
    ("reap-strike", lambda p, lad: _reaped(p, lad, rows=0)),
    ("walltime", lambda p, lad: _walltime(p, lad, ran=50.0, want=pq.JobState.PENDING)),
    ("walltime-long", lambda p, lad: _walltime(p, lad, ran=115.0, want=pq.JobState.LONG)),
    ("late-finish", _late_finish),
    ("retry", _retry),
    ("cancel", _cancel),
    ("claim-job", _claim_job),
    ("aside", _aside),
    ("other-worker", _other_worker),
    ("one-claim", _one_claim),
    ("exclude", _excluded),
)


def transitions(_args: list[str]) -> list[str]:
    lad = LADDERS["cpu-score"]
    if (lad.lease_minutes, lad.margin_minutes, lad.startup_minutes, lad.strikes, lad.fit_mem,
            lad.rungs) != (120, 10, 5, 3, 1.5, (4, 12, 24, 56, 88)):
        return [f"cpu-score is not the cases' (lease 120, margin 10, startup 5, strikes 3, "
                f"fit_mem 1.5, rungs 4-88): {lad}"]
    bad: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pool_check-") as tmp:
        for name, case in TRANSITIONS:
            notes = case(Path(tmp) / f"{name}.db", lad)
            print(f"{name}: " + ("ok" if not notes else "FAILS: " + "; ".join(notes)),
                  flush=True)
            if notes:
                bad.append(name)
    return bad


class Guard(NamedTuple):
    """A guard: what runs it (its arguments -> its failures), and its arguments' names, in order
    (the command line takes exactly these)."""
    run: Callable[[list[str]], list[str]]
    args: tuple[str, ...]


# ---- class ----------------------------------------------------------------------------------

CLASS_NODES = ("orion", "quadro2")
CLASS_ITEM = "cycle_native_betweenness_contrast"    # leave-one-out: cycle_desire's proposals
CLASS_BLOCKS = 12        # plan: the stored lineup blocks with the most stored work
# Facts of the stored rows (2026-10-21) that no reader supplies, pinning the scoring-only ruling
# (pool_plan.PROPOSED_AND_SCORED): scored alone, every stored lineup network fits a rung of
# cpu-score, and 62803's cycle_desire network (16.6 GB scored alone, 165 GB where it was proposed
# in the same process) is m24's.
SCORED_ALONE_BLOCK = "ZAF.9.3.1_1_62803"
SCORED_ALONE_CLASS = "m24"


def _now() -> str:
    return datetime.now(UTC).strftime(STAMP)


class Stored(NamedTuple):
    """A (block, item)'s stored basis, recomputed here: its largest peak, its longest wall."""
    peak_gb: float
    minutes: float


def _stored(runs: Sequence[pp.Observed]) -> dict[tuple[str, str], Stored]:
    out: dict[tuple[str, str], Stored] = {}
    for o in runs:
        k = (o.block, o.item)
        out[k] = (Stored(peak_gb=max(out[k].peak_gb, o.peak_gb),
                         minutes=max(out[k].minutes, o.minutes)) if k in out
                  else Stored(peak_gb=o.peak_gb, minutes=o.minutes))
    return out


def _lane_wrong(j: pq.JobSeed, *, peak: float, gb: float, minutes: float,
                lad: Ladder) -> list[str]:
    """`j` against its estimate (its basis `peak`, needing `gb` and `minutes`), both ways: a job a
    rung holds whose minutes fit a fresh lease is a pool's, in that rung's class; any other is
    the long lane's, needing fit_mem times its peak, in the class of the rung holding its gb (the
    top rung's past every rung)."""
    r = rung_of(gb, lad)
    want = class_name(lad.cards, lad.rungs[-1] if r is None else r)
    if r is not None and minutes <= pq.fresh_minutes(lad):
        return ([] if (j.lane, j.cls) == (pq.Lane.POOL, want) else
                [f"{j.block} {j.item}: {j.lane} in {j.cls}, want pool in {want} ({gb:.1f} GB, "
                 f"{minutes:.0f} min)"])
    need = lad.fit_mem * peak
    if (j.lane, j.cls) == (pq.Lane.LONG, want) and math.isclose(j.gb, need, rel_tol=1e-9):
        return []
    return [f"{j.block} {j.item}: {j.lane} in {j.cls} needing {j.gb:.1f} GB, want long in {want} "
            f"needing {need:.1f} ({gb:.1f} GB, {minutes:.0f} min)"]


def _failing(notes: Sequence[str]) -> str:
    return (" -> ok" if not notes else " -> FAILS: " + "; ".join(notes[:5])
            + (f" ({len(notes)} in all)" if len(notes) > 5 else ""))


@dataclass(frozen=True, kw_only=True)
class _Without:
    """`family` with `block`'s runs removed: that block as if never run (the leave-one-out's)."""

    family: pp.Family
    block: str

    @property
    def name(self) -> str:
        return self.family.name

    def stored(self, block: str, item: str) -> pp.Observed | None:
        return None if block == self.block else self.family.stored(block, item)

    def observed(self) -> Sequence[pp.Observed]:
        return [o for o in self.family.observed() if o.block != self.block]


def _stored_classes(lineup: pp.Family, lad: Ladder, areas: dict[str, float]) -> list[str]:
    stored = _stored(lineup.observed())
    est = pp.Estimator(family=lineup, ladder=lad, areas=areas, history=(),
                       fit_min_km2=pp.FIT_MIN_KM2, fit_min_blocks=pp.FIT_MIN_BLOCKS)
    seed = pp.plan_wave(wave="pool-check", commit=COMMIT, family=lineup.name, ladder=lad,
                        estimator=est, nodes=CLASS_NODES, usable_gb=pp.USABLE_GB,
                        ids=sorted({b for b, _ in stored}), items=sorted({i for _, i in stored}),
                        line=lambda b, i: f"echo {b} {i}", created=_now())
    notes: list[str] = []
    by: Counter[str] = Counter()
    lanes: Counter[str] = Counter()
    for j in seed.jobs:
        if (j.block, j.item) not in stored:
            continue                     # no stored run: the fit's (leave-one-out's to check)
        s = stored[(j.block, j.item)]
        gb, minutes = 1.2 * s.peak_gb, 1.25 * s.minutes
        r = rung_of(gb, lad)
        by["past 88" if r is None else class_name(0, r)] += 1
        lanes[j.lane.value] += 1
        notes += _lane_wrong(j, peak=s.peak_gb, gb=gb, minutes=minutes, lad=lad)
        if not math.isclose(j.minutes, minutes, rel_tol=1e-9):
            notes.append(f"{j.block} {j.item}: {j.minutes:.2f} minutes, want 1.25 x "
                         f"{s.minutes:.2f}")
    past = by["past 88"]
    if past:
        notes.append(f"{past} stored lineup networks past the top rung: scored alone, none is")
    fact = next((j for j in seed.jobs if (j.block, j.item) == (SCORED_ALONE_BLOCK, CLASS_ITEM)),
                None)
    if fact is None or (fact.lane, fact.cls) != (pq.Lane.POOL, SCORED_ALONE_CLASS):
        notes.append(f"{SCORED_ALONE_BLOCK}'s {CLASS_ITEM}: "
                     + ("not planned" if fact is None else f"{fact.lane} in {fact.cls}")
                     + f", want pool in {SCORED_ALONE_CLASS} (scored alone)")
    print(f"stored: {len(stored)} stored (block, method)s of {len({b for b, _ in stored})} blocks "
          f"under cpu-score, by class " + ", ".join(f"{c} {by[c]}" for c in sorted(
              by, key=lambda c: math.inf if c == "past 88" else int(c[1:])))
          + "; by lane " + ", ".join(f"{k} {n}" for k, n in sorted(lanes.items()))
          + ("" if fact is None else f"; {SCORED_ALONE_BLOCK}'s {CLASS_ITEM} {fact.lane} in "
                                     f"{fact.cls}")
          + _failing(notes), flush=True)
    return ["stored"] if notes else []


class Held(NamedTuple):
    """A leave-one-out estimate against its block's measured peak, and the rate that set it."""
    ratio: float          # the estimate over the measured peak
    block: str
    measured: float       # GB
    area: float           # km2
    gb: float             # the estimate
    scope: str            # whose jobs the fit was over: the item's or the family's
    by: str               # the block whose per-km2 peak is the worst of those jobs
    rate: float           # GB/km2


def _left_out(propose: pp.Family, lad: Ladder, areas: dict[str, float],
              large: Sequence[pp.Observed], item: str) -> tuple[list[Held], list[str]]:
    """`item`'s leave-one-out over its jobs in `large` (the family's on blocks of at least
    FIT_MIN_KM2): each estimate against 1.5 times the worst per-km2 peak of the others (the item's
    when they are on FIT_MIN_BLOCKS blocks, else the family's: recomputed here) times its area."""
    jobs = [o for o in large if o.item == item]
    notes: list[str] = []
    held: list[Held] = []
    for blk in sorted({o.block for o in jobs}):
        est = pp.Estimator(family=_Without(family=propose, block=blk), ladder=lad, areas=areas,
                           history=(), fit_min_km2=pp.FIT_MIN_KM2,
                           fit_min_blocks=pp.FIT_MIN_BLOCKS).estimate(blk, item)
        others = [o for o in large if o.block != blk]
        own = [o for o in others if o.item == item]
        scope, pool = (("item", own) if len({o.block for o in own}) >= pp.FIT_MIN_BLOCKS
                       else ("family", others))
        worst = max(pool, key=lambda o: o.peak_gb / areas[o.block])
        rate = worst.peak_gb / areas[worst.block]
        want = 1.5 * rate * areas[blk]
        measured = max(o.peak_gb for o in jobs if o.block == blk)
        if not math.isclose(est.gb(), want, rel_tol=1e-9):
            notes.append(f"{item} {blk}: {est.gb():.2f} GB, want 1.5 x {rate:.3f} GB/km2 ({scope}) "
                         f"x {areas[blk]:.3f} km2 = {want:.2f}")
        if not est.source.startswith(f"fit:{scope}:"):
            notes.append(f"{item} {blk}: source {est.source!r}, want the {scope}'s fit")
        held.append(Held(ratio=est.gb() / measured, block=blk, measured=measured,
                         area=areas[blk], gb=est.gb(), scope=scope, by=worst.block, rate=rate))
    return held, notes


def _leave_one_out(propose: pp.Family, lad: Ladder, areas: dict[str, float]) -> list[str]:
    large = [o for o in propose.observed() if areas[o.block] >= pp.FIT_MIN_KM2]
    bad: list[str] = []
    for item in sorted({o.item for o in large}):
        held, notes = _left_out(propose, lad, areas, large, item)
        below = [h for h in held if h.ratio < 1.0]
        if item == CLASS_ITEM:
            notes += [f"{item} {h.block}: {h.gb:.1f} GB, below its measured {h.measured:.1f}"
                      for h in below]
        t = min(held)
        scopes = Counter(h.scope for h in held)
        print(f"leave-one-out {item}: {sum(1 for o in large if o.item == item)} jobs on "
              f"{len(held)} blocks of at least {pp.FIT_MIN_KM2:g} km2, fit by "
              + ", ".join(f"{s} {n}" for s, n in sorted(scopes.items()))
              + f"; {len(below)} below the measured peak; the tightest {t.block} "
              f"({t.measured:.1f} GB on {t.area:.2f} km2, {t.measured / t.area:.1f} GB/km2) at "
              f"{t.gb:.1f} GB, by {t.by}'s {t.rate:.1f} GB/km2 ({t.scope})"
              + (" -> ok" if not notes else " -> FAILS: " + "; ".join(notes[:5])
                 + (f" ({len(notes)} in all)" if len(notes) > 5 else "")), flush=True)
        bad += [f"leave-one-out {item}"] if notes else []
    if not any(o.item == CLASS_ITEM for o in large):
        print(f"leave-one-out: no {CLASS_ITEM} job on a block of at least "
              f"{pp.FIT_MIN_KM2:g} km2 -> FAILS", flush=True)
        bad.append("leave-one-out")
    return bad


def _placed(lineup: pp.Family, lad: Ladder, areas: dict[str, float]) -> list[str]:
    stored = _stored(lineup.observed())
    work = dict.fromkeys(sorted({b for b, _ in stored}), 0.0)
    for (b, _), s in stored.items():
        work[b] += s.minutes
    blocks = sorted(sorted(work, key=lambda b: -work[b])[:CLASS_BLOCKS])
    items = sorted({i for _, i in stored})
    est = pp.Estimator(family=lineup, ladder=lad, areas=areas, history=(),
                       fit_min_km2=pp.FIT_MIN_KM2, fit_min_blocks=pp.FIT_MIN_BLOCKS)
    usable = pp.USABLE_GB
    seed = pp.plan_wave(wave="pool-check", commit=COMMIT, family=lineup.name, ladder=lad,
                        estimator=est, nodes=CLASS_NODES, usable_gb=usable, ids=blocks,
                        items=items, line=lambda b, i: f"echo {b} {i}", created=_now())
    rung_gb = {class_name(lad.cards, r): float(r) for r in lad.rungs}
    notes: list[str] = []
    need = dict.fromkeys(blocks, 0.0)            # the most a group's jobs need of its node
    pooled = dict.fromkeys(blocks, 0.0)          # a group's pool minutes
    on: dict[str, set[str]] = {}
    for j in seed.jobs:
        e = est.estimate(j.block, j.item)
        notes += _lane_wrong(j, peak=e.peak_gb, gb=e.gb(), minutes=e.needed_minutes(), lad=lad)
        hold = rung_gb[j.cls] if j.lane == pq.Lane.POOL else j.gb
        if hold > usable[j.node]:
            notes.append(f"{j.block} {j.item}: {j.lane} needing {hold:.0f} GB on {j.node} "
                         f"({usable[j.node]:g} GB usable)")
        need[j.block] = max(need[j.block], hold)
        if j.lane == pq.Lane.POOL:
            pooled[j.block] += j.minutes
        on.setdefault(j.block, set()).add(j.node)
        if j.group != j.block:
            notes.append(f"job {j.id}: group {j.group}, block {j.block}")
    split = {b: sorted(n) for b, n in on.items() if len(n) != 1}
    if split:
        notes.append(f"blocks on more than one node: {split}")
    if len(seed.jobs) != len(blocks) * len(items):
        notes.append(f"{len(seed.jobs)} jobs, not {len(blocks)} x {len(items)}")
    # The placement replayed: the groups longest-processing-time first on their pool minutes,
    # each to the least loaded of the nodes that hold its every job (a long job's minutes outside
    # the balance).
    load = dict.fromkeys(CLASS_NODES, 0.0)
    for b in sorted(blocks, key=lambda b: -pooled[b]):
        able = [n for n in CLASS_NODES if need[b] <= usable[n]]
        want = min(able, key=lambda n: load[n]) if able else None
        got = sorted(on[b])
        if got != [want]:
            loads = ", ".join(f"{n} {load[n]:.0f}" for n in CLASS_NODES)
            notes.append(f"{b} on {', '.join(got)}, want {want} ({pooled[b]:.0f} pool minutes, "
                         f"needing {need[b]:.0f} GB; loads {loads})")
        load[got[0]] += pooled[b]
    long = [f"{j.block} {j.item} ({j.minutes:.0f} min, {j.gb:.0f} GB, {j.node})"
            for j in seed.jobs if j.lane == pq.Lane.LONG]
    print(f"plan: {len(blocks)} blocks, {len(seed.jobs)} jobs over "
          + ", ".join(f"{n} ({usable[n]:g} GB usable)" for n in CLASS_NODES) + ": pool minutes "
          + ", ".join(f"{n} {load[n]:.0f} ({sum(1 for b in on if n in on[b])} blocks)"
                      for n in CLASS_NODES)
          + f"; long: {', '.join(long) or 'none'}" + _failing(notes), flush=True)
    return ["plan"] if notes else []


def classes(args: list[str]) -> list[str]:
    [rows] = args
    root = Path(rows)
    lad = LADDERS["cpu-score"]
    if (lad.stored_mem, lad.stored_time, lad.fit_mem, lad.rungs) != (1.2, 1.25, 1.5,
                                                                      (4, 12, 24, 56, 88)):
        return [f"cpu-score is not the guard's (stored_mem 1.2, stored_time 1.25, fit_mem 1.5, "
                f"rungs 4-88): {lad}"]
    areas = pp.areas_of(root / "consolidated_rows" / "summary_features.csv", {})
    lineup, propose = pp.FAMILIES["lineup-score"](root), pp.FAMILIES["propose"](root)
    return (_stored_classes(lineup, lad, areas) + _leave_one_out(propose, lad, areas)
            + _placed(lineup, lad, areas))


# ---- the worker: walltime, oom, identity, worker --------------------------------------------

WORKER = "w1"                     # the worker the cases run (pool_worker.work)
OTHER_COMMIT = "f" * 40           # identity: a checkout at another commit than the queue's
WALLTIME_LEFT = 40.0              # walltime: minutes of walltime left
WALLTIME_JOBS = (20.0, 29.0, 31.0, 60.0)     # ... the jobs' estimates (minutes)
OOM_BEFORE = 5                    # oom: the fake count before a job
NO_CGROUP = "the guard's: no cgroup to read"  # oom: an Unreadable evidence's why
#: A child that prints its own oom_score_adj (bash's `read` reads it in the shell itself) and
#: kills itself with SIGKILL, as the OOM killer would.
SIGKILLED = 'read -r adj < /proc/self/oom_score_adj; echo "oom_score_adj $adj"; kill -KILL $$'
TASK_JOB = "4242"                 # worker: the Slurm array job of pool.py's tasks
RUN = f"pool-check-{NODE}-m4-s1"         # the run (a lease) of the cases' workers
NEXT_RUN = f"pool-check-{NODE}-m4-l2"    # ... and the next lease's
WALLTIME_SPENT = 15.0             # walltime, the clock moving: the minutes each job spends
RAISE_SLEEP = 5.0                 # raises: the child's sleep after its first line (s)
RAISE_LIMIT = 2.0                 # ... the most the worker takes to raise (s): the child killed
#: group: a compound line whose shell forks a grandchild (sh, not the list's last command), which
#: prints its pid, then `started`, and sleeps RAISE_SLEEP
GRANDCHILD = ("true && sh -c 'echo \"grandchild $$\"; echo started; "
              f"exec sleep {RAISE_SLEEP:g}'; echo after")
GONE_WAIT = 2.0                   # ... the most the grandchild takes to be gone after the kill (s)
QUADRO = "quadro2"                # a node other than NODE


class Clock:
    """A fake clock for pool_worker.work: `now` stands still but for `wait`, a poll at a time."""

    def __init__(self, *, start: float, poll: float) -> None:
        self.t, self.poll, self.waits = start, poll, 0

    def now(self) -> float:
        return self.t

    def wait(self) -> None:
        self.t += self.poll
        self.waits += 1


@dataclass(kw_only=True)
class Counted:
    """An OomEvidence whose count the guard raises after a job, as the kernel would on an OOM
    kill of a process of the worker's cgroup."""

    count: int

    @property
    def about(self) -> str:
        return f"the guard's count, {self.count}"

    def kills(self) -> int | None:
        return self.count


def _line(i: int, *, minutes: float, line: str, peak_gb: float) -> pq.JobSeed:
    """Job `i` of `_seed_job` (cpu-score, on NODE) running `line`."""
    return replace(_seed_job(i, minutes=minutes, peak_gb=peak_gb, ladder=LADDERS["cpu-score"]),
                   line=line)


def _m4(i: int, *, minutes: float, line: str) -> pq.JobSeed:
    """A job of class m4 under cpu-score (peak 3 GB) running `line`."""
    return _line(i, minutes=minutes, line=line, peak_gb=3.0)


class Worked(NamedTuple):
    """What a case's worker did: its exit, its log, the jobs and attempts after, its polls."""

    exit: int
    log: list[str]
    jobs: dict[int, pq.Job]
    attempts: list[Attempt]
    waits: int
    workers: list[pq.Worker]


After = Callable[[pq.Queue, pq.Claim, Clock], None]


def _nothing(q: pq.Queue, c: pq.Claim, clock: Clock) -> None:
    """A case's step after a job: none."""


def _work(path: Path, seed: pq.Seed, *, worker: str, run: str, commit: str,
          minutes_left: float, oom: pw.OomEvidence, after: After) -> Worked:
    """pool_worker.work as `worker` of `run` (of NODE, m4) on a queue at `path` made from `seed`
    (or opened, made already), its checkout at `commit`, `minutes_left` of walltime on a fake
    clock (Clock), each job a real child (pool_worker.run_job in `path`'s directory), `after`
    called after each (with the clock, which it may move)."""
    q: pq.Queue = pq.SqliteQueue.open(path, seed)
    clock = Clock(start=time.time(), poll=seed.header.ladder.poll_seconds)
    log: list[str] = []

    def run_line(c: pq.Claim) -> pw.Ran:
        ran = pw.run_job(c.job.line, cwd=path.parent, env=os.environ, log=log.append,
                         wrote=lambda: q.wrote(c.attempt))
        after(q, c, clock)
        return ran

    ex = pw.work(queue=q, worker=worker, restarts=0, run=run, node=NODE, cls="m4",
                 commit=commit, ends_at=clock.now() + 60.0 * minutes_left, now=clock.now,
                 wait=clock.wait, run_line=run_line, oom=oom, log=log.append)
    return Worked(exit=int(ex), log=log, jobs={j.id: j for j in q.jobs()},
                  attempts=_attempts(path), waits=clock.waits, workers=q.workers())


def _tried(w: Worked) -> list[tuple[int, str | None, int | None]]:
    """Each attempt's job, outcome and code, in order."""
    return [(a.job, a.outcome, a.rc) for a in w.attempts]


def _state(w: Worked, i: int, *, state: pq.JobState, cls: str, strikes: int) -> list[str]:
    j = w.jobs[i]
    got = (j.state, j.cls, j.lane, j.strikes)
    want = (state, cls, pq.Lane.POOL, strikes)
    return [] if got == want else [f"job {i} {got}, want {want}"]


def walltime(_args: list[str]) -> list[str]:
    lad = LADDERS["cpu-score"]
    if lad.margin_minutes != 10:
        return [f"cpu-score's margin is {lad.margin_minutes}, not the guard's 10"]
    jobs = [_m4(i, minutes=m, line="true") for i, m in enumerate(WALLTIME_JOBS, 1)]

    def spend(q: pq.Queue, c: pq.Claim, clock: Clock) -> None:
        clock.t += 60.0 * WALLTIME_SPENT

    bad: list[str] = []
    # the clock still: 29 then 20 fit the 30 minutes; the clock moving 15 minutes a job: after
    # the 29-minute job 15 are left, which the 20-minute job no longer fits
    for name, after, want in (("still", _nothing, [29.0, 20.0]),
                              ("moving", spend, [29.0])):
        with tempfile.TemporaryDirectory(prefix="pool_check-") as tmp:
            w = _work(Path(tmp) / "queue.db", _seed(lad, jobs), worker=WORKER, run=RUN,
                      commit=COMMIT, minutes_left=WALLTIME_LEFT, oom=Counted(count=0),
                      after=after)
        ran = [w.jobs[a.job].minutes for a in w.attempts]
        pending = sorted(j.minutes for j in w.jobs.values() if j.state == pq.JobState.PENDING)
        left = sorted(set(WALLTIME_JOBS) - set(want))
        notes = ([] if ran == want else [f"ran jobs estimated {ran}, want {want}"]) + (
            [] if pending == left else [f"pending {pending}, want {left}"]) + (
            [] if w.exit == pw.Exit.RETIRED else [f"exit {w.exit}, want 75"])
        print(f"walltime {name}: {WALLTIME_LEFT:g} minutes left less a margin of "
              f"{lad.margin_minutes:g}" + ("" if after is _nothing else
                                           f", each job spending {WALLTIME_SPENT:g}")
              + f", jobs estimated {list(WALLTIME_JOBS)}: ran {ran}, pending {pending}, exit "
              f"{w.exit}" + _failing(notes), flush=True)
        bad += [name] if notes else []
    return bad


def _bump_job_1(ev: Counted) -> After:
    """After job 1: the count one more (its OOM kill)."""
    def bump(q: pq.Queue, c: pq.Claim, clock: Clock) -> None:
        if c.job.id == 1:
            ev.count += 1
    return bump


def _oom_cases(tmp: Path) -> list[str]:
    lad = LADDERS["cpu-score"]
    bad: list[str] = []
    adj = Path("/proc/self/oom_score_adj")
    mine = int(adj.read_text())
    if mine >= pw.OOM_SCORE_ADJ:
        return [f"this process's oom_score_adj is {mine}: a child's {pw.OOM_SCORE_ADJ} would be "
                "inherited, not set by the worker"]
    # promote: job 1 SIGKILLed with one more kill counted, then job 2
    ev = Counted(count=OOM_BEFORE)
    w = _work(tmp / "promote.db", _seed(lad, [_m4(1, minutes=60.0, line=SIGKILLED),
                                             _m4(2, minutes=30.0, line="true")]),
              worker=WORKER, run=RUN, commit=COMMIT,
              minutes_left=120.0, oom=ev, after=_bump_job_1(ev))
    printed = [x for x in w.log if x.startswith("oom_score_adj ")]
    notes = _state(w, 1, state=pq.JobState.PENDING, cls="m12", strikes=0) + _state(
        w, 2, state=pq.JobState.DONE, cls="m4", strikes=0)
    notes += [] if _tried(w) == [(1, "oom", -9), (2, "done", 0)] else [f"attempts {_tried(w)}"]
    notes += [] if printed == [f"oom_score_adj {pw.OOM_SCORE_ADJ}"] else [
        f"the child printed {printed}, want oom_score_adj {pw.OOM_SCORE_ADJ}"]
    notes += [] if w.exit == pw.Exit.DRAINED else [f"exit {w.exit}, want 0"]
    now = int(adj.read_text())
    notes += [] if now == mine else [f"the worker's own oom_score_adj {mine} -> {now}"]
    print(f"oom promote: a SIGKILL with the count {OOM_BEFORE} -> {ev.count}: attempts "
          f"{_tried(w)}, job 1 {w.jobs[1].state} in {w.jobs[1].cls}, the child at {printed}, "
          f"the worker's own {now}, exit {w.exit}" + _failing(notes), flush=True)
    bad += ["promote"] if notes else []
    # a SIGKILL with the count unchanged, and with no count to read: FAILED, a strike, and the
    # job the next lease's worker's (this one retires: the only job pending is one it attempted)
    for name, evidence in (("unchanged", Counted(count=OOM_BEFORE)),
                           ("unreadable", pw.Unreadable(why=NO_CGROUP))):
        w = _work(tmp / f"{name}.db", _seed(lad, [_m4(1, minutes=60.0, line=SIGKILLED)]),
                  worker=WORKER, run=RUN, commit=COMMIT,
                  minutes_left=120.0, oom=evidence, after=_nothing)
        notes = _state(w, 1, state=pq.JobState.PENDING, cls="m4", strikes=1)
        notes += [] if _tried(w) == [(1, "failed", -9)] else [f"attempts {_tried(w)}"]
        notes += [] if w.exit == pw.Exit.RETIRED else [f"exit {w.exit}, want 75"]
        why = [x for x in w.log if NO_CGROUP in x and x.startswith("job 1: failed")]
        if name == "unreadable" and len(why) != 1:
            notes.append(f"the log says why {len(why)} times, want once")
        print(f"oom {name}: a SIGKILL ({evidence.about}): attempts {_tried(w)}, job 1 "
              f"{w.jobs[1].state} with {w.jobs[1].strikes} strikes, exit {w.exit}"
              + _failing(notes), flush=True)
        bad += [name] if notes else []
    return bad


class Fixture(NamedTuple):
    """A /proc/self/cgroup and the files under the cgroup root (relative path -> text), with the
    evidence they should resolve to (None: Unreadable) and its count."""

    name: str
    proc: str
    files: dict[str, str]
    want: type[pw.CgroupV2] | type[pw.CgroupV1] | None
    at: str
    kills: int | None


V2_EVENTS = "low 0\nhigh 0\nmax 7\noom 3\noom_kill 2\noom_group_kill 0\n"
V1_CONTROL = "oom_kill_disable 0\nunder_oom 0\noom_kill 1\n"
SLURM_V2 = "system.slice/slurmstepd.scope/job_1/step_batch/user/task_0"
SLURM_V1 = "slurm/uid_1/job_1/step_batch"
FIXTURES = (
    Fixture(name="v2", proc=f"0::/{SLURM_V2}\n", files={f"{SLURM_V2}/memory.events": V2_EVENTS},
            want=pw.CgroupV2, at=SLURM_V2, kills=2),
    Fixture(name="v1", proc=f"12:memory:/{SLURM_V1}\n7:cpu,cpuacct:/{SLURM_V1}\n",
            files={f"memory/{SLURM_V1}/memory.oom_control": V1_CONTROL},
            want=pw.CgroupV1, at=f"memory/{SLURM_V1}", kills=1),
    Fixture(name="hybrid", proc=f"12:memory:/{SLURM_V1}\n0::/user.slice/session-1.scope\n",
            files={f"memory/{SLURM_V1}/memory.oom_control": V1_CONTROL,
                   "user.slice/session-1.scope/cgroup.procs": "1\n"},
            want=pw.CgroupV1, at=f"memory/{SLURM_V1}", kills=1),
    Fixture(name="v2-no-memory", proc=f"0::/{SLURM_V2}\n",
            files={f"{SLURM_V2}/cgroup.procs": "1\n"}, want=None, at="", kills=None),
    Fixture(name="v1-before-4.13", proc=f"12:memory:/{SLURM_V1}\n",
            files={f"memory/{SLURM_V1}/memory.oom_control": "oom_kill_disable 0\nunder_oom 0\n"},
            want=None, at="", kills=None),
    Fixture(name="none", proc="", files={}, want=None, at="", kills=None),
)


def _readers(tmp: Path) -> list[str]:
    """pool_worker.oom_evidence on each of FIXTURES, under a root of its own; and v2's count read
    afresh at each call (a kill counted after the probe is seen)."""
    notes: list[str] = []
    for f in FIXTURES:
        root = tmp / "cgroups" / f.name
        root.mkdir(parents=True)
        for rel, text in f.files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text(text)
        ev = pw.oom_evidence(f.proc, root)
        if f.want is None:
            ok = isinstance(ev, pw.Unreadable)
        else:
            ok = isinstance(ev, f.want) and ev.path == root / f.at and ev.kills() == f.kills
        if not ok:
            notes.append(f"{f.name}: {ev.about} ({type(ev).__name__}), want "
                         + ("Unreadable" if f.want is None else
                            f"{f.want.__name__} at {f.at}, count {f.kills}"))
        if f.name == "v2" and ok:
            (root / f.at / "memory.events").write_text(V2_EVENTS.replace("oom_kill 2",
                                                                         "oom_kill 3"))
            if ev.kills() != 3:
                notes.append(f"v2: the count after a kill {ev.kills()}, want 3 (read afresh)")
    here = pw.oom_evidence(Path("/proc/self/cgroup").read_text(), Path("/sys/fs/cgroup"))
    print(f"oom readers: {', '.join(f.name for f in FIXTURES)} (this machine: {here.about}, "
          f"count {here.kills()})" + _failing(notes), flush=True)
    return ["readers"] if notes else []


def oom(_args: list[str]) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="pool_check-") as tmp:
        return _oom_cases(Path(tmp)) + _readers(Path(tmp))


def identity(_args: list[str]) -> list[str]:
    lad = LADDERS["cpu-score"]
    bad: list[str] = []
    with tempfile.TemporaryDirectory(prefix="pool_check-") as tmp:
        w = _work(Path(tmp) / "pool.db", _seed(lad, [_m4(1, minutes=30.0, line="true")]),
                  worker=WORKER, run=RUN, commit=OTHER_COMMIT,
                  minutes_left=120.0, oom=Counted(count=0),
                  after=_nothing)
        notes = ([] if w.exit == pw.Exit.OWN_FAULT else [f"exit {w.exit}, want 1"]) + (
            [] if not w.attempts else [f"attempts {_tried(w)}, want none"]) + (
            [] if not w.workers else [f"registered {[x.id for x in w.workers]}, want none"]) + (
            [] if any(COMMIT in x and OTHER_COMMIT in x for x in w.log) else
            ["the log names not both commits"])
        print(f"identity work: a queue at {COMMIT[:8]}, the worker at {OTHER_COMMIT[:8]}: exit "
              f"{w.exit}, attempts {_tried(w)}" + _failing(notes), flush=True)
        bad += ["work"] if notes else []
        long = _line(1, minutes=30.0, line="true", peak_gb=100.0)
        path = Path(tmp) / "long.db"
        q: pq.Queue = pq.SqliteQueue.open(path, _seed(lad, [long]))
        log: list[str] = []
        ex = pw.work_one(queue=q, worker="long", restarts=0, run="pool-check-long-1", job=long.id,
                         commit=OTHER_COMMIT, ends_at=time.time() + 3600.0, now=time.time,
                         run_line=lambda c: pw.run_job(c.job.line, cwd=Path(tmp), env=os.environ,
                                                       log=log.append, wrote=lambda: None),
                         oom=Counted(count=0), log=log.append)
        tried = _attempts(path)
        notes = ([] if long.lane == pq.Lane.LONG else [f"the job's lane {long.lane}"]) + (
            [] if ex == pw.Exit.OWN_FAULT else [f"exit {int(ex)}, want 1"]) + (
            [] if not tried else [f"{len(tried)} attempts, want none"])
        print(f"identity work_one: the long lane's job, the worker at another commit: exit "
              f"{int(ex)}, {len(tried)} attempts" + _failing(notes), flush=True)
        bad += ["work_one"] if notes else []
    return bad


class Wave(NamedTuple):
    """A wave as `wave submit` ships it, before any process has opened its queue."""

    home: Path
    queue: Path
    inputs: Path
    checkout: Path


def _shipped(tmp: Path, seed: pq.Seed) -> Wave:
    """`seed` shipped to `tmp`/home/reblock-waves/<its wave>/inputs/jobs.json, and a checkout at
    COMMIT (cluster_submit's manifest)."""
    home = tmp / "home"
    wave = home / "reblock-waves" / seed.header.wave
    inputs = wave / "inputs"
    inputs.mkdir(parents=True)
    (inputs / "jobs.json").write_text(pq.seed_json(seed))
    checkout = tmp / "checkout"
    checkout.mkdir()
    (checkout / MANIFEST).write_text(json.dumps({"commit": COMMIT, "files": {}}))
    return Wave(home=home, queue=wave / "queue.db", inputs=inputs, checkout=checkout)


def _task(w: Wave, argv: list[str], *, task: int, run: str, on: str,
          restarts: int | None) -> tuple[int, list[str]]:
    """pool.py's `argv` as task `task` of run `run` on node `on`, requeued `restarts` times (None:
    never, Slurm's SLURM_RESTART_COUNT unset), its environment as cluster_submit and Slurm set it
    (two hours of walltime), in this process: its exit and its log."""
    environ = {**os.environ, "HOME": str(w.home), "SLURMD_NODENAME": on,
               "SLURM_ARRAY_JOB_ID": TASK_JOB,
               "SLURM_ARRAY_TASK_ID": str(task), "CLUSTER_SUBMIT_RUN": run,
               "SLURM_JOB_END_TIME": str(int(time.time() + 7200)),
               "CLUSTER_SUBMIT_CHECKOUT": str(w.checkout)}
    environ.pop("SLURM_RESTART_COUNT", None)
    if restarts is not None:
        environ["SLURM_RESTART_COUNT"] = str(restarts)
    out = io.StringIO()
    with redirect_stdout(out):
        ex = pl.main(argv, environ)
    return int(ex), out.getvalue().splitlines()


def _absent_queue(tmp: Path) -> list[str]:
    lad = LADDERS["cpu-score"]
    home = tmp / "home"
    inputs = home / "reblock-waves" / "pool-check" / "inputs"
    checks_env = (f'test "$CLUSTER_SUBMIT_INPUTS" = {shlex.quote(str(inputs))} && '
                  f'test "${common.BANK_ENV}" = {shlex.quote(str(inputs / "bank.pkl"))} && '
                  f'test "$(pwd -P)" = {shlex.quote(str((tmp / "checkout").resolve()))}')
    w = _shipped(tmp, _seed(lad, [_m4(1, minutes=60.0, line="exit 3"),
                                  _m4(2, minutes=30.0, line=checks_env)]))
    there_before = w.queue.exists()
    run = f"pool-check-{NODE}-m4-s1"
    ex, log = _task(w, ["work", "--wave", "pool-check", "--node", NODE, "--class", "m4"],
                    task=0, run=run, on=NODE, restarts=None)
    q = pq.SqliteQueue.open(w.queue, _seed(lad, []))
    jobs = {j.id: j for j in q.jobs()}
    tried = [(a.job, a.outcome, a.rc) for a in _attempts(w.queue)]
    workers = [(x.id, x.run, x.node, x.cls, x.retired) for x in q.workers()]
    notes = (["the queue was there before the worker"] if there_before else []) + (
        [] if ex == pw.Exit.RETIRED else [f"exit {ex}, want 75"]) + (
        [] if tried == [(1, "failed", 3), (2, "done", 0)] else
        [f"attempts {tried}, want job 1 failed, then job 2 done (its environment and directory "
         "the wave's)"]) + (
        [] if (jobs[1].state, jobs[1].strikes, jobs[2].state) == (
            pq.JobState.PENDING, 1, pq.JobState.DONE) else
        [f"jobs {jobs[1].state} ({jobs[1].strikes} strikes), {jobs[2].state}"]) + (
        [] if workers == [(f"{TASK_JOB}_0", run, NODE, "m4", True)] else [f"workers {workers}"])
    print(f"worker absent-queue: pool.py work on a wave whose queue no process had made: exit "
          f"{ex}, attempts {tried}, workers {workers}" + _failing(notes), flush=True)
    return ["absent-queue"] if notes else []


def _bad_node(tmp: Path) -> list[str]:
    """Jobs 1-6 failing (a bad node), job 7 not, three workers of one lease in turn."""
    lad = LADDERS["cpu-score"]
    bad: list[str] = []
    seed = _seed(lad, [_m4(i, minutes=70.0 - 5 * i, line="exit 1") for i in range(1, 7)]
                 + [_m4(7, minutes=10.0, line="true")])
    runs = [_work(tmp / "bad-node.db", seed, worker=f"w{k}", run=RUN, commit=COMMIT,
                  minutes_left=120.0, oom=Counted(count=0), after=_nothing) for k in (1, 2, 3)]
    w = runs[-1]
    by = {k: [(a.job, a.outcome) for a in w.attempts if a.worker == f"w{k}"] for k in (1, 2, 3)}
    want = {1: [(1, "failed"), (2, "failed"), (3, "failed")],
            2: [(4, "failed"), (5, "failed"), (6, "failed")], 3: [(7, "done")]}
    exits = [x.exit for x in runs]
    notes = ([] if exits == [1, 1, 75] else [f"exits {exits}, want 1, 1, 75"]) + (
        [] if by == want else [f"attempts by worker {by}, want {want}"]) + [
        n for i in range(1, 7) for n in _state(w, i, state=pq.JobState.PENDING, cls="m4",
                                                strikes=1)] + _state(
        w, 7, state=pq.JobState.DONE, cls="m4", strikes=0)
    print(f"worker bad-node: jobs 1-6 failing, job 7 not, three workers of one lease: attempts "
          f"{by}, strikes {[w.jobs[i].strikes for i in range(1, 8)]}, exits {exits}"
          + _failing(notes), flush=True)
    bad += ["bad-node"] if notes else []
    # one job, failing: one strike, never claimed again by this worker, which retires
    w = _work(tmp / "one-job.db", _seed(lad, [_m4(1, minutes=60.0, line="exit 1")]),
              worker=WORKER, run=RUN, commit=COMMIT, minutes_left=120.0, oom=Counted(count=0),
              after=_nothing)
    notes = ([] if w.exit == pw.Exit.RETIRED else [f"exit {w.exit}, want 75"]) + (
        [] if _tried(w) == [(1, "failed", 1)] else [f"attempts {_tried(w)}, want one"]) + _state(
        w, 1, state=pq.JobState.PENDING, cls="m4", strikes=1)
    print(f"worker one-job: its only job failing: attempts {_tried(w)}, job 1 {w.jobs[1].state} "
          f"with {w.jobs[1].strikes} strikes, exit {w.exit}" + _failing(notes), flush=True)
    return bad + (["one-job"] if notes else [])


def _progress(tmp: Path) -> list[str]:
    lad = LADDERS["cpu-score"]
    rows = [str(tmp / "rows" / f"r{k}.parquet") for k in (1, 2)]
    code = (f"import sys; sys.path.insert(0, {str(HERE)!r}); import clearings, pandas as pd; "
            "from pathlib import Path; "
            f"[clearings.Refuse().write(pd.DataFrame({{'a': [1]}}), Path(p)) "
            f"for p in {rows!r}]; sys.exit(1)")
    line = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    seed = _seed(lad, [_m4(1, minutes=30.0, line=line)])
    first = _work(tmp / "progress.db", seed, worker=WORKER, run=RUN, commit=COMMIT,
                  minutes_left=120.0,
                  oom=Counted(count=0), after=_nothing)
    w = _work(tmp / "progress.db", seed, worker="w2", run=NEXT_RUN, commit=COMMIT,
              minutes_left=120.0,
              oom=Counted(count=0), after=_nothing)
    counted = [a.rows for a in w.attempts]
    printed = [x for x in first.log if x.startswith(row_marker.ROW_WRITTEN)]
    notes = ([] if counted == [2, 0] else [f"rows by attempt {counted}, want [2, 0] (the first "
                                           "written, no strike; the next refused, a strike)"]) + (
        [] if printed == [f"{row_marker.ROW_WRITTEN}{p}" for p in rows] else
        [f"ROW_WRITTEN lines {printed}"]) + _state(
        w, 1, state=pq.JobState.PENDING, cls="m4", strikes=1) + (
        [] if (first.exit, w.exit) == (pw.Exit.RETIRED, pw.Exit.RETIRED) else
        [f"exits {first.exit}, {w.exit}, want 75, 75"])
    print(f"worker progress: a job writing two rows (clearings.write_once) and exiting 1, run by "
          f"two workers: rows by attempt {counted}, job 1 {w.jobs[1].state} with "
          f"{w.jobs[1].strikes} strikes, exits {first.exit}, {w.exit}" + _failing(notes),
          flush=True)
    return ["progress"] if notes else []


def _idle(tmp: Path) -> list[str]:
    """A worker on an open queue with nothing pending: it waits, and retires after the ladder's
    idle minutes. Only an open queue idles a worker (a closed one with nothing pending drains it
    at once, exit 0), and only a Screened feed (decision 7, not built) leaves a queue open: its
    open seed stands in for Screened's, as `_end_open`'s does."""
    lad = LADDERS["cpu-score"]
    polls = math.ceil(60 * lad.idle_minutes / lad.poll_seconds)
    bad: list[str] = []
    for name, left, want in (("idle", 120.0, polls), ("idle-at-end", lad.margin_minutes / 2, 0)):
        seed = pq.Seed(header=replace(_header(lad), feed=OPEN_FEED), jobs=(), closed=False)
        w = _work(tmp / f"{name}.db", seed, worker=WORKER, run=RUN, commit=COMMIT,
                  minutes_left=left,
                  oom=Counted(count=0), after=_nothing)
        notes = ([] if w.exit == pw.Exit.RETIRED else [f"exit {w.exit}, want 75"]) + (
            [] if w.waits == want else [f"{w.waits} polls, want {want}"])
        print(f"worker {name}: an open queue with nothing pending, {left:g} minutes left: "
              f"{w.waits} polls of {lad.poll_seconds:g} s, exit {w.exit}" + _failing(notes),
              flush=True)
        bad += [name] if notes else []
    return bad


def _stepped_aside(tmp: Path) -> list[str]:
    lad = LADDERS["cpu-score"]
    asked: list[list[str]] = []

    def ask(q: pq.Queue, c: pq.Claim, clock: Clock) -> None:
        if c.job.id == 1:
            asked.append(q.ask_aside(node=NODE, cls="m4", units=1, since=time.time() - 1.0))

    w = _work(tmp / "aside.db", _seed(lad, [_m4(1, minutes=60.0, line="true"),
                                            _m4(2, minutes=30.0, line="true")]),
              worker=WORKER, run=RUN, commit=COMMIT,
              minutes_left=120.0, oom=Counted(count=0), after=ask)
    notes = ([] if asked == [[WORKER]] else [f"asked {asked}"]) + (
        [] if w.exit == pw.Exit.RETIRED else [f"exit {w.exit}, want 75"]) + (
        [] if _tried(w) == [(1, "done", 0)] else [f"attempts {_tried(w)}, want job 1's"]) + (
        _state(w, 2, state=pq.JobState.PENDING, cls="m4", strikes=0))
    print(f"worker aside: asked aside during job 1: attempts {_tried(w)}, job 2 "
          f"{w.jobs[2].state}, exit {w.exit}" + _failing(notes), flush=True)
    return ["aside"] if notes else []


def _long_lane(tmp: Path) -> list[str]:
    lad = LADDERS["cpu-score"]
    long = _line(1, minutes=30.0, line="true", peak_gb=100.0)
    w = _shipped(tmp, _seed(lad, [long]))
    run = "pool-check-long-1"
    first, _ = _task(w, ["work", "--wave", "pool-check", "--job", "1"], task=0, run=run, on=NODE,
                     restarts=None)
    again, _ = _task(w, ["work", "--wave", "pool-check", "--job", "1"], task=1, run=run, on=NODE,
                     restarts=None)
    q = pq.SqliteQueue.open(w.queue, _seed(lad, []))
    job = q.jobs()[0]
    workers = [(x.id, x.node, x.cls, x.retired) for x in q.workers()]
    want = [(f"{TASK_JOB}_{k}", NODE, pq.Lane.LONG.value, True) for k in (0, 1)]
    notes = ([] if long.lane == pq.Lane.LONG else [f"the job's lane {long.lane}"]) + (
        [] if (first, again) == (0, 1) else [f"exits {first}, {again}, want 0, 1"]) + (
        [] if job.state == pq.JobState.DONE else [f"job 1 {job.state}"]) + (
        [] if len(_attempts(w.queue)) == 1 else [f"{len(_attempts(w.queue))} attempts"]) + (
        [] if workers == want else [f"workers {workers}, want {want}"])
    print(f"worker long-lane: pool.py work --job 1, twice: exits {first}, {again}, job 1 "
          f"{job.state}, workers {workers}" + _failing(notes), flush=True)
    return ["long-lane"] if notes else []


class _Interrupted(RuntimeError):
    """The error a case's log raises mid-job."""


class Interrupted(NamedTuple):
    """What a worker did whose log raised mid-job (`_interrupt`)."""

    raised: str | None      # None: the log's own error, as wanted; else what happened instead
    took: float             # seconds, the worker's start to its error
    printed: list[str]      # the job's lines its log was given
    notes: list[str]        # the job, its attempt and the worker against the case's want


def _interrupt(tmp: Path, line: str) -> Interrupted:
    """pool_worker.work running one job, `line`, whose log raises at the job's `started` line:
    the error the log's own, the job back to PENDING as if never claimed (its attempt RETIRED,
    no strike), the worker retired."""
    lad = LADDERS["cpu-score"]
    path = tmp / "interrupted.db"
    q: pq.Queue = pq.SqliteQueue.open(path, _seed(lad, [_m4(1, minutes=30.0, line=line)]))
    printed: list[str] = []

    def interrupting(text: str) -> None:
        printed.append(text)
        if text == "started":
            raise _Interrupted("the worker's log failed mid-job")

    def run_line(c: pq.Claim) -> pw.Ran:
        return pw.run_job(c.job.line, cwd=tmp, env=os.environ, log=interrupting,
                          wrote=lambda: q.wrote(c.attempt))

    t0 = time.monotonic()
    try:
        ex = pw.work(queue=q, worker=WORKER, restarts=0, run=RUN, node=NODE, cls="m4",
                     commit=COMMIT, ends_at=time.time() + 7200.0, now=time.time,
                     wait=lambda: None, run_line=run_line, oom=Counted(count=0),
                     log=lambda text: None)
        raised: str | None = f"nothing, exit {int(ex)}"
    except _Interrupted:
        raised = None
    except Exception as e:      # another error than the log's: what the guard reports
        raised = f"{type(e).__name__}: {e}"
    took = time.monotonic() - t0
    job = q.jobs()[0]
    tried = [(a.job, a.outcome) for a in _attempts(path)]
    retired = [x.retired for x in q.workers()]
    notes = ([] if (job.state, job.strikes) == (pq.JobState.PENDING, 0) else
             [f"job 1 {job.state} with {job.strikes} strikes, want pending with none"]) + (
        [] if tried == [(1, pq.Outcome.RETIRED.value)] else [f"attempts {tried}"]) + (
        [] if retired == [True] else [f"workers retired {retired}"])
    return Interrupted(raised=raised, took=took, printed=printed, notes=notes)


def _raised(tmp: Path) -> list[str]:
    """run_line raising mid-job (the worker's log, at the child's first line): the child killed
    within RAISE_LIMIT (it sleeps RAISE_SLEEP), and `_interrupt`'s wants."""
    got = _interrupt(tmp, f"echo started; exec sleep {RAISE_SLEEP:g}")
    notes = ([] if got.raised is None else [f"raised {got.raised}, want the log's own error"]) + (
        [] if got.took < RAISE_LIMIT else [f"took {got.took:.1f} s (the child sleeps "
                                           f"{RAISE_SLEEP:g}): not killed"]) + got.notes
    print(f"worker raises: the log raising at the child's first line: "
          f"{got.raised or 'its error'}, in {got.took:.2f} s" + _failing(notes), flush=True)
    return ["raises"] if notes else []


def _running(pid: int) -> bool:
    """Whether process `pid` runs: neither gone nor a zombie (dead, its parent yet to reap it)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _group(tmp: Path) -> list[str]:
    """A compound line (GRANDCHILD: its shell forks a grandchild, which prints its pid) whose
    worker's log raises mid-job: the job's whole group killed, the grandchild gone within
    GONE_WAIT; and `_interrupt`'s wants. A grandchild left running is killed here after."""
    got = _interrupt(tmp, GRANDCHILD)
    pids = [int(x.split()[1]) for x in got.printed if x.startswith("grandchild ")]
    deadline = time.monotonic() + GONE_WAIT
    while pids and _running(pids[0]) and time.monotonic() < deadline:
        time.sleep(0.05)
    left = [p for p in pids if _running(p)]
    for p in left:              # the guard's own: no process outlives it
        os.kill(p, signal.SIGKILL)
    notes = ([] if got.raised is None else [f"raised {got.raised}, want the log's own error"]) + (
        [] if got.took < RAISE_LIMIT else [f"took {got.took:.1f} s: not killed"]) + (
        [] if len(pids) == 1 else [f"the grandchild printed {len(pids)} pids, want one"]) + (
        [] if not left else [f"the grandchild {left} still ran {GONE_WAIT:g} s after the kill"]
    ) + got.notes
    print(f"worker group: a compound line's grandchild ({pids}), the log raising mid-job: "
          f"{got.raised or 'its error'} in {got.took:.2f} s, the grandchild "
          + ("still running" if left else "gone") + _failing(notes), flush=True)
    return ["group"] if notes else []


def _requeued(tmp: Path) -> list[str]:
    """A task requeued by Slurm (its array id kept) while holding job 1, its earlier self never
    retired, run by pool.py at SLURM_RESTART_COUNT 1: registered again (no refusal), its earlier
    self reaped (job 1 PENDING with a strike: it wrote no row; the reap recorded), job 2 run, job 1
    not (its lease attempted it), retired (75); registering again at the same count is refused."""
    lad = LADDERS["cpu-score"]
    seed = _seed(lad, [_m4(1, minutes=60.0, line="true"), _m4(2, minutes=30.0, line="true")])
    w = _shipped(tmp, seed)
    task = f"{TASK_JOB}_0"
    q = pq.SqliteQueue.open(w.queue, seed)
    q.register(worker=task, run=RUN, node=NODE, cls="m4", ends_at=time.time() + 7200.0,
               restarts=0)
    held = q.claim(worker=task, minutes_left=pq.fresh_minutes(lad))
    argv = ["work", "--wave", "pool-check", "--node", NODE, "--class", "m4"]
    try:
        ex: int | str = _task(w, argv, task=0, run=RUN, on=NODE, restarts=1)[0]
    except ValueError as e:
        ex = f"refused: {e}"
    try:
        again: int | str = _task(w, argv, task=0, run=RUN, on=NODE, restarts=1)[0]
    except ValueError:
        again = "refused"
    jobs = {j.id: j for j in q.jobs()}
    tried = [(a.job, a.outcome) for a in _attempts(w.queue)]
    want_reaped = [pq.Reaped(worker=task, run=RUN, job=1, outcome=pq.Outcome.REAPED,
                             walltime=False)]
    workers = [(x.id, x.retired) for x in q.workers()]
    notes = ([] if held is not None and held.job.id == 1 else [f"the first claim {held}"]) + (
        [] if ex == pw.Exit.RETIRED else [f"restart 1: {ex}, want exit 75"]) + (
        [] if again == "refused" else [f"restart 1 again: exit {again}, want refused"]) + (
        [] if tried == [(1, "reaped"), (2, "done")] else [f"attempts {tried}"]) + (
        [] if (jobs[1].state, jobs[1].strikes, jobs[2].state) == (
            pq.JobState.PENDING, 1, pq.JobState.DONE) else
        [f"jobs {jobs[1].state} ({jobs[1].strikes} strikes), {jobs[2].state}"]) + (
        [] if q.reaped() == want_reaped else [f"reaped {q.reaped()}, want {want_reaped}"]) + (
        [] if workers == [(task, True)] else [f"workers {workers}"])
    print(f"worker requeued: task {task} holding job 1, requeued (SLURM_RESTART_COUNT 1): exit "
          f"{ex}, attempts {tried}, reaped {[(r.worker, r.job) for r in q.reaped()]}; the same "
          f"restart again: {again}" + _failing(notes), flush=True)
    return ["requeued"] if notes else []


def _refusals(tmp: Path) -> list[str]:
    """pool.py refusing a --class its wave's ladder lacks, a --node it does not run on, and a
    --job of another node's."""
    lad = LADDERS["cpu-score"]
    w = _shipped(tmp, _seed(lad, [_m4(1, minutes=30.0, line="true")]))
    pooled = ["work", "--wave", "pool-check", "--node", NODE, "--class"]
    notes: list[str] = []
    for k, (name, argv, on, names) in enumerate((
            ("class", [*pooled, "m5"], NODE, ["m5", *(class_name(0, r) for r in lad.rungs)]),
            ("node", [*pooled, "m4"], QUADRO, [NODE, QUADRO]),
            ("job", ["work", "--wave", "pool-check", "--job", "1"], QUADRO, ["1", NODE, QUADRO]))):
        try:
            ex, _ = _task(w, argv, task=k, run=RUN, on=on, restarts=None)
            notes.append(f"--{name}: exit {ex}, want refused")
        except SystemExit as e:
            missing = [n for n in names if n not in str(e.code)]
            if missing:
                notes.append(f"--{name}: {e.code!r} names not {missing}")
    if w.queue.exists():
        q = pq.SqliteQueue.open(w.queue, _seed(lad, []))
        notes += [f"registered {[x.id for x in q.workers()]}"] if q.workers() else []
        notes += [f"attempts {_attempts(w.queue)}"] if _attempts(w.queue) else []
    print(f"worker refusals: pool.py work with --class m5 (cpu-score's: "
          f"{', '.join(class_name(0, r) for r in lad.rungs)}), --node {NODE} on {QUADRO}, and "
          f"--job 1 ({NODE}'s) on {QUADRO}" + _failing(notes), flush=True)
    return ["refusals"] if notes else []


def worker(_args: list[str]) -> list[str]:
    bad: list[str] = []
    for case in (_absent_queue, _bad_node, _progress, _idle, _stepped_aside, _raised, _group,
                 _requeued, _refusals, _long_lane):
        with tempfile.TemporaryDirectory(prefix="pool_check-") as tmp:
            bad += case(Path(tmp))
    return bad


# ---- the steward: dead-claim, aside, end ------------------------------------------------------

STEWARD_TASK = "4343_0"           # the steward's own Slurm task, which a believed squeue lists
STALE_AFTER = timedelta(minutes=30)          # the market's stale_after, as the guards pass it
HOLD_SLEEP = 600.0                # dead-claim: the SIGKILLed worker's wait for its kill (s)
HOLD_WAIT = 60.0                  # ... the most the guard waits for its claim (s)
WALLTIME_RAN = 50.0               # ... the minutes the walltime-killed worker ran its job
AGE_LIMIT = 3600.0                # by-age: a claim older than this reaped (s)
ASIDE_WORKERS = (("wave-a", 4), ("wave-b", 2))   # aside: the waves and their live workers
ASIDE_YIELDS = 3                  # ... the campaign's yields on (QUADRO, m4)
GHOST_WORKERS = 6                 # ... the workers a dead steward's summary last counted there
END_POLL = 0.2                    # end: the steward's poll (s); its lease three polls long

Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def _squeue(*tasks: str) -> Runner:
    """A squeue answering `tasks`, a line each."""
    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), 0, "".join(f"{x}\n" for x in tasks), "")
    return run


def _squeue_fails(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """A squeue that cannot reach its controller."""
    return subprocess.CompletedProcess(list(argv), 1, "",
                                       "squeue: error: Unable to contact slurm controller\n")


def _squeue_hangs(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """A squeue that outlives its timeout."""
    raise subprocess.TimeoutExpired(list(argv), ps.SQUEUE_TIMEOUT)


def _liveness(run: Runner) -> ps.Squeue:
    return ps.Squeue(run=run, user="pool-check", own=STEWARD_TASK)


def _wave_seed(name: str, jobs: Sequence[pq.JobSeed]) -> pq.Seed:
    """A fixed wave `name`'s seed under cpu-score."""
    return pq.Seed(header=replace(_header(LADDERS["cpu-score"]), wave=name), jobs=tuple(jobs),
                   closed=True)


def _round(q: pq.Queue, wave_dir: Path, waves: Path, *, liveness: ps.Liveness,
           allocation: Allocation | None, now: datetime) -> psum.Summary:
    """One steward round of the wave in `wave_dir` at `now`, `allocation` on the board."""
    return ps.round_(queue=q, wave_dir=wave_dir, waves=waves, liveness=liveness,
                     feed=ps.Fixed(), allocation=lambda: allocation, stale_after=STALE_AFTER,
                     now=lambda: now, log=lambda line: None)


def _holder(path: Path, seed: pq.Seed, worker: str, run: str, ends_at: float,
            out: MpQueue[int]) -> None:
    """A worker (a child process) that registers, claims, says which job, and waits to be
    SIGKILLed holding its claim."""
    q = pq.SqliteQueue.open(path, seed)
    q.register(worker=worker, run=run, node=NODE, cls="m4", ends_at=ends_at, restarts=0)
    c = q.claim(worker=worker, minutes_left=pq.fresh_minutes(seed.header.ladder))
    out.put(0 if c is None else c.job.id)
    time.sleep(HOLD_SLEEP)


class Workers(NamedTuple):
    """dead-claim's workers (Slurm tasks): SIGKILLed holding job 1, live holding job 2 (the
    longest), past its walltime holding job 3."""
    killed: str
    live: str
    ended: str


def _three(waves: Path, *, real: bool) -> tuple[Path, pq.SqliteQueue, Workers, float, list[str]]:
    """dead-claim's wave: jobs 1 (30 min), 2 (100) and 3 (20) of m4, each held by one of three
    workers of three leases registered at t0: `live` (job 2, its walltime 3 h off), `killed` (job
    1, the same; a real child SIGKILLed holding it when `real`, else registered here and left),
    `ended` (job 3, its walltime WALLTIME_RAN minutes off). The wave's directory and queue, the
    workers, t0, and what went wrong."""
    lad = LADDERS["cpu-score"]
    seed = _wave_seed("pool-check", [_small(lad, 1, minutes=30.0), _small(lad, 2, minutes=100.0),
                                     _small(lad, 3, minutes=20.0)])
    d = waves / "pool-check"
    q = pq.SqliteQueue.open(d / "queue.db", seed)
    ws = Workers(killed=f"{TASK_JOB}_0", live=f"{TASK_JOB}_1", ended=f"{TASK_JOB}_2")
    t0 = time.time()
    notes: list[str] = []

    def lease(w: str) -> str:
        return f"pool-check-{NODE}-m4-{w}"

    q.register(worker=ws.live, run=lease(ws.live), node=NODE, cls="m4", ends_at=t0 + 3 * 3600,
               restarts=0)
    c = q.claim(worker=ws.live, minutes_left=pq.fresh_minutes(lad))
    notes += [] if c is not None and c.job.id == 2 else [f"live's claim {c}, want job 2"]
    if real:
        ctx = mp.get_context("fork")
        out: MpQueue[int] = ctx.Queue()
        p = ctx.Process(target=_holder, args=(d / "queue.db", seed, ws.killed, lease(ws.killed),
                                               t0 + 3 * 3600, out))
        p.start()
        try:
            held: int | None = out.get(timeout=HOLD_WAIT)
        except queue.Empty:
            held = None
        p.kill()                # SIGKILL: no `finally`, no retire
        p.join()
        notes += [] if p.exitcode == -signal.SIGKILL else [f"the holder exited {p.exitcode}"]
    else:
        q.register(worker=ws.killed, run=lease(ws.killed), node=NODE, cls="m4",
                   ends_at=t0 + 3 * 3600, restarts=0)
        k = q.claim(worker=ws.killed, minutes_left=pq.fresh_minutes(lad))
        held = None if k is None else k.job.id
    notes += [] if held == 1 else [f"the killed worker held {held}, want job 1"]
    q.register(worker=ws.ended, run=lease(ws.ended), node=NODE, cls="m4",
               ends_at=t0 + 60 * WALLTIME_RAN, restarts=0)
    e = q.claim(worker=ws.ended, minutes_left=WALLTIME_RAN - lad.margin_minutes)
    notes += [] if e is not None and e.job.id == 3 else [f"ended's claim {e}, want job 3"]
    return d, q, ws, t0, notes


def _dead_round(tmp: Path) -> list[str]:
    """The round after a worker's SIGKILL (the spec's guard 3, the plan's Review Focus 4)."""
    waves = tmp / "waves"
    d, q, ws, t0, notes = _three(waves, real=True)
    now = datetime.fromtimestamp(t0 + 2 * 3600, UTC)
    s = _round(q, d, waves, liveness=_liveness(_squeue(STEWARD_TASK, ws.live)), allocation=None,
               now=now)
    jobs = {j.id: j for j in q.jobs()}

    def lease(w: str) -> str:
        return f"pool-check-{NODE}-m4-{w}"

    want = (psum.ReapedWorker(task=ws.killed, run=lease(ws.killed), held_job=True,
                              walltime=False),
            psum.ReapedWorker(task=ws.ended, run=lease(ws.ended), held_job=True, walltime=True))
    live = (psum.RegisteredWorker(task=ws.live, run=lease(ws.live)),)
    on_disk = psum.read_summary((d / ps.SUMMARY).read_text(), "summary.json")
    ledger = (d / ps.LEDGER).read_text().splitlines()
    got = {i: (j.state, j.strikes, j.worker) for i, j in jobs.items()}
    want_jobs = {1: (pq.JobState.PENDING, 1, None), 2: (pq.JobState.CLAIMED, 0, ws.live),
                 3: (pq.JobState.PENDING, 0, None)}
    notes += ([] if got == want_jobs else [f"jobs {got}, want {want_jobs}"]) + (
        [] if math.isclose(jobs[3].minutes, WALLTIME_RAN, abs_tol=0.1) else
        [f"job 3's minutes {jobs[3].minutes:.2f}, want {WALLTIME_RAN:g} (what it ran)"]) + (
        [] if s.reaped == want else [f"reaped {s.reaped}, want {want}"]) + (
        [] if s.registered == live else [f"registered {s.registered}, want {live}"]) + (
        [] if on_disk == s else ["summary.json reads back otherwise"]) + (
        [] if len(ledger) == 1 else [f"{len(ledger)} ledger lines, want 1"])
    print(f"dead-claim round: a worker SIGKILLed holding job 1, one live holding job 2, one past "
          f"its walltime holding job 3; squeue lists the live one and the steward's own: jobs "
          f"{({i: (x[0].value, *x[1:]) for i, x in got.items()})}, job 3's minutes "
          f"{jobs[3].minutes:.1f}, reaped "
          f"{[(r.task, r.held_job, r.walltime) for r in s.reaped]}" + _failing(notes),
          flush=True)
    return ["round"] if notes else []


def _abstains(tmp: Path) -> list[str]:
    """A squeue that fails, one that outlives its timeout, and one answering without the
    steward's own task: nobody reaped."""
    bad: list[str] = []
    for name, run in (("fails", _squeue_fails), ("hangs", _squeue_hangs),
                      ("without-own", None)):
        waves = tmp / name / "waves"
        d, q, ws, t0, notes = _three(waves, real=False)
        answer = _squeue(ws.live) if run is None else run
        s = _round(q, d, waves, liveness=_liveness(answer), allocation=None,
                   now=datetime.fromtimestamp(t0 + 2 * 3600, UTC))
        held = {j.id: j.worker for j in q.jobs()}
        want = {1: ws.killed, 2: ws.live, 3: ws.ended}
        notes += ([] if held == want else [f"held {held}, want {want}"]) + (
            [] if s.reaped == () else [f"reaped {s.reaped}"])
        print(f"dead-claim {name}: nobody reaped" + _failing(notes), flush=True)
        bad += [name] if notes else []
    return bad


def dead_claim(_args: list[str]) -> list[str]:
    bad: list[str] = []
    for case in (_dead_round, _abstains):
        with tempfile.TemporaryDirectory(prefix="pool_check-") as tmp:
            bad += case(Path(tmp))
    return bad


def _allocation(*, at: datetime, mode: str, yields: int) -> Allocation:
    """An allocation of `mode` at `at` in which the campaign holds 6 units of cpu-score's m4 on
    QUADRO and yields `yields` of them, made as the keeper would (read back from its JSON)."""
    m4 = LADDERS["cpu-score"].bundle(4)
    none = Bundle(cards=0, cpus=0, mem_gb=0)
    stamp = at.strftime(STAMP)
    tier = FittedTier(coverage=0.9, lease_hours=2.0, hours=0.0, reserve=none, covered=1.0,
                      started=1.0)
    node = NodeShare(state="mixed", tier="active", lease_hours=2.0,
                     total=Bundle(cards=0, cpus=64, mem_gb=256), others=none, outside=none,
                     reserve=none, capacity=Bundle(cards=0, cpus=64, mem_gb=248),
                     owed=m4.times(yields), owed_to=("another-user's-job",), why="the guard's",
                     campaigns={ps.CAMPAIGN: (Place(bundle=m4, demand=6, urgency=None,
                                                    allocated=6, held=6, yields=yields),)})
    a = Allocation(at=stamp, mode=mode, pool=Bundle(cards=0, cpus=64, mem_gb=256),
                   fit=Fit(at=stamp, window_days=14, jobs=0, quiet=tier, active=tier),
                   campaigns={ps.CAMPAIGN: Standing(weight=1.0, share=1.0, spend=0.0,
                                                    demand_at=stamp, stale=False, runs=())},
                   nodes={QUADRO: node}, outside=(), refused={})
    return read_allocation(allocation_json(a, indent=1), "the guard's allocation")


class Third(NamedTuple):
    """A third wave under `waves` whose directory holds only a summary.json (`text`): a dead
    steward's, or one of another format; and the key of the aside's ledger record (`stale` or
    `refused`) that should name it."""
    wave: str
    text: str
    left_out: str


def _ghost(at: datetime) -> Third:
    """A dead steward's summary written at `at`: wave-ghost CLOSED with GHOST_WORKERS live workers
    on (QUADRO, m4) as it last counted them."""
    s = psum.Summary(wave="wave-ghost", commit=COMMIT, ladder="cpu-score",
                     at=at.strftime(STAMP), round=7,
                     state=psum.WaveState.CLOSED, metric=None,
                     lines=(psum.ClassLine(node=QUADRO, cls="m4", cards=0, cpus=2, mem_gb=4,
                                           pending=GHOST_WORKERS, claimed=GHOST_WORKERS,
                                           minutes=30.0, long=0, done=0, failed=0,
                                           workers=GHOST_WORKERS),),
                     reaped=(), registered=(), long=())
    return Third(wave=s.wave, text=psum.summary_json(s), left_out="stale")


def _aside_case(tmp: Path, *, name: str, t0: datetime, mode: str, age: timedelta,
                third: Third | None, want: tuple[int, ...]) -> list[str]:
    """ASIDE_WORKERS' waves sharing (QUADRO, m4) under one `waves`, their workers live, and
    `third` beside them: each steward's first round with no allocation, at `t0`, then two rounds
    each under an allocation of `mode` `age` old yielding ASIDE_YIELDS there; the workers each wave
    has asked after each round against `want` (the first rounds', then the second's), and the
    third named in the first wave's ledger, at its first round under the allocation, as left out
    (`third.left_out`)."""
    waves = tmp / "waves"
    lad = LADDERS["cpu-score"]
    made: dict[str, tuple[Path, pq.SqliteQueue, list[str]]] = {}
    for w, n in ASIDE_WORKERS:
        jobs = [replace(_small(lad, i, minutes=30.0), node=QUADRO) for i in range(1, 2 * n + 1)]
        q = pq.SqliteQueue.open(waves / w / "queue.db", _wave_seed(w, jobs))
        ids = [f"{w}-{k}" for k in range(n)]
        for x in ids:
            q.register(worker=x, run=f"{w}-{QUADRO}-m4-s1", node=QUADRO, cls="m4",
                       ends_at=t0.timestamp() + 7200, restarts=0)
        made[w] = (waves / w, q, ids)
    if third is not None:
        (waves / third.wave).mkdir(parents=True)
        (waves / third.wave / ps.SUMMARY).write_text(third.text)
    everyone = _liveness(_squeue(STEWARD_TASK, *(x for _, _, ids in made.values() for x in ids)))
    alloc = _allocation(at=t0 - age, mode=mode, yields=ASIDE_YIELDS)
    got: list[int] = []
    notes: list[str] = []
    try:
        for d, q, _ in made.values():
            _round(q, d, waves, liveness=everyone, allocation=None, now=t0)
        for _ in (1, 2):
            for d, q, ids in made.values():
                _round(q, d, waves, liveness=everyone, allocation=alloc, now=t0)
                got.append(sum(1 for x in ids if q.asked_aside(x)))
    except ValueError as e:
        notes.append(f"a round raised {e}")
    notes += [] if tuple(got) == want else [f"asked {got}, want {list(want)}"]
    if third is not None and not notes:
        first = waves / ASIDE_WORKERS[0][0] / ps.LEDGER
        record = json.loads(first.read_text().splitlines()[1])["aside"]
        named = record[third.left_out] if isinstance(record, dict) else None
        path = str(waves / third.wave / ps.SUMMARY)
        notes += [] if isinstance(named, dict) and path in named else [
            f"the ledger's aside {record}, want {path} among its {third.left_out}"]
    print(f"aside {name}: waves {dict(ASIDE_WORKERS)} on {QUADRO} m4"
          + ("" if third is None else f" and {third.wave}'s summary ({third.left_out})")
          + f", an allocation ({mode}, {age} old) yielding {ASIDE_YIELDS}: asked after each "
          f"round {got}" + _failing(notes), flush=True)
    return [name] if notes else []


def aside(_args: list[str]) -> list[str]:
    bad: list[str] = []
    t0 = datetime.fromtimestamp(math.floor(time.time()), UTC)
    fresh, stale = timedelta(0), STALE_AFTER + timedelta(minutes=1)
    odd = Third(wave="wave-odd", text=json.dumps({"format": psum.SUMMARY_FORMAT + 1}),
                left_out="refused")
    for name, mode, age, third, want in (
            ("active", "active", fresh, None, (2, 1, 2, 1)),
            ("stale", "active", stale, None, (0, 0, 0, 0)),
            ("shadow", "shadow", fresh, None, (0, 0, 0, 0)),
            ("ghost", "active", fresh, _ghost(t0 - stale), (2, 1, 2, 1)),
            ("unreadable", "active", fresh, odd, (2, 1, 2, 1))):
        with tempfile.TemporaryDirectory(prefix="pool_check-") as tmp:
            bad += _aside_case(Path(tmp), name=name, t0=t0, mode=mode, age=age, third=third,
                               want=want)
    return bad


def _rows(path: Path, table: str) -> list[tuple[object, ...]]:
    db = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
    try:
        return [tuple(r) for r in db.execute(f"SELECT * FROM {table} ORDER BY id")]
    finally:
        db.close()


class Ended(NamedTuple):
    """What a steward did (`_steward_run`)."""
    exit: int
    summary: psum.Summary
    ledger: list[dict[str, object]]
    final: bool                 # final.db written
    inputs: list[str]           # what is left in the wave's inputs
    log: list[str]


def _inputs(d: Path, seed: pq.Seed) -> None:
    """A wave's shipped inputs: its seed, its bank, a tar per block and a --with file."""
    inputs = d / ps.INPUTS
    inputs.mkdir(parents=True, exist_ok=True)
    (inputs / "jobs.json").write_text(pq.seed_json(seed))
    for name in (ps.BANK, *(f"{j.block}.tar" for j in seed.jobs), "with.txt"):
        (inputs / name).write_bytes(b"")


@dataclass(frozen=True)
class _Unclosed:
    """A feed that grants nothing yet and never closes its queue, as Screened's would between
    its grants (decision 7)."""

    def round(self, queue: pq.Queue) -> None:
        return None


def _steward_run(q: pq.Queue, d: Path, waves: Path, *, listed: Sequence[str],
                 feed: ps.Feed) -> Ended:
    """pool_steward.steward on the wave in `d` fed by `feed`, squeue listing `listed` and its own
    task, its lease three of END_POLL's polls long (so at most three rounds)."""
    log: list[str] = []
    ex = ps.steward(queue=q, wave_dir=d, waves=waves,
                    liveness=_liveness(_squeue(STEWARD_TASK, *listed)), feed=feed,
                    allocation=lambda: None, stale_after=STALE_AFTER,
                    now=lambda: datetime.now(UTC), log=log.append,
                    ends_at=time.time() + 3 * END_POLL, poll_seconds=END_POLL)
    return Ended(exit=int(ex),
                 summary=psum.read_summary((d / ps.SUMMARY).read_text(), "summary.json"),
                 ledger=[json.loads(x) for x in (d / ps.LEDGER).read_text().splitlines()],
                 final=(d / ps.FINAL).exists(),
                 inputs=sorted(p.name for p in (d / ps.INPUTS).iterdir()), log=log)


def _end_finish(tmp: Path) -> list[str]:
    """A wave whose last job finishes: its end."""
    lad = LADDERS["cpu-score"]
    waves = tmp / "waves"
    d = waves / "pool-check"
    jobs = [_small(lad, 1, minutes=30.0), _small(lad, 2, minutes=20.0)]
    d.mkdir(parents=True)
    c = Case(d / "queue.db", lad, jobs)
    _inputs(d, _seed(lad, jobs))
    w = c.worker("w1", cls="m4", minutes=120, lease="s1").name
    for _ in jobs:
        got = c.q.claim(worker=w, minutes_left=c.fresh)
        if got is not None:
            c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.DONE, peak_gb=3.1,
                       minutes=20.0, rc=0)
    e = _steward_run(c.q, d, waves, listed=[w], feed=ps.Fixed())
    same = e.final and all(_rows(d / ps.FINAL, x) == _rows(d / "queue.db", x)
                           for x in ("jobs", "attempts"))
    notes = c.notes + ([] if e.exit == 0 else [f"exit {e.exit}, want 0"]) + (
        [] if same else ["final.db is not the queue's jobs and attempts"]) + (
        [] if e.inputs == ["jobs.json", "with.txt"] else [f"inputs left {e.inputs}"]) + (
        [] if e.summary.state == psum.WaveState.FINISHED else [f"summary {e.summary.state}"]) + (
        [] if "end" in e.ledger[-1] and e.ledger[-1]["end"] == "finished" else
        [f"the ledger's last {e.ledger[-1]}"])
    print(f"end finish: both jobs done: exit {e.exit}, final.db {e.final}, inputs left "
          f"{e.inputs}, {e.summary.state}" + _failing(notes), flush=True)
    return ["finish"] if notes else []


def _end_retry(tmp: Path) -> list[str]:
    """A retry request of a FAILED job, one of a job the wave lacks, and a malformed one (not
    JSON)."""
    lad = LADDERS["cpu-score"]
    waves = tmp / "waves"
    d = waves / "pool-check"
    s = _small(lad)
    d.mkdir(parents=True)
    c = Case(d / "queue.db", lad, [s])
    _inputs(d, _seed(lad, [s]))
    _strike_thrice(c, s)
    requests = d / ps.REQUESTS
    requests.mkdir()
    (requests / "2026-10-21T00:00:00Z.json").write_text(json.dumps({"retry": [s.id]}))
    (requests / "2026-10-21T00:00:01Z.json").write_text(json.dumps({"retry": [99]}))
    malformed = "2026-10-21T00:00:02Z.json"
    (requests / malformed).write_text('{"retry": [1,')
    try:
        e: Ended | None = _steward_run(c.q, d, waves, listed=["w1", "w2", "w3"],
                                       feed=ps.Fixed())
        raised = None
    except ValueError as err:
        e, raised = None, str(err)
    if e is None:
        print(f"end retry: the steward raised {raised} -> FAILS", flush=True)
        return ["retry"]
    first = e.ledger[0]["requests"]
    done = first if isinstance(first, list) else []
    retried = [x for x in done if isinstance(x, dict) and "retried" in x and x["retried"] == 1]
    refused = [x["request"] for x in done if isinstance(x, dict) and "refused" in x]
    logged = [x for x in e.log if x.startswith(f"request {malformed}: ") and "refused" in x]
    notes = c.notes + ([] if e.exit == pw.Exit.RETIRED else [f"exit {e.exit}, want 75"]) + (
        [] if (c.job(s.id).state, c.job(s.id).strikes) == (pq.JobState.PENDING, 0) else
        [f"job 1 {c.job(s.id).state} with {c.job(s.id).strikes} strikes"]) + (
        [] if len(retried) == 1 and refused == ["2026-10-21T00:00:01Z.json", malformed] else
        [f"the ledger's requests {done}"]) + (
        [] if len(logged) == 1 else [f"the malformed request logged {len(logged)} times"]) + (
        [] if not list(requests.iterdir()) else ["requests left"]) + (
        [] if not e.final else ["final.db written"])
    print(f"end retry: job 1 FAILED, a retry of it, of job 99 and a malformed one: job 1 "
          f"{c.job(s.id).state} with {c.job(s.id).strikes} strikes, refused {refused}, exit "
          f"{e.exit}" + _failing(notes), flush=True)
    return ["retry"] if notes else []


def _end_open(tmp: Path) -> list[str]:
    """A wave whose every job is done but whose queue is open (its feed may grant more, as
    Screened's would): no end. It guards `_ended`'s closed() for the Screened feed (decision 7,
    not built): its open seed stands in for Screened's."""
    lad = LADDERS["cpu-score"]
    waves = tmp / "waves"
    d = waves / "pool-check"
    s = _small(lad)
    seed = pq.Seed(header=replace(_header(lad), feed=OPEN_FEED), jobs=(s,), closed=False)
    q = pq.SqliteQueue.open(d / "queue.db", seed)
    _inputs(d, seed)
    q.register(worker="w1", run=RUN, node=NODE, cls="m4", ends_at=time.time() + 7200.0,
               restarts=0)
    got = q.claim(worker="w1", minutes_left=pq.fresh_minutes(lad))
    if got is not None:
        q.finish(got.attempt, worker="w1", outcome=pq.Outcome.DONE, peak_gb=3.1, minutes=20.0,
                 rc=0)
    e = _steward_run(q, d, waves, listed=["w1"], feed=_Unclosed())
    notes = ([] if got is not None and q.jobs()[0].state == pq.JobState.DONE else
             ["job 1 not done"]) + (
        [] if e.exit == pw.Exit.RETIRED else [f"exit {e.exit}, want 75"]) + (
        [] if not e.final else ["final.db written"]) + (
        [] if e.summary.state == psum.WaveState.OPEN else [f"summary {e.summary.state}"])
    print(f"end open: its one job done, its queue open: exit {e.exit}, final.db {e.final}, "
          f"{e.summary.state}" + _failing(notes), flush=True)
    return ["open"] if notes else []


def _end_cancel(tmp: Path) -> list[str]:
    """A cancel request with a job pending, one claimed (its worker live), one LONG and one
    done; then the claimed one's worker finishing it, failed."""
    lad = LADDERS["cpu-score"]
    waves = tmp / "waves"
    d = waves / "pool-check"
    jobs = [_small(lad, 1, minutes=30.0), _small(lad, 2, minutes=60.0),
            _seed_job(3, minutes=30.0, peak_gb=100.0, ladder=lad), _small(lad, 4, minutes=10.0)]
    d.mkdir(parents=True)
    c = Case(d / "queue.db", lad, jobs)
    _inputs(d, _seed(lad, jobs))
    w = c.worker("w1", cls="m4", minutes=120, lease="s1").name
    held = c.q.claim(worker=w, minutes_left=c.fresh)          # job 2, the longest
    other = c.worker("w2", cls="m4", minutes=120, lease="s2").name
    four = c.q.claim_job(worker=other, job=4)
    if four is not None:
        c.q.finish(four.attempt, worker=other, outcome=pq.Outcome.DONE, peak_gb=3.1,
                   minutes=5.0, rc=0)
    (d / ps.REQUESTS).mkdir()
    (d / ps.REQUESTS / "2026-10-21T00:00:00Z.json").write_text(json.dumps({"cancel": True}))
    e = _steward_run(c.q, d, waves, listed=[w, other], feed=ps.Fixed())
    states = {j.id: j.state for j in c.q.jobs()}
    want = {1: pq.JobState.CANCELLED, 2: pq.JobState.CANCELLED, 3: pq.JobState.CANCELLED,
            4: pq.JobState.DONE}
    late = None if held is None else c.q.finish(held.attempt, worker=w,
                                                outcome=pq.Outcome.FAILED, peak_gb=3.1,
                                                minutes=25.0, rc=1)
    notes = c.notes + ([] if held is not None and held.job.id == 2 else [f"claim {held}"]) + (
        [] if e.exit == 0 else [f"exit {e.exit}, want 0"]) + (
        [] if states == want else [f"jobs {states}, want {want}"]) + (
        [] if e.final else ["no final.db"]) + (
        [] if e.inputs == ["jobs.json", "with.txt"] else [f"inputs left {e.inputs}"]) + (
        [] if e.summary.state == psum.WaveState.CANCELLED else [f"summary {e.summary.state}"]) + (
        [] if late == pq.JobState.CANCELLED else [f"the claimed job's later failure: {late}"])
    print(f"end cancel: jobs {({i: s.value for i, s in states.items()})}, exit {e.exit}, "
          f"final.db {e.final}, {e.summary.state}; "
          f"the claimed job's later failure leaves it {late}" + _failing(notes), flush=True)
    return ["cancel"] if notes else []


def _end_long(tmp: Path) -> list[str]:
    """A wave whose one job not done waits for the long lane: no end."""
    lad = LADDERS["cpu-score"]
    waves = tmp / "waves"
    d = waves / "pool-check"
    jobs = [_small(lad, 1, minutes=30.0), _seed_job(2, minutes=30.0, peak_gb=100.0, ladder=lad)]
    d.mkdir(parents=True)
    c = Case(d / "queue.db", lad, jobs)
    _inputs(d, _seed(lad, jobs))
    w = c.worker("w1", cls="m4", minutes=120, lease="s1").name
    got = c.q.claim(worker=w, minutes_left=c.fresh)
    if got is not None:
        c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.DONE, peak_gb=3.1, minutes=20.0,
                   rc=0)
    e = _steward_run(c.q, d, waves, listed=[w], feed=ps.Fixed())
    want = (psum.LongJob(job=2, block=jobs[1].block, item=jobs[1].item, node=NODE,
                         mem_gb=math.ceil(jobs[1].gb), minutes=30.0),)
    notes = c.notes + ([] if e.exit == pw.Exit.RETIRED else [f"exit {e.exit}, want 75"]) + (
        [] if not e.final else ["final.db written"]) + (
        [] if ps.BANK in e.inputs else ["the bank deleted"]) + (
        [] if e.summary.state == psum.WaveState.CLOSED else [f"summary {e.summary.state}"]) + (
        [] if e.summary.long == want else [f"long {e.summary.long}, want {want}"])
    print(f"end long: job 1 done, job 2 waiting for the long lane: exit {e.exit}, final.db "
          f"{e.final}, {e.summary.state}, long {[x.job for x in e.summary.long]}"
          + _failing(notes), flush=True)
    return ["long"] if notes else []


def _end_command(tmp: Path) -> list[str]:
    """pool.py `steward` as a cluster_submit task runs it (its environment set: HOME, the Slurm
    array ids and end time, USER), a relative --board with no allocation on it, on a shipped wave
    whose one job is done and whose worker has retired (so squeue is never asked): the metric
    named (metrics.RUNTIME), the wave ended (0), its summary FINISHED and naming the metric."""
    import metrics  # here: its numpy and scipy are this case's alone
    lad = LADDERS["cpu-score"]
    s = _small(lad)
    w = _shipped(tmp, _seed(lad, [s]))
    (w.inputs / ps.BANK).write_bytes(b"")
    c = Case(w.queue, lad, [s])
    x = c.worker("w1", cls="m4", minutes=120, lease="s1").name
    got = c.q.claim(worker=x, minutes_left=c.fresh)
    if got is not None:
        c.q.finish(got.attempt, worker=x, outcome=pq.Outcome.DONE, peak_gb=3.1, minutes=20.0,
                   rc=0)
    c.q.retire(x)
    environ = {**os.environ, "HOME": str(w.home), "SLURM_ARRAY_JOB_ID": STEWARD_TASK.split("_")[0],
               "SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_END_TIME": str(int(time.time() + 3600)),
               "USER": "pool-check"}
    with redirect_stdout(io.StringIO()):
        ex = int(pl.main(["steward", "--wave", "pool-check", "--board", "market",
                          "--stale-after-minutes", "30"], environ))
    d = w.queue.parent
    summary = psum.read_summary((d / ps.SUMMARY).read_text(), "summary.json")
    notes = c.notes + ([] if ex == 0 else [f"exit {ex}, want 0"]) + (
        [] if c.q.metric() == metrics.RUNTIME else [f"the queue's metric {c.q.metric()}"]) + (
        [] if (summary.state, summary.metric) == (psum.WaveState.FINISHED, metrics.RUNTIME) else
        [f"summary {summary.state} naming {summary.metric}"]) + (
        [] if (d / ps.FINAL).exists() and not (w.inputs / ps.BANK).exists() else
        ["no final.db, or the bank kept"])
    print(f"end command: pool.py steward on a wave whose job is done: exit {ex}, "
          f"{summary.state}, the metric named" + _failing(notes), flush=True)
    return ["command"] if notes else []


def _end_fixed_open(tmp: Path) -> list[str]:
    """A fixed wave's seed that leaves its queue open (nothing would close it, so its wave would
    never end): refused by Seed (ValueError naming the wave) and by read_seed of a jobs.json
    edited so (its `closed` false)."""
    lad = LADDERS["cpu-score"]
    notes: list[str] = []
    try:
        pq.Seed(header=_header(lad), jobs=(_small(lad),), closed=False)
        notes.append("Seed accepted it")
    except ValueError as e:
        notes += [] if "pool-check" in str(e) else [f"Seed refused it without naming the wave: {e}"]
    edited = json.loads(pq.seed_json(_seed(lad, [_small(lad)])))
    edited["closed"] = False
    try:
        pq.read_seed(json.dumps(edited), "jobs.json")
        notes.append("read_seed accepted it")
    except ValueError as e:
        notes += [] if "pool-check" in str(e) else [f"read_seed refused it without naming the "
                                                    f"wave: {e}"]
    print("end fixed-open: a fixed feed's seed with its queue open, refused by Seed and "
          "read_seed" + _failing(notes), flush=True)
    return ["fixed-open"] if notes else []


def end(_args: list[str]) -> list[str]:
    bad: list[str] = []
    for case in (_end_finish, _end_retry, _end_cancel, _end_long, _end_open, _end_fixed_open,
                 _end_command):
        with tempfile.TemporaryDirectory(prefix="pool_check-") as tmp:
            bad += case(Path(tmp))
    return bad


GUARDS: dict[str, Guard] = {"claim": Guard(run=claim, args=()),
                            "transitions": Guard(run=transitions, args=()),
                            "class": Guard(run=classes, args=("rows",)),
                            "walltime": Guard(run=walltime, args=()),
                            "oom": Guard(run=oom, args=()),
                            "identity": Guard(run=identity, args=()),
                            "worker": Guard(run=worker, args=()),
                            "dead-claim": Guard(run=dead_claim, args=()),
                            "aside": Guard(run=aside, args=()),
                            "end": Guard(run=end, args=())}


# ---- faults ---------------------------------------------------------------------------------

@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _split_claim(self: pq.SqliteQueue, *, worker: str, minutes_left: float) -> pq.Claim | None:
    """SqliteQueue.claim with its select and its update in two transactions, SPLIT_SLEEP apart."""
    with self._tx() as db:
        w = self._live(db, worker)
        pq._one_claim(db, worker)
        job = self._pick(db, w, minutes_left=minutes_left)
    if job is None:
        return None
    time.sleep(SPLIT_SLEEP)
    with self._tx() as db:
        return self._take(db, w, job)


_PROMOTED = pq._promoted


def _dropped_on_top(ladder: Ladder, *, cls: str, lane: pq.Lane, gb: float, peak_gb: float,
                    peak: float) -> pq.Placed:
    """pool_queue._promoted, with a pool's job past the top rung failed, not sent long."""
    p = _PROMOTED(ladder, cls=cls, lane=lane, gb=gb, peak_gb=peak_gb, peak=peak)
    return (p._replace(state=pq.JobState.FAILED, lane=lane)
            if lane == pq.Lane.POOL and p.lane == pq.Lane.LONG else p)


_FITTED = pp.Estimator._fitted


def _unmargined(self: pp.Estimator, block: str, item: str) -> pp.Estimate:
    """pool_plan.Estimator._fitted without the fit's margin (fit_mem)."""
    return _FITTED(self, block, item)._replace(mem_margin=1.0)


def _family_pool(runs: Sequence[pp.Observed], *, item: str, fit_min_blocks: int) -> pp.Pool:
    """pool_plan._pool with every item fitted at the family's rate."""
    return pp.Pool(scope="family", runs=list(runs))


_SIZED = pp._sized


def _all_long(e: pp.Estimate, *, ladder: Ladder) -> pp.Sized:
    """pool_plan._sized with every job sent to the long lane, needing fit_mem times its peak."""
    gb = ladder.fit_mem * e.peak_gb
    return _SIZED(e, ladder=ladder)._replace(lane=pq.Lane.LONG, gb=gb, hold_gb=gb)


_CLAIM = pq.SqliteQueue.claim


def _claim_any(self: pq.SqliteQueue, *, worker: str, minutes_left: float) -> pq.Claim | None:
    """SqliteQueue.claim ignoring the walltime left: every pending job fits."""
    return _CLAIM(self, worker=worker, minutes_left=math.inf)


def _no_exclusion() -> AbstractContextManager[None]:
    """pool_queue's claim excluding nothing: a lease's worker may claim a job it attempted."""
    return _patched(pq, "_attempted_by", lambda db, run: [])


_MINUTES_LEFT = pw._minutes_left


def _left_once() -> AbstractContextManager[None]:
    """pool_worker._minutes_left computed once a worker (its first answer for an end time, ever
    after): the walltime spent by a job never counted."""
    first: dict[float, float] = {}

    def once(*, ends_at: float, now: float, margin_minutes: float) -> float:
        if ends_at not in first:
            first[ends_at] = _MINUTES_LEFT(ends_at=ends_at, now=now,
                                           margin_minutes=margin_minutes)
        return first[ends_at]
    return _patched(pw, "_minutes_left", once)


@contextmanager
def _unchecked() -> Iterator[None]:
    """pool.py's checks of --node, --job's node and --class patched out."""
    with _patched(pl, "check_node", lambda node, environ: None), _patched(
            pl, "check_job_node", lambda job, queue, environ: None), _patched(
            pl, "check_class", lambda cls, header: None):
        yield


def _cancel_waiting(self: pq.SqliteQueue) -> int:
    """SqliteQueue.cancel leaving a claimed job to run on (CLAIMED), only the waiting ones
    CANCELLED: its later failure takes it back to PENDING."""
    with self._tx() as db:
        return db.execute("UPDATE jobs SET state = ? WHERE state IN (?, ?)",
                          (pq.JobState.CANCELLED.value, pq.JobState.PENDING.value,
                           pq.JobState.LONG.value)).rowcount


def _reap_by_age(queue: pq.Queue, liveness: ps.Liveness, *, now: float) -> list[pq.Reaped]:
    """pool_steward._reap deciding by a claim's age, not squeue's word (mycooc's measured and
    rejected threshold): every worker whose open attempt started AGE_LIMIT before `now`."""
    if not isinstance(queue, pq.SqliteQueue):
        raise TypeError(f"by-age reads a SqliteQueue's attempts, not a {type(queue).__name__}")
    db = sqlite3.connect(queue.path.absolute().as_uri() + "?mode=ro", uri=True)
    try:
        old = [r[0] for r in db.execute("SELECT worker FROM attempts WHERE end IS NULL AND"
                                        " start < ? ORDER BY id", (now - AGE_LIMIT,))]
    finally:
        db.close()
    return [r for w in old for r in [queue.reap(w, now=now)] if r is not None]


def _blind(self: ps.Squeue, workers: Sequence[str]) -> frozenset[str]:
    """pool_steward.Squeue.dead reading a squeue that fails (or answers without the steward's own
    task) as an empty answer: every worker it does not list has ended."""
    try:
        out = self.run(["squeue", "-h", "-r", "-u", self.user, "-o", "%i"])
        listed = set(out.stdout.split()) if out.returncode == 0 else set()
    except (OSError, subprocess.SubprocessError):
        listed = set()
    return frozenset(w for w in workers if w not in listed)


def _whole_yields(*, yields: int, workers: Mapping[str, int], wave: str) -> int:
    """pool_summary.aside_share asking all the yields of a wave's own workers there."""
    return min(yields, workers[wave])


def _ended_with_long(jobs: Sequence[pq.Job], *, closed: bool) -> bool:
    """pool_steward._ended counting a job waiting for the long lane as ended."""
    return closed and all(j.state in (*ps.ENDED, pq.JobState.LONG) for j in jobs)


def _ended_open(jobs: Sequence[pq.Job], *, closed: bool) -> bool:
    """pool_steward._ended ignoring whether the queue is closed."""
    return all(j.state in ps.ENDED for j in jobs)


_LIVE_SUMMARIES = ps._live_summaries


def _unread_fatal(waves: Path, wave_dir: Path, *, now: datetime,
                  stale_after: timedelta) -> ps.Others:
    """pool_steward._live_summaries letting a summary that does not read stop the round."""
    for path in sorted(waves.glob(f"*/{ps.SUMMARY}")):
        psum.read_summary(path.read_text(), str(path))
    return _LIVE_SUMMARIES(waves, wave_dir, now=now, stale_after=stale_after)


def _requests_fatal(queue: pq.Queue, requests: Path, *,
                    log: Callable[[str], None]) -> list[dict[str, object]]:
    """pool_steward._requests letting a request that does not read, or that the queue refuses,
    stop the round."""
    done: list[dict[str, object]] = []
    for path in sorted(requests.glob("*.json")):
        what = ps.read_request(path.read_text(), str(path)).apply(queue)
        path.unlink()
        done.append({"request": path.name, **what})
    return done


_OUTCOME = pw._outcome


def _oom_failed(rc: int, *, before: int | None, after: int | None) -> pq.Outcome:
    """pool_worker._outcome recording an OOM as FAILED."""
    o = _OUTCOME(rc, before=before, after=after)
    return pq.Outcome.FAILED if o == pq.Outcome.OOM else o


_OOM_EVIDENCE = pw.oom_evidence


def _unified_first(proc_self_cgroup: str, root: Path) -> pw.OomEvidence:
    """pool_worker.oom_evidence taking a v2 line first, wherever v1's memory controller is."""
    v2 = [line.split(":", 2)[2] for line in proc_self_cgroup.splitlines()
          if line.startswith("0::")]
    if not v2:
        return _OOM_EVIDENCE(proc_self_cgroup, root)
    found = pw.CgroupV2(path=root / v2[0].lstrip("/"))
    return found if found.kills() is not None else pw.Unreadable(why=f"{found.about} unread")


FAULTS: dict[str, dict[str, Callable[[], AbstractContextManager[None]]]] = {
    "claim": {"split-tx": lambda: _patched(pq.SqliteQueue, "claim", _split_claim)},
    "transitions": {"late-finish": lambda: _patched(pq, "_holds", lambda job, attempt: True),
                    "drop-on-top": lambda: _patched(pq, "_promoted", _dropped_on_top),
                    "any-worker": lambda: _patched(pq, "_mine", lambda attempt, worker: True),
                    "second-claim": lambda: _patched(pq, "_one_claim", lambda db, worker: None),
                    "no-exclude": _no_exclusion,
                    "claimed-runs-on": lambda: _patched(pq.SqliteQueue, "cancel",
                                                        _cancel_waiting)},
    "class": {"median-rate": lambda: _patched(pp, "_worst_rate", statistics.median),
              "no-fit-margin": lambda: _patched(pp.Estimator, "_fitted", _unmargined),
              "family-rate": lambda: _patched(pp, "_pool", _family_pool),
              "all-long": lambda: _patched(pp, "_sized", _all_long),
              "no-capacity": lambda: _patched(pp, "_able", lambda nodes, *, usable_gb, need:
                                              list(nodes)),
              "with-proposed": lambda: _patched(pp, "PROPOSED_AND_SCORED", frozenset()),
              "one-run": lambda: _patched(pp, "_demanding", lambda *, peak, wall: peak)},
    "walltime": {"no-fit": lambda: _patched(pq.SqliteQueue, "claim", _claim_any),
                 "left-once": _left_once},
    "oom": {"fail-on-oom": lambda: _patched(pw, "_outcome", _oom_failed),
            "unified-first": lambda: _patched(pw, "oom_evidence", _unified_first)},
    "identity": {"no-identity": lambda: _patched(pw, "_refused", lambda header, commit: None)},
    "worker": {"fail-worker": lambda: _patched(pw, "FAILED_IN_A_ROW", 1),
               "no-exclude": _no_exclusion,
               "no-retire": lambda: _patched(pq.SqliteQueue, "retire", lambda self, worker: 0),
               "no-kill": lambda: _patched(pw, "_kill", lambda proc: None),
               "pid-only": lambda: _patched(pw, "_kill", lambda proc: proc.kill()),
               "no-rebirth": lambda: _patched(pq, "_reborn", lambda had, restarts: False),
               "no-checks": _unchecked},
    "dead-claim": {"by-age": lambda: _patched(ps, "_reap", _reap_by_age),
                   "blind": lambda: _patched(ps.Squeue, "dead", _blind)},
    "aside": {"whole-yields": lambda: _patched(psum, "aside_share", _whole_yields),
              "stale-kept": lambda: _patched(ps, "_stale", lambda s, *, now, stale_after: False),
              "unread-fatal": lambda: _patched(ps, "_live_summaries", _unread_fatal)},
    "end": {"early-end": lambda: _patched(ps, "_ended", _ended_with_long),
            "claimed-runs-on": lambda: _patched(pq.SqliteQueue, "cancel", _cancel_waiting),
            "open-ended": lambda: _patched(ps, "_ended", _ended_open),
            "request-fatal": lambda: _patched(ps, "_requests", _requests_fatal),
            "open-fixed": lambda: _patched(pq, "_open_fixed", lambda header, *, closed: False)},
}


# ---- the command line -----------------------------------------------------------------------

def main(guard: str, fault: str | None, args: list[str]) -> None:
    if fault is None:
        bad = GUARDS[guard].run(args)
    else:
        with FAULTS[guard][fault]():
            bad = GUARDS[guard].run(args)
    verdict = "FAILS on " + ", ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


def both_ways(guards: list[str], args: list[str]) -> None:
    """Each guard as it is, then under each of its faults (FAULTS), one after another in this
    process, `args` passed to each. Raises naming every run not as `main` expects."""
    wrong = []
    for guard in guards:
        for fault in (None, *FAULTS[guard]):
            try:
                main(guard, fault, args)
            except SystemExit as e:
                wrong.append(f"{guard}{'' if fault is None else f' under {fault}'}: {e.code}")
    print(f"both ways, {', '.join(guards)}: "
          + ("as expected" if not wrong else "NOT as expected: " + "; ".join(wrong)), flush=True)
    if wrong:
        raise SystemExit(1)


def command_line(a: list[str]) -> None:
    """`<guard> [--fault <name>] [args ...]` or `both-ways <guards,> [args ...]` (the module's
    doc)."""
    if len(a) >= 2 and a[0] == "both-ways":
        names = a[1].split(",")
        unknown = [g for g in names if g not in GUARDS]
        if unknown:
            raise SystemExit(f"no guard {', '.join(unknown)} (guards: {', '.join(GUARDS)})")
        both_ways(names, a[2:])
        return
    if not a or a[0] not in GUARDS:
        raise SystemExit(__doc__)
    guard, faults = GUARDS[a[0]], FAULTS[a[0]]
    fault, args = None, a[1:]
    if args[:1] == ["--fault"]:
        if len(args) < 2 or args[1] not in faults:
            raise SystemExit(f"{a[0]} has no fault {args[1] if len(args) > 1 else '(none named)'}"
                             f" (faults: {', '.join(faults)})")
        fault, args = args[1], args[2:]
    if len(args) != len(guard.args):
        takes = " ".join(f"<{n}>" for n in guard.args) or "no arguments"
        raise SystemExit(f"{a[0]} takes {takes}, got {args} (faults: {', '.join(faults)})")
    main(a[0], fault, args)


if __name__ == "__main__":
    command_line(sys.argv[1:])
