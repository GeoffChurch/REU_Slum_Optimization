"""Guards of the lease pool (docs/superpowers/specs/2026-10-21-lease-pool-design.md), each with its
faults (a patch in one process, no file edited, as gap_clear_check.py's): a guard passes on the
code as it is and must fail under each of its faults.

    S=research/roadless/pool_check.py      # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S claim [fault]         # fault: split-tx
    uv run python $S transitions [fault]   # faults: late-finish, drop-on-top, any-worker,
                                           # second-claim
    uv run python $S both-ways <guards,> [args ...]   # each guard, then under each of its faults

A guard's arguments follow its fault (`<guard> [fault] [args ...]`). Each guard declares its
arguments (GUARDS), and run alone it refuses any other count of them, naming its faults, so a
mistyped fault is no argument run unfaulted; `both-ways` passes its arguments to every guard it
names. `claim` and `transitions` take none.

`claim`: CLAIM_JOBS jobs of one node and one class under cpu-score, estimated evenly over (0, 110]
minutes (a fresh lease less its margin: every job a plan would pool), in a queue in a temporary
directory on local disk; CLAIM_WORKERS processes (fork), started together, each opens the queue
(the first makes it from the seed), registers a worker and claims with a fresh worker's 110
minutes left (every job fits: the fit is `walltime`'s), finishes the job done and claims again
until nothing is left, sleeping 0 to CLAIM_SLEEP between operations. Fails on a job
with two attempts, a job not done, an attempt finished by a worker other than its claimer (the
queue's attempts against what each process says it finished), and a worker whose claims'
estimates ever rise (each claim is the longest pending at its time, so a worker's never rise).
Fault `split-tx`: SqliteQueue.claim selects in one transaction and updates in a second,
SPLIT_SLEEP apart: some job is claimed twice.

`transitions`: on a queue under the preset cpu-score (lease 120, margin 10, strikes 3), each case a
fresh queue, the job as it stands after each event against the job it should be (every field:
its state, class, lane, strikes, gb, peak_gb and minutes among them):
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
  (past 120 - 10).
- `late-finish`: a reaped worker's late finish DONE -> the attempt recorded (its rc and peak), the
  job as the reaper left it (the spec's 3.2, "Done").
- `retry`: retry of a FAILED job -> PENDING, strikes 0.
- `cancel`: every PENDING and LONG job CANCELLED, a CLAIMED one untouched.
- `claim-job`: a pool's claim leaves a LONG job; claim_job takes it (CLAIMED); its finish FAILED
  without rows -> LONG, strikes 1.
- `aside`: ask_aside(units=2) of class m4 on a node with three live workers of it, one idle, and
  an idle worker of m12 -> the idle m4 worker, then the one whose job ends soonest; asked again
  with the same `since` -> none.
- `other-worker`: a worker's finish DONE of another worker's attempt is refused (ValueError), the
  job still the claimer's and the attempt unrecorded.
- `one-claim`: a worker holding a claim is refused another, by claim and by claim_job
  (ValueError), the other job left PENDING.
Faults: `late-finish` (finish ignores whose claim it is: the reaped worker's late DONE marks the
job done), `drop-on-top` (past the top rung the job is FAILED, not LONG: the spec's "OOM promotes;
it never drops"), `any-worker` (finish takes any worker's word for an attempt: the other worker's
DONE marks the job done), `second-claim` (a worker holding a claim may claim again).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import queue
import random
import sqlite3
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from itertools import pairwise
from multiprocessing.queues import Queue as MpQueue
from multiprocessing.synchronize import Barrier
from pathlib import Path
from typing import NamedTuple

from cluster_submit.context import STAMP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pool_queue as pq  # noqa: E402
from pool_ladder import LADDERS, Ladder, class_name, rung_of  # noqa: E402

NODE = "orion"
COMMIT = "0123456789abcdef0123456789abcdef01234567"   # a wave's commit (any, the same for all)
CLAIM_JOBS = 200         # claim: jobs, one class, one node, estimated evenly over (0, 110]
CLAIM_WORKERS = 4        # ... processes, each one worker
CLAIM_SLEEP = 0.010      # ... the most it sleeps between operations (s)
CLAIM_WAIT = 600.0       # ... the most the guard waits for a process (s)
SPLIT_SLEEP = 0.005      # split-tx: between the claim's select and its update (s)


# ---- the queues the guards make, as a wave's plan makes them --------------------------------

def _header(ladder: Ladder) -> pq.Header:
    return pq.Header(wave="pool-check", commit=COMMIT, family="pool-check", ladder=ladder,
                     created=datetime.now(UTC).strftime(STAMP), feed="fixed")


def _seed_job(i: int, *, minutes: float, peak_gb: float, ladder: Ladder) -> pq.JobSeed:
    """Job `i` as a plan makes it from a stored peak and a time estimate of `minutes`: needing
    the peak times stored_mem, of the smallest rung holding that; the long lane's (of the top
    class) past the top rung or past a fresh lease's minutes (pool_queue.fresh_minutes)."""
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
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
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
                   ends_at=time.time() + 3600.0)
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
        self.path = path
        self.notes: list[str] = []

    def job(self, i: int) -> pq.Job:
        return next(j for j in self.q.jobs() if j.id == i)

    def worker(self, name: str, *, cls: str, minutes: float) -> Registered:
        """Register `name`, a worker of `cls` with `minutes` of its walltime left."""
        ends = time.time() + 60.0 * minutes
        self.q.register(worker=name, run=f"pool-check-{NODE}-{cls}-s1", node=NODE, cls=cls,
                        ends_at=ends)
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
    w = c.worker("w1", cls=s.cls, minutes=120).name
    got = c.claim(w, left=110, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.DONE, peak_gb=3.1,
                        minutes=25.0, rc=0)
        c.after("finish DONE", st, _as_job(s, state=pq.JobState.DONE))
    return c.notes


def _failed_progress(path: Path, lad: Ladder) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    w = c.worker("w1", cls=s.cls, minutes=120).name
    got = c.claim(w, left=110, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        c.q.wrote(got.attempt)
        c.q.wrote(got.attempt)
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.FAILED, peak_gb=3.1,
                        minutes=25.0, rc=1)
        c.after("finish FAILED, two rows written", st, _as_job(s, state=pq.JobState.PENDING))
    return c.notes


def _strike_thrice(c: Case, s: pq.JobSeed, w: str) -> None:
    """Three claims of `s` by `w`, each finished FAILED with no row written."""
    for n, state in enumerate((pq.JobState.PENDING, pq.JobState.PENDING, pq.JobState.FAILED), 1):
        got = c.claim(w, left=110, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w)._replace(
            strikes=n - 1))
        if got is None:
            return
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.FAILED, peak_gb=3.1,
                        minutes=25.0, rc=1)
        c.after(f"finish FAILED {n}, no row", st, _as_job(s, state=state)._replace(strikes=n))


def _failed_strikes(path: Path, lad: Ladder) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    w = c.worker("w1", cls=s.cls, minutes=120).name
    _strike_thrice(c, s, w)
    return c.notes


def _oom_promote(path: Path, lad: Ladder) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    w = c.worker("w1", cls=s.cls, minutes=120).name
    got = c.claim(w, left=110, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
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
    w = c.worker("w1", cls=s.cls, minutes=120).name
    got = c.claim(w, left=110, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        st = c.q.finish(got.attempt, worker=w, outcome=pq.Outcome.OOM, peak_gb=88.0,
                        minutes=5.0, rc=-9)
        c.after("finish OOM, peak 88 in m88", st, _as_job(s, state=pq.JobState.LONG)._replace(
            lane=pq.Lane.LONG, peak_gb=88.0, gb=132.0))
    return c.notes


def _reaped(path: Path, lad: Ladder, *, rows: int) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    w = c.worker("w1", cls=s.cls, minutes=120).name
    got = c.claim(w, left=110, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        for _ in range(rows):
            c.q.wrote(got.attempt)
        r = c.q.reap(w, now=time.time())
        c.check("reap", r == pq.Reaped(worker=w, run=f"pool-check-{NODE}-m4-s1", job=s.id,
                                       outcome=pq.Outcome.REAPED), f"{r}")
        c.after(f"reap before its end, {rows} rows", None, _as_job(
            s, state=pq.JobState.PENDING)._replace(strikes=0 if rows else 1))
    return c.notes


def _walltime(path: Path, lad: Ladder, *, ran: float, want: pq.JobState) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    reg = c.worker("w1", cls=s.cls, minutes=ran)
    w = reg.name
    got = c.claim(w, left=ran - lad.margin_minutes, want=_as_job(
        s, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        r = c.q.reap(w, now=reg.ends + 60.0)
        c.check("reap", r.job == s.id and r.outcome == pq.Outcome.WALLTIME, f"{r}")
        c.after(f"reap after its end, {ran:g} minutes run", None, _as_job(s, state=want)._replace(
            minutes=ran, lane=pq.Lane.LONG if want == pq.JobState.LONG else pq.Lane.POOL))
    return c.notes


def _late_finish(path: Path, lad: Ladder) -> list[str]:
    s = _small(lad)
    c = Case(path, lad, [s])
    w = c.worker("w1", cls=s.cls, minutes=120).name
    got = c.claim(w, left=110, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
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
    w = c.worker("w1", cls=s.cls, minutes=120).name
    _strike_thrice(c, s, w)
    n = c.q.retry([s.id])
    c.check("retry", n == 1, f"{n} retried, want 1")
    c.after("retry", None, _as_job(s, state=pq.JobState.PENDING))
    return c.notes


def _cancel(path: Path, lad: Ladder) -> list[str]:
    held, waits = _small(lad, 1, minutes=30.0), _small(lad, 3, minutes=20.0)
    long = _seed_job(2, minutes=30.0, peak_gb=100.0, ladder=lad)
    c = Case(path, lad, [held, long, waits])
    c.check("the seed", long.lane == pq.Lane.LONG, f"job 2's lane {long.lane}")
    w = c.worker("w1", cls=held.cls, minutes=120).name
    got = c.claim(w, left=110, want=_as_job(held, state=pq.JobState.CLAIMED, worker=w))
    if got is not None:
        n = c.q.cancel()
        c.check("cancel", n == 2, f"{n} cancelled, want 2")
        c.after("cancel", None, _as_job(held, state=pq.JobState.CLAIMED, worker=w))
        c.after("cancel", None, _as_job(long, state=pq.JobState.CANCELLED))
        c.after("cancel", None, _as_job(waits, state=pq.JobState.CANCELLED))
    return c.notes


def _claim_job(path: Path, lad: Ladder) -> list[str]:
    s = _seed_job(1, minutes=30.0, peak_gb=100.0, ladder=lad)
    c = Case(path, lad, [s])
    pool = c.worker("pool", cls=s.cls, minutes=120).name
    c.check("a pool's claim", c.q.claim(worker=pool, minutes_left=110) is None,
            "took the LONG job")
    w = c.worker("long", cls=s.cls, minutes=3 * 24 * 60).name
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
    c.worker("w-m12", cls="m12", minutes=120)
    for name, s in (("w-a", jobs[0]), ("w-b", jobs[1])):
        w = c.worker(name, cls=s.cls, minutes=120).name
        c.claim(w, left=110, want=_as_job(s, state=pq.JobState.CLAIMED, worker=w))
    c.worker("w-c", cls="m4", minutes=120)
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
    w = c.worker("w1", cls=s.cls, minutes=120).name
    other = c.worker("w2", cls=s.cls, minutes=120).name
    claimed = _as_job(s, state=pq.JobState.CLAIMED, worker=w)
    got = c.claim(w, left=110, want=claimed)
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
    w = c.worker("w1", cls=first.cls, minutes=120).name
    if c.claim(w, left=110, want=_as_job(first, state=pq.JobState.CLAIMED, worker=w)) is None:
        return c.notes
    tries: tuple[tuple[str, Callable[[], pq.Claim | None]], ...] = (
        ("claim", lambda: c.q.claim(worker=w, minutes_left=110)),
        ("claim_job", lambda: c.q.claim_job(worker=w, job=second.id)))
    for how, again in tries:
        try:
            got = again()
            c.notes.append(f"a second {how} by {w}: {got and got.job.id}, want refused")
        except ValueError:
            pass
    c.after("the second claims", None, _as_job(second, state=pq.JobState.PENDING))
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
)


def transitions(_args: list[str]) -> list[str]:
    lad = LADDERS["cpu-score"]
    if (lad.lease_minutes, lad.margin_minutes, lad.strikes, lad.fit_mem, lad.rungs) != (
            120, 10, 3, 1.5, (4, 12, 24, 56, 88)):
        return [f"cpu-score is not the cases' (lease 120, margin 10, strikes 3, fit_mem 1.5, "
                f"rungs 4-88): {lad}"]
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


GUARDS: dict[str, Guard] = {"claim": Guard(run=claim, args=()),
                            "transitions": Guard(run=transitions, args=())}


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
        job = self._pick(db, w, minutes_left)
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


FAULTS: dict[str, dict[str, Callable[[], AbstractContextManager[None]]]] = {
    "claim": {"split-tx": lambda: _patched(pq.SqliteQueue, "claim", _split_claim)},
    "transitions": {"late-finish": lambda: _patched(pq, "_holds", lambda job, attempt: True),
                    "drop-on-top": lambda: _patched(pq, "_promoted", _dropped_on_top),
                    "any-worker": lambda: _patched(pq, "_mine", lambda attempt, worker: True),
                    "second-claim": lambda: _patched(pq, "_one_claim", lambda db, worker: None)},
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
    """`<guard> [fault] [args ...]` or `both-ways <guards,> [args ...]` (the module's doc)."""
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
    named = len(a) >= 2 and a[1] in faults
    args = a[2:] if named else a[1:]
    if len(args) != len(guard.args):
        takes = " ".join(f"<{n}>" for n in guard.args) or "no arguments"
        raise SystemExit(f"{a[0]} takes {takes}, got {args} (faults: {', '.join(faults)})")
    main(a[0], a[1] if named else None, args)


if __name__ == "__main__":
    command_line(sys.argv[1:])
