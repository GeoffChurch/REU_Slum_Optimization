"""A wave's queue (docs/superpowers/specs/2026-10-21-lease-pool-design.md, 3.2 and 3.4's
promotion), after GeoffChurch/mycooc's scheduler/jobs.py: the jobs a wave grants, which the
workers of its pools claim, run and record, each claim one attempt.

- **Storage.** SQLite in journal_mode DELETE (WAL's shared memory is not coherent over NFS, the
  cluster's home: mycooc's lesson 1); each operation opens its own connection and does all it
  reads and writes in one BEGIN IMMEDIATE transaction, so two workers can never claim one job
  while the lock holds. It is behind one Protocol (`Queue`), so it can move to Postgres with
  mycooc's without its callers changing.
- **Tables.** `header` (one row: the wave's header as JSON, written once, and the metric's name),
  `jobs` (the seed's jobs, unique on block and item, each with its state, strikes and the claim
  holding it), `attempts` (one per claim: the job, the worker and its node and class, start, end,
  outcome, peak GB, minutes, exit code and the rows it wrote), `workers` (each registered worker's
  run, node, class, Slurm end time, restarts, and when it registered and retired), `reaped` (each
  reap: the worker, the job it held and its outcome, whether its walltime had ended), `aside` (the
  workers asked to step aside) and `closed`.
- **A claim** takes, of the PENDING jobs of the worker's node and class whose estimate fits the
  minutes it has left, the longest (then the lowest id), so a lease's big jobs come first and
  small ones fill its tail; a LONG job is the long lane's, claimed by id (`claim_job`). A worker
  holds one claim at a time: a second is refused where claims are made. No worker claims a job
  any worker of its run has attempted (its lease: the run's tasks, all on one node), read in the
  claim's transaction from the attempts of the run's registered workers: a bad node spends one
  strike a job across the whole lease, and a job that failed for its own reasons is retried by
  the next lease (the controller's ruling, Option A).
- **Done is recorded only by the claim's holder**, only while the claim is still its own: a
  reaped worker's late finish stays an attempt (its exit code, peak and minutes recorded, its
  outcome the reaper's) and the job stays as the reaper left it.
- **Strikes** (the plan's Ruling 6, mycooc's lesson 7): an attempt that ends failed, or is reaped
  before its walltime, with no row written, is a strike; after the ladder's `strikes` the job is
  FAILED, for a person (`retry`). A walltime kill is no strike (it raises the time estimate to
  what the attempt ran, and past `fresh_minutes` the job goes long), nor an OOM (it
  promotes), nor a failure that wrote a row (the job ran further: run again, it keeps what stands).
- **OOM promotes; it never drops**: an OOM's job moves a rung up, needing the ladder's `fit_mem`
  times its peak (the peak that killed it is a floor of its need, as uncertain as a fit), its
  estimates never lowered; past the top rung it goes to the long lane.
- **A dead worker's claim goes back**, as if never claimed (lesson 6), on evidence only: the
  worker's own exit (`retire`), the steward's reap on squeue's word (`reap`, which decides a
  walltime kill by the worker's Slurm end time against `now`), or its task registering again
  requeued (Slurm keeps a requeued task's array id and counts its restarts: `register` reaps its
  earlier self). Every reap is recorded (`reaped`), in the same transaction, for the summary. A
  job waits, between claims, in its lane's state: PENDING in the pool's, LONG in the long lane's.
- **A cancel** ends every job not ended, a claimed one too: no later end moves it (`cancel`).
- Timestamps are for reading (lesson 2): no worker's death or job's fate is decided by elapsed
  time; an attempt's start is read to raise a killed job's estimate to what it ran, and
  `ask_aside` counts the workers asked that retired after an allocation, as mycooc's does.
"""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import NamedTuple, Protocol

from pool_ladder import Ladder, class_name

FORMAT = 1      # the seed's (jobs.json), read exactly


class JobState(StrEnum):
    PENDING = "pending"         # waiting for a worker of its pool (node and class)
    CLAIMED = "claimed"         # an attempt of a worker's holds it
    DONE = "done"               # its command exited 0
    FAILED = "failed"           # the ladder's strikes made no progress: for a person (`retry`)
    LONG = "long"               # waiting for the long lane: a run of its own (`claim_job`)
    CANCELLED = "cancelled"     # will not run


class Lane(StrEnum):
    POOL = "pool"               # claimed by a pool's worker of its node and class
    LONG = "long"               # run by a run of its own, sized per job (the spec's 3.8)


class Outcome(StrEnum):
    DONE = "done"               # the command exited 0 (the worker's)
    FAILED = "failed"           # it exited otherwise (the worker's)
    OOM = "oom"                 # it was killed for memory (the worker's)
    REAPED = "reaped"           # its worker died before its walltime (the steward's)
    WALLTIME = "walltime"       # its worker was killed at its walltime (the steward's)
    RETIRED = "retired"         # its worker exited holding it (the worker's own `retire`)
    CANCELLED = "cancelled"


#: What a worker reports of its job's command (`finish`); the rest are the queue's.
FINISHED = (Outcome.DONE, Outcome.FAILED, Outcome.OOM)


def fresh_minutes(ladder: Ladder) -> float:
    """The minutes a fresh worker of `ladder` can give a job: its lease less the margin and its
    startup (a worker's first claim comes after its checkout, imports and opening the queue). A
    job estimated past them is the long lane's: no worker could claim it."""
    return ladder.lease_minutes - ladder.margin_minutes - ladder.startup_minutes


class FeedName(StrEnum):
    """The feed a wave's header records, which pool_steward.FEEDS resolves to its Feed."""
    FIXED = "fixed"             # every job granted at submit (pool_steward.Fixed): its seed
                                # closes the queue it makes
    SCREENED = "screened"       # decision 7's jobs granted as screens decide: not built (FEEDS
                                # lacks it, so pool.py refuses its steward); a queue of it open,
                                # as the guards' open waves are


@dataclass(frozen=True, kw_only=True)
class Header:
    """A wave's: its name, the commit every run of it is at, the family of its work (the
    estimators'), its ladder (recorded whole), when it was planned (cluster_submit's STAMP) and its
    feed (FeedName)."""

    wave: str
    commit: str
    family: str
    ladder: Ladder
    created: str
    feed: FeedName


@dataclass(frozen=True, kw_only=True)
class JobSeed:
    """A job as a wave grants it: one command (`line`) for one (block, item), in `group` (its
    block: a group's jobs run on one node), on `node`, of class `cls`; its estimate, needing `gb`
    (the basis `peak_gb` times a margin) and `minutes`, from `source`; and its lane."""

    id: int
    block: str
    item: str
    group: str
    line: str
    node: str
    cls: str
    gb: float
    minutes: float
    peak_gb: float
    source: str
    lane: Lane


@dataclass(frozen=True, kw_only=True)
class Seed:
    """A wave's jobs.json: its header, its jobs, and whether the queue is closed from the start
    (a fixed wave's is, and must be: `_open_fixed`). Its jobs' ids and (block, item)s are each
    unique, and a pool's job is of a class of the header's ladder and fits a fresh lease
    (`fresh_minutes`): one that did not would wait forever, no worker able to claim it, its count
    in the demand all the while."""

    header: Header
    jobs: tuple[JobSeed, ...]
    closed: bool

    def __post_init__(self) -> None:
        ids = [j.id for j in self.jobs]
        work = [(j.block, j.item) for j in self.jobs]
        if len(set(ids)) != len(ids) or len(set(work)) != len(work):
            raise ValueError(f"wave {self.header.wave}: a job id or (block, item) twice")
        if _open_fixed(self.header, closed=self.closed):
            raise ValueError(f"wave {self.header.wave}: its feed is {FeedName.FIXED} (every job "
                             "granted at submit) and its seed leaves the queue open: nothing "
                             "would ever close it, and its steward ends a wave only once its "
                             "queue is closed")
        lad = self.header.ladder
        classes = {class_name(lad.cards, r) for r in lad.rungs}
        odd = [j.id for j in self.jobs if j.lane == Lane.POOL and j.cls not in classes]
        if odd:
            raise ValueError(f"wave {self.header.wave}: pool jobs {odd[:5]} of no class of "
                             f"{lad.name} ({sorted(classes)})")
        slow = [j.id for j in self.jobs if j.lane == Lane.POOL and j.minutes > fresh_minutes(lad)]
        if slow:
            raise ValueError(f"wave {self.header.wave}: pool jobs {slow[:5]} estimated past a "
                             f"fresh lease's {fresh_minutes(lad):g} minutes (the long lane's)")


def _open_fixed(header: Header, *, closed: bool) -> bool:
    """Whether a seed of `header` that is `closed` or not is a fixed feed's left open: its feed
    grants nothing after submit, so it never closes the queue, and the wave would never end."""
    return header.feed == FeedName.FIXED and not closed


class Job(NamedTuple):
    """A job as the queue holds it: the seed's fields (its class, lane and estimates as they
    stand), its state, its strikes, and the worker whose claim holds it (None unless CLAIMED)."""

    id: int
    block: str
    item: str
    group: str
    line: str
    node: str
    cls: str
    gb: float
    minutes: float
    peak_gb: float
    source: str
    lane: Lane
    state: JobState
    strikes: int
    worker: str | None


class Claim(NamedTuple):
    job: Job            # as claimed
    attempt: int        # the attempt it is; `wrote` and `finish` name it


class Worker(NamedTuple):
    id: str             # its Slurm task
    run: str
    node: str
    cls: str
    ends_at: float      # its Slurm end time (epoch seconds)
    retired: bool


class Reaped(NamedTuple):
    """A reaped worker: the job it held, what its attempt came to (both None: it held none), and
    whether its walltime had ended (the reaper's `now` at or past its Slurm end time): what a
    lease's `more` reads (the spec's 3.7, mycooc's lesson 5)."""

    worker: str
    run: str
    job: int | None
    outcome: Outcome | None
    walltime: bool


class Queue(Protocol):
    def header(self) -> Header: ...
    def closed(self) -> bool: ...
    def close(self) -> None: ...
    def metric(self) -> str | None: ...
    def name_metric(self, name: str) -> None: ...
    def register(self, *, worker: str, run: str, node: str, cls: str, ends_at: float,
                 restarts: int) -> None: ...
    def retire(self, worker: str) -> int: ...
    def claim(self, *, worker: str, minutes_left: float) -> Claim | None: ...
    def claim_job(self, *, worker: str, job: int) -> Claim | None: ...
    def wrote(self, attempt: int) -> None: ...
    def kept(self, attempt: int) -> None: ...
    def finish(self, attempt: int, *, worker: str, outcome: Outcome, peak_gb: float,
               minutes: float, rc: int) -> JobState: ...
    def reap(self, worker: str, *, now: float) -> Reaped | None: ...
    def reaped(self) -> list[Reaped]: ...
    def ask_aside(self, *, node: str, cls: str, units: int, since: float) -> list[str]: ...
    def asked_aside(self, worker: str) -> bool: ...
    def retry(self, ids: Sequence[int]) -> int: ...
    def cancel(self) -> int: ...
    def jobs(self) -> list[Job]: ...
    def workers(self) -> list[Worker]: ...
    def backup(self, path: Path) -> None: ...


# ---- the seed's format: every key by name, exactly ------------------------------------------

class Obj:
    """A JSON object read exactly: each key by name, its type checked, and none left unread
    (`done`); every refusal a ValueError naming `where`. The pool's one reader of its JSON files
    (the seed here, pool_summary's summary.json, pool_steward's requests)."""

    def __init__(self, value: object, where: str) -> None:
        if not isinstance(value, dict):
            raise ValueError(f"{where}: not an object: {value!r}")
        self._d: dict[str, object] = value
        self._unread = set(value)
        self.where = where

    def _take(self, key: str) -> object:
        if key not in self._d:
            raise ValueError(f"{self.where}: no {key!r}")
        self._unread.discard(key)
        return self._d[key]

    def text(self, key: str) -> str:
        v = self._take(key)
        if not isinstance(v, str):
            raise ValueError(f"{self.where}: {key} {v!r} is not a string")
        return v

    def text_or_null(self, key: str) -> str | None:
        """A key whose value is a string or null (the key itself is required)."""
        v = self._take(key)
        if v is not None and not isinstance(v, str):
            raise ValueError(f"{self.where}: {key} {v!r} is neither a string nor null")
        return v

    def count(self, key: str) -> int:
        v = self._take(key)
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{self.where}: {key} {v!r} is not an integer")
        return v

    def number(self, key: str) -> float:
        v = self._take(key)
        if isinstance(v, bool) or not isinstance(v, int | float):
            raise ValueError(f"{self.where}: {key} {v!r} is not a number")
        return float(v)

    def flag(self, key: str) -> bool:
        v = self._take(key)
        if not isinstance(v, bool):
            raise ValueError(f"{self.where}: {key} {v!r} is not true or false")
        return v

    def items(self, key: str) -> list[object]:
        v = self._take(key)
        if not isinstance(v, list):
            raise ValueError(f"{self.where}: {key} {v!r} is not a list")
        return v

    def obj(self, key: str) -> Obj:
        return Obj(self._take(key), f"{self.where}.{key}")

    def done(self) -> None:
        if self._unread:
            raise ValueError(f"{self.where}: unknown keys {sorted(self._unread)}")


def _ladder_json(lad: Ladder) -> dict[str, object]:
    return {"name": lad.name, "cards": lad.cards, "cpus": lad.cpus, "rungs": list(lad.rungs),
            "lease_minutes": lad.lease_minutes, "margin_minutes": lad.margin_minutes,
            "startup_minutes": lad.startup_minutes,
            "strikes": lad.strikes, "idle_minutes": lad.idle_minutes,
            "poll_seconds": lad.poll_seconds, "stored_mem": lad.stored_mem,
            "fit_mem": lad.fit_mem, "stored_time": lad.stored_time, "about": lad.about}


def _ladder(o: Obj) -> Ladder:
    rungs: list[int] = []
    for r in o.items("rungs"):
        if isinstance(r, bool) or not isinstance(r, int):
            raise ValueError(f"{o.where}: a rung {r!r} is not an integer")
        rungs.append(r)
    lad = Ladder(name=o.text("name"), cards=o.count("cards"), cpus=o.count("cpus"),
                 rungs=tuple(rungs),
                 lease_minutes=o.number("lease_minutes"),
                 margin_minutes=o.number("margin_minutes"),
                 startup_minutes=o.number("startup_minutes"), strikes=o.count("strikes"),
                 idle_minutes=o.number("idle_minutes"), poll_seconds=o.number("poll_seconds"),
                 stored_mem=o.number("stored_mem"), fit_mem=o.number("fit_mem"),
                 stored_time=o.number("stored_time"), about=o.text("about"))
    o.done()
    return lad


def _header_json(h: Header) -> dict[str, object]:
    return {"wave": h.wave, "commit": h.commit, "family": h.family,
            "ladder": _ladder_json(h.ladder), "created": h.created, "feed": h.feed.value}


def _header(o: Obj) -> Header:
    h = Header(wave=o.text("wave"), commit=o.text("commit"), family=o.text("family"),
               ladder=_ladder(o.obj("ladder")), created=o.text("created"),
               feed=FeedName(o.text("feed")))
    o.done()
    return h


def _stored_header(text: str, where: str) -> Header:
    """The header as the queue's `header` table holds it (JSON, read exactly)."""
    return _header(Obj(json.loads(text), where))


def _job_json(j: JobSeed) -> dict[str, object]:
    return {"id": j.id, "block": j.block, "item": j.item, "group": j.group, "line": j.line,
            "node": j.node, "cls": j.cls, "gb": j.gb, "minutes": j.minutes,
            "peak_gb": j.peak_gb, "source": j.source, "lane": j.lane.value}


def _job_seed(o: Obj) -> JobSeed:
    lane = o.text("lane")
    if lane not in {x.value for x in Lane}:
        raise ValueError(f"{o.where}: lane {lane!r} is none of {[x.value for x in Lane]}")
    j = JobSeed(id=o.count("id"), block=o.text("block"), item=o.text("item"),
                group=o.text("group"), line=o.text("line"), node=o.text("node"),
                cls=o.text("cls"), gb=o.number("gb"), minutes=o.number("minutes"),
                peak_gb=o.number("peak_gb"), source=o.text("source"), lane=Lane(lane))
    o.done()
    return j


def seed_json(seed: Seed) -> str:
    """A wave's jobs.json: format 1, its header, its jobs and whether it is closed."""
    return json.dumps({"format": FORMAT, "header": _header_json(seed.header),
                       "jobs": [_job_json(j) for j in seed.jobs], "closed": seed.closed},
                      indent=1, allow_nan=False) + "\n"


def read_seed(text: str, where: str) -> Seed:
    """seed_json's output, read exactly (every key by name, of its type, none unknown); anything
    else is refused (ValueError naming `where`)."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where}: not JSON: {e}") from e
    o = Obj(value, where)
    fmt = o.count("format")
    if fmt != FORMAT:
        raise ValueError(f"{where}: format {fmt}, not {FORMAT}")
    seed = Seed(header=_header(o.obj("header")),
                jobs=tuple(_job_seed(Obj(j, f"{where}.jobs[{i}]"))
                           for i, j in enumerate(o.items("jobs"))),
                closed=o.flag("closed"))
    o.done()
    return seed


# ---- the transitions --------------------------------------------------------------------------

def _waiting(lane: Lane) -> JobState:
    """Where a job of `lane` waits between claims."""
    return JobState.PENDING if lane == Lane.POOL else JobState.LONG


def _mine(attempt: sqlite3.Row, worker: str) -> bool:
    """Whether `attempt` is `worker`'s: the claim it made (only its claimer finishes it)."""
    return bool(attempt["worker"] == worker)


def _holds(job: sqlite3.Row, attempt: int) -> bool:
    """Whether `attempt` is still `job`'s claim (not reaped, retired or taken over since)."""
    return bool(job["state"] == JobState.CLAIMED and job["attempt"] == attempt)


class Placed(NamedTuple):
    """Where an OOM leaves a job (`_promoted`)."""

    state: JobState
    lane: Lane
    cls: str
    gb: float
    peak_gb: float


def _promoted(ladder: Ladder, *, cls: str, lane: Lane, gb: float, peak_gb: float,
              peak: float) -> Placed:
    """A job of `cls` in `lane` killed for memory at `peak` GB: it needs `fit_mem` times that peak,
    its estimates never lowered; a pool's job moves a rung up, and past the top rung to the long
    lane; a long job stays long, its run to be sized by the new need."""
    need, basis = max(gb, ladder.fit_mem * peak), max(peak_gb, peak)
    if lane == Lane.POOL:
        rungs = {class_name(ladder.cards, r): r for r in ladder.rungs}
        if cls not in rungs:
            raise ValueError(f"class {cls} is no rung of {ladder.name} ({sorted(rungs)})")
        above = [r for r in ladder.rungs if r > rungs[cls]]
        if above:
            return Placed(state=JobState.PENDING, lane=Lane.POOL,
                          cls=class_name(ladder.cards, above[0]), gb=need, peak_gb=basis)
    return Placed(state=JobState.LONG, lane=Lane.LONG, cls=cls, gb=need, peak_gb=basis)


class Struck(NamedTuple):
    """Where a strike leaves a job (`_struck`)."""

    state: JobState
    strikes: int


def _struck(ladder: Ladder, *, lane: Lane, strikes: int) -> Struck:
    """A strike more: FAILED at the ladder's strikes, else waiting in its lane again."""
    n = strikes + 1
    return Struck(state=JobState.FAILED if n >= ladder.strikes else _waiting(lane), strikes=n)


def _reborn(had: int, restarts: int) -> bool:
    """Whether a task registered at restart count `had` registers again as itself requeued: Slurm
    keeps a requeued task's array id and counts its restarts (SLURM_RESTART_COUNT)."""
    return restarts > had


def _attempted_by(db: sqlite3.Connection, run: str) -> list[int]:
    """The jobs any worker registered in `run` (a lease) has attempted, by id."""
    return [r["job"] for r in db.execute(
        "SELECT DISTINCT a.job AS job FROM attempts a JOIN workers k ON k.id = a.worker"
        " WHERE k.run = ? ORDER BY a.job", (run,))]


def _close(db: sqlite3.Connection) -> None:
    """The queue closed, once (its time the first close's)."""
    if db.execute("SELECT 1 FROM closed").fetchone() is None:
        db.execute("INSERT INTO closed (at) VALUES (?)", (time.time(),))


def _one_claim(db: sqlite3.Connection, worker: str) -> None:
    """Refuse `worker` a claim while it holds one (ValueError naming it): a worker runs one job at
    a time, so whatever settles its claim (finish, retire, reap) settles one."""
    held = db.execute("SELECT id FROM jobs WHERE worker = ? AND state = ?",
                      (worker, JobState.CLAIMED.value)).fetchone()
    if held is not None:
        raise ValueError(f"worker {worker!r} holds job {held['id']} already: one claim at a time")


def _job(row: sqlite3.Row) -> Job:
    return Job(id=row["id"], block=row["block"], item=row["item"], group=row["grp"],
               line=row["line"], node=row["node"], cls=row["cls"], gb=row["gb"],
               minutes=row["minutes"], peak_gb=row["peak_gb"], source=row["source"],
               lane=Lane(row["lane"]), state=JobState(row["state"]), strikes=row["strikes"],
               worker=row["worker"])


def _inserted(cur: sqlite3.Cursor) -> int:
    if cur.lastrowid is None:
        raise RuntimeError("an INSERT gave no row id")
    return cur.lastrowid


# ---- SQLite ---------------------------------------------------------------------------------

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS header (one INTEGER PRIMARY KEY CHECK (one = 1),"
    " header TEXT NOT NULL, metric TEXT)",
    "CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY, block TEXT NOT NULL,"
    " item TEXT NOT NULL, grp TEXT NOT NULL, line TEXT NOT NULL, node TEXT NOT NULL,"
    " cls TEXT NOT NULL, gb REAL NOT NULL, minutes REAL NOT NULL, peak_gb REAL NOT NULL,"
    " source TEXT NOT NULL, lane TEXT NOT NULL, state TEXT NOT NULL, strikes INTEGER NOT NULL,"
    " worker TEXT, attempt INTEGER, UNIQUE (block, item))",
    "CREATE INDEX IF NOT EXISTS jobs_waiting ON jobs (node, cls, state, minutes)",
    "CREATE TABLE IF NOT EXISTS workers (id TEXT PRIMARY KEY, run TEXT NOT NULL,"
    " node TEXT NOT NULL, cls TEXT NOT NULL, ends_at REAL NOT NULL, registered REAL NOT NULL,"
    " retired REAL, restarts INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS reaped (id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " worker TEXT NOT NULL REFERENCES workers (id), run TEXT NOT NULL, job INTEGER,"
    " outcome TEXT, walltime INTEGER NOT NULL, at REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " job INTEGER NOT NULL REFERENCES jobs (id), worker TEXT NOT NULL REFERENCES workers (id),"
    " node TEXT NOT NULL, cls TEXT NOT NULL, start REAL NOT NULL, end REAL, outcome TEXT,"
    " peak_gb REAL, minutes REAL, rc INTEGER, rows INTEGER NOT NULL, kept INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS aside (worker TEXT PRIMARY KEY REFERENCES workers (id),"
    " asked_at REAL NOT NULL)",
    "CREATE TABLE IF NOT EXISTS closed (at REAL NOT NULL)",
)


@contextmanager
def _transaction(path: Path) -> Iterator[sqlite3.Connection]:
    """One operation: its own connection, in journal_mode DELETE, all in one BEGIN IMMEDIATE."""
    db = sqlite3.connect(path, timeout=120, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("BEGIN IMMEDIATE")
        try:
            yield db
        except BaseException:
            if db.in_transaction:      # else a failed ROLLBACK would mask the error
                db.execute("ROLLBACK")
            raise
        db.execute("COMMIT")
    finally:
        db.close()


def _grant(db: sqlite3.Connection, jobs: Sequence[JobSeed]) -> None:
    """Insert each of `jobs` not there yet, waiting in its lane; one there already is left as it
    stands, and one whose id or (block, item) is another job's there is refused."""
    have = {r["id"]: (r["block"], r["item"])
            for r in db.execute("SELECT id, block, item FROM jobs")}
    ids = {work: i for i, work in have.items()}
    for j in jobs:
        work = (j.block, j.item)
        if j.id in have and have[j.id] != work:
            raise ValueError(f"job {j.id} is {have[j.id]} in the queue, {work} in the seed")
        if work in ids and ids[work] != j.id:
            raise ValueError(f"{work} is job {ids[work]} in the queue, {j.id} in the seed")
        if j.id in have:
            continue
        db.execute("INSERT INTO jobs (id, block, item, grp, line, node, cls, gb, minutes,"
                   " peak_gb, source, lane, state, strikes) VALUES"
                   " (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                   (j.id, j.block, j.item, j.group, j.line, j.node, j.cls, j.gb, j.minutes,
                    j.peak_gb, j.source, j.lane.value, _waiting(j.lane).value))


class SqliteQueue:
    """The Queue in one SQLite file (the module's doc). Made by `open`."""

    def __init__(self, path: Path, header: Header) -> None:
        self.path = path
        self._header = header

    @classmethod
    def open(cls, path: Path, seed: Seed) -> SqliteQueue:
        """The queue at `path`, made if absent: its tables, its header, and the seed's jobs
        granted once (unique on block and item); closed when the seed is. Any process of a wave
        may be the first (the plan's Ruling 7). A seed of another wave or commit than the header
        there is refused (ValueError)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with _transaction(path) as db:
            for statement in _SCHEMA:
                db.execute(statement)
            row = db.execute("SELECT header FROM header").fetchone()
            if row is None:
                db.execute("INSERT INTO header (one, header) VALUES (1, ?)",
                           (json.dumps(_header_json(seed.header), allow_nan=False),))
                header = seed.header
            else:
                header = _stored_header(row["header"], f"{path}: header")
                if (header.wave, header.commit) != (seed.header.wave, seed.header.commit):
                    raise ValueError(f"{path} is wave {header.wave} at {header.commit}, the "
                                     f"seed {seed.header.wave} at {seed.header.commit}")
            _grant(db, seed.jobs)
            if seed.closed and db.execute("SELECT 1 FROM closed").fetchone() is None:
                db.execute("INSERT INTO closed (at) VALUES (?)", (time.time(),))
        return cls(path, header)

    def _tx(self) -> AbstractContextManager[sqlite3.Connection]:
        return _transaction(self.path)

    # -- the header ----------------------------------------------------------------------------

    def header(self) -> Header:
        """The header the queue was made with (written once)."""
        return self._header

    def closed(self) -> bool:
        """Whether the wave grants nothing more: a worker with nothing to claim then exits."""
        with self._tx() as db:
            return db.execute("SELECT 1 FROM closed").fetchone() is not None

    def close(self) -> None:
        with self._tx() as db:
            _close(db)

    def metric(self) -> str | None:
        """The name of the wave's metric, once named."""
        with self._tx() as db:
            name: str | None = db.execute("SELECT metric FROM header").fetchone()["metric"]
            return name

    def name_metric(self, name: str) -> None:
        """Name the wave's metric: once; the same name again is a no-op, another refused."""
        with self._tx() as db:
            had = db.execute("SELECT metric FROM header").fetchone()["metric"]
            if had is not None and had != name:
                raise ValueError(f"wave {self._header.wave}'s metric is {had}, not {name}")
            db.execute("UPDATE header SET metric = ?", (name,))

    # -- workers -------------------------------------------------------------------------------

    def register(self, *, worker: str, run: str, node: str, cls: str, ends_at: float,
                 restarts: int) -> None:
        """`worker` (its Slurm task) of `run` starts on `node` in class `cls`, its walltime
        ending at `ends_at`, Slurm having restarted it `restarts` times. A worker registers once
        a restart: registered again at a higher count, it is its task requeued (which keeps its
        array id), and its earlier self, if it never retired, is reaped first as the steward
        would reap it (the job it held settled, the reap recorded); then it is registered afresh,
        its end time the new one, an ask to step aside kept. Registered again at its count or a
        lower one, or of another run, node or class, it is refused (ValueError)."""
        with self._tx() as db:
            now = time.time()
            w = db.execute("SELECT * FROM workers WHERE id = ?", (worker,)).fetchone()
            if w is None:
                db.execute("INSERT INTO workers (id, run, node, cls, ends_at, registered, restarts)"
                           " VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (worker, run, node, cls, ends_at, now, restarts))
                return
            if not _reborn(w["restarts"], restarts):
                raise ValueError(f"worker {worker!r} has registered already (restart "
                                 f"{w['restarts']}, now {restarts})")
            if (w["run"], w["node"], w["cls"]) != (run, node, cls):
                raise ValueError(f"worker {worker!r} of {w['run']} on {w['node']} in {w['cls']} "
                                 f"registers again of {run} on {node} in {cls}")
            if w["retired"] is None:
                self._settle(db, w, now=now)
            db.execute("UPDATE workers SET ends_at = ?, registered = ?, retired = NULL,"
                       " restarts = ? WHERE id = ?", (ends_at, now, restarts, worker))

    @staticmethod
    def _worker_row(db: sqlite3.Connection, worker: str) -> sqlite3.Row:
        row: sqlite3.Row | None = db.execute("SELECT * FROM workers WHERE id = ?",
                                             (worker,)).fetchone()
        if row is None:
            raise ValueError(f"worker {worker!r} has not registered")
        return row

    @staticmethod
    def _held(db: sqlite3.Connection, worker: str) -> sqlite3.Row | None:
        """The job `worker`'s claim holds, if any: one at most, as `_one_claim` keeps it where
        claims are made (more is that invariant broken: RuntimeError)."""
        rows = db.execute("SELECT * FROM jobs WHERE worker = ? AND state = ?",
                          (worker, JobState.CLAIMED.value)).fetchall()
        if len(rows) > 1:
            raise RuntimeError(f"worker {worker!r} holds jobs {[r['id'] for r in rows]}: one "
                               "claim at a time is broken")
        return rows[0] if rows else None

    def retire(self, worker: str) -> int:
        """`worker` has stopped: it claims nothing more, and the job it held goes back to wait in
        its lane, as if never claimed, its attempt RETIRED: 1 if it held one, else 0 (and 0
        again)."""
        with self._tx() as db:
            self._worker_row(db, worker)
            now = time.time()
            db.execute("UPDATE workers SET retired = COALESCE(retired, ?) WHERE id = ?",
                       (now, worker))
            job = self._held(db, worker)
            if job is None:
                return 0
            db.execute("UPDATE attempts SET end = ?, outcome = ? WHERE id = ?",
                       (now, Outcome.RETIRED.value, job["attempt"]))
            db.execute("UPDATE jobs SET state = ?, worker = NULL, attempt = NULL WHERE id = ?",
                       (_waiting(Lane(job["lane"])).value, job["id"]))
            return 1

    def reap(self, worker: str, *, now: float) -> Reaped | None:
        """`worker` is gone (squeue no longer lists it: the steward's evidence): retired, the job
        it held settled, and the reap recorded (`reaped`). With `now` at or past its Slurm end
        time it was killed at its walltime: no strike, the job's time estimate raised to what the
        attempt ran, and past `fresh_minutes` the job goes long. Before it, it died: a strike if
        the attempt wrote no row. A worker retired already (its own exit came first) is left as
        it is, and nothing is reaped (None)."""
        with self._tx() as db:
            w = self._worker_row(db, worker)
            if w["retired"] is not None:
                return None
            return self._settle(db, w, now=now)

    def _settle(self, db: sqlite3.Connection, w: sqlite3.Row, *, now: float) -> Reaped:
        """Reap `w` (a worker not retired) at `now`, inside the caller's transaction (`reap`'s
        doc), and record it."""
        worker = w["id"]
        db.execute("UPDATE workers SET retired = ? WHERE id = ?", (time.time(), worker))
        walltime = now >= w["ends_at"]
        job = self._held(db, worker)
        reaped = Reaped(worker=worker, run=w["run"], job=None, outcome=None, walltime=walltime)
        if job is not None:
            a = db.execute("SELECT * FROM attempts WHERE id = ?", (job["attempt"],)).fetchone()
            lane, strikes, minutes = Lane(job["lane"]), job["strikes"], job["minutes"]
            ladder = self._header.ladder
            if walltime:
                outcome, end = Outcome.WALLTIME, w["ends_at"]
                ran = (end - a["start"]) / 60
                minutes = max(minutes, ran)
                if minutes > fresh_minutes(ladder):
                    lane = Lane.LONG
                state = _waiting(lane)
            else:
                outcome, ran, end = Outcome.REAPED, None, time.time()
                state = _waiting(lane)
                if not a["rows"]:
                    struck = _struck(ladder, lane=lane, strikes=strikes)
                    state, strikes = struck.state, struck.strikes
            db.execute("UPDATE attempts SET end = ?, outcome = ?, minutes = ? WHERE id = ?",
                       (end, outcome.value, ran, a["id"]))
            db.execute("UPDATE jobs SET state = ?, lane = ?, minutes = ?, strikes = ?,"
                       " worker = NULL, attempt = NULL WHERE id = ?",
                       (state.value, lane.value, minutes, strikes, job["id"]))
            reaped = reaped._replace(job=job["id"], outcome=outcome)
        db.execute("INSERT INTO reaped (worker, run, job, outcome, walltime, at)"
                   " VALUES (?, ?, ?, ?, ?, ?)",
                   (worker, reaped.run, reaped.job,
                    None if reaped.outcome is None else reaped.outcome.value, int(walltime),
                    time.time()))
        return reaped

    def reaped(self) -> list[Reaped]:
        """Every reap of the wave (the steward's, and a requeued task's earlier self's), in
        order: what its summary lists."""
        with self._tx() as db:
            return [Reaped(worker=r["worker"], run=r["run"], job=r["job"],
                           outcome=None if r["outcome"] is None else Outcome(r["outcome"]),
                           walltime=bool(r["walltime"]))
                    for r in db.execute("SELECT * FROM reaped ORDER BY id")]

    # -- claims --------------------------------------------------------------------------------

    @classmethod
    def _live(cls, db: sqlite3.Connection, worker: str) -> sqlite3.Row:
        w = cls._worker_row(db, worker)
        if w["retired"] is not None:
            raise ValueError(f"worker {worker!r} has retired")
        return w

    @staticmethod
    def _pick(db: sqlite3.Connection, w: sqlite3.Row, *, minutes_left: float) -> sqlite3.Row | None:
        """The PENDING job of `w`'s node and class that no worker of `w`'s run has attempted
        (`_attempted_by`), with the longest estimate fitting `minutes_left` (then the lowest
        id)."""
        ex = _attempted_by(db, w["run"])
        row: sqlite3.Row | None = db.execute(
            "SELECT * FROM jobs WHERE node = ? AND cls = ? AND state = ? AND minutes <= ?"
            f" AND id NOT IN ({', '.join('?' * len(ex))}) ORDER BY minutes DESC, id LIMIT 1",
            (w["node"], w["cls"], JobState.PENDING.value, minutes_left, *ex)).fetchone()
        return row

    @staticmethod
    def _take(db: sqlite3.Connection, w: sqlite3.Row, job: sqlite3.Row) -> Claim:
        """`job` becomes `w`'s, with a new attempt."""
        attempt = _inserted(db.execute(
            "INSERT INTO attempts (job, worker, node, cls, start, rows, kept)"
            " VALUES (?, ?, ?, ?, ?, 0, 0)",
            (job["id"], w["id"], w["node"], w["cls"], time.time())))
        db.execute("UPDATE jobs SET state = ?, worker = ?, attempt = ? WHERE id = ?",
                   (JobState.CLAIMED.value, w["id"], attempt, job["id"]))
        return Claim(job=_job(db.execute("SELECT * FROM jobs WHERE id = ?",
                                         (job["id"],)).fetchone()), attempt=attempt)

    def claim(self, *, worker: str, minutes_left: float) -> Claim | None:
        """The next job for `worker` (registered, not retired and holding no claim; else
        ValueError): of its node and class, PENDING, attempted by no worker of its run (its
        lease), estimated within `minutes_left` (its walltime left less the margin), the longest
        first; None if none is. The exclusion is read in the claim's one transaction."""
        with self._tx() as db:
            w = self._live(db, worker)
            _one_claim(db, worker)
            job = self._pick(db, w, minutes_left=minutes_left)
            return None if job is None else self._take(db, w, job)

    def claim_job(self, *, worker: str, job: int) -> Claim | None:
        """Job `job` for `worker` (the long lane's run of it), if it is waiting (LONG, or a
        pool's PENDING); None if it is not. A job of another node, or a worker holding a claim,
        is refused (ValueError)."""
        with self._tx() as db:
            w = self._live(db, worker)
            _one_claim(db, worker)
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job,)).fetchone()
            if row is None:
                raise ValueError(f"no job {job}")
            if row["node"] != w["node"]:
                raise ValueError(f"job {job} is {row['node']}'s, worker {worker!r} on {w['node']}")
            if row["state"] != _waiting(Lane(row["lane"])):
                return None
            return self._take(db, w, row)

    def wrote(self, attempt: int) -> None:
        """`attempt`'s job wrote a row (its worker counts clearings' ROW_WRITTEN lines)."""
        with self._tx() as db:
            if not db.execute("UPDATE attempts SET rows = rows + 1 WHERE id = ?",
                              (attempt,)).rowcount:
                raise ValueError(f"no attempt {attempt}")

    def kept(self, attempt: int) -> None:
        """`attempt`'s job found a row standing and kept it (its worker counts clearings'
        row_marker.KEPT lines): it ran less than its whole work, so it measures no job
        (pool_plan.history_of)."""
        with self._tx() as db:
            if not db.execute("UPDATE attempts SET kept = kept + 1 WHERE id = ?",
                              (attempt,)).rowcount:
                raise ValueError(f"no attempt {attempt}")

    def finish(self, attempt: int, *, worker: str, outcome: Outcome, peak_gb: float,
               minutes: float, rc: int) -> JobState:
        """`worker` has run `attempt`'s command: DONE (exit 0), FAILED or OOM, at `peak_gb`, in
        `minutes`, exiting `rc`. The attempt is recorded; the job moves only while the attempt
        is still its claim (else it stays as the reaper or a later claim left it): DONE; FAILED
        back to wait if a row was written, else a strike; OOM promoted (`_promoted`). The job's
        state after."""
        if outcome not in FINISHED:
            raise ValueError(f"a worker finishes a job {', '.join(FINISHED)}, not {outcome}")
        with self._tx() as db:
            a = db.execute("SELECT * FROM attempts WHERE id = ?", (attempt,)).fetchone()
            if a is None or not _mine(a, worker):
                raise ValueError(f"attempt {attempt} is not {worker!r}'s"
                                 + ("" if a is None else f" but {a['worker']!r}'s"))
            if a["rc"] is not None:
                raise ValueError(f"attempt {attempt} has finished already")
            db.execute("UPDATE attempts SET end = ?, outcome = COALESCE(outcome, ?), peak_gb = ?,"
                       " minutes = ?, rc = ? WHERE id = ?",
                       (time.time(), outcome.value, peak_gb, minutes, rc, attempt))
            job = db.execute("SELECT * FROM jobs WHERE id = ?", (a["job"],)).fetchone()
            if not _holds(job, attempt):
                return JobState(job["state"])
            ladder, lane = self._header.ladder, Lane(job["lane"])
            placed = Placed(state=JobState.DONE, lane=lane, cls=job["cls"], gb=job["gb"],
                            peak_gb=job["peak_gb"])
            strikes = job["strikes"]
            if outcome == Outcome.OOM:
                placed = _promoted(ladder, cls=job["cls"], lane=lane, gb=job["gb"],
                                   peak_gb=job["peak_gb"], peak=peak_gb)
            elif outcome == Outcome.FAILED:
                state = _waiting(lane)
                if not a["rows"]:
                    struck = _struck(ladder, lane=lane, strikes=strikes)
                    state, strikes = struck.state, struck.strikes
                placed = placed._replace(state=state)
            db.execute("UPDATE jobs SET state = ?, lane = ?, cls = ?, gb = ?, peak_gb = ?,"
                       " strikes = ?, worker = NULL, attempt = NULL WHERE id = ?",
                       (placed.state.value, placed.lane.value, placed.cls, placed.gb,
                        placed.peak_gb, strikes, job["id"]))
            return placed.state

    # -- stepping aside ------------------------------------------------------------------------

    def ask_aside(self, *, node: str, cls: str, units: int, since: float) -> list[str]:
        """Ask `node`'s live workers of `cls` to step aside until `units` of them have been,
        counting those asked already that still run or retired after `since` (the time of the
        allocation that owes `units`: one it made before they retired counted them as held, one
        made after did not). Idle workers first (a bundle freed at once), then those whose job
        is expected to end soonest (its attempt's start plus its estimate). The workers newly
        asked (mycooc's, restricted to the class)."""
        with self._tx() as db:
            asked = db.execute(
                "SELECT COUNT(*) FROM aside a JOIN workers w ON w.id = a.worker WHERE w.node = ?"
                " AND w.cls = ? AND (w.retired IS NULL OR w.retired > ?)",
                (node, cls, since)).fetchone()[0]
            if units <= asked:
                return []
            rows = db.execute(
                "SELECT w.id AS id, t.start + 60 * j.minutes AS ends FROM workers w"
                " LEFT JOIN jobs j ON j.worker = w.id AND j.state = ?"
                " LEFT JOIN attempts t ON t.id = j.attempt"
                " WHERE w.node = ? AND w.cls = ? AND w.retired IS NULL"
                " AND w.id NOT IN (SELECT worker FROM aside)",
                (JobState.CLAIMED.value, node, cls)).fetchall()
            chosen: list[str] = [r["id"] for r in sorted(rows, key=lambda r: (
                r["ends"] is not None, 0.0 if r["ends"] is None else r["ends"], r["id"]))
            ][:units - asked]
            now = time.time()
            db.executemany("INSERT INTO aside (worker, asked_at) VALUES (?, ?)",
                           [(w, now) for w in chosen])
            return chosen

    def asked_aside(self, worker: str) -> bool:
        """Whether `worker` has been asked to step aside (ask_aside)."""
        with self._tx() as db:
            return db.execute("SELECT 1 FROM aside WHERE worker = ?",
                              (worker,)).fetchone() is not None

    # -- a person's requests, through the steward ----------------------------------------------

    def retry(self, ids: Sequence[int]) -> int:
        """Each FAILED job of `ids` back to wait in its lane, its strikes 0 (their count); a job
        of them not FAILED (done or claimed since the request) is left; an id of no job is
        refused (ValueError)."""
        with self._tx() as db:
            n = 0
            for i in ids:
                row = db.execute("SELECT state, lane FROM jobs WHERE id = ?", (i,)).fetchone()
                if row is None:
                    raise ValueError(f"no job {i}")
                if row["state"] == JobState.FAILED:
                    db.execute("UPDATE jobs SET state = ?, strikes = 0 WHERE id = ?",
                               (_waiting(Lane(row["lane"])).value, i))
                    n += 1
            return n

    def cancel(self) -> int:
        """Every job not ended (PENDING, LONG or CLAIMED) CANCELLED (their count): a claimed one's
        claim released and its attempt's outcome CANCELLED, ended now. No later end moves it
        again: its worker's finish moves a job only while the attempt is still its claim (the
        attempt records the code, peak and minutes under its CANCELLED), and a retire or reap
        settles only a CLAIMED job. So the wave ends at once, its jobs as the cancel left them
        (the plan's Ruling 12; a job that kept running writes rows that stand, as any does). The
        queue is closed too (the spec's 3.9): a cancelled wave grants nothing more, so an open
        one's feed never adds a job to it, and it ends."""
        with self._tx() as db:
            _close(db)
            claimed = JobState.CLAIMED.value
            db.execute("UPDATE attempts SET end = ?, outcome = ? WHERE id IN"
                       " (SELECT attempt FROM jobs WHERE state = ?)",
                       (time.time(), Outcome.CANCELLED.value, claimed))
            return db.execute("UPDATE jobs SET state = ?, worker = NULL, attempt = NULL"
                              " WHERE state IN (?, ?, ?)",
                              (JobState.CANCELLED.value, JobState.PENDING.value,
                               JobState.LONG.value, claimed)).rowcount

    # -- reading -------------------------------------------------------------------------------

    def jobs(self) -> list[Job]:
        """Every job, by id."""
        with self._tx() as db:
            return [_job(r) for r in db.execute("SELECT * FROM jobs ORDER BY id")]

    def workers(self) -> list[Worker]:
        """Every worker registered, in the order they registered."""
        with self._tx() as db:
            return [Worker(id=r["id"], run=r["run"], node=r["node"], cls=r["cls"],
                           ends_at=r["ends_at"], retired=r["retired"] is not None)
                    for r in db.execute("SELECT * FROM workers ORDER BY registered, id")]

    def backup(self, path: Path) -> None:
        """A snapshot of the queue at `path` (SQLite's backup, the wave's final.db), written to
        a temporary file beside it and renamed over it. The one operation outside BEGIN
        IMMEDIATE: SQLite's backup takes its own read lock, every page in one step (a consistent
        snapshot), and refuses a source connection that holds a write transaction (LOCKED,
        which Python's backup retries forever)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        os.close(fd)
        try:
            db = sqlite3.connect(self.path, timeout=120, isolation_level=None)
            try:
                db.execute("PRAGMA journal_mode=DELETE")
                target = sqlite3.connect(tmp)
                try:
                    db.backup(target, pages=-1)
                finally:
                    target.close()
            finally:
                db.close()
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


def read_header(path: Path) -> Header:
    """The header of the queue at `path`, or of a snapshot of one (a wave's final.db), read
    exactly; read-only, one SELECT (so a path with no queue is refused, never made one: sqlite3's
    OperationalError, and a queue with no header is a ValueError)."""
    db = sqlite3.connect(path.absolute().as_uri() + "?mode=ro", uri=True)
    try:
        row = db.execute("SELECT header FROM header").fetchone()
    finally:
        db.close()
    if row is None:
        raise ValueError(f"{path}: no header")
    return _stored_header(row[0], f"{path}: header")
