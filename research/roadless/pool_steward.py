"""A wave's steward (docs/superpowers/specs/2026-10-21-lease-pool-design.md, 3.6, with the plan's
Rulings 1, 5, 6 and 12), after GeoffChurch/mycooc's executor (its scheduler/liveness.py, and its
gpu_demand's step-aside): one CPU lease per wave that runs a round every poll (`round_`):

1. **Requests**: each `<wave>/requests/*.json`, in name order, read exactly (`{"retry": [<job
   id>, ...]}` or `{"cancel": true}`), applied, deleted and recorded in the ledger; one that does
   not read, or that the queue refuses (a job it lacks), deleted and recorded as refused.
2. **Reap** on squeue's evidence alone (`Liveness`): a registered worker that squeue no longer
   lists, in an answer listing the steward's own task, has ended, and is reaped (pool_queue's
   `reap`: its job back as if never claimed, a strike if its attempt wrote no row, Ruling 6;
   past its walltime no strike, the job's estimate raised). Never on elapsed time (mycooc's
   lesson 2); a squeue that fails, or answers without the steward's own task, reaps nobody (lesson
   15: an unreachable scheduler is no verdict).
3. **Aside** (Ruling 5): the market's allocation through its mode (shadow asks nobody), unless
   older than the market's stale_after; for each (node, bundle) the campaign yields there that is
   a class of this wave's ladder, its share of those units among the live waves' workers there
   (pool_summary.aside_share: the other waves counted by their latest summaries, one that does
   not read left out and said in the ledger), asked of its own live workers (pool_queue's
   `ask_aside`: idle ones first, counting its own asks since that allocation).
4. **Feed** (`Feed`): `Fixed`, nothing after the seed (decision 7; `Screened` waits for the
   re-run).
5. **summary.json** written whole (a temporary name, then a rename), then the round's line
   appended to `ledger.jsonl`.

`steward` runs rounds until two no longer fit before its lease's end (75: its continuation is its
own rule's), or every job is done, failed or cancelled: then the queue's final snapshot
(`final.db`), the bank and tars deleted from the wave's inputs, a last summary (FINISHED, or
CANCELLED after a cancel), 0.

It writes no demand (Ruling 1): this machine's demand step states the one campaign's, reading
each live wave's summary.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import pool_summary as psum
from cluster_submit import Bundle
from cluster_submit.context import STAMP, stamped
from cluster_submit.gpu import ALLOCATION, GPU, Allocation, mode_named, read_allocation
from pool_ladder import class_name
from pool_queue import Job, JobState, Obj, Queue, Reaped
from pool_worker import Exit

#: The one campaign, as the market's allocation names it ("<project>/<campaign>": Ruling 1).
CAMPAIGN = "reblock/reblock"
#: A wave's files, in its directory ($HOME/reblock-waves/<w>/).
SUMMARY = "summary.json"
LEDGER = "ledger.jsonl"
FINAL = "final.db"
REQUESTS = "requests"
INPUTS = "inputs"
BANK = "bank.pkl"           # the wave's bank, deleted at its end
TARS = "*.tar"              # --unpack's tars, deleted at its end
#: How long a round waits for squeue: it answers in seconds, and a round must not wait on a
#: wedged controller (mycooc's). A timeout is no answer: nobody reaped.
SQUEUE_TIMEOUT = 60.0
#: The states in which a job will not run again (but by a person's `retry`).
ENDED = (JobState.DONE, JobState.FAILED, JobState.CANCELLED)


# ---- liveness ---------------------------------------------------------------------------------

class Liveness(Protocol):
    def dead(self, workers: Sequence[str]) -> frozenset[str]:
        """Those of `workers` (Slurm tasks) known to have ended; none when it cannot tell."""
        ...


def run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """`argv`, its output captured, within SQUEUE_TIMEOUT (subprocess.TimeoutExpired past it)."""
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=SQUEUE_TIMEOUT,
                          check=False)


def dead_in(listed: str, workers: Sequence[str], *, own: str) -> frozenset[str]:
    """The `workers` squeue's listing (a task a line) does not name; none when it does not name
    `own`, the asker's own task (a confused controller's empty answer would call every worker
    dead)."""
    alive = set(listed.split())
    if own not in alive:
        return frozenset()
    return frozenset(w for w in workers if w not in alive)


@dataclass(frozen=True, kw_only=True)
class Squeue:
    """mycooc's SlurmWorkers: asks the controller (`run`, the command's runner) for `user`'s
    pending and running tasks (`squeue -h -r -u <user> -o %i`); `own` is the steward's own task,
    which a believed answer lists. squeue rather than sacct: on ai, sacct on a compute node cannot
    reach the accounting database (mycooc, 2026-10-15)."""

    run: Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
    user: str
    own: str

    def dead(self, workers: Sequence[str]) -> frozenset[str]:
        if not workers:
            return frozenset()
        try:
            out = self.run(["squeue", "-h", "-r", "-u", self.user, "-o", "%i"])
        except (OSError, subprocess.SubprocessError):
            return frozenset()
        if out.returncode != 0:
            return frozenset()
        return dead_in(out.stdout, workers, own=self.own)


# ---- the feed ---------------------------------------------------------------------------------

class Feed(Protocol):
    def round(self, queue: Queue) -> None:
        """Grant the round's jobs (and close the queue once it grants no more)."""
        ...


@dataclass(frozen=True)
class Fixed:
    """The jobs granted at submit, nothing after: the seed closed the queue when it made it."""

    def round(self, queue: Queue) -> None:
        return None


#: The feeds, by the name the queue's header records (`Header.feed`): resolved once, no default.
FEEDS: dict[str, Callable[[], Feed]] = {"fixed": Fixed}


# ---- requests ---------------------------------------------------------------------------------

class Request(Protocol):
    def apply(self, queue: Queue) -> dict[str, object]:
        """Apply it to `queue`: what it did, for the ledger."""
        ...


@dataclass(frozen=True, kw_only=True)
class Retry:
    """`wave <w> retry`: these FAILED jobs back to wait, their strikes 0 (pool_queue's `retry`)."""

    ids: tuple[int, ...]

    def apply(self, queue: Queue) -> dict[str, object]:
        return {"retry": list(self.ids), "retried": queue.retry(self.ids)}


@dataclass(frozen=True)
class Cancel:
    """`wave <w> cancel`: every job not ended CANCELLED (pool_queue's `cancel`; Ruling 12)."""

    def apply(self, queue: Queue) -> dict[str, object]:
        return {"cancel": True, "cancelled": queue.cancel()}


def read_request(text: str, where: str) -> Request:
    """A request file's text, read exactly: `{"retry": [<job id>, ...]}` or `{"cancel": true}`;
    anything else a ValueError naming `where`."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where}: not JSON: {e}") from e
    o = Obj(value, where)
    req: Request
    if "retry" in value:
        ids: list[int] = []
        for v in o.items("retry"):
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(f"{where}: retry: {v!r} is not a job id")
            ids.append(v)
        req = Retry(ids=tuple(ids))
    else:
        if not o.flag("cancel"):
            raise ValueError(f"{where}: cancel false asks nothing")
        req = Cancel()
    o.done()
    return req


def _requests(queue: Queue, requests: Path, *, log: Callable[[str], None]
              ) -> list[dict[str, object]]:
    """Each request in `requests`, in name order: applied (or refused), deleted, and what came of
    it (the ledger's)."""
    done: list[dict[str, object]] = []
    for path in sorted(requests.glob("*.json")):
        what: dict[str, object]
        try:
            what = read_request(path.read_text(), str(path)).apply(queue)
        except ValueError as e:
            what = {"refused": str(e)}
        path.unlink()
        log(f"request {path.name}: {what}")
        done.append({"request": path.name, **what})
    return done


# ---- a round ----------------------------------------------------------------------------------

def _reap(queue: Queue, liveness: Liveness, *, now: float) -> list[Reaped]:
    """Every live worker `liveness` says has ended, reaped at `now` (epoch seconds), in the order
    they registered. The workers are read before squeue is asked, so each was running when it
    answered: one it does not list has ended. One that retired on its own meanwhile is no reap."""
    live = [w.id for w in queue.workers() if not w.retired]
    gone = liveness.dead(live)
    out: list[Reaped] = []
    for w in live:
        if w in gone:
            r = queue.reap(w, now=now)
            if r is not None:
                out.append(r)
    return out


def _live_summaries(waves: Path, wave_dir: Path) -> tuple[list[psum.Summary], dict[str, str]]:
    """The other waves' latest summaries under `waves` whose stewards still run (OPEN or CLOSED),
    and each summary that does not read here, with why."""
    found: list[psum.Summary] = []
    refused: dict[str, str] = {}
    own = wave_dir.resolve()
    for path in sorted(waves.glob(f"*/{SUMMARY}")):
        if path.parent.resolve() == own:
            continue
        try:
            s = psum.read_summary(path.read_text(), str(path))
        except (OSError, ValueError) as e:
            refused[str(path)] = str(e)
            continue
        if s.state in psum.LIVE:
            found.append(s)
    return found, refused


def _workers_at(s: psum.Summary, node: str, bundle: Bundle) -> int:
    """The live workers `s` counts on `node` holding `bundle`."""
    return sum(line.workers for line in s.lines if line.node == node
               and Bundle(cards=line.cards, cpus=line.cpus, mem_gb=line.mem_gb) == bundle)


def _aside(queue: Queue, *, wave_dir: Path, waves: Path,
           allocation: Callable[[], Allocation | None], stale_after: timedelta,
           now: datetime) -> dict[str, object]:
    """Ask the wave's share of the campaign's yields of its workers (the module's doc, 3): what it
    did, for the ledger."""
    a = allocation()
    if a is None:
        return {"none": "no allocation read from the board"}
    at = stamped(a.at)
    if now - at > stale_after:
        return {"none": f"the allocation of {a.at} is older than {stale_after}"}
    mode = mode_named(a.mode)
    if mode is None:            # read_allocation refuses any other
        raise ValueError(f"the allocation of {a.at}: mode {a.mode!r} is none of cluster_submit's")
    owed = mode.aside(lambda: a.yields(CAMPAIGN))
    h = queue.header()
    lad = h.ladder
    classes = {lad.bundle(r): class_name(lad.cards, r) for r in lad.rungs}
    mine = [(node, b, units) for node, held in sorted(owed.items())
            for b, units in held.items() if b in classes]
    if not mine:
        return {"mode": a.mode, "at": a.at, "asked": {}}
    others, refused = _live_summaries(waves, wave_dir)
    live = [w for w in queue.workers() if not w.retired]
    asked: dict[str, object] = {}
    for node, b, units in mine:
        cls = classes[b]
        counts = {s.wave: _workers_at(s, node, b) for s in others}
        counts[h.wave] = sum(1 for w in live if (w.node, w.cls) == (node, cls))
        share = psum.aside_share(yields=units, workers=counts, wave=h.wave)
        asked[f"{node} {cls}"] = {"yields": units, "workers": counts, "share": share,
                                  "asked": queue.ask_aside(node=node, cls=cls, units=share,
                                                           since=at.timestamp())}
    return {"mode": a.mode, "at": a.at, "asked": asked, "refused": refused}


def _summary(queue: Queue, *, round_no: int, at: datetime,
             state: psum.WaveState) -> psum.Summary:
    """The wave's summary now: `state`, at `at`, of round `round_no`."""
    h = queue.header()
    jobs = queue.jobs()
    return psum.Summary(
        wave=h.wave, commit=h.commit, at=at.strftime(STAMP), round=round_no, state=state,
        metric=queue.metric(), lines=psum.lines_of(jobs, queue.workers(), h.ladder),
        reaped=tuple(psum.ReapedWorker(task=r.worker, run=r.run, held_job=r.job is not None,
                                       walltime=r.walltime) for r in queue.reaped()),
        long=psum.long_jobs(jobs))


def _write_whole(path: Path, text: str) -> None:
    """`text` at `path` whole: written to a temporary name beside it, then renamed over it."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _append(path: Path, line: dict[str, object]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(line, allow_nan=False) + "\n")


def round_(*, queue: Queue, wave_dir: Path, waves: Path, liveness: Liveness, feed: Feed,
           allocation: Callable[[], Allocation | None], stale_after: timedelta,
           now: Callable[[], datetime], log: Callable[[str], None]) -> psum.Summary:
    """One round of the wave in `wave_dir` (its queue `queue`; every wave's directory under
    `waves`): requests, reap, aside, feed, summary, the ledger's line (the module's doc). Its
    number follows the wave's last summary's, whichever lease of the steward wrote it."""
    t = now()
    path = wave_dir / SUMMARY
    n = 1 + (psum.read_summary(path.read_text(), str(path)).round if path.exists() else 0)
    requests = _requests(queue, wave_dir / REQUESTS, log=log)
    reaped = _reap(queue, liveness, now=t.timestamp())
    for r in reaped:
        log(f"reaped {r.worker} of {r.run}: "
            + ("idle" if r.job is None else f"job {r.job} {r.outcome}")
            + (", past its walltime" if r.walltime else ""))
    aside = _aside(queue, wave_dir=wave_dir, waves=waves, allocation=allocation,
                   stale_after=stale_after, now=t)
    feed.round(queue)
    s = _summary(queue, round_no=n, at=t,
                 state=psum.WaveState.CLOSED if queue.closed() else psum.WaveState.OPEN)
    _write_whole(path, psum.summary_json(s))
    _append(wave_dir / LEDGER, {
        "round": n, "at": s.at, "requests": requests,
        "reaped": [{"task": r.worker, "run": r.run, "job": r.job,
                    "outcome": None if r.outcome is None else r.outcome.value,
                    "walltime": r.walltime} for r in reaped],
        "aside": aside, "state": s.state.value})
    log(f"round {n}: {len(requests)} requests, {len(reaped)} reaped, aside {aside}; "
        + ", ".join(f"{x.node} {x.cls} {x.pending} pending {x.claimed} claimed {x.workers} "
                    f"workers" for x in s.lines))
    return s


# ---- the steward ------------------------------------------------------------------------------

def _ended(jobs: Sequence[Job]) -> bool:
    """Whether every job of the wave has ended (ENDED): a job waiting for the long lane has not."""
    return all(j.state in ENDED for j in jobs)


def _end(queue: Queue, *, wave_dir: Path, round_no: int, now: Callable[[], datetime],
         log: Callable[[str], None]) -> Exit:
    """The wave's end: its final snapshot, its bank and tars deleted, its last summary (CANCELLED
    if a cancel ended a job, else FINISHED), in that order (a summary saying it ended is there only
    once the snapshot is), and the ledger's line; 0."""
    queue.backup(wave_dir / FINAL)
    inputs = wave_dir / INPUTS
    gone = [p for p in (inputs / BANK, *sorted(inputs.glob(TARS))) if p.exists()]
    for p in gone:
        p.unlink()
    jobs = queue.jobs()
    state = (psum.WaveState.CANCELLED if any(j.state == JobState.CANCELLED for j in jobs)
             else psum.WaveState.FINISHED)
    last = _summary(queue, round_no=round_no, at=now(), state=state)
    _write_whole(wave_dir / SUMMARY, psum.summary_json(last))
    _append(wave_dir / LEDGER, {"round": round_no, "at": last.at, "end": state.value,
                                "final": FINAL, "deleted": [p.name for p in gone]})
    log(f"wave {last.wave} {state}: {FINAL} written, {len(gone)} inputs deleted; exit 0")
    return Exit.DRAINED


def steward(*, queue: Queue, wave_dir: Path, waves: Path, liveness: Liveness, feed: Feed,
            allocation: Callable[[], Allocation | None], stale_after: timedelta,
            now: Callable[[], datetime], log: Callable[[str], None], ends_at: float,
            poll_seconds: float) -> Exit:
    """The wave's steward (the module's doc): a round every `poll_seconds` until every job has
    ended (its end: 0) or two rounds no longer fit before `ends_at` (epoch seconds on `now`'s
    clock: 75)."""
    h = queue.header()
    log(f"=== steward of wave {h.wave} at {h.commit}: a round every {poll_seconds:g} s, "
        f"{(ends_at - now().timestamp()) / 60:.1f} minutes of its lease left")
    while True:
        started = time.monotonic()
        s = round_(queue=queue, wave_dir=wave_dir, waves=waves, liveness=liveness, feed=feed,
                   allocation=allocation, stale_after=stale_after, now=now, log=log)
        if _ended(queue.jobs()):
            return _end(queue, wave_dir=wave_dir, round_no=s.round, now=now, log=log)
        if ends_at - now().timestamp() < 2 * poll_seconds:
            log("two rounds no longer fit before the lease's end: retired (75)")
            return Exit.RETIRED
        time.sleep(max(0.0, poll_seconds - (time.monotonic() - started)))


def board_allocation(board: Path, *, log: Callable[[str], None]) -> Allocation | None:
    """The market's allocation on `board` (`<board>/gpu/allocation.json`), read now; None when it
    is absent, and when it cannot be read or does not read here (said in `log`): it asks nobody."""
    path = board / GPU / ALLOCATION
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as e:
        log(f"{path} cannot be read ({e}): asking nobody aside")
        return None
    try:
        return read_allocation(text, str(path))
    except ValueError as e:
        log(f"{path} does not read here ({e}): asking nobody aside")
        return None
