"""A wave's summary (docs/superpowers/specs/2026-10-21-lease-pool-design.md, 3.6 step 5, with the
plan's Rulings 1, 2, 5 and 13): what its steward writes each round to `<wave>/summary.json`, whole,
and what the rest read of the wave (a lease's `more`, this machine's demand step, `wave status`,
the other waves' stewards stepping aside).

- **Lines** (`lines_of`): one per (node, class) with a job or a live pool worker there, the
  class's bundle (the ladder's cards and CPUs, the rung's memory), its jobs pending, claimed (a
  pool's), waiting for or running in the long lane, done and failed, the mean needed minutes of
  the pending and claimed ones, and its live workers (the long lane's are no class's: a run each,
  stated as runs are).
- **Reaped**: every worker the wave's reaps took (pool_queue's `reaped`), each with whether it
  held a job and whether its walltime had ended (what `more` reads: mycooc's lesson 5).
- **Long**: the jobs waiting for a long run (LONG), each with what a run of it needs.
- **Its ladder's name** (the queue header's, a name of pool_ladder.LADDERS): what this machine's
  `Waves.template` starts a class's pool lease with (`--preset`), so the queue's header stays
  the one record of a wave's ladder and no run's arguments repeat it (the controller's ruling,
  2026-10-21).
- **Read exactly** (`read_summary`): format SUMMARY_FORMAT, every key by name, of its type, none
  unknown; anything else a ValueError naming where it was read (Ruling 13: the format changes
  only while no wave is live). Several waves' summaries on the cluster are read with one command
  (`read_summaries`), each read so.
- **Stepping aside across waves** (`aside_share`, Ruling 5): the campaign's yields on a (node,
  bundle) shared among the live waves there in proportion to their workers, largest remainder.
"""
from __future__ import annotations

import base64
import json
import math
import shlex
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import NamedTuple

from cluster_submit.context import STAMP
from cluster_submit.remote import Login
from pool_ladder import Ladder, class_name
from pool_queue import Job, JobState, Lane, Obj, Worker

SUMMARY_FORMAT = 1


class WaveState(StrEnum):
    OPEN = "open"               # its queue may still be fed (Screened's: decision 7)
    CLOSED = "closed"           # its queue grants nothing more; jobs remain
    FINISHED = "finished"       # every job done or failed: the steward ended it
    CANCELLED = "cancelled"     # a cancel ended it


#: The states of a wave whose steward still runs rounds.
LIVE = (WaveState.OPEN, WaveState.CLOSED)


class ClassLine(NamedTuple):
    """A (node, class) of a wave: its bundle, its jobs by state, and its live workers."""

    node: str
    cls: str
    cards: int
    cpus: int
    mem_gb: int
    pending: int
    claimed: int        # a pool's jobs claimed
    minutes: float      # the mean needed minutes of the pending and claimed jobs, 0 with none
    long: int           # the long lane's jobs not ended: waiting (LONG) or running (CLAIMED)
    done: int
    failed: int
    workers: int        # live (registered, not retired)


class ReapedWorker(NamedTuple):
    task: str           # its Slurm task
    run: str
    held_job: bool
    walltime: bool      # its walltime had ended when it was reaped


class LongJob(NamedTuple):
    """A job waiting for a long run: where, and what a run of it needs."""

    job: int
    block: str
    item: str
    node: str
    mem_gb: int         # its need, rounded up
    minutes: float


@dataclass(frozen=True, kw_only=True)
class Summary:
    """A wave's summary.json: the wave, its commit, its ladder's name, when (STAMP) and at which
    of its steward's rounds, its state and metric, its lines, every worker reaped, and the jobs
    waiting for a long run."""

    wave: str
    commit: str
    ladder: str
    at: str
    round: int
    state: WaveState
    metric: str | None
    lines: tuple[ClassLine, ...]
    reaped: tuple[ReapedWorker, ...]
    long: tuple[LongJob, ...]


def lines_of(jobs: Sequence[Job], workers: Sequence[Worker],
             ladder: Ladder) -> tuple[ClassLine, ...]:
    """One line per (node, class) with a job (in any state) or a live pool worker, by node and
    rung, its bundle `(ladder.cards, ladder.cpus, rung)`. A job or a pool worker of no class of
    `ladder` is refused (ValueError); the long lane's workers (class `long`) are no line's."""
    rungs = {class_name(ladder.cards, r): r for r in ladder.rungs}
    pool = [w for w in workers if not w.retired and w.cls != Lane.LONG.value]
    keys = {(j.node, j.cls) for j in jobs} | {(w.node, w.cls) for w in pool}
    odd = sorted({c for _, c in keys if c not in rungs})
    if odd:
        raise ValueError(f"classes {odd} are no rung of {ladder.name} ({sorted(rungs)})")
    lines: list[ClassLine] = []
    for node, cls in sorted(keys, key=lambda k: (k[0], rungs[k[1]])):
        here = [j for j in jobs if (j.node, j.cls) == (node, cls)]
        by = Counter((j.state, j.lane) for j in here)
        states = Counter(j.state for j in here)
        busy = [j for j in here if j.lane == Lane.POOL
                and j.state in (JobState.PENDING, JobState.CLAIMED)]
        lines.append(ClassLine(
            node=node, cls=cls, cards=ladder.cards, cpus=ladder.cpus, mem_gb=rungs[cls],
            pending=by[(JobState.PENDING, Lane.POOL)], claimed=by[(JobState.CLAIMED, Lane.POOL)],
            minutes=sum(j.minutes for j in busy) / len(busy) if busy else 0.0,
            long=by[(JobState.LONG, Lane.LONG)] + by[(JobState.CLAIMED, Lane.LONG)],
            done=states[JobState.DONE], failed=states[JobState.FAILED],
            workers=sum(1 for w in pool if (w.node, w.cls) == (node, cls))))
    return tuple(lines)


def long_jobs(jobs: Sequence[Job]) -> tuple[LongJob, ...]:
    """The jobs waiting for a long run (LONG), by id."""
    return tuple(LongJob(job=j.id, block=j.block, item=j.item, node=j.node,
                         mem_gb=math.ceil(j.gb), minutes=j.minutes)
                 for j in jobs if j.state == JobState.LONG)


def aside_share(*, yields: int, workers: Mapping[str, int], wave: str) -> int:
    """`wave`'s share of the `yields` units its campaign steps aside on one (node, bundle): the
    largest-remainder apportionment of them over the live waves there in proportion to their
    `workers` (by wave, `wave`'s own among them), remainders' ties to the earlier wave name (the
    plan's Ruling 5). No workers there: 0."""
    if wave not in workers:
        raise ValueError(f"wave {wave} is not among the waves counted ({sorted(workers)})")
    total = sum(workers.values())
    if total == 0:
        return 0
    floors = {w: yields * n // total for w, n in workers.items()}
    left = yields - sum(floors.values())
    ahead = sorted(workers, key=lambda w: (-(yields * workers[w] % total), w))[:left]
    return floors[wave] + (1 if wave in ahead else 0)


# ---- summary.json, exactly --------------------------------------------------------------------

def summary_json(s: Summary) -> str:
    """summary.json's text: one line of JSON (and its newline)."""
    return json.dumps({
        "format": SUMMARY_FORMAT, "wave": s.wave, "commit": s.commit, "ladder": s.ladder,
        "at": s.at, "round": s.round, "state": s.state.value, "metric": s.metric,
        "lines": [line._asdict() for line in s.lines],
        "reaped": [r._asdict() for r in s.reaped],
        "long": [j._asdict() for j in s.long]}, allow_nan=False) + "\n"


def _line(o: Obj) -> ClassLine:
    line = ClassLine(node=o.text("node"), cls=o.text("cls"), cards=o.count("cards"),
                     cpus=o.count("cpus"), mem_gb=o.count("mem_gb"), pending=o.count("pending"),
                     claimed=o.count("claimed"), minutes=o.number("minutes"),
                     long=o.count("long"), done=o.count("done"), failed=o.count("failed"),
                     workers=o.count("workers"))
    o.done()
    return line


def _reaped(o: Obj) -> ReapedWorker:
    r = ReapedWorker(task=o.text("task"), run=o.text("run"), held_job=o.flag("held_job"),
                     walltime=o.flag("walltime"))
    o.done()
    return r


def _long(o: Obj) -> LongJob:
    j = LongJob(job=o.count("job"), block=o.text("block"), item=o.text("item"),
                node=o.text("node"), mem_gb=o.count("mem_gb"), minutes=o.number("minutes"))
    o.done()
    return j


def read_summary(text: str, where: str) -> Summary:
    """summary_json's output, read exactly (every key by name, of its type, none unknown); a
    summary of another format, or anything else, is refused (ValueError naming `where`)."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where}: not JSON: {e}") from e
    o = Obj(value, where)
    fmt = o.count("format")
    if fmt != SUMMARY_FORMAT:
        raise ValueError(f"{where}: format {fmt}, not {SUMMARY_FORMAT}")
    at = o.text("at")
    try:
        datetime.strptime(at, STAMP)
    except ValueError as e:
        raise ValueError(f"{where}: at {at!r} is not a time ({STAMP})") from e
    state = o.text("state")
    if state not in {s.value for s in WaveState}:
        raise ValueError(f"{where}: state {state!r} is none of {[s.value for s in WaveState]}")
    s = Summary(wave=o.text("wave"), commit=o.text("commit"), ladder=o.text("ladder"), at=at,
                round=o.count("round"), state=WaveState(state), metric=o.text_or_null("metric"),
                lines=tuple(_line(Obj(v, f"{where}.lines[{i}]"))
                            for i, v in enumerate(o.items("lines"))),
                reaped=tuple(_reaped(Obj(v, f"{where}.reaped[{i}]"))
                             for i, v in enumerate(o.items("reaped"))),
                long=tuple(_long(Obj(v, f"{where}.long[{i}]"))
                           for i, v in enumerate(o.items("long"))))
    o.done()
    return s


class Summaries(NamedTuple):
    """Waves' summaries as `read_summaries` found them: those that read, and why each other did
    not, both by wave."""

    read: dict[str, Summary]
    unread: dict[str, str]


#: How `read_summaries`'s command marks a summary it could not read (base64 never prints it).
_UNREAD = "!"


def read_summaries(login: Login, paths: Mapping[str, str]) -> Summaries:
    """The summaries at `paths` (by wave, each as `login`'s commands name it), read with one
    command, each exactly (read_summary) and its `wave` the one asked: those that read, and why
    each other did not (absent or unreadable there, not a summary, of another format, another
    wave's). A command that fails, or that answers otherwise than one line a wave, in order, is
    refused (SystemExit): nothing is read, and whoever asked asks again. No paths: none, and no
    command."""
    waves = list(paths)
    if not waves:
        return Summaries(read={}, unread={})
    command = "".join(f"printf '%s ' {shlex.quote(w)}; base64 -w0 {shlex.quote(paths[w])} "
                      f"2>/dev/null || printf %s {shlex.quote(_UNREAD)}; echo\n" for w in waves)
    r = login.probe(command)
    lines = r.stdout.splitlines()
    named = [line.partition(" ")[0] for line in lines]
    if r.returncode or named != waves:
        raise SystemExit(f"the summaries of {', '.join(waves)} on {login.host} could not be read "
                         f"(exit {r.returncode}, waves answered {named}): "
                         f"{r.stderr.strip()[-300:]}")
    out = Summaries(read={}, unread={})
    for w, line in zip(waves, lines, strict=True):
        body = line.partition(" ")[2]
        where = f"{login.host}:{paths[w]}"
        if body == _UNREAD:
            out.unread[w] = f"{where}: absent, or not readable there"
            continue
        try:
            text = base64.b64decode(body, validate=True).decode()
        except ValueError as e:                 # binascii.Error, UnicodeDecodeError
            out.unread[w] = f"{where}: not base64 of UTF-8 text: {e}"
            continue
        try:
            s = read_summary(text, where)
        except ValueError as e:
            out.unread[w] = str(e)
            continue
        if s.wave != w:
            out.unread[w] = f"{where}: wave {s.wave}'s summary, not {w}'s"
            continue
        out.read[w] = s
    return out
