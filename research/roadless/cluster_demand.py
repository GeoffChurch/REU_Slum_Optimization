"""reblock's campaign demand in the cluster's GPU market (GeoffChurch/cluster_submit's README, "GPU
market").

The market's keeper charges each job of the project's to the campaign whose demand names its run,
and every other one to the project's hand account (cluster_submit's room.HAND), whose demand
cluster_submit's sync writes itself (room.hand_demand, `gpu/demand/reblock/hand.json`, at the
project's hand_weight): the chain bases of this registry's unfinished runs that no campaign claims,
and what they hold and wait for. Waves.lease (cluster.py) claims the pool runs alone, so the hand
account has every other run, a `blocks` or `cpu` run, a wave's steward and its long runs. The
keeper refuses every demand file naming a run another names (gpuboard.demands: a run is one
campaign's), both of them, so this demand names and states only what the hand account does not:
the campaign's pool leases (owner's ruling, 2026-10-28, with fair-reference admission). It is
stated from here, over ssh, by `state`.

- One campaign, the project's (`reblock`), not one a wave (the controller's ruling, 2026-10-21):
  its spend accumulates over the waves, so a new wave does not rank first in fair share. Its file
  has one writer, this machine's step (`state`, after every verb of cluster.STATES_DEMAND, the
  watch timer's 150-s sync among them), and a wave's steward writes none: it publishes its wave's
  lines in its summary.json, which the step reads for every live wave with one ssh call beside
  squeue (the lease-pool plan's Ruling 1: two writers would each rewrite the other's whole file
  with what it cannot see; a steward sees neither the registry nor which runs are whose leases).
  While a wave is live its steward is an unfinished run, so the timer syncs and the demand is
  restated every 150 s, well inside the market's stale_after. Its runs are the chain bases of this
  registry's unfinished pool runs (`campaign_runs`; cs.gpu.chain_base: a lease's continuations,
  `-l<N>`, are its chain's), the leases room.hand_demand leaves out, and no other.
- A line per (node, bundle) of its waves' pools, across its waves. A running task of one of this
  registry's unfinished pool runs (`pools`: what cluster.Waves.lease says each holds) is its
  wave's, on its pool's (node, bundle), never at its tres-alloc (the same bundle: the plan's
  Ruling 3); no other running task of the project's is stated (`campaign_task`: a steward's, a
  long run's, a `blocks` or `cpu` run's are the hand account's). A pool run's waiting tasks are no
  units: its wave's jobs pending are; no other waiting task is stated either.
- Each wave's pool lines (Ruling 2): for each (node, class) line of a live wave's summary
  (pool_summary.lines_of), the larger of its workers running there (its pool runs' running tasks)
  and its jobs there pending or claimed (`pool_units`: held workers are units, since the keeper
  treats what is held and not demanded as free; a finished wave's fall to 0 once its workers have
  exited, and a line of 0 units is no line), summed over the waves that share the (node, bundle).
  A live wave whose summary does not read (absent, of another format, another wave's), or whose
  records do not (cluster.demand_waves: a pool run whose --mem is no rung of its preset, live
  stewards at two commits), has its running workers stated alone, a unit each, and is said
  (Stated.unread): one wave's fault leaves the rest of the campaign stated. So is any other
  running worker no line read covers (one of a wave no longer live, retiring once its steward has
  ended it), stated alone and not said. A pool run whose record does not say what its workers
  hold has its running tasks stated at what Slurm holds (tres-alloc: a unit each), and its waiting
  ones as no units.
- minutes, with units: the mean of the units' times, each unit's as it is stated. A running
  worker's is its time limit as Slurm holds it (squeue's TimeLimit), an upper bound on its busy
  time; a pool line's units' the mean of its jobs' estimates (pending and claimed), or of its
  running workers' time limits when it has none (Ruling 2).

The demand is written whole (a temporary file, then a rename, as cs.gpu.write_demand does on a
shared home) and never removed: with no lease left it names none and demands nothing, and its
campaign keeps its account. cluster.py writes it after every verb that submits a run or changes
what the registry knows of one, the watch timer's sync among them, and on its own as
`cluster.py demand`.
"""
from __future__ import annotations

import base64
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from cluster_submit import Bundle
from cluster_submit.context import STAMP
from cluster_submit.gpu import Demand, DemandLine, chain_base, demand_json, demand_path, run_of
from cluster_submit.gpuboard import Abstained, Running, Seen, look
from cluster_submit.registry import RunRecord
from cluster_submit.remote import Login
from cluster_submit.slurm import time_seconds
from pool_summary import ClassLine, Summaries, Summary

# Every task of the account's with its time limit (squeue -r: an array's tasks one a line).
LIMITS = 'squeue -h -r --me -O "JobArrayID:|,TimeLimit:"'


class PoolRun(NamedTuple):
    """An unfinished pool run of this registry (cluster.Waves.lease): its wave, its node, and the
    bundle each of its workers holds there."""
    wave: str
    node: str
    bundle: Bundle


class Stated(NamedTuple):
    """The campaign's demand, and the waves whose lines it states and does not (the module's
    doc)."""
    demand: Demand
    waves: tuple[str, ...]      # the live waves whose summaries read: their lines stated
    unread: tuple[str, ...]     # the waves whose summaries or records did not: their running
                                # workers stated alone


class Written(NamedTuple):
    """What `state` did: whether it wrote the demand, and what the demand it wrote lacks (each
    said): the waves whose lines it does not state (their running workers stated alone) and the
    records that did not read (a pool run among them unseen)."""
    written: bool
    unread: tuple[str, ...]
    unreadable: tuple[str, ...]

    def whole(self) -> bool:
        """Written, lacking nothing."""
        return self.written and not self.unread and not self.unreadable


def limits_of(text: str) -> dict[str, float]:
    """LIMITS's output: each task's time limit in minutes, by its id. A line that does not read is
    refused (ValueError), naming it."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        job, sep, limit = line.partition("|")
        if not sep:
            raise ValueError(f"squeue's time limit line is not job|limit: {line!r}")
        out[job.strip()] = time_seconds(limit.strip()) / 60
    return out


def array_of(job: str) -> str:
    """The array job of a task as squeue -r names it (`<array>_<task>`; a lone job, itself)."""
    return job.partition("_")[0]


def finished(rec: RunRecord) -> bool:
    return rec.finished_at is not None


def held(running: Sequence[Running], limits: dict[str, float]
         ) -> dict[tuple[str, Bundle], list[float]]:
    """Each (node, bundle) the running tasks hold, with each task's time limit: a unit each (the
    module's doc)."""
    out: dict[tuple[str, Bundle], list[float]] = {}
    for j in running:
        if len(j.nodes) != 1:
            raise ValueError(f"task {j.job} runs on {len(j.nodes)} nodes: what it holds on each "
                             "is not known")
        out.setdefault((j.nodes[0], j.bundle), []).append(limits[j.job])
    return out


def pool_units(line: ClassLine, running: int) -> int:
    """A wave's pool line's units (Ruling 2): the larger of its workers `running` there and its
    jobs there pending or claimed."""
    return max(running, line.pending + line.claimed)


def wave_lines(read: Mapping[str, Summary], workers: Mapping[PoolRun, Sequence[float]]
               ) -> dict[tuple[str, Bundle], list[float]]:
    """The waves' pool lines (the module's doc), a unit's minutes for each unit, by (node, bundle),
    summed over the waves: of each summary that read (`read`, by wave), each line's units
    (`pool_units`, of its wave's `workers` running there: each pool task running, with its time
    limit, by its run's wave, node and bundle) at the mean of its jobs' estimates, or of its running
    workers' time limits when it has none; then each running worker no such line covers (its
    wave's summary did not read, or has no line where it runs), a unit at its own time limit."""
    out: dict[tuple[str, Bundle], list[float]] = {}
    alone = dict(workers)
    for w in sorted(read):
        for x in read[w].lines:
            bundle = Bundle(cards=x.cards, cpus=x.cpus, mem_gb=x.mem_gb)
            mine = alone.pop(PoolRun(wave=w, node=x.node, bundle=bundle), ())
            units = pool_units(x, len(mine))
            if not units:
                continue            # nothing to run there and no worker holding it: no line
            minutes = x.minutes if x.pending + x.claimed else sum(mine) / len(mine)
            out.setdefault((x.node, bundle), []).extend([minutes] * units)
    for r, mine in alone.items():
        out.setdefault((r.node, r.bundle), []).extend(mine)
    return out


def pool_task(job: str, pooled: Mapping[str, PoolRun | None]) -> bool:
    """Whether task `job` is a pool lease's (`pooled`: each unfinished pool run's array job, with
    what its workers hold, or None when its record does not say)."""
    return array_of(job) in pooled


def campaign_task(j: Running, pooled: Mapping[str, PoolRun | None], *, user: str,
                  project: str) -> bool:
    """Whether running task `j` is the campaign's to state: the account's (`user`), of `project`'s
    runs (its job's name), and a pool lease's (`pool_task`). Every other task of the project's is
    the hand account's (room.hand_demand), and stated there."""
    return j.user == user and run_of(project, j.name) is not None and pool_task(j.job, pooled)


def at_tres_alloc(j: Running, pooled: Mapping[str, PoolRun | None]) -> bool:
    """Whether running task `j` is stated at what Slurm holds (tres-alloc), a unit: one that is no
    pool lease's (none reaches here: `campaign_task`), or a lease's whose record does not say what
    it holds; every other lease's task its wave's line states."""
    return not pool_task(j.job, pooled) or pooled[array_of(j.job)] is None


def campaign_runs(pools: Mapping[str, PoolRun | None]) -> tuple[str, ...]:
    """The runs the campaign's demand names: the chain base of each of this registry's unfinished
    pool runs (`pools`, by name), the leases room.hand_demand leaves out, and no other (a run two
    demands name is refused in both: gpuboard.demands)."""
    return tuple(sorted({chain_base(r) for r in pools}))


def demand_of(recs: Sequence[RunRecord], seen: Seen, limits: dict[str, float], *,
              pools: Mapping[str, PoolRun | None], waves: Summaries, project: str, campaign: str,
              weight: float, at: str) -> Stated:
    """The campaign's demand as `campaign`, at `weight`, written `at` (STAMP), from what squeue
    lists (`seen`; `limits`, each task's time limit in minutes, read after it: a task it lacks
    left squeue between the reads, and is left out), this registry's records (`recs`), its
    unfinished pool runs (`pools`, by name, each with what its workers hold, or None when its
    record does not say) and its live waves' summaries as they were read (`waves`: those that
    read, and those that did not, or whose records did not) (the module's doc)."""
    pooled = {a.job_id: pools[r.name] for r in recs if r.name in pools
              for a in r.arrays if a.job_id is not None}
    running = [j for j in seen.running if j.job in limits
               and campaign_task(j, pooled, user=seen.user, project=project)]
    workers: dict[PoolRun, list[float]] = {}
    for j in running:
        if pool_task(j.job, pooled) and (pool := pooled[array_of(j.job)]) is not None:
            workers.setdefault(pool, []).append(limits[j.job])
    busy = held([j for j in running if at_tres_alloc(j, pooled)], limits)
    waved = wave_lines(waves.read, workers)
    lines = []
    for key in dict.fromkeys([*busy, *waved]):
        minutes = [*busy.get(key, []), *waved.get(key, [])]
        lines.append(DemandLine(node=key[0], bundle=key[1], units=len(minutes),
                                minutes=sum(minutes) / len(minutes) if minutes else 0.0,
                                urgency=None))
    return Stated(Demand(project=project, campaign=campaign, at=at, weight=weight,
                         runs=campaign_runs(pools), lines=tuple(lines)),
                  waves=tuple(sorted(waves.read)), unread=tuple(sorted(waves.unread)))


def board_script(board: str, demand: Demand) -> str:
    """The remote shell, run in the login directory, that writes `demand` whole at its path below
    `board` (cs.gpu.demand_path): a temporary file of this shell's own beside it, no reader's glob
    matching it, then a rename. Exit 2 when it could not."""
    path = demand_path(Path(board), demand.project, demand.campaign)
    tmp = shlex.quote(str(path.with_name(f".{demand.campaign}."))) + "$$.tmp"
    payload = base64.b64encode(demand_json(demand).encode()).decode()
    return (f"mkdir -p {shlex.quote(str(path.parent))} || exit 2\n"
            f"echo {payload} | base64 -d > {tmp} && mv -f {tmp} {shlex.quote(str(path))} "
            f"|| {{ rm -f {tmp}; exit 2; }}\n")


def state(*, recs: Sequence[RunRecord], unreadable: Sequence[str],
          pools: Mapping[str, PoolRun | None], summaries: Callable[[], Summaries], login: Login,
          board: str, project: str, campaign: str, weight: float, now: datetime) -> Written:
    """The campaign's demand, from squeue, this registry's records (`recs`; `unreadable`, those
    that did not read), its unfinished pool runs (`pools`, by name) and its live waves' summaries
    (`summaries`, read with one ssh call once squeue has been; a wave whose records did not read
    among those that did not), written on the board `board` (Cluster.board()) through `login`:
    what it did (Written). Nothing is written when squeue could not be read (a blind step states
    nothing it could not see; the last demand stands, and still attributes once stale); a wave
    whose summary or records do not read, and a record that does not, are said and left out of
    what is written (the wave's running workers stated alone). A command reading the summaries
    that fails is refused (SystemExit, pool_summary.read_summaries'), nothing written, as with a
    blind squeue."""
    seen = look(login)
    if isinstance(seen, Abstained):
        print(f"[demand] nothing stated: {seen.why}", file=sys.stderr)
        return Written(written=False, unread=(), unreadable=())
    limits = limits_of(login.sh(LIMITS))
    waves = summaries()
    s = demand_of(recs, seen, limits, pools=pools, waves=waves, project=project,
                  campaign=campaign, weight=weight, at=now.strftime(STAMP))
    login.sh(board_script(board, s.demand))
    said = "; ".join(f"{line.units} of ({line.bundle.text()}) on {line.node}"
                     + (f", {line.minutes:.0f} min a unit" if line.units else "")
                     for line in s.demand.lines) or "nothing"
    print(f"[demand] {s.demand.name()}: {len(s.demand.runs)} pool runs"
          + (f", the lines of waves {', '.join(s.waves)}" if s.waves else "") + f"; {said}")
    for w in s.unread:
        print(f"[demand] wave {w} does not read, its running workers stated alone: "
              f"{waves.unread[w]}", file=sys.stderr)
    if unreadable:
        print(f"[demand] unreadable records (a wave's pool run among them unseen): "
              f"{', '.join(unreadable)}", file=sys.stderr)
    return Written(written=True, unread=s.unread, unreadable=tuple(unreadable))
