"""reblock's demand in the cluster's GPU market (GeoffChurch/cluster_submit's README, "GPU market").

Every campaign that runs on the cluster states demand, a CPU-only one too: the market's keeper
attributes what a campaign's jobs hold and charges it, and a job no demand names is held outside
the market, charged to nobody (on 2026-10-21 reblock's cp-roads4-desire-g3a and -g3b held about
1.39 TB of orion's 1.5 TB so). It is stated from here, over ssh, by `state`.

- One campaign, the project's (`reblock`), not one a wave (the controller's ruling, 2026-10-21):
  its spend accumulates over the waves, so a new wave does not rank first in fair share. Its file
  has one writer, this machine's step (`state`, after every verb of cluster.STATES_DEMAND, the
  watch timer's 150-s sync among them), and a wave's steward writes none: it publishes its wave's
  lines in its summary.json, which the step reads for every live wave with one ssh call beside
  squeue (the lease-pool plan's Ruling 1: two writers would each rewrite the other's whole file
  with what it cannot see; a steward sees neither the registry's throttles nor which runs are
  whose leases). While a wave is live its steward is an unfinished run, so the timer syncs and the
  demand is restated every 150 s, well inside the market's stale_after. Its runs are every run of
  the project's squeue lists (its jobs' names, `{project}-{run}-{label}`: so the keeper attributes
  them whichever checkout last wrote the file) and every unfinished run of this checkout's
  registry (its waves' pool, steward and long runs among them), each as a chain's base
  (cs.gpu.chain_base: a pool lease's continuations, `-l<N>`, are its chain's).
- A line per (node, bundle) the project's tasks hold or wait for, across its waves, read off
  squeue as the keeper reads it (gpuboard.look): a running task's bundle is what Slurm holds
  (tres-alloc), a waiting one's what it asks for, so a task whose memory was changed in Slurm
  counts as Slurm has it now; cards 1 for a task with a GPU (the blocks kind's), 0 for a CPU one.
  A running task of one of this registry's unfinished pool runs (`pools`: what cluster.Waves.lease
  says each holds) is its wave's, on its pool's (node, bundle), never at its tres-alloc (the same
  bundle: the plan's Ruling 3).
- units, the controller's ruling (2026-10-21), as mycooc counts its running work: every running
  task of the project's on that node and bundle but a pool's, plus the waiting tasks of this
  registry's runs that each run's own throttle (`--gpus`, each array's %N: RunRecord.gpus, less
  its running tasks) would start now, in Slurm's priority order, among those that wait for the
  node's resources (gpuboard.waits_for: not on a dependency, a hold, the array's throttle or a
  limit), none for a cancelled run; so a wave's stewards and long runs are stated as any run is.
  The keeper's capacity subtracts only what the account holds outside the market, so what a
  campaign holds and does not demand it treats as free.
- Each wave's pool lines (Ruling 2): for each (node, class) line of a live wave's summary
  (pool_summary.lines_of), the larger of its workers running there (its pool runs' running tasks)
  and its jobs there pending or claimed (`pool_units`: held workers are units, since the keeper
  treats what is held and not demanded as free; a finished wave's fall to 0 once its workers have
  exited, and a line of 0 units is no line), summed over the waves that share the (node, bundle)
  and with the other runs' units there. A pool run's waiting tasks are no units: its wave's jobs
  pending are. A live wave whose summary does not read (absent, of another format, another
  wave's) has its running workers stated alone, a unit each, and is said (Stated.unread). So is
  any other running worker no line read covers (one of a wave no longer live, retiring once its
  steward has ended it), stated alone and not said.
- minutes, with units: the mean of the units' times, each unit's as it is stated. A running or
  waiting task's is its time limit as Slurm holds it (squeue's TimeLimit: the run's --time, a
  resubmit's, the site's default, or a later scontrol change), an upper bound on its busy time; a
  pool line's units' the mean of its jobs' estimates (pending and claimed), or of its running
  workers' time limits when it has none (Ruling 2).
- A waiting task of an unpinned run (blocks without --node) requires no node: it may run on every
  node whose card holds it. A line on each would claim its unit once per node, and the keeper
  allocates line by line: it is stated nowhere for now (Stated.unplaced; the lease-pool design
  places tasks on nodes). The waiting tasks of a run another checkout's registry holds are that
  registry's to state (Stated.elsewhere): only its own step knows their throttle.

The demand is written whole (a temporary file, then a rename, as cs.gpu.write_demand does on a
shared home) and never removed: with no run left it names none and demands nothing, and its
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

# Re-exported so a guard can replace it (cluster_check.py `demand`, fault any-reason).
from cluster_submit.gpuboard import waits_for as waits_for
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
    """The project's demand, the waves whose lines it states, and what it does not state (the
    module's doc)."""
    demand: Demand
    unplaced: int       # this registry's waiting tasks that would start now but name no node
    elsewhere: int      # waiting tasks of the project's runs this registry does not hold
    waves: tuple[str, ...]      # the live waves whose summaries read: their lines stated
    unread: tuple[str, ...]     # the live waves whose summaries did not: their running workers
                                # stated alone


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


def room(rec: RunRecord, running: int) -> int:
    """How many more of an array's tasks its throttle starts now: RunRecord.gpus (each array's %N)
    less its tasks running."""
    return max(0, rec.gpus - running)


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


def demand_of(recs: Sequence[RunRecord], seen: Seen, limits: dict[str, float], *,
              pools: Mapping[str, PoolRun], waves: Summaries, project: str, campaign: str,
              weight: float, at: str, nodes: Sequence[str]) -> Stated:
    """The project's demand as `campaign`, at `weight`, written `at` (STAMP), from what squeue
    lists (`seen`; `limits`, each task's time limit in minutes, read after it: a task it lacks
    left squeue between the reads, and is left out), this registry's runs (`recs`), its unfinished
    pool runs (`pools`, by name) and its live waves' summaries as they were read (`waves`: those
    that read, and those that did not); `nodes` the site's, on which a task of an unpinned run
    may wait (the module's doc)."""
    def ours(job: str, name: str, user: str) -> bool:
        return user == seen.user and run_of(project, name) is not None and job in limits

    running = [j for j in seen.running if ours(j.job, j.name, j.user)]
    pending = [p for p in seen.pending if ours(p.job, p.name, p.user)]
    live = [r for r in recs if not finished(r)]
    names = [j.name for j in running] + [p.name for p in pending]
    runs = {chain_base(r) for n in names if (r := run_of(project, n)) is not None}
    runs |= {chain_base(r.name) for r in live}
    owner = {a.job_id: r for r in live for a in r.arrays if a.job_id is not None}
    pooled = {a: pools[r.name] for a, r in owner.items() if r.name in pools}
    workers: dict[PoolRun, list[float]] = {}
    for j in running:
        if array_of(j.job) in pooled:
            workers.setdefault(pooled[array_of(j.job)], []).append(limits[j.job])
    busy = held([j for j in running if array_of(j.job) not in pooled], limits)
    wanted: dict[tuple[str, Bundle], list[float]] = {}
    left = {a: room(r, sum(1 for j in running if array_of(j.job) == a))
            for a, r in owner.items()}
    unplaced = elsewhere = 0
    for p in pending:                       # in Slurm's priority order (gpuboard.PENDING)
        if len(p.required) > 1:
            raise ValueError(f"task {p.job} requires {len(p.required)} nodes: a task of a "
                             "reblock run runs on one")
        a = array_of(p.job)
        if a in pooled:
            continue                        # a pool lease's: its wave's pending jobs are units
        node = p.required[0] if p.required else None
        if node is not None:
            wanted.setdefault((node, p.bundle), [])
        if a not in owner:
            elsewhere += 1
            continue
        if (owner[a].cancelled_at is not None or not left[a]
                or not any(waits_for(p, n) for n in ((node,) if node is not None else nodes))):
            continue
        left[a] -= 1
        if node is None:
            unplaced += 1
        else:
            wanted[(node, p.bundle)].append(limits[p.job])
    waved = wave_lines(waves.read, workers)
    lines = []
    for key in dict.fromkeys([*busy, *wanted, *waved]):
        minutes = [*busy.get(key, []), *wanted.get(key, []), *waved.get(key, [])]
        lines.append(DemandLine(node=key[0], bundle=key[1], units=len(minutes),
                                minutes=sum(minutes) / len(minutes) if minutes else 0.0,
                                urgency=None))
    return Stated(Demand(project=project, campaign=campaign, at=at, weight=weight,
                         runs=tuple(sorted(runs)), lines=tuple(lines)), unplaced, elsewhere,
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


def state(*, recs: Sequence[RunRecord], unreadable: Sequence[str], pools: Mapping[str, PoolRun],
          summaries: Callable[[], Summaries], login: Login, board: str, project: str,
          campaign: str, weight: float, nodes: Sequence[str], now: datetime) -> int:
    """The project's demand, from squeue, this registry's records (`recs`; `unreadable`, those
    that did not read), its unfinished pool runs (`pools`, by name) and its live waves' summaries
    (`summaries`, read with one ssh call once squeue has been), written on the board `board`
    (Cluster.board()) through `login`. 0 once written; 1 when squeue could not be read (nothing
    is written: a blind step states nothing it could not see; the last demand stands, and still
    attributes once stale), or when a live wave's summary does not read or a record does not
    (each said; the rest stated, that wave's running workers alone). A command reading the
    summaries that fails is refused (SystemExit, pool_summary.read_summaries'), nothing written,
    as with a blind squeue."""
    seen = look(login)
    if isinstance(seen, Abstained):
        print(f"[demand] nothing stated: {seen.why}", file=sys.stderr)
        return 1
    limits = limits_of(login.sh(LIMITS))
    waves = summaries()
    s = demand_of(recs, seen, limits, pools=pools, waves=waves, project=project,
                  campaign=campaign, weight=weight, at=now.strftime(STAMP), nodes=nodes)
    login.sh(board_script(board, s.demand))
    said = "; ".join(f"{line.units} of ({line.bundle.text()}) on {line.node}"
                     + (f", {line.minutes:.0f} min a unit" if line.units else "")
                     for line in s.demand.lines) or "nothing"
    print(f"[demand] {s.demand.name()}: {len(s.demand.runs)} runs"
          + (f", the lines of waves {', '.join(s.waves)}" if s.waves else "") + f"; {said}"
          + (f"; {s.unplaced} waiting tasks of unpinned runs stated nowhere" if s.unplaced
             else "")
          + (f"; {s.elsewhere} waiting tasks of other checkouts' runs left to their registry"
             if s.elsewhere else ""))
    for w in s.unread:
        print(f"[demand] wave {w}'s summary does not read, its running workers stated alone: "
              f"{waves.unread[w]}", file=sys.stderr)
    if unreadable:
        print(f"[demand] unreadable records, their waiting tasks not stated (and a wave's "
              f"steward or pool among them unseen): {', '.join(unreadable)}", file=sys.stderr)
    return 1 if s.unread or unreadable else 0
