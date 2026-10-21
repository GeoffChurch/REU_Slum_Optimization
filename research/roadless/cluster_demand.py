"""reblock's waves in the cluster's GPU market (GeoffChurch/cluster_submit's README, "GPU market").

Every campaign that runs on the cluster states demand, a CPU-only one too: the market's keeper
attributes what a campaign's jobs hold and charges it, and a job no demand names is held outside
the market, charged to nobody (on 2026-10-21 reblock's cp-roads4-desire-g3a and -g3b held about
1.39 TB of orion's 1.5 TB so). reblock has no executor on the cluster to state it, as
GeoffChurch/mycooc's does: it is stated from here, over ssh, by `step` and `state`.

- A wave is its run: the campaign is the run's name and its runs the run alone (a chain's base;
  no reblock run is a lease, so none is continued).
- A line per (node, bundle) the run's tasks hold or wait for, read off squeue as the keeper reads
  it (gpuboard.look): a running task's bundle is what Slurm holds (tres-alloc), a waiting one's
  what it asks for, so a task whose memory was changed in Slurm counts as Slurm has it now; cards
  1 for a task with a GPU (the blocks kind's), 0 for a CPU one (the cpu kind's).
- units: the run's waiting tasks that its own throttle would start now (`--gpus`, each array's %N:
  RunRecord.gpus, less its running tasks), in Slurm's priority order, among those that wait for
  the node's resources (gpuboard.waits_for: not on a dependency, a hold, the array's throttle or
  a limit). So 0 once every task has started, and 0 for a cancelled run, which wants nothing.
  Beyond its throttle a wave could keep no further worker busy now: the README's units.
- minutes, with units: the run's walltime (a resubmit's --time, else the submit's, else the site's
  default_time), the one time a reblock run states for its tasks; an upper bound on a unit's busy
  time, so the keeper's planning errs toward charging the wave more, not less.
- A waiting task of an unpinned run (blocks without --node) requires no node: it may run on every
  node whose card holds it. A line on each would claim its unit once per node, and the keeper
  allocates line by line. Where its unit is demanded is the owner's to decide: until then it is
  stated nowhere, and every step says how many there are (Stated.unplaced).

Each step states the demand of every unfinished run of this machine's registry (written whole:
a temporary file, then a rename, as cs.gpu.write_demand does on a shared home) and removes the
demand of every finished one still on the board, which holds nothing and claims nothing. A run of
another registry (a worktree's) is that registry's to state. cluster.py runs a step after every
verb that submits a run or changes what the registry knows of one, the watch timer's sync among
them, and on its own as `cluster.py demand`.
"""
from __future__ import annotations

import base64
import shlex
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

from cluster_submit import Bundle
from cluster_submit.context import STAMP
from cluster_submit.gpu import DEMAND, GPU, Demand, DemandLine, demand_json
from cluster_submit.gpuboard import Abstained, Seen, look

# Re-exported so a guard can replace it (cluster_check.py `demand`, fault any-reason).
from cluster_submit.gpuboard import waits_for as waits_for
from cluster_submit.registry import Registry, RunRecord
from cluster_submit.remote import Login
from cluster_submit.slurm import time_seconds


class Stated(NamedTuple):
    """An unfinished run's demand, and its waiting tasks that would start now but name no node
    (an unpinned run's): stated nowhere (the module's doc)."""
    demand: Demand
    unplaced: int


class Step(NamedTuple):
    """What one step does on the board."""
    stated: tuple[Stated, ...]              # each unfinished run's demand, written
    finished: tuple[str, ...]               # each finished run, whose demand is removed
    refused: tuple[tuple[str, str], ...]    # (run, why) for an unfinished run not stated


def walltime_minutes(rec: RunRecord, *, submit_time: str | None, default_time: str) -> float:
    """`rec`'s tasks' time limit in minutes, as its arrays were submitted with it: a resubmit's
    `--time` (rec.overrides), else the submit's (`submit_time`, its --time), else the site's
    default_time (cluster_submit's arrays.script)."""
    if rec.overrides.time is not None:
        limit = rec.overrides.time
    elif submit_time is not None:
        limit = submit_time
    else:
        limit = default_time
    return time_seconds(limit) / 60


def array_of(job: str) -> str:
    """The array job of a task as squeue -r names it (`<array>_<task>`; a lone job, itself)."""
    return job.partition("_")[0]


def room(rec: RunRecord, running: int) -> int:
    """How many more of an array's tasks its throttle starts now: RunRecord.gpus (each array's %N)
    less its tasks running."""
    return max(0, rec.gpus - running)


def finished(rec: RunRecord) -> bool:
    return rec.finished_at is not None


def demand_of(rec: RunRecord, seen: Seen, *, project: str, weight: float, at: str,
              nodes: Sequence[str], minutes: float) -> Stated:
    """Unfinished run `rec`'s demand from what squeue lists of it (`seen`, the account's jobs of
    its arrays), at `weight`, written `at` (STAMP), a unit's busy time `minutes`; `nodes` the
    site's, on which a task of an unpinned run may wait (the module's doc)."""
    arrays = {a.job_id for a in rec.arrays if a.job_id is not None}
    running = [j for j in seen.running if j.user == seen.user and array_of(j.job) in arrays]
    pending = [p for p in seen.pending if p.user == seen.user and array_of(p.job) in arrays]
    units: dict[tuple[str, Bundle], int] = {}
    for j in running:
        if len(j.nodes) != 1:
            raise ValueError(f"task {j.job} runs on {len(j.nodes)} nodes: what it holds on each "
                             "is not known")
        units.setdefault((j.nodes[0], j.bundle), 0)
    left = {a: room(rec, sum(1 for j in running if array_of(j.job) == a)) for a in arrays}
    unplaced = 0
    for p in pending:                       # in Slurm's priority order (gpuboard.PENDING)
        if len(p.required) > 1:
            raise ValueError(f"task {p.job} requires {len(p.required)} nodes: a task of a "
                             "reblock run runs on one")
        node = p.required[0] if p.required else None
        if node is not None:
            units.setdefault((node, p.bundle), 0)
        a = array_of(p.job)
        if (rec.cancelled_at is not None or not left[a]
                or not any(waits_for(p, n) for n in ((node,) if node is not None else nodes))):
            continue
        left[a] -= 1
        if node is None:
            unplaced += 1
        else:
            units[(node, p.bundle)] += 1
    lines = tuple(DemandLine(node=n, bundle=b, units=u, minutes=minutes if u else 0.0,
                             urgency=None) for (n, b), u in units.items())
    return Stated(Demand(project=project, campaign=rec.name, at=at, weight=weight,
                         runs=(rec.name,), lines=lines), unplaced)


def step(recs: Sequence[RunRecord], seen: Seen, *, project: str, weight: float, at: str,
         nodes: Sequence[str], minutes: Callable[[RunRecord], float]) -> Step:
    """Every run of `recs` (one registry's): an unfinished one's demand (demand_of, a unit's
    minutes from `minutes`), a finished one's removal. A run whose demand cannot be made (its
    record does not parse as a submit, its name is no chain's base) is refused, alone."""
    stated: list[Stated] = []
    refused: list[tuple[str, str]] = []
    for rec in recs:
        if finished(rec):
            continue
        try:
            stated.append(demand_of(rec, seen, project=project, weight=weight, at=at,
                                    nodes=nodes, minutes=minutes(rec)))
        except (ValueError, SystemExit) as e:
            refused.append((rec.name, str(e)))
    return Step(tuple(stated), tuple(r.name for r in recs if finished(r)), tuple(refused))


def board_script(board: str, project: str, demands: Sequence[Demand]) -> str:
    """The remote shell, run in the login directory, that writes each of `demands` whole as
    `<board>/gpu/demand/<project>/<campaign>.json` (a temporary file of this shell's own beside
    it, no reader's glob matching it, then a rename) and then removes the demand of each run its
    standard input names (one a line) that is there, printing each it removed. Exit 2 when a
    write fails."""
    where = shlex.quote(str(Path(board) / GPU / DEMAND / project))
    lines = [f"mkdir -p {where} && cd {where} || exit 2"]
    for d in demands:
        payload = base64.b64encode(demand_json(d).encode()).decode()
        tmp = f".{d.campaign}.$$.tmp"
        lines.append(f"echo {payload} | base64 -d > {tmp} && mv -f {tmp} "
                     f"{shlex.quote(d.campaign + '.json')} || {{ rm -f {tmp}; exit 2; }}")
    lines += ['gone=$(cat)',
              'for f in *.json; do',
              '  [ -e "$f" ] || continue',
              '  if printf "%s\\n" "$gone" | grep -qxF -- "${f%.json}"; then',
              '    rm -f -- "$f" && echo "${f%.json}"',
              '  fi',
              'done']
    return "\n".join(lines) + "\n"


def state(*, registry: Registry, login: Login, board: str, project: str, weight: float,
          nodes: Sequence[str], minutes: Callable[[RunRecord], float], now: datetime) -> int:
    """One step for every run of `registry`, on the board `board` (Cluster.board()) through
    `login`: 0 when every run was stated, 1 when squeue could not be read (nothing is written: a
    blind step states nothing it could not see), a record does not read or a run was refused (each
    said; the rest are stated)."""
    recs, unreadable = registry.all()
    seen = look(login)
    if isinstance(seen, Abstained):
        print(f"[demand] nothing stated: {seen.why}", file=sys.stderr)
        return 1
    s = step(recs, seen, project=project, weight=weight, at=now.strftime(STAMP), nodes=nodes,
             minutes=minutes)
    removed = login.sh(board_script(board, project, [x.demand for x in s.stated]),
                       input="".join(f"{name}\n" for name in s.finished)).split()
    for x in s.stated:
        said = "; ".join(f"{line.units} units of ({line.bundle.text()}) on {line.node}"
                         for line in x.demand.lines) or "nothing in squeue"
        busy = {line.minutes for line in x.demand.lines if line.units}
        print(f"[demand] {x.demand.campaign}: {said}"
              + (f" ({busy.pop():.0f} min a unit)" if busy else "")
              + (f"; {x.unplaced} waiting tasks of an unpinned run stated nowhere (they name "
                 "no node)" if x.unplaced else ""))
    for name in removed:
        print(f"[demand] {name}: finished, its demand removed")
    for name, why in s.refused:
        print(f"[demand] {name}: not stated: {why}", file=sys.stderr)
    if unreadable:
        print(f"[demand] unreadable records, not stated: {', '.join(unreadable)}",
              file=sys.stderr)
    return 1 if s.refused or unreadable else 0
