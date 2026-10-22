"""The lease pool's command line on the cluster
(docs/superpowers/specs/2026-10-21-lease-pool-design.md, 3.5, 3.6 and 3.8): what a pool's lease, a
long run and a wave's steward run, each a task of a cluster_submit run.

    python -u research/roadless/pool.py work --wave <w> --node <n> --class <c>   # a pool's task
    python -u research/roadless/pool.py work --wave <w> --job <id>                # a long run's
    python -u research/roadless/pool.py steward --wave <w> --board <path> \
        --stale-after-minutes <m>                                                # the steward's

Every Strategy is built here, where the command line and the task's environment are read, and
passed to pool_worker (`work`, `work_one`) and pool_steward (`steward`):
- the queue: `$HOME/reblock-waves/<w>/queue.db`, made from the wave's seed
  (`$HOME/reblock-waves/<w>/inputs/jobs.json`) by whichever process of the wave opens it first
  (the plan's Ruling 7: no worker waits for its steward's first round);
- the worker: `$SLURM_ARRAY_JOB_ID`_`$SLURM_ARRAY_TASK_ID` (as squeue -r names it), restarted
  `$SLURM_RESTART_COUNT` times (Slurm sets it once a task is requeued: absent, none), of the run
  `$CLUSTER_SUBMIT_RUN`, its walltime ending at `$SLURM_JOB_END_TIME`, its commit its checkout's
  (`$CLUSTER_SUBMIT_CHECKOUT`, cluster_submit's manifest, key `commit`); it runs on its jobs'
  node (`$SLURMD_NODENAME`: a block's jobs run on its node), a pool's `--node`, a long run's
  `--job`'s, and a pool's `--class` is one of the wave's ladder's (each a SystemExit naming what
  it is not);
- the OOM evidence: resolved from /proc/self/cgroup (pool_worker.oom_evidence);
- a job: `bash -c <its line>` in the checkout, its environment the worker's with the wave's
  inputs (`CLUSTER_SUBMIT_INPUTS=<wave>/inputs`, where cluster_submit's would name the run's own)
  and bank (`REBLOCK_BLOCK_BANK=<wave>/inputs/bank.pkl`: common.BANK_ENV, which guard `worker`
  checks the child's environment by; common is not imported here, its scipy and geopandas would
  sit in every worker's cgroup);
- the clock (time.time) and a poll's wait (the ladder's poll_seconds);
- the steward's: its liveness (pool_steward.Squeue: `squeue` as `$USER`, its own task
  `$SLURM_ARRAY_JOB_ID`_`$SLURM_ARRAY_TASK_ID`), its feed (pool_steward.FEEDS, by the queue
  header's `feed`), the allocation read from `<board>/gpu/allocation.json` each round (a relative
  `--board` is below `$HOME`, as the login node names it; absent or unreadable, it asks nobody
  aside), the market's stale_after (`--stale-after-minutes`), a round every STEWARD_POLL_SECONDS
  and its lease's end (`$SLURM_JOB_END_TIME`); and the wave's metric, named in the queue's header
  at its first round (metrics.RUNTIME at the wave's commit: the spec's 3.11).

Nothing here needs a GPU. The exit is pool_worker.Exit's (0 drained, done or the wave ended, 1 the
worker's own fault, 75 retired).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pool_queue as pq
import pool_steward as ps
import pool_worker as pw
from cluster_submit.project import NAME
from cluster_submit.remote_build import MANIFEST
from pool_ladder import class_name

#: Where the cgroups are: the task's (/proc/self/cgroup) and their hierarchies' mount.
PROC_SELF_CGROUP = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")
#: The waves' home, below $HOME (the spec's 3.2).
WAVES = "reblock-waves"
#: The steward's round: every minute (the spec's 3.6). Its cost only, never a row's.
STEWARD_POLL_SECONDS = 60.0


def _env(environ: Mapping[str, str], name: str) -> str:
    if name not in environ:
        raise SystemExit(f"pool.py work: ${name} is not set (a task of a cluster_submit run on "
                         "Slurm has it)")
    return environ[name]


def checkout_commit(checkout: Path) -> str:
    """The commit of `checkout`: cluster_submit's manifest's `commit` (its other keys are its
    own); a manifest without one is a SystemExit."""
    path = checkout / MANIFEST
    value = json.loads(path.read_text())
    if not (isinstance(value, dict) and "commit" in value and isinstance(value["commit"], str)):
        raise SystemExit(f"{path}: no commit")
    commit: str = value["commit"]
    return commit


def check_node(node: str, environ: Mapping[str, str]) -> None:
    """A pool's worker of `node` runs on it (`$SLURMD_NODENAME`), else a SystemExit naming both:
    its jobs are its node's, whose results can differ by machine."""
    here = _env(environ, "SLURMD_NODENAME")
    if here != node:
        raise SystemExit(f"pool.py work: --node {node}, but this task runs on {here}")


def check_job_node(job: int, queue: pq.Queue, environ: Mapping[str, str]) -> None:
    """A long run of job `job` runs on the job's node (`$SLURMD_NODENAME`), else a SystemExit
    naming both, as a pool's worker does on its `--node`; a job the queue lacks is work_one's to
    refuse."""
    here = _env(environ, "SLURMD_NODENAME")
    for j in queue.jobs():
        if j.id == job and j.node != here:
            raise SystemExit(f"pool.py work: --job {job} is {j.node}'s, but this task runs on "
                             f"{here}")


def restarts_of(environ: Mapping[str, str]) -> int:
    """The times Slurm has restarted this task: `$SLURM_RESTART_COUNT`, which Slurm sets once it
    requeues a job (absent: never)."""
    return int(environ["SLURM_RESTART_COUNT"]) if "SLURM_RESTART_COUNT" in environ else 0


def check_class(cls: str, header: pq.Header) -> None:
    """A pool's worker's class is one of its wave's ladder's, else a SystemExit listing them."""
    lad = header.ladder
    classes = [class_name(lad.cards, r) for r in lad.rungs]
    if cls not in classes:
        raise SystemExit(f"pool.py work: --class {cls} is no class of wave {header.wave}'s ladder "
                         f"{lad.name} ({', '.join(classes)})")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="pool.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    verbs = ap.add_subparsers(dest="verb", required=True)
    w = verbs.add_parser("work", help="a pool's worker (--node, --class) or a long run (--job)")
    w.add_argument("--wave", required=True)
    w.add_argument("--node", help="the node whose jobs a pool's worker claims")
    w.add_argument("--class", dest="cls", help="the class whose jobs a pool's worker claims")
    w.add_argument("--job", type=int, help="the one job a long run runs")
    w.set_defaults(verb_run=_work)
    s = verbs.add_parser("steward", help="a wave's steward")
    s.add_argument("--wave", required=True)
    s.add_argument("--board", required=True,
                   help="the market's board (a relative one is below $HOME)")
    s.add_argument("--stale-after-minutes", type=float, required=True,
                   help="the market's stale_after: an allocation older asks nobody aside")
    s.set_defaults(verb_run=_steward)
    return ap


def _log(line: str) -> None:
    print(line, flush=True)


def _queue(wave: Path) -> pq.SqliteQueue:
    """The queue of the wave in `wave`, made from its seed if no process has made it."""
    seed = wave / ps.INPUTS / ps.SEED
    return pq.SqliteQueue.open(wave / "queue.db", pq.read_seed(seed.read_text(), str(seed)))


def _work(ap: argparse.ArgumentParser, a: argparse.Namespace,
          environ: Mapping[str, str]) -> pw.Exit:
    long_run = a.job is not None and a.node is None and a.cls is None
    pooled = a.job is None and a.node is not None and a.cls is not None
    if not (long_run or pooled):
        ap.error("work takes --node and --class (a pool's worker), or --job alone (a long run)")
    if pooled:
        check_node(a.node, environ)
    worker = f"{_env(environ, 'SLURM_ARRAY_JOB_ID')}_{_env(environ, 'SLURM_ARRAY_TASK_ID')}"
    restarts = restarts_of(environ)
    run = _env(environ, "CLUSTER_SUBMIT_RUN")
    ends_at = float(_env(environ, "SLURM_JOB_END_TIME"))
    checkout = Path(_env(environ, "CLUSTER_SUBMIT_CHECKOUT"))
    commit = checkout_commit(checkout)
    oom = pw.oom_evidence(PROC_SELF_CGROUP.read_text(), CGROUP_ROOT)
    wave = Path(_env(environ, "HOME")) / WAVES / a.wave
    inputs = wave / ps.INPUTS
    queue = _queue(wave)
    child_env = {**environ, "CLUSTER_SUBMIT_INPUTS": str(inputs),
                 "REBLOCK_BLOCK_BANK": str(inputs / ps.BANK)}

    def run_line(c: pq.Claim) -> pw.Ran:
        return pw.run_job(c.job.line, cwd=checkout, env=child_env, log=_log,
                          wrote=lambda: queue.wrote(c.attempt),
                          kept=lambda: queue.kept(c.attempt))

    if a.job is not None:
        check_job_node(a.job, queue, environ)
        return pw.work_one(queue=queue, worker=worker, restarts=restarts, run=run, job=a.job,
                           commit=commit, ends_at=ends_at, now=time.time, run_line=run_line,
                           oom=oom, log=_log)
    check_class(a.cls, queue.header())
    poll = queue.header().ladder.poll_seconds
    return pw.work(queue=queue, worker=worker, restarts=restarts, run=run, node=a.node,
                   cls=a.cls, commit=commit, ends_at=ends_at, now=time.time,
                   wait=lambda: time.sleep(poll), run_line=run_line, oom=oom, log=_log)


def _steward(ap: argparse.ArgumentParser, a: argparse.Namespace,
             environ: Mapping[str, str]) -> pw.Exit:
    # here, not at the top: its numpy and scipy would sit in every worker's cgroup
    import metrics
    home = Path(_env(environ, "HOME"))
    wave = home / WAVES / a.wave
    queue = _queue(wave)
    feed = queue.header().feed
    if feed not in ps.FEEDS:
        raise SystemExit(f"pool.py steward: wave {a.wave}'s feed {feed!r} is none of "
                         f"{', '.join(ps.FEEDS)}")
    queue.name_metric(metrics.RUNTIME)
    board = home / a.board          # an absolute --board stays itself
    own = f"{_env(environ, 'SLURM_ARRAY_JOB_ID')}_{_env(environ, 'SLURM_ARRAY_TASK_ID')}"
    return ps.steward(queue=queue, wave_dir=wave, waves=home / WAVES,
                      liveness=ps.Squeue(run=ps.run_command, user=_env(environ, "USER"),
                                         own=own),
                      feed=ps.FEEDS[feed](),
                      allocation=lambda: ps.board_allocation(board, log=_log),
                      stale_after=timedelta(minutes=a.stale_after_minutes),
                      now=lambda: datetime.now(UTC), log=_log,
                      ends_at=float(_env(environ, "SLURM_JOB_END_TIME")),
                      poll_seconds=STEWARD_POLL_SECONDS)


def main(argv: Sequence[str], environ: Mapping[str, str]) -> pw.Exit:
    """`work` or `steward` (the module's doc), with `environ` the task's environment."""
    ap = _parser()
    a = ap.parse_args(argv)
    if not NAME.fullmatch(a.wave):      # a wave's name is a run name's (the spec's 3.9)
        ap.error(f"--wave {a.wave!r} is not a wave's name ({NAME.pattern})")
    run: Callable[[argparse.ArgumentParser, argparse.Namespace, Mapping[str, str]],
                  pw.Exit] = a.verb_run
    return run(ap, a, environ)


if __name__ == "__main__":
    sys.exit(int(main(sys.argv[1:], os.environ)))
