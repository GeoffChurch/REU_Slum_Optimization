"""The lease pool's command line on the cluster
(docs/superpowers/specs/2026-10-21-lease-pool-design.md, 3.5 and 3.8): what a pool's lease and a
long run run, each a task of a cluster_submit run.

    python -u research/roadless/pool.py work --wave <w> --node <n> --class <c>   # a pool's task
    python -u research/roadless/pool.py work --wave <w> --job <id>                # a long run's

Every Strategy is built here, where the command line and the task's environment are read, and
passed to pool_worker (`work`, `work_one`):
- the queue: `$HOME/reblock-waves/<w>/queue.db`, made from the wave's seed
  (`$HOME/reblock-waves/<w>/inputs/jobs.json`) by whichever process of the wave opens it first
  (the plan's Ruling 7: no worker waits for its steward's first round);
- the worker: `$SLURM_ARRAY_JOB_ID`_`$SLURM_ARRAY_TASK_ID`, of the run `$CLUSTER_SUBMIT_RUN`, its
  walltime ending at `$SLURM_JOB_END_TIME`, its commit its checkout's (`$CLUSTER_SUBMIT_CHECKOUT`,
  cluster_submit's manifest, key `commit`); a pool's worker's `--node` must be the node it runs
  on (`$SLURMD_NODENAME`: a block's jobs run on its node) and its `--class` one of the wave's
  ladder's (each a SystemExit naming what it is not);
- the OOM evidence: resolved from /proc/self/cgroup (pool_worker.oom_evidence);
- a job: `bash -c <its line>` in the checkout, its environment the worker's with the wave's
  inputs (`CLUSTER_SUBMIT_INPUTS=<wave>/inputs`, where cluster_submit's would name the run's own)
  and bank (`REBLOCK_BLOCK_BANK=<wave>/inputs/bank.pkl`: common.BANK_ENV, which guard `worker`
  checks the child's environment by; common is not imported here, its scipy and geopandas would
  sit in every worker's cgroup);
- the clock (time.time) and a poll's wait (the ladder's poll_seconds).

Nothing here needs a GPU. The exit is pool_worker.Exit's (0 drained or done, 1 the worker's own
fault, 75 retired).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import pool_queue as pq
import pool_worker as pw
from cluster_submit.remote_build import MANIFEST
from pool_ladder import class_name

#: A wave's name (the spec's 3.9): its directory's, so nothing that leaves it.
WAVE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
#: Where the cgroups are: the task's (/proc/self/cgroup) and their hierarchies' mount.
PROC_SELF_CGROUP = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")


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
    return ap


def main(argv: Sequence[str], environ: Mapping[str, str]) -> pw.Exit:
    """`work` (the module's doc), with `environ` the task's environment."""
    ap = _parser()
    a = ap.parse_args(argv)
    long_run = a.job is not None and a.node is None and a.cls is None
    pooled = a.job is None and a.node is not None and a.cls is not None
    if not (long_run or pooled):
        ap.error("work takes --node and --class (a pool's worker), or --job alone (a long run)")
    if not WAVE_NAME.fullmatch(a.wave):
        ap.error(f"--wave {a.wave!r} is not a wave's name ({WAVE_NAME.pattern})")
    if pooled:
        check_node(a.node, environ)
    worker = f"{_env(environ, 'SLURM_ARRAY_JOB_ID')}_{_env(environ, 'SLURM_ARRAY_TASK_ID')}"
    run = _env(environ, "CLUSTER_SUBMIT_RUN")
    ends_at = float(_env(environ, "SLURM_JOB_END_TIME"))
    checkout = Path(_env(environ, "CLUSTER_SUBMIT_CHECKOUT"))
    commit = checkout_commit(checkout)
    oom = pw.oom_evidence(PROC_SELF_CGROUP.read_text(), CGROUP_ROOT)
    wave = Path(_env(environ, "HOME")) / "reblock-waves" / a.wave
    inputs = wave / "inputs"
    seed = inputs / "jobs.json"
    queue = pq.SqliteQueue.open(wave / "queue.db", pq.read_seed(seed.read_text(), str(seed)))
    child_env = {**environ, "CLUSTER_SUBMIT_INPUTS": str(inputs),
                 "REBLOCK_BLOCK_BANK": str(inputs / "bank.pkl")}

    def log(line: str) -> None:
        print(line, flush=True)

    def run_line(c: pq.Claim) -> pw.Ran:
        return pw.run_job(c.job.line, cwd=checkout, env=child_env, log=log,
                          wrote=lambda: queue.wrote(c.attempt))

    if a.job is not None:
        return pw.work_one(queue=queue, worker=worker, run=run, job=a.job, commit=commit,
                           ends_at=ends_at, now=time.time, run_line=run_line, oom=oom, log=log)
    check_class(a.cls, queue.header())
    poll = queue.header().ladder.poll_seconds
    return pw.work(queue=queue, worker=worker, run=run, node=a.node, cls=a.cls, commit=commit,
                   ends_at=ends_at, now=time.time, wait=lambda: time.sleep(poll),
                   run_line=run_line, oom=oom, log=log)


if __name__ == "__main__":
    sys.exit(int(main(sys.argv[1:], os.environ)))
