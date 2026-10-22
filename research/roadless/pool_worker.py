"""The pool's worker (docs/superpowers/specs/2026-10-21-lease-pool-design.md, 3.5; the plan's
Rulings 6 and 7; the spec's decision 9), after GeoffChurch/mycooc's scheduler/worker.py `work`:
every task of a pool's lease alike claims a job of its node and class from its wave's queue, runs
it as a child, records it and claims again, until the queue is drained, nothing pending fits its
walltime, it has idled, or it is asked to step aside. pool.py builds what it needs where it reads
its command line (the queue, the OOM evidence, the clock, the job's runner) and passes it in.

- **The loop** (`work`): refuse a queue of another commit than the worker's checkout (3.11),
  before registering; register; then, at each job boundary: asked to step aside, retire (75);
  claim the longest pending job of its node and class estimated within the walltime left less the
  ladder's margin (recomputed at each claim), never one any worker of its lease has attempted
  (pool_queue's claim: a bad node spends one strike a job across the lease, and a job that failed
  for its own reasons is the next lease's); none such, retire (75: mycooc's lesson 4); nothing
  pending, drained (0) on a closed queue,
  else wait a poll, retiring (75) once it has idled the ladder's idle minutes, or at once when no
  job could start in the walltime left (an idle worker killed at its walltime is a failed task,
  which ends its lease's chain). Its every exit retires it from the queue (`finally`), so a job
  it held goes back as if never claimed.
- **A job** (`run_job`): `bash -c <line>` in the checkout, stdout and stderr merged and streamed
  into the worker's log between the job's markers; each row_marker.ROW_WRITTEN line counted into
  the attempt as it comes (so a reaper sees what a killed job had written); its code, and its
  peak (`ru_maxrss`, the child's and its descendants' maximum) from `os.wait4`. The child runs at
  oom_score_adj 1000, set by itself before it execs (raising its own needs no privilege), so a
  cgroup's OOM killer takes the job, not the worker (the plan's Review Focus 4); every process it
  forks inherits it. It leads a process group of its own (a new session), so a kill on the
  worker's error path takes the group: a compound line (`tar -xf ... && python ...`, where bash
  forks) leaves no grandchild running past its attempt. The session changes no cgroup: the job
  stays in its worker's, whose OOM count is read, and Slurm's cgroup process tracking kills it
  with the task (ai's `scontrol show config`, 2026-10-21: ProctrackType proctrack/cgroup,
  TaskPlugin task/cgroup,task/affinity; under proctrack/pgid a new session would outlive it).
- **Its outcome** (`_outcome`): exit 0 is DONE; otherwise an increase of the cgroup's OOM-kill
  count across the child (`OomEvidence`, read before and after) is OOM, which promotes the job
  (pool_queue); anything else is FAILED, a SIGKILL with no increase or no readable count among
  them (decision 9: no guess), and a strike unless the attempt wrote a row (pool_queue).
- **A failed job never fails its worker** (3.5): cluster_submit continues a lease only when none
  of its tasks failed, so the job's state carries the failure. The worker's only non-routine exit
  (1) is its own fault: a queue of another commit, or FAILED_IN_A_ROW different jobs failed in a
  row (a bad node, as quadro2's CUDA was): the lease then ends and a person looks. An exception
  (an unreadable queue) exits 1 as well.
- **The long lane** (`work_one`, the spec's 3.8): a run of its own claims its one job by id, runs
  it and records it: 0 when done, 1 otherwise.
- **No SIGTERM handler**: retiring on SIGTERM needs all three legs of the wiki's "Self-retiring
  claim on walltime" (the shell's delivery, forwarding, the child's retirement) or is worse than
  none (the spec's 3.2); the walltime fit and the steward's reaper cover a walltime kill.

The OOM evidence (`oom_evidence`) is resolved from /proc/self/cgroup at each worker's start and
logged once: cgroup v2's `memory.events` (its `oom_kill`), v1's `memory.oom_control` (its
`oom_kill`, kernel 4.13 on), or `Unreadable` with why, every OOM then a strike. The victim's own
cgroup counts an OOM kill (the kernel records it on the killed process's memcg), and the job runs
in its worker's, so the worker's cgroup is the one read; ai's is confirmed by a forced OOM in the
smoke waves (the plan's Task 10).
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import NamedTuple, Protocol

import row_marker
from pool_queue import Claim, Header, JobState, Lane, Outcome, Queue

#: Different jobs failed in a row after which a worker exits 1, its own fault (a bad node): a
#: worker's exit, never a row, so no parameter of a run's identity.
FAILED_IN_A_ROW = 3

#: The child's oom_score_adj: the most, so the kernel's OOM killer takes it before its worker.
OOM_SCORE_ADJ = 1000


class Exit(IntEnum):
    DRAINED = 0         # the queue closed, nothing of its node and class pending (work_one: done;
                        # the steward: its wave ended)
    OWN_FAULT = 1       # another commit's queue, jobs failed in a row (work_one: not done)
    RETIRED = 75        # routine: asked aside, nothing pending fits its walltime, idle (the
                        # steward: two rounds no longer fit its lease)


# ---- the OOM evidence -----------------------------------------------------------------------

class OomEvidence(Protocol):
    """The worker's cgroup's count of processes killed for memory, read before and after each job:
    an increase is the job's OOM."""

    @property
    def about(self) -> str: ...

    def kills(self) -> int | None:
        """The count now; None when it cannot be read."""
        ...


def _count(path: Path, key: str) -> int | None:
    """The value of `key` in the flat-keyed file `path` (`<key> <value>` lines); None when the
    file cannot be read or has no such line."""
    try:
        text = path.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == key:
            try:
                return int(parts[1])
            except ValueError:
                return None
    return None


@dataclass(frozen=True, kw_only=True)
class CgroupV2:
    """cgroup v2: `oom_kill` of the cgroup's memory.events (hierarchical, so the cgroup's own and
    its descendants': the job runs in its worker's cgroup)."""

    path: Path          # the cgroup's directory

    @property
    def about(self) -> str:
        return f"cgroup v2, oom_kill of {self.path / 'memory.events'}"

    def kills(self) -> int | None:
        return _count(self.path / "memory.events", "oom_kill")


@dataclass(frozen=True, kw_only=True)
class CgroupV1:
    """cgroup v1: `oom_kill` of the memory controller's memory.oom_control (kernel 4.13 on)."""

    path: Path          # the cgroup's directory in the memory controller's hierarchy

    @property
    def about(self) -> str:
        return f"cgroup v1, oom_kill of {self.path / 'memory.oom_control'}"

    def kills(self) -> int | None:
        return _count(self.path / "memory.oom_control", "oom_kill")


@dataclass(frozen=True, kw_only=True)
class Unreadable:
    """No count to read (`why`): every job killed is FAILED, an OOM among them (decision 9)."""

    why: str

    @property
    def about(self) -> str:
        return f"no OOM-kill count: {self.why}"

    def kills(self) -> int | None:
        return None


def oom_evidence(proc_self_cgroup: str, root: Path) -> OomEvidence:
    """The OOM evidence of the cgroup `proc_self_cgroup` (the text of /proc/self/cgroup) names,
    its hierarchies mounted under `root` (/sys/fs/cgroup), probed once: a v1 line whose
    controllers include `memory` (`<id>:memory:<path>`, mounted at `root/<controllers>`), else the
    v2 line (`0::<path>`, at `root`), its count read now; Unreadable when neither is there or the
    count cannot be read. The v1 memory line comes first: a controller is bound to one hierarchy,
    so on a hybrid system (a `0::` line beside v1's) the unified one holds no memory controller."""
    v1: list[tuple[str, str]] = []
    v2: list[str] = []
    for line in proc_self_cgroup.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, path = parts
        if "memory" in controllers.split(","):
            v1.append((controllers, path))
        elif hierarchy == "0" and controllers == "":
            v2.append(path)
    found: OomEvidence
    if v1:
        controllers, path = v1[0]
        found = CgroupV1(path=root / controllers / path.lstrip("/"))
    elif v2:
        found = CgroupV2(path=root / v2[0].lstrip("/"))
    else:
        return Unreadable(why="/proc/self/cgroup names neither v1's memory controller nor a v2 "
                              f"hierarchy: {proc_self_cgroup.strip()!r}")
    if found.kills() is None:
        return Unreadable(why=f"{found.about} cannot be read")
    return found


# ---- a job ------------------------------------------------------------------------------------

class Ran(NamedTuple):
    """A job's run. Its peak is the child's and its waited descendants' maximum RSS (`ru_maxrss`,
    GB of 2**30 bytes as gap_calib.peak_gb's), at least the worker's own RSS when it forked (the
    kernel counts the forked image's before the exec): a floor of about 0.1 GB, far below a rung."""

    rc: int             # the shell's code, -signal when killed
    peak_gb: float
    minutes: float      # its wall
    rows: int           # the row_marker.ROW_WRITTEN lines it printed


def _last_for_the_oom_killer() -> None:
    """In the child, before it execs: its oom_score_adj OOM_SCORE_ADJ, the kernel's first choice
    for an OOM kill (an unprivileged process may raise its own)."""
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write(str(OOM_SCORE_ADJ))


def _kill(proc: subprocess.Popen[str]) -> None:
    """SIGKILL the job's process group: its shell and every process the shell forked. The group
    is there while its leader is unreaped, which it is until `run_job` waits for it."""
    os.killpg(proc.pid, signal.SIGKILL)


def run_job(line: str, *, cwd: Path, env: Mapping[str, str], log: Callable[[str], None],
            wrote: Callable[[], None]) -> Ran:
    """`bash -c <line>` in `cwd` with `env`, at oom_score_adj OOM_SCORE_ADJ, leading a process
    group of its own, its output (stdout and stderr merged, undecodable bytes replaced) given to
    `log` line by line as it comes, `wrote()` called for each line starting with
    row_marker.ROW_WRITTEN; its code and peak from `os.wait4` once its output ends. If reading it
    raises, its group is killed and the child reaped before the error goes on, so no process of
    the job outlives its worker's attempt."""
    start = time.monotonic()
    # preexec_fn runs between fork and exec, unsafe only beside other threads: the worker has
    # none; start_new_session's setsid runs before it, so the job leads its group from its exec
    proc = subprocess.Popen(["bash", "-c", line], cwd=cwd, env=dict(env), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1, start_new_session=True,
                            preexec_fn=_last_for_the_oom_killer)
    if proc.stdout is None:
        raise RuntimeError("Popen gave no stdout pipe")
    rows = 0
    try:
        for out in proc.stdout:
            text = out.rstrip("\n")
            log(text)
            if text.startswith(row_marker.ROW_WRITTEN):
                rows += 1
                wrote()
    except BaseException:
        _kill(proc)
        proc.wait()         # Popen's: an already reaped child is no new error over this one
        raise
    finally:
        proc.stdout.close()
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)     # so Popen never waits again
    return Ran(rc=proc.returncode, peak_gb=usage.ru_maxrss / 2 ** 20,
               minutes=(time.monotonic() - start) / 60, rows=rows)


def _outcome(rc: int, *, before: int | None, after: int | None) -> Outcome:
    """A job's outcome from its code and the cgroup's OOM-kill count before and after it."""
    if rc == 0:
        return Outcome.DONE
    if before is not None and after is not None and after > before:
        return Outcome.OOM
    return Outcome.FAILED


def _attempt(c: Claim, *, queue: Queue, worker: str, run_line: Callable[[Claim], Ran],
             oom: OomEvidence, log: Callable[[str], None]) -> Outcome:
    """Run `c`'s job between its markers and record it (pool_queue's finish): its outcome."""
    j = c.job
    log(f"=== job {j.id} {j.block} {j.item} start")
    before = oom.kills()
    ran = run_line(c)
    after = oom.kills()
    outcome = _outcome(ran.rc, before=before, after=after)
    state = queue.finish(c.attempt, worker=worker, outcome=outcome, peak_gb=ran.peak_gb,
                         minutes=ran.minutes, rc=ran.rc)
    log(f"=== job {j.id} {outcome} rc {ran.rc} {ran.minutes:.2f} min peak {ran.peak_gb:.2f} GB "
        f"rows {ran.rows}")
    if outcome == Outcome.FAILED and (before is None or after is None):
        log(f"job {j.id}: failed, not told from an OOM ({oom.about})")
    if state != JobState.DONE:
        log(f"job {j.id} now {state}")
    return outcome


def _minutes_left(*, ends_at: float, now: float, margin_minutes: float) -> float:
    """The minutes a worker can give a job it claims at `now`: to its walltime's end less the
    margin."""
    return (ends_at - now) / 60 - margin_minutes


def _refused(header: Header, commit: str) -> str | None:
    """Why a worker at `commit` refuses the queue of `header` (another commit: the spec's 3.11),
    or None."""
    if header.commit == commit:
        return None
    return (f"wave {header.wave}'s queue is at {header.commit}, this worker's checkout at "
            f"{commit}: refused, exit 1")


# ---- the worker -------------------------------------------------------------------------------

def work(*, queue: Queue, worker: str, restarts: int, run: str, node: str, cls: str,
         commit: str, ends_at: float, now: Callable[[], float], wait: Callable[[], None],
         run_line: Callable[[Claim], Ran], oom: OomEvidence, log: Callable[[str], None]) -> Exit:
    """`worker` (a Slurm task of `run`, restarted `restarts` times, on `node` in class `cls`, its
    checkout at `commit`, its walltime ending at `ends_at`, epoch seconds on `now`'s clock;
    registered again when requeued: pool_queue's `register`) claims and runs its jobs until
    it exits (the module's doc), each by `run_line`, never one its lease has attempted, `wait`ing
    a poll on an open queue with nothing pending; the queue's ladder gives its margin and idle
    minutes."""
    header = queue.header()
    refused = _refused(header, commit)
    if refused is not None:
        log(refused)
        return Exit.OWN_FAULT
    lad = header.ladder
    log(f"=== worker {worker} of {run}" + (f" (restart {restarts})" if restarts else "")
        + f": wave {header.wave} at {commit}, {node} {cls}, "
        f"{(ends_at - now()) / 60:.1f} minutes of walltime left; {oom.about}")
    queue.register(worker=worker, run=run, node=node, cls=cls, ends_at=ends_at,
                   restarts=restarts)
    try:
        failed: list[int] = []              # the jobs of the attempts failed in a row
        idle_since: float | None = None
        while True:
            if queue.asked_aside(worker):
                log("asked to step aside: retired (75)")
                return Exit.RETIRED
            left = _minutes_left(ends_at=ends_at, now=now(), margin_minutes=lad.margin_minutes)
            c = queue.claim(worker=worker, minutes_left=left)
            if c is None:
                pending = sum(1 for j in queue.jobs()
                              if (j.node, j.cls, j.state) == (node, cls, JobState.PENDING))
                if pending:
                    log(f"{pending} jobs of {node} {cls} pending, none both unattempted by this "
                        f"lease ({run}) and estimated within the {left:.1f} minutes left less "
                        "the margin: retired (75)")
                    return Exit.RETIRED
                if queue.closed():
                    log(f"the queue closed, nothing of {node} {cls} pending: drained (0)")
                    return Exit.DRAINED
                if left <= 0:
                    log("nothing pending, and no job could start in the walltime left: "
                        "retired (75)")
                    return Exit.RETIRED
                idle_since = now() if idle_since is None else idle_since
                if now() - idle_since >= 60 * lad.idle_minutes:
                    log(f"nothing pending for {lad.idle_minutes:g} minutes: retired (75)")
                    return Exit.RETIRED
                wait()
                continue
            idle_since = None
            outcome = _attempt(c, queue=queue, worker=worker, run_line=run_line, oom=oom, log=log)
            failed = [*failed, c.job.id] if outcome == Outcome.FAILED else []
            if len(set(failed)) >= FAILED_IN_A_ROW:
                log(f"jobs {', '.join(map(str, sorted(set(failed))))} failed in a row: this "
                    "node's or worker's fault?, exit 1")
                return Exit.OWN_FAULT
    finally:
        queue.retire(worker)


def work_one(*, queue: Queue, worker: str, restarts: int, run: str, job: int, commit: str,
             ends_at: float, now: Callable[[], float], run_line: Callable[[Claim], Ran],
             oom: OomEvidence, log: Callable[[str], None]) -> Exit:
    """The long lane's run of job `job` (the spec's 3.8): `worker` claims that job, runs it by
    `run_line` and records it; 0 when it is done, 1 otherwise (a queue of another commit, no such
    job, a job not waiting, a job that failed or was killed for memory). It registers on the
    job's node in the class `long`, no pool's, so no pool's step-aside or demand counts it, at
    its restart count `restarts` (as `work`'s worker)."""
    header = queue.header()
    refused = _refused(header, commit)
    if refused is not None:
        log(refused)
        return Exit.OWN_FAULT
    held = [j for j in queue.jobs() if j.id == job]
    if not held:
        log(f"wave {header.wave} has no job {job}: exit 1")
        return Exit.OWN_FAULT
    [j] = held
    log(f"=== worker {worker} of {run}: the long lane's job {job} of wave {header.wave} at "
        f"{commit}, on {j.node}, estimated {j.minutes:.0f} minutes, "
        f"{(ends_at - now()) / 60:.1f} minutes of walltime left; {oom.about}")
    queue.register(worker=worker, run=run, node=j.node, cls=Lane.LONG.value, ends_at=ends_at,
                   restarts=restarts)
    try:
        c = queue.claim_job(worker=worker, job=job)
        if c is None:
            now_state = next(x.state for x in queue.jobs() if x.id == job)
            log(f"job {job} is {now_state}, not waiting: nothing run, exit 1")
            return Exit.OWN_FAULT
        outcome = _attempt(c, queue=queue, worker=worker, run_line=run_line, oom=oom, log=log)
        return Exit.DRAINED if outcome == Outcome.DONE else Exit.OWN_FAULT
    finally:
        queue.retire(worker)
