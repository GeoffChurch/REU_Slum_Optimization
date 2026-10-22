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
  the attempt as it comes (so a reaper sees what a killed job had written), and each
  row_marker.KEPT line too (a row it found standing: the attempt measures no whole job); its
  code from `os.wait4`, and its peak: the larger of wait4's `ru_maxrss` (the child's and its
  waited descendants' maximum) and the largest session sum its watch saw. The child runs at
  oom_score_adj 1000, set by itself before it execs (raising its own needs no privilege), so a
  cgroup's OOM killer takes the job, not the worker (the plan's Review Focus 4); every process it
  forks inherits it. It leads a process group of its own (a new session), so a kill on the
  worker's error path, or for memory, takes the group: a compound line (`tar -xf ... && python
  ...`, where bash forks) leaves no grandchild running past its attempt. The session changes no
  cgroup: the job stays in its worker's, whose OOM count is read, and Slurm's cgroup process
  tracking kills it with the task (ai's `scontrol show config`, 2026-10-21: ProctrackType
  proctrack/cgroup, TaskPlugin task/cgroup,task/affinity; under proctrack/pgid a new session
  would outlive it).
- **Its memory** (`_Watch`, the owner's "Worker enforces it", 2026-10-22): ai's Slurm writes no
  memory limit (jobs 81808/81809: memory.max, memory.swap.max and memory.high `max` at every
  level of a job's cgroup v2 tree, and a --mem=1G job touched 1.5 GB to rc 0, memory.events oom
  0), so no kernel ever kills a job for passing its class, and the OOM promotion would never fire
  while a job outgrowing its class risked the node's OOM killer taking other users' jobs. So a
  thread of the worker, started after the fork, sums the RSS of every live process of the job's
  session every WATCH_SECONDS (`_members`: the session the job leads, so a compound line's
  grandchildren count; not the cgroup's memory.current, which counts page cache, so the NFS reads
  of a job's inputs would trip it), and once the sum passes the job's limit, SIGKILLs the session
  (`_kill`) and records the measure (`Overrun`). The limit is the class's memory (the GB its
  lease's task asked: pool.py resolves it once from the class, or a long run's --mem) less the
  worker's own RSS read as the job starts, the rest of what the task holds.
- **Its outcome** (`_outcome`): exit 0 is DONE; otherwise the worker's own kill for memory is
  OOM, and so is an increase of the cgroup's OOM-kill count across the child (`OomEvidence`,
  read before and after: a kernel's kill, should Slurm ever enforce the limit), either of which
  promotes the job (pool_queue); anything else is FAILED, a SIGKILL with no such evidence among
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

The cgroup's OOM evidence (`oom_evidence`), the second source beside the worker's own kill, is
resolved from /proc/self/cgroup at each worker's start and logged once: cgroup v2's
`memory.events` (its `oom_kill`), v1's `memory.oom_control` (its `oom_kill`, kernel 4.13 on), or
`Unreadable` with why, every OOM it alone sees then a strike. The victim's own
cgroup counts an OOM kill (the kernel records it on the killed process's memcg), and the job runs
in its worker's, so the worker's cgroup is the one read; ai's is confirmed by a forced OOM in the
smoke waves (the plan's Task 10).
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
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

#: How often the worker sums its job's RSS (`_Watch`). It changes no row: a job's rows are the
#: same whichever class runs it, and only which spikes are caught moves with it (a spike past the
#: limit shorter than a period can be missed: `_outcome`), so it is no parameter of a run's
#: identity. A second bounds a job's overrun to what it allocates in a second; a scan of /proc
#: costs about 6 us a process (4 ms for 660 here), about 1% of a thread on a node of 2000.
WATCH_SECONDS = 1.0

#: The most the worker waits for its watch to end once its job has (s): the watch's one blocking
#: step is a scan of /proc (milliseconds). A watch still running then can kill nothing (`_Watch`).
WATCH_STOP_SECONDS = 10.0

#: Where the job's processes are listed and read.
PROC = Path("/proc")
PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")


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

class Overrun(NamedTuple):
    """The worker's kill of a job for memory: its session's summed RSS when it passed its limit
    (GB of 2**30 bytes)."""

    rss_gb: float
    limit_gb: float

    @property
    def evidence(self) -> str:
        return f"worker: rss {self.rss_gb:.2f} GB > limit {self.limit_gb:.2f} GB"


class Ran(NamedTuple):
    """A job's run. Its peak is the larger of the child's and its waited descendants' maximum RSS
    (`ru_maxrss`: a maximum over processes, at least the worker's own RSS when it forked, as the
    kernel counts the forked image's before the exec, a floor of about 0.1 GB, far below a rung)
    and the largest sum over its session's processes the watch saw (`_Watch`: the measure a class
    bounds, which outlives a shell killed before it waited for its children), GB of 2**30 bytes as
    gap_calib.peak_gb's."""

    rc: int             # the shell's code, -signal when killed
    peak_gb: float
    minutes: float      # its wall
    rows: int           # the row_marker.ROW_WRITTEN lines it printed
    kept: int           # ... and its row_marker.KEPT lines
    overrun: Overrun | None     # the worker killed it for memory


def _last_for_the_oom_killer() -> None:
    """In the child, before it execs: its oom_score_adj OOM_SCORE_ADJ, the kernel's first choice
    for an OOM kill (an unprivileged process may raise its own)."""
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write(str(OOM_SCORE_ADJ))


class Member(NamedTuple):
    """A live process of a job's session."""

    pid: int
    rss_gb: float


def _stat(text: bytes) -> tuple[int, float]:
    """The session and RSS (GB) of a /proc/<pid>/stat: its fields after the command's closing
    parenthesis (a command may hold any byte, `)` and spaces among them), the 6th (session) and
    24th (rss, pages: the counter /proc/<pid>/status shows as VmRSS, anonymous, file and shmem
    pages resident)."""
    fields = text[text.rindex(b")") + 2:].split()
    return int(fields[3]), int(fields[21]) * PAGE_BYTES / 2 ** 30


def own_rss_gb() -> float:
    """The worker's own RSS (GB)."""
    return _stat((PROC / "self" / "stat").read_bytes())[1]


def _members(session: int) -> list[Member]:
    """Every live process of session `session` with its RSS, each from one read of its
    /proc/<pid>/stat (its session and RSS of one instant, never of a pid reused between two
    reads); a process gone between the listing and its read is skipped. A process that left the
    session (setsid) is not counted: no job of reblock's does."""
    found: list[Member] = []
    for name in os.listdir(PROC):
        if not name.isdigit():
            continue
        try:
            with open(PROC / name / "stat", "rb") as f:
                text = f.read()
        except OSError:         # gone since the listing
            continue
        sid, rss_gb = _stat(text)
        if sid == session:
            found.append(Member(pid=int(name), rss_gb=rss_gb))
    return found


def _kill(leader: int) -> None:
    """SIGKILL the job led by `leader`: its process group (its shell and every process the shell
    forked), then any other process of its session (one that left the group). The group is there
    while its leader is unreaped, which it is until `run_job` waits for it; a process gone
    meanwhile is no error."""
    try:
        os.killpg(leader, signal.SIGKILL)
    except ProcessLookupError:
        pass
    for m in _members(leader):
        try:
            os.kill(m.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class Watched(NamedTuple):
    """What a job's watch saw: the largest sum of its session's RSS (GB), its kill, and the error
    it stopped on (the job killed then: never left to run unwatched)."""

    peak_gb: float
    overrun: Overrun | None
    error: BaseException | None


class _Watch:
    """A thread summing the RSS of session `session`'s processes every WATCH_SECONDS (the first
    a period after the job's start), which kills the session (`_kill`) once the sum passes
    `limit_gb` and stops; `stop` ends it, after which it kills nothing (the kill and the stop
    take one lock, so a stop returning means no kill can follow: the leader is unreaped until
    then, so no pid it kills is another's). It is started after the fork, so the child's
    preexec_fn ran beside no thread."""

    def __init__(self, *, session: int, limit_gb: float) -> None:
        self._session, self._limit_gb = session, limit_gb
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._peak_gb = 0.0
        self._overrun: Overrun | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=f"watch-{session}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stopped.wait(WATCH_SECONDS):
                gb = sum(m.rss_gb for m in _members(self._session))
                self._peak_gb = max(self._peak_gb, gb)
                if gb > self._limit_gb:
                    with self._lock:
                        if not self._stopped.is_set():
                            _kill(self._session)
                            self._overrun = Overrun(rss_gb=gb, limit_gb=self._limit_gb)
                    return
        except Exception as e:      # the worker raises it once the job is reaped
            with self._lock:
                if not self._stopped.is_set():
                    _kill(self._session)
                self._error = e

    def stop(self) -> Watched:
        """Stop watching (never waiting past WATCH_STOP_SECONDS): what it saw."""
        with self._lock:
            self._stopped.set()
        self._thread.join(WATCH_STOP_SECONDS)
        with self._lock:
            return Watched(peak_gb=self._peak_gb, overrun=self._overrun, error=self._error)


def run_job(line: str, *, cwd: Path, env: Mapping[str, str], memory_gb: float,
            log: Callable[[str], None], wrote: Callable[[], None],
            kept: Callable[[], None]) -> Ran:
    """`bash -c <line>` in `cwd` with `env`, at oom_score_adj OOM_SCORE_ADJ, leading a session
    and process group of its own, its output (stdout and stderr merged, undecodable bytes
    replaced) given to `log` line by line as it comes, `wrote()` called for each line starting
    with row_marker.ROW_WRITTEN and `kept()` for each starting with row_marker.KEPT; watched
    (`_Watch`) against `memory_gb` (its class's memory) less the worker's own RSS now; its code
    from `os.wait4` once it has exited, and its peak (Ran's). If reading it raises, its session
    is killed and the child reaped before the error goes on, so no process of the job outlives
    its worker's attempt; an error of the watch's is raised once the job is reaped."""
    own_gb = own_rss_gb()
    limit_gb = memory_gb - own_gb
    if limit_gb <= 0:
        raise ValueError(f"a class of {memory_gb:g} GB holds no job beside its worker's own "
                         f"{own_gb:.2f} GB")
    start = time.monotonic()
    # preexec_fn runs between fork and exec, unsafe only beside other threads: the worker has
    # none at the fork (its watch starts after); start_new_session's setsid runs before it, so
    # the job leads its group and session from its exec
    proc = subprocess.Popen(["bash", "-c", line], cwd=cwd, env=dict(env), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1, start_new_session=True,
                            preexec_fn=_last_for_the_oom_killer)
    if proc.stdout is None:
        raise RuntimeError("Popen gave no stdout pipe")
    watch = _Watch(session=proc.pid, limit_gb=limit_gb)
    watch.start()
    rows = held = 0
    try:
        for out in proc.stdout:
            text = out.rstrip("\n")
            log(text)
            if text.startswith(row_marker.ROW_WRITTEN):
                rows += 1
                wrote()
            elif text.startswith(row_marker.KEPT):
                held += 1
                kept()
    except BaseException:
        _kill(proc.pid)
        watch.stop()        # before the reap: no kill of the watch's after it
        proc.wait()         # Popen's: an already reaped child is no new error over this one
        raise
    finally:
        proc.stdout.close()
    # its output may end before it does (a job writing to a file): exited, not yet reaped, so its
    # pid and group stay its own while the watch stops
    os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
    watched = watch.stop()
    _, status, usage = os.wait4(proc.pid, 0)
    proc.returncode = os.waitstatus_to_exitcode(status)     # so Popen never waits again
    if watched.error is not None:
        raise watched.error
    return Ran(rc=proc.returncode, peak_gb=max(usage.ru_maxrss / 2 ** 20, watched.peak_gb),
               minutes=(time.monotonic() - start) / 60, rows=rows, kept=held,
               overrun=watched.overrun)


def _outcome(rc: int, *, overrun: Overrun | None, before: int | None,
             after: int | None) -> Outcome:
    """A job's outcome from its code, the worker's kill for memory and the cgroup's OOM-kill count
    before and after it. The worker's kill is evidence enough: it measured the job past its limit
    and killed it itself, while no kernel did (the count unchanged on ai, where Slurm sets no
    limit); with none, the count's increase is a kernel's OOM kill, should Slurm enforce one. A
    job exiting 0 is DONE whatever the evidence (a kill racing its own end: its work is whole).
    A spike past the limit shorter than WATCH_SECONDS, which the watch missed, leaves the job
    DONE too: its attempt records the peak (os.wait4's, at least), so the next wave's history
    classes it higher (pool_plan); no outcome of its own."""
    if rc == 0:
        return Outcome.DONE
    if overrun is not None:
        return Outcome.OOM
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
    outcome = _outcome(ran.rc, overrun=ran.overrun, before=before, after=after)
    state = queue.finish(c.attempt, worker=worker, outcome=outcome, peak_gb=ran.peak_gb,
                         minutes=ran.minutes, rc=ran.rc)
    log(f"=== job {j.id} {outcome} rc {ran.rc} {ran.minutes:.2f} min peak {ran.peak_gb:.2f} GB "
        f"rows {ran.rows} kept {ran.kept}")
    if ran.overrun is not None:
        log(f"job {j.id}: killed for memory ({ran.overrun.evidence})")
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
