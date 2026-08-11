"""Run a module under instrumentation, to find out HOW a long background run is being killed.

**It worked, and the answer is: the agent harness kills them.** Five runs had been stopped
mid-flight (C9 at 2/10, C20 at 2/12 and 5/12, the first region run at 73 min, and the six-region
replication at 57 min) with the machine ruled out -- the cgroup reports `oom_kill 0` with no memory
limit and pressure-stall is flat (`avg10=0.00`) for memory, CPU and IO. That evidence was read as
"the kill is external". It is not external at all. On 2026-08-11 this wrapper caught one and the
`siginfo` named the sender:

    !! SIGNAL SIGTERM (15) si_code=0 si_uid=1641171234
       FROM pid=1638454 comm='claude' ppid=2872 cmdline='claude'

`si_code=0` is `SI_USER` -- a deliberate `kill()` -- and `comm='claude'` is the Claude Code process
running the session. An earlier 103-minute run survived while this one died at 57, so it is not a
plain wall-clock timeout and the exact trigger is still unidentified; the *source* is not in
question.

**The remedy is not this tool, it is `setsid`.** Launch long work into its own session so harness
signals cannot reach it, and make the job resumable anyway:

    setsid nohup pixi run python -u -m scripts.perf.<name> > log 2>&1 < /dev/null &

Detaching costs the harness's completion notification, so poll the log or the output file. Keep
running long jobs under this wrapper regardless -- if something else ever kills one, this is what
will say what.

## What it can and cannot see

* **Catchable signals** (TERM, HUP, INT, QUIT, USR1, USR2) are caught via `signal.sigwaitinfo` in a
  dedicated thread rather than an ordinary handler, because `sigwaitinfo` returns a `siginfo`
  carrying **`si_pid` and `si_uid` -- the sender**. An ordinary Python handler cannot see that. The
  sender's `/proc/<pid>/cmdline` is read immediately, before it can exit.
* **SIGKILL cannot be caught by anything**, by design. What identifies it is the wrapper's exit
  status: 128+N, so 137 = SIGKILL and 143 = SIGTERM. "Something chose to hard-kill rather than ask
  politely" is itself a useful distinction -- a supervisor enforcing a limit behaves differently
  from a user pressing stop.
* A **heartbeat** line every `HEARTBEAT_S` records wall time, RSS, and the PARENT pid. If the parent
  changes to 1, the run was orphaned (its parent died) rather than signalled, which would point at
  the shell/session rather than at the job.

## The fork-pool hazard

`pthread_sigmask` is inherited across `fork`, and `greedy_arterial_*` forks a 16-worker pool. Left
alone, those workers would inherit a mask blocking SIGTERM and could not be terminated -- exactly
the orphaned-worker failure this repo has hit before. `os.register_at_fork(after_in_child=...)`
restores the default mask in every child, so only the parent is instrumented.

Usage:  pixi run python -m scripts.perf.instrumented <module> <logfile>
"""
from __future__ import annotations

import faulthandler
import importlib
import os
import signal
import sys
import threading
import time
from pathlib import Path

WATCHED = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGQUIT,
           signal.SIGUSR1, signal.SIGUSR2, signal.SIGXCPU, signal.SIGPIPE)
HEARTBEAT_S = 20.0


def _log(path: Path, msg: str) -> None:
    with path.open("a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        fh.flush()
        os.fsync(fh.fileno())          # survive a hard kill


def _describe(pid: int) -> str:
    if pid <= 0:
        return "(kernel or unknown)"
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().strip()
        comm = Path(f"/proc/{pid}/comm").read_text().strip()
        ppid = ""
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                ppid = line.split()[1]
        return f"pid={pid} comm={comm!r} ppid={ppid} cmdline={cmd[:200]!r}"
    except OSError:
        return f"pid={pid} (already gone)"


def _signal_watcher(path: Path) -> None:
    while True:
        info = signal.sigwaitinfo(set(WATCHED))
        name = signal.Signals(info.si_signo).name
        _log(path, f"!! SIGNAL {name} ({info.si_signo}) si_code={info.si_code} "
                   f"si_uid={info.si_uid} FROM {_describe(info.si_pid)}")
        faulthandler.dump_traceback(file=sys.stderr)
        if info.si_signo in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT, signal.SIGHUP):
            _log(path, f"   exiting on {name}")
            os._exit(128 + info.si_signo)


def _heartbeat(path: Path) -> None:
    t0 = time.monotonic()
    while True:
        try:
            rss_kb = int(Path("/proc/self/statm").read_text().split()[1]) * (
                os.sysconf("SC_PAGE_SIZE") // 1024)
        except (OSError, ValueError):
            rss_kb = -1
        _log(path, f".. alive {time.monotonic() - t0:7.0f}s  rss={rss_kb / 1024:8.0f} MB  "
                   f"ppid={os.getppid()}  threads={threading.active_count()}")
        time.sleep(HEARTBEAT_S)


def main() -> int:
    module, logfile = sys.argv[1], Path(sys.argv[2])
    logfile.parent.mkdir(parents=True, exist_ok=True)
    _log(logfile, f"== start pid={os.getpid()} ppid={os.getppid()} pgid={os.getpgid(0)} "
                  f"sid={os.getsid(0)} module={module}")
    _log(logfile, f"   parent is {_describe(os.getppid())}")

    signal.pthread_sigmask(signal.SIG_BLOCK, set(WATCHED))
    # forked children (the arterial worker pool) must NOT inherit the block, or they become
    # unkillable orphans -- a failure this repo has hit before
    os.register_at_fork(after_in_child=lambda: signal.pthread_sigmask(
        signal.SIG_UNBLOCK, set(WATCHED)))
    threading.Thread(target=_signal_watcher, args=(logfile,), daemon=True).start()
    threading.Thread(target=_heartbeat, args=(logfile,), daemon=True).start()

    try:
        importlib.import_module(module).main()
    except BaseException as exc:                                      # noqa: BLE001
        _log(logfile, f"== raised {type(exc).__name__}: {exc}")
        raise
    _log(logfile, "== finished normally")
    return 0


if __name__ == "__main__":
    sys.exit(main())
