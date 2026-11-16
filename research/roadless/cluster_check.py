"""Guards on cluster.py's kinds, without a submit: each plans a run from a command line, routes and
scripts it as cluster_submit's submit does (the plan a dry submit prints and the sbatch it
stages), and checks what a task would get.

    C=research/roadless/cluster_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $C cpu <ids,> <items,>    # a print, checked by eye
    uv run python $C ids [fault]            # fault: no-check
    uv run python $C script [fault]         # fault: no-script-check
    uv run python $C cut [fault]            # fault: resumable
    uv run python $C both-ways <guards,>    # each guard, then under each of its faults

Each guard but `cpu` has its faults (FAULTS: a patch in this process, no file edited, as
gap_clear_check.py's): it passes on the code as it is, and must fail under each fault; run alone
it exits 1 when it fails as it is, and under a fault exits 0 when it fails (the fault seen) and
names the fault when it does not. `both-ways` runs each guard as it is and under each of its
faults, in one process, and exits 1 naming every run not as expected.

`cpu`: a `cpu` run on each node of the site (any of them: a CPU run's node is any the site file
names), at a throttle of its own (CPU_THROTTLES: uncapped, room is admission's), is one CPU array
pinned there (`--nodelist`), and one without `--node` one CPU array placed by cluster_submit at
admission (cs.Placed: no `--nodelist`); each with no GPU request, the given cores and memory,
every (block, item) one task that unpacks the block's tar through cluster_unpack.py and runs the
command with both placeholders filled. A `--node` the site file lacks is refused. The tars are empty
stand-ins (a plan checks only that they exist). The command is gap_calib.py's `preset`, a
(block, item) task.

`ids` (phase 3: an unknown block id raises at the boundary): phase 2b's misspelt id (MISTYPED for
SPELLED: the source holds it, under its building floor, and yielded nothing for it) is refused by
the parser every entry point reads its ids with (common.ids_of), by the bank's writer
(common.write_bank), by a `blocks` submit's plan before any task is staged, and by build_blocks
reading a bank that lacks it; the id spelt right parses and plans. Fault `no-check`: the check
removed (common._known a no-op): the parser and the bank's writer let the mistyped id through,
the bank written without it, as in phase 2b.

`script` (a plan reads the repo at its commit, cluster_submit's README): a plan of either kind
reads its command's script at the run's commit (Kind.plan's sha), from git, never the working
tree. A script the working tree holds and the commit lacks (gap_calib.py at the repo's first
commit) is refused naming the commit; one the commit holds and the working tree lacks (GONE, at
the commit before the one that deleted it) plans. Fault `no-script-check`: cluster._script_at a
no-op: the absent script plans.

`cut` (cluster_submit's Kind.restart_hours, its debt spec's section 3): every kind's work is lost
whole when cut (restart_hours math.inf), so a run's walltime is its own `--time` on every node,
however short the node's length L* is: for each kind's restart as admission reads it
(room.TestedGate: Kind.restart_hours by the kind's name), at each --time of TIMES, on each node
at the spec's seeded length (SEEDED_LENGTHS) and on one where nothing bounds a job, the walltime
harm.json gives (harm.Published.walltime) is the run's own. A reblock job is one task (one array
element runs one line), and a task killed at its walltime loses all its work: no later job
continues it, so a length shorter than the task would kill it again at every resubmit. Fault
`resumable`: every kind's restart the records' median task (MEDIAN_TASK_HOURS, the spec's "a
restart of the task's length"): L* cuts the runs, and the guard sees it.

(`gpu`, the retired raster's sizing checked against its scorer's measured peaks, went with it:
NOTES, "The complex as the runtime: the raster retired".)
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import cluster  # noqa: E402
import cluster_submit as cs  # noqa: E402
from cluster_submit import RunPlan  # noqa: E402
from cluster_submit.arrays import Array, route, script  # noqa: E402
from cluster_submit.cli import parser  # noqa: E402
from cluster_submit.harm import NO_BOOK, VICTIMS, Length, Published, Standing  # noqa: E402
from cluster_submit.remote import SubprocessRunner  # noqa: E402
from cluster_submit.slurm import time_seconds  # noqa: E402
from cluster_submit.submit import throttle_checked  # noqa: E402

PROJECT = cluster.PROJECT
COMMAND = ["--", "research/roadless/gap_calib.py", "preset", "{id}", "{item}", "cpu"]
MISTYPED = "ZAF.9.3.1_1_7851"   # ids: phase 2b's misspelling of SPELLED in a device-guard submit
SPELLED = "ZAF.9.5.4_1_7851"
# The commit a plan reads its script at (Kind.plan's sha): HEAD, as a submit's clean HEAD.
def _rev(rev: str) -> str:
    return subprocess.run(["git", "-C", str(PROJECT.repo), "rev-parse", rev], check=True,
                          capture_output=True, text=True).stdout.strip()


SHA = _rev("HEAD")
# `script`: a script the history deleted, and the commit that deleted it.
GONE, GONE_BY = "research/roadless/fft_homog/bench.py", "2c4f57d007a89269fc63c220be68fa423a5b7223"
# `cut`: each node's L* at a 6-minute restart as GeoffChurch/cluster_submit's debt spec seeded it
# (docs/superpowers/specs/2026-11-15-debt-and-short-jobs.md, section 3, "Illustrative": decayed to
# 2026-11-14T20:40Z), in hours; at a restart of s hours it is that times sqrt(s / 0.1).
SEEDED_LENGTHS = {"h100": 0.6, "tesla1": 1.1, "tesla2": 1.3, "quadro1": 1.6, "quadro2": 2.0}
SEEDED_RESTART_HOURS = 0.1
# `cut`: the run's own --time, from a short one to the longest the records ask (cp-roads4-desire-
# g3a's 7 days); 23:30:00 is the site's default (ai.toml's default_time) and the commonest asked.
TIMES = ("2:00:00", "23:30:00", "7-00:00:00")
# `cut`'s fault: the median completed task of the `blocks` and `cpu` runs in reblock's run records
# (.cluster_submit/runs/*/sacct.txt, 401 runs submitted 2026-10-03 to 2026-10-28: 12381 tasks,
# median 2.27 minutes), as the restart a cut costs.
MEDIAN_TASK_HOURS = 2.27 / 60


def _parsed(argv: list[str]) -> argparse.Namespace:
    return parser(PROJECT).parse_args(argv)


def _plan(argv: list[str], workdir: Path, *, sha: str
          ) -> tuple[argparse.Namespace, RunPlan, tuple[Array, ...]]:
    """The kind's plan for `argv` at commit `sha`, routed and checked as submit does
    (cluster_submit's route and throttle_checked): (args, plan, arrays)."""
    args = _parsed(argv)
    kind = {k.name: k for k in PROJECT.kinds}[args.kind]
    workdir.mkdir(parents=True, exist_ok=True)
    plan = kind.plan(args, workdir, sha=sha)
    arrays, too_big = route(PROJECT.cluster, plan.tasks, placement=plan.resources.placement)
    if too_big:
        raise SystemExit(f"{len(too_big)} tasks too big for any card")
    throttle_checked(PROJECT, arrays, args.gpus)
    return args, plan, arrays


def _refused(argv: list[str], workdir: Path, *, sha: str) -> str:
    """The refusal `argv` meets (argparse's or a plan's), or raise if it plans."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            _plan(argv, workdir, sha=sha)
    except SystemExit as e:
        return f"{e.code}: {err.getvalue().strip().splitlines()[-1] if err.getvalue() else ''}"
    raise AssertionError(f"planned, not refused: {argv}")


#: `cpu`'s throttles, taken in turn over the site's nodes: a CPU run's `--gpus` is its own, uncapped
#: by the project's cards (room is admission's), so one past what a node's cores hold plans too.
CPU_THROTTLES = (36, 64, 200)


def check_cpu(ids: list[str], items: list[str]) -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for b in ids:
            with tarfile.open(tmp / f"{b}.tar", "w"):
                pass
        common = ["--cpus", "1", "--mem", "4", "--time", "2:00:00", "--items", ",".join(items),
                  "--unpack", str(tmp / "{id}.tar"), ",".join(ids), *COMMAND]
        lines = [f'python -u research/roadless/cluster_unpack.py "$CLUSTER_SUBMIT_INPUTS/'
                 f'{b}.tar" && python -u research/roadless/gap_calib.py preset {b} {item} '
                 'cpu' for b in ids for item in items]
        env = cluster.Cpu().env.at(SubprocessRunner(), PROJECT.repo, SHA, root=PROJECT.root())
        # every node of the site, each at a throttle of its own (a CPU run's is uncapped: room is
        # admission's), and unpinned, placed by cluster_submit at admission
        nodes: list[str | None] = [n.name for n in PROJECT.cluster.nodes]
        for i, node in enumerate([*nodes, None]):
            at_once = CPU_THROTTLES[i % len(CPU_THROTTLES)]
            pin = [] if node is None else ["--node", node]
            argv = ["submit", "cpu", "guard", "--dry", *pin, "--gpus", str(at_once), *common]
            args, plan, arrays = _plan(argv, tmp / "work", sha=SHA)
            [a] = arrays
            text = script(PROJECT, "guard", SHA, a, plan.resources, plan.prelude, args.gpus, env,
                          named="guard")
            head = [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
            want = {"#SBATCH --cpus-per-task=1", "#SBATCH --mem=4G",
                    f"#SBATCH --array=0-{len(ids) * len(items) - 1}%{at_once}"}
            listed = [ln for ln in head if ln.startswith("#SBATCH --nodelist")]
            gpu = [ln for ln in head if "gres" in ln or "--gpus" in ln]
            where = "placed" if node is None else node
            assert a.card_gb == 0 and a.node == node, f"{where}: {a.label} array on {a.node}"
            assert plan.resources.placement.pinned == node, f"{where}: {plan.resources.placement}"
            assert listed == ([] if node is None else [f"#SBATCH --nodelist={node}"]), \
                f"{where}: {listed}"
            assert want <= set(head), f"{where}: header lacks {want - set(head)}"
            assert not gpu, f"{where}: a GPU request in a CPU array: {gpu}"
            assert list(a.tasks) == lines, f"{where}: task lines {a.tasks[:2]}"
            assert all(t.cpu for t in plan.tasks)
            print(f"{where}: one CPU array of {len(a.tasks)} tasks "
                  f"{plan.resources.placement.text()}, at most {at_once} at once, 1 cpu and 4 GB "
                  "a task, no GPU request")
        off_site = ["submit", "cpu", "guard", "--dry", "--node", "nosuchnode", "--gpus", "4",
                    *common]
        print(f"--node off the site refused: {_refused(off_site, tmp / 'work', sha=SHA)}")


def _refuses(what: str, fn: Callable[[], object]) -> str | None:
    """None if calling `fn` raises ValueError naming MISTYPED (or the bank's own KeyError), else
    the problem."""
    try:
        got = fn()
    except ValueError as e:
        return None if MISTYPED in str(e) else f"{what} raised without naming {MISTYPED}: {e}"
    except KeyError:
        return None
    return f"{what} accepted {MISTYPED}: {got!r}"[:300]


def check_ids() -> list[str]:
    """`ids` (the module's docstring): the problems found."""
    import common
    both = f"{SPELLED},{MISTYPED}"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        argv = ["submit", "blocks", "guard", "--dry", "--gb", "8", both, "--",
                "research/roadless/gap_calib.py", "warm", "{id}"]
        bad = [p for p in (
            _refuses("common.ids_of", lambda: common.ids_of(both)),
            _refuses("common.write_bank",
                     lambda: sorted(common.write_bank([SPELLED, MISTYPED], tmp / "bank.pkl"))),
            _refuses("a blocks submit's plan", lambda: _plan(argv, tmp / "work", sha=SHA)[1].tasks),
        ) if p is not None]
        common.write_bank([SPELLED], tmp / "one.pkl")
        old = os.environ.get(common.BANK_ENV)
        os.environ[common.BANK_ENV] = str(tmp / "one.pkl")
        try:
            p = _refuses("build_blocks on a bank without it",
                         lambda: [b.block_id for b in common.build_blocks([MISTYPED])])
            bad += [] if p is None else [p]
            if [b.block_id for b in common.build_blocks([SPELLED])] != [SPELLED]:
                bad.append(f"build_blocks did not read {SPELLED} from the bank")
        finally:
            if old is None:
                del os.environ[common.BANK_ENV]
            else:
                os.environ[common.BANK_ENV] = old
        _args, plan, _arrays = _plan([*argv[:6], SPELLED, *argv[7:]], tmp / "work2", sha=SHA)
        if common.ids_of(SPELLED) != [SPELLED] or len(plan.tasks) != 1:
            bad.append(f"{SPELLED} does not parse and plan as one task")
    print(f"ids: {MISTYPED} (for {SPELLED}) refused by common.ids_of, common.write_bank, a blocks "
          f"submit's plan and build_blocks on a bank -> {'ok' if not bad else 'FAILS'}",
          flush=True)
    return bad


def check_script() -> list[str]:
    """`script` (the module's docstring): the problems found."""
    absent, gone_at = "research/roadless/gap_calib.py", _rev(f"{GONE_BY}^")
    first = subprocess.run(["git", "-C", str(PROJECT.repo), "rev-list", "--max-parents=0",
                            "HEAD"], check=True, capture_output=True, text=True).stdout.split()[-1]
    assert (PROJECT.repo / absent).is_file() and not (PROJECT.repo / GONE).exists()
    bad: list[str] = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for kind, opts in (("blocks", ["--gb", "8"]),
                           ("cpu", ["--node", "orion", "--cpus", "1", "--mem", "4"])):
            head = ["submit", kind, "guard", "--dry", *opts, "--items", "x", SPELLED, "--"]
            try:
                why = _refused([*head, absent, "{id}", "{item}"], tmp / "work", sha=first)
                if first[:12] not in why:
                    bad.append(f"{kind}: {absent} at {first[:12]} refused without naming the "
                               f"commit: {why}")
            except AssertionError:
                bad.append(f"{kind}: {absent} planned at {first[:12]}, which lacks it")
            try:
                _plan([*head, GONE, "{id}", "{item}"], tmp / "work", sha=gone_at)
            except SystemExit as e:
                bad.append(f"{kind}: {GONE} refused at {gone_at[:12]}, which holds it: {e}")
    print(f"script: a script read at the run's commit by both kinds -> "
          f"{'ok' if not bad else 'FAILS'}", flush=True)
    return bad


# `cut`: a node of harm.json where nothing bounds a job (no arrival there: Length.hours None).
UNBOUNDED = "unbounded"


def _seeded() -> Published:
    """harm.json in active mode with each node's seeded length (SEEDED_LENGTHS, at a restart of
    1 h) and UNBOUNDED's none; nothing else in it is read by a walltime."""
    at = "2026-11-14T20:40:00Z"
    lengths = {n: Length(None, None, x / math.sqrt(SEEDED_RESTART_HOURS), 0.0, 0.0)
               for n, x in SEEDED_LENGTHS.items()}
    lengths[UNBOUNDED] = Length(None, None, None, 0.0, 0.0)
    return Published(at=at, mode="active", trust_hours=6.0, stale_after_minutes=30.0,
                     user="reblock", usable={}, usage={}, debt={}, debt_weight=1.0,
                     lengths=lengths, budget_days=28.0,
                     standings=tuple(Standing(victims=v, limit=1.0, at=at, accounts={},
                                              projects={}, over=None, short={})
                                     for v in VICTIMS),
                     book=NO_BOOK)


def check_cut() -> list[str]:
    """`cut` (the module's docstring): the problems found."""
    # as admission reads them (room.TestedGate: Kind.restart_hours by the kind's name)
    restarts = {k.name: k.restart_hours for k in PROJECT.kinds}
    published = _seeded()
    bad: list[str] = []
    for kind, s in restarts.items():
        for t in TIMES:
            own = timedelta(seconds=time_seconds(t))
            for node in (*SEEDED_LENGTHS, UNBOUNDED):
                got = published.walltime(node, restart_hours=s, maximum=own)
                if got != own:
                    bad.append(f"{kind} at --time {t} on {node}: walltime {got}, short of its own")
    print(f"cut: {', '.join(sorted(restarts))} keep their own --time on every node "
          f"({', '.join(SEEDED_LENGTHS)}, {UNBOUNDED}) -> {'ok' if not bad else 'FAILS'}",
          flush=True)
    return bad


@contextlib.contextmanager
def _resumable() -> Iterator[None]:
    """Every kind's restart the records' median task (fault `resumable`)."""
    global PROJECT
    old = PROJECT

    def resumable(k: cs.Kind) -> cs.Kind:
        assert dataclasses.is_dataclass(k) and not isinstance(k, type), k   # cluster.py's kinds
        return dataclasses.replace(k, restart_hours=MEDIAN_TASK_HOURS)

    PROJECT = dataclasses.replace(old, kinds=tuple(resumable(k) for k in old.kinds))
    try:
        yield
    finally:
        PROJECT = old


@contextlib.contextmanager
def _no_script_check() -> Iterator[None]:
    """cluster._script_at a no-op (fault `no-script-check`)."""
    old = cluster._script_at
    cluster._script_at = lambda args, sha: None
    try:
        yield
    finally:
        cluster._script_at = old


@contextlib.contextmanager
def _no_check() -> Iterator[None]:
    """common._known a no-op (fault `no-check`)."""
    import common
    old = common._known
    common._known = lambda ids, have, *, where: None
    try:
        yield
    finally:
        common._known = old


GUARDS: dict[str, Callable[[], list[str]]] = {"ids": check_ids, "script": check_script,
                                               "cut": check_cut}
FAULTS: dict[str, dict[str, Callable[[], contextlib.AbstractContextManager[None]]]] = {
    "ids": {"no-check": _no_check},
    "script": {"no-script-check": _no_script_check},
    "cut": {"resumable": _resumable}}


# ---- the command line -----------------------------------------------------------------------

def main(guard: str, fault: str | None) -> None:
    if fault is None:
        bad = GUARDS[guard]()
    else:
        with FAULTS[guard][fault]():
            bad = GUARDS[guard]()
    verdict = "FAILS on " + "; ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


def both_ways(guards: list[str]) -> None:
    """Each guard as it is, then under each of its faults (FAULTS), one after another in this
    process. Raises naming every run not as `main` expects."""
    wrong = []
    for guard in guards:
        for fault in (None, *FAULTS[guard]):
            try:
                main(guard, fault)
            except SystemExit as e:
                wrong.append(f"{guard}{'' if fault is None else f' under {fault}'}: {e.code}")
    print(f"both ways, {', '.join(guards)}: "
          + ("as expected" if not wrong else "NOT as expected: " + "; ".join(wrong)), flush=True)
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "cpu":
        check_cpu(a[1].split(","), a[2].split(","))
        raise SystemExit(0)
    if len(a) == 2 and a[0] == "both-ways" and set(a[1].split(",")) <= set(GUARDS):
        both_ways(a[1].split(","))
        raise SystemExit(0)
    if len(a) not in (1, 2) or a[0] not in GUARDS or (len(a) == 2 and a[1] not in FAULTS[a[0]]):
        raise SystemExit(__doc__)
    main(a[0], a[1] if len(a) == 2 else None)
