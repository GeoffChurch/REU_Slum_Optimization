"""Guards on cluster.py's kinds, without a submit: each plans a run from a command line, routes and
scripts it as cluster_submit's submit does (the plan a dry submit prints and the sbatch it
stages), and checks what a task would get.

    PYTHONPATH=. uv run python research/roadless/cluster_check.py cpu <ids,> <items,>
    PYTHONPATH=. uv run python research/roadless/cluster_check.py ids [no-check]
    PYTHONPATH=. uv run python research/roadless/cluster_check.py script [no-script-check]
    PYTHONPATH=. uv run python research/roadless/cluster_check.py demand \
        [finished-named|no-throttle|any-reason|no-running]

`cpu`: a `cpu` run on each node of CPU_NODES at its own throttle is one CPU array pinned there
(`--nodelist`), with no GPU request, the given cores and memory, every (block, item) one task that
unpacks the block's tar through cluster_unpack.py and runs the command with both placeholders
filled; a run without `--node` is refused, and one past the node's throttle. The tars are empty
stand-ins (a plan checks only that they exist). The command is gap_calib.py's `preset`, a
(block, item) task.

`ids` (phase 3: an unknown block id raises at the boundary): phase 2b's misspelt id (MISTYPED for
SPELLED: the source holds it, under its building floor, and yielded nothing for it) is refused by
the parser every entry point reads its ids with (common.ids_of), by the bank's writer
(common.write_bank), by a `blocks` submit's plan before any task is staged, and by build_blocks
reading a bank that lacks it; the id spelt right parses and plans. Fault `no-check`: the check
removed (common._known a no-op): the parser and the bank's writer let the mistyped id through,
the bank written without it, as in phase 2b. It follows gap_clear_check.py's convention: exit 0
when the guard passes, or fails under its fault.

`script` (a plan reads the repo at its commit, cluster_submit's README): a plan of either kind
reads its command's script at the run's commit (Kind.plan's sha), from git, never the working
tree. A script the working tree holds and the commit lacks (gap_calib.py at the repo's first
commit) is refused naming the commit; one the commit holds and the working tree lacks (GONE, at
the commit before the one that deleted it) plans. Fault `no-script-check`: cluster._script_at a
no-op: the absent script plans. Exit as for `ids`.

`demand` (cluster_demand: reblock's one campaign in the GPU market): its demand from a made-up
squeue and seven made-up runs of this registry, and the board script run on a scratch board. Held
units: every running task of the project's, cp-roads4-desire-g3b's as it stood on 2026-10-21
among them (throttle 2, one running, three waiting, their memory changed in Slurm), a cancelled
run's and another checkout's run's too, aggregated by (node, bundle) at the mean of their time
limits; waiting units: those each run's throttle starts now (one of g3b's), none held by a
dependency or a begin time, none of a cancelled run; an unpinned run's waiting tasks stated
nowhere and counted, another checkout's left to it and counted; its runs every run squeue lists
of the project's and every unfinished one of the registry, not a finished one, another user's,
another project's, or a task that left squeue between the two reads. Faults: `finished-named`
(cluster_demand.finished never true: a finished run named), `no-throttle` (cluster_demand.room
unbounded: units past a run's throttle), `any-reason` (cluster_demand.waits_for always true: a
task held by its dependency counted), `no-running` (cluster_demand.held's units dropped: what the
waves hold stated as units 0). Exit as for `ids`.

(`gpu`, the retired raster's sizing checked against its scorer's measured peaks, went with it:
NOTES, "The complex as the runtime: the raster retired".)
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import cluster  # noqa: E402
import cluster_demand  # noqa: E402
from cluster_submit import Bundle, RunPlan  # noqa: E402
from cluster_submit.arrays import Array, route, script  # noqa: E402
from cluster_submit.cli import parser  # noqa: E402
from cluster_submit.gpu import DemandLine, demand_path, read_demand  # noqa: E402
from cluster_submit.gpuboard import Pending, Running, Seen  # noqa: E402
from cluster_submit.registry import NO_OVERRIDES, ArrayRecord, RunRecord  # noqa: E402
from cluster_submit.remote import Local, SubprocessRunner  # noqa: E402
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
    arrays, too_big = route(PROJECT.cluster, plan.tasks, node=plan.resources.node)
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


def check_cpu(ids: list[str], items: list[str]) -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        for b in ids:
            with tarfile.open(tmp / f"{b}.tar", "w"):
                pass
        common = ["--cpus", "1", "--mem", "4", "--time", "2:00:00", "--items", ",".join(items),
                  "--unpack", str(tmp / "{id}.tar"), ",".join(ids), *COMMAND]
        for node, at_once in cluster.CPU_NODES.items():
            argv = ["submit", "cpu", "guard", "--dry", "--node", node, "--gpus", str(at_once),
                    *common]
            args, plan, arrays = _plan(argv, tmp / "work", sha=SHA)
            [a] = arrays
            env = cluster.Cpu().env.at(SubprocessRunner(), PROJECT.repo, SHA,
                                       root=PROJECT.root())
            text = script(PROJECT, "guard", SHA, a, plan.resources, plan.prelude, args.gpus, env)
            head = [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]
            want = {f"#SBATCH --nodelist={node}", "#SBATCH --cpus-per-task=1",
                    "#SBATCH --mem=4G", f"#SBATCH --array=0-{len(ids) * len(items) - 1}%{at_once}"}
            gpu = [ln for ln in head if "gres" in ln or "--gpus" in ln]
            lines = [f'python -u research/roadless/cluster_unpack.py "$CLUSTER_SUBMIT_INPUTS/'
                     f'{b}.tar" && python -u research/roadless/gap_calib.py preset {b} {item} '
                     'cpu' for b in ids for item in items]
            assert a.card_gb == 0 and a.node == node, f"{node}: {a.label} array on {a.node}"
            assert want <= set(head), f"{node}: header lacks {want - set(head)}"
            assert not gpu, f"{node}: a GPU request in a CPU array: {gpu}"
            assert list(a.tasks) == lines, f"{node}: task lines {a.tasks[:2]}"
            assert plan.resources.node == node and all(t.cpu for t in plan.tasks)
            print(f"{node}: one CPU array of {len(a.tasks)} tasks pinned there, at most "
                  f"{at_once} at once, 1 cpu and 4 GB a task, no GPU request")
            over = ["submit", "cpu", "guard", "--dry", "--node", node, "--gpus", str(at_once + 1),
                    *common]
            print(f"  --gpus {at_once + 1} refused: {_refused(over, tmp / 'work', sha=SHA)}")
        bare = ["submit", "cpu", "guard", "--dry", "--gpus", "4", *common]
        print(f"without --node refused: {_refused(bare, tmp / 'work', sha=SHA)}")


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


def _record(name: str, kind: str, job: str, *, gpus: int, finished: bool,
            cancelled: bool) -> RunRecord:
    """A made-up run's record, as a submit of `kind` writes it: one array, job `job`, at most
    `gpus` tasks at once."""
    opts = (["--gb", "32"] if kind == "blocks"
            else ["--node", "orion", "--cpus", "1", "--mem", "540", "--items", "x"])
    argv = ("submit", kind, name, *opts, "--gpus", str(gpus), "a,b", "--",
            "research/roadless/road_rescore.py", "{id}", "{item}")
    stamp = "2026-10-21T12:00:00Z"
    return RunRecord(name=name, kind=kind, commit=SHA, argv=argv, overrides=NO_OVERRIDES,
                     submitted_at=stamp, attempt=f"guard:{name}",
                     arrays=(ArrayRecord(job, 32 if kind == "blocks" else 0, ("t",) * 4),),
                     gpus=gpus, chained_after=(), last_sync=stamp,
                     gone_at=stamp if finished else None, finished_at=stamp if finished else None,
                     cancelled_at=stamp if cancelled else None, unpulled=(), unstarted=(),
                     continues=None, continued_by=None, settled=False)


def check_demand() -> list[str]:
    """`demand` (the module's docstring): the problems found."""
    def cpu(gb: float, cpus: int = 1) -> Bundle:
        return Bundle(cards=0, cpus=cpus, mem_gb=gb)

    def name(run: str, label: str = "cpu", project: str = "reblock") -> str:
        return f"{project}-{run}-{label}"

    def run(job: str, run: str, node: str, b: Bundle, *, user: str = "me",
            label: str = "cpu", project: str = "reblock") -> Running:
        return Running(job, user, (node,), b, datetime(2026, 10, 21, tzinfo=UTC),
                       name(run, label, project))

    def wait(job: str, run: str, b: Bundle, reason: str = "Resources",
             node: str | None = "orion", label: str = "cpu") -> Pending:
        return Pending(job, "me", (node,) if node else (), () if node else ("orion", "quadro2"),
                       reason, name(run, label), b)

    def rec(name: str, job: str, gpus: int, *, kind: str = "cpu", finished: bool = False,
            cancelled: bool = False) -> RunRecord:
        return _record(name, kind, job, gpus=gpus, finished=finished, cancelled=cancelled)

    card = Bundle(cards=1, cpus=4, mem_gb=64)
    recs = [rec("g3a", "100", 1), rec("g3b", "110", 2), rec("held", "300", 4),
            rec("cancelled", "400", 4, cancelled=True), rec("unpinned", "500", 4, kind="blocks"),
            rec("done", "600", 4, finished=True), rec("plain", "800", 4)]
    seen = Seen("me", running=(
        run("100_0", "g3a", "orion", cpu(850, 2)), run("110_0", "g3b", "orion", cpu(540, 2)),
        run("400_0", "cancelled", "orion", cpu(540, 2)),
        run("500_0", "unpinned", "tesla1", card, label="gpu32"),
        run("900_0", "elsewhere", "orion", cpu(540, 2)),
        run("910_0", "x", "orion", cpu(8, 2), user="other"),
        run("920_0", "y", "orion", cpu(8, 2), project="mycooc"),
        run("950_0", "gone", "orion", cpu(8, 2))), pending=(
        wait("110_1", "g3b", cpu(375)), wait("110_2", "g3b", cpu(360)),
        wait("110_3", "g3b", cpu(280)), wait("300_0", "held", cpu(540), "Dependency"),
        wait("300_1", "held", cpu(540), "BeginTime"), wait("400_1", "cancelled", cpu(540)),
        wait("500_1", "unpinned", card, node=None, label="gpu32"),
        wait("500_2", "unpinned", card, node=None, label="gpu32"),
        wait("800_0", "plain", cpu(540)), wait("900_1", "elsewhere", cpu(375))), nodes=())
    week, six = 7 * 24 * 60.0, 6 * 24 * 60.0
    limits = {"100_0": week, "110_0": six, "110_1": six, "110_2": six, "110_3": six,
              "300_0": 60.0, "300_1": 60.0, "400_0": 60.0, "400_1": 60.0, "500_0": 120.0,
              "500_1": 120.0, "500_2": 120.0, "800_0": 30.0, "900_0": 600.0, "900_1": 600.0,
              "910_0": 60.0, "920_0": 60.0}           # 950_0 left squeue between the reads
    at = "2026-10-21T12:00:00Z"
    s = cluster_demand.demand_of(recs, seen, limits, project="reblock", campaign="reblock",
                                 weight=1.0, at=at,
                                 nodes=tuple(n.name for n in PROJECT.cluster.nodes))
    want = tuple(DemandLine(node=n, bundle=b, units=u, minutes=m, urgency=None) for n, b, u, m in (
        ("orion", cpu(850, 2), 1, week), ("orion", cpu(540, 2), 3, (six + 60 + 600) / 3),
        ("tesla1", card, 1, 120.0), ("orion", cpu(375), 1, six), ("orion", cpu(360), 0, 0.0),
        ("orion", cpu(280), 0, 0.0), ("orion", cpu(540), 1, 30.0)))
    runs = ("cancelled", "elsewhere", "g3a", "g3b", "held", "plain", "unpinned")

    def shown(ls: tuple[DemandLine, ...]) -> list[tuple[str, str, int, float]]:
        return [(x.node, x.bundle.text(), x.units, round(x.minutes, 3)) for x in ls]

    d = s.demand
    bad = []
    if d.lines != want:
        bad.append(f"lines {shown(d.lines)}, wanted {shown(want)}")
    if (s.unplaced, s.elsewhere) != (2, 1):
        bad.append(f"{s.unplaced} unplaced and {s.elsewhere} elsewhere, wanted 2 and 1")
    if d.runs != runs:
        bad.append(f"runs {d.runs}, wanted {runs}")
    if (d.project, d.campaign, d.at, d.weight) != ("reblock", "reblock", at, 1.0):
        bad.append(f"campaign {d.name()}, at {d.at}, weight {d.weight}")
    with tempfile.TemporaryDirectory() as tmp:
        board = Path(tmp) / "market"
        path, beside = demand_path(board, "reblock", "reblock"), demand_path(board, "reblock", "w")
        beside.parent.mkdir(parents=True)
        beside.write_text("{}\n")
        Local("here", SubprocessRunner()).sh(cluster_demand.board_script(str(board), d))
        if not path.is_file() or read_demand(path.read_text(), str(path)) != d:
            bad.append(f"the board's {path.name} is not the demand")
        left = sorted(p.name for p in board.rglob("*") if p.is_file())
        if left != ["reblock.json", "w.json"]:
            bad.append(f"the board holds {left}, not the demand beside the other file")
    print(f"demand: reblock's demand from a squeue and seven runs, on a scratch board -> "
          f"{'ok' if not bad else 'FAILS'}", flush=True)
    return bad


@contextlib.contextmanager
def _finished_named() -> Iterator[None]:
    """cluster_demand.finished never true (fault `finished-named`)."""
    old = cluster_demand.finished
    cluster_demand.finished = lambda rec: False
    try:
        yield
    finally:
        cluster_demand.finished = old


@contextlib.contextmanager
def _no_throttle() -> Iterator[None]:
    """cluster_demand.room unbounded (fault `no-throttle`)."""
    old = cluster_demand.room
    cluster_demand.room = lambda rec, running: 10**6
    try:
        yield
    finally:
        cluster_demand.room = old


@contextlib.contextmanager
def _any_reason() -> Iterator[None]:
    """cluster_demand.waits_for always true (fault `any-reason`)."""
    old = cluster_demand.waits_for
    cluster_demand.waits_for = lambda p, node: True
    try:
        yield
    finally:
        cluster_demand.waits_for = old


@contextlib.contextmanager
def _no_running() -> Iterator[None]:
    """cluster_demand.held's lines kept with no units (fault `no-running`: what a wave holds
    stated as units 0, the README's way for a wave that wants no more)."""
    old = cluster_demand.held
    cluster_demand.held = lambda running, limits: {k: [] for k in old(running, limits)}
    try:
        yield
    finally:
        cluster_demand.held = old


# The command line's fault names, each the fault it injects.
DEMAND_FAULTS: dict[str, Callable[[], contextlib.AbstractContextManager[None]]] = {
    "finished-named": _finished_named, "no-throttle": _no_throttle, "any-reason": _any_reason,
    "no-running": _no_running}


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


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "cpu":
        check_cpu(a[1].split(","), a[2].split(","))
    elif a[0] == "ids" and len(a) in (1, 2) and a[1:] in ([], ["no-check"]):
        with _no_check() if a[1:] else contextlib.nullcontext():
            bad = check_ids()
        print(f"ids{' under the fault no-check' if a[1:] else ''}: "
              + ("FAILS on " + "; ".join(bad) if bad else "passes"), flush=True)
        if bool(bad) != bool(a[1:]):
            raise SystemExit(1 if bad else "the fault no-check went unseen: the guard guards "
                                           "nothing")
    elif a[0] == "script" and a[1:] in ([], ["no-script-check"]):
        with _no_script_check() if a[1:] else contextlib.nullcontext():
            bad = check_script()
        print(f"script{' under the fault no-script-check' if a[1:] else ''}: "
              + ("FAILS on " + "; ".join(bad) if bad else "passes"), flush=True)
        if bool(bad) != bool(a[1:]):
            raise SystemExit(1 if bad else "the fault no-script-check went unseen: the guard "
                                           "guards nothing")
    elif a[0] == "demand" and len(a) <= 2 and set(a[1:]) <= set(DEMAND_FAULTS):
        with DEMAND_FAULTS[a[1]]() if a[1:] else contextlib.nullcontext():
            bad = check_demand()
        print(f"demand{f' under the fault {a[1]}' if a[1:] else ''}: "
              + ("FAILS on " + "; ".join(bad) if bad else "passes"), flush=True)
        if bool(bad) != bool(a[1:]):
            raise SystemExit(1 if bad else f"the fault {a[1]} went unseen: the guard guards "
                                           "nothing")
    else:
        raise SystemExit(f"unknown check {a[0]!r}")
