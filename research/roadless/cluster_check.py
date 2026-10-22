"""Guards on cluster.py's kinds, without a submit: each plans a run from a command line, routes and
scripts it as cluster_submit's submit does (the plan a dry submit prints and the sbatch it
stages), and checks what a task would get.

    C=research/roadless/cluster_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $C cpu <ids,> <items,>    # a print, checked by eye
    uv run python $C ids [fault]            # fault: no-check
    uv run python $C script [fault]         # fault: no-script-check
    uv run python $C demand [fault]         # faults: finished-named, no-throttle, any-reason,
                                            # no-running
    uv run python $C kinds [fault]          # fault: lease-bundle
    uv run python $C more [fault]           # faults: continue-idle, finished-dead, task-reaps,
                                            # hung-waits, reap-first
    uv run python $C template [fault]       # faults: other-mem, other-node, continuation-name,
                                            # reused-name, with-at, first-wave, market-always
    uv run python $C both-ways <guards,>    # each guard, then under each of its faults

Each guard but `cpu` has its faults (FAULTS: a patch in this process, no file edited, as
gap_clear_check.py's): it passes on the code as it is, and must fail under each fault; run alone
it exits 1 when it fails as it is, and under a fault exits 0 when it fails (the fault seen) and
names the fault when it does not. `both-ways` runs each guard as it is and under each of its
faults, in one process, and exits 1 naming every run not as expected.

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
the bank written without it, as in phase 2b.

`script` (a plan reads the repo at its commit, cluster_submit's README): a plan of either kind
reads its command's script at the run's commit (Kind.plan's sha), from git, never the working
tree. A script the working tree holds and the commit lacks (gap_calib.py at the repo's first
commit) is refused naming the commit; one the commit holds and the working tree lacks (GONE, at
the commit before the one that deleted it) plans. Fault `no-script-check`: cluster._script_at a
no-op: the absent script plans.

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
waves hold stated as units 0).

The lease pool's guards (docs/superpowers/specs/2026-10-21-lease-pool-design.md; the plan's
Rulings 3, 4, 10 and 11, and the controller's: a summary names its ladder) run cluster.py's kinds
and Waves against a real cluster_submit Context over a registry in a temporary directory, its
runner answering every ssh command with bash in a temporary cluster home, so the summaries are
read as the login node reads them:

`kinds`: each kind planned from a command line at HEAD (as a dry submit plans it, routed and
throttle-checked): a `pool` of cpu-score at 12 GB on quadro2 is one CPU array pinned there, 2
CPUs, 12 GB, 2:00:00 (its sbatch header too), no inputs, its one line `pool.py work --wave w
--node quadro2 --class m12`; one of gpu-presets at 32 GB on tesla1 one array of tesla1's card,
its task asking 8 GB of it, 4 CPUs and 32 GB, `--class g32`; refused: a gpu-presets pool on orion
(its cards unusable), a `--mem` that is no rung of the preset, a cpu-score pool on tesla1 (no
node of CPU_NODES, whose throttle it would run at: Ruling 10) and one past quadro2's throttle; a
`steward` lands on orion with 2 CPUs and 3 GB for 2:00:00, its line `pool.py steward --wave w
--board cluster_submit/market --stale-after-minutes 30` (the market's stale_after); a `long` of
90 GB and 3000 minutes asks 90 GB and 2-02:00:00, its line `pool.py work --wave w --job 7`, and
one of 7 days and a minute is refused. `Waves.lease` of each one's record: a pool's
`Lease("reblock", node, its class's bundle)`, which is what its plan asks Slurm for (a card per
task or none, the plan's CPUs and memory, on its node: Ruling 3), and None for the steward, the
long run, a `cpu` run and a `blocks` run. Fault `lease-bundle`: Waves.lease gives a CPU class 1
CPU, so a CPU pool's lease is not what its plan asks for.

`more` (the spec's guard 8, the plan's Review Focus 1 and 3): `pool_more` of a finished lease of
wave `w` two workers wide (tasks 500_0 and 500_1, out of squeue at GONE_AT), its steward live
unless a case says otherwise and its summary on the cluster (or not), `cs.launches.width` patched
(the market's arithmetic is cluster_submit's, not this guard's) to call the kind's own rule,
`today`, except in `market`. Cases: `today`, three m12 jobs pending on quadro2 and 500_1 killed
at its walltime and reaped there holding a job: Continue(2); `market`, the same with `width`
answering the market's Continue(10): Continue(3), capped by the jobs pending; `routine`, no task
killed: Continue(2); `idle`, 500_1 reaped holding no job: Done; `unreaped`, 500_1 registered and
not reaped yet: raises; `hung`, 500_1 never registered (hung in its startup): Done; `requeued`,
500_1 requeued (its registering again reaps its earlier self, holding a job, before that one's
end time) and registered again, not yet reaped: raises; `requeued-late`, the same with the earlier
self reaped past its end time (the requeue waited that long): raises; `requeued-hung`, its earlier
self reaped before its end time and the requeued task never registered: Done; `old`, the summary
written before GONE_AT: raises; `absent`, no summary: raises; `format-2`, a summary of format 2:
raises; `unsettled`, the wave's steward finished and its continuation undecided: Continue(2);
`no-steward`, the steward finished and settled: Done; `none-pending`, no m12 job pending on
quadro2 (some claimed there, some pending elsewhere): Done; and the steward's own `more`:
Continue(1) while its summary is CLOSED, Done once FINISHED. Faults: `continue-idle`
(cluster._idle_killed finding none: idle and hung continued), `finished-dead` (a steward live
only while unfinished: unsettled Done), `task-reaps` (a killed task's reap read by its task alone,
any run's and any time's: requeued-hung continued on its earlier self's), `hung-waits` (every
killed task waited for, registered or not: hung raises) and `reap-first` (a walltime reap decides
a killed task still registered: requeued-late continued on its earlier self's).

`template` (Ruling 4, Review Focus 2): live waves `a` (its steward at HEAD) and `b` (at HEAD^),
with 3 and 7 jobs of m12 (cpu-score) pending on quadro2 and none of m24 pending on orion, a pool
lease of `b` there already named b-quadro2-m12-s1, a live wave `c` whose summary is absent and a
live wave `d` whose summary names a ladder LADDERS lacks (no-such, with 9 jobs pending there).
The template for (quadro2, 0/2/12) is `b`'s: kind `pool`, run b-quadro2-m12-s2, commit `b`'s
steward's, args `--wave b --preset cpu-score --node quadro2 --mem 12 --gpus 40`; and it passes
every refusal of cluster_submit's `launches.start`, by launches' own code: its arguments parsed
with `--at` its commit read that commit and no --dry, `chain_base` of its run is the run, the
registry lacks it, the pool kind's plan of it passes `launches.line_checked` for that node and
bundle, and `Waves.lease` of its record is the campaign's lease of that bundle there. Done, naming
the bundle and node: a bundle no live wave has a line of (quadro2, 0/2/24), a node no wave serves
(tesla1), lines with nothing pending (orion, 0/2/24; its why naming `c` and `d`, left out, too),
and another campaign's. A registry whose only live wave is `c`: raises; whose only live wave is
`d`: raises, naming its ladder; with no live wave (only a steward finished and settled): Done.
`Waves.starts(market)` is the market's starts while a wave is live, and () with none, the market
not asked. Faults, one per refusal the template must not trip, one for its choice and one for
its starts: `other-mem` (its `--mem` the next rung: line_checked refuses the bundle),
`other-node` (its `--node` another node), `continuation-name` (`<w>-<node>-<class>-l1`),
`reused-name` (`-s1`, which the registry has), `with-at` (its args carrying `--at` HEAD: the parse
reads another commit than the template's), `first-wave` (the first live wave by name, whatever
its pending jobs) and `market-always` (Waves.starts asking the market with no wave live).

(`gpu`, the retired raster's sizing checked against its scorer's measured peaks, went with it:
NOTES, "The complex as the runtime: the raster retired".)
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import cluster  # noqa: E402
import cluster_demand  # noqa: E402
import cluster_submit as cs  # noqa: E402
import pool_summary as psum  # noqa: E402
from cluster_submit import Bundle, RunPlan  # noqa: E402
from cluster_submit.arrays import Array, route, script  # noqa: E402
from cluster_submit.cli import parser  # noqa: E402
from cluster_submit.context import Context, Shell, kind_of, utcnow  # noqa: E402
from cluster_submit.gpu import DemandLine, chain_base, demand_path, read_demand  # noqa: E402
from cluster_submit.gpuboard import Pending, Running, Seen  # noqa: E402
from cluster_submit.registry import NO_OVERRIDES, ArrayRecord, Registry, RunRecord  # noqa: E402
from cluster_submit.remote import Completed, Local, SubprocessRunner  # noqa: E402
from cluster_submit.submit import throttle_checked  # noqa: E402
from pool_ladder import Ladder  # noqa: E402

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


# ---- the lease pool's guards ------------------------------------------------------------------

GONE_AT = "2026-10-21T12:00:00Z"        # more: when the lease left squeue
LEASE = "w-quadro2-m12-s1"              # more: the lease, two workers wide
KILLED = "500_1"                        # more: its task killed at its walltime


def _context(repo: Path, home: Path) -> Context:
    """A cluster_submit Context over a registry in `repo`, its runner answering every ssh command
    with bash in `home` (the cluster home, as the login node runs a command)."""
    def run(argv: Sequence[str], *, input: str | None = None) -> Completed:
        if argv[0] != "ssh":
            raise AssertionError(f"a call that is no ssh command: {list(argv)}")
        r = subprocess.run(["bash", "-c", argv[-1]], cwd=home, input=input, text=True,
                           capture_output=True)
        return Completed(r.returncode, r.stdout, r.stderr)

    home.mkdir(parents=True, exist_ok=True)
    return Context(project=PROJECT, run=run, registry=Registry(repo), units_dir=repo / "units",
                   now=utcnow, entry=HERE / "cluster.py", argv=(), machine="cluster-check",
                   caller=Shell(), sleep=lambda s: None, parse=parser(PROJECT).parse_args)


def _run(name: str, kind: str, args: Sequence[str], *, commit: str, width: int,
         finished: bool, settled: bool) -> RunRecord:
    """A run's record as a submit of `kind` named `name` with `args` writes it: one array (job
    500, `width` tasks), out of squeue at GONE_AT and finished (its continuation decided, or not:
    `settled`), or live."""
    stamp = "2026-10-21T10:00:00Z"
    return RunRecord(name=name, kind=kind, commit=commit, argv=("submit", kind, name, *args),
                     overrides=NO_OVERRIDES, submitted_at=stamp, attempt=f"guard:{name}",
                     arrays=(ArrayRecord("500", 0, ("t",) * width),), gpus=4, chained_after=(),
                     last_sync=stamp, gone_at=GONE_AT if finished else None,
                     finished_at=GONE_AT if finished else None, cancelled_at=None, unpulled=(),
                     unstarted=(), continues=None, continued_by=None, settled=settled)


def _steward_run(wave: str, *, commit: str, finished: bool, settled: bool) -> RunRecord:
    return _run(f"{wave}-steward", "steward", ("--wave", wave, "--board", "market"),
                commit=commit, width=1, finished=finished, settled=settled)


def _pool_run(name: str, wave: str, *, node: str, mem: int, finished: bool) -> RunRecord:
    return _run(name, "pool", ("--wave", wave, "--preset", "cpu-score", "--node", node, "--mem",
                               str(mem), "--gpus", str(cluster.CPU_NODES[node])),
                commit=SHA, width=2, finished=finished, settled=False)


def _line(node: str, mem: int, *, pending: int, claimed: int) -> psum.ClassLine:
    """A line of cpu-score's class `mem` on `node`."""
    return psum.ClassLine(node=node, cls=f"m{mem}", cards=0, cpus=2, mem_gb=mem, pending=pending,
                          claimed=claimed, minutes=30.0, long=0, done=0, failed=0,
                          workers=claimed)


def _summary(wave: str, *, at: str, state: psum.WaveState, ladder: str,
             lines: Sequence[psum.ClassLine], reaped: Sequence[psum.ReapedWorker],
             registered: Sequence[psum.RegisteredWorker]) -> str:
    return psum.summary_json(psum.Summary(
        wave=wave, commit=SHA, ladder=ladder, at=at, round=3, state=state, metric=None,
        lines=tuple(lines), reaped=tuple(reaped), registered=tuple(registered), long=()))


def _put(home: Path, wave: str, text: str | None) -> None:
    """`wave`'s summary.json on the cluster `text`, or none."""
    path = home / cluster.WAVES / wave / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if text is None:
        path.unlink(missing_ok=True)
    else:
        path.write_text(text)


def _answer(fn: Callable[[], cs.Continue | cs.Done]) -> str:
    """What a `more` answered: `Continue(<workers>)`, `Done`, or `raises` (SystemExit)."""
    try:
        got = fn()
    except SystemExit:
        return "raises"
    return f"Continue({got.workers})" if isinstance(got, cs.Continue) else "Done"


@contextlib.contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _width_today(ctx: Context, rec: RunRecord, *,
                 today: Callable[[], cs.Continue | cs.Done]) -> cs.Continue | cs.Done:
    return today()


def _width_market(ctx: Context, rec: RunRecord, *,
                  today: Callable[[], cs.Continue | cs.Done]) -> cs.Continue | cs.Done:
    return cs.Continue(workers=10)


def check_kinds() -> list[str]:
    """`kinds` (the module's docstring): the problems found."""
    bad: list[str] = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ctx = _context(tmp / "repo", tmp / "home")
        cpu_pool = ("--wave", "w", "--preset", "cpu-score", "--node", "quadro2", "--mem", "12",
                    "--gpus", "40")
        gpu_pool = ("--wave", "w", "--preset", "gpu-presets", "--node", "tesla1", "--mem", "32",
                    "--gpus", "4")
        steward = ("--wave", "w", "--board", "cluster_submit/market")
        long = ("--wave", "w", "--preset", "cpu-score", "--job", "7", "--node", "orion", "--mem",
                "90", "--minutes", "3000")
        work = "python -u research/roadless/pool.py"
        want = {   # kind, run, args: card GB of its array, its node, resources, task GB, line
            ("pool", "w-quadro2-m12-s1", cpu_pool): (0, cs.Resources(
                cpus=2, mem_gb=12, time="2:00:00", node="quadro2"), 0.0,
                f"{work} work --wave w --node quadro2 --class m12"),
            ("pool", "w-tesla1-g32-s1", gpu_pool): (32, cs.Resources(
                cpus=4, mem_gb=32, time="2:00:00", node="tesla1"), cluster.CARD_GB,
                f"{work} work --wave w --node tesla1 --class g32"),
            ("steward", "w-steward", steward): (0, cs.Resources(
                cpus=2, mem_gb=3, time="2:00:00", node="orion"), 0.0,
                f"{work} steward --wave w --board cluster_submit/market "
                "--stale-after-minutes 30"),
            ("long", "w-long-7", long): (0, cs.Resources(
                cpus=2, mem_gb=90, time="2-02:00:00", node="orion"), 0.0,
                f"{work} work --wave w --job 7")}
        leases = {"w-quadro2-m12-s1": cs.Lease("reblock", "quadro2", Bundle(0, 2, 12)),
                  "w-tesla1-g32-s1": cs.Lease("reblock", "tesla1", Bundle(1, 4, 32))}
        for (kind, run, args), (card, res, gb, line) in want.items():
            argv = ["submit", kind, run, "--dry", *args]
            sub, plan, arrays = _plan(argv, tmp / run, sha=SHA)
            got = ([(a.card_gb, a.node, list(a.tasks)) for a in arrays], plan.resources,
                   [t.gpu_gb for t in plan.tasks], plan.inputs, plan.prelude)
            if got != ([(card, res.node, [line])], res, [gb], (), (cluster.TASK_ENV,)):
                bad.append(f"{kind} {run}: planned {got}")
            env = cluster.ENV.at(SubprocessRunner(), PROJECT.repo, SHA, root=PROJECT.root())
            head = {ln for ln in script(PROJECT, run, SHA, arrays[0], plan.resources,
                                        plan.prelude, sub.gpus, env).splitlines()
                    if ln.startswith("#SBATCH")}
            sbatch = {f"#SBATCH --nodelist={res.node}", f"#SBATCH --cpus-per-task={res.cpus}",
                      f"#SBATCH --mem={res.mem_gb}G", f"#SBATCH --time={res.time}"}
            if not sbatch <= head:
                bad.append(f"{kind} {run}: its sbatch header lacks {sorted(sbatch - head)}")
            rec = _run(run, kind, args, commit=SHA, width=1, finished=False, settled=False)
            lease = cluster.Waves(registry=ctx.registry).lease(ctx, rec)
            if lease != leases.get(run):
                bad.append(f"{kind} {run}: Waves.lease {lease}, want {leases.get(run)}")
            asks = {Bundle(cards=0 if t.cpu else 1, cpus=plan.resources.cpus,
                           mem_gb=plan.resources.mem_gb) for t in plan.tasks}
            if lease is not None and (lease.node, {lease.bundle}) != (plan.resources.node, asks):
                bad.append(f"{kind} {run}: its lease is {lease.bundle.text()} on {lease.node}, "
                           f"its plan asks {[b.text() for b in asks]} on {plan.resources.node}")
        others = [_run("c", "cpu", ("--node", "orion", "--cpus", "1", "--mem", "8", "--items",
                                    "x", "a", "--", "research/roadless/gap_calib.py", "{id}",
                                    "{item}"), commit=SHA, width=1, finished=False,
                       settled=False),
                  _run("b", "blocks", ("--gb", "8", "a", "--", "research/roadless/gap_calib.py",
                                       "{id}"), commit=SHA, width=1, finished=False,
                       settled=False)]
        bad += [f"{r.kind}: Waves.lease {x}" for r in others
                if (x := cluster.Waves(registry=ctx.registry).lease(ctx, r)) is not None]
        refusals = {   # what is refused, and a word its why says
            "a gpu-presets pool on orion": ("pool", ("--wave", "w", "--preset", "gpu-presets",
                                                     "--node", "orion", "--mem", "32", "--gpus",
                                                     "4"), "unusable"),
            "a --mem of 13 GB": ("pool", (*cpu_pool[:7], "13", *cpu_pool[8:]), "rung"),
            "a cpu-score pool on tesla1": ("pool", ("--wave", "w", "--preset", "cpu-score",
                                                    "--node", "tesla1", "--mem", "12"),
                                           "CPU_NODES"),
            "a cpu-score pool on quadro2 at --gpus 41": ("pool", (*cpu_pool[:9], "41"),
                                                         "at most 40"),
            "a long run of 7 days and a minute": ("long", (*long[:11], "10081"), "7 days")}
        for what, (kind, args, word) in refusals.items():
            try:
                why = _refused(["submit", kind, "guard", "--dry", *args], tmp / "refused",
                               sha=SHA)
                if word not in why:
                    bad.append(f"{what}: refused without saying {word!r}: {why}")
            except AssertionError:
                bad.append(f"{what}: planned")
    print(f"kinds: pool, steward and long planned, refused and leased -> "
          f"{'ok' if not bad else 'FAILS'}", flush=True)
    return bad


def _one_cpu_lease(self: cluster.Waves, ctx: Context, rec: RunRecord) -> cs.Lease | None:
    """Waves.lease giving a CPU class 1 CPU (fault `lease-bundle`)."""
    x = _LEASE(self, ctx, rec)
    if x is None or x.bundle.cards:
        return x
    return x._replace(bundle=Bundle(cards=0, cpus=1, mem_gb=x.bundle.mem_gb))


_LEASE = cluster.Waves.lease
_REGISTERED = cluster._registered


def _registered_unreaped(s: psum.Summary, run: str) -> frozenset[str]:
    """`run`'s tasks registered in `s` less those it has a walltime reap of (fault `reap-first`:
    a reap decides a killed task, an earlier self's of a requeued one too)."""
    return _REGISTERED(s, run) - set(cluster._walltime_reaps(s, run))


class StewardState(Enum):
    """A `more` case's wave steward: live, finished with its continuation undecided, or settled."""

    LIVE = "live"
    UNSETTLED = "unsettled"
    SETTLED = "settled"

    @property
    def finished(self) -> bool:
        return self is not StewardState.LIVE

    @property
    def settled(self) -> bool:
        return self is StewardState.SETTLED


class MoreCase(NamedTuple):
    """A case of `more`: the summary on the cluster (None: none), the tasks killed at their
    walltime, the wave's steward, the width's patch, and the answer."""

    summary: str | None
    killed: tuple[str, ...]
    steward: StewardState
    width: Callable[..., cs.Continue | cs.Done]
    want: str


def check_more() -> list[str]:
    """`more` (the module's docstring): the problems found."""
    closed = psum.WaveState.CLOSED
    after, before = "2026-10-21T12:01:00Z", "2026-10-21T11:59:00Z"
    lines = (_line("orion", 12, pending=4, claimed=0), _line("quadro2", 12, pending=3, claimed=2),
             _line("quadro2", 24, pending=5, claimed=0))
    held = psum.ReapedWorker(task=KILLED, run=LEASE, held_job=True, walltime=True)
    idle = held._replace(held_job=False)
    # its earlier self, reaped when the task registered again requeued (its node lost): before
    # its end time, or past it (the requeue waited that long)
    earlier, earlier_late = held._replace(walltime=False), held
    registered = (psum.RegisteredWorker(task=KILLED, run=LEASE),)

    def summary(*, at: str = after, of: Sequence[psum.ClassLine] = lines,
                reaped: Sequence[psum.ReapedWorker] = (held,),
                live: Sequence[psum.RegisteredWorker] = ()) -> str:
        return _summary("w", at=at, state=closed, ladder="cpu-score", lines=of, reaped=reaped,
                        registered=live)

    none_pending = (lines[0], _line("quadro2", 12, pending=0, claimed=2), lines[2])
    k, t, alive = (KILLED,), _width_today, StewardState.LIVE
    cases = {
        "today": MoreCase(summary(), k, alive, t, "Continue(2)"),
        "market": MoreCase(summary(), k, alive, _width_market, "Continue(3)"),
        "routine": MoreCase(summary(reaped=()), (), alive, t, "Continue(2)"),
        "idle": MoreCase(summary(reaped=(idle,)), k, alive, t, "Done"),
        "unreaped": MoreCase(summary(reaped=(), live=registered), k, alive, t, "raises"),
        "hung": MoreCase(summary(reaped=()), k, alive, t, "Done"),
        "requeued": MoreCase(summary(reaped=(earlier,), live=registered), k, alive, t,
                             "raises"),
        "requeued-late": MoreCase(summary(reaped=(earlier_late,), live=registered), k, alive, t,
                                  "raises"),
        "requeued-hung": MoreCase(summary(reaped=(earlier,)), k, alive, t, "Done"),
        "old": MoreCase(summary(at=before), k, alive, t, "raises"),
        "absent": MoreCase(None, k, alive, t, "raises"),
        "format-2": MoreCase(json.dumps({**json.loads(summary()), "format": 2}), k, alive, t,
                             "raises"),
        "unsettled": MoreCase(summary(), k, StewardState.UNSETTLED, t, "Continue(2)"),
        "no-steward": MoreCase(summary(), k, StewardState.SETTLED, t, "Done"),
        "none-pending": MoreCase(summary(of=none_pending), k, alive, t, "Done")}
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ctx = _context(tmp / "repo", tmp / "home")
        lease = _pool_run(LEASE, "w", node="quadro2", mem=12, finished=True)
        ctx.registry.write(lease)
        got: dict[str, str] = {}
        for name, c in cases.items():
            ctx.registry.write(_steward_run("w", commit=SHA, finished=c.steward.finished,
                                            settled=c.steward.settled))
            _put(tmp / "home", "w", c.summary)
            with _patched(cs.launches, "width", c.width):
                got[name] = _answer(functools.partial(cluster.pool_more, ctx, lease,
                                                      killed=c.killed))
        steward = _steward_run("w", commit=SHA, finished=True, settled=False)
        for name, state, want in (("steward-closed", closed, "Continue(1)"),
                                  ("steward-finished", psum.WaveState.FINISHED, "Done")):
            _put(tmp / "home", "w", _summary("w", at=after, state=state, ladder="cpu-score",
                                             lines=lines, reaped=(), registered=()))
            got[name] = _answer(lambda: cluster.Steward().more(ctx, steward, killed=()))
            cases[name] = MoreCase(None, (), StewardState.UNSETTLED, _width_today, want)
    bad = [f"{name} {got[name]}, want {c.want}" for name, c in cases.items()
           if got[name] != c.want]
    print(f"more: {', '.join(f'{n} {a}' for n, a in got.items())} -> "
          f"{'ok' if not bad else 'FAILS'}", flush=True)
    return bad


class _Want(NamedTuple):
    """A `template` case answered Done: the campaign, node and bundle asked, and the words its
    why says."""

    campaign: str
    node: str
    bundle: Bundle
    says: tuple[str, ...]


def _start_refusals(ctx: Context, t: cs.Template, node: str, bundle: Bundle,
                    work: Path) -> list[str]:
    """Each refusal of cluster_submit's launches.start `t` trips there, by launches' own code;
    and whether Waves.lease of its record is the campaign's lease of `bundle` on `node`."""
    bad: list[str] = []
    argv = ("submit", t.kind, t.run, "--at", t.commit, *t.args)
    sub = ctx.parse(argv)
    if sub.at != t.commit or sub.dry:
        bad.append(f"its arguments parse to --at {sub.at} (dry {sub.dry}), its commit {t.commit}")
    if chain_base(t.run) != t.run:
        bad.append(f"its run {t.run} reads as a continuation of {chain_base(t.run)}")
    if ctx.registry.exists(t.run):
        bad.append(f"its run {t.run} is in the registry")
    plan = kind_of(PROJECT, t.kind).plan(sub, work, sha=t.commit)
    try:
        cs.launches.line_checked(plan, t.run, cs.Start(campaign=cluster.CAMPAIGN, node=node,
                                                       bundle=bundle, workers=3,
                                                       lease_hours=2.0))
    except SystemExit as e:
        bad.append(f"line_checked: {e}")
    rec = _run(t.run, t.kind, t.args, commit=t.commit, width=1, finished=False, settled=False)
    lease = cluster.Waves(registry=ctx.registry).lease(ctx, rec)
    if lease != cs.Lease(cluster.CAMPAIGN, node, bundle):
        bad.append(f"its lease {lease}")
    return bad


def check_template() -> list[str]:
    """`template` (the module's docstring): the problems found."""
    closed, after = psum.WaveState.CLOSED, "2026-10-21T12:01:00Z"
    older = _rev("HEAD^")
    m12, m24 = Bundle(0, 2, 12), Bundle(0, 2, 24)
    s = cs.Start(campaign="reblock", node="quadro2", bundle=m12, workers=3, lease_hours=2.0)
    asked: list[str] = []

    def market() -> tuple[cs.Start, ...]:
        asked.append("asked")
        return (s,)

    def no_such(home: Path) -> None:
        _put(home, "d", _summary("d", at=after, state=closed, ladder="no-such", lines=(
            _line("quadro2", 12, pending=9, claimed=0),), reaped=(), registered=()))

    bad: list[str] = []
    got: list[str] = []
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        home = tmp / "home"
        ctx = _context(tmp / "repo", home)
        waves = cluster.Waves(registry=ctx.registry)
        for wave, commit, pending in (("a", SHA, 3), ("b", older, 7)):
            ctx.registry.write(_steward_run(wave, commit=commit, finished=False, settled=False))
            _put(home, wave, _summary(wave, at=after, state=closed, ladder="cpu-score", lines=(
                _line("orion", 24, pending=0, claimed=pending),
                _line("quadro2", 12, pending=pending, claimed=0)), reaped=(), registered=()))
        for wave in ("c", "d"):     # c's summary absent; d's names a ladder LADDERS lacks
            ctx.registry.write(_steward_run(wave, commit=SHA, finished=False, settled=False))
        no_such(home)
        ctx.registry.write(_pool_run("b-quadro2-m12-s1", "b", node="quadro2", mem=12,
                                     finished=False))
        t = waves.template(ctx, "reblock", "quadro2", m12)
        want = cs.Template(kind="pool", run="b-quadro2-m12-s2", commit=older,
                           args=("--wave", "b", "--preset", "cpu-score", "--node", "quadro2",
                                 "--mem", "12", "--gpus", "40"))
        if t != want:
            bad.append(f"the template for quadro2's m12 {t}, want {want}")
        if isinstance(t, cs.Template):
            got.append(f"{t.run} at {t.commit[:7]}")
            bad += [f"its start: {p}" for p in _start_refusals(ctx, t, "quadro2", m12,
                                                                tmp / "plan")]
        for w in (_Want("reblock", "quadro2", m24, ("quadro2", m24.text())),
                  _Want("reblock", "tesla1", m12, ("tesla1", m12.text())),
                  _Want("reblock", "orion", m24, ("orion", m24.text(), "wave c", "wave d",
                                                  "no-such")),
                  _Want("mycooc", "quadro2", m12, ("mycooc",))):
            x = waves.template(ctx, w.campaign, w.node, w.bundle)
            if not isinstance(x, cs.Done) or not all(word in x.why for word in w.says):
                bad.append(f"{w.campaign} on {w.node} of {w.bundle.text()}: {x}, want Done "
                           f"saying {w.says}")
            got.append(f"{w.campaign} on {w.node}, {w.bundle.text()}: {type(x).__name__}")
        if waves.starts(market) != (s,) or asked != ["asked"]:
            bad.append(f"Waves.starts with live waves: {asked}, not the market's starts")
    for case, (live, says) in {"unread": (("c",), "wave c"),
                               "no-such-ladder": (("d",), "no-such"),
                               "none-live": ((), None)}.items():
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            ctx = _context(tmp / "repo", tmp / "home")
            waves = cluster.Waves(registry=ctx.registry)
            for wave in live:
                ctx.registry.write(_steward_run(wave, commit=SHA, finished=False, settled=False))
            ctx.registry.write(_steward_run("e", commit=SHA, finished=True, settled=True))
            no_such(tmp / "home")
            try:
                x = waves.template(ctx, "reblock", "quadro2", m12)
                ok = says is None and isinstance(x, cs.Done)
                got.append(f"{case}: {type(x).__name__}")
            except SystemExit as e:
                ok = says is not None and says in str(e)
                got.append(f"{case}: raises")
            if not ok:
                bad.append(f"{case}: {got[-1]}, want {'Done' if says is None else 'raises'}"
                           f"{'' if says is None else f' naming {says}'}")
            if not live:
                asked.clear()
                started = waves.starts(market)
                got.append(f"{case}: starts {started}")
                if started or asked:
                    bad.append(f"Waves.starts with no live wave: {started}, the market "
                               f"{'asked' if asked else 'not asked'}")
    print(f"template: {'; '.join(got)} -> {'ok' if not bad else 'FAILS'}", flush=True)
    return bad


_POOL_ARGS, _START_NAME = cluster._pool_args, cluster._start_name


def _other_mem(*, wave: str, ladder: Ladder, node: str, rung: int) -> tuple[str, ...]:
    """The template's --mem the rung above its class's (fault `other-mem`)."""
    args = list(_POOL_ARGS(wave=wave, ladder=ladder, node=node, rung=rung))
    args[args.index("--mem") + 1] = str(ladder.rungs[ladder.rungs.index(rung) + 1])
    return tuple(args)


def _other_node(*, wave: str, ladder: Ladder, node: str, rung: int) -> tuple[str, ...]:
    """The template's --node another node of CPU_NODES (fault `other-node`)."""
    args = list(_POOL_ARGS(wave=wave, ladder=ladder, node=node, rung=rung))
    args[args.index("--node") + 1] = next(n for n in sorted(cluster.CPU_NODES) if n != node)
    return tuple(args)


def _with_at(*, wave: str, ladder: Ladder, node: str, rung: int) -> tuple[str, ...]:
    """The template's args carrying --at HEAD (fault `with-at`)."""
    return (*_POOL_ARGS(wave=wave, ladder=ladder, node=node, rung=rung), "--at", SHA)


GUARDS: dict[str, Callable[[], list[str]]] = {
    "ids": check_ids, "script": check_script, "demand": check_demand, "kinds": check_kinds,
    "more": check_more, "template": check_template}
FAULTS: dict[str, dict[str, Callable[[], contextlib.AbstractContextManager[None]]]] = {
    "ids": {"no-check": _no_check},
    "script": {"no-script-check": _no_script_check},
    "demand": DEMAND_FAULTS,
    "kinds": {"lease-bundle": lambda: _patched(cluster.Waves, "lease", _one_cpu_lease)},
    "more": {"continue-idle": lambda: _patched(cluster, "_idle_killed",
                                               lambda killed, reaps: []),
             "finished-dead": lambda: _patched(cluster, "_live",
                                               lambda rec: rec.finished_at is None),
             "task-reaps": lambda: _patched(cluster, "_walltime_reaps",
                                            lambda s, run: {r.task: r for r in s.reaped}),
             "hung-waits": lambda: _patched(cluster, "_unreaped",
                                            lambda killed, registered: list(killed)),
             "reap-first": lambda: _patched(cluster, "_registered", _registered_unreaped)},
    "template": {
        "other-mem": lambda: _patched(cluster, "_pool_args", _other_mem),
        "other-node": lambda: _patched(cluster, "_pool_args", _other_node),
        "continuation-name": lambda: _patched(
            cluster, "_start_name", lambda ctx, wave, node, cls: f"{wave}-{node}-{cls}-l1"),
        "reused-name": lambda: _patched(
            cluster, "_start_name", lambda ctx, wave, node, cls: f"{wave}-{node}-{cls}-s1"),
        "with-at": lambda: _patched(cluster, "_pool_args", _with_at),
        "first-wave": lambda: _patched(cluster, "_chosen",
                                       lambda choices: min(choices, key=lambda c: c.wave)),
        "market-always": lambda: _patched(cluster.Waves, "starts",
                                          lambda self, market: tuple(market()))}}


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
