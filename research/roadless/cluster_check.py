"""Guards on cluster.py's kinds, without a submit: each plans a run from a command line, routes and
scripts it as cluster_submit's submit does (the plan a dry submit prints and the sbatch it
stages), and checks what a task would get.

    C=research/roadless/cluster_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $C cpu <ids,> <items,>    # a print, checked by eye
    uv run python $C ids [fault]            # fault: no-check
    uv run python $C script [fault]         # fault: no-script-check
    uv run python $C demand [fault]         # faults: finished-named, no-throttle, any-reason,
                                            # no-running, count-done, every-node, queue-only
    uv run python $C kinds [fault]          # fault: lease-bundle
    uv run python $C more [fault]           # faults: continue-idle, finished-dead, task-reaps,
                                            # hung-waits, reap-first, none-unreaped
    uv run python $C template [fault]       # faults: other-mem, other-node, continuation-name,
                                            # reused-name, with-at, first-wave, market-always
    uv run python $C wave [fault]           # faults: no-keep, block-names, no-script-check,
                                            # no-check, fit-capped, uncapped, any-node,
                                            # pools-only, fit-margin, unfinished-only,
                                            # first-cancel-stops, record-reused, no-bank-code,
                                            # common-only, drop-regions, clean-always,
                                            # outside-import, producers-unseen
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

`demand` (cluster_demand: reblock's one campaign in the GPU market, with the lease-pool plan's
Rulings 1 and 2): its demand from a made-up squeue, seven made-up runs of this registry and three
waves' runs, their summaries read from a cluster home made here, and the board script run on a
scratch board. Held units: every running task of the project's but a pool lease's,
cp-roads4-desire-g3b's as it stood on 2026-10-21 among them (throttle 2, one running, three
waiting, their memory changed in Slurm), a cancelled run's and another checkout's run's too, the
waves' two running stewards (one unit each at the steward's bundle on orion) and a long run,
aggregated by (node, bundle) at the mean of their time limits; waiting units: those each run's
throttle starts now (one of g3b's), none held by a dependency or a begin time, none of a cancelled
run; an unpinned run's waiting tasks stated nowhere and counted, another checkout's left to it and
counted. The waves' pool lines, each wave's queue made in that home and its summary.json its
steward's (pool_steward._summary, its lines pool_summary.lines_of's): waves `a` and `b` share
(quadro2, cpu-score's m4), `a` with 5 jobs pending, 2 claimed and 3 workers running there and one
job pending on orion (m12), `b` with nothing pending or claimed and 2 workers running there (idle,
about to retire), and `c` finished, its m12 jobs on quadro2 done and its worker exited; a lease of
`a` waits in Slurm on (quadro2, m4). The live waves (cluster._live_stewards: `c`'s steward finished
and unsettled) are a, b and c; the registry's pool runs (cluster.pool_runs) its three unfinished
ones; (quadro2, m4) states 9 units, max(3, 5 + 2) and max(2, 0) (Ruling 2), at the units' mean of
a's jobs' estimates and b's workers' time limits; the waiting lease no unit; (orion, m12) 1; `c` no
line; and with `b`'s summary absent, `b` unread and its two workers stated alone (the same line).
Its runs every run squeue lists of the project's and every unfinished one of the registry (every
pool, steward and long run among them), not a finished one, another user's, another project's, or a
task that left squeue between the two reads. Faults: `finished-named` (cluster_demand.finished
never true: a finished run named), `no-throttle` (cluster_demand.room unbounded: units past a run's
throttle), `any-reason` (cluster_demand.waits_for always true: a task held by its dependency
counted), `no-running` (cluster_demand.held's units dropped: what the waves hold stated as units
0), `count-done` (pool_summary.lines_of counting a DONE job as pending: the finished wave `c`
states units), `every-node` (lines_of counting each job on every node of its wave: a group's jobs
counted on each of the wave's nodes) and `queue-only` (cluster_demand.pool_units the jobs pending
or claimed alone: `b`'s idle workers dropped, against Ruling 2).

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
killed task waited for, registered or not: hung raises), `reap-first` (a walltime reap decides
a killed task still registered: requeued-late continued on its earlier self's) and
`none-unreaped` (cluster._unreaped finding none, the registered-means-raise rule gone: unreaped,
requeued and requeued-late decided on reaps that are not theirs).

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

`wave` (the spec's 3.8 and 3.9, cluster_wave.py): `cluster.py wave submit ... --dry` command lines
at HEAD through cluster.main, cluster_wave's edge a registry in a temporary directory, a runner
that records any call to the cluster (none is made), each run's dry submit planned as `kinds` plans
one, and a rows root of stored lineup-score rows made here (WAVE_ROWS: no stored row of this
checkout's is read): `wave submit g-1 --preset cpu-score --family lineup-score --nodes
orion,quadro2 --items clearance,resistance_lp --unpack <tars>/{id}.tar <WAVE_BLOCKS> --
road_rescore.py lineup-score <scratch> {id} {item}` writes a seed of one job per (block, item) in
order, each block's on one node, each line unpacking its block's tar through cluster_unpack.py and
ending `--keep-standing`; prints each node's classes; plans no pool, its steward `--at` HEAD on the
default board, and one long run, of the job whose stored wall (100 minutes, 125 with its margin)
passes a fresh lease's 105: `--mem` 45, 1.5 times its stored peak of 30 GB, and `--minutes` 188
(1.5 times 125, rounded up), every planned run's name a run name's. Refused: the misspelt id of
`ids` (MISTYPED), by the bank's writer, before anything is planned; a script the commit lacks
(GONE), naming the commit; `{item}` without --items, and --items without `{item}`; a tar whose name
does not end in `.tar`; gpu-presets on orion (its cards unusable); the Ruling's fitted long jobs
whose estimate passes 7 days (cycle_desire on both blocks, no stored row: the family's longest wall
under 1 km2, 11000 minutes), naming them; a wave with a local record (`waves/g-10` here); and,
cluster_wave._uncommitted answering every path changed, an uncommitted bank code, naming BANK_FILES
(BANK_CODE's paths, spelt here: the other cases read this checkout's bank code as committed,
whatever it is where the guard runs). Planned with long runs capped at `--minutes` 10080 (7 days):
a stored long job (slow_stored: 7500 minutes with its margin), and fitted ones whose estimate fits
inside 7 days, 1.5 times it past them (fitted_slow on both blocks: its own fit over its five
blocks, 7000 minutes). cluster_wave._uncommitted on a git repo made here lists common.py, a file
below src/reblock and one below conf once they are changed (and no other file), and nothing while
they are committed. A bank of one block written by common.write_bank in a subprocess (BANK_PROBE)
loads its modules of this repo (not of its .venv) from below BANK_CODE's paths alone, common.py
among them, and a file below src/reblock (conf is read, not imported: BANK_CODE lists it whole).
Then the requests, on a registry with wave `w` live (its steward, a pool lease running, one
finished with its continuation undecided, one finished and settled, a long run running, another
wave's pool) and its seed of three jobs on the cluster: `wave w retry 2`, `wave w retry failed` and
`wave w cancel` leave in `reblock-waves/w/requests/` exactly three files, reading as Retry((2,)),
Retry((1, 2, 3)) and Cancel(); `retry 9` is refused naming 9; `cancel` cancels the running pool
lease, the undecided one and the long run, every one though the first cancel fails (exit 1), and
neither the settled lease, the steward nor the other wave's pool. Faults: `no-keep` (the job lines
without `--keep-standing`), `block-names` (long runs named `<w>-long-<block>`: no run name),
`no-script-check` and `no-check` (as `script`'s and `ids`'), `fit-capped` (the Ruling's refusal
gone: fitted jobs past 7 days capped), `fit-margin` (the Ruling as first written, 1.5 times the
estimate past 7 days: fitted_slow refused), `uncapped` (long runs at 1.5 times their estimate,
uncapped: the long kind refuses slow_stored's), `any-node` (cluster_wave._servable a no-op:
gpu-presets planned on orion), `pools-only` (cancel leaving the long runs), `unfinished-only`
(cluster_wave._undone as unfinished alone: the undecided lease left), `first-cancel-stops` (cancel
stopping at its first failure), `record-reused` (cluster_wave._recorded a no-op), `no-bank-code`
(BANK_CODE empty), `common-only` (BANK_CODE common.py alone: the Block producers' change unseen),
`drop-regions` (BANK_CODE without regions.py: the probe's imports name it), `clean-always`
(cluster_wave._uncommitted answering none), and two faults of the probe alone (BANK_PROBE changed,
BANK_CODE as it is, so no other case sees them): `outside-import` (the probe imports pool_ladder
beside common, as a common.py importing a module of this repo outside BANK_CODE would: the probe
names pool_ladder.py) and `producers-unseen` (the probe blind to src/, as with reblock loaded from
outside the repo: it sees no file below src/reblock).

(`gpu`, the retired raster's sizing checked against its scorer's measured peaks, went with it:
NOTES, "The complex as the runtime: the raster retired".)
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import cluster  # noqa: E402
import cluster_demand  # noqa: E402
import cluster_submit as cs  # noqa: E402
import cluster_wave  # noqa: E402
import pool_plan as pp  # noqa: E402
import pool_queue as pq  # noqa: E402
import pool_steward  # noqa: E402
import pool_summary as psum  # noqa: E402
from cluster_submit import Bundle, RunPlan  # noqa: E402
from cluster_submit.arrays import Array, route, script  # noqa: E402
from cluster_submit.cli import parser  # noqa: E402
from cluster_submit.context import STAMP, Context, Shell, kind_of, utcnow  # noqa: E402
from cluster_submit.gpu import DemandLine, chain_base, demand_path, read_demand  # noqa: E402
from cluster_submit.gpuboard import Pending, Running, Seen  # noqa: E402
from cluster_submit.project import NAME  # noqa: E402
from cluster_submit.registry import NO_OVERRIDES, ArrayRecord, Registry, RunRecord  # noqa: E402
from cluster_submit.remote import Completed, Local, SubprocessRunner  # noqa: E402
from cluster_submit.submit import throttle_checked  # noqa: E402
from pool_ladder import LADDERS, Ladder  # noqa: E402

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


@contextlib.contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _demand_run(name: str, kind: str, args: Sequence[str], job: str, *, width: int,
                finished: bool) -> RunRecord:
    """A run of a `demand` wave: `_run`'s record of it at HEAD, `width` tasks of array job `job`;
    finished, its continuation undecided, or live."""
    r = _run(name, kind, args, commit=SHA, width=width, finished=finished, settled=False)
    return replace(r, arrays=(ArrayRecord(job, 0, ("t",) * width),))


def _demand_pool(name: str, wave: str, job: str, *, node: str, mem: int, width: int,
                 finished: bool) -> RunRecord:
    """A lease of `demand` wave `wave`'s cpu-score pool of `mem` GB on `node` (`_demand_run`)."""
    return _demand_run(name, "pool", ("--wave", wave, "--preset", "cpu-score", "--node", node,
                                      "--mem", str(mem), "--gpus", str(cluster.CPU_NODES[node])),
                       job, width=width, finished=finished)


def _demand_job(i: int, *, node: str, mem: int, minutes: float) -> pq.JobSeed:
    """Job `i` of a `demand` wave: of cpu-score's class `mem` on `node`, estimated at `minutes`."""
    block = f"guard.demand{i}"
    return pq.JobSeed(id=i, block=block, item="clearance", group=block, line=f"echo {block}",
                      node=node, cls=f"m{mem}", gb=mem - 1.0, minutes=minutes,
                      peak_gb=(mem - 1.0) / 1.2, source="rows:guard", lane=pq.Lane.POOL)


def _demand_summaries(home: Path) -> None:
    """The `demand` waves' queues in `home`'s WAVES (each wave's directory, as on the cluster), as
    their workers leave them, and each one's summary.json as its steward writes it
    (pool_steward._summary: its lines pool_summary.lines_of's): `a`, CLOSED, 7 jobs of m4 on
    quadro2 estimated at 10 to 70 minutes, 2 of them claimed by workers of a-quadro2-m4-s1 and 5
    pending, its third worker idle, and one m12 job on orion pending (25 minutes); `b`, CLOSED, its
    one m4 job on quadro2 done and its two workers there idle (about to retire); `c`, FINISHED,
    its two m12 jobs on quadro2 done and its one worker retired (its lease has exited)."""
    lad = LADDERS["cpu-score"]
    now = datetime.now(UTC)

    def made(wave: str, jobs: Sequence[pq.JobSeed]) -> pq.SqliteQueue:
        h = pq.Header(wave=wave, commit=SHA, family="guard", ladder=lad,
                      created=now.strftime(STAMP), feed=pq.FIXED)
        return pq.SqliteQueue.open(home / cluster.WAVES / wave / "queue.db",
                                   pq.Seed(header=h, jobs=tuple(jobs), closed=True))

    def workers(q: pq.SqliteQueue, run: str, tasks: Sequence[str], *, node: str,
                cls: str) -> None:
        for t in tasks:
            q.register(worker=t, run=run, node=node, cls=cls, ends_at=now.timestamp() + 7200,
                       restarts=0)

    def claimed(q: pq.SqliteQueue, worker: str) -> pq.Claim:
        c = q.claim(worker=worker, minutes_left=1000.0)
        if c is None:
            raise AssertionError(f"wave {q.header().wave}'s {worker} claimed nothing")
        return c

    def done(q: pq.SqliteQueue, worker: str) -> None:
        q.finish(claimed(q, worker).attempt, worker=worker, outcome=pq.Outcome.DONE,
                 peak_gb=1.0, minutes=1.0, rc=0)

    a = made("a", [*(_demand_job(i, node="quadro2", mem=4, minutes=10.0 * i)
                     for i in range(1, 8)), _demand_job(8, node="orion", mem=12, minutes=25.0)])
    workers(a, "a-quadro2-m4-s1", ("1100_0", "1100_1", "1100_2"), node="quadro2", cls="m4")
    claimed(a, "1100_0")
    claimed(a, "1100_1")
    b = made("b", [_demand_job(1, node="quadro2", mem=4, minutes=30.0)])
    workers(b, "b-quadro2-m4-s1", ("1200_0", "1200_1"), node="quadro2", cls="m4")
    done(b, "1200_0")
    c = made("c", [_demand_job(i, node="quadro2", mem=12, minutes=20.0) for i in (1, 2)])
    workers(c, "c-quadro2-m12-s1", ("1400_0",), node="quadro2", cls="m12")
    done(c, "1400_0")
    done(c, "1400_0")
    c.retire("1400_0")
    for q, state in ((a, psum.WaveState.CLOSED), (b, psum.WaveState.CLOSED),
                     (c, psum.WaveState.FINISHED)):
        _put(home, q.header().wave, psum.summary_json(
            pool_steward._summary(q, round_no=1, at=now, state=state)))


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
    m4, steward = cpu(4, 2), cpu(3, 2)
    recs = [rec("g3a", "100", 1), rec("g3b", "110", 2), rec("held", "300", 4),
            rec("cancelled", "400", 4, cancelled=True), rec("unpinned", "500", 4, kind="blocks"),
            rec("done", "600", 4, finished=True), rec("plain", "800", 4),
            *(_demand_run(f"{w}-steward", "steward", ("--wave", w, "--board", "market"), job,
                          width=1, finished=w == "c")
              for w, job in (("a", "1000"), ("b", "1010"), ("c", "1020"))),
            _demand_pool("a-quadro2-m4-s1", "a", "1100", node="quadro2", mem=4, width=3,
                         finished=False),
            _demand_pool("a-quadro2-m4-s2", "a", "1110", node="quadro2", mem=4, width=1,
                         finished=False),
            _demand_pool("b-quadro2-m4-s1", "b", "1200", node="quadro2", mem=4, width=2,
                         finished=False),
            _demand_pool("c-quadro2-m12-s1", "c", "1400", node="quadro2", mem=12, width=1,
                         finished=True),
            _demand_run("a-long-7", "long", ("--wave", "a", "--preset", "cpu-score", "--job",
                                             "7", "--node", "orion", "--mem", "90",
                                             "--minutes", "3000"), "1300", width=1,
                        finished=False)]
    seen = Seen("me", running=(
        run("100_0", "g3a", "orion", cpu(850, 2)), run("110_0", "g3b", "orion", cpu(540, 2)),
        run("400_0", "cancelled", "orion", cpu(540, 2)),
        run("500_0", "unpinned", "tesla1", card, label="gpu32"),
        run("900_0", "elsewhere", "orion", cpu(540, 2)),
        run("910_0", "x", "orion", cpu(8, 2), user="other"),
        run("920_0", "y", "orion", cpu(8, 2), project="mycooc"),
        run("950_0", "gone", "orion", cpu(8, 2)),
        run("1000_0", "a-steward", "orion", steward), run("1010_0", "b-steward", "orion", steward),
        run("1300_0", "a-long-7", "orion", cpu(90, 2)),
        *(run(f"1100_{k}", "a-quadro2-m4-s1", "quadro2", m4) for k in range(3)),
        *(run(f"1200_{k}", "b-quadro2-m4-s1", "quadro2", m4) for k in range(2))), pending=(
        wait("110_1", "g3b", cpu(375)), wait("110_2", "g3b", cpu(360)),
        wait("110_3", "g3b", cpu(280)), wait("300_0", "held", cpu(540), "Dependency"),
        wait("300_1", "held", cpu(540), "BeginTime"), wait("400_1", "cancelled", cpu(540)),
        wait("500_1", "unpinned", card, node=None, label="gpu32"),
        wait("500_2", "unpinned", card, node=None, label="gpu32"),
        wait("800_0", "plain", cpu(540)), wait("900_1", "elsewhere", cpu(375)),
        wait("1110_0", "a-quadro2-m4-s2", m4, node="quadro2")), nodes=())
    week, six = 7 * 24 * 60.0, 6 * 24 * 60.0
    limits = {"100_0": week, "110_0": six, "110_1": six, "110_2": six, "110_3": six,
              "300_0": 60.0, "300_1": 60.0, "400_0": 60.0, "400_1": 60.0, "500_0": 120.0,
              "500_1": 120.0, "500_2": 120.0, "800_0": 30.0, "900_0": 600.0, "900_1": 600.0,
              "910_0": 60.0, "920_0": 60.0,           # 950_0 left squeue between the reads
              "1000_0": 120.0, "1010_0": 120.0, "1300_0": 3000.0, "1110_0": 120.0,
              **{f"1100_{k}": 120.0 for k in range(3)},
              **{f"1200_{k}": 100.0 for k in range(2)}}   # b's lease capped by the market
    at = "2026-10-21T12:00:00Z"
    pools = {"a-quadro2-m4-s1": cluster_demand.PoolRun(wave="a", node="quadro2", bundle=m4),
             "a-quadro2-m4-s2": cluster_demand.PoolRun(wave="a", node="quadro2", bundle=m4),
             "b-quadro2-m4-s1": cluster_demand.PoolRun(wave="b", node="quadro2", bundle=m4)}
    # (quadro2, m4): a's max(3 running, 5 pending + 2 claimed) at its jobs' mean (40 minutes),
    # and b's max(2 running, none) at its workers' time limits; c's m12 line there none
    want = tuple(DemandLine(node=n, bundle=b, units=u, minutes=m, urgency=None) for n, b, u, m in (
        ("orion", cpu(850, 2), 1, week), ("orion", cpu(540, 2), 3, (six + 60 + 600) / 3),
        ("tesla1", card, 1, 120.0), ("orion", steward, 2, 120.0),
        ("orion", cpu(90, 2), 1, 3000.0), ("orion", cpu(375), 1, six),
        ("orion", cpu(360), 0, 0.0), ("orion", cpu(280), 0, 0.0), ("orion", cpu(540), 1, 30.0),
        ("orion", cpu(12, 2), 1, 25.0), ("quadro2", m4, 9, (7 * 40.0 + 2 * 100.0) / 9)))
    runs = ("a-long-7", "a-quadro2-m4-s1", "a-quadro2-m4-s2", "a-steward", "b-quadro2-m4-s1",
            "b-steward", "cancelled", "elsewhere", "g3a", "g3b", "held", "plain", "unpinned")

    def shown(ls: tuple[DemandLine, ...]) -> list[tuple[str, str, int, float]]:
        return [(x.node, x.bundle.text(), x.units, round(x.minutes, 3)) for x in ls]

    def stated(s: cluster_demand.Stated, *, waves: tuple[str, ...],
               unread: tuple[str, ...]) -> list[str]:
        """What is wrong with `s`, the demand stated with the summaries of `waves` read and those
        of `unread` not."""
        d = s.demand
        bad = []
        if d.lines != want:
            bad.append(f"lines {shown(d.lines)}, wanted {shown(want)}")
        if (s.waves, s.unread) != (waves, unread):
            bad.append(f"the lines of waves {s.waves} stated and {s.unread} unread, wanted "
                       f"{waves} and {unread}")
        return bad

    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        home = Path(tmp) / "home"
        login = _context(Path(tmp) / "repo", home).remote()
        _demand_summaries(home)
        live = sorted(cluster._live_stewards(recs, cluster._submitted))
        if live != ["a", "b", "c"]:
            bad.append(f"live waves {live}, want a, b and c (c's steward finished, unsettled)")
        got = cluster.pool_runs(recs)
        if got != pools:
            bad.append(f"pool runs {got}, want {pools}")
        s = cluster_demand.demand_of(
            recs, seen, limits, pools=got, waves=cluster.wave_summaries(login, live),
            project="reblock", campaign="reblock", weight=1.0, at=at,
            nodes=tuple(n.name for n in PROJECT.cluster.nodes))
        bad += stated(s, waves=("a", "b", "c"), unread=())
        _put(home, "b", None)
        alone = cluster_demand.demand_of(
            recs, seen, limits, pools=got, waves=cluster.wave_summaries(login, live),
            project="reblock", campaign="reblock", weight=1.0, at=at,
            nodes=tuple(n.name for n in PROJECT.cluster.nodes))
        bad += [f"b's summary absent: {x}" for x in stated(alone, waves=("a", "c"),
                                                           unread=("b",))]
    d = s.demand
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
    print(f"demand: reblock's demand from a squeue, seven runs and waves a, b and c (b's summary "
          f"read, then absent), on a scratch board -> {'ok' if not bad else 'FAILS'}", flush=True)
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


_LINES_OF = psum.lines_of


def _count_done(jobs: Sequence[pq.Job], workers: Sequence[pq.Worker],
                ladder: Ladder) -> tuple[psum.ClassLine, ...]:
    """pool_summary.lines_of counting a DONE job as pending (fault `count-done`: a finished wave
    states units)."""
    return _LINES_OF([j._replace(state=pq.JobState.PENDING) if j.state == pq.JobState.DONE
                      else j for j in jobs], workers, ladder)


def _every_node(jobs: Sequence[pq.Job], workers: Sequence[pq.Worker],
                ladder: Ladder) -> tuple[psum.ClassLine, ...]:
    """pool_summary.lines_of counting each job on every node its wave has a job on (fault
    `every-node`: a group's jobs counted on each of the wave's nodes)."""
    nodes = sorted({j.node for j in jobs})
    return _LINES_OF([j._replace(node=n) for j in jobs for n in nodes], workers, ladder)


# The command line's fault names, each the fault it injects.
DEMAND_FAULTS: dict[str, Callable[[], contextlib.AbstractContextManager[None]]] = {
    "finished-named": _finished_named, "no-throttle": _no_throttle, "any-reason": _any_reason,
    "no-running": _no_running,
    "count-done": lambda: _patched(psum, "lines_of", _count_done),
    "every-node": lambda: _patched(psum, "lines_of", _every_node),
    "queue-only": lambda: _patched(cluster_demand, "pool_units",
                                   lambda line, running: line.pending + line.claimed)}


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


# ---- the wave verb ----------------------------------------------------------------------------

WAVE_BLOCKS = ("ZAF.9.5.4_1_9712", "ZAF.9.5.4_1_9717")  # wave: two of TUNING's large blocks
FIT_BLOCK = "guard.fit"         # wave: a stored run's block beside them, under 1 km2, in the fit
#: wave: the blocks of fitted_slow's stored runs, under 1 km2: pool_plan.FIT_MIN_BLOCKS of them,
#: so its fit is its own
FITTED = tuple(f"guard.fitted{k}" for k in range(1, 6))
SCORE = "research/roadless/road_rescore.py"
COMMON = "research/roadless/common.py"
PRODUCERS = "src/reblock"       # wave: the Block producers' package, which a bank's write imports
#: wave: the bank's code as cluster_wave.BANK_CODE lists it, in its order
BANK_FILES = (COMMON, "research/roadless/regions.py", "research/roadless/gap_exits.py",
              "research/roadless/digests.py", "scripts/__init__.py",
              "scripts/fetch_kblock_fixtures.py", PRODUCERS, "conf")
PRODUCER = "src/reblock/derive/guard.py"   # wave: a Block producer's file, below src/reblock
CONFIG = "conf/guard.yaml"                 # wave: a config file, below conf


class _Stored(NamedTuple):
    """A stored lineup-score row of the `wave` guard's rows root: a network scored."""

    block: str
    item: str
    peak_gb: float
    minutes: float


WAVE_ROWS = (
    _Stored(WAVE_BLOCKS[0], "clearance", 3.0, 20.0),
    _Stored(WAVE_BLOCKS[0], "resistance_lp", 9.0, 30.0),
    _Stored(WAVE_BLOCKS[1], "clearance", 3.0, 25.0),
    _Stored(WAVE_BLOCKS[1], "resistance_lp", 30.0, 100.0),  # 125 min with its margin: long
    _Stored(WAVE_BLOCKS[1], "slow_stored", 5.0, 6000.0),    # 7500: 1.5 times it past 7 days
    _Stored(FIT_BLOCK, "slow", 2.0, 11000.0),   # the family's longest wall under 1 km2: past 7 days
    *(_Stored(f, "fitted_slow", 2.0, 7000.0)    # its own fit: 7000 minutes, inside 7 days
      for f in FITTED))


def _wave_rows(root: Path) -> None:
    """The `wave` guard's rows root: each of WAVE_ROWS a row file as road_rescore writes one (the
    lineup-score family reads them), and a features CSV giving FIT_BLOCK's and FITTED's areas
    (WAVE_BLOCKS', from the bank)."""
    import pandas as pd
    d = root / "road_rows" / "guard" / "area_guard"
    d.mkdir(parents=True)
    for r in WAVE_ROWS:
        pd.DataFrame({"block": [r.block], "method": [r.item], "peak_rss_gb": [r.peak_gb],
                      "t_total": [r.minutes * 60]}).to_parquet(d / f"{r.block}__{r.item}.parquet")
    features = root / "consolidated_rows" / "summary_features.csv"
    features.parent.mkdir(parents=True)
    features.write_text("block,area_m2\n" + "".join(f"{f},500000\n" for f in (FIT_BLOCK, *FITTED)))


class _Planned(NamedTuple):
    """A run's dry submit as the `wave` guard's verb planned it."""

    argv: tuple[str, ...]
    args: argparse.Namespace
    plan: RunPlan


def _planner(work: Path, planned: list[_Planned]) -> cluster_wave.Verb:
    """The `wave` guard's verb: each call a dry submit, planned as `kinds` plans one (the kind's
    plan, routed and throttle-checked), and recorded; anything else is a failure of the guard."""
    def verb(argv: Sequence[str]) -> int:
        args = _parsed(list(argv))
        if args.verb != "submit" or not args.dry:
            raise AssertionError(f"a dry wave ran {list(argv)}")
        _args, plan, _arrays = _plan(list(argv), work / args.run, sha=args.at)
        planned.append(_Planned(argv=tuple(argv), args=args, plan=plan))
        return 0
    return verb


def _offline(repo: Path, calls: list[list[str]]) -> Context:
    """A Context over a registry in `repo` whose runner records every call (a dry wave makes
    none) and answers it as ssh does when the cluster cannot be reached."""
    def run(argv: Sequence[str], *, input: str | None = None) -> Completed:
        calls.append(list(argv))
        return Completed(255, "", "the guard's cluster is unreachable")

    return Context(project=PROJECT, run=run, registry=Registry(repo), units_dir=repo / "units",
                   now=utcnow, entry=HERE / "cluster.py", argv=(), machine="cluster-check",
                   caller=Shell(), sleep=lambda s: None, parse=parser(PROJECT).parse_args)


class _Waved(NamedTuple):
    """What a `wave` command line came to through cluster.main: its refusal (None: it ran, exit
    0), what it printed, and the dry submits it planned."""

    refused: str | None
    out: str
    planned: list[_Planned]


def _waved(edge: cluster_wave.Edge, planned: list[_Planned], argv: Sequence[str]) -> _Waved:
    """`cluster.py wave <argv>` through cluster.main, on `edge`."""
    planned.clear()
    out, err = io.StringIO(), io.StringIO()
    refused: str | None = None
    try:
        with (contextlib.redirect_stdout(out), contextlib.redirect_stderr(err),
              _patched(cluster_wave, "here", lambda a: edge)):
            code = cluster.main(["wave", *argv])
        if code:
            refused = f"exit {code}: {err.getvalue().strip()[-500:]}"
    except SystemExit as e:
        refused = f"{e.code}: {err.getvalue().strip()[-500:]}"
    return _Waved(refused=refused, out=out.getvalue(), planned=list(planned))


def _submit_line(wave: str, *opts: str, preset: str, ids: Sequence[str],
                 command: Sequence[str]) -> list[str]:
    """`wave submit <wave> --preset <preset> --family lineup-score --dry <opts> <ids> --
    <command>`."""
    return ["submit", wave, "--preset", preset, "--family", "lineup-score", "--dry", *opts,
            ",".join(ids), "--", *command]


def _main_wave(got: _Waved, *, work: Path, scratch: str) -> list[str]:
    """The `wave` guard's main case, g-1 (the module's docstring): its problems."""
    if got.refused is not None:
        return [f"g-1 refused: {got.refused}"]
    bad: list[str] = []
    seed = pq.read_seed((work / "g-1" / pool_steward.SEED).read_text(), "g-1's seed")
    items = ("clearance", "resistance_lp")
    want = [(b, i) for b in WAVE_BLOCKS for i in items]
    if [(j.id, j.block, j.item) for j in seed.jobs] != [(k, *w) for k, w in enumerate(want, 1)]:
        bad.append(f"jobs {[(j.id, j.block, j.item) for j in seed.jobs]}, want one per (block, "
                   f"item): {want}")
    node = {b: {j.node for j in seed.jobs if j.block == b} for b in WAVE_BLOCKS}
    if any(len(n) != 1 for n in node.values()):
        bad.append(f"a block's jobs on several nodes: {node}")
    lines = {j.line for j in seed.jobs
             if j.line != f'python -u research/roadless/cluster_unpack.py "$CLUSTER_SUBMIT_INPUTS/'
                          f'{j.block}.tar" && python -u {SCORE} lineup-score {scratch} {j.block} '
                          f"{j.item} --keep-standing"}
    if lines:
        bad.append(f"job lines {sorted(lines)[:2]}")
    printed = got.out.splitlines()
    for n in sorted({j.node for j in seed.jobs if j.lane == pq.Lane.POOL}):
        counts = Counter(j.cls for j in seed.jobs if j.node == n and j.lane == pq.Lane.POOL)
        shown = [ln for ln in printed if ln.startswith(f"  {n}: ")]
        if len(shown) != 1 or not all(f"{c} {k}" in shown[0] for c, k in counts.items()):
            bad.append(f"{n}'s classes {dict(counts)} not printed: {shown}")
    longs = [j for j in seed.jobs if j.lane == pq.Lane.LONG]
    stored = WAVE_ROWS[3]
    if [(j.block, j.item) for j in longs] != [(stored.block, stored.item)]:
        bad.append(f"long jobs {longs}, want ({stored.block}, {stored.item})'s alone")
        return bad
    [long] = longs
    kinds = [p.args.kind for p in got.planned]
    if kinds != ["steward", "long"]:
        bad.append(f"planned {kinds}, want a steward and one long run, no pool")
    names = [p.args.run for p in got.planned if not NAME.fullmatch(p.args.run)]
    if names:
        bad.append(f"planned runs whose names are no run names: {names}")
    steward = ("submit", "steward", "g-1-steward", "--at", SHA, "--wave", "g-1", "--board",
               PROJECT.cluster.board(), "--dry")
    mem, minutes = 1.5 * stored.peak_gb, math.ceil(1.5 * 1.25 * stored.minutes)
    run = ("submit", "long", f"g-1-long-{long.id}", "--at", SHA, "--wave", "g-1", "--preset",
           "cpu-score", "--job", str(long.id), "--node", long.node, "--mem", f"{mem:g}",
           "--minutes", str(minutes), "--dry")
    if [p.argv for p in got.planned] != [steward, run]:
        bad.append(f"planned {[p.argv for p in got.planned]}, want {[steward, run]}")
    elif got.planned[1].plan.resources != cs.Resources(cpus=2, mem_gb=int(mem),
                                                       time=cluster.slurm_time(minutes),
                                                       node=long.node):
        bad.append(f"the long run's resources {got.planned[1].plan.resources}")
    return bad


def _wave_seed(wave: str) -> str:
    """A seed of three cpu-score jobs of `wave`, as a submit ships it (the requests' case)."""
    header = pq.Header(wave=wave, commit=SHA, family="lineup-score", ladder=LADDERS["cpu-score"],
                       created="2026-10-21T10:00:00Z", feed=pq.FIXED)
    jobs = tuple(pq.JobSeed(id=k, block=f"b{k}", item="", group=f"b{k}", line="true",
                            node="orion", cls="m4", gb=1.0, minutes=10.0, peak_gb=1.0,
                            source="smoke", lane=pq.Lane.POOL) for k in (1, 2, 3))
    return pq.seed_json(pq.Seed(header=header, jobs=jobs, closed=True))


def _wave_requests(tmp: Path) -> tuple[list[str], str]:
    """The `wave` guard's requests (the module's docstring): its problems, and what it saw."""
    home = tmp / "home"
    ctx = _context(tmp / "repo", home)
    for r in (_steward_run("w", commit=SHA, finished=False, settled=False),
              _pool_run("w-quadro2-m12-s1", "w", node="quadro2", mem=12, finished=False),
              _run("w-orion-m4-s1", "pool", ("--wave", "w", "--preset", "cpu-score", "--node",
                                            "orion", "--mem", "4", "--gpus", "64"),
                   commit=SHA, width=1, finished=True, settled=True),
              _pool_run("w-orion-m12-s1", "w", node="orion", mem=12, finished=True),
              _run("w-long-3", "long", ("--wave", "w", "--preset", "cpu-score", "--job", "3",
                                        "--node", "orion", "--mem", "90", "--minutes", "300"),
                   commit=SHA, width=1, finished=False, settled=False),
              _pool_run("v-quadro2-m12-s1", "v", node="quadro2", mem=12, finished=False)):
        ctx.registry.write(r)
    inputs = home / cluster.WAVES / "w" / pool_steward.INPUTS
    inputs.mkdir(parents=True)
    (inputs / pool_steward.SEED).write_text(_wave_seed("w"))
    verbs: list[tuple[str, ...]] = []

    def verb(argv: Sequence[str]) -> int:
        """Every cancel recorded; the first fails (the others must be tried all the same)."""
        verbs.append(tuple(argv))
        return 1 if len(verbs) == 1 else 0

    edge = cluster_wave.Edge(ctx=ctx, verb=verb, rows=tmp, features=tmp / "features.csv",
                             records=tmp / "waves")
    bad: list[str] = []
    with contextlib.redirect_stdout(io.StringIO()):
        for argv in (["w", "retry", "2"], ["w", "retry", "failed"]):
            cluster_wave.main(argv, edge=edge)
        with contextlib.redirect_stderr(io.StringIO()):
            cancelled = cluster_wave.main(["w", "cancel"], edge=edge)
        if cancelled.code != 1:
            bad.append(f"cancel, its first cancel failing, exits {cancelled.code}, not 1")
        try:
            cluster_wave.main(["w", "retry", "9"], edge=edge)
            bad.append("retry 9 (no job of w's) requested")
        except SystemExit as e:
            if "9" not in str(e.code):
                bad.append(f"retry 9 refused without naming 9: {e.code}")
    folder = home / cluster.WAVES / "w" / pool_steward.REQUESTS
    files = sorted(p.name for p in folder.iterdir())
    read = [pool_steward.read_request((folder / f).read_text(), f) for f in files]
    want = [pool_steward.Retry(ids=(2,)), pool_steward.Retry(ids=(1, 2, 3)),
            pool_steward.Cancel()]
    if read != want or [f.rpartition("-")[2] for f in files] != ["retry.json", "retry.json",
                                                                  "cancel.json"]:
        bad.append(f"requests {files}: {read}, want {want}")
    cancels = sorted(verbs)
    want_cancels = [("cancel", r) for r in ("w-long-3", "w-orion-m12-s1", "w-quadro2-m12-s1")]
    if cancels != want_cancels:
        bad.append(f"cancel ran {cancels}, want the running pool lease's, the unsettled one's and "
                   "the long run's, every one though the first failed")
    return bad, f"requests {len(files)}, cancels {[c[1] for c in cancels]}"


def _uncommitted_check(repo: Path) -> list[str]:
    """cluster_wave._uncommitted on a git repo made in `repo`: BANK_CODE committed, none listed;
    common.py, a file below src/reblock and one below conf changed (and one outside BANK_CODE),
    those three."""
    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", "-c",
                        "commit.gpgsign=false", "-c", "user.name=guard", "-c",
                        "user.email=guard@localhost", *args], check=True, capture_output=True)

    for f in (COMMON, PRODUCER, CONFIG):
        (repo / f).parent.mkdir(parents=True)
        (repo / f).write_text("# committed\n")
    (repo / "other.py").write_text("")
    git("init", "-q")
    git("add", ".")
    git("commit", "-qm", "guard")
    bad = []
    if cluster_wave._uncommitted(repo, cluster_wave.BANK_CODE):
        bad.append("a committed common.py listed as uncommitted")
    for f in (COMMON, PRODUCER, CONFIG, "other.py"):
        (repo / f).write_text("# changed\n")
    got = cluster_wave._uncommitted(repo, cluster_wave.BANK_CODE)
    if sorted(got) != sorted([COMMON, PRODUCER, CONFIG]):
        bad.append(f"{COMMON}, {PRODUCER} and {CONFIG} changed (and another file): {got} "
                   "listed, want those three")
    return bad


#: wave: a bank written in a subprocess, which prints every file of this repo (not of its .venv)
#: a module it imported was loaded from (each module's spec: has_location, origin), as JSON.
BANK_PROBE = """
import json, sys
from pathlib import Path
from types import ModuleType
repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path[:0] = [str(repo / "research" / "roadless"), str(repo)]
import common
common.write_bank(sys.argv[3].split(","), out)
files = set()
for m in list(sys.modules.values()):
    if not isinstance(m, ModuleType) or m.__spec__ is None or not m.__spec__.has_location:
        continue
    f = Path(m.__spec__.origin).resolve()
    if f.is_relative_to(repo) and not f.is_relative_to(repo / ".venv"):
        files.add(f.relative_to(repo).as_posix())
print(json.dumps(sorted(files)))
"""


def _covered(path: str, code: Sequence[str]) -> bool:
    """Whether `path` (below the repo) is a path of `code` or below one."""
    p = Path(path)
    return any(p == Path(c) or Path(c) in p.parents for c in code)


def _bank_imports_check(tmp: Path) -> tuple[list[str], str]:
    """The `wave` guard's import case: a bank of WAVE_BLOCKS[1] written by common.write_bank in a
    subprocess; every file of this repo its modules were loaded from lies below a path of
    cluster_wave.BANK_CODE, common.py and a file below PRODUCERS among them (a probe that saw
    none of src/reblock, loaded from elsewhere, would pass whatever BANK_CODE says of it). Its
    problems, and what it saw."""
    tmp.mkdir()
    r = subprocess.run([sys.executable, "-I", "-c", BANK_PROBE, str(PROJECT.repo),
                        str(tmp / "bank.pkl"), WAVE_BLOCKS[1]], cwd=tmp, capture_output=True,
                       text=True)
    if r.returncode or not (tmp / "bank.pkl").is_file():
        return [f"the bank's probe failed ({r.returncode}): {r.stderr.strip()[-500:]}"], "no probe"
    files: list[str] = json.loads(r.stdout.splitlines()[-1])
    outside = [f for f in files if not _covered(f, cluster_wave.BANK_CODE)]
    bad = ([f"a bank's write imports {', '.join(outside)}, below no path of BANK_CODE"]
           if outside else [])
    if not any(f == COMMON for f in files):
        bad.append(f"the bank's probe imported no {COMMON}: {files}")
    if not any(_covered(f, (PRODUCERS,)) for f in files):
        bad.append(f"the bank's probe imported no file below {PRODUCERS}: {files}")
    return bad, (f"bank imports {len(files)} files of this repo, "
                 + (f"{len(outside)} outside BANK_CODE" if outside else "every one in BANK_CODE"))


def _probe_fault(old: str, new: str) -> contextlib.AbstractContextManager[None]:
    """BANK_PROBE with `old` replaced by `new` (the probe's faults: the bank is written in a
    subprocess, which no patch in this process reaches)."""
    if BANK_PROBE.count(old) != 1:
        raise AssertionError(f"BANK_PROBE has {old!r} {BANK_PROBE.count(old)} times, not once")
    return _patched(sys.modules[__name__], "BANK_PROBE", BANK_PROBE.replace(old, new))


def _clean(repo: Path, paths: Sequence[str]) -> list[str]:
    """cluster_wave._uncommitted answering none (the `wave` cases' common.py, committed)."""
    return []


def _dirty(repo: Path, paths: Sequence[str]) -> list[str]:
    """cluster_wave._uncommitted answering every path (the `wave` case of a dirty common.py)."""
    return list(paths)


def _wave_cases(edge: cluster_wave.Edge, planned: list[_Planned], *, tmp: Path, work: Path
                ) -> tuple[list[str], list[str]]:
    """The `wave` guard's command lines (the module's docstring): its problems, and what it saw."""
    bad: list[str] = []
    got: list[str] = []
    tars, scratch = tmp / "tars", str(tmp / "scratch")
    tars.mkdir()
    for blk in WAVE_BLOCKS:
        with tarfile.open(tars / f"{blk}.tar", "w"):
            pass
    score = [SCORE, "lineup-score", scratch, "{id}", "{item}"]
    a, b = WAVE_BLOCKS
    cpu = "cpu-score"
    main = _waved(edge, planned, _submit_line(
        "g-1", "--nodes", "orion,quadro2", "--items", "clearance,resistance_lp", "--unpack",
        str(tars / "{id}.tar"), preset=cpu, ids=WAVE_BLOCKS, command=score))
    bad += _main_wave(main, work=work, scratch=scratch)
    got.append("g-1 " + ("planned" if main.refused is None else "refused"))
    refusals = {   # case: (its command line, the words its refusal says)
        "misspelt": (_submit_line("g-2", "--nodes", "orion", "--items", "clearance",
                                  preset=cpu, ids=(b, MISTYPED), command=score),
                     (MISTYPED, "does not yield")),
        "no-script": (_submit_line("g-3", "--nodes", "orion", "--items", "clearance",
                                   preset=cpu, ids=(b,), command=(GONE, "{id}", "{item}")),
                      (SHA[:12],)),
        "item-unfilled": (_submit_line("g-4", "--nodes", "orion", preset=cpu, ids=(b,),
                                       command=score), ("{item}",)),
        "item-unused": (_submit_line("g-5", "--nodes", "orion", "--items", "clearance",
                                     preset=cpu, ids=(b,), command=score[:-1]), ("{item}",)),
        "tgz": (_submit_line("g-6", "--nodes", "orion", "--items", "clearance", "--unpack",
                             str(tars / "{id}.tgz"), preset=cpu, ids=(b,), command=score),
                (".tar",)),
        "gpu-on-orion": (_submit_line("g-7", "--nodes", "orion", "--items", "clearance",
                                      preset="gpu-presets", ids=(b,), command=score),
                         ("unusable",)),
        "fitted-past-7-days": (_submit_line("g-8", "--nodes", "orion,quadro2", "--items",
                                            "cycle_desire", preset=cpu, ids=WAVE_BLOCKS,
                                            command=score),
                               (a, b, "cycle_desire", "7 days")),
        "recorded": (_submit_line("g-10", "--nodes", "orion", "--items", "clearance",
                                  preset=cpu, ids=(b,), command=score),
                     ("waves/g-10", "record"))}
    (edge.records / "g-10").mkdir(parents=True)
    for case, (argv, words) in refusals.items():
        x = _waved(edge, planned, argv)
        if x.refused is None or x.planned:
            bad.append(f"{case}: planned {[p.args.run for p in x.planned]}, not refused")
        elif not all(w in x.refused for w in words):
            bad.append(f"{case}: refused without saying {words}: {x.refused}")
        got.append(f"{case} {'refused' if x.refused is not None else 'planned'}")
    for item, ids, source in (("slow_stored", (b,), "rows:"),
                              ("fitted_slow", WAVE_BLOCKS, "fit:item:")):
        wave = f"g-9-{item.replace('_', '-')}"
        capped = _waved(edge, planned, _submit_line(
            wave, "--nodes", "orion", "--items", item, preset=cpu, ids=ids, command=score))
        longs = [p for p in capped.planned if p.args.kind == "long"]
        seed = work / wave / pool_steward.SEED
        jobs = pq.read_seed(seed.read_text(), item).jobs if seed.is_file() else ()
        if (capped.refused is not None or len(longs) != len(ids)
                or any(p.args.minutes != 10080 or p.plan.resources.time != "7-00:00:00"
                       for p in longs)
                or [j.source[:len(source)] for j in jobs] != [source] * len(ids)):
            bad.append(f"{item}: {capped.refused or [p.argv for p in longs]} "
                       f"({[j.source for j in jobs]}), want {len(ids)} long runs of --minutes "
                       f"10080 (7 days), their estimates {source}")
        got.append(f"{item} " + ("refused" if capped.refused is not None
                                 else f"--minutes {[p.args.minutes for p in longs]}"))
    with _patched(cluster_wave, "_uncommitted", _dirty):
        dirty = _waved(edge, planned, _submit_line(
            "g-11", "--nodes", "orion", "--items", "clearance", preset=cpu, ids=(b,),
            command=score))
    listed = (dirty.refused or "").partition(": uncommitted changes")[0]  # the paths it names
    if dirty.refused is None or dirty.planned or listed.split(", ") != list(BANK_FILES):
        bad.append(f"the bank's code uncommitted: {dirty.refused or 'planned'}, want refused "
                   f"naming {', '.join(BANK_FILES)}")
    got.append(f"dirty bank code {'refused' if dirty.refused is not None else 'planned'}")
    return bad, got


def check_wave() -> list[str]:
    """`wave` (the module's docstring): the problems found."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = tmp / "rows"
        _wave_rows(rows)
        calls: list[list[str]] = []
        planned: list[_Planned] = []
        ctx = _offline(tmp / "repo", calls)
        edge = cluster_wave.Edge(ctx=ctx, verb=_planner(tmp / "plans", planned), rows=rows,
                                 features=rows / "consolidated_rows" / "summary_features.csv",
                                 records=rows / "waves")
        # this checkout's common.py may be uncommitted where the guard runs (another agent's
        # edit): the cases read it committed, and _uncommitted_check checks the check
        with _patched(cluster_wave, "_uncommitted", _clean):
            bad, got = _wave_cases(edge, planned, tmp=tmp,
                                   work=ctx.registry.root.parent / cluster_wave.WORK)
        if calls:
            bad.append(f"a dry wave called the cluster: {calls[:2]}")
        bad += _uncommitted_check(tmp / "git")
        imports, saw = _bank_imports_check(tmp / "probe")
        bad += imports
        got.append(saw)
        more, saw = _wave_requests(tmp / "requests")
        bad += more
        got.append(saw)
    print(f"wave: {'; '.join(got)} -> {'ok' if not bad else 'FAILS'}", flush=True)
    return bad


_JOB_LINE = cluster_wave._job_line


def _no_keep(line: str, *, block: str, item: str, tar: Path | None) -> str:
    """A job's line without --keep-standing (fault `no-keep`)."""
    return _JOB_LINE(line, block=block, item=item, tar=tar).removesuffix(
        f" {cluster_wave.KEEP}")


def _block_names(ctx: Context, wave: str, job: psum.LongJob) -> str:
    """A long run named for its job's block (fault `block-names`)."""
    return f"{wave}-long-{job.block}"


def _uncapped(minutes: float) -> int:
    """A long run's walltime uncapped (fault `uncapped`)."""
    return math.ceil(cluster_wave.LONG_TIME * minutes)


def _any_node(*, wave: str, ladder: Ladder, nodes: Sequence[str], commit: str,
              work: Path) -> None:
    """Every node taken as serving every ladder (fault `any-node`)."""


def _fit_margin(jobs: Sequence[pq.JobSeed]) -> list[pq.JobSeed]:
    """The Ruling as first written: a fitted job refused when 1.5 times its estimate passes 7
    days (fault `fit-margin`: fitted_slow refused)."""
    return [j for j in jobs if j.lane == pq.Lane.LONG and j.source.startswith(pp.FIT_SOURCE)
            and cluster_wave.LONG_TIME * j.minutes > cluster_wave.LONG_MAX_MINUTES]


GUARDS: dict[str, Callable[[], list[str]]] = {
    "ids": check_ids, "script": check_script, "demand": check_demand, "kinds": check_kinds,
    "more": check_more, "template": check_template, "wave": check_wave}
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
             "reap-first": lambda: _patched(cluster, "_registered", _registered_unreaped),
             "none-unreaped": lambda: _patched(cluster, "_unreaped",
                                               lambda killed, registered: [])},
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
                                          lambda self, market: tuple(market()))},
    "wave": {
        "no-keep": lambda: _patched(cluster_wave, "_job_line", _no_keep),
        "block-names": lambda: _patched(cluster_wave, "_long_name", _block_names),
        "no-script-check": _no_script_check,
        "no-check": _no_check,
        "fit-capped": lambda: _patched(cluster_wave, "_unbounded", lambda jobs: []),
        "uncapped": lambda: _patched(cluster_wave, "_long_minutes", _uncapped),
        "any-node": lambda: _patched(cluster_wave, "_servable", _any_node),
        "pools-only": lambda: _patched(cluster_wave, "CANCELLED_KINDS", (cluster.Pool.name,)),
        "fit-margin": lambda: _patched(cluster_wave, "_unbounded", _fit_margin),
        "unfinished-only": lambda: _patched(cluster_wave, "_undone",
                                            lambda rec: rec.finished_at is None),
        "first-cancel-stops": lambda: _patched(cluster_wave, "_every", cluster_wave._each),
        "record-reused": lambda: _patched(cluster_wave, "_recorded", lambda records, wave: None),
        "no-bank-code": lambda: _patched(cluster_wave, "BANK_CODE", ()),
        "common-only": lambda: _patched(cluster_wave, "BANK_CODE", (COMMON,)),
        "drop-regions": lambda: _patched(
            cluster_wave, "BANK_CODE",
            tuple(c for c in cluster_wave.BANK_CODE if c != "research/roadless/regions.py")),
        "clean-always": lambda: _patched(cluster_wave, "_uncommitted", _clean),
        "outside-import": lambda: _probe_fault("import common\n",
                                               "import common\nimport pool_ladder\n"),
        "producers-unseen": lambda: _probe_fault(
            'not f.is_relative_to(repo / ".venv")',
            'not f.is_relative_to(repo / ".venv") and not f.is_relative_to(repo / "src")')}}


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
