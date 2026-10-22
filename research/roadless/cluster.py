"""reblock's cluster launcher: the project and its two kinds of run, on GeoffChurch/cluster_submit.

`blocks` runs one task per block on a GPU: a script command with `{id}` in it, with the block
bank shipped as an input (a node has no source data; common.build_blocks reads the bank). The
tasks run the gap complex (gap_clear: cuDSS on the card, gap_factor.CuDSS), and `--gb`, the GPU
memory every task needs, routes each to the smallest card that holds it, or with `--node` to that
node's cards (a node's mesh cache is its own disk: a region's tasks pinned to one node build its
mesh once, NOTES "Exits as leaks: the region model on the complex (5810@major)"). (The
retired raster's tasks were sized from the cells of their grid, `--mesh` and `--scans`, at
GB_PER_MILLION GB per million unknowns: NOTES, "The complex as the runtime: the raster
retired".) With `--items`, one task per (block, item), the item in the command's `{item}` (a
region's configurations and presets: region_leaks). `--unpack` ships a tar per block of inputs
git does not carry (parquet rows, at their repo paths) and unpacks it in the task's checkout
before the command. Results (parquet rows) come back to the same paths here, never overwriting a
local file.

`cpu` runs one task per (block, item) on the CPUs of one node and no GPU: a script command with
`{id}` and `{item}` in it, for every block of the ids and every item of `--items`, so a block's
items (the arms of an experiment, say) spread over the node's cores rather than running one
after another in one task. No GPU sizing and no GPU request: every task is CPU only. The node is
required and is one of CPU_NODES (`--nodelist`: results can differ by machine, so a run says
where it ran, and every item of a block runs on the one node; two nodes are two runs); so are
the cores and memory of a task (`--cpus`, `--mem`). `--gpus N`, cluster_submit's throttle, is
how many of the run's tasks run at once: a run of CPU tasks holds no GPU, so the project's GPU
budget does not cap it, CPU_NODES does. The bank ships as for `blocks`, and `--unpack` too, but
each task unpacks its block's tar through cluster_unpack.py, which renames every file into place:
a block's tasks unpack one tar at once into the checkout they share.

    uv run python research/roadless/cluster.py setup
    uv run python research/roadless/cluster.py submit blocks <run> --gb G [--node N] [--time T] \
        [--gpus N] [--items <items,|@file>] [--with FILE ...] [--unpack TAR] <ids,|@file> \
        -- <script> <args with {id} [and {item}]>
    uv run python research/roadless/cluster.py submit cpu <run> --node orion|quadro2 --cpus C \
        --mem G --items <items,|@file> [--gpus N] [--time T] [--unpack TAR] <ids,|@file> \
        -- <script> <args with {id} and {item}>
    uv run python research/roadless/cluster.py submit steward <w>-steward --at <commit> \
        --wave <w> --board <board>
    uv run python research/roadless/cluster.py submit pool <w>-<node>-<class> --at <commit> \
        --wave <w> --preset <p> --node <node> --mem <GB> --gpus <throttle>
    uv run python research/roadless/cluster.py submit long <w>-long-<id> --at <commit> \
        --wave <w> --preset <p> --job <id> --node <node> --mem <GB> --minutes <m>
    uv run python research/roadless/cluster.py status|sync [<run>]
    uv run python research/roadless/cluster.py wait <run> [--timeout 3h]
        # exit 0 passed, 3 failed, 4 timed out, 5 blind
    uv run python research/roadless/cluster.py resubmit <run> [--gb G] [--time T]
        # its failed / missing blocks as <run>-r1
    uv run python research/roadless/cluster.py cancel <run>
    uv run python research/roadless/cluster.py watch start|stop
    uv run python research/roadless/cluster.py gpu      # the GPU market's last allocation
    uv run python research/roadless/cluster.py demand
        # state reblock's demand in the GPU market now (after submit, resubmit, sync, wait and
        # cancel it is stated anyway: cluster_demand)

A wave (docs/superpowers/specs/2026-10-21-lease-pool-design.md) is reblock's work as leases of
ai's GPU/CPU market. Its jobs, one command per (block, item), are rows of a queue on the cluster's
home (pool_queue), each on its block's node and in the memory class its estimate needs (a rung of
the wave's preset, pool_ladder.LADDERS); three kinds of run work it, every one at the wave's
commit:

- `pool`: a lease of identical workers of one wave, node and class (`pool.py work`), each
  claiming the longest job that fits the walltime it has left and retiring with 75 when none
  does. A pool run is reblock's lease in the market (`Waves.lease`: campaign CAMPAIGN, its node,
  its class's bundle). It is continued (`pool_more`) while its class has jobs pending on its node
  and every worker killed at its walltime held a job (GeoffChurch/mycooc's lesson 5), at the
  market's width capped by the jobs pending (`cs.launches.width`); the market starts a class's
  lease where it allocates the campaign more than its leases there hold (`Waves.template`): the
  pool of the live wave with the most jobs of that class pending on that node, at that wave's
  commit, a fresh chain `<w>-<node>-<class>-s<N>`.
- `steward`: one lease a wave on STEWARD_NODE, 2 CPUs and 3 GB (`pool.py steward`): every
  minute it reaps dead claims on squeue's evidence, asks workers to step aside for the wave's
  share of the market's yields, applies the requests of the `wave` verb and writes the wave's
  summary.json, which `pool_more`, `Waves.template` and the demand read; it ends the wave, and
  is continued (its own `more`) until then. Its pull (WaveRecord) mirrors the wave's summary,
  ledger and final queue into research/roadless/waves/<w>/ (git-ignored, as rows are).
- `long`: one job no lease can hold, run alone and sized for it (`pool.py work --job`).

The `wave` verb (cluster_wave.py) submits a wave whole (its steward and long runs: the market
starts its pools) and reads and steers it. A wave's state lives on the cluster, in
`<WAVES>/<w>/` below its home (the queue, the inputs shipped once, summary.json, the ledger, the
requests), never in a run's directory, which the keeper or `gc` releases once its run settles.

reblock states its demand in ai's GPU market as one campaign, CAMPAIGN (cluster_submit's README,
"GPU market"; cluster_demand.py): what its waves' tasks hold and wait for, so the market's keeper
charges it.

Before a CPU run, the node's load: `ssh ai sinfo -N -p defq -o '%N %c %T %C'` (cores allocated,
idle, other, total) and `scontrol show node <node>` (AllocMem of RealMemory).
"""
from __future__ import annotations

import argparse
import math
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import NamedTuple

import cluster_demand
import cluster_submit as cs
import pool
import pool_summary as psum
from cluster_submit.cli import VERB_NAMES, parser
from cluster_submit.context import stamped, utcnow
from cluster_submit.project import Source
from cluster_submit.registry import Registry
from cluster_submit.remote import Remote, SubprocessRunner
from cluster_submit.tree import blob_at
from pool_ladder import LADDERS, STEWARD, Ladder, class_name
from pool_steward import FINAL, LEDGER, SUMMARY
from pool_worker import Exit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
# The nodes a `cpu` run may go to, and the most of its tasks each runs at once (`--gpus`): the
# cluster's two nodes whose GPUs are unusable (sites/ai.toml) and whose CPUs are fine (owner
# 2026-10-12: "run the 13- and 60-block stages on orion/quadro2"). Slurm allocates whole cores
# (CR_CORE_MEMORY: a 1-cpu task holds both of a core's threads), so fewer run at once than these
# allow: 36 single-CPU tasks on orion, 32 on quadro2, less what others hold, and what --mem leaves.
CPU_NODES = {
    "orion": 64,        # 72 CPUs, 36 cores; idle on 2026-10-12
    "quadro2": 40,      # 64 CPUs, 32 cores; shared (another user's jobs: 8 CPUs, 320 of 376 GB)
}
# The shared env of uv.lock (cluster_submit's "Envs": one per lock, built once on the login node,
# reblock and ext/topology editable against each task's checkout): reblock has no extras, and
# pyproject's default-groups are every group, cupy and pyamg (`gpu`) among them, so it names none
# beside them.
ENV = cs.Uv(extras=(), groups=())
# Every task's environment: single-threaded, unbuffered, no bytecode (cluster_env.sh).
TASK_ENV = "source research/roadless/cluster_env.sh"
# A `blocks` or `cpu` task's: its environment and the run's shipped bank (common.build_blocks). A
# wave's runs ship none: pool.py hands each job the wave's bank.
PRELUDE = (TASK_ENV, 'export REBLOCK_BLOCK_BANK="$CLUSTER_SUBMIT_INPUTS/bank.pkl"')


def _listed(arg: str) -> list[str]:
    """A comma-separated list, or @file of whitespace-separated entries. Block ids are checked
    where the plan writes the bank (common.write_bank: an id the source does not yield raises,
    before any task is staged)."""
    return Path(arg[1:]).read_text().split() if arg.startswith("@") else arg.split(",")


def _command(args: argparse.Namespace, *placeholders: str) -> str:
    """The task command after `--`, with every placeholder in it."""
    # REMAINDER (which drops the `--`) swallows an option placed after the ids
    command: list[str] = args.command
    if not command or command[0].startswith("-"):
        raise SystemExit(f"usage: submit {args.kind} <run> [options] <ids> -- <command>")
    line = " ".join(command)
    for p in placeholders:
        if p not in line:
            raise SystemExit(f"the command needs {p}")
    return line


def _script_at(args: argparse.Namespace, sha: str) -> None:
    """The command's script is a file of the run's commit: read there, from git, never the
    working tree (cluster_submit's "A plan reads the repo at its commit")."""
    script = args.command[0]
    if blob_at(SubprocessRunner(), REPO, sha, script) is None:
        raise SystemExit(f"{script} is not a file at {sha[:12]}, the commit the run's tasks run")


def _unpacks(unpack: str | None, ids: list[str]) -> dict[str, Path]:
    """`--unpack`'s tar per block (its path with {id}), each present; none without it."""
    if unpack is None:
        return {}
    if "{id}" not in Path(unpack).name:
        raise SystemExit("--unpack needs {id} in the tar's name")
    tars = {b: Path(unpack.replace("{id}", b)) for b in ids}
    absent = [str(t) for t in tars.values() if not t.is_file()]
    if absent:
        raise SystemExit(f"--unpack: {len(absent)} tars absent, e.g. {absent[0]}")
    return tars


def _unpack_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--unpack", metavar="TAR",
                        help="a tar per block, its path with {id}: shipped beside the bank "
                             "and unpacked in the task's checkout before its command (inputs "
                             "git does not carry, at their repo paths)")


@dataclass(frozen=True)
class Blocks:
    name: str = "blocks"
    env: cs.Env = ENV
    pull: cs.IntoTree = cs.IntoTree("research/roadless", ("*.parquet",))
    routine_exits: frozenset[int] = frozenset()     # every non-zero exit is a failure

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("ids", help="block ids, comma-separated, or @file of ids")
        parser.add_argument("--time", help="sbatch --time (default: the cluster's)")
        parser.add_argument("--gb", type=float, required=True,
                            help="the GPU memory each task needs, in GB: its card's size")
        parser.add_argument("--node", help="the node every task runs on (--nodelist; its card "
                                           "holds --gb): unpinned, the smallest card that does")
        parser.add_argument("--items", help="items, comma-separated, or @file: one task per "
                                            "(block, item), the item in the command's {item}")
        parser.add_argument("--with", dest="extra", type=Path, action="append", default=[],
                            help="a file shipped beside the block bank (repeatable); a task "
                                 "reads it as $CLUSTER_SUBMIT_INPUTS/<its name>")
        _unpack_argument(parser)
        parser.add_argument("command", nargs=argparse.REMAINDER,
                            help="-- <script> <args with {id}>")

    def plan(self, args: argparse.Namespace, workdir: Path, *, sha: str) -> cs.RunPlan:
        import common  # geopandas and the block sources: only a submit needs them

        items = [] if args.items is None else _listed(args.items)
        line = _command(args, "{id}", *(["{item}"] if items else []))
        _script_at(args, sha)
        ids = _listed(args.ids)
        if args.gb <= 0:
            raise SystemExit(f"--gb {args.gb}: a task's GPU memory is positive")
        if len(set(items)) != len(items) or not all(items):
            raise SystemExit(f"--items: empty or repeated entries in {items}")
        unpack = _unpacks(args.unpack, ids)

        def task_line(b: str, item: str) -> str:
            run = f"python -u {line.replace('{id}', b).replace('{item}', item)}"
            if args.unpack is None:
                return run
            return f'tar -xf "$CLUSTER_SUBMIT_INPUTS/{unpack[b].name}" && {run}'

        common.write_bank(ids, workdir / "bank.pkl")
        # without items the command has no {item}: one task a block, its line as it is
        tasks = tuple(cs.Task(task_line(b, item), args.gb) for b in ids
                      for item in (items or ["{item}"]))
        return cs.RunPlan(
            tasks=tasks, resources=cs.Resources(cpus=4, mem_gb=64, time=args.time, node=args.node),
            inputs=(workdir / "bank.pkl", *args.extra, *unpack.values()),
            prelude=PRELUDE, describe=(f"{len(ids)} blocks: {line}",))

    def more(self, ctx: cs.Context, rec: cs.RunRecord, *,
             killed: tuple[str, ...]) -> cs.Continue | cs.Done:
        return cs.Done("never asked: no exit of its is routine (cluster_submit's \"Leases\")")


@dataclass(frozen=True)
class Cpu:
    """One task per (block, item), CPU only, on one node of CPU_NODES (the module's docstring)."""

    name: str = "cpu"
    env: cs.Env = ENV
    pull: cs.IntoTree = cs.IntoTree("research/roadless", ("*.parquet",))
    routine_exits: frozenset[int] = frozenset()     # every non-zero exit is a failure

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("ids", help="block ids, comma-separated, or @file of ids")
        parser.add_argument("--node", required=True, choices=sorted(CPU_NODES),
                            help="the node every task runs on (--nodelist)")
        parser.add_argument("--cpus", type=int, required=True,
                            help="cores per task (sbatch --cpus-per-task; 1 for a "
                                 "single-threaded task)")
        parser.add_argument("--mem", type=int, required=True,
                            help="memory per task, GB (sbatch --mem)")
        parser.add_argument("--items", required=True,
                            help="items, comma-separated, or @file: one task per (block, item), "
                                 "the item in the command's {item}")
        parser.add_argument("--time", help="sbatch --time (default: the cluster's)")
        _unpack_argument(parser)
        parser.add_argument("command", nargs=argparse.REMAINDER,
                            help="-- <script> <args with {id} and {item}>")

    def plan(self, args: argparse.Namespace, workdir: Path, *, sha: str) -> cs.RunPlan:
        import common  # geopandas and the block sources: only a submit needs them

        line = _command(args, "{id}", "{item}")
        _script_at(args, sha)
        if args.cpus < 1 or args.mem < 1:
            raise SystemExit(f"--cpus and --mem are at least 1: {args.cpus}, {args.mem}")
        at_once = CPU_NODES[args.node]
        if args.gpus > at_once:
            raise SystemExit(f"--gpus {args.gpus}: at most {at_once} tasks at once on "
                             f"{args.node} (CPU_NODES)")
        ids, items = _listed(args.ids), _listed(args.items)
        for what, names in (("ids", ids), ("items", items)):
            if len(set(names)) != len(names) or not all(names):
                raise SystemExit(f"--{what}: empty or repeated entries in {names}")
        unpack = _unpacks(args.unpack, ids)

        def task_line(b: str, item: str) -> str:
            run = f"python -u {line.replace('{id}', b).replace('{item}', item)}"
            if args.unpack is None:
                return run
            return (f'python -u research/roadless/cluster_unpack.py '
                    f'"$CLUSTER_SUBMIT_INPUTS/{unpack[b].name}" && {run}')

        common.write_bank(ids, workdir / "bank.pkl")
        tasks = tuple(cs.Task(task_line(b, item), 0.0) for b in ids for item in items)
        return cs.RunPlan(
            tasks=tasks,
            resources=cs.Resources(cpus=args.cpus, mem_gb=args.mem, time=args.time,
                                   node=args.node),
            inputs=(workdir / "bank.pkl", *unpack.values()), prelude=PRELUDE,
            describe=(f"{len(ids)} blocks x {len(items)} items on {args.node}: {line}",))

    def more(self, ctx: cs.Context, rec: cs.RunRecord, *,
             killed: tuple[str, ...]) -> cs.Continue | cs.Done:
        return cs.Done("never asked: no exit of its is routine (cluster_submit's \"Leases\")")


# ---- the lease pool's kinds (docs/superpowers/specs/2026-10-21-lease-pool-design.md) ----------

#: The waves' home, below the cluster home (pool.py's, the one definition): each wave's queue,
#: inputs, summary, ledger and final queue are in <WAVES>/<w>/ (the spec's 3.2).
WAVES = pool.WAVES
#: Where every wave's steward runs (the plan's Ruling 11): a node of CPU_NODES.
STEWARD_NODE = "orion"
#: A steward's lease, the active tier's 2 h (Ruling 9): it retires with 75 before its end, and its
#: own `more` continues it while its wave is live.
STEWARD_LEASE_MINUTES = 120.0
#: What a GPU pool's or long run's task asks of its card, GB: the largest GPU peak stored is 7.6 GB
#: and every usable card holds 32, so the class varies host memory only (the spec's 3.4); --node
#: picks the card.
CARD_GB = 8.0
#: The longest a long run may ask for (sbatch --time): the partition's 7 days (the spec's 3.8).
LONG_MAX_MINUTES = 7 * 24 * 60.0
#: Where the steward's pull mirrors a wave's record, below the repo (the spec's 3.10, Ruling 14),
#: and what of <WAVES>/<w>/ it mirrors, those present.
WAVE_RECORDS = "research/roadless/waves"
WAVE_RECORD_FILES = (SUMMARY, LEDGER, FINAL)


def slurm_time(minutes: float) -> str:
    """`minutes`, rounded up to the second, as sbatch --time: H:MM:SS under a day, D-HH:MM:SS
    from one (Slurm reads both)."""
    s = math.ceil(minutes * 60)
    days, s = divmod(s, 24 * 3600)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{days}-{h:02d}:{m:02d}:{s:02d}" if days else f"{h}:{m:02d}:{s:02d}"


def _wave(name: str) -> str:
    """A wave's name (pool.WAVE_NAME: its directory's), or a SystemExit."""
    if not pool.WAVE_NAME.fullmatch(name):
        raise SystemExit(f"--wave {name!r} is not a wave's name ({pool.WAVE_NAME.pattern})")
    return name


def _rung_checked(ladder: Ladder, mem: int) -> None:
    """`mem` is a rung of `ladder`, else a SystemExit naming its rungs."""
    if mem not in ladder.rungs:
        raise SystemExit(f"--mem {mem}: no rung of {ladder.name} "
                         f"({', '.join(str(r) for r in ladder.rungs)} GB)")


def throttle(ladder: Ladder, node: str) -> int:
    """A pool lease's `--gpus` (Ruling 10), so a lease the market widens runs whole: a CPU
    class's, on a node of CPU_NODES, that node's; a GPU class's, the project's budget. A CPU class
    on another node is refused (SystemExit)."""
    if ladder.cards:
        return PROJECT.max_gpus
    if node not in CPU_NODES:
        raise SystemExit(f"a CPU class's pool runs on a node of CPU_NODES "
                         f"({', '.join(sorted(CPU_NODES))}), not {node}")
    return CPU_NODES[node]


def _preset_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--preset", required=True, choices=sorted(LADDERS),
                        help="the wave's ladder (pool_ladder.LADDERS): its cards, CPUs and "
                             "rungs, and its lease's length")


@dataclass(frozen=True)
class Pool:
    """A lease of a wave's pool: identical workers of one node and class (the module's
    docstring)."""

    name: str = "pool"
    env: cs.Env = ENV
    pull: cs.IntoTree = cs.IntoTree("research/roadless", ("*.parquet",))
    routine_exits: frozenset[int] = frozenset({int(Exit.RETIRED)})   # retired: 75

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--wave", required=True, help="the wave whose jobs its workers claim")
        _preset_argument(parser)
        parser.add_argument("--node", required=True,
                            help="the node every worker runs on, whose jobs it claims")
        parser.add_argument("--mem", type=int, required=True,
                            help="the class: a rung of the preset, GB a worker")

    def plan(self, args: argparse.Namespace, workdir: Path, *, sha: str) -> cs.RunPlan:
        ladder = LADDERS[args.preset]
        wave = _wave(args.wave)
        _rung_checked(ladder, args.mem)
        at_once = throttle(ladder, args.node)
        if args.gpus > at_once:
            raise SystemExit(f"--gpus {args.gpus}: at most {at_once} workers at once of a "
                             f"{ladder.name} pool on {args.node} (Ruling 10)")
        cls = class_name(ladder.cards, args.mem)
        line = (f"python -u research/roadless/pool.py work --wave {wave} --node {args.node} "
                f"--class {cls}")
        return cs.RunPlan(
            tasks=(cs.Task(line, CARD_GB if ladder.cards else 0.0),),
            resources=cs.Resources(cpus=ladder.cpus, mem_gb=args.mem,
                                   time=slurm_time(ladder.lease_minutes), node=args.node),
            inputs=(), prelude=(TASK_ENV,),
            describe=(f"a lease of wave {wave}'s pool of {cls} ({ladder.name}) on {args.node}: "
                      f"{line}",))

    def more(self, ctx: cs.Context, rec: cs.RunRecord, *,
             killed: tuple[str, ...]) -> cs.Continue | cs.Done:
        return pool_more(ctx, rec, killed=killed)


@dataclass(frozen=True)
class WaveRecord:
    """The steward's pull: its wave's summary, ledger and final queue (those present) mirrored
    from <WAVES>/<w>/ into WAVE_RECORDS/<w>/ here, each replacing what is there (the wave's latest
    state, not write-once results). It leaves nothing on the cluster that is not the wave's own."""

    def pull(self, remote: Remote, src: Source) -> None:
        wave: str = src.args.wave
        here = src.repo / WAVE_RECORDS / wave
        here.mkdir(parents=True, exist_ok=True)
        remote.rsync("-a", *(f"--include={f}" for f in WAVE_RECORD_FILES), "--exclude=*",
                     f"{src.host}:{WAVES}/{wave}/", f"{here}/")

    def unpulled(self, remote: Remote, src: Source) -> tuple[str, ...]:
        return ()           # a mirror: nothing is left unpulled for a local file's sake

    def retained(self) -> tuple[str, ...]:
        return ()           # the wave's state is in <WAVES>/<w>/, no run directory's


@dataclass(frozen=True)
class Steward:
    """A wave's steward: one lease on STEWARD_NODE (the module's docstring)."""

    name: str = "steward"
    env: cs.Env = ENV
    pull: WaveRecord = WaveRecord()
    routine_exits: frozenset[int] = frozenset({int(Exit.RETIRED)})   # retired: 75

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--wave", required=True, help="the wave it stewards")
        parser.add_argument("--board", required=True,
                            help="the market's board, where it reads the allocation (relative: "
                                 "below the cluster home; PROJECT.cluster.board())")

    def plan(self, args: argparse.Namespace, workdir: Path, *, sha: str) -> cs.RunPlan:
        wave = _wave(args.wave)
        market = PROJECT.gpu_market
        if market is None:
            raise SystemExit("reblock is in no GPU market: a steward has no allocation to read")
        minutes = market.stale_after.total_seconds() / 60
        line = (f"python -u research/roadless/pool.py steward --wave {wave} "
                f"--board {shlex.quote(args.board)} --stale-after-minutes {minutes:g}")
        return cs.RunPlan(
            tasks=(cs.Task(line, 0.0),),
            resources=cs.Resources(cpus=STEWARD.cpus, mem_gb=math.ceil(STEWARD.mem_gb),
                                   time=slurm_time(STEWARD_LEASE_MINUTES), node=STEWARD_NODE),
            inputs=(), prelude=(TASK_ENV,),
            describe=(f"wave {wave}'s steward on {STEWARD_NODE}: {line}",))

    def more(self, ctx: cs.Context, rec: cs.RunRecord, *,
             killed: tuple[str, ...]) -> cs.Continue | cs.Done:
        """Continued, one task, while its wave is live (its summary OPEN or CLOSED); Done once
        the wave has ended; raises when its summary does not read (asked again next sync)."""
        wave = ctx.parse(rec.argv).wave
        s = wave_summary(ctx, wave)
        if s.state not in psum.LIVE:
            return cs.Done(f"wave {wave} is {s.state.value} (its summary of {s.at})")
        return cs.Continue(workers=1)


@dataclass(frozen=True)
class Long:
    """One job of a wave no lease can hold, run alone and sized for it (the spec's 3.8)."""

    name: str = "long"
    env: cs.Env = ENV
    pull: cs.IntoTree = cs.IntoTree("research/roadless", ("*.parquet",))
    routine_exits: frozenset[int] = frozenset()     # every non-zero exit is a failure

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--wave", required=True, help="the job's wave")
        _preset_argument(parser)
        parser.add_argument("--job", type=int, required=True, help="the job's id in its queue")
        parser.add_argument("--node", required=True, help="the job's node (--nodelist)")
        parser.add_argument("--mem", type=int, required=True, help="its memory, GB (sbatch --mem)")
        parser.add_argument("--minutes", type=float, required=True,
                            help=f"its walltime, minutes (sbatch --time; at most "
                                 f"{LONG_MAX_MINUTES:g}, the partition's 7 days)")

    def plan(self, args: argparse.Namespace, workdir: Path, *, sha: str) -> cs.RunPlan:
        ladder = LADDERS[args.preset]
        wave = _wave(args.wave)
        if args.mem < 1:
            raise SystemExit(f"--mem {args.mem}: a run's memory is at least 1 GB")
        if not 0 < args.minutes <= LONG_MAX_MINUTES:
            raise SystemExit(f"--minutes {args.minutes:g}: more than 0 and at most "
                             f"{LONG_MAX_MINUTES:g} (the partition's 7 days)")
        line = f"python -u research/roadless/pool.py work --wave {wave} --job {args.job}"
        return cs.RunPlan(
            tasks=(cs.Task(line, CARD_GB if ladder.cards else 0.0),),
            resources=cs.Resources(cpus=ladder.cpus, mem_gb=args.mem,
                                   time=slurm_time(args.minutes), node=args.node),
            inputs=(), prelude=(TASK_ENV,),
            describe=(f"wave {wave}'s job {args.job} on {args.node}, alone: {line}",))

    def more(self, ctx: cs.Context, rec: cs.RunRecord, *,
             killed: tuple[str, ...]) -> cs.Continue | cs.Done:
        return cs.Done("never asked: no exit of its is routine (cluster_submit's \"Leases\")")


# ---- a wave as the market's leases see it -----------------------------------------------------

def _live(rec: cs.RunRecord) -> bool:
    """Whether steward run `rec` keeps its wave live: unfinished, or finished and not yet settled
    (whether its own `more` continues it is undecided: the sync that finished it may decide its
    pool leases first, or its `more` may have raised). Settled, its continuation (if any) is the
    run that does (the controller's ruling, 2026-10-21)."""
    return rec.finished_at is None or not rec.settled


def live_stewards(ctx: cs.Context) -> dict[str, cs.RunRecord]:
    """The live waves of this registry, by name: each wave's live steward run (`_live`). An
    unreadable record is refused (SystemExit): a steward among them would go unseen, and a Done
    on its wave's leases is final. Two live stewards of a wave at two commits are refused."""
    recs, unreadable = ctx.registry.all()
    if unreadable:
        raise SystemExit(f"unreadable records: {', '.join(unreadable)}: a wave's steward among "
                         "them would go unseen")
    live: dict[str, cs.RunRecord] = {}
    for r in recs:
        if r.kind != Steward.name or not _live(r):
            continue
        wave: str = ctx.parse(r.argv).wave
        if wave in live and live[wave].commit != r.commit:
            raise SystemExit(f"wave {wave} has live stewards at two commits: "
                             f"{live[wave].name} at {live[wave].commit[:12]}, {r.name} at "
                             f"{r.commit[:12]}")
        live[wave] = r
    return live


def wave_summaries(ctx: cs.Context, waves: Sequence[str]) -> psum.Summaries:
    """The summaries of `waves` on the cluster, read with one ssh call (pool_summary's
    `read_summaries`)."""
    return psum.read_summaries(ctx.remote(), {w: f"{WAVES}/{w}/{SUMMARY}" for w in waves})


def wave_summary(ctx: cs.Context, wave: str) -> psum.Summary:
    """`wave`'s summary on the cluster; raises (SystemExit) when it is absent or does not read
    (asked again next sync)."""
    got = wave_summaries(ctx, [wave])
    if wave not in got.read:
        raise SystemExit(f"wave {wave}'s summary does not read ({got.unread[wave]}); the next sync "
                         "asks again")
    return got.read[wave]


def _walltime_reaps(s: psum.Summary, run: str) -> dict[str, psum.ReapedWorker]:
    """The reaps in `s` of `run`'s workers at or past their walltime, by task: a requeued task's
    earlier self, reaped when it registered again (its node lost, before its walltime), is not
    its walltime kill's."""
    return {r.task: r for r in s.reaped if r.run == run and r.walltime}


def _registered(s: psum.Summary, run: str) -> frozenset[str]:
    """The tasks of `run` registered in `s` and neither retired nor reaped yet."""
    return frozenset(w.task for w in s.registered if w.run == run)


def _unreaped(killed: Sequence[str], reaps: Mapping[str, psum.ReapedWorker],
              registered: frozenset[str]) -> list[str]:
    """The tasks of `killed` registered and not yet reaped at their walltime: whether they held a
    job is not known yet."""
    return [t for t in killed if t not in reaps and t in registered]


def _idle_killed(killed: Sequence[str], reaps: Mapping[str, psum.ReapedWorker]) -> list[str]:
    """The tasks of `killed` (none of them unreaped) that held no job: reaped idle, or never
    registered (hung in its startup)."""
    return [t for t in killed if t not in reaps or not reaps[t].held_job]


def pool_more(ctx: cs.Context, rec: cs.RunRecord, *,
              killed: tuple[str, ...]) -> cs.Continue | cs.Done:
    """Whether pool lease `rec`, finished, is continued (Pool.more; the spec's 3.7), read from its
    wave's summary (one ssh call), in this order:
    - no live steward of its wave in this registry (`live_stewards`): Done (its wave is over
      here);
    - its summary absent, unreadable or of another format: raise (asked again next sync);
    - the summary written before the lease left squeue (`gone_at`: its last reaps may predate the
      lease's end), or a task of `killed` (at its walltime) registered and not yet reaped at its
      walltime, by this run (`_walltime_reaps`: a requeued task's earlier reap is not it): raise;
    - a task of `killed` reaped holding no job, or never registered (hung in its startup): Done
      (GeoffChurch/mycooc's lesson 5: a lease whose worker sat idle to its walltime has nothing
      to continue for);
    - no job of its class pending on its node: Done;
    - else Continue at the market's width (`cs.launches.width`; in shadow, its own width), at
      most the jobs pending."""
    a = ctx.parse(rec.argv)
    wave: str = a.wave
    if wave not in live_stewards(ctx):
        return cs.Done(f"no steward of wave {wave} runs from this registry: its wave is over here")
    s = wave_summary(ctx, wave)
    if rec.gone_at is None:
        raise SystemExit(f"{rec.name} is finished with no time it left squeue: its summary's age "
                         "cannot be judged")
    if stamped(s.at) < stamped(rec.gone_at):
        raise SystemExit(f"wave {wave}'s summary is of {s.at}, before {rec.name} left squeue "
                         f"({rec.gone_at}): its steward may not have reaped it yet; the next sync "
                         "asks again")
    reaps = _walltime_reaps(s, rec.name)
    unreaped = _unreaped(killed, reaps, _registered(s, rec.name))
    if unreaped:
        raise SystemExit(f"{rec.name}'s tasks {', '.join(unreaped)}, killed at their walltime, are "
                         f"not yet reaped (wave {wave}'s summary of {s.at}): whether they held a "
                         "job is not known; the next sync asks again")
    idle = _idle_killed(killed, reaps)
    if idle:
        return cs.Done(f"its tasks {', '.join(idle)} were killed at their walltime holding no job "
                       "(idle, or never registered)")
    ladder = LADDERS[a.preset]
    cls = class_name(ladder.cards, a.mem)
    pending = sum(x.pending for x in s.lines if (x.node, x.cls) == (a.node, cls))
    if not pending:
        return cs.Done(f"wave {wave} has no job of {cls} pending on {a.node} (its summary of "
                       f"{s.at})")
    c = cs.launches.width(ctx, rec, today=lambda: cs.Continue(
        workers=min(cs.launches.width_of(rec), pending)))
    if isinstance(c, cs.Done):
        return c
    return cs.Continue(workers=min(c.workers, pending))


class _Choice(NamedTuple):
    """A live wave with jobs of a start's class pending on its node."""

    wave: str
    pending: int
    ladder: Ladder
    rung: int


def _chosen(choices: Sequence[_Choice]) -> _Choice:
    """The wave with the most jobs pending, ties to the earlier name (Ruling 4)."""
    return min(choices, key=lambda c: (-c.pending, c.wave))


def _start_name(ctx: cs.Context, wave: str, node: str, cls: str) -> str:
    """A started lease's name, `<w>-<node>-<class>-s<N>`, N the first this registry lacks: never a
    run it has, nor a continuation's (`-l<N>`)."""
    n = 1
    while ctx.registry.exists(f"{wave}-{node}-{cls}-s{n}"):
        n += 1
    return f"{wave}-{node}-{cls}-s{n}"


def _pool_args(*, wave: str, ladder: Ladder, node: str, rung: int) -> tuple[str, ...]:
    """A pool's submit arguments after its run's name (no --at: the template's commit is)."""
    return ("--wave", wave, "--preset", ladder.name, "--node", node, "--mem", str(rung),
            "--gpus", str(throttle(ladder, node)))


@dataclass(frozen=True)
class Waves:
    """reblock's campaign as the market's launches see it (cluster_submit's Campaigns, README "GPU
    market"; its docs/migrating/reblock.md, "a campaign names the template of a lease the market
    starts"): its pool runs are its leases; its stewards and long runs, and the `blocks` and
    `cpu` runs, are none. `registry` is the project's (every verb's Context has the same), which
    `starts`, given no Context, reads for a live steward."""

    registry: Registry

    def lease(self, ctx: cs.Context, rec: cs.RunRecord) -> cs.Lease | None:
        """A pool run: CAMPAIGN's lease on its --node, each worker its class's bundle."""
        if rec.kind != Pool.name:
            return None
        a = ctx.parse(rec.argv)
        return cs.Lease(CAMPAIGN, a.node, LADDERS[a.preset].bundle(a.mem))

    def template(self, ctx: cs.Context, campaign: str, node: str,
                 bundle: cs.Bundle) -> cs.Template | cs.Done:
        """The pool lease the market starts for `bundle` on `node` (Ruling 4): of the live waves
        (`live_stewards`), their summaries read with one ssh call, those whose ladder has a rung
        of `bundle` and whose summary has jobs of that class pending on `node`, the one with the
        most (ties to the earlier name), at its steward's commit, a fresh chain (`_start_name`),
        its throttle Ruling 10's. A live wave whose summary does not read, or names a ladder this
        machine's LADDERS lacks, is left out, and named in a Done's why. Done when no live wave
        has such jobs, for another campaign, and with no live wave; raises when no live wave's
        summary reads with a ladder of LADDERS (the next sync asks again)."""
        if campaign != CAMPAIGN:
            return cs.Done(f"reblock's one campaign is {CAMPAIGN}, not {campaign}")
        live = live_stewards(ctx)
        if not live:
            return cs.Done(f"no wave is live in this registry (no unfinished steward), so none "
                           f"has a job of {bundle.text()} on {node}")
        got = wave_summaries(ctx, sorted(live))
        left_out = {**got.unread, **{
            w: (f"it names the ladder {s.ladder!r}, none of this machine's "
                f"({', '.join(sorted(LADDERS))}): its pools cannot be planned here")
            for w, s in got.read.items() if s.ladder not in LADDERS}}
        usable = {w: s for w, s in got.read.items() if w not in left_out}
        if not usable:
            unread = "; ".join(f"wave {w}'s: {why}" for w, why in sorted(left_out.items()))
            raise SystemExit(f"no live wave's summary reads ({unread}): the next sync asks again")
        choices: list[_Choice] = []
        for w, s in sorted(usable.items()):
            ladder = LADDERS[s.ladder]
            for rung in ladder.rungs:
                cls = class_name(ladder.cards, rung)
                pending = sum(x.pending for x in s.lines if (x.node, x.cls) == (node, cls))
                if ladder.bundle(rung) == bundle and pending:
                    choices.append(_Choice(wave=w, pending=pending, ladder=ladder, rung=rung))
        if not choices:
            unread = "".join(f"; wave {w}'s summary left out: {why}"
                             for w, why in sorted(left_out.items()))
            return cs.Done(f"no live wave has a job of {bundle.text()} pending on {node}{unread}")
        c = _chosen(choices)
        cls = class_name(c.ladder.cards, c.rung)
        return cs.Template(kind=Pool.name, run=_start_name(ctx, c.wave, node, cls),
                           commit=live[c.wave].commit,
                           args=_pool_args(wave=c.wave, ladder=c.ladder, node=node, rung=c.rung))

    def starts(self, market: Callable[[], Sequence[cs.Start]]) -> tuple[cs.Start, ...]:
        """The market's starts on freed capacity, while a wave is live (a steward of `registry`'s
        `_live`, or a record that does not read: the market's own reading of the registry refuses
        it); none, and no allocation read, while none is: no template could answer one, and a
        stale allocation would fail every sync."""
        recs, unreadable = self.registry.all()
        if not unreadable and not any(r.kind == Steward.name and _live(r) for r in recs):
            return ()
        return tuple(market())


# sacct_grace: how long a run is out of squeue before a task of it with no exit code and no sacct
# row is taken as cancelled before it started (cluster_submit's "Settling a run": 10 minutes, the
# cluster's accounting lag many times over; cx-raster, 2026-10-15, had 19 such tasks).
# attribute_cache: how long ai's login node may answer a stat from cached attributes (its NFSv3
# home's acregmax and acdirmax, the client's defaults: 60 s), so a run is finished, with its last
# pull, only once it has been out of squeue that long (cluster_submit's "Settling a run": mycooc's
# ev2-lib-20k-c, finished a second after it left squeue, lost its last row so, 2026-10-21).
# market: the disk market of ai's home (its keeper holds reblock's files once a launch publishes
# them). checkout_setup: nothing to link into a checkout. gpu_market: ai's GPU market (owner
# 2026-10-21, at GeoffChurch/mycooc's asking): reblock states its demand there (cluster_demand,
# after the verbs of STATES_DEMAND), so the keeper attributes what its tasks hold and charges it.
# Never `keeper install` from here: the keeper is GeoffChurch/mycooc's to install, and the newest
# install wins (one from here would deploy this pin's cluster_submit, market file and GPU file over
# mycooc's). campaigns: Waves, whose pool runs are its leases (the market widens and starts them
# in active mode), and none of whose other runs (a steward, a long run, a `blocks` or `cpu` run)
# is one.
PROJECT = cs.Project(name="reblock", repo=REPO, cluster=cs.load_cluster("ai"),
                     kinds=(Blocks(), Cpu(), Pool(), Steward(), Long()),
                     max_gpus=4,   # owner, 2026-10-03: a shared cluster
                     checkout_setup=(), sacct_grace=timedelta(minutes=10),
                     attribute_cache=timedelta(seconds=60),
                     market=cs.load_market("ai"), gpu_market=cs.load_gpu_market("ai"),
                     campaigns=Waves(registry=Registry(REPO)))

# reblock's one campaign in the GPU market (the controller's ruling, 2026-10-21: one for the
# project, so its spend accumulates over its waves) is named for the project, and its weight among
# reblock's campaigns, the only one, is 1 (the project's own weight, against mycooc's and
# bookgen's, is the market file's).
CAMPAIGN = PROJECT.name
CAMPAIGN_WEIGHT = 1.0
# The verbs after which reblock states its demand: those that submit a run or change what the
# registry knows of one (a dry submit or resubmit submits nothing, and is left out). The watch
# timer runs `sync`, so the demand is refreshed every 150 s while a run is unfinished.
STATES_DEMAND = frozenset({"submit", "resubmit", "sync", "wait", "cancel"})
if not STATES_DEMAND <= set(VERB_NAMES):
    raise SystemExit(f"STATES_DEMAND names verbs cluster_submit has not: "
                     f"{sorted(STATES_DEMAND - set(VERB_NAMES))}")


def state_demand(registry: Registry) -> int:
    """The project's demand (cluster_demand), from squeue and `registry`: 0 once written."""
    return cluster_demand.state(
        registry=registry, login=Remote(PROJECT.cluster.host, SubprocessRunner()),
        board=PROJECT.cluster.board(), project=PROJECT.name, campaign=CAMPAIGN,
        weight=CAMPAIGN_WEIGHT, nodes=tuple(n.name for n in PROJECT.cluster.nodes),
        now=utcnow())


def main(argv: list[str]) -> int:
    """cluster_submit's verbs, each followed, for those of STATES_DEMAND, by writing reblock's
    demand, whose failure is said and leaves the verb's exit as it was (a submit that took, a
    wait's verdict); and `demand`, that write alone, its exit its own."""
    if argv == ["demand"]:
        return state_demand(Registry(REPO))
    code = cs.main(PROJECT, argv)
    args = parser(PROJECT).parse_args(argv)
    if args.verb in STATES_DEMAND and not (args.verb in ("submit", "resubmit") and args.dry):
        try:
            if state_demand(Registry(REPO)):
                print("[demand] the next sync states it again", file=sys.stderr)
        except (SystemExit, ValueError, OSError) as e:      # ssh, squeue's output, the registry
            print(f"[demand] not stated: {e}; the next sync states it again", file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.path[:0] = [str(HERE), str(REPO)]   # common and the repo's scripts package, for Blocks.plan
    raise SystemExit(main(sys.argv[1:]))
