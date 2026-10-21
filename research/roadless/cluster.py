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
    uv run python research/roadless/cluster.py status|sync [<run>]
    uv run python research/roadless/cluster.py wait <run> [--timeout 3h]
        # exit 0 passed, 3 failed, 4 timed out, 5 blind
    uv run python research/roadless/cluster.py resubmit <run> [--gb G] [--time T]
        # its failed / missing blocks as <run>-r1
    uv run python research/roadless/cluster.py cancel <run>
    uv run python research/roadless/cluster.py watch start|stop

Before a CPU run, the node's load: `ssh ai sinfo -N -p defq -o '%N %c %T %C'` (cores allocated,
idle, other, total) and `scontrol show node <node>` (AllocMem of RealMemory).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import cluster_submit as cs
from cluster_submit.remote import SubprocessRunner
from cluster_submit.tree import blob_at

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
# Every task's environment: single-threaded, no bytecode, the shipped bank (common.build_blocks).
PRELUDE = ("source research/roadless/cluster_env.sh",
           'export REBLOCK_BLOCK_BANK="$CLUSTER_SUBMIT_INPUTS/bank.pkl"')


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


# sacct_grace: how long a run is out of squeue before a task of it with no exit code and no sacct
# row is taken as cancelled before it started (cluster_submit's "Settling a run": 10 minutes, the
# cluster's accounting lag many times over; cx-raster, 2026-10-15, had 19 such tasks).
# attribute_cache: how long ai's login node may answer a stat from cached attributes (its NFSv3
# home's acregmax and acdirmax, the client's defaults: 60 s), so a run is finished, with its last
# pull, only once it has been out of squeue that long (cluster_submit's "Settling a run": mycooc's
# ev2-lib-20k-c, finished a second after it left squeue, lost its last row so, 2026-10-21).
# market: the disk market of ai's home (its keeper holds reblock's files once a launch publishes
# them). checkout_setup: nothing to link into a checkout. gpu_market None: reblock's waves are held
# outside the GPU market's shadow accounts (never `keeper install` from here: it would remove the
# GPU market's files the keeper reads; the keeper is installed from GeoffChurch/mycooc). campaigns:
# none of reblock's runs is a campaign's lease.
PROJECT = cs.Project(name="reblock", repo=REPO, cluster=cs.load_cluster("ai"),
                     kinds=(Blocks(), Cpu()), max_gpus=4,   # owner, 2026-10-03: a shared cluster
                     checkout_setup=(), sacct_grace=timedelta(minutes=10),
                     attribute_cache=timedelta(seconds=60),
                     market=cs.load_market("ai"), gpu_market=None, campaigns=cs.NoCampaigns())

if __name__ == "__main__":
    sys.path[:0] = [str(HERE), str(REPO)]   # common and the repo's scripts package, for Blocks.plan
    raise SystemExit(cs.main(PROJECT))
