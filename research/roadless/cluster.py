"""reblock's cluster launcher: the project and its two kinds of run, on GeoffChurch/cluster_submit.

`blocks` runs one task per block: a script command with `{id}` in it, with the block bank
shipped as an input (a node has no source data; common.build_blocks reads the bank). Each task
needs GPU memory for the eps world, every inside cell x 8 headings, about GB_PER_MILLION GB per
million unknowns, and cluster_submit routes it to the smallest card that holds it. `--mesh` is
the grid the tasks use (common.mesh_of: <h>, <h>a<d0>x<smax> or
<h>g<d0>x<smax>f<factor>p<power>o<opening>), its cells counted here, SIZING_WORKERS blocks at a
time (a goal mesh's: its target); `--scans` says the tasks' conductance scans the fine lattice
(a sightline metric or search: SoftSightline), whose rasters peak when the solves' own memory is
down to what a run holds, so the estimate is the larger of the two; `--gb` sets a floor under
the estimate, to send a block known to need a bigger card to one. `--complex`: the tasks run the
gap complex (gap_clear: cuDSS on the card, gap_factor.CuDSS), which has no raster grid: no
`--mesh`, the card sized by `--gb` alone. `--unpack` ships a tar per
block of inputs git does not carry (parquet rows, at their repo paths) and unpacks it in the
task's checkout before the command. Results (parquet rows) come back to the same paths here,
never overwriting a local file.

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
    uv run python research/roadless/cluster.py submit blocks <run> [--time T] [--gb G] [--gpus N] \
        [--mesh M | --complex] [--scans] [--with FILE ...] [--unpack TAR] <ids,|@file> \
        -- <script> <args with {id}>
    uv run python research/roadless/cluster.py submit cpu <run> --node orion|quadro2 --cpus C \
        --mem G --items <items,|@file> [--gpus N] [--time T] [--unpack TAR] <ids,|@file> \
        -- <script> <args with {id} and {item}>
    uv run python research/roadless/cluster.py status|sync [<run>]
    uv run python research/roadless/cluster.py wait <run> [--timeout 3h]    # exit 0 passed, 3 failed, 4 timed out, 5 blind
    uv run python research/roadless/cluster.py resubmit <run> [--gb G] [--time T]   # its failed / missing blocks as <run>-r1
    uv run python research/roadless/cluster.py cancel <run>
    uv run python research/roadless/cluster.py watch start|stop

Before a CPU run, the node's load: `ssh ai sinfo -N -p defq -o '%N %c %T %C'` (cores allocated,
idle, other, total) and `scontrol show node <node>` (AllocMem of RealMemory).
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import cluster_submit as cs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GB_PER_MILLION = 0.95   # measured (2026-10-12): road_rescore.py's scoring of the 13 tuning blocks
                        # peaks at 0.83 -- 0.87 GB per million unknowns on the nine over 14M
                        # (22422's 0.866 the most), at 1.1 -- 1.4 on the four under 13M (a few
                        # GB fixed, which any card holds), above 0.7 (the rate before) on all 13
                        # and on item 4's 60, where 1558, 14401 and 20023 went to 32 GB cards
                        # they would not fit and ran on bigger ones by luck. 0.95 is 22422's rate
                        # with 10% to spare: a peak the 1 s sampling misses, and the spread
                        # between runs (the 13's large blocks peaked up to 2% above item 4's).
                        # SIMP and the greedies peak at ~0.65 (NOTES), so they get room to spare.
                        # cluster_check.py `gpu` checks recorded runs' peaks against it; raise it
                        # if a block runs out
HELD_PER_MILLION = 0.6  # --scans: what a run holds between solves (472's translucent greedy on
                        # a5x8, 34M unknowns: 19.8 GiB) ...
RASTER_BYTES = 40       # ... plus five float64 rasters of the fine lattice (the painted field,
                        # F, B, the vjp's weight and its sum): 472, 260M pixels, peaks at 30.0 GiB
SIZING_WORKERS = 8      # processes counting the blocks' cells at submit: speed only (a shared
                        # machine; the count is shapely and integer numpy, no BLAS threads)
# The nodes a `cpu` run may go to, and the most of its tasks each runs at once (`--gpus`): the
# cluster's two nodes whose GPUs are unusable (sites/ai.toml) and whose CPUs are fine (owner
# 2026-10-12: "run the 13- and 60-block stages on orion/quadro2"). Slurm allocates whole cores
# (CR_CORE_MEMORY: a 1-cpu task holds both of a core's threads), so fewer run at once than these
# allow: 36 single-CPU tasks on orion, 32 on quadro2, less what others hold, and what --mem leaves.
CPU_NODES = {
    "orion": 64,        # 72 CPUs, 36 cores; idle on 2026-10-12
    "quadro2": 40,      # 64 CPUs, 32 cores; shared (another user's jobs: 8 CPUs, 320 of 376 GB)
}
# The checkout's .venv from uv.lock, with every default group: cupy and pyamg (`gpu`) among them.
ENV = cs.Uv()
# Every task's environment: single-threaded, no bytecode, the shipped bank (common.build_blocks).
PRELUDE = ("source research/roadless/cluster_env.sh",
           'export REBLOCK_BLOCK_BANK="$CLUSTER_SUBMIT_INPUTS/bank.pkl"')


def _listed(arg: str) -> list[str]:
    """A comma-separated list, or @file of whitespace-separated entries."""
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
    pull: cs.IntoTree = cs.IntoTree("research/roadless", ("*.parquet",))
    routine_exits: frozenset[int] = frozenset()     # every non-zero exit is a failure

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("ids", help="block ids, comma-separated, or @file of ids")
        parser.add_argument("--time", help="sbatch --time (default: the cluster's)")
        parser.add_argument("--gb", type=float, default=0.0,
                            help="a floor under each task's GPU-memory estimate, in GB")
        parser.add_argument("--mesh", default="0.5",
                            help="the grid the tasks use (<h> or <h>a<d0>x<smax>: relax.py's "
                                 ".h<h> / .a<d0>x<smax>, or one of resolution_check.py's); "
                                 "sets the memory estimate")
        parser.add_argument("--complex", action="store_true",
                            help="the tasks run the gap complex (no raster grid): the card "
                                 "sized by --gb alone")
        parser.add_argument("--scans", action="store_true",
                            help="the tasks' conductance scans the fine lattice (ss/sl metric "
                                 "or search): size for its rasters too")
        parser.add_argument("--with", dest="extra", type=Path, action="append", default=[],
                            help="a file shipped beside the block bank (repeatable); a task "
                                 "reads it as $CLUSTER_SUBMIT_INPUTS/<its name>")
        _unpack_argument(parser)
        parser.add_argument("command", nargs=argparse.REMAINDER,
                            help="-- <script> <args with {id}>")

    def plan(self, args: argparse.Namespace, workdir: Path) -> cs.RunPlan:
        import common   # geopandas and the block sources: only a submit needs them

        line = _command(args, "{id}")
        ids = _listed(args.ids)
        if args.complex:
            if args.scans or args.gb <= 0:
                raise SystemExit("--complex: no --scans, and --gb sizes the card")
        else:
            mesh = common.mesh_of(args.mesh)
            if not _names(line, args.mesh, mesh):
                raise SystemExit(f"--mesh {args.mesh} but the command does not use it")
        unpack = _unpacks(args.unpack, ids)

        def task_line(b: str) -> str:
            run = f"python -u {line.replace('{id}', b)}"
            if args.unpack is None:
                return run
            return f'tar -xf "$CLUSTER_SUBMIT_INPUTS/{unpack[b].name}" && {run}'

        bank = common.write_bank(ids, workdir / "bank.pkl")
        if args.complex:
            gb = [0.0] * len(ids)
        else:
            with ProcessPoolExecutor(min(SIZING_WORKERS, len(ids))) as ex:
                gb = list(ex.map(_gb, [mesh] * len(ids), [bank[b] for b in ids],
                                 [args.scans] * len(ids)))
        tasks = tuple(cs.Task(task_line(b), max(g, args.gb))
                      for b, g in zip(ids, gb, strict=True))
        return cs.RunPlan(
            tasks=tasks, resources=cs.Resources(cpus=4, mem_gb=64, time=args.time, node=None),
            inputs=(workdir / "bank.pkl", *args.extra, *unpack.values()),
            prelude=PRELUDE, describe=(f"{len(ids)} blocks: {line}",))

    def more(self, ctx: cs.Context, rec: cs.RunRecord) -> bool:
        return False    # never asked: no exit of its is routine (cluster_submit's "Leases")


@dataclass(frozen=True)
class Cpu:
    """One task per (block, item), CPU only, on one node of CPU_NODES (the module's docstring)."""

    name: str = "cpu"
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

    def plan(self, args: argparse.Namespace, workdir: Path) -> cs.RunPlan:
        import common   # geopandas and the block sources: only a submit needs them

        line = _command(args, "{id}", "{item}")
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

    def more(self, ctx: cs.Context, rec: cs.RunRecord) -> bool:
        return False    # never asked: no exit of its is routine (cluster_submit's "Leases")


def _gb(mesh, block, scans: bool) -> float:
    """A task's GPU memory on `block` (a pool worker's task): the eps world's solves, every cell
    x 8 headings, or (`scans`) what a run holds plus the scans' rasters, whichever is more."""
    unknowns = mesh.cells(block) * 8 / 1e6
    solves = unknowns * GB_PER_MILLION
    if not scans:
        return solves
    return max(solves, unknowns * HELD_PER_MILLION + mesh.pixels(block) * RASTER_BYTES / 2 ** 30)


def _names(line: str, token: str, mesh) -> bool:
    """Whether the command uses `mesh`: the token among its words or comma lists
    (resolution_check.py), or its suffix in a SIMP plan (relax.py; h 0.5 unnamed)."""
    return token in line.replace(",", " ").split() or mesh.suffix in line


# sacct_grace: how long a run is out of squeue before a task of it with no exit code and no sacct
# row is taken as cancelled before it started (cluster_submit's "Settling a run": 10 minutes, the
# cluster's accounting lag many times over; cx-raster, 2026-10-15, had 19 such tasks).
PROJECT = cs.Project(name="reblock", repo=REPO, cluster=cs.load_cluster("ai"), env=ENV,
                     kinds=(Blocks(), Cpu()), max_gpus=4,   # owner, 2026-10-03: a shared cluster
                     sacct_grace=timedelta(minutes=10))

if __name__ == "__main__":
    sys.path[:0] = [str(HERE), str(REPO)]   # common and the repo's scripts package, for Blocks.plan
    raise SystemExit(cs.main(PROJECT))
