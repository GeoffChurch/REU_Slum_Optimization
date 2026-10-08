"""reblock's cluster launcher: the project and its one kind of run, on GeoffChurch/cluster_submit.

`blocks` runs one task per block: a script command with `{id}` in it, with the block bank
shipped as an input (a node has no source data; common.build_blocks reads the bank). Each task
needs GPU memory for the eps world, every inside cell x 8 headings, about GB_PER_MILLION GB per
million unknowns, and cluster_submit routes it to the smallest card that holds it. `--mesh` is
the grid the tasks use (lifted.mesh_of: <h> or <h>a<d0>x<smax>, the latter's cells counted
here, SIZING_WORKERS blocks at a time); `--gb` sets a floor under the estimate, to send a block
known to need a bigger card to one. Results (parquet rows) come back to the same paths here,
never overwriting a local file.

    uv run python research/roadless/cluster.py setup
    uv run python research/roadless/cluster.py submit blocks <run> [--time T] [--gb G] [--gpus N] \
        [--mesh M] [--with FILE ...] <ids,|@file> -- <script> <args with {id}>
    uv run python research/roadless/cluster.py status|sync [<run>]
    uv run python research/roadless/cluster.py wait <run> [--timeout 3h]    # exit 0 passed, 3 failed, 4 timed out, 5 blind
    uv run python research/roadless/cluster.py resubmit <run> [--gb G] [--time T]   # its failed / missing blocks as <run>-r1
    uv run python research/roadless/cluster.py cancel <run>
    uv run python research/roadless/cluster.py watch start|stop
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cluster_submit as cs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GB_PER_MILLION = 0.7    # measured: the translucent greedy on 30796 peaks at 31.2 GiB for 48.7M
                        # unknowns after the K-cycle fix (NOTES); raise it if a block runs out
SIZING_WORKERS = 8      # processes counting the blocks' cells at submit: speed only (a shared
                        # machine; the count is shapely and integer numpy, no BLAS threads)
# The checkout's .venv from uv.lock, with every default group: cupy and pyamg (`gpu`) among them.
ENV = cs.Uv()


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
        parser.add_argument("--with", dest="extra", type=Path, action="append", default=[],
                            help="a file shipped beside the block bank (repeatable); a task "
                                 "reads it as $CLUSTER_SUBMIT_INPUTS/<its name>")
        parser.add_argument("command", nargs=argparse.REMAINDER,
                            help="-- <script> <args with {id}>")

    def plan(self, args: argparse.Namespace, workdir: Path) -> cs.RunPlan:
        import common   # geopandas and the block sources: only a submit needs them
        import lifted

        # REMAINDER (which drops the `--`) swallows an option placed after the ids
        command: list[str] = args.command
        if not command or command[0].startswith("-"):
            raise SystemExit("usage: submit blocks <run> [--time T] [--gb G] <ids> -- <command>")
        line = " ".join(command)
        if "{id}" not in line:
            raise SystemExit("the command needs {id}")
        mesh = lifted.mesh_of(args.mesh)
        if not _names(line, args.mesh, mesh):
            raise SystemExit(f"--mesh {args.mesh} but the command does not use it")
        ids = (Path(args.ids[1:]).read_text().split() if args.ids.startswith("@")
               else args.ids.split(","))
        bank = common.write_bank(ids, workdir / "bank.pkl")
        with ProcessPoolExecutor(min(SIZING_WORKERS, len(ids))) as ex:
            cells = list(ex.map(_cells, [mesh] * len(ids), [bank[b] for b in ids]))
        tasks = tuple(
            cs.Task(f"python -u {line.replace('{id}', b)}",
                    max(n * 8 / 1e6 * GB_PER_MILLION, args.gb))
            for b, n in zip(ids, cells, strict=True))
        return cs.RunPlan(
            tasks=tasks, resources=cs.Resources(cpus=4, mem_gb=64, time=args.time, node=None),
            inputs=(workdir / "bank.pkl", *args.extra),
            prelude=("source research/roadless/cluster_env.sh",
                     'export REBLOCK_BLOCK_BANK="$CLUSTER_SUBMIT_INPUTS/bank.pkl"'),
            describe=(f"{len(ids)} blocks: {line}",))


def _cells(mesh, block) -> float:
    """`mesh`'s cells on `block` (a pool worker's task)."""
    return mesh.cells(block.boundary, list(block.buildings.outlines),
                      list(block.streets.geometry))


def _names(line: str, token: str, mesh) -> bool:
    """Whether the command uses `mesh`: the token among its words or comma lists
    (resolution_check.py), or its suffix in a SIMP plan (relax.py; h 0.5 unnamed)."""
    return token in line.replace(",", " ").split() or mesh.suffix in line


PROJECT = cs.Project(name="reblock", repo=REPO, cluster=cs.clusters.AI, env=ENV,
                     kinds=(Blocks(),), max_gpus=4)     # owner, 2026-10-03: a shared cluster

if __name__ == "__main__":
    sys.path[:0] = [str(HERE), str(REPO)]   # common and the repo's scripts package, for Blocks.plan
    raise SystemExit(cs.main(PROJECT))
