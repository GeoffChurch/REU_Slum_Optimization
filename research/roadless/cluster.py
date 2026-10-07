"""reblock's cluster launcher: the project and its one kind of run, on GeoffChurch/cluster_submit.

`blocks` runs one task per block: a script command with `{id}` in it, with the block bank
shipped as an input (a node has no source data; common.build_blocks reads the bank). Each task
needs GPU memory for the eps world, every inside cell x 8 headings, about GB_PER_MILLION GB per
million unknowns, and cluster_submit routes it to the smallest card that holds it. `--gb` sets a
floor under that estimate, to send a block known to need a bigger card to one. Results (parquet
rows) come back to the same paths here, never overwriting a local file.

    uv run python research/roadless/cluster.py setup
    uv run python research/roadless/cluster.py submit blocks <run> [--time T] [--gb G] [--gpus N] [--h H] [--with FILE ...] <ids,|@file> -- <script> <args with {id}>
    uv run python research/roadless/cluster.py status|sync [<run>]
    uv run python research/roadless/cluster.py wait <run> [--timeout 3h]    # exit 0 passed, 3 failed, 4 timed out, 5 blind
    uv run python research/roadless/cluster.py resubmit <run> [--gb G] [--time T]   # its failed / missing blocks as <run>-r1
    uv run python research/roadless/cluster.py cancel <run>
    uv run python research/roadless/cluster.py watch start|stop
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cluster_submit as cs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GB_PER_MILLION = 0.7    # measured: the translucent greedy on 30796 peaks at 31.2 GiB for 48.7M
                        # unknowns after the K-cycle fix (NOTES); raise it if a block runs out
# The checkout's .venv from uv.lock, with every default group: cupy and pyamg (`gpu`) among them.
ENV = cs.Uv()


@dataclass(frozen=True)
class Blocks:
    name: str = "blocks"
    pull: cs.IntoTree = cs.IntoTree("research/roadless", ("*.parquet",))

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("ids", help="block ids, comma-separated, or @file of ids")
        parser.add_argument("--time", help="sbatch --time (default: the cluster's)")
        parser.add_argument("--gb", type=float, default=0.0,
                            help="a floor under each task's GPU-memory estimate, in GB")
        parser.add_argument("--h", type=float, default=0.5,
                            help="the grid spacing the tasks use (relax.py's .h<h>; scales the "
                                 "memory estimate, which assumes h 0.5)")
        parser.add_argument("--with", dest="extra", type=Path, action="append", default=[],
                            help="a file shipped beside the block bank (repeatable); a task "
                                 "reads it as $CLUSTER_SUBMIT_INPUTS/<its name>")
        parser.add_argument("command", nargs=argparse.REMAINDER,
                            help="-- <script> <args with {id}>")

    def plan(self, args: argparse.Namespace, workdir: Path) -> cs.RunPlan:
        import common   # geopandas and the block sources: only a submit needs them

        # REMAINDER (which drops the `--`) swallows an option placed after the ids
        command: list[str] = args.command
        if not command or command[0].startswith("-"):
            raise SystemExit("usage: submit blocks <run> [--time T] [--gb G] <ids> -- <command>")
        line = " ".join(command)
        if "{id}" not in line:
            raise SystemExit("the command needs {id}")
        if args.h != 0.5 and f".h{args.h:g}" not in line:
            raise SystemExit(f"--h {args.h:g} but the command's plan has no .h{args.h:g}")
        ids = (Path(args.ids[1:]).read_text().split() if args.ids.startswith("@")
               else args.ids.split(","))
        bank = common.write_bank(ids, workdir / "bank.pkl")
        tasks = tuple(
            cs.Task(f"python -u {line.replace('{id}', b)}",
                    max(bank[b].boundary.area / args.h ** 2 * 8 / 1e6 * GB_PER_MILLION, args.gb))
            for b in ids)
        return cs.RunPlan(
            tasks=tasks, resources=cs.Resources(cpus=4, mem_gb=64, time=args.time),
            inputs=(workdir / "bank.pkl", *args.extra),
            prelude=("source research/roadless/cluster_env.sh",
                     'export REBLOCK_BLOCK_BANK="$CLUSTER_SUBMIT_INPUTS/bank.pkl"'),
            describe=(f"{len(ids)} blocks: {line}",))


PROJECT = cs.Project(name="reblock", repo=REPO, cluster=cs.clusters.AI, env=ENV,
                     kinds=(Blocks(),), max_gpus=4)     # owner, 2026-10-03: a shared cluster

if __name__ == "__main__":
    sys.path[:0] = [str(HERE), str(REPO)]   # common and the repo's scripts package, for Blocks.plan
    raise SystemExit(cs.main(PROJECT))
