"""reblock's cluster launcher: the project and its one kind of run, on GeoffChurch/cluster_submit.

`blocks` runs one task per block: a script command with `{id}` in it, with the block bank
shipped as an input (a node has no source data; common.build_blocks reads the bank). Each task
needs GPU memory for the eps world, every inside cell x 8 headings, about GB_PER_MILLION GB per
million unknowns, and cluster_submit routes it to the smallest card that holds it. `--gb` sets a
floor under that estimate, to send a block known to need a bigger card to one. Results (parquet
rows) come back to the same paths here, never overwriting a local file.

    pixi run -e launch python research/roadless/cluster.py setup
    pixi run -e launch python research/roadless/cluster.py submit blocks <run> <ids,|@file> [--time T] [--gb G] -- <script> <args with {id}>
    pixi run -e launch python research/roadless/cluster.py status|sync [<run>]
    pixi run -e launch python research/roadless/cluster.py cancel <run>
    pixi run -e launch python research/roadless/cluster.py watch start|stop
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cluster_submit as cs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
GB_PER_MILLION = 1.0    # an estimate (the eps world); raise it if a block runs out of memory
PIXI = cs.Pixi()
DEPS = "~/.cache/reblock-research/pydeps"   # the research deps outside the lock
# cupy with the CUDA 12 toolkit from pip (the cluster's /usr/local/cuda is 11.8, the nodes'
# drivers 12.8); pyamg without its deps (numpy and scipy are the env's). The numpy that the
# [ctk] extra drags in would shadow the env's own on PYTHONPATH, so it goes.
SETUP = (
    f"test -d {DEPS}/cupy || {PIXI.run_prefix()} python -m pip install -q --target {DEPS} "
    "'cupy-cuda12x[ctk]==14.2.0'",
    f"test -d {DEPS}/pyamg || {PIXI.run_prefix()} python -m pip install -q --target {DEPS} "
    "--no-deps pyamg==5.3.0",
    f"rm -rf {DEPS}/numpy {DEPS}/numpy-*.dist-info {DEPS}/numpy.libs",
)


@dataclass(frozen=True)
class Blocks:
    name: str = "blocks"
    pull: cs.IntoTree = cs.IntoTree("research/roadless", ("*.parquet",))

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("ids", help="block ids, comma-separated, or @file of ids")
        parser.add_argument("--time", help="sbatch --time (default: the cluster's)")
        parser.add_argument("--gb", type=float, default=0.0,
                            help="a floor under each task's GPU-memory estimate, in GB")
        parser.add_argument("command", nargs=argparse.REMAINDER,
                            help="-- <script> <args with {id}>")

    def plan(self, args: argparse.Namespace, workdir: Path) -> cs.RunPlan:
        import common   # geopandas and the block sources: only a submit needs them

        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        line = " ".join(command)
        if "{id}" not in line:
            raise SystemExit("the command needs {id}")
        ids = (Path(args.ids[1:]).read_text().split() if args.ids.startswith("@")
               else args.ids.split(","))
        bank = common.write_bank(ids, workdir / "bank.pkl")
        tasks = tuple(
            cs.Task(f"python -u {line.replace('{id}', b)}",
                    max(bank[b].boundary.area / 0.25 * 8 / 1e6 * GB_PER_MILLION, args.gb))
            for b in ids)
        return cs.RunPlan(
            tasks=tasks, resources=cs.Resources(cpus=4, mem_gb=64, time=args.time),
            inputs=(workdir / "bank.pkl",),
            prelude=("source research/roadless/cluster_env.sh",
                     'export REBLOCK_BLOCK_BANK="$CLUSTER_SUBMIT_INPUTS/bank.pkl"'),
            describe=(f"{len(ids)} blocks: {line}",))


PROJECT = cs.Project(name="reblock", repo=REPO, cluster=cs.clusters.AI, env=PIXI,
                     kinds=(Blocks(),), max_gpus=4,     # owner, 2026-10-03: a shared cluster
                     setup_commands=SETUP)

if __name__ == "__main__":
    sys.path.insert(0, str(HERE))       # common, for Blocks.plan
    raise SystemExit(cs.main(PROJECT))
