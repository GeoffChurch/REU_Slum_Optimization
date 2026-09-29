"""Displacement-aware group Steiner: edge cost = length + lam * homes under the edge's corridor.

    pixi run python sweep.py <workers> <k> <lam,lam,...> [first_blocks_file]

Trees go to trees_k{k}_lam{lam}/, rows to rows_k{k}_lam{lam}/. The LP bound and MILP are on the
WEIGHTED objective; scoring (score_all, arm gst_k{k}_lam{lam}) uses the real metrics.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gst  # noqa: E402
import study  # noqa: E402

LP_LIMIT, EXACT_LIMIT, EXACT_MAX_GROUPS = 300.0, 180.0, 250
_BLOCKS: list = []
FAST = os.environ.get("GST_FAST") == "1"      # no LP: dual ascent + heuristics (fast_tree)
WHICH = os.environ.get("GST_BLOCKS", "small")


class _Done(Exception):
    pass


def one(task: tuple[int, int, float]) -> None:
    i, k, lam = task
    block = _BLOCKS[i]
    tag = f"{'fast_' if FAST else ''}k{k}_lam{lam:g}"
    rows, trees = HERE / f"rows_{tag}", HERE / f"trees_{tag}"
    out = rows / f"{block.block_id}.parquet"
    if out.exists():
        return
    row: dict = dict(block=block.block_id, n_parcels=len(block.parcels), k=k, lam=lam)
    t0 = time.time()
    try:
        base = gst.build_instance(block, k)
        row["n_unreachable"] = base.n_unreachable
        if base.groups:
            ed = gst.edge_displacement(base, block)
            inst = gst.weighted(base, ed, lam)
            if FAST:
                tree, ub, lb, st = gst.fast_tree(inst)
                row.update(lb=lb, ub=ub, **st, tree_len=gst.tree_cost(base, tree))
                raise _Done
            lb, xe, f, st = gst.lp_bound(inst, time_limit=LP_LIMIT)
            tree, ub, _ = gst.upper_bound(inst, x_edge=xe, restarts=20)
            row.update(lb=lb, ub=ub, **st)
            if ub - lb > 1e-6 * max(ub, 1.0) and len(inst.groups) <= EXACT_MAX_GROUPS:
                opt, bound, st2, mt = gst.exact_tree(inst, f, time_limit=EXACT_LIMIT)
                row.update(milp_bound=bound, opt=opt, **st2)
                if opt is not None and opt < ub - 1e-6 and mt is not None:
                    tree = mt
            row["tree_len"] = gst.tree_cost(base, tree)
        else:
            tree = set()
    except _Done:
        pass
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    if "error" not in row:
        roads = gst.roads_gdf(base, tree, block)
        tmp = trees / f"{block.block_id}.{os.getpid()}.tmp"
        roads.to_parquet(tmp)
        os.replace(tmp, trees / f"{block.block_id}.parquet")
    row["total_s"] = time.time() - t0
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    pd.DataFrame([row]).to_parquet(tmp)
    os.replace(tmp, out)
    study.log(f"{tag} {block.block_id} {row['n_parcels']}: len {row.get('tree_len', 0):.0f} "
              f"gap {(row.get('ub', 0) - row.get('lb', 0)) / max(row.get('lb', 1e-9), 1e-9):.2%} "
              f"{row['total_s']:.0f}s {row.get('error', '')}")


def main() -> None:
    global _BLOCKS
    workers, k = int(sys.argv[1]), int(sys.argv[2])
    lams = [float(x) for x in sys.argv[3].split(",")]
    first = set(Path(sys.argv[4]).read_text().split()) if len(sys.argv) > 4 else set()
    for lam in lams:
        tag = f"{'fast_' if FAST else ''}k{k}_lam{lam:g}"
        (HERE / f"rows_{tag}").mkdir(exist_ok=True)
        (HERE / f"trees_{tag}").mkdir(exist_ok=True)
    _BLOCKS = study.blocks(WHICH)
    order = sorted(range(len(_BLOCKS)), key=lambda i: (_BLOCKS[i].block_id not in first,
                                                       len(_BLOCKS[i].parcels)))
    tasks = [(i, k, lam) for i in order for lam in lams]
    study.log(f"sweep k={k} lams={lams}: {len(tasks)} tasks, {len(first)} blocks first")
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        list(pool.imap_unordered(one, tasks))
    study.log("sweep: done")


if __name__ == "__main__":
    main()
