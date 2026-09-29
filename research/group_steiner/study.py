"""Group Steiner solver vs topology on topology's own objective.

    pixi run python study.py small <workers>      # the 220 consensus recipients
    pixi run python study.py large <workers>      # the 36 large blocks (topology cannot run)
    pixi run python study.py report

Per block: heuristic (SPH + prune + MST) alone, LP lower bound (directed cuts), LP-guided restarts,
exact MILP where the gap is open and the block small enough; and the solver's roads scored through
the canonical street-first prefix (`prefix_to_depth` 1), as topo_obj scores every other method.
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
GAPS = HERE
TOPO = HERE
sys.path.insert(0, str(GAPS))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(TOPO))
import gst  # noqa: E402

LP_LIMIT = 900.0
EXACT_LIMIT = 600.0
EXACT_MAX_GROUPS = 400
_BLOCKS: list = []


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (HERE / "progress.log").open("a") as f:
        f.write(line + "\n")


def one(task: tuple[int, int]) -> None:
    from topo_obj import score
    i, k = task
    block = _BLOCKS[i]
    rows_dir, trees_dir = HERE / f"rows_k{k}", HERE / f"trees_k{k}"
    out = rows_dir / f"{block.block_id}.parquet"
    if out.exists():
        return
    row: dict = dict(block=block.block_id, n_bldg=len(block.buildings),
                     n_parcels=len(block.parcels), k=k)
    t0 = time.time()
    try:
        inst = gst.build_instance(block, k)
        row.update(nodes=inst.n, edges=len(inst.ec), groups=len(inst.groups),
                   build_s=time.time() - t0)
        if not inst.groups:                  # already at the target: the empty tree is optimal
            row.update(ub=0.0, lb=0.0, sph=0.0, opt=0.0)
            empty = gst.roads_gdf(inst, set(), block)
            tmp_t = trees_dir / f"{block.block_id}.{os.getpid()}.tmp"
            empty.to_parquet(tmp_t)
            os.replace(tmp_t, trees_dir / f"{block.block_id}.parquet")
        else:
            t = time.time()
            tree0, ub0, st = gst.upper_bound(inst, restarts=0)
            row.update(sph=st["sph_first"], ub_improve=ub0, heur_s=time.time() - t)
            lb, xe, f, st = gst.lp_bound(inst, time_limit=LP_LIMIT)
            row.update(lb=lb, **st)
            t = time.time()
            tree, ub, _ = gst.upper_bound(inst, x_edge=xe, restarts=20)
            row.update(ub=ub, restart_s=time.time() - t)
            opt = None
            if ub - lb > 1e-6 * max(ub, 1.0) and len(inst.groups) <= EXACT_MAX_GROUPS:
                opt, bound, st, milp_tree = gst.exact_tree(inst, f, time_limit=EXACT_LIMIT)
                row.update(milp_bound=bound, **st)
                if opt is not None and opt < ub - 1e-6:
                    row["ub_above_opt"] = ub - opt
                    assert milp_tree is not None
                    assert abs(gst.tree_cost(inst, milp_tree) - opt) < 1e-6 * max(opt, 1.0)
                    tree = milp_tree
            elif ub - lb <= 1e-6 * max(ub, 1.0):
                opt = ub
            row["opt"] = opt
            roads = gst.roads_gdf(inst, tree, block)
            tmp_t = trees_dir / f"{block.block_id}.{os.getpid()}.tmp"
            roads.to_parquet(tmp_t)
            os.replace(tmp_t, trees_dir / f"{block.block_id}.parquet")
            sc = score(block, roads)
            row.update({f"canon_{k}": v for k, v in sc.items()})
            row["tree_m"] = float(roads.geometry.length.sum())
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    row["total_s"] = time.time() - t0
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    pd.DataFrame([row]).to_parquet(tmp)
    os.replace(tmp, out)
    gap = (row.get("ub", np.nan) - row.get("lb", np.nan)) / max(row.get("lb", np.nan), 1e-9)
    log(f"k={k} {block.block_id} {row['n_parcels']} parcels: ub {row.get('ub', np.nan):.1f} "
        f"lb {row.get('lb', np.nan):.1f} gap {gap:.2%} opt {row.get('opt')} "
        f"{row['total_s']:.0f}s {row.get('error', '')}")


def blocks(which: str) -> list:
    if which == "small":
        import study220
        return sorted(study220.build_blocks(study220.recipients()), key=lambda b: len(b.parcels))
    import large_study as LS
    LS.build()
    return sorted((b for _, b in LS._BLOCKS.values()), key=lambda b: len(b.parcels))


def run(which: str, workers: int, ks: tuple[int, ...]) -> None:
    global _BLOCKS
    for k in ks:
        (HERE / f"rows_k{k}").mkdir(exist_ok=True)
        (HERE / f"trees_k{k}").mkdir(exist_ok=True)
    _BLOCKS = blocks(which)
    log(f"{which}: {len(_BLOCKS)} blocks, k in {ks}")
    tasks = [(i, k) for i in range(len(_BLOCKS)) for k in ks]   # smallest blocks first
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        list(pool.imap_unordered(one, tasks))
    log(f"{which}: done")


if __name__ == "__main__":
    if sys.argv[1] in ("small", "large"):
        run(sys.argv[1], int(sys.argv[2]), tuple(int(k) for k in sys.argv[3].split(",")))
