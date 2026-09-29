"""Two-exit access (gst2) over a block sample: trees to trees_2r_k{k}_lam{lam}/, rows to rows_2r_.../.

    GST_BLOCKS=small|large pixi run python sweep2.py <workers> <k> <lam,...> [first_blocks_file]
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gst2  # noqa: E402
import study  # noqa: E402

SECTOR_M = float(os.environ.get("GST_SECTOR_M", "150"))   # scale at which the two exits must differ, m
WHICH = os.environ.get("GST_BLOCKS", "small")
# MODE: "group" (either route may end anywhere in the group), "corner" (both routes to one corner,
# mandatory), or "prize<p>" (one route always, the second only if it costs <= p more)
MODE = os.environ.get("GST_MODE", "group")
PREFIX = {"group": "2r"}.get(MODE, "2rc" if MODE == "corner" else f"2rp{MODE[5:]}")
if SECTOR_M != 150.0:
    PREFIX += f"_s{SECTOR_M:g}"
_BLOCKS: list = []


def one(task) -> None:
    i, k, lam = task
    block = _BLOCKS[i]
    tag = f"{PREFIX}_k{k}_lam{lam:g}"
    out = HERE / f"rows_{tag}" / f"{block.block_id}.parquet"
    if out.exists():
        return
    row: dict = dict(block=block.block_id, n_parcels=len(block.parcels), k=k, lam=lam)
    t0 = time.time()
    try:
        inst = gst2.build(block, k=k, sector_m=SECTOR_M)
        homes = gst2.edge_homes(inst, block)
        cost = inst.length + lam * homes
        if MODE == "group":
            built, st = gst2.solve(inst, cost)
        elif MODE == "corner":
            built, st = gst2.solve2(inst, cost, mode="corner", prize=None)
        else:
            built, st = gst2.solve2(inst, cost, prize=float(MODE[5:]))
        row.update(st, groups=len(inst.groups), sectors=inst.n_sectors,
                   tree_len=float(inst.length[built].sum()))
        roads = gst2.roads(inst, built, block)
        tmp = HERE / f"trees_{tag}" / f"{block.block_id}.{os.getpid()}.tmp"
        roads.to_parquet(tmp)
        os.replace(tmp, HERE / f"trees_{tag}" / f"{block.block_id}.parquet")
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    row["total_s"] = time.time() - t0
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    pd.DataFrame([row]).to_parquet(tmp)
    os.replace(tmp, out)
    study.log(f"{tag} {block.block_id} {row['n_parcels']}: {row.get('tree_len', 0):.0f} m, "
              f"infeasible {row.get('infeasible_groups')} {row['total_s']:.0f}s {row.get('error', '')}")


def main() -> None:
    global _BLOCKS
    workers, k = int(sys.argv[1]), int(sys.argv[2])
    lams = [float(x) for x in sys.argv[3].split(",")]
    first = set(Path(sys.argv[4]).read_text().split()) if len(sys.argv) > 4 else set()
    for lam in lams:
        (HERE / f"rows_{PREFIX}_k{k}_lam{lam:g}").mkdir(exist_ok=True)
        (HERE / f"trees_{PREFIX}_k{k}_lam{lam:g}").mkdir(exist_ok=True)
    _BLOCKS = study.blocks(WHICH)
    order = sorted(range(len(_BLOCKS)), key=lambda i: (_BLOCKS[i].block_id not in first,
                                                       -len(_BLOCKS[i].parcels)))
    tasks = [(i, k, lam) for i in order for lam in lams]
    study.log(f"sweep2 {WHICH} {MODE} k={k} lams={lams}: {len(tasks)} tasks")
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=1) as pool:
        list(pool.imap_unordered(one, tasks))
    study.log("sweep2: done")


if __name__ == "__main__":
    main()
