"""The lineup on the 220 recipients, scored roadless and by today's permeability, on the SAME
prefixes: the canonical street-first prefix at each displacement budget in BUDGETS (and the whole
network). Lens A is budget 0.10; Lens B is read afterwards from the curve (both metrics are
monotone along the prefix order).

    PYTHONPATH=. pixi run python research/roadless/study.py run <workers> <h> <ell> [K]
    PYTHONPATH=. pixi run python research/roadless/study.py report <h> <ell> [K]
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
import common  # noqa: E402
import lifted  # noqa: E402

BUDGETS = (0.02, 0.05, 0.10, 0.15, 0.20)
_BLOCKS: list = []
_ARMS: dict = {}
_CFG: dict = {}


def rows_dir(h: float, ell: float, K: int) -> Path:
    return HERE / f"rows_h{h:g}_ell{ell:g}_K{K}"


def one(i: int) -> None:
    try:
        _one(i)
    except Exception as e:                       # log and move on: one bad block must not kill the pool
        print(f"{time.strftime('%H:%M:%S')} {_BLOCKS[i].block_id} FAILED "
              f"{type(e).__name__}: {e}"[:300], flush=True)


def _one(i: int) -> None:
    from reblock.budget import displacement, prefix_to_displacement
    from reblock.compare import load_permeability_config
    from reblock.derivations import propose
    from reblock.permeability import EgressContext, permeability
    h, ell, K = _CFG["h"], _CFG["ell"], _CFG["K"]
    out = rows_dir(h, ell, K) / f"{_BLOCKS[i].block_id}.parquet"
    if out.exists():
        return
    b = _BLOCKS[i]
    t0 = time.time()
    pcfg = load_permeability_config(common.REPO / "conf")
    ctx = EgressContext.of(b, pcfg.params)
    p = lifted.Params(ell_m=ell, K=K)
    carve = common.Scorer(b, h, p, rule=common.Carve())
    obl = common.Scorer(b, h, p, rule=common.Obliterate())
    n = len(b.buildings)
    rows = []
    for name in common.LINEUP:
        roads = propose(_ARMS[name], b).roads
        if roads is None or len(roads) == 0:
            continue
        seen: dict[int, dict] = {}
        for bud in (*BUDGETS, None):
            pre = roads if bud is None else prefix_to_displacement(b, roads, bud)
            key = len(pre)
            if key in seen:                       # budget unreachable: same (whole) prefix
                rows.append({**seen[key], "budget": bud if bud is not None else np.nan,
                             "full": bud is None})
                continue
            r = dict(block=b.block_id, n=n, arm=name, n_roads=len(pre),
                     road_m=float(pre.geometry.length.sum()),
                     D=displacement(b.buildings, pre) / n,
                     P_old=permeability(ctx, pre), P_carve=carve.perm(pre),
                     stranded=carve.stranded)
            if bud == 0.10:
                r["P_obl"] = obl.perm(pre)
            seen[key] = r
            rows.append({**r, "budget": bud if bud is not None else np.nan, "full": bud is None})
    df = pd.DataFrame(rows)
    df["P0_carve"] = carve.P0
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, out)
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} n={n} {time.time() - t0:.0f}s", flush=True)


def run(workers: int, h: float, ell: float, K: int) -> None:
    global _BLOCKS, _ARMS, _CFG
    _CFG = dict(h=h, ell=ell, K=K)
    rows_dir(h, ell, K).mkdir(exist_ok=True)
    _BLOCKS = common.build_blocks(common.recipients())
    _ARMS = common.arms()
    order = list(range(len(_BLOCKS)))[::-1]          # largest first: the long poles start early
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for _ in pool.imap_unordered(one, order):
            pass


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        K = int(sys.argv[5]) if len(sys.argv) > 5 else 8
        run(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), K)
