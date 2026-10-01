"""Is it the metric or the search? Under each along-conductance, the roadless P score (p 1) at 10%
displaced (area population) of the road lineup's corridors (carved) and of the greedy's
clearing (its perm1, linear to D = 0.10). If roads overtake the greedy under a road-favouring
conductance, lanes are worth it and the greedy is not finding them.

    PYTHONPATH=. pixi run python research/roadless/along_lineup.py <ids,> <workers> [picker] [alongs,]
"""
from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from clear import rows_dir  # noqa: E402
from study import SHORT  # noqa: E402

_B: dict = {}
_ARMS: dict = {}
_CFG: dict = {}


def one(job: tuple[str, str]) -> list[dict]:
    from reblock.derivations import propose
    bid, along = job
    b = _B[bid]
    pop = common.POPULATIONS["area"]
    sc = common.Scorer(b, 0.5, lifted.Params(3.0, 8, lifted.along_of(along)), population=pop)
    w = pop.weights(sc.polys)
    rows = []
    for arm, m in _ARMS.items():
        pre = common.prefix_to(b, propose(m, b).roads, 0.10, w)
        rows.append(dict(block=bid, along=along, arm=SHORT[arm],
                         D=common.displacement(b, pre, w), P=sc.perm(pre)))
    g = pd.read_parquet(rows_dir(_CFG["picker"], 0.5, "area", 2.0, along)
                        / f"{bid}.parquet").sort_values("D")
    rows.append(dict(block=bid, along=along, arm="clear_bldg", D=0.10,
                     P=float(np.interp(0.10, g.D, g.perm1))))
    print(f"{bid} {along} done", flush=True)
    return rows


def main(ids: list[str], workers: int, picker: str, alongs: list[str]) -> None:
    global _B, _ARMS, _CFG
    _CFG = dict(picker=picker)
    _B = {b.block_id: b for b in common.build_blocks(ids)}
    _ARMS = common.arms()
    jobs = [(i, a) for i in ids for a in alongs]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        rows = [r for rs in pool.imap_unordered(one, jobs) for r in rs]
    d = pd.DataFrame(rows)
    d.to_parquet(HERE / f"along_lineup_{picker}.parquet")
    piv = d.pivot_table(index=["block", "along"], columns="arm", values="P")
    pd.set_option("display.width", 200)
    print("\nroadless P score (p 1) at 10% displaced, per block and along-conductance")
    print(piv.round(3).to_string())
    roads = [c for c in piv.columns if c != "clear_bldg"]
    gap = (piv.clear_bldg - piv[roads].max(axis=1)).unstack("along")
    print("\ngreedy minus the best road arm")
    print(gap.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1].split(","), int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "S0.01cat",
         sys.argv[4].split(",") if len(sys.argv) > 4 else ["uni", "sl3", "sl10"])
