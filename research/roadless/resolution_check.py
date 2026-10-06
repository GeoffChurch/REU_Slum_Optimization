"""Where resolution matters: the same clearings scored exactly at several grid spacings.

SIMP's coarse-to-fine `.c1` lost 30848's gate (NOTES, "SIMP: every 2nd candidate ..."): a
passage narrower than a cell stops conducting. This scores fixed clearings -- several per block,
from different methods -- at h 0.5 (the metric's), 0.75 and 1.0. Where a coarser grid keeps
each clearing's score and their order, a block can be optimized coarse; where it does not, the
block has gates below that spacing. Buildings, their costs and a clearing do not depend on h.

    PYTHONPATH=. uv run python research/roadless/resolution_check.py <id> <p> <cpu|gpu> <along> \\
        <clearings.parquet> <h,h,...> <budget>

clearings.parquet: block, name, cleared (building indices); in the cluster's inputs
($CLUSTER_SUBMIT_INPUTS) or a local path. A name ending `_steps` is a greedy's pick order, cut to
`budget` as polish_greedy.py cuts it (every building in order while it fits) and reported
without the suffix. Rows in res_rows/<along>/<id>_p<p>.parquet: block,
name, h, perm (Lens A at that h), cells (inside cells at that h).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import relax  # noqa: E402


def _cut(order: list[int], cost: np.ndarray, budget: float) -> list[int]:
    out, left = [], budget + 1e-12
    for j in order:
        if cost[j] <= left:
            out.append(j)
            left -= cost[j]
    return out


def main(bid: str, power: float, device: str, along: str, clearings: str, hs: list[float],
         budget: float) -> None:
    src = Path(os.environ["CLUSTER_SUBMIT_INPUTS"]) / clearings \
        if "CLUSTER_SUBMIT_INPUTS" in os.environ else Path(clearings)
    todo = pd.read_parquet(src)
    todo = todo[todo.block == bid]
    if todo.empty:
        raise SystemExit(f"{bid}: no clearings in {src}")
    rows = []
    for h in hs:
        t0 = time.time()
        c = relax._clearing(bid, device, along, h)
        rel = relax.Relaxation(c, power)
        cells = int(c.sc.grid.inside.sum())
        for name, cleared in zip(todo.name, todo.cleared):
            if name.endswith("_steps"):
                name, cleared = name[:-len("_steps")], _cut(list(cleared), c.cost, budget)
            r = np.zeros(c.n)
            r[np.asarray(cleared, dtype=np.int64)] = 1.0
            rows.append(dict(block=bid, name=name, h=h, perm=rel.perm(rel.exact(r)), cells=cells))
        print(f"{bid} h {h:g}: {cells} cells, "
              + ", ".join(f"{r['name']} {r['perm']:.4f}" for r in rows if r["h"] == h)
              + f" ({time.time() - t0:.0f}s)", flush=True)
        del rel, c
    out = HERE / "res_rows" / along / f"{bid}_p{power:g}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[1]), a[2], a[3], a[4], [float(x) for x in a[5].split(",")], float(a[6]))
