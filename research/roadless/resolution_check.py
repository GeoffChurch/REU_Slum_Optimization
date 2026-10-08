"""Where resolution matters: the same clearings scored exactly at several grid spacings.

SIMP's coarse-to-fine `.c1` lost 30848's gate (NOTES, "SIMP: every 2nd candidate ..."): a
passage narrower than a cell stops conducting. This scores fixed clearings -- several per block,
from different methods -- at h 0.5 (the metric's), 0.75 and 1.0. Where a coarser grid keeps
each clearing's score and their order, a block can be optimized coarse; where it does not, the
block has gates below that spacing. Buildings, their costs and a clearing do not depend on h.

    PYTHONPATH=. uv run python research/roadless/resolution_check.py <id> <p> <cpu|gpu> <along> \\
        <clearings.parquet> <mesh,mesh,...> <budget>

mesh: <h> (a uniform grid) or <h>a<d0>x<smax> (lifted.AdaptiveMesh: h within d0 metres of
every footprint, street and the block edge, coarser with distance up to smax).

clearings.parquet: block, name, cleared (building indices); in the cluster's inputs
($CLUSTER_SUBMIT_INPUTS) or a local path. A name ending `_steps` is a greedy's pick order, cut to
`budget` as polish_greedy.py cuts it (every building in order while it fits) and reported
without the suffix. Rows in res_rows/<along>/<id>_p<p>_h<mesh,mesh,...>.parquet:
block, name, h (the finest cell), mesh (its name), perm (Lens A on that mesh), cells.
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
import lifted  # noqa: E402
import relax  # noqa: E402


def main(bid: str, power: float, device: str, along: str, clearings: str, tokens: list[str],
         budget: float) -> None:
    src = Path(os.environ["CLUSTER_SUBMIT_INPUTS"]) / clearings \
        if "CLUSTER_SUBMIT_INPUTS" in os.environ else Path(clearings)
    todo = pd.read_parquet(src)
    todo = todo[todo.block == bid]
    if todo.empty:
        raise SystemExit(f"{bid}: no clearings in {src}")
    rows = []
    for token in tokens:
        mesh = lifted.mesh_of(token)
        t0 = time.time()
        c = relax._clearing(bid, device, along, mesh)
        rel = relax.Relaxation(c, power)
        cells = int(c.sc.grid.inside.sum())
        for name, cleared in zip(todo.name, todo.cleared):
            if name.endswith("_steps"):
                name = name[:-len("_steps")]
                cleared = np.flatnonzero(relax.cut_to_budget(list(cleared), c.cost, budget))
            r = np.zeros(c.n)
            r[np.asarray(cleared, dtype=np.int64)] = 1.0
            rows.append(dict(block=bid, name=name, h=mesh.h, mesh=mesh.name,
                             perm=rel.perm(rel.exact(r)), cells=cells))
        print(f"{bid} {mesh.name}: {cells} cells, "
              + ", ".join(f"{r['name']} {r['perm']:.4f}" for r in rows if r["mesh"] == mesh.name)
              + f" ({time.time() - t0:.0f}s)", flush=True)
        del rel, c
    out = HERE / "res_rows" / along / f"{bid}_p{power:g}_h{','.join(tokens)}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[1]), a[2], a[3], a[4], a[5].split(","), float(a[6]))
