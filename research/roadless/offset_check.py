"""Grid alignment alone: the same clearings scored exactly on one grid spacing under several
lattice offsets (the uniform mesh's cell centres shifted against the buildings).

resolution_check.py moves the spacing, which moves the alignment too; this holds h and moves
only the alignment (NOTES, "The metric's resolution floor"). OFFSETS[0] is lifted.OFFSET, the
metric's own; the others shift the lattice by half a cell on both axes, and by a quarter and
three quarters.

    PYTHONPATH=. uv run python research/roadless/offset_check.py <id> <p> <cpu|gpu> <along> \\
        <clearings.parquet> <h>

clearings.parquet as resolution_check.py's (block, name, cleared), here or in the cluster's inputs.
Rows in res_rows/<along>/<id>_p<p>_h<h>_offsets.parquet: block, name, h, off (the offset, as
fractions of h), perm.
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

OFFSETS = (lifted.OFFSET, (0.8713, 0.6931), (0.1213, 0.9431), (0.6213, 0.4431))


def main(bid: str, power: float, device: str, along: str, clearings: str, h: float) -> None:
    src = Path(os.environ["CLUSTER_SUBMIT_INPUTS"]) / clearings \
        if "CLUSTER_SUBMIT_INPUTS" in os.environ else Path(clearings)
    todo = pd.read_parquet(src)
    todo = todo[todo.block == bid]
    if todo.empty:
        raise SystemExit(f"{bid}: no clearings in {src}")
    rows = []
    for off in OFFSETS:
        t0 = time.time()
        c = relax._clearing(bid, device, along, lifted.UniformMesh(h, offset=off))
        rel = relax.Relaxation(c, power)
        for name, cleared in zip(todo.name, todo.cleared, strict=True):
            r = np.zeros(c.n)
            r[np.asarray(cleared, dtype=np.int64)] = 1.0
            rows.append(dict(block=bid, name=name, h=h, off=f"{off[0]:.4f},{off[1]:.4f}",
                             perm=rel.perm(rel.exact(r))))
        print(f"{bid} h{h:g} offset {off}: "
              + ", ".join(f"{r['name']} {r['perm']:.4f}" for r in rows[-len(todo):])
              + f" ({time.time() - t0:.0f}s)", flush=True)
        del rel, c
    out = HERE / "res_rows" / along / f"{bid}_p{power:g}_h{h:g}_offsets.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], float(a[1]), a[2], a[3], a[4], float(a[5]))
