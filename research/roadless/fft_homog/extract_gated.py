"""Rasters of two gated blocks (22422, 30848) and the greedy's (S0.01cat, lifted uni, J_2)
clearings through steps 1, 2 and 5. Run from the reblock repo root. Writes fabric_<id>.npz."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

sys.path.insert(0, "research/roadless")
import common  # noqa: E402
import lifted  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = Path("research/roadless/clear_rows_S0.01cat_h0.5_area_p2")
p = lifted.Params(3.0, 8, solver=lifted.solver_of("cpu"))
for bid in ("ZAF.9.3.1_1_22422", "ZAF.9.3.1_1_30848"):
    [b] = common.build_blocks([bid])
    g = lifted.UniformMesh(0.5, offset=lifted.OFFSET).build(b, p, common.POPULATIONS["area"])
    polys = np.asarray(b.buildings.outlines)
    w = common.POPULATIONS["area"].weights(polys)
    reach = lifted.grounded(g, g.ff0, p)
    f, stranded, owner = lifted.demand(g, polys, reach, w)
    lab = g.label_sub(polys)
    rows = pd.read_parquet(RUN / f"{bid}.parquet")
    out = dict(ff0=g.ff0.astype(np.float32), inside=g.inside, ground=g.ground, f=f,
               owner=owner.astype(np.int32), w=w, stranded=stranded, area=shapely.area(polys),
               block_area=b.boundary.area)
    for step in (1, 2, 5):
        s = rows[(rows.step >= 1) & (rows.step <= step)]
        ids = np.unique(np.concatenate([np.asarray(c, dtype=np.int64) for c in s.cleared]))
        removed = np.zeros(len(polys), dtype=bool)
        removed[ids] = True
        cl = (lab >= 0) & removed[np.maximum(lab, 0)]
        out[f"ff_step{step}"] = (g.isub & (~g.bsub | cl)).mean(axis=-1).astype(np.float32)
        out[f"perm_step{step}"] = float(rows[rows.step == step].perm.iloc[0])
        out[f"D_step{step}"] = float(rows[rows.step == step].D.iloc[0])
        out[f"n_step{step}"] = len(ids)
    np.savez_compressed(HERE / f"fabric_{bid.split('_')[-1]}.npz", **out)
    print(bid, g.ff0.shape, len(polys), "stranded", int(stranded.sum()), "area",
          round(b.boundary.area),
          {k: out[k] for k in out if k.startswith(("perm", "n_", "D_"))}, flush=True)
