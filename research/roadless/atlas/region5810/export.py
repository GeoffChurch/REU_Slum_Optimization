"""5810@major's flow fields on its composite mesh (0.5a5x8), painted to 1 m rasters.
`base`: the flow before any clearing and the no-buildings prior; `arm <name> <cleared.npy>`:
the flow after a clearing. Per cell, flux S = sum_k |phi_k| over the cell's side in fine cells
(a flux per metre, comparable across cell sizes); escape time u."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path("/home/gchurchill/src/reblock/research/roadless")
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from current_map import flow  # noqa: E402

BID = "ZAF.9.3.1_1_5810@major"
OUT = Path(sys.argv[1])
mode = sys.argv[2]
t0 = time.time()
[b] = common.build_blocks([BID])
p = lifted.Params(3.0, 8, solver=lifted.solver_of("gpu"))
mesh = common.mesh_of("0.5a5x8")
sc = common.Scorer(b, mesh, p, population=common.POPULATIONS["area"])
g = sc.grid
side = (2.0 ** g.level)                                  # in fine cells
print(f"{len(g.level)} cells, scorer {time.time() - t0:.0f} s", flush=True)


def raster(v):
    """(n,) -> 1 m raster (ny/2, nx/2), north row first; NaN outside the region."""
    R = g.paint(np.asarray(v, dtype=float), np)
    ins = g.paint(np.ones(len(g.level)), np) > 0
    R[~ins] = np.nan
    ny, nx = R.shape
    R = R[: ny // 2 * 2, : nx // 2 * 2].reshape(ny // 2, 2, nx // 2, 2)
    return np.nanmean(R, axis=(1, 3))[::-1].astype("float32")


def save(name, f):
    np.save(OUT / f"S_{name}.npy", raster(np.abs(f.phi).sum(axis=0) / side))
    np.save(OUT / f"u_{name}.npy", raster(np.where(f.u > 0, f.u, np.nan)))
    print(name, "P", f.P, f"{time.time() - t0:.0f} s", flush=True)


if mode == "base":
    save("before", flow(sc, g.ff0, p))
    save("open", flow(sc, g.isub.mean(axis=-1), p))
    open0 = raster((g.ff0 > 0).astype(float))
    np.save(OUT / "open0.npy", open0)
    xs = g.x0 - g.h / 2
    ys = g.y0 - g.h / 2
    ny, nx = g.shape
    (OUT / "frame.json").write_text(json.dumps({"x0": xs, "y0": ys, "w": nx // 2 * 2 * g.h,
                                                "h": ny // 2 * 2 * g.h, "px": 1.0,
                                                "cells": int(len(g.level))}))
else:
    name, cleared = sys.argv[3], np.load(sys.argv[4])
    lab = g.label_sub(sc.polys)
    after = (g.isub & (~g.bsub | np.isin(lab, cleared))).mean(axis=-1)
    save(name, flow(sc, after, p))
