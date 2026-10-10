"""Rasterize block 5810 once (h 0.5, lifted.OFFSET) and save what the prototype needs.

Run from the reblock repo root:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. uv run python <this>
Writes fabric_5810.npz beside this script.
"""
import sys
import time
from pathlib import Path

import numpy as np
import shapely

sys.path.insert(0, "research/roadless")
import common  # noqa: E402
import lifted  # noqa: E402

OUT = Path(__file__).resolve().parent / "fabric_5810.npz"
t0 = time.time()
[b] = common.build_blocks(["ZAF.9.3.1_1_5810"])
t1 = time.time()
p = lifted.Params(3.0, 8, solver=lifted.solver_of("cpu"))
g = lifted.UniformMesh(0.5, offset=lifted.OFFSET).build(b, p, common.POPULATIONS["area"])
t2 = time.time()
polys = np.asarray(b.buildings.outlines)
w = common.POPULATIONS["area"].weights(polys)
reach = lifted.grounded(g, g.ff0, p)
f, stranded, owner = lifted.demand(g, polys, reach, w)
t3 = time.time()
np.savez_compressed(
    OUT, ff0=g.ff0.astype(np.float32), inside=g.inside, ground=g.ground, building=g.building,
    reach=reach, f=f, owner=owner.astype(np.int32), w=w, stranded=stranded,
    x0=g.x0, y0=g.y0, h=g.h, area=shapely.area(polys), n_buildings=len(polys),
    block_area=b.boundary.area)
print(f"source {t1 - t0:.1f}s grid {t2 - t1:.1f}s demand {t3 - t2:.1f}s shape {g.ff0.shape} "
      f"inside {int(g.inside.sum())} ground {int(g.ground.sum())} buildings {len(polys)} "
      f"stranded {int(stranded.sum())} block_area {b.boundary.area:.0f}")
