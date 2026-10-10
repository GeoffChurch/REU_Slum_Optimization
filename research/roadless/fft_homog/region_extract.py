"""Rasterize 5810@major once on the metric's uniform lattice (h 0.5, lifted.OFFSET) and save
what the region_*.py scripts need: the fabric (as extract.py), the per-cell footprint labels as
(cell, building, count) triples (as extract2.py, to open any building exactly), the building
centroids (pick.py's ring-cell means, and the polygons' own), and a check that the two stored
clearings' indices are in b.buildings.outlines order (their cleared population share D).

Run from the reblock repo root:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. uv run python research/roadless/fft_homog/region_extract.py
Writes OUT/fabric_region.npz.
"""
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import common  # noqa: E402
import lifted  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import region_common as rc  # noqa: E402
import shapely  # noqa: E402

with rc.Monitor("extract: build the region Block"):
    [b] = common.build_blocks([rc.BID])
polys = np.asarray(b.buildings.outlines)
print(f"{rc.BID}: {len(polys)} buildings, area {b.boundary.area / 1e6:.3f} km^2", flush=True)

with rc.Monitor("extract: uniform h 0.5 grid, demand, labels"):
    t = time.time()
    p = lifted.Params(3.0, 8, solver=lifted.solver_of("cpu"))
    g = lifted.UniformMesh(rc.H, offset=lifted.OFFSET).build(b, p, common.POPULATIONS["area"])
    print(f"grid {g.inside.shape} ({g.inside.size / 1e6:.1f}M px), inside "
          f"{int(g.inside.sum()) / 1e6:.2f}M, ground {int(g.ground.sum())}, "
          f"{time.time() - t:.0f} s", flush=True)
    w = common.POPULATIONS["area"].weights(polys)
    reach = lifted.grounded(g, g.ff0, p)
    f, stranded, owner = lifted.demand(g, polys, reach, w)
    print(f"demand: stranded {int(stranded.sum())}, {time.time() - t:.0f} s", flush=True)
    lab = g.label_sub(polys).reshape(-1, g.S * g.S)
    cells, subs = np.nonzero(lab >= 0)
    key = cells.astype(np.int64) * len(polys) + lab[cells, subs]
    del lab, subs
    ukey, cnts = np.unique(key, return_counts=True)
    del key, cells
    sub_cell, sub_bldg = ukey // len(polys), (ukey % len(polys)).astype(np.int32)
    print(f"labels: {len(ukey)} (cell, building) pairs, {time.time() - t:.0f} s", flush=True)

ny, nx = g.inside.shape
# pick.py's centroids: the mean of each building's ring cells (rows, columns), with their count
rr, cc = np.nonzero(owner >= 0)
k = owner[rr, cc]
cnt = np.bincount(k, minlength=len(polys))
cy = np.bincount(k, weights=rr, minlength=len(polys)) / np.maximum(cnt, 1)
cx = np.bincount(k, weights=cc, minlength=len(polys)) / np.maximum(cnt, 1)
cen = shapely.centroid(polys)
px = (shapely.get_x(cen) - g.x0) / g.h            # cell (i, j) is centred at (x0 + j h, y0 + i h)
py = (shapely.get_y(cen) - g.y0) / g.h
area = shapely.area(polys)

# the stored clearings: indices into b.buildings.outlines? (their D from the population share)
st = rc.stored_clearings()
cheap_row = pd.read_parquet(rc.CHEAP).iloc[0]
simp_row = pd.read_parquet(rc.SIMP).iloc[0]
for name, row, Dcol in (("cheap", cheap_row, "D"), ("simp", simp_row, "simp_D")):
    ids = st[name]
    D = area[ids].sum() / area.sum()
    print(f"{name}: {len(ids)} buildings, stored n {row.n} (here {len(polys)}), stored D "
          f"{row[Dcol]:.6f}, D from these indices {D:.6f}, max index {ids.max()}", flush=True)
    assert row.n == len(polys) and abs(D - row[Dcol]) < 1e-9

np.savez_compressed(
    rc.OUT / "fabric_region.npz", ff0=g.ff0.astype(np.float32), inside=g.inside,
    ground=g.ground, reach=reach, f=f, owner=owner.astype(np.int32), w=w, stranded=stranded,
    x0=g.x0, y0=g.y0, h=g.h, area=area, cy=cy, cx=cx, cnt=cnt, py=py, px=px,
    sub_cell=sub_cell, sub_bldg=sub_bldg, sub_count=cnts.astype(np.int16),
    n_buildings=len(polys), region_area=b.boundary.area, boundary_wkt=b.boundary.wkt)
print("saved", rc.OUT / "fabric_region.npz", flush=True)
