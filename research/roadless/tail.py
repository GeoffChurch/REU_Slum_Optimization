"""Per-home escape time tau_i (demand-weighted mean potential over the home's ring cells):
how concentrated is P0?"""
import sys

sys.path.insert(0, 'research/roadless')
import numpy as np

import common
import lifted

blocks = common.build_blocks(common.recipients())
for i in [int(a) for a in sys.argv[1].split(",")]:
    b = blocks[i]
    p = lifted.Params(ell_m=3.0, K=8)
    sc = common.Scorer(b, lifted.UniformMesh(0.5, offset=lifted.OFFSET), p)
    sol = lifted.solve(sc.grid, sc.free0, sc.f, p)
    free = sc.free0 > 0
    nc = int(free.sum())
    uc = np.zeros(nc)
    live = sol.unk_cell >= 0
    _, _, m, _ = lifted.axes(p.K)
    uc[live] = sol.u.reshape(-1, p.K) @ (m / np.pi)
    ucell = np.zeros(free.shape)
    ucell[free] = uc
    contrib = sc.f * ucell                                  # per cell, sums to P0
    # attribute to buildings by nearest footprint of each demand cell
    rr, cc = np.nonzero(sc.f > 0)
    import shapely
    pts = shapely.points(sc.grid.xy[rr, cc, 0], sc.grid.xy[rr, cc, 1])
    idx, _ = sc.tree.query_nearest(pts, return_distance=True, all_matches=False)
    owner = np.full(len(rr), -1)
    owner[idx[0]] = idx[1]
    tau = np.bincount(owner, weights=contrib[rr, cc], minlength=len(sc.polys))
    tau = tau[tau > 0]
    s = np.sort(tau)[::-1]
    print(f"{b.block_id} n={len(sc.polys)} P0 {sc.P0:.0f} sum tau {tau.sum():.0f}: mean {tau.mean():.1f} "
          f"median {np.median(tau):.1f} max {s[0]:.1f} max/mean {s[0] / tau.mean():.1f}; "
          f"top 1/5/10% homes hold {s[:max(1, len(s)//100)].sum() / s.sum():.2f} / "
          f"{s[:max(1, len(s)//20)].sum() / s.sum():.2f} / {s[:max(1, len(s)//10)].sum() / s.sum():.2f} of P0; "
          f"top 10 tau {np.round(s[:10], 0).tolist()}", flush=True)
