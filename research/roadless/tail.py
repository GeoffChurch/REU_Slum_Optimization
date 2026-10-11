"""Per-home share of P0, tau_i = w_i u_i (u_i the injection-weighted mean potential over the
home's cells): how concentrated is P0?"""
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
    tau = sc.w * sc.u0                                      # per home, sums to P0
    tau = tau[tau > 0]                                      # (nan: enclosed)
    s = np.sort(tau)[::-1]
    print(f"{b.block_id} n={len(sc.polys)} P0 {sc.P0:.0f} sum tau {tau.sum():.0f}: mean {tau.mean():.1f} "
          f"median {np.median(tau):.1f} max {s[0]:.1f} max/mean {s[0] / tau.mean():.1f}; "
          f"top 1/5/10% homes hold {s[:max(1, len(s)//100)].sum() / s.sum():.2f} / "
          f"{s[:max(1, len(s)//20)].sum() / s.sum():.2f} / {s[:max(1, len(s)//10)].sum() / s.sum():.2f} of P0; "
          f"top 10 tau {np.round(s[:10], 0).tolist()}", flush=True)
