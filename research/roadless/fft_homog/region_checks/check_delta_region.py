"""Region-scale check: FineBase.delta (as the resolve runs used it, rtol 1e-8) against a fresh
fine solve, on two sampled tile clearings."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import region_common as rc
import region_tiles as rt
import region_fine as rf
rt.setup(("50",), fine=False)
fab = rt.G["fab"]
fb = rf.FineBase(fab, u0=np.load(rc.OUT / "fine_u0.npy"))
res = pd.concat([pd.read_csv(p) for p in sorted(rc.OUT.glob("tiles_resolve_[ab].csv"))])
for _, r in res.sort_values("dJ_fine").iloc[[0, len(res) // 2]].iterrows():
    ids = rt.clearing(int(r.r0), int(r.c0), r.rule)
    oc = rc.open_buildings(fab, ids)
    t = time.time(); d = fb.delta(oc, rtol=1e-8); td = time.time() - t
    t = time.time(); f = fb.full(oc); tf = time.time() - t
    dJf = f["J"] - fb.J0
    print(f"{int(r.r0)},{int(r.c0)} {r.rule}: resolve run {r.dJ_fine:.8g}, delta now {d['dJ']:.8g} "
          f"({d['iters']} it {td:.0f}s), fresh {dJf:.8g} ({f['iters']} it {tf:.0f}s), rel "
          f"{(d['dJ'] - dJf) / dJf:.1e}", flush=True)
