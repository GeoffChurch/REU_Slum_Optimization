"""Check FineBase.delta against a fresh solve on block 5810 (the earlier study's fabric)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import region_common as rc
import region_fine as rf
S = str(rc.OUT.parent / "fft_homog") + "/"
d = np.load(S + "fabric_5810.npz"); C = np.load(S + "clearings_5810.npz")
order = np.argsort(C["sub_bldg"], kind="stable"); sb = C["sub_bldg"][order]; n = len(d["w"])
cen = np.load(S + "centroids.npy")
fab = rc.Fabric(o=d["ff0"].astype(float), inside=d["inside"], ground=d["ground"], f=d["f"],
                owner=d["owner"], w=d["w"], live=~d["stranded"], area=d["area"], cy=cen[0],
                cx=cen[1], cnt=cen[2], py=cen[0], px=cen[1], sc=C["sub_cell"][order], sb=sb,
                sn=C["sub_count"][order], starts=np.searchsorted(sb, np.arange(n + 1)))
t = time.time(); fb = rf.FineBase(fab); print("base", fb.N0, fb.iters, f"{time.time()-t:.1f}s J0 {fb.J0:.10g}", flush=True)
rng = np.random.default_rng(0)
for trial in range(4):
    r0, c0 = [(1200, 1000), (800, 1400), (1500, 600), (400, 1800)][trial]
    m = (fab.cnt > 0) & (fab.cy >= r0) & (fab.cy < r0 + 100) & (fab.cx >= c0) & (fab.cx < c0 + 100)
    ids = np.flatnonzero(m)
    ids = rng.permutation(ids)[: max(1, len(ids) // 3)]
    oc = rc.open_buildings(fab, ids)
    for rtol in (1e-7, 1e-9):
        dl = fb.delta(oc, rtol=rtol)
        print(f"tile {r0},{c0} {len(ids)} bldg rtol {rtol:g}: dJ {dl['dJ']:.10g} it {dl['iters']} new {dl['n_new']} loc {dl['n_loc']} {dl['seconds']:.2f}s", flush=True)
    fu = fb.full(oc)
    print(f"   full: dJ {fu['J'] - fb.J0:.10g} ({fu['iters']} it {fu['seconds']:.1f}s); rel diff {(dl['dJ'] - (fu['J'] - fb.J0)) / (fu['J'] - fb.J0):.2e}", flush=True)
