import sys, time; sys.path.insert(0, 'research/roadless')
import numpy as np, common, lifted
t=time.time(); blocks = common.build_blocks(common.recipients()); print("blocks", len(blocks), f"{time.time()-t:.0f}s", flush=True)
sizes=[len(b.buildings) for b in blocks]; print("buildings quantiles", np.percentile(sizes,[0,25,50,75,90,100]).round(), flush=True)
areas=[b.boundary.area for b in blocks]; print("area m2 quantiles", np.percentile(areas,[0,25,50,75,90,100]).round(), flush=True)
for i in (20, 110, 200):
    b = blocks[i]
    for h in (1.0, 0.5):
        mesh = lifted.UniformMesh(h, offset=lifted.OFFSET)
        t=time.time(); sc = common.Scorer(b, mesh, lifted.Params(ell_m=3.0, K=16))
        print(b.block_id, len(b.buildings), f"area {b.boundary.area:.0f} h={h} free cells {sc.free0.sum()} P0 {sc.P0:.1f} fallback {sc.n_fallback} stranded {sc.stranded:.3f} {time.time()-t:.1f}s", flush=True)
