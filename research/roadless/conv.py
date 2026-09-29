"""h and K convergence of the roadless score on real blocks: P0, and perm' of each lineup method's
Lens A (10% displacement) prefix."""
import sys, time; sys.path.insert(0, 'research/roadless')
import numpy as np, common, lifted
from reblock.derivations import propose
from reblock.budget import prefix_to_displacement
blocks = common.build_blocks(common.recipients())
arms = common.arms()
idx = [int(a) for a in sys.argv[1].split(",")]
hs = [float(a) for a in sys.argv[2].split(",")]
K = int(sys.argv[3]); ell = float(sys.argv[4])
off = tuple(float(a) for a in sys.argv[5].split(",")) if len(sys.argv) > 5 else (0.3713, 0.1931)
names = ["clearance_looped", "cycle_native", "resistance_lp", "greedy_arterial_access_displacement"]
for i in idx:
    b = blocks[i]
    pre = {}
    for n in names:
        r = propose(arms[n], b).roads
        pre[n] = prefix_to_displacement(b, r, 0.10)
    for h in hs:
        t = time.time()
        sc = common.Scorer(b, h, lifted.Params(ell_m=ell, K=K), offset=off)
        vals = {n: sc.perm(pre[n]) for n in names}
        print(f"{b.block_id} n={len(b.buildings)} K={K} ell={ell} h={h} off={off}: P0 {sc.P0:9.1f} stranded {sc.stranded:.3f} "
              + " ".join(f"{n[:12]} {v:.3f}" for n, v in vals.items()) + f"  {time.time()-t:.0f}s", flush=True)
