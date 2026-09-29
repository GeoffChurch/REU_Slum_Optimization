import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import study220, gst2, numpy as np
from reblock.budget import max_access_depth, displacement
from reblock.derive.access import ParcelAdjacency, STREET_TOL
from collections import Counter
for bid in sys.argv[1:]:
    b = study220.build_blocks([bid])[0]
    inst = gst2.build(b, k=1, sector_m=150); homes = gst2.edge_homes(inst, b); cost = inst.length + 30 * homes
    adj = ParcelAdjacency.of(b, STREET_TOL)
    for label, kw in [("group-2r", None), ("corner-2r", dict(mode="corner", prize=None)), ("prize 25", dict(prize=25.0)), ("prize 100", dict(prize=100.0)), ("prize 0", dict(prize=0.0))]:
        t = time.time()
        built, st = gst2.solve(inst, cost) if kw is None else gst2.solve2(inst, cost, **kw)
        r = gst2.roads(inst, built, b)
        deg = Counter()
        for i in np.flatnonzero(built):
            deg[int(inst.eu[i])] += 1; deg[int(inst.ev[i])] += 1
        dead = sum(1 for v, d in deg.items() if d == 1 and v > inst.n_sectors)
        print(f"{bid} {label:9s} road {r.geometry.length.sum():5.0f} m homes {displacement(b.buildings, r):5.1f} depth {max_access_depth(adj, r)} dead ends {dead:3d} pairs {st.get('pairs','-')} {time.time()-t:.1f}s", flush=True)
