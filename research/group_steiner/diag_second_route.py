"""What limits interior loops in the redundancy-prize solver?

Replays solve2's prize augmentation (no pruning) and, for every group that needs a decision, records
its distance from the street (metres along the boundary graph) and the extra cost of a second route
(price of the best disjoint pair minus the price of the single route) under two rules:
  sector   the second route must leave through a different street sector (the shipped rule)
  relaxed  the second route may end at ANY node of the network built so far (option (a))

    PYTHONPATH=<repo> pixi run python diag_second_route.py <lam> <prize> <block> [<block> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gst2  # noqa: E402
import study220  # noqa: E402

lam, prize = float(sys.argv[1]), float(sys.argv[2])
rows = []
for bid in sys.argv[3:]:
    b = study220.build_blocks([bid])[0]
    inst = gst2.build(b, k=1, sector_m=150)
    homes = gst2.edge_homes(inst, b)
    cost = inst.length + lam * homes
    built = np.zeros(len(inst.eu), bool)
    allu = np.ones(len(inst.eu), bool)

    def gdist(w):
        g = sp.csr_matrix((np.r_[w, w] + gst2.EPS, (np.r_[inst.eu, inst.ev], np.r_[inst.ev, inst.eu])),
                          shape=(inst.n, inst.n))
        return dijkstra(g, indices=np.arange(1, inst.n_sectors + 1), min_only=True)

    dist, dist_m = gdist(cost), gdist(inst.length)
    order = sorted(range(len(inst.groups)), key=lambda i: -float(np.min(dist[inst.groups[i]])))

    def price(used):
        return float(sum(cost[e] for e in used if not built[e]))

    for gi in order:
        grp = inst.groups[gi]
        corners = grp[np.argsort(dist[grp])][:4]
        if built.any() and any(gst2.routes(inst, np.array([v]), built) >= 2 for v in corners):
            continue
        free = np.where(built, 0.0, cost)
        one = gst2._single(inst, grp, free)
        bn = np.unique(np.r_[inst.eu[built], inst.ev[built]])
        bn = bn[bn > inst.n_sectors]
        best = {}
        for name, extra in (("sector", None), ("relaxed", bn)):
            cand = [(gst2.suurballe(inst, np.array([v]), free, allu, extra), v) for v in corners]
            cand = [(price(u), u) for u, v in cand if u is not None]
            best[name] = min(cand, key=lambda t: t[0]) if cand else (np.inf, None)
        deg = np.zeros(inst.n, np.int64)
        np.add.at(deg, inst.eu[built], 1)
        np.add.at(deg, inst.ev[built], 1)
        through = bool((deg[grp] >= 2).any())      # a corner of the group already on a through-lane
        p1 = price(one) if one is not None else np.inf
        rows.append(dict(block=bid, depth_m=float(np.min(dist_m[grp])), one=p1,
                         extra_sector=best["sector"][0] - p1, extra_relaxed=best["relaxed"][0] - p1, through=through))
        # follow the SHIPPED decision (sector rule) so the network is the real one
        if best["sector"][1] is not None and best["sector"][0] - p1 <= prize:
            built[list(best["sector"][1])] = True
        elif one is not None:
            built[list(one)] = True
    print(bid, len(b.parcels), "groups decided", sum(r["block"] == bid for r in rows), flush=True)

import pandas as pd  # noqa: E402
d = pd.DataFrame(rows)
d["bin"] = pd.cut(d.depth_m, [-1, 25, 50, 100, 200, 1e9], labels=["<25 m", "25-50", "50-100", "100-200", ">200"])
g = d.groupby("bin", observed=True)
out = pd.DataFrame({
    "groups": g.size(),
    "route 1 cost (median)": g.one.median().round(0),
    "sector: median": g.extra_sector.median().round(0),
    "sector: share<=prize": g.extra_sector.apply(lambda x: (x <= prize).mean()).round(2),
    "relaxed: median": g.extra_relaxed.median().round(0),
    "relaxed: share<=prize": g.extra_relaxed.apply(lambda x: (x <= prize).mean()).round(2),
    "share with a corner on a through-lane": g.through.mean().round(2),
})
print(f"\nlam {lam:g}, prize {prize:g}; second-route extra cost in metres + lam x homes\n{out.to_string()}")
