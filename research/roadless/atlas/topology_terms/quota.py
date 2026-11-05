"""The quota floor of topology's first objective: the least new parcel-boundary road giving at least
q parcels a corner on a road, swept over q (gst.py's quota section; floor.py is its q = every
parcel). Per q: the quota LP's cut loop (a lower bound at every round), the quota heuristic from
plain, LP-guided and perturbed costs and from the full floor tree (runs/steiner.parquet's edges)
trimmed, then the MILP with the LP's cuts (its dual bound and any incumbent).

What it bounds: networks on gst's graph (topology's own parcel-boundary graph, streets contracted)
whose served parcels are those with a corner on the network, every home kept (c = 0). That is
topology's family and its line P@1. It is NOT a strict bound on Shock Reblock or the Loop Network:
their roads leave that graph (cutting through parcels serves a parcel without a corner) and remove
homes.

q counts PARCELS (= homes on 41785, one building each), the street-fronting ones included: they
are served by the street at zero road, so the instance's groups are the others and the quota on
groups is q - n_free.

    Q=research/roadless/atlas/topology_terms/quota.py   # from the repo root, PYTHONPATH=.
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
        PYTHONDONTWRITEBYTECODE=1 uv run python $Q grid <block>                 # the q list
    ... $Q one <block> <q> <lp seconds> <milp seconds>   # runs/quota/q<q>.json, .parquet
    ... $Q polish <block>             # every q's ub from the larger q's trees, trimmed
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))           # research/roadless: common, road_rescore
import common  # noqa: E402
import gst  # noqa: E402
import road_rescore as rr  # noqa: E402

from reblock.derive.access import STREET_TOL  # noqa: E402

OUT = HERE / "runs" / "quota"
# P@1 levels (share of parcels): denser at the top, where the floor is steep.
LEVELS = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97,
          0.98, 0.99, 0.995, 1.0)
RESTARTS = 20
SEED = 0
PURGE_AFTER = 3
MAX_ROUNDS = 400


def free_parcels(b) -> int:
    """Parcels the street alone serves (within STREET_TOL of it)."""
    street = shapely.union_all(np.array(b.streets.geometry, dtype=object))
    return int(shapely.dwithin(np.array(b.parcels.geometry, dtype=object), street,
                               STREET_TOL).sum())


def instance(b) -> tuple[gst.Instance, int]:
    inst = gst.build_instance(b, 0)
    n_free = free_parcels(b)
    if len(inst.groups) + n_free != len(b.parcels) or inst.n_unreachable:
        raise AssertionError(f"groups {len(inst.groups)} + free {n_free} != parcels "
                             f"{len(b.parcels)} (unreachable {inst.n_unreachable})")
    return inst, n_free


def grid(n_parcels: int) -> list[int]:
    return sorted({math.ceil(p * n_parcels - 1e-9) for p in LEVELS})


def edges_of(inst: gst.Instance, lines) -> set[tuple[int, int]]:
    """A tree's lines (gst.roads_gdf's) back to instance edges, by endpoints."""
    key = {}
    for u, v in zip(inst.eu.tolist(), inst.ev.tolist(), strict=True):
        a = inst.xy[u] if u != gst.ROOT else inst.root_end[(u, v)]
        c = inst.xy[v] if v != gst.ROOT else inst.root_end[(u, v)]
        key[tuple(np.round(np.r_[a, c], 6))] = (u, v)
        key[tuple(np.round(np.r_[c, a], 6))] = (u, v)
    out = set()
    for g in lines:
        xy = shapely.get_coordinates(g)
        out.add(key[tuple(np.round(np.r_[xy[0], xy[-1]], 6))])
    return out


def one(bid: str, q: int, *, lp_s: float, milp_s: float) -> None:
    [b] = common.build_blocks([bid])
    t0 = time.time()
    inst, n_free = instance(b)
    qg = q - n_free
    rec: dict = dict(block=bid, q=q, parcels=len(b.parcels), free=n_free, q_groups=qg,
                     groups=len(inst.groups), lp_cap_s=lp_s, milp_cap_s=milp_s)
    if qg <= 0:
        rec.update(lb=0.0, ub=0.0, proven=True, tree_m=0.0)
        (OUT / f"q{q}.json").write_text(json.dumps(rec, indent=1) + "\n")
        return
    lb, xe, f, cuts, st = gst.quota_lp_bound(inst, qg, time_limit=lp_s, max_rounds=MAX_ROUNDS,
                                             purge_after=PURGE_AFTER)
    rec.update(lp_lb=lb, **st)
    print(rec, flush=True)
    import geopandas as gpd  # noqa: PLC0415
    floor_tree = edges_of(inst, gpd.read_parquet(HERE / "runs" / "steiner.parquet").geometry)
    t = time.time()
    tree, ub = gst.quota_upper_bound(inst, qg, x_edge=xe, restarts=RESTARTS, seed=SEED,
                                     starts=[floor_tree])
    rec.update(heur_ub=ub, heur_s=time.time() - t)
    milp_tree, bound, st = gst.quota_exact(inst, f, cuts, qg, time_limit=milp_s)
    rec.update(milp_bound=bound, milp_ub=(gst.tree_cost(inst, milp_tree) if milp_tree else None),
               **st)
    if milp_tree is not None and gst.tree_cost(inst, milp_tree) < ub - 1e-9:
        tree, ub = milp_tree, gst.tree_cost(inst, milp_tree)
    rec["ub"] = ub
    rec["lb"] = max(lb, bound)            # bound is the MILP optimum when proven
    rec["served_groups"] = gst.n_served(inst, tree)
    roads = gst.roads_gdf(inst, tree, b)
    rec["tree_m"] = float(roads.geometry.length.sum())
    rec["total_s"] = time.time() - t0
    roads.to_parquet(OUT / f"q{q}.parquet")
    (OUT / f"q{q}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)


def polish(bid: str) -> None:
    """Upper bounds from every LARGER q's tree trimmed to q (gst.quota_improve): a tree serving
    q' >= q is a start the per-q runs did not have (they started from the 1630.7 m floor tree
    only). Rewrites q<q>.json's ub (polish_ub records it) and .parquet where that improves."""
    [b] = common.build_blocks([bid])
    inst, n_free = instance(b)
    import geopandas as gpd  # noqa: PLC0415
    recs = {int(p.stem[1:]): json.loads(p.read_text()) for p in OUT.glob("q*.json")}
    trees = {q: edges_of(inst, gpd.read_parquet(OUT / f"q{q}.parquet").geometry)
             for q in recs if (OUT / f"q{q}.parquet").exists()}
    for q in sorted(recs):
        qg = q - n_free
        if qg <= 0:
            continue
        cands = [gst.quota_improve(inst, t, qg) for q2, t in trees.items() if q2 >= q]
        best = min(cands, key=lambda t: gst.tree_cost(inst, t))
        c = gst.tree_cost(inst, best)
        rec = recs[q]
        rec["polish_ub"] = c
        if c < rec["ub"] - 1e-9:
            rec["ub"] = c
            trees[q] = best
            gst.roads_gdf(inst, best, b).to_parquet(OUT / f"q{q}.parquet")
            rec["tree_m"] = c
            rec["served_groups"] = gst.n_served(inst, best)
        (OUT / f"q{q}.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(q, round(rec["lb"], 1), round(c, 1), round(rec["ub"], 1), flush=True)


if __name__ == "__main__":
    rr._check_single_threaded()
    OUT.mkdir(parents=True, exist_ok=True)
    match sys.argv[1:]:
        case ["grid", bid]:
            [b] = common.build_blocks([bid])
            print(" ".join(map(str, grid(len(b.parcels)))))
        case ["polish", bid]:
            polish(bid)
        case ["one", bid, q, lp_s, milp_s]:
            one(bid, int(q), lp_s=float(lp_s), milp_s=float(milp_s))
        case _:
            raise SystemExit(__doc__)
