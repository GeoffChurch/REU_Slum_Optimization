"""One greedy step of the lineup's arterial, timed piece by piece, at a committed network of S
segments: the first S roads of the block's stored whole arterial network (street-first order),
for several S. Calls the same functions, in the same order, as `engines._greedy_shortlist`'s loop
body, with `n_eval` of the shortlist scored (the loop scores all k=512; extrapolated linearly).

usage: PYTHONPATH=. python microbench.py <block_id> <S,S,...> <n_eval> <out.csv>
"""
import sys
import time
from collections import defaultdict

sys.path.insert(0, "research/roadless")
import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402
from shapely import STRtree  # noqa: E402

import common  # noqa: E402
import road_rescore  # noqa: E402
from reblock.derive import access  # noqa: E402
from reblock.derive.access import STREET_TOL, ParcelAdjacency, past_every_parcel  # noqa: E402
from reblock.methods.arterial import costs, objectives, realize, scoring  # noqa: E402
from reblock.methods.arterial.engines import _step_network  # noqa: E402
from reblock.methods.arterial.primitives import (  # noqa: E402
    _anchor_points, _candidate_chords, _deep_targets, _snap_graph)
from reblock.methods.arterial.shortlist import FirstOrder, RankContext  # noqa: E402
from reblock.methods.boundary_graph import _boundary_graph  # noqa: E402

bid, Ss, n_eval, out = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], int(sys.argv[3]), \
    sys.argv[4]
acc = defaultdict(float)


def timed(name, fn):
    def w(*a, **k):
        t = time.perf_counter()
        r = fn(*a, **k)
        acc[name] += time.perf_counter() - t
        return r
    return w


realize.SnapToBoundary.realize = timed("e_snap", realize.SnapToBoundary.realize)
objectives._Unioned.of = timed("e_gain", objectives._Unioned.of)
objectives._union_with = timed("e_gain_union", objectives._union_with)
objectives._explode = timed("e_gain_explode", objectives._explode)
objectives.parcel_access_layers = timed("e_gain_peel", objectives.parcel_access_layers)
access.street_connectivity = timed("street_connectivity", access.street_connectivity)
costs._NewlyDisplaced.of = timed("e_cost", costs._NewlyDisplaced.of)

[b] = common.build_blocks([bid])
m = road_rescore.lineup_arms()[road_rescore.ARTERIAL]
ordered = gpd.read_parquet(f"research/roadless/road_rows/lineup-caf14fa2/scratch/networks/{bid}/"
                           f"{road_rescore.ARTERIAL}.ordered.parquet")
hw = m.road_width_m / 2
adjacency = ParcelAdjacency.of(b, STREET_TOL)
sg = _snap_graph(_boundary_graph(b.parcels))
streets = list(b.streets.geometry)
scorer = m.objective.for_block(adjacency)
parcel_tree = STRtree(list(b.parcels.geometry))
building_tree = STRtree(shapely.points(b.buildings.xy))
ids = b.parcels["parcel_id"]
sel = FirstOrder(m.engine.k, threads=1)
rows = []
for S in Ss:
    committed = list(ordered.geometry.iloc[:S])
    acc.clear()
    r = dict(block=bid, S=S)
    t = time.perf_counter()
    anchors = _anchor_points([*streets, *committed], m.n_anchors, m.max_anchors)
    r["t_anchors"] = time.perf_counter() - t
    t = time.perf_counter()
    net = _step_network(committed, b, hw)
    curr = net.roads if S else None
    targets = _deep_targets(adjacency, curr, m.top_k)
    r["t_net_targets"] = time.perf_counter() - t
    t = time.perf_counter()
    cands = _candidate_chords(anchors, targets)
    r["t_candidates"] = time.perf_counter() - t
    t = time.perf_counter()
    depths = access.parcel_access_layers(adjacency, curr, unreached=past_every_parcel)
    short = sel.select(cands, RankContext(depths=depths.loc[ids].to_numpy(dtype=float),
                                          parcel_tree=parcel_tree, building_tree=building_tree,
                                          half_width_m=hw, step=S))
    r["t_rank"] = time.perf_counter() - t
    t = time.perf_counter()
    scoring._STEP_STATE = scoring.step_state(b, sg=sg, realizer=m.realizer, objective=scorer,
                                             cost=m.cost, committed=net)
    r["t_step_state"] = time.perf_counter() - t
    acc.clear()
    t = time.perf_counter()
    for c in short[:n_eval]:
        scoring.eval_candidate(c)
    te = time.perf_counter() - t
    scoring._STEP_STATE = None
    n = min(n_eval, len(short))
    r.update(anchors=len(anchors), candidates=len(cands), shortlisted=len(short),
             ms_per_eval=1000 * te / n, **{f"ms_{k}": 1000 * v / n for k, v in acc.items()})
    r["step_s_est"] = (r["t_anchors"] + r["t_net_targets"] + r["t_candidates"] + r["t_rank"]
                       + r["t_step_state"] + te / n * len(short))
    rows.append(r)
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
pd.DataFrame(rows).to_csv(out, index=False)
