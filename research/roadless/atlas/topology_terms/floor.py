"""The floor of topology's first objective on a block: the least new parcel-boundary road that gives
every parcel a corner on a road (universal access), solved as the group Steiner tree it is (gst.py,
vendored from the research/group-steiner branch; the method of its study220 runs, study.py `one`):
the shortest-path heuristic, the directed-cut LP bound, LP-guided restarts, and the MILP where the
gap stays open. Writes runs/steiner.parquet (the tree's new edges, 7 m wide like topology's),
runs/steiner.ordered.parquet (common.canonical_order, the Demo's prefix order) and
runs/steiner.json (upper and lower bounds, the proven optimum if any, timings).

    F=research/roadless/atlas/topology_terms/floor.py     # from the repo root, PYTHONPATH=.
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
        PYTHONDONTWRITEBYTECODE=1 uv run python $F <block> <lp seconds> <milp seconds>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))           # research/roadless: common, road_rescore
import common  # noqa: E402
import gst  # noqa: E402
import road_rescore as rr  # noqa: E402

RUNS = HERE / "runs"


def floor(bid: str, *, lp_s: float, milp_s: float) -> None:
    [b] = common.build_blocks([bid])
    t0 = time.time()
    inst = gst.build_instance(b, 0)
    rec: dict = dict(block=bid, nodes=inst.n, edges=len(inst.ec), groups=len(inst.groups),
                     unreachable_groups=inst.n_unreachable, build_s=time.time() - t0)
    print(rec, flush=True)
    t = time.time()
    _tree0, ub0, st = gst.upper_bound(inst, restarts=0)
    rec.update(sph=st["sph_first"], ub_improve=ub0, heur_s=time.time() - t)
    lb, xe, f, st = gst.lp_bound(inst, time_limit=lp_s)
    rec.update(lb=lb, **st)
    print(rec, flush=True)
    t = time.time()
    tree, ub, _ = gst.upper_bound(inst, x_edge=xe, restarts=20)
    rec.update(ub=ub, restart_s=time.time() - t)
    opt = ub if ub - lb <= 1e-6 * max(ub, 1.0) else None
    if opt is None:
        o, bound, st, milp_tree = gst.exact_tree(inst, f, time_limit=milp_s)
        rec.update(milp_bound=bound, **st)
        if o is not None:
            opt = o
            if o < ub - 1e-6:
                assert milp_tree is not None
                tree = milp_tree
    rec["opt"] = opt
    roads = gst.roads_gdf(inst, tree, b)
    rec["tree_m"] = float(roads.geometry.length.sum())
    rec["total_s"] = time.time() - t0
    roads.to_parquet(RUNS / "steiner.parquet")
    common.canonical_order(b, roads).to_parquet(RUNS / "steiner.ordered.parquet")
    (RUNS / "steiner.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    rr._check_single_threaded()
    RUNS.mkdir(exist_ok=True)
    match sys.argv[1:]:
        case [bid, lp_s, milp_s]:
            floor(bid, lp_s=float(lp_s), milp_s=float(milp_s))
        case _:
            raise SystemExit(__doc__)
