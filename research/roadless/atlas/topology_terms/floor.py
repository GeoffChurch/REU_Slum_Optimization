"""The floor of topology's first objective on a block: the least new parcel-boundary road that gives
every parcel a corner on a road (universal access), solved as the group Steiner tree it is: gst.py's
quota formulation at q = every parcel (quota.solve; at that q every y_g is 1 and it is the plain
directed-cut formulation). The cut LP run to convergence, the heuristic (shortest path, LP-guided
and perturbed restarts, trimming), then the MILP on the LP's cuts, which proves the optimum on
41785 (1580.4 m; the heuristic alone gives 1630.7 m).

Writes runs/steiner.parquet (the tree's new edges, 7 m wide like topology's),
runs/steiner.ordered.parquet (common.canonical_order, the Demo's prefix order), runs/steiner.json
(bounds, timings), runs/steiner.cuts.npz (the LP's converged cut set: valid at every q, quota.py's
seed) and runs/quota/q<every parcel>.json / .parquet (the quota table's top row).

    F=research/roadless/atlas/topology_terms/floor.py     # from the repo root, PYTHONPATH=.
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
        PYTHONDONTWRITEBYTECODE=1 uv run python $F <block> <lp seconds> <milp seconds>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))           # research/roadless: common, road_rescore
import common  # noqa: E402
import quota  # noqa: E402
import road_rescore as rr  # noqa: E402

RUNS = HERE / "runs"


def floor(bid: str, *, lp_s: float, milp_s: float) -> None:
    [b] = common.build_blocks([bid])
    n = len(b.parcels)
    s = quota.solve(b, n, lp_s=lp_s, milp_s=milp_s, seed=[], starts=[])
    if not s.rec["lp_converged"]:
        print("WARNING: the LP did not converge; its cuts are still a valid seed", flush=True)
    s.roads.to_parquet(RUNS / "steiner.parquet")
    common.canonical_order(b, s.roads).to_parquet(RUNS / "steiner.ordered.parquet")
    quota.save_cuts(RUNS / "steiner.cuts.npz", s.cuts)
    text = json.dumps(s.rec, indent=1) + "\n"
    (RUNS / "steiner.json").write_text(text)
    quota.OUT.mkdir(parents=True, exist_ok=True)
    s.roads.to_parquet(quota.OUT / f"q{n}.parquet")
    (quota.OUT / f"q{n}.json").write_text(text)
    print(json.dumps(s.rec), flush=True)


if __name__ == "__main__":
    rr._check_single_threaded()
    RUNS.mkdir(exist_ok=True)
    match sys.argv[1:]:
        case [bid, lp_s, milp_s]:
            floor(bid, lp_s=float(lp_s), milp_s=float(milp_s))
        case _:
            raise SystemExit(__doc__)
