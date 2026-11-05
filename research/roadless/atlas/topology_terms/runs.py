"""The networks topology_terms scores that no earlier study left on 41785 (terms.py):

  topology <alpha> <seed>   one more sample of the original topology algorithm (TopologyMethod, as
                            atlas/topology_run.py runs the kept one: 7 m roads, the Cape Town
                            footprints block), for the best-of-N its papers report; its roads and
                            timing to runs/topology.a<alpha>.s<seed>.{parquet,json}. (The
                            2026-11-04 runs wrote through Path.with_suffix, which took ".s<seed>"
                            for a suffix: each alpha's last finisher overwrote the others, so
                            only a100 s0 and a10 s1 kept their roads; every run's length and
                            timing stands in its log, runs/topology.a<alpha>.s<seed>.log.)
  cycle_native              the Loop Network (road_rescore.lineup_arms()["cycle_native"], the
                            lineup's arm the 5810 build-out page shows), fresh, in its canonical
                            order (common.canonical_order: the order the Demo cuts its prefixes
                            in); to runs/cycle_native.{parquet,ordered.parquet,json}.

Single-threaded (road_rescore.SINGLE), refused otherwise.

    R=research/roadless/atlas/topology_terms/runs.py   # from the repo root, PYTHONPATH=.
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
        PYTHONDONTWRITEBYTECODE=1 uv run python $R topology 10 1 <block>
"""
from __future__ import annotations

import json
import resource
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))           # research/roadless: common, road_rescore
import common  # noqa: E402
import road_rescore as rr  # noqa: E402

RUNS = HERE / "runs"
ROAD_W = 7.0             # m: topology_run.py's (the kept run's) road width


def info(bid: str, t0: float, c0: float, roads, **extra) -> dict:  # type: ignore[no-untyped-def]
    return dict(block=bid, wall_s=time.time() - t0, cpu_s=time.process_time() - c0,
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                host=socket.gethostname(), n_roads=len(roads),
                road_m=float(roads.geometry.length.sum()),
                finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)


def topology(bid: str, alpha: float, seed: int) -> None:
    from reblock.methods.topology import TopologyMethod
    [b] = common.build_blocks([bid])
    stem = f"topology.a{alpha:g}.s{seed}"
    t0, c0 = time.time(), time.process_time()
    p = TopologyMethod(alpha=alpha, seed=seed, road_width_m=ROAD_W).propose(b)
    assert p.roads is not None
    p.roads.to_parquet(RUNS / f"{stem}.parquet")
    rec = info(bid, t0, c0, p.roads, alpha=alpha, seed=seed, road_width_m=ROAD_W)
    (RUNS / f"{stem}.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)


def cycle_native(bid: str) -> None:
    method = rr.lineup_arms()["cycle_native"]
    [b] = common.build_blocks([bid])
    t0, c0 = time.time(), time.process_time()
    p = method.propose(b)
    assert p.roads is not None
    p.roads.to_parquet(RUNS / "cycle_native.parquet")
    common.canonical_order(b, p.roads).to_parquet(RUNS / "cycle_native.ordered.parquet")
    rec = info(bid, t0, c0, p.roads, identity=repr(method.identity),
               road_width_m=float(method.road_width_m))
    (RUNS / "cycle_native.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    rr._check_single_threaded()
    RUNS.mkdir(exist_ok=True)
    match sys.argv[1:]:
        case ["topology", alpha, seed, bid]:
            topology(bid, float(alpha), int(seed))
        case ["cycle_native", bid]:
            cycle_native(bid)
        case _:
            raise SystemExit(__doc__)
