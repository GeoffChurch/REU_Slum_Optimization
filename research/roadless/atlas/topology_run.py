"""The original topology algorithm (Brelsford and Bettencourt's, reblock.methods.topology) run once
on a block, timed as the six runs explore_proto/build.py's extrapolation is fitted on (topo_obj.py
`time`, GeoffChurch/REU_Slum_Optimization branch research/group-steiner): TopologyMethod(alpha 10,
seed 0, road width 7 m), single-threaded (NUMBA_NUM_THREADS=1, refused otherwise), the block from
the Cape Town footprints source (common.build_blocks, study220's source). Writes into the block's
atlas directory (blocks/<block>/): topology_roads.parquet (the proposal's roads, the block's CRS)
and topology.json (wall and CPU seconds, peak resident memory, the host, the roads' count and
length, the run's parameters).

    PYTHONPATH=. NUMBA_NUM_THREADS=1 uv run python research/roadless/atlas/topology_run.py <block>
"""
from __future__ import annotations

import json
import os
import platform
import resource
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))               # research/roadless: common
import common  # noqa: E402

from reblock.methods.topology import TopologyMethod  # noqa: E402

BLOCKS = HERE / "blocks"     # the per-block atlas: one directory per block id
ALPHA, SEED, ROAD_W = 10.0, 0, 7.0                 # topo_obj.py `time`'s TopologyMethod


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def main(bid: str) -> None:
    if os.environ.get("NUMBA_NUM_THREADS") != "1":
        raise SystemExit("NUMBA_NUM_THREADS=1: the fit's runs are single-threaded")
    out = BLOCKS / bid
    out.mkdir(parents=True, exist_ok=True)
    [b] = common.build_blocks([bid])
    method = TopologyMethod(alpha=ALPHA, seed=SEED, road_width_m=ROAD_W)
    print(f"{bid}: {len(b.buildings)} buildings; topology alpha {ALPHA:g} seed {SEED} on "
          f"{socket.gethostname()} ({cpu_model()})", flush=True)
    w0, c0 = time.time(), time.process_time()
    p = method.propose(b)
    wall, cpu = time.time() - w0, time.process_time() - c0
    roads = p.roads
    assert roads is not None
    roads.to_parquet(out / "topology_roads.parquet")
    info = dict(block=bid, n_buildings=len(b.buildings), alpha=ALPHA, seed=SEED,
                road_width_m=ROAD_W, wall_s=wall, cpu_s=cpu,
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                host=socket.gethostname(), cpu=cpu_model(), n_roads=len(roads),
                road_m=float(roads.geometry.length.sum()),
                finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (out / "topology.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1), flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
