"""The original topology algorithm (Brelsford and Bettencourt's, reblock.methods.topology) run once
on a block, timed as the six runs explore_proto/build.py's extrapolation is fitted on (topo_obj.py
`time`, GeoffChurch/REU_Slum_Optimization branch research/group-steiner): TopologyMethod(alpha 10,
seed 0, road width 7 m), single-threaded (NUMBA_NUM_THREADS=1, refused otherwise), the block from
the Cape Town footprints source (common.build_blocks, study220's source). Writes into the block's
kept directory (block_site.Site.kept, atlas/blocks/<block>/): topology_roads.parquet (the
proposal's roads, the block's CRS) and topology.json (wall and CPU seconds, peak resident memory,
the host, the roads' count and length, the run's parameters).

`score` scores those roads as the build-out scores a network (compute.frames_of on one frame, the
whole network: its corridor carved as a kappa-10 road under RUNTIME, each home's escape time, the
sheet), into the Site's networks as TOPOLOGY_KEY's run at<DDD> (an answer at its own D, as
networks.py names one), so the Demo can draw it beside the
build-out; its D, L_P and C go into topology.json beside the timing.

    T=research/roadless/atlas/topology_run.py         # from the repo root, PYTHONPATH=.
    NUMBA_NUM_THREADS=1 uv run python $T run <block>
    uv run python $T score <block>
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
sys.path.insert(0, str(HERE / "roads5810"))
import common  # noqa: E402
from block_site import Site  # noqa: E402

from reblock.methods.topology import TopologyMethod  # noqa: E402

ALPHA, SEED, ROAD_W = 10.0, 0, 7.0                 # topo_obj.py `time`'s TopologyMethod
TOPOLOGY_KEY = "topology.a10.s0.w7"                # its frames' name in the Site's networks


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def run(site: Site) -> None:
    if os.environ.get("NUMBA_NUM_THREADS") != "1":
        raise SystemExit("NUMBA_NUM_THREADS=1: the fit's runs are single-threaded")
    bid = site.bid
    out = site.kept
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


def score(site: Site) -> None:
    """The kept roads scored (module docstring), their D, L_P and C into topology.json."""
    import geopandas as gpd  # noqa: PLC0415
    import networks as nw  # noqa: PLC0415  (the build-out's sources; heavy, so here)
    import numpy as np  # noqa: PLC0415
    import road_simp as rs  # noqa: PLC0415
    import shapely  # noqa: PLC0415

    from reblock.budget import road_corridor  # noqa: PLC0415
    t0 = time.time()
    [b] = common.build_blocks([site.bid])
    roads = gpd.read_parquet(site.kept / "topology_roads.parquet")
    ground = shapely.intersection(road_corridor(roads), b.boundary)
    none = np.zeros(len(b.buildings.outlines), dtype=bool)
    metric = nw.cp.gcl.complex_of(nw.metrics.RUNTIME,
                                  factorization=nw.cp.gap_factor.SuperLU())
    _m, _lens, row = rs.scored(b, metric, ground, cleared=none)
    a = nw.Answer(budget=row["D_exact"], ground=ground, lines=roads.geometry.to_numpy(),
                  cleared=none, L_P=row["L_P"], D=row["D_exact"])
    site.networks.mkdir(parents=True, exist_ok=True)
    run_name = nw.run_name(a.D)
    nw.cp.frames_of(TOPOLOGY_KEY, run_name, b, nw.One(b, a), t0,
                    frames=nw.cp.Pass(targets=(), full=True), data=site.networks,
                    meshes=site.networks / ".meshes")
    [r] = nw.info_of(TOPOLOGY_KEY, run_name, data=site.networks)["frames"]
    if abs(r["L_P"] - a.L_P) > 1e-6:
        raise AssertionError(f"L_P {r['L_P']} on the frames' mesh, {a.L_P} on its own")
    [C] = nw.added_of(TOPOLOGY_KEY, data=site.networks).values()
    info = json.loads((site.kept / "topology.json").read_text())
    info |= dict(D=r["D_area"], L_P=r["L_P"], L_J2=r["L_J2"], C=C, scored_key=TOPOLOGY_KEY,
                 scored_run=run_name, metric=metric.name)
    (site.kept / "topology.json").write_text(json.dumps(info, indent=1) + "\n")
    print(f"{site.bid} topology: D {r['D_area']:.4f}, L_P {r['L_P']:.4f}, C {C:.0f} m^2, "
          f"{r['road_m']:.0f} m ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["run", bid]:
            run(Site.of(bid))
        case ["score", bid]:
            score(Site.of(bid))
        case _:
            raise SystemExit(__doc__)
