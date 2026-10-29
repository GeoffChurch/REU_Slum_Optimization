"""The original topology algorithm's own build order on a block (owner 2026-10-29: is topology more
flattered by its own order or by the lineup's drainage order, and what does its time per step look
like?). The proposal keeps no order (reblock.methods.topology reads its new roads off the graph at
the end), so the run of topology_run.py is repeated as it was (TopologyMethod alpha 10, seed 0,
7 m roads, one thread) with the vendored algorithm watched from outside, nothing in ext/topology
edited and nothing it computes changed (`watching`, for the run's duration only):
  MyGraph.add_road_segment  each edge as the greedy appends it to the graph's road_edges: its
                            index, the wall-clock time and its two endpoints (the parcel graph's
                            frame);
  MyGraph.define_interior_parcels  called once after each chosen path: the path boundaries;
  build_all_roads (reblock.methods.topology's name for it)  when the greedy starts (what came
                            before it, the parcel graph and its streets, is setup).
The logged edges are matched to the proposal's roads by their endpoints (`in_order`: the frames
differ by the parcel graph's origin, found as the shift most endpoints share), every road once,
and the network is refused unless it is the kept run's (topology_run.py's: the same 546
segments, geometry for geometry). Writes beside it in the block's kept directory:
topology_built_roads.parquet (the roads in build order: per road `index`, its build position,
`path`, the chosen path it came in, 1 .., and `t`, seconds from the run's start) and
topology_built.json (the run's timing: wall, CPU, setup, peak memory, host).

    T=research/roadless/atlas/topology_order.py       # from the repo root, PYTHONPATH=.
    NUMBA_NUM_THREADS=1 uv run python $T run <block>
    uv run python $T timing <block>                    # per segment, per path and per frame
"""
from __future__ import annotations

import json
import os
import resource
import socket
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, NamedTuple

import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topology_run as tr  # noqa: E402  (the kept run's setup; puts common and block_site on the path)
from block_site import Site  # noqa: E402
from topology import MyGraph  # noqa: E402  (ext/topology, the vendored algorithm)

import reblock.methods.topology as rmt  # noqa: E402

common = tr.common
MATCH_M = 1e-6           # m: a logged edge's endpoints against a road's (the same floats, shifted)


class Logged(NamedTuple):
    """One add_road_segment call: the path it belongs to, when, and its two endpoints."""
    path: int
    t: float
    ends: tuple[float, float, float, float]


class Watch(NamedTuple):
    log: list[Logged]
    started: list[float]              # build_all_roads' start, seconds from t0 (one entry)


@contextmanager
def watching(t0: float) -> Iterator[Watch]:
    """The three seams wrapped for the duration (module docstring), restored after."""
    w = Watch(log=[], started=[])
    path = [0]
    add, define, build = (MyGraph.add_road_segment, MyGraph.define_interior_parcels,
                          rmt.build_all_roads)

    def add_logged(self: Any, edge: Any) -> Any:
        a, b = edge.nodes
        w.log.append(Logged(path=path[0], t=time.time() - t0, ends=(a.x, a.y, b.x, b.y)))
        return add(self, edge)

    def define_logged(self: Any, *args: Any, **kwargs: Any) -> Any:
        path[0] += 1
        return define(self, *args, **kwargs)

    def build_logged(*args: Any, **kwargs: Any) -> Any:
        w.started.append(time.time() - t0)
        return build(*args, **kwargs)

    MyGraph.add_road_segment = add_logged               # type: ignore[method-assign]
    MyGraph.define_interior_parcels = define_logged     # type: ignore[method-assign]
    rmt.build_all_roads = build_logged                  # type: ignore[attr-defined]
    try:
        yield w
    finally:
        MyGraph.add_road_segment = add                  # type: ignore[method-assign]
        MyGraph.define_interior_parcels = define        # type: ignore[method-assign]
        rmt.build_all_roads = build                     # type: ignore[attr-defined]


def in_order(roads: gpd.GeoDataFrame, log: list[Logged]) -> gpd.GeoDataFrame:
    """`roads` (world frame) in the logged order with `index`, `path` (renumbered 1 ..) and `t`;
    refuses a road no logged edge matches."""
    xy = np.asarray([list(g.coords[0]) + list(g.coords[-1]) for g in roads.geometry])
    lg = np.asarray([e.ends for e in log])
    shifts = (xy[:, None, :2] - lg[None, :, :2]).reshape(-1, 2)
    vals, counts = np.unique(np.round(shifts, 2), axis=0, return_counts=True)
    near = np.abs(shifts - vals[np.argmax(counts)]).max(axis=1) < 0.02
    o = shifts[near].mean(axis=0)
    mid = cKDTree((xy[:, :2] + xy[:, 2:]) / 2)
    first: dict[int, int] = {}
    for k, e in enumerate(lg):
        a, b = e[:2] + o, e[2:] + o
        d, i = mid.query((a + b) / 2)
        i = int(i)
        ends = np.sort(np.stack([a, b]), axis=0)
        mine = np.sort(np.stack([xy[i, :2], xy[i, 2:]]), axis=0)
        if d < MATCH_M and np.abs(ends - mine).max() < MATCH_M and i not in first:
            first[i] = k
    if len(first) != len(roads):
        raise AssertionError(f"{len(roads) - len(first)} of {len(roads)} roads not in the log")
    rows = sorted(first, key=lambda i: first[i])
    paths = np.asarray([log[first[i]].path for i in rows])
    _, renum = np.unique(paths, return_inverse=True)
    return roads.iloc[rows].reset_index(drop=True).assign(
        index=np.arange(len(rows)), path=renum + 1, t=[log[first[i]].t for i in rows])


def same_network(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> bool:
    """The two road sets equal geometry for geometry (order and direction aside)."""
    def keys(g: gpd.GeoDataFrame) -> list[str]:
        return sorted(shapely.to_wkt(shapely.normalize(g.geometry.to_numpy()),
                                     rounding_precision=6))
    return len(a) == len(b) and keys(a) == keys(b)


def run(site: Site) -> None:
    if os.environ.get("NUMBA_NUM_THREADS") != "1":
        raise SystemExit("NUMBA_NUM_THREADS=1: the fit's runs are single-threaded")
    kept = gpd.read_parquet(site.kept / "topology_roads.parquet")
    [b] = common.build_blocks([site.bid])
    method = rmt.TopologyMethod(alpha=tr.ALPHA, seed=tr.SEED, road_width_m=tr.ROAD_W)
    print(f"{site.bid}: {len(b.buildings)} buildings; topology alpha {tr.ALPHA:g} seed {tr.SEED} "
          f"watched, on {socket.gethostname()} ({tr.cpu_model()})", flush=True)
    w0, c0 = time.time(), time.process_time()
    with watching(w0) as w:
        p = method.propose(b)
    wall, cpu = time.time() - w0, time.process_time() - c0
    assert p.roads is not None
    if not same_network(p.roads, kept):
        raise SystemExit(f"seed {tr.SEED} did not reproduce the kept run: {len(p.roads)} roads "
                         f"against {len(kept)}")
    roads = in_order(p.roads, w.log)
    roads.to_parquet(site.kept / "topology_built_roads.parquet")
    [started] = w.started
    info = dict(block=site.bid, alpha=tr.ALPHA, seed=tr.SEED, road_width_m=tr.ROAD_W,
                wall_s=wall, cpu_s=cpu, greedy_start_s=started,
                first_road_s=float(roads.t.min()), last_road_s=float(roads.t.max()),
                n_roads=len(roads), n_paths=int(roads.path.max()), n_logged=len(w.log),
                peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20,
                host=socket.gethostname(), cpu=tr.cpu_model(), same_as_kept=True,
                finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    (site.kept / "topology_built.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1), flush=True)


def summary(x: np.ndarray) -> str:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return (f"n {len(x)}, median {med:.2f} s, IQR {q1:.2f} -- {q3:.2f} s, max {x.max():.1f} s, "
            f"total {x.sum():.0f} s")


def timing(site: Site) -> None:
    """Per segment and per path the seconds since the last, and the trend over the run (the
    run's thirds by build index)."""
    roads = gpd.read_parquet(site.kept / "topology_built_roads.parquet")
    info = json.loads((site.kept / "topology_built.json").read_text())
    t = roads.t.to_numpy()
    seg = np.diff(np.concatenate([[info["greedy_start_s"]], t]))
    path_t = roads.groupby("path").t.max().to_numpy()
    per_path = np.diff(np.concatenate([[info["greedy_start_s"]], path_t]))
    print(f"wall {info['wall_s']:.0f} s; greedy starts at {info['greedy_start_s']:.1f} s "
          f"({info['greedy_start_s'] / info['wall_s']:.2%}), first road at "
          f"{info['first_road_s']:.1f} s ({info['first_road_s'] / info['wall_s']:.2%}), last at "
          f"{info['last_road_s']:.0f} s")
    print(f"per segment: {summary(seg)}")
    print(f"per path:    {summary(per_path)}")
    for k, part in enumerate(np.array_split(per_path, 3)):
        print(f"  paths' third {k + 1}: {summary(part)}")


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["run", bid]:
            run(Site.of(bid))
        case ["timing", bid]:
            timing(Site.of(bid))
        case _:
            raise SystemExit(__doc__)
