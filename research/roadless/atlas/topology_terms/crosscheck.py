"""The original code's parcel-to-parcel travel matrix on topology's kept network (the cross-check of
terms.py's `travel`): topology's own parcel graph (reblock.derive.parcel_graph.to_parcel_graph,
the streets marked as TopologyMethod.propose marks them), each of the kept run's 546 segments
(atlas/blocks/<block>/topology_roads.parquet) marked a road on the graph edge it came from, then
ext/topology's shortest_path_p2p_matrix(full=False, travelcost=False) run unchanged: road-only
travel, road nodes within 5 m joined, each parcel's centroid tied to its corners by zero-length
edges, the mean over n(n-1). Writes runs/crosscheck_topology.npz: the matrix (NaN unreachable),
its mean as the function returns it, each face's centroid in the block's CRS (to match faces to
block.parcels), and the seconds.

    X=research/roadless/atlas/topology_terms/crosscheck.py   # from the repo root, PYTHONPATH=.
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
        PYTHONDONTWRITEBYTECODE=1 uv run python $X <block>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))           # research/roadless: common, road_rescore
sys.path.insert(0, str(HERE.parent))               # atlas: block_site
import common  # noqa: E402
import road_rescore as rr  # noqa: E402
from block_site import Site  # noqa: E402
from topology.graph import my_graph_helpers as mgh  # noqa: E402

from reblock.derive.parcel_graph import to_parcel_graph  # noqa: E402
from reblock.methods.topology import _mark_streets_as_roads  # noqa: E402

RUNS = HERE / "runs"
MATCH_M = 1e-6       # m: a kept segment's endpoints against its graph edge's (the same floats)


def marked(b, roads: gpd.GeoDataFrame):  # type: ignore[no-untyped-def]
    """Topology's parcel graph with the streets and `roads` (each a graph edge) marked roads."""
    ppg = to_parcel_graph(b)
    g = ppg.graph
    _mark_streets_as_roads(g, b.streets, ppg.origin)
    ox, oy = ppg.origin
    by_ends = {}
    for e in g.myedges():
        a, c = e.nodes
        by_ends[tuple(sorted([(round(a.x, 6), round(a.y, 6)), (round(c.x, 6), round(c.y, 6))]))] = e
    for line in roads.geometry:
        (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
        key = tuple(sorted([(round(x0 - ox, 6), round(y0 - oy, 6)),
                            (round(x1 - ox, 6), round(y1 - oy, 6))]))
        if key not in by_ends:
            raise AssertionError(f"a kept segment matches no graph edge: {line.wkt}")
        g.add_road_segment(by_ends[key])
    g.define_interior_parcels()
    if g.interior_parcels:
        raise AssertionError(f"{len(g.interior_parcels)} interior parcels left: not the kept run")
    return g, ppg.origin


def main(bid: str) -> None:
    site = Site.of(bid)
    [b] = common.build_blocks([bid])
    g, (ox, oy) = marked(b, gpd.read_parquet(site.kept / "topology_roads.parquet"))
    t0 = time.time()
    mat, mean = mgh.shortest_path_p2p_matrix(g, full=False, travelcost=False)
    sec = time.time() - t0
    cen = np.asarray([(f.centroid.x + ox, f.centroid.y + oy) for f in g.inner_facelist])
    np.savez(RUNS / "crosscheck_topology.npz", T=np.asarray(mat, dtype=float), mean=mean,
             centroid=cen, seconds=sec)
    print(f"{bid}: {len(cen)} faces, mean {mean:.3f} m, {sec:.0f} s", flush=True)


if __name__ == "__main__":
    rr._check_single_threaded()
    match sys.argv[1:]:
        case [bid]:
            main(bid)
        case _:
            raise SystemExit(__doc__)
