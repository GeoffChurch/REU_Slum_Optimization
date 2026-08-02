"""Guards for the two LP constraint families that carry the method's claims.

Both are fault-injected: the constraint each test names was deleted from the LP and the test was
confirmed to fail, then restored. A guard nobody has broken on purpose guards nothing.
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.budget import _noded_graph, building_radii, displacement
from reblock.contracts import Block
from reblock.methods.resistance_lp import ResistanceLPReblocker, solve_coverage_lp

CORRIDOR_M = 3.0


def _grid_block(nx_: int = 6, ny: int = 6, step: float = 10.0) -> Block:
    """A dense lattice of square parcels with one building each, streeted on the south edge only.

    Deep interior parcels are the point: with street on one side only, the LP has to drive roads
    inward past buildings, which is where a displacement budget can actually bind.
    """
    polys, pts = [], []
    for i in range(nx_):
        for j in range(ny):
            x0, y0 = i * step, j * step
            polys.append(Polygon([(x0, y0), (x0 + step, y0), (x0 + step, y0 + step),
                                  (x0, y0 + step)]))
            pts.append(Point(x0 + step / 2, y0 + step / 2))
    parcels = gpd.GeoDataFrame({"parcel_id": [str(k) for k in range(len(polys))]},
                               geometry=polys, crs=CRS.from_epsg(32734))
    streets = gpd.GeoDataFrame(geometry=[LineString([(-step, 0.0), (nx_ * step + step, 0.0)])],
                               crs=CRS.from_epsg(32734))
    return Block(block_id="lp-test", crs=CRS.from_epsg(32734),
                 boundary=Polygon([(0, 0), (nx_ * step, 0), (nx_ * step, ny * step),
                                   (0, ny * step)]),
                 parcels=parcels, streets=streets,
                 building_points=gpd.GeoDataFrame(geometry=pts, crs=CRS.from_epsg(32734)))


@pytest.mark.parametrize("cap", [0.05, 0.15])
def test_displacement_budget_is_respected(cap: float) -> None:
    """The `sum_b u_b <= D*n` row must bind: output displacement cannot exceed the cap.

    This is the method's whole premise -- it budgets displacement rather than metres -- so if the
    cap does not hold, nothing measured about it means anything.

    FAULT INJECTION: neutering `_round`'s running-max displacement check drives displacement to
    0.989 against both caps, failing here. Note that deleting the LP's own `sum_b u_b` budget row
    does NOT fail this test -- the rounding enforces the cap independently -- which is why
    `test_lp_respects_displacement_budget` guards that row separately.
    """
    block = _grid_block()
    roads = ResistanceLPReblocker(max_displacement=cap).propose(block).roads
    assert roads is not None and len(roads) > 0, "no roads: the cap is vacuous"
    radii = building_radii(block.building_points)
    got = displacement(block.building_points, radii, roads) / len(
        block.building_points)
    assert got <= cap + 1e-9, f"displacement {got:.4f} exceeds cap {cap}"


def test_every_road_reaches_the_street() -> None:
    """`x_p <= z_s` must make connectivity structural: no floating segments.

    A segment that does not connect to the street is not a road, and an LP that can buy one is
    optimizing a fiction -- which is exactly the failure mode route (A) was shelved for.

    FAULT INJECTION: two independent faults fail this. Committing each path LEAF-first instead of
    street-first orphans 6 roads; replacing the path-structured rounding with a raw per-segment
    `z`-ordered commit -- the free-edge rounding route (A) was shelved for -- orphans far more.
    Budget truncation is what exposes both: a prefix of a street-first path still reaches the
    street, a suffix does not.
    """
    block = _grid_block()
    roads = ResistanceLPReblocker(max_displacement=0.15).propose(block).roads
    assert roads is not None and len(roads) > 0

    graph = _noded_graph(roads, block.streets)
    street_nodes = set(_noded_graph(
        gpd.GeoDataFrame(geometry=[], crs=block.crs), block.streets))
    reach: set[object] = set()
    for comp in nx.connected_components(graph):
        if comp & street_nodes:
            reach |= comp
    orphan = [g for g in roads.geometry
              if not all(tuple(np.round(c, 2)) in reach for c in g.coords)]
    assert not orphan, f"{len(orphan)}/{len(roads)} roads do not reach the street"


def test_lp_respects_displacement_budget() -> None:
    """`solve_coverage_lp`'s own `sum_b u_b <= D*n` row, guarded directly.

    The end-to-end test above cannot see this row, because `_round` re-enforces the cap on the way
    out. A hand-built instance with no rounding in the way can: three unit-length segments, one
    displacing building 0, one displacing building 1, one displacing nothing, and a budget of one
    building. The LP must decline to build both of the first two.

    FAULT INJECTION: deleting the `sum_b u_b` row lets the LP build everything, giving displacement
    2.0 against a budget of 1.0, failing here.
    """
    seg_len = np.array([10.0, 10.0, 10.0])
    path_segs = [[0], [1], [2]]
    seg_edges = [np.array([0]), np.array([1]), np.array([2])]
    seg_disp = [(np.array([0]), np.array([1.0])),
                (np.array([1]), np.array([1.0])),
                (np.zeros(0, dtype=np.int64), np.zeros(0))]
    z = solve_coverage_lp(path_segs, seg_len, seg_edges, seg_disp,
                          edge_gain=np.array([5.0, 5.0, 1.0]), n_buildings=2,
                          base_c=np.zeros(2), disp_budget=1.0, len_budget=100.0)

    used = np.zeros(2)
    for s, (bidx, c) in enumerate(seg_disp):
        if bidx.size:
            used[bidx] = np.maximum(used[bidx], c * z[s])
    assert used.sum() <= 1.0 + 1e-6, f"LP displaced {used.sum():.3f} against a budget of 1.0"
    assert z[2] > 0.99, "the free segment carries gain at no displacement and must be built"


def test_segment_displacement_sees_buildings_beside_a_LONG_span() -> None:
    """The prefilter must measure to the segment, not to its vertices.

    An earlier version queried balls around each vertex, so a building beside the MIDDLE of a long
    span -- further than `half_width_m + rmax` from either end -- was never considered and silently
    contributed zero. 8.31% of this method's own substrate edges are long enough for that, so it
    made the LP under-count what it had spent.

    FAULT INJECTION: restore the per-vertex `query_ball_point` prefilter and the midpoint building
    drops out, failing the first assertion while the endpoint one still passes.
    """
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import LineString, Point

    from reblock.methods.resistance_lp import segment_displacement

    crs = "EPSG:32734"
    half, r = 3.5, 4.0
    # 100 m span; its midpoint is 50 m from either end, far outside any (half + rmax) ball
    seg = LineString([(0.0, 0.0), (100.0, 0.0)])
    mid, end = Point(50.0, 2.0), Point(2.0, 2.0)     # both 2 m off the line, inside half + r
    pts = gpd.GeoDataFrame(geometry=[mid, end], crs=crs)
    radii = np.array([r, r])

    (idx, c), = segment_displacement([seg], pts, radii, half)
    assert 0 in idx, "a building beside the MIDDLE of a long span must be counted"
    assert 1 in idx, "a building beside its END must still be counted"
    # both sit 2 m off the line, so d = max(2 - 3.5, 0) = 0 -> fully displaced
    assert c[list(idx).index(0)] == pytest.approx(1.0)
