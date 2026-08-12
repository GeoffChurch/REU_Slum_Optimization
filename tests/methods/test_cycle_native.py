"""Guards for the property that defines this method: its output is BRIDGELESS.

Fault-injected -- the fault named in each docstring was applied, the test confirmed to fail, and the
code restored. A guard nobody has broken on purpose guards nothing.
"""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.budget import _rnd
from reblock.contracts import Block
from reblock.methods.cycle_native import CycleNativeReblocker

UTM = CRS.from_epsg(32734)


def _block(nx_: int = 7, ny: int = 7, step: float = 10.0) -> Block:
    """A lattice with street on ONE side only, so the interior must be reached and returned from."""
    polys, pts = [], []
    for i in range(nx_):
        for j in range(ny):
            x0, y0 = i * step, j * step
            polys.append(Polygon([(x0, y0), (x0 + step, y0), (x0 + step, y0 + step),
                                  (x0, y0 + step)]))
            pts.append(Point(x0 + step / 2, y0 + step / 2))
    return Block(
        block_id="cyc", crs=UTM,
        boundary=Polygon([(0, 0), (nx_ * step, 0), (nx_ * step, ny * step), (0, ny * step)]),
        parcels=gpd.GeoDataFrame({"parcel_id": [str(k) for k in range(len(polys))]},
                                 geometry=polys, crs=UTM),
        streets=gpd.GeoDataFrame(
            geometry=[LineString([(-step, 0.0), (nx_ * step + step, 0.0)])], crs=UTM),
        building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))


def _road_graph(block: Block, roads: gpd.GeoDataFrame) -> nx.Graph:
    from shapely.ops import unary_union
    noded = unary_union([*roads.geometry, *block.streets.geometry])
    pieces = list(noded.geoms) if hasattr(noded, "geoms") else [noded]
    g: nx.Graph = nx.Graph()
    for piece in pieces:
        cs = list(piece.coords)
        for a, b in zip(cs, cs[1:], strict=False):
            na, nb = _rnd(a), _rnd(b)
            if na != nb:
                g.add_edge(na, nb)
    return g


def test_output_is_bridgeless() -> None:
    """No road this method emits may be a bridge -- that is the whole point of the method.

    Bridgelessness is what makes the output strongly orientable (Robbins' theorem), i.e. what makes
    every road one-way-able with no repair pass. It is not an incidental property.

    FAULT INJECTION: making the return leg reuse the outbound edges (dropping the `keep` mask that
    removes them from the second Dijkstra) fails this test -- though via the vacuity guard below
    rather than the bridge assertion. The retraced return duplicates the outbound path, so it adds
    displacement without adding permeability, no candidate ever clears `gain > 0`, and the method
    emits NOTHING. Worth stating precisely: the guard that fires is `len(roads) > 0`, which is why
    that assertion is here and not merely defensive.
    """
    block = _block()
    roads = CycleNativeReblocker(max_displacement=0.20).propose(block).roads
    assert roads is not None and len(roads) > 0, "no roads: bridgelessness is vacuous"

    g = _road_graph(block, roads)
    road_nodes = set()
    for geom in roads.geometry:
        cs = list(geom.coords)
        for a, b in zip(cs, cs[1:], strict=False):
            na, nb = _rnd(a), _rnd(b)
            if na != nb:
                road_nodes |= {na, nb}
    bridges = [e for e in nx.bridges(g) if set(e) <= road_nodes] if nx.is_connected(g) else []
    assert not bridges, f"{len(bridges)} emitted road edges are bridges: {bridges[:3]}"


@pytest.mark.parametrize("cap", [0.05, 0.15])
def test_respects_its_displacement_budget(cap: float) -> None:
    """The method's INTENDED stopping rule must bind (`max_cycles` is a safety valve above it).

    FAULT INJECTION: removing the `if d > self.max_displacement: continue` candidate rejection lets
    it run to 0.30+ displacement against a 0.05 cap, failing here.
    """
    from reblock.budget import building_radii, displacement

    block = _block()
    roads = CycleNativeReblocker(max_displacement=cap).propose(block).roads
    assert roads is not None
    if not len(roads):
        return
    radii = building_radii(block.building_points)
    got = displacement(block.building_points, radii, roads) / len(block.building_points)
    assert got <= cap + 1e-9, f"displacement {got:.4f} exceeds its own cap {cap}"


@pytest.mark.parametrize("cap", [1, 3])
def test_max_cycles_caps_the_greedy(cap: int) -> None:
    """`max_cycles` must bind when it is the binding rule.

    It was a bare `range(60)` in the greedy loop: unconfigurable and untested, and it silently ended
    the reported curve short of `max_displacement` in every settlement region measured (all of them
    emitting exactly 120 segments = 60 cycles x 2 legs).

    FAULT INJECTION: restoring the literal `range(60)` makes cap=1 emit 20+ roads against the 2
    asserted here, failing this test.
    """
    roads = CycleNativeReblocker(max_displacement=0.9, max_cycles=cap).propose(_block()).roads
    assert roads is not None
    # Each accepted move appends its outbound leg plus, when a bridgeless return exists, the return
    # leg -- so at most two roads per cycle.
    assert len(roads) <= 2 * cap, f"max_cycles={cap} emitted {len(roads)} roads"


def test_max_cycles_is_what_binds_not_an_earlier_stop() -> None:
    """The cap test above passes vacuously if the greedy stops early for some OTHER reason, which
    would defend the bug rather than catch it. Raising the cap on the same block must buy more road
    -- proving the cap is the live constraint and the assertion above is load-bearing."""
    block = _block()
    small = CycleNativeReblocker(max_displacement=0.9, max_cycles=2).propose(block).roads
    large = CycleNativeReblocker(max_displacement=0.9, max_cycles=8).propose(block).roads
    assert small is not None and large is not None
    assert len(large) > len(small), f"cap 8 gave {len(large)} roads, cap 2 gave {len(small)}"
