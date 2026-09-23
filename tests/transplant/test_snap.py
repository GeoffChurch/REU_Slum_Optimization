"""Both snaps land every vertex on the recipient's substrate; only the routed one keeps every hop on
a substrate edge."""
from __future__ import annotations

from typing import cast

import geopandas as gpd
import numpy as np
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union

from reblock.buildings import SpacingDiscs
from reblock.contracts import Block
from reblock.methods.substrates import ChordSubstrate
from reblock.transplant.snap import NearestNodeSnap, RoutedSnap

UTM = CRS.from_epsg(32734)


def _grid(w: int, h: int, cell: float) -> Block:
    polys = [Polygon([(i * cell, j * cell), ((i + 1) * cell, j * cell),
                      ((i + 1) * cell, (j + 1) * cell), (i * cell, (j + 1) * cell)])
             for j in range(h) for i in range(w)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    return Block(block_id="grid", crs=UTM, boundary=cast(Polygon, unary_union(polys)),
                 parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (w * cell, 0)])], crs=UTM),
                 building_geometries=gpd.GeoDataFrame(
                     geometry=[Point(p.centroid.x, p.centroid.y) for p in polys], crs=UTM),
                 source_content_hash=None, building_tier=SpacingDiscs)


BLOCK = _grid(6, 5, 10.0)
LINES = gpd.GeoDataFrame(geometry=[
    LineString([(1.0, 1.0), (28.0, 43.0), (57.0, 2.0)]),
    MultiLineString([[(2.0, 48.0), (41.0, 18.0)], [(33.0, 33.0), (34.0, 34.0)]]),
], crs=UTM)


def _nodes() -> set[tuple[float, float]]:
    return {(float(x), float(y)) for x, y in ChordSubstrate().build(BLOCK).pts}


def _edges() -> set[frozenset[tuple[float, float]]]:
    g = ChordSubstrate().build(BLOCK)
    return {frozenset({(float(g.pts[u][0]), float(g.pts[u][1])),
                       (float(g.pts[v][0]), float(g.pts[v][1]))})
            for u, v in zip(g.rows, g.cols, strict=True)}


def _vertices(out: gpd.GeoDataFrame) -> list[list[tuple[float, float]]]:
    return [[(float(x), float(y)) for x, y in np.asarray(g.coords)] for g in out.geometry]


def test_nearest_node_puts_every_vertex_on_a_substrate_node_and_drops_a_collapsed_part() -> None:
    """The last part is under a metre long, so both its ends snap to one node and it goes."""
    out = NearestNodeSnap(ChordSubstrate()).snap(LINES, BLOCK)
    nodes = _nodes()
    lines = _vertices(out)
    assert len(lines) == 2
    for line in lines:
        assert all(v in nodes for v in line)
        assert all(a != b for a, b in zip(line, line[1:], strict=False))
    assert out.crs == BLOCK.crs


def test_routed_snap_keeps_every_hop_on_a_substrate_edge() -> None:
    """FAULT INJECTION: joining consecutive snapped nodes straight (the nearest-node snap's rule)
    fails the edge check -- the first line's nodes are nowhere near adjacent."""
    edges = _edges()
    lines = _vertices(RoutedSnap(ChordSubstrate()).snap(LINES, BLOCK))
    assert len(lines) == 2
    for line in lines:
        assert all(frozenset({a, b}) in edges for a, b in zip(line, line[1:], strict=False))
    straight = _vertices(NearestNodeSnap(ChordSubstrate()).snap(LINES, BLOCK))
    assert not all(frozenset({a, b}) in edges
                   for line in straight for a, b in zip(line, line[1:], strict=False))


def test_both_snaps_start_and_end_on_the_same_nodes() -> None:
    """Routing only fills in between the snapped vertices; it never moves them."""
    near = _vertices(NearestNodeSnap(ChordSubstrate()).snap(LINES, BLOCK))
    routed = _vertices(RoutedSnap(ChordSubstrate()).snap(LINES, BLOCK))
    for a, b in zip(near, routed, strict=True):
        assert (a[0], a[-1]) == (b[0], b[-1])
        assert set(a) <= set(b)
