from typing import cast

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block
from reblock.methods.substrates import (
    GridSubstrate,
    PrebuiltSubstrate,
    RoutingGraph,
    Substrate,
    _pack_edges,
)

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="grid", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_pack_edges_is_symmetric_and_sorted() -> None:
    pts = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    pts2, rows, cols, edist = _pack_edges(pts, {frozenset((0, 1)), frozenset((0, 2))})
    assert len(rows) == len(cols) == len(edist) == 4          # 2 undirected -> 4 directed
    undirected = {frozenset((int(a), int(b))) for a, b in zip(rows, cols, strict=True)}
    assert undirected == {frozenset((0, 1)), frozenset((0, 2))}
    # lengths correct (3-4-5 triangle and the unit edge)
    assert set(np.round(edist, 6)) == {5.0, 1.0}


def test_grid_substrate_builds_valid_routing_graph() -> None:
    graph = GridSubstrate(res=1.0).build(_grid_block(4))
    assert isinstance(graph, RoutingGraph)
    assert graph.pts.shape[1] == 2 and len(graph.pts) > 0
    assert len(graph.rows) == len(graph.cols) == len(graph.edist)
    assert graph.net_tol == pytest.approx(1.5)                 # res * 1.5
    assert GridSubstrate(res=1.0).identity == ("grid", 1.0)
    assert GridSubstrate(res=1.0).tag == "grid"


def test_prebuilt_substrate_round_trips() -> None:
    g = RoutingGraph(pts=np.array([[0.0, 0.0], [1.0, 0.0]]),
                     rows=np.array([0, 1]), cols=np.array([1, 0]),
                     edist=np.array([1.0, 1.0]), net_tol=0.5)
    sub: Substrate = PrebuiltSubstrate(g)
    assert sub.build(_grid_block(2)) is g
    assert sub.tag == "prebuilt"
