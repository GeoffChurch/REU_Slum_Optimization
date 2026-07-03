from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block
from reblock.methods.topology import TopologyMethod

UTM = CRS.from_epsg(32643)


def _grid(n: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_proposes_roads_for_interior_parcel() -> None:
    proposal = TopologyMethod().propose(_grid(3))
    assert proposal.method == "topology" and proposal.crs == UTM
    assert proposal.roads is not None and len(proposal.roads) >= 1
    assert proposal.roads.geometry.length.sum() > 0
    assert proposal.edges is not None
    assert set(proposal.edges.columns) >= {"road", "interior", "barrier"}
    assert proposal.edges.crs == UTM
    assert len(proposal.edges) > 0


def test_propose_is_deterministic_across_runs() -> None:
    block = _grid(3)
    a = TopologyMethod(seed=0).propose(block)
    b = TopologyMethod(seed=0).propose(block)
    assert a.roads is not None and b.roads is not None
    assert sorted(g.wkt for g in a.roads.geometry) == sorted(g.wkt for g in b.roads.geometry)


def test_all_interior_parcels_connected() -> None:
    import random

    from topology import build_all_roads

    from reblock.derive.parcel_graph import to_parcel_graph
    ppg = to_parcel_graph(_grid(3))
    ppg.graph.define_roads()  # type: ignore[no-untyped-call]
    ppg.graph.define_interior_parcels()  # type: ignore[no-untyped-call]
    random.seed(0)
    build_all_roads(ppg.graph, alpha=2.0, vquiet=True)  # type: ignore[no-untyped-call]
    ppg.graph.define_interior_parcels()  # type: ignore[no-untyped-call]
    assert len(ppg.graph.interior_parcels) == 0
