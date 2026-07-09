from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, street_connectivity
from reblock.methods.dijkstra import DijkstraReblocker
from reblock.methods.mesh import MeshReblocker

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


def test_mesh_adds_loops_over_the_dijkstra_forest() -> None:
    block = _grid_block(5)
    tree = DijkstraReblocker().propose(block).roads
    mesh = MeshReblocker().propose(block).roads
    assert tree is not None and mesh is not None
    assert len(mesh) > len(tree)                       # closed >=1 loop
    conn = street_connectivity(block.streets, mesh, STREET_TOL)
    assert conn.connected_frac == 1.0


def test_mesh_is_deterministic() -> None:
    block = _grid_block(5)
    r1 = MeshReblocker().propose(block).roads
    r2 = MeshReblocker().propose(block).roads
    assert r1 is not None and r2 is not None
    assert [g.wkt for g in r1.geometry] == [g.wkt for g in r2.geometry]


def test_mesh_more_direct_than_the_tree() -> None:
    from reblock.budget import network_efficiency
    block = _grid_block(5)
    _, d_tree = network_efficiency(block, DijkstraReblocker().propose(block).roads)
    _, d_mesh = network_efficiency(block, MeshReblocker().propose(block).roads)
    assert d_mesh >= d_tree                            # crossings straighten inter-parcel travel


def test_mesh_identity_and_proposal_metadata() -> None:
    block = _grid_block(5)
    proposal = MeshReblocker().propose(block)
    assert MeshReblocker().identity == ("mesh",)
    assert proposal.proposal_id == "mesh" and proposal.method == "mesh"
    assert proposal.roads is not None and len(proposal.roads) > 0
