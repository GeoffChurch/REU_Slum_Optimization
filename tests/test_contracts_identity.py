from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block, Proposal

UTM = CRS.from_epsg(32643)


def _block(hash_: str) -> Block:
    parcels = gpd.GeoDataFrame({"parcel_id": [0]},
                               geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=streets, source_content_hash=hash_)


def test_block_identity_composes_hash_and_id() -> None:
    assert _block("deadbeef").identity == ("deadbeef", "b")


def test_block_identity_is_none_when_hash_empty() -> None:
    assert _block("").identity is None            # default => uncacheable


def test_block_identity_is_hashable() -> None:
    hash((_block("h").identity,))                 # usable as a dict/joblib key


def test_proposal_identity_from_block_identity_and_proposal_id() -> None:
    p = Proposal(block_id="b", crs=UTM, block_identity=("deadbeef", "b"),
                 proposal_id="topology_a2.0_s0")
    assert p.identity == (("deadbeef", "b"), "topology_a2.0_s0")


def test_proposal_identity_is_none_without_block_identity() -> None:
    assert Proposal(block_id="b", crs=UTM, proposal_id="peel").identity is None
