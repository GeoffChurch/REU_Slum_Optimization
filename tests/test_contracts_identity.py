from typing import cast

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.buildings import AreaDiscs, SpacingDiscs
from reblock.contracts import Block, Proposal
from tests.block_fixtures import no_buildings

UTM = CRS.from_epsg(32643)


def _block(hash_: str | None, tier: object) -> Block:
    parcels = gpd.GeoDataFrame({"parcel_id": [0]},
                               geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=streets, source_content_hash=hash_,
                 building_geometries=no_buildings(UTM),
                 building_tier=tier)  # type: ignore[arg-type]


def test_block_identity_composes_hash_id_and_tier() -> None:
    assert (_block("deadbeef", SpacingDiscs).identity
            == ("deadbeef", "b", "reblock.buildings.SpacingDiscs"))


def test_block_identity_distinguishes_building_tiers() -> None:
    """The cache-key guard. Methods read `block.buildings`, so one block at two tiers gives two
    answers; if the tier were missing from the key, switching tiers would hit the cache and hand
    back the OTHER tier's result -- no error, plausible numbers. Watched failing: with the tier
    dropped from `Block.identity` these two keys are equal and this assertion fires."""
    assert _block("h", SpacingDiscs).identity != _block("h", AreaDiscs).identity


def test_block_identity_sees_through_hydra_partial() -> None:
    """Hydra's `_partial_: true` hands the source a functools.partial, not the class. It must key
    identically to the bare class, or config-built and hand-built blocks would never share a cache
    entry for the same tier."""
    from functools import partial
    assert _block("h", partial(SpacingDiscs)).identity == _block("h", SpacingDiscs).identity


def test_block_identity_is_none_when_uncacheable() -> None:
    assert _block(None, SpacingDiscs).identity is None


def test_block_rejects_an_empty_hash() -> None:
    """"" is not a content hash. Were it accepted, every such block would share the cache key
    ("", block_id, tier) and hand each other results; uncacheable is spelled None."""
    with pytest.raises(ValueError, match="source_content_hash"):
        _block("", SpacingDiscs)


def test_block_identity_is_hashable() -> None:
    hash((_block("h", SpacingDiscs).identity,))   # usable as a dict/joblib key


def test_proposal_identity_from_block_identity_and_proposal_id() -> None:
    p = Proposal(block_id="b", crs=UTM, roads=None, edges=None, proposal_id="topology_a2.0_s0",
                 method="topology", params={}, block_identity=("deadbeef", "b"))
    assert p.identity == (("deadbeef", "b"), "topology_a2.0_s0")


def test_proposal_identity_is_none_without_block_identity() -> None:
    assert Proposal(block_id="b", crs=UTM, roads=None, edges=None, proposal_id="peel",
                    method="peel", params={}, block_identity=None).identity is None
