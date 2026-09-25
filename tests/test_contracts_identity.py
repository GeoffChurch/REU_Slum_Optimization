from dataclasses import replace
from typing import cast

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon

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


def _proposal(roads: gpd.GeoDataFrame | None, label: str = "m") -> Proposal:
    return Proposal(block_id="b", crs=UTM, roads=roads, edges=None, proposal_id=label,
                    method=label, params={})


def _roads(*xs: float) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"width_m": [7.0] * len(xs)},
                            geometry=[LineString([(x, 0), (x, 5)]) for x in xs], crs=UTM)


def test_a_proposals_identity_is_its_roads_not_its_label() -> None:
    """The derivations keyed on a proposal (`access_after`, `geometric_after`) read its roads, so
    its identity is their content: different roads never share a key, equal roads always do.

    FAULT INJECTION: an identity built from `proposal_id` fails the first assertion (one label,
    two road sets -- `cycle_native` on two substrates) and the second (two labels, one road set).
    """
    assert _proposal(_roads(0, 1), "same").identity != _proposal(_roads(0, 2), "same").identity
    assert _proposal(_roads(0, 1), "one").identity == _proposal(_roads(0, 1), "two").identity
    assert _proposal(None).identity == _proposal(None).identity


def test_a_truncated_proposal_is_its_own_key() -> None:
    """`replace(p, roads=prefix)` keeps the label; the key must still move, or a lens prefix is
    served the full network's cached access fields."""
    full = _proposal(_roads(0, 1, 2))
    prefix = replace(full, roads=_roads(0, 1))       # its first two roads, under its label
    assert prefix.identity != full.identity


def test_a_proposals_identity_ignores_the_frame_index() -> None:
    """No derivation reads the index, so a reindexed copy of the same roads is the same key."""
    roads = _roads(0, 1)
    shifted = roads.set_axis([10, 11])
    assert _proposal(roads).identity == _proposal(shifted).identity


def test_a_proposals_identity_sees_every_column() -> None:
    """Width is not geometry, and a wider road displaces more: it must move the key."""
    wide = _roads(0, 1).assign(width_m=9.0)
    assert _proposal(_roads(0, 1)).identity != _proposal(wide).identity
