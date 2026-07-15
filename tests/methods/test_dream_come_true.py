# tests/methods/test_dream_come_true.py
from collections.abc import Hashable

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block, Proposal
from reblock.methods.dream_come_true import DreamComeTrueReblocker

UTM = CRS.from_epsg(32734)
Bbox = tuple[float, float, float, float]


def _block() -> Block:
    # A 100 m square block; its street is the south edge (y=0).
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[boundary], crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _cacheable_block() -> Block:
    # Same block, but with a non-empty source_content_hash so block.identity is a real tuple
    # (not None) -- lets a test distinguish the cacheable vs. live eval-cache branch.
    boundary = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[boundary], crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=UTM)
    return Block(block_id="b", crs=UTM, boundary=boundary, parcels=parcels, streets=streets,
                 source_content_hash="h")


class _StubSource:
    """Structurally a DesireLineSource: returns fixed lines in the block CRS, ignoring the bbox --
    exercises the method's clip/dedupe without any network."""

    def __init__(self, lines: list[LineString], ident: Hashable = ("stub",)) -> None:
        self._lines = lines
        self._ident = ident

    @property
    def identity(self) -> Hashable:
        return self._ident

    def desire_lines(self, bbox_wgs84: Bbox, crs: CRS) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(geometry=self._lines, crs=UTM)


def test_propose_keeps_interior_paths_and_drops_those_on_the_street() -> None:
    interior = LineString([(50, 20), (50, 80)])          # a vertical interior path
    on_street = LineString([(10, 0), (90, 0)])           # runs along the south-edge street
    outside = LineString([(150, 150), (160, 160)])       # outside the boundary
    method = DreamComeTrueReblocker(source=_StubSource([interior, on_street, outside]))
    prop = method.propose(_block())
    assert isinstance(prop, Proposal) and prop.roads is not None
    lengths = sorted(round(g.length) for g in prop.roads.geometry)
    assert lengths == [60]                               # only the interior path survives


def test_propose_empty_coverage_returns_empty_roads_without_crashing() -> None:
    method = DreamComeTrueReblocker(source=_StubSource([]))
    prop = method.propose(_block())
    assert prop.roads is not None and prop.roads.empty


def test_identity_propagates_none_from_uncacheable_source() -> None:
    # A live (snapshot-less) source reports identity None; the method must propagate it.
    method = DreamComeTrueReblocker(source=_StubSource([], ident=None))
    assert method.identity is None


def test_cacheable_source_encodes_config_into_proposal_identity() -> None:
    # A stable-snapshot source (real identity) on a block with a real identity yields a cacheable
    # Proposal; two methods differing only in corridor_m must NOT collide in the eval cache.
    block = _cacheable_block()
    assert block.identity is not None
    src_a = _StubSource([], ident=("osm", ("path",), "abc"))
    src_b = _StubSource([], ident=("osm", ("path",), "abc"))
    prop_a = DreamComeTrueReblocker(source=src_a, corridor_m=3.0).propose(block)
    prop_b = DreamComeTrueReblocker(source=src_b, corridor_m=5.0).propose(block)
    assert prop_a.identity is not None
    assert prop_a.proposal_id != prop_b.proposal_id      # config encoded -> no eval-cache collision
    assert prop_a.identity != prop_b.identity


def test_live_source_makes_proposal_uncacheable_even_on_a_real_block() -> None:
    # A live source's roads can drift, so its eval must bypass the cache regardless of the block:
    # Proposal.identity must be None even when block.identity is a real tuple.
    block = _cacheable_block()
    assert block.identity is not None
    method = DreamComeTrueReblocker(source=_StubSource([], ident=None))
    prop = method.propose(block)
    assert prop.identity is None
