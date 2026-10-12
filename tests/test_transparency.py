"""The transparency Strategy (`reblock.transparency`): factors checked where a method reads them,
and `PricedOverlap`, the priced displacement beside `IncrementalOverlap`.

Fault-injected where a docstring says so: the fault was applied, the test seen to fail, the code
restored.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pytest
from numpy.typing import NDArray
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon, box

from reblock.buildings import ANCHOR_COL, Discs, Footprints, IncrementalOverlap
from reblock.contracts import Block
from reblock.derive_graph import config_identity
from reblock.transparency import Opaque, PricedOverlap, Transparency, factors_of, price_tag

UTM = CRS.from_epsg(32734)


@dataclass(frozen=True)
class Given:
    """Whatever factors it is handed, for any block."""

    values: tuple[float, ...]

    @property
    def identity(self) -> Hashable:
        return ("given", self.values)

    @property
    def tag(self) -> str:
        return price_tag(self.identity)

    def factors(self, block: Block) -> NDArray[np.float64]:
        del block
        return np.asarray(self.values, dtype=np.float64)


def _block(n: int = 3) -> Block:
    """`n` 4 x 4 m footprints in a row of 10 m parcels, street along the bottom."""
    parcels = [box(10.0 * i, 0.0, 10.0 * (i + 1), 10.0) for i in range(n)]
    outlines = [box(10.0 * i + 3.0, 3.0, 10.0 * i + 7.0, 7.0) for i in range(n)]
    return Block(
        block_id="row", crs=UTM, boundary=box(0.0, 0.0, 10.0 * n, 10.0),
        parcels=gpd.GeoDataFrame({"parcel_id": [str(i) for i in range(n)]}, geometry=parcels,
                                 crs=UTM),
        streets=gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (10.0 * n, 0.0)])], crs=UTM),
        building_geometries=gpd.GeoDataFrame({ANCHOR_COL: [o.centroid for o in outlines]},
                                             geometry=outlines, crs=UTM),
        source_content_hash=None, building_tier=Footprints)


def test_opaque_is_every_factor_one_and_satisfies_the_protocol() -> None:
    block = _block()
    assert isinstance(Opaque(), Transparency)
    np.testing.assert_array_equal(factors_of(Opaque(), block), np.ones(3))
    assert Opaque().identity == config_identity(Opaque()) is not None
    assert Opaque().tag == ""                   # it changes no road set, so no label


def test_a_price_tag_follows_the_identity() -> None:
    a, b = Given((1.0, 0.5, 1.0)), Given((1.0, 0.25, 1.0))
    assert a.tag.startswith(":price-") and a.tag != b.tag
    assert a.tag == Given((1.0, 0.5, 1.0)).tag


@pytest.mark.parametrize("values", [(1.0, 0.0, 1.0), (1.0, 1.5, 1.0), (1.0, float("nan"), 1.0),
                                    (-0.5, 1.0, 1.0)], ids=["zero", "above-one", "nan", "negative"])
def test_factors_outside_zero_to_one_are_refused(values: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\]"):
        factors_of(Given(values), _block())


def test_factors_must_be_one_per_building() -> None:
    with pytest.raises(ValueError, match="for 3 buildings"):
        factors_of(Given((1.0, 0.5)), _block())


def _corridor(x0: float, x1: float) -> Polygon:
    """A 7 m corridor along y = 5 from x0 to x1, flat-ended so the shares it takes are exact."""
    return LineString([(x0, 5.0), (x1, 5.0)]).buffer(3.5, cap_style="flat")


def test_opaque_prices_exactly_as_overlap_counts() -> None:
    """At factor 1 the priced increment IS `IncrementalOverlap.delta`, float for float -- the
    reason `Opaque` methods choose exactly as they did -- and the priced total is the total."""
    block = _block()
    priced = PricedOverlap(block.buildings, factors=np.ones(3))
    plain = IncrementalOverlap(block.buildings)
    for piece in (_corridor(0.0, 6.0), _corridor(4.0, 16.0), _corridor(14.0, 29.0)):
        inc = priced.increment(piece)
        assert inc.true == plain.delta(piece) and inc.priced == inc.true
        priced.add(piece)
        plain.add(piece)
        assert priced.priced_total() == plain.total() == priced.total()


def test_a_building_is_priced_at_its_own_factor() -> None:
    """Each building's added share counts times ITS factor: half of building 0 at 0.5, all of
    building 1 at 0.25, building 2 untouched."""
    block = _block()
    f = np.array([0.5, 0.25, 1.0])
    priced = PricedOverlap(block.buildings, factors=f)
    piece = _corridor(5.0, 17.0)          # x 5 .. 17: half of building 0 (x 3-7), all of 1
    inc = priced.increment(piece)
    assert inc.true == pytest.approx(0.5 + 1.0)
    assert inc.priced == pytest.approx(0.5 * 0.5 + 1.0 * 0.25)
    priced.add(piece)
    assert priced.priced_total() == pytest.approx(0.5 * 0.5 + 1.0 * 0.25)
    # Only the share a piece ADDS is priced: covering building 0's other half adds 0.5 * 0.5.
    assert priced.increment(_corridor(0.0, 6.0)).priced == pytest.approx(0.5 * 0.5)


def test_a_building_with_no_extent_is_priced_at_its_factor() -> None:
    """A zero-area building is displaced whole when the corridor covers its point."""
    pts = gpd.GeoDataFrame(geometry=[Point(5.0, 5.0), Point(25.0, 5.0)], crs=UTM)
    priced = PricedOverlap(Discs(pts, np.array([0.0, 2.0])), factors=np.array([0.4, 1.0]))
    inc = priced.increment(_corridor(0.0, 10.0))
    assert inc.true == 1.0 and inc.priced == pytest.approx(0.4)


def test_factors_must_match_the_buildings() -> None:
    with pytest.raises(ValueError, match="factors for 3 buildings"):
        PricedOverlap(_block().buildings, factors=np.ones(2))
