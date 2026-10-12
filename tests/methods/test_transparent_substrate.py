"""`reblock.transparency.TransparentSubstrate`: the route half of transparency, a multiplier on the
base's edge weights.

Fault-injected where a docstring says so: the fault was applied, the test seen to fail, the code
restored.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace

import geopandas as gpd
import numpy as np
import pytest
from numpy.typing import NDArray
from pyproj import CRS
from shapely.geometry import LineString, box

from reblock.buildings import ANCHOR_COL, Footprints
from reblock.contracts import Block
from reblock.derive_graph import closure_hash
from reblock.methods.desire_substrate import DesireWarpedSubstrate
from reblock.methods.substrates import ChordSubstrate, PrebuiltSubstrate, RoutingGraph
from reblock.transparency import Opaque, TransparentSubstrate, price_tag
from tests.methods.test_desire_substrate import _Lines

UTM = CRS.from_epsg(32734)

# name: (outline, factor). Buildings 3 and 4 overlap on x 32-34, 5 and 6 on x 42-44, 7 and 8 on
# x 52-54: footprints overlap in the data (hundreds of pairs on a large block).
BUILDINGS: dict[str, tuple[tuple[float, float, float, float], float]] = {
    "b0": ((0.0, 0.0, 4.0, 4.0), 0.5),
    "b1": ((10.0, 0.0, 14.0, 4.0), 0.2),
    "b2": ((20.0, 0.0, 24.0, 4.0), 1.0),
    "b3": ((30.0, 0.0, 34.0, 4.0), 0.5),
    "b4": ((32.0, 0.0, 36.0, 4.0), 1.0),
    "b5": ((40.0, 0.0, 44.0, 4.0), 1.0),
    "b6": ((42.0, 0.0, 46.0, 4.0), 1.0),
    "b7": ((50.0, 0.0, 54.0, 4.0), 0.5),
    "b8": ((52.0, 0.0, 56.0, 4.0), 0.5),
}

# (from, to, the multiplier the docstring's rule gives)
EDGES: list[tuple[tuple[float, float], tuple[float, float], float]] = [
    ((0.0, 10.0), (4.0, 10.0), 1.0),             # in the open
    ((1.0, 2.0), (3.0, 2.0), 0.5),               # wholly inside b0
    ((12.0, 2.0), (16.0, 2.0), (1.0 + 0.2) / 2),  # half inside b1
    ((21.0, 2.0), (23.0, 2.0), 1.0),             # wholly inside an opaque building
    ((2.0, 2.0), (12.0, 2.0), (2 * 0.5 + 6 * 1.0 + 2 * 0.2) / 10),   # b0, the open, b1
    ((32.5, 2.0), (33.5, 2.0), 0.5),             # in b3 (0.5) and b4 (1): the smaller
    ((42.5, 2.0), (43.5, 2.0), 1.0),             # in two opaque buildings: not dearer than open
    ((52.5, 2.0), (53.5, 2.0), 0.5),             # in two at 0.5: 0.5, not 0.25 or 0
]


@dataclass(frozen=True)
class ByAnchor:
    """Factors keyed by each building's anchor, so they follow the BUILDING whatever its row."""

    by_anchor: tuple[tuple[tuple[float, float], float], ...]

    @property
    def identity(self) -> Hashable:
        return ("by-anchor", self.by_anchor)

    @property
    def tag(self) -> str:
        return price_tag(self.identity)

    def factors(self, block: Block) -> NDArray[np.float64]:
        lookup = dict(self.by_anchor)
        return np.array([lookup[(float(x), float(y))] for x, y in block.buildings.xy])


def _anchor(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)


FACTORS = ByAnchor(tuple((_anchor(b), f) for b, f in BUILDINGS.values()))


def _block(order: list[str]) -> Block:
    """The buildings in the row order `order`."""
    outlines = [box(*BUILDINGS[name][0]) for name in order]
    return Block(
        block_id="transparent", crs=UTM, boundary=box(-5.0, -5.0, 60.0, 15.0),
        parcels=gpd.GeoDataFrame({"parcel_id": ["p"]}, geometry=[box(-5.0, -5.0, 60.0, 15.0)],
                                 crs=UTM),
        streets=gpd.GeoDataFrame(geometry=[LineString([(-5.0, -5.0), (60.0, -5.0)])], crs=UTM),
        building_geometries=gpd.GeoDataFrame({ANCHOR_COL: [o.centroid for o in outlines]},
                                             geometry=outlines, crs=UTM),
        source_content_hash=None, building_tier=Footprints)


def _graph() -> RoutingGraph:
    """EDGES as a symmetric COO, each weighted three times its length -- a base whose weights are
    not lengths, as a warped base's are not."""
    pts = np.array([p for a, b, _ in EDGES for p in (a, b)], dtype=np.float64)
    rows = np.array([i for k in range(len(EDGES)) for i in (2 * k, 2 * k + 1)], dtype=np.int64)
    cols = np.array([i for k in range(len(EDGES)) for i in (2 * k + 1, 2 * k)], dtype=np.int64)
    edist = 3.0 * np.hypot(*(pts[rows] - pts[cols]).T)
    return RoutingGraph(pts, rows, cols, edist, net_tol=0.5)


def _multipliers(block: Block) -> NDArray[np.float64]:
    base = _graph()
    built = TransparentSubstrate(base=PrebuiltSubstrate(base), transparency=FACTORS).build(block)
    np.testing.assert_array_equal(built.pts, base.pts)
    np.testing.assert_array_equal(built.rows, base.rows)
    np.testing.assert_array_equal(built.cols, base.cols)
    return np.asarray(built.edist / base.edist, dtype=np.float64)


@pytest.mark.parametrize("order", [list(BUILDINGS), list(reversed(BUILDINGS)),
                                   ["b4", "b0", "b7", "b2", "b8", "b1", "b6", "b3", "b5"]],
                         ids=["rows", "reversed", "shuffled"])
def test_each_edge_is_scaled_by_the_factors_along_it(order: list[str]) -> None:
    """Open ground keeps its weight, a building of factor f scales by f, half inside by (1 + f) / 2,
    and an overlap takes the smaller factor -- with the factors following each BUILDING in
    whatever row order the block holds them.

    FAULT INJECTION: an off-by-one in the building index -- `factors = np.roll(factors_of(...),
    1)` in `TransparentSubstrate.build` -- fails this in every order; so does summing the
    overlapping buildings instead of taking the smaller factor (the three overlap edges)."""
    want = np.repeat([m for _, _, m in EDGES], 2)
    np.testing.assert_allclose(_multipliers(_block(order)), want, rtol=1e-12, atol=0.0)


def test_opaque_leaves_every_weight_exactly_as_it_was() -> None:
    block = _block(list(BUILDINGS))
    base = ChordSubstrate().build(block)
    built = TransparentSubstrate(base=ChordSubstrate(), transparency=Opaque()).build(block)
    np.testing.assert_array_equal(built.edist, base.edist)


def test_it_composes_with_a_warped_base() -> None:
    """The multiplier is the edges' own: the same over a desire-warped base as over the plain
    one, so the warped weights are scaled exactly as the plain ones are."""
    block = _block(list(BUILDINGS))
    warped = DesireWarpedSubstrate(base=ChordSubstrate(), desire_source=_Lines(2.0),
                                   buffer_m=1.0, eps=0.1, gamma=1.0)
    plain = (TransparentSubstrate(base=ChordSubstrate(), transparency=FACTORS).build(block).edist
             / ChordSubstrate().build(block).edist)
    over_warped = (TransparentSubstrate(base=warped, transparency=FACTORS).build(block).edist
                   / warped.build(block).edist)
    np.testing.assert_allclose(over_warped, plain, rtol=1e-12)
    assert (plain < 1.0).any(), "no edge crosses a transparent building: the check is vacuous"


def test_identity_carries_its_code_its_base_and_its_factors() -> None:
    sub = TransparentSubstrate(base=ChordSubstrate(), transparency=FACTORS)
    ident = sub.identity
    assert isinstance(ident, tuple)
    assert ident[0] == closure_hash("reblock.transparency")
    other = ByAnchor(((_anchor(BUILDINGS["b0"][0]), 0.25), *FACTORS.by_anchor[1:]))
    assert replace(sub, transparency=other).identity != ident
    assert replace(sub, transparency=Opaque()).identity != ident
    assert sub.tag.startswith("chord_diag+transparent:")
    assert replace(sub, transparency=other).tag != sub.tag


def test_factors_with_no_address_make_it_uncacheable() -> None:
    @dataclass(frozen=True)
    class Live:
        identity = None
        tag = price_tag(None)

        def factors(self, block: Block) -> NDArray[np.float64]:
            return np.ones(len(block.buildings))

    assert TransparentSubstrate(base=ChordSubstrate(), transparency=Live()).identity is None


def test_a_zero_factor_is_refused() -> None:
    zero = ByAnchor(((_anchor(BUILDINGS["b0"][0]), 0.0), *FACTORS.by_anchor[1:]))
    with pytest.raises(ValueError, match=r"must lie in \(0, 1\]"):
        TransparentSubstrate(base=PrebuiltSubstrate(_graph()), transparency=zero).build(
            _block(list(BUILDINGS)))
