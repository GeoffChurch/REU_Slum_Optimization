"""Build order, and adjacency measured in the frame's own units.

`_block_adjacency` runs `dwithin(STREET_TOL)` with `STREET_TOL = 0.5`. That is 0.5 metres in a
projected frame and ~55 km in lon/lat, so a builder handed a geographic frame used to treat every
block in a metro as adjacent to every other -- and returned a plausible-looking region assembled
from blocks kilometres apart, with nothing raised. See the design's §1.5.
"""
from __future__ import annotations

from typing import cast

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import Polygon
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.region import (
    ConvexHullRegionBuilder,
    DenseClusterRegionBuilder,
    IdentityRegionBuilder,
    RegionBuilder,
    ShapeStandardizingRegionBuilder,
    region_block,
)

# A 4x4 grid of 100 m blocks with a 2 m street gap, placed in UTM 33S near Cape Town. 100 m is
# small enough that 0.5 DEGREES (~55 km) swallows the whole grid, which is what makes the
# geographic-frame bug observable at all. The 2 m gap is deliberately ABOVE STREET_TOL (0.5 m):
# it keeps every cell mutually non-adjacent in the metric frame, so the CRS-invariance test below
# (the only one that reprojects) compares two ZERO-growth computations -- exact, with no
# reprojection-roundoff tie-break sensitivity near the adjacency threshold.
CELL, GAP, ORIGIN_X, ORIGIN_Y = 100.0, 2.0, 260000.0, 6240000.0


def _grid(counts: dict[tuple[int, int], float] | None = None,
         gap: float = GAP) -> gpd.GeoDataFrame:
    ids, polys, ns = [], [], []
    for r in range(4):
        for c in range(4):
            x0 = ORIGIN_X + c * (CELL + gap)
            y0 = ORIGIN_Y + r * (CELL + gap)
            ids.append(f"{r}_{c}")
            polys.append(Polygon([(x0, y0), (x0 + CELL, y0), (x0 + CELL, y0 + CELL),
                                  (x0, y0 + CELL)]))
            ns.append((counts or {}).get((r, c), 10.0))
    return gpd.GeoDataFrame({"block_id": ids, "building_count": ns},
                            geometry=polys, crs="EPSG:32734")


@pytest.mark.parametrize("builder", [
    DenseClusterRegionBuilder(max_buildings=40),
    ShapeStandardizingRegionBuilder(max_buildings=40),
    IdentityRegionBuilder(),
    ConvexHullRegionBuilder(),
], ids=["dense_cluster", "shape_standardizing", "identity", "convex_hull"])
def test_geographic_frame_grows_the_same_region_as_its_projected_twin(
    builder: RegionBuilder,
) -> None:
    """A frame's CRS must not change which blocks a builder picks.

    This is the whole of the §1.5 bug: `dwithin(0.5)` in lon/lat is ~55 km, so before the fix
    `dense_cluster` on the geographic twin pulled in blocks by score alone, ignoring adjacency.
    """
    utm = _grid()
    geo = utm.to_crs("EPSG:4326")
    assert builder.build(utm, [["1_1"]]) == builder.build(geo, [["1_1"]])


def test_dense_cluster_returns_accretion_order_not_sorted_order() -> None:
    """The order blocks were ADDED in, which is what RegionGrow teaches and what pins its
    TypeScript to production. A sorted result throws it away.

    The grid is rigged so accretion order and sorted order disagree: the seed is `1_1`, and its
    neighbours are weighted so the greedy walks DOWN-then-LEFT while `sorted()` would put `0_1`
    before `1_0`. If this test ever passes against a `sorted()` implementation, the rigging has
    stopped working and the test guards nothing -- check by reverting the builder.

    `gap=0.3` (< STREET_TOL): the module default GAP (2 m) is deliberately non-adjacent (see the
    module docstring), so growth needs a smaller gap here to be reachable at all.
    """
    # depth proxy is sqrt(n*A)/P; A and P are equal for every cell, so a higher count wins.
    grid = _grid({(1, 0): 30.0, (0, 1): 20.0, (2, 1): 15.0}, gap=0.3)
    got = DenseClusterRegionBuilder(max_buildings=70).build(grid, [["1_1"]])[0]

    assert got[0] == "1_1", "the seed comes first"
    assert got == ["1_1", "1_0", "0_1", "2_1"], got
    assert got != sorted(got), "accretion order must not coincide with sorted order here"


def test_shape_standardizing_returns_accretion_order() -> None:
    """Same contract, the other growing builder."""
    grid = _grid(gap=0.3)
    got = ShapeStandardizingRegionBuilder(max_buildings=40).build(grid, [["1_1"]])[0]
    assert got[0] == "1_1", "the seed comes first"
    assert len(got) == len(set(got)), "no block appears twice"


def test_non_growing_builders_return_sorted_order() -> None:
    """`identity` and `convex_hull` have no accretion to report, so sorted IS their build order --
    stated as a test so the contract is one sentence for all four builders."""
    grid = _grid()
    assert IdentityRegionBuilder().build(grid, [["1_1", "0_0"]]) == [["0_0", "1_1"]]
    hull = ConvexHullRegionBuilder().build(grid, [["0_0", "1_1"]])[0]
    assert hull == sorted(hull)


def _adjacent_blocks() -> tuple[Block, Block]:
    """Two touch-adjacent 3x3-parcel Blocks, side by side, each with a real (non-"") content
    hash. `tests/scoring_fixtures.py` has no `two_adjacent_blocks()` helper (checked), so this
    copies the CONSTRUCTION `tests/test_region.py::_grid_block` uses to build `region_block`
    fixtures elsewhere -- not its assertions, which belong to that module's own tests.

    The two blocks' parcels occupy disjoint spatial extents (x in [0, 3) vs. [3, 6)), so
    concatenating them in a different order visibly permutes the parcel GEOMETRY sequence unless
    something re-canonicalizes it -- which is exactly what the test below (
    `test_region_block_is_independent_of_member_order`) checks for. Non-empty
    `source_content_hash` makes `region_block`'s hash comparison exercise the real
    sorted-hash-join branch instead of the trivial "" == "" one.
    """
    def _block(x0: int, block_id: str, source_content_hash: str) -> Block:
        polys = [Polygon([(x0 + i, j), (x0 + i + 1, j), (x0 + i + 1, j + 1), (x0 + i, j + 1)])
                for i in range(3) for j in range(3)]
        parcels = gpd.GeoDataFrame({"parcel_id": list(range(9))}, geometry=polys,
                                   crs="EPSG:32734")
        boundary = cast(Polygon, unary_union(polys))
        streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs="EPSG:32734")
        return Block(block_id=block_id, crs=CRS.from_epsg(32734), boundary=boundary,
                    parcels=parcels, streets=streets, source_content_hash=source_content_hash)

    return _block(0, "a", "srcA"), _block(3, "b", "srcB")


def test_region_block_is_independent_of_member_order() -> None:
    """`region_block` must give the same parcels whatever order its members arrive in.

    `_shared_parts` does `pd.concat([b.parcels for b in blocks])` then
    `parcels["parcel_id"] = range(len(parcels))`, so member order RENUMBERS every parcel -- while
    `block_id` and `source_content_hash` are both built from `sorted(...)` and do NOT change. A
    cached derivation keyed on that unchanged hash would then be reused against differently
    numbered parcels. Task 1 makes builders return accretion order, which is exactly when unsorted
    members become reachable, so this is the guard that makes that change safe.
    """
    a, b = _adjacent_blocks()
    forward = region_block([a, b])
    reverse = region_block([b, a])

    assert forward.block_id == reverse.block_id
    assert forward.source_content_hash == reverse.source_content_hash
    assert list(forward.parcels["parcel_id"]) == list(reverse.parcels["parcel_id"])
    assert forward.parcels.geometry.equals(reverse.parcels.geometry), (
        "parcel geometry must be identical, not merely equivalent as a set")
