"""Guards for the shape-standardizing RegionBuilder.

The one property that matters, and the one the substitute builder lacked: it must score the SHAPE
OF THE UNION as it grows, not the candidate block in isolation. `DenseClusterRegionBuilder` ranks
each frontier block by `sqrt(n*A)/P` computed on that block alone, which is why its regions came out
as tendrils whose outline is a growth artifact -- the confound the Phase 3 donor-material test
cannot tolerate.
"""
from __future__ import annotations

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.region import (
    DenseClusterRegionBuilder,
    Isoperimetric,
    Rectangularity,
    ShapeStandardizingRegionBuilder,
    Squareness,
)

UTM = CRS.from_epsg(32734)
CELL = 100.0


def _grid(k: int = 5, counts: dict[tuple[int, int], float] | None = None) -> gpd.GeoDataFrame:
    """A k x k grid of square blocks, ids "r_c". `counts` overrides building_count per cell."""
    rows = []
    for r in range(k):
        for c in range(k):
            x0, y0 = c * CELL, r * CELL
            rows.append({
                "block_id": f"{r}_{c}",
                "building_count": (counts or {}).get((r, c), 10.0),
                "geometry": Polygon([(x0, y0), (x0 + CELL, y0),
                                     (x0 + CELL, y0 + CELL), (x0, y0 + CELL)]),
            })
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=UTM)


def _outline(ids: list[str], geoms: gpd.GeoDataFrame) -> float:
    """Isoperimetric quotient of the union those ids form -- 1.0 is a circle."""
    from shapely.ops import unary_union

    sel = geoms[geoms["block_id"].isin(ids)]
    u = unary_union(list(sel.geometry))
    return Isoperimetric().score(u)


def test_objectives_peak_on_their_own_ideal_shape():
    # Each objective must actually measure what it claims, or the accretion optimises nothing.
    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    strip = Polygon([(0, 0), (100, 0), (100, 1), (0, 1)])
    assert Isoperimetric().score(square) > Isoperimetric().score(strip)
    # rectangularity is blind to elongation: BOTH are perfect rectangles
    assert Rectangularity().score(square) == pytest.approx(1.0)
    assert Rectangularity().score(strip) == pytest.approx(1.0)
    # squareness is not: it multiplies in the aspect ratio
    assert Squareness().score(square) == pytest.approx(1.0)
    assert Squareness().score(strip) < 0.05


def test_it_scores_the_UNION_not_the_candidate_block():
    """The distinguishing property. Every block here is an identical square, so a builder that
    scores candidates in isolation cannot tell them apart and falls back to its tie-break, while one
    scoring the union must plump the region toward a square.

    FAULT INJECTION: score `shape_geoms[j]` instead of `union.union(shape_geoms[j])` and the chosen
    region degrades to the tie-break order, failing the compactness assertion below.
    """
    geoms = _grid(5)
    got = ShapeStandardizingRegionBuilder(max_buildings=40).build(geoms, [["2_2"]])[0]
    assert len(got) == 4, got
    # 4 identical squares: the compact 2x2 block scores 0.785, an L or a 1x4 strip far less
    assert _outline(got, geoms) > 0.7, f"grew a non-compact region: {got}"


def test_it_beats_dense_cluster_on_outline_for_the_same_budget():
    # The reason this builder exists. building_count is uniform, so dense-cluster's sqrt(n*A)/P is
    # tied everywhere and its tie-break walks block_id order -- exactly the uncontrolled outline the
    # Phase 3 test cannot use.
    geoms = _grid(6)
    seed, budget = [["3_3"]], 60
    shaped = ShapeStandardizingRegionBuilder(max_buildings=budget).build(geoms, seed)[0]
    dense = DenseClusterRegionBuilder(max_buildings=budget).build(geoms, seed)[0]
    assert len(shaped) == len(dense), (shaped, dense)
    assert _outline(shaped, geoms) > _outline(dense, geoms), (
        f"shape-standardizing {_outline(shaped, geoms):.3f} did not beat dense-cluster "
        f"{_outline(dense, geoms):.3f}")


def test_isoperimetric_TIES_ITSELF_INTO_a_bad_shape_which_is_why_it_is_not_the_default():
    """The reason the "obvious first guess" is not the default, pinned as a measurement.

    Polyomino perimeters tie constantly -- a 1x3 strip and an L-tromino both have area 3 and
    perimeter 8 -- so on grid-like fabric the greedy cannot discriminate, falls back to the
    `block_id` tie-break, and lands somewhere the compact option is no longer reachable from. An
    objective that ties everywhere standardizes nothing, which is the growth-artifact outline this
    builder exists to remove.
    """
    geoms = _grid(5)
    seed, budget = [["2_2"]], 40
    iso = ShapeStandardizingRegionBuilder(objective=Isoperimetric(), max_buildings=budget).build(
        geoms, seed)[0]
    rect = ShapeStandardizingRegionBuilder(objective=Rectangularity(), max_buildings=budget).build(
        geoms, seed)[0]
    sq = ShapeStandardizingRegionBuilder(objective=Squareness(), max_buildings=budget).build(
        geoms, seed)[0]

    # squareness finds the 2x2; the other two do not -- and isoperimetric misses it on its OWN
    # metric, scoring 0.503 where the 2x2 it declined to build scores 0.785
    assert _outline(sq, geoms) == pytest.approx(0.785, abs=0.01), sq
    assert _outline(iso, geoms) < 0.6, iso
    assert _outline(rect, geoms) < 0.6, rect


def test_the_seed_is_always_kept_even_alone_over_budget():
    geoms = _grid(3)
    got = ShapeStandardizingRegionBuilder(max_buildings=1).build(geoms, [["1_1"]])[0]
    assert got == ["1_1"]


def test_growth_is_contiguous_and_deterministic():
    geoms = _grid(5)
    runs = [ShapeStandardizingRegionBuilder(max_buildings=50).build(geoms, [["0_0"]])[0]
            for _ in range(3)]
    assert runs[0] == runs[1] == runs[2], runs
    from shapely.ops import unary_union
    sel = geoms[geoms["block_id"].isin(runs[0])]
    assert unary_union(list(sel.geometry)).geom_type == "Polygon"   # one piece, no holes/islands


def test_it_works_without_building_counts():
    # A non-kblock source: the budget becomes a block count, and the builder must still run.
    geoms = _grid(4).drop(columns=["building_count"])
    got = ShapeStandardizingRegionBuilder(max_buildings=4).build(geoms, [["1_1"]])[0]
    assert len(got) == 4 and "1_1" in got


def test_an_unknown_seed_id_fails_with_a_named_error():
    geoms = _grid(3)
    with pytest.raises(ValueError, match="nope"):
        ShapeStandardizingRegionBuilder().build(geoms, [["nope"]])
