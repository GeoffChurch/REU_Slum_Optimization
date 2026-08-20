"""The identity the TypeScript widget implements, pinned against shapely.

    dist(p, U_i buffer(L_i, w_i/2)) == min_i max(0, dist(p, L_i) - w_i/2)

A buffer IS the set of points within w/2 of the line, and distance to a union is the minimum over
its parts, so this is exact. Shapely's buffer is an inscribed POLYGON, slightly smaller than the
true round buffer, so it reports slightly larger distances -- which is why the closed form comes out
higher, and why the gap closes as `quad_segs` rises.

No block load here on purpose: see the plan's global constraint on block-loading tests. Task 3's
one slow test carries the real-data check across all eight methods.
"""
from typing import cast

import numpy as np
import pytest
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

from reblock.budget import displacement_from_distance
from reblock.contracts import Block
from scripts._default_road import closed_form_distance, default_roads, segments

UTM = CRS.from_epsg(32643)
# Every width at or above min_road_width_m (permeability.py:125); a narrower road is one this
# pipeline raises on, so it has no business in a fixture even for a pure-arithmetic test.
ROADS = [(LineString([(0, 0), (100, 20), (140, 90)]), 7.0),
         (LineString([(20, 80), (120, 10)]), 9.0),
         (LineString([(60, 60), (60, 140)]), 14.0)]


def _frame() -> tuple[GeoDataFrame, np.ndarray, np.ndarray, np.ndarray]:
    roads = GeoDataFrame({"width_m": [w for _, w in ROADS]},
                         geometry=[g for g, _ in ROADS], crs=UTM)
    rng = np.random.default_rng(0)
    pts = rng.uniform(-20, 160, size=(4000, 2))
    radii = rng.uniform(1.0, 6.0, size=len(pts))
    return roads, pts[:, 0], pts[:, 1], radii


def test_the_closed_form_reproduces_shapely_to_within_its_discretisation() -> None:
    roads, px, py, radii = _frame()
    corridor = unary_union([g.buffer(w / 2.0) for g, w in ROADS])
    truth = np.array([Point(x, y).distance(corridor) for x, y in zip(px, py, strict=True)])
    mine = closed_form_distance(px, py, segments(roads))
    assert np.abs(mine - truth).max() < 0.01, (
        f"max distance disagreement {np.abs(mine - truth).max():.4g} m exceeds shapely's own "
        f"discretisation scale at quad_segs=16")
    assert (mine <= truth + 1e-9).all(), (
        "the closed form must never exceed shapely's distance: shapely's buffer is INSCRIBED, so "
        "it is the one that reports distances too large")


def test_the_residual_is_shapelys_discretisation_and_not_our_error() -> None:
    """The decisive test. If the formula were wrong the gap would be constant in `quad_segs`; it
    falls quadratically, which identifies the residual as shapely's polygonal buffer."""
    roads, px, py, _ = _frame()
    mine = closed_form_distance(px, py, segments(roads))
    errs = []
    for qs in (16, 64, 256):
        corridor = unary_union([g.buffer(w / 2.0, quad_segs=qs) for g, w in ROADS])
        truth = np.array([Point(x, y).distance(corridor) for x, y in zip(px, py, strict=True)])
        errs.append(float(np.abs(mine - truth).max()))
    assert errs[0] > errs[1] > errs[2], f"error did not fall with resolution: {errs}"
    assert errs[2] < errs[0] / 100.0, (
        f"a 16x resolution rise should cut a quadratic error ~256x; got {errs[0]:.3g} -> "
        f"{errs[2]:.3g}. A constant residual would mean the formula is wrong, not shapely coarse.")


def test_no_roads_costs_nothing_rather_than_producing_nan() -> None:
    _, px, py, radii = _frame()
    d = closed_form_distance(px, py, np.empty((0, 5)))
    assert np.isinf(d).all(), "an empty road set must give infinite distance, not zero"
    assert displacement_from_distance(radii, d) == 0.0


def test_a_zero_length_road_is_its_own_endpoint() -> None:
    roads = GeoDataFrame({"width_m": [7.0]},
                         geometry=[LineString([(0, 0), (0, 0)])], crs=UTM)
    d = closed_form_distance(np.array([10.0]), np.array([0.0]), segments(roads))
    assert np.isfinite(d[0]), f"degenerate road produced {d[0]}"
    assert d[0] == pytest.approx(10.0 - 3.5)


def _synthetic_block(n: int = 4, cell: float = 25.0) -> Block:
    """A square block with one building per cell -- enough for `default_roads`, which needs only
    the parcel hull and the building points."""
    polys, ids, pts = [], [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(i * cell, j * cell), ((i + 1) * cell, j * cell),
                                  ((i + 1) * cell, (j + 1) * cell), (i * cell, (j + 1) * cell)]))
            ids.append(i * n + j)
            pts.append(Point((i + 0.5) * cell, (j + 0.5) * cell))
    parcels = GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast("Polygon | MultiPolygon", parcels.geometry.union_all())
    return Block(block_id="s", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=GeoDataFrame(geometry=[boundary.boundary], crs=UTM),
                 building_points=GeoDataFrame(geometry=pts, crs=UTM))


def test_default_roads_are_reproducible_disjoint_and_inside_the_block() -> None:
    block = _synthetic_block()
    a, b = default_roads(block, 7.0), default_roads(block, 7.0)
    assert len(a) == 2
    for i in (0, 1):
        assert a.geometry.iloc[i].equals(b.geometry.iloc[i]), f"road {i + 1} is not reproducible"
    hull = block.parcels.union_all()
    for i, g in enumerate(a.geometry):
        assert g.length > 0, f"road {i + 1} is degenerate"
        assert hull.buffer(1e-6).contains(g), f"road {i + 1} leaves the block"
    assert not a.geometry.iloc[0].buffer(3.5).intersects(a.geometry.iloc[1].buffer(3.5)), (
        "the two default corridors already overlap, so merging them is not something the reader "
        "does")
