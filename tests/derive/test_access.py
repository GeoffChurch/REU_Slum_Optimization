from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.buildings import SpacingDiscs
from reblock.contracts import Block
from reblock.derive.access import parcel_access_layers
from tests.block_fixtures import no_buildings

UTM = CRS.from_epsg(32643)


def _grid_block(n: int, x0: float = 0.0) -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(x0+i, j), (x0+i+1, j), (x0+i+1, j+1), (x0+i, j+1)]))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets,
                 source_content_hash=None, building_geometries=no_buildings(UTM),
                 building_tier=SpacingDiscs)


def test_2x2_all_on_street() -> None:
    assert parcel_access_layers(_grid_block(2), None).max() == 1


def test_3x3_centre_is_layer_2() -> None:
    layers = parcel_access_layers(_grid_block(3), None)
    assert layers.max() == 2
    assert (layers == 2).sum() == 1        # exactly the centre parcel (id 4)
    assert layers.loc[4] == 2


def test_strip_is_honest_not_degenerate() -> None:
    # 1xN strip, only the far-left parcel touches the (left-edge) street -> depth N
    polys = [Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)]) for i in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(5))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    block = Block(block_id="s", crs=UTM, boundary=cast(Polygon, parcels.geometry.union_all()),
                  parcels=parcels, streets=streets,
                  source_content_hash=None, building_geometries=no_buildings(UTM),
                  building_tier=SpacingDiscs)
    assert parcel_access_layers(block, None).max() == 5     # weak-dual wrongly gives 1


def test_indexed_by_parcel_id_survives_reorder() -> None:
    # Give non-positional parcel_ids (1000+) AND shuffle the rows so a parcel's
    # position in the frame no longer equals its id -> a positional/RangeIndex
    # bug can't accidentally return the right layer (loc[1004] would KeyError or
    # return the wrong parcel's layer).
    base = _grid_block(3)
    reordered = base.parcels.copy()
    reordered["parcel_id"] = reordered["parcel_id"] + 1000
    reordered = reordered.sample(frac=1, random_state=1).reset_index(drop=True)
    block = Block(block_id="g", crs=base.crs, boundary=base.boundary,
                  parcels=reordered, streets=base.streets,
                  source_content_hash=None, building_geometries=no_buildings(base.crs),
                  building_tier=SpacingDiscs)
    layers = parcel_access_layers(block, None)
    assert layers.index.name == "parcel_id"
    # original centre id 4 -> now 1004; still the sole layer-2 parcel, by id
    assert layers.loc[1004] == 2
    assert (layers == 2).sum() == 1


def test_nonzero_origin() -> None:
    assert parcel_access_layers(_grid_block(3, x0=1000.0), None).max() == 2


def test_added_road_reduces_depth() -> None:
    block = _grid_block(3)
    connector = gpd.GeoDataFrame(geometry=[LineString([(1, 0), (1, 1)])], crs=UTM)
    assert parcel_access_layers(block, connector).max() == 1   # centre now reached


def test_diagonal_touch_is_not_adjacency() -> None:
    # L-shape. Only p0 is seeded (left-edge street). p0-p1 share an EDGE, p1-p2
    # share an EDGE, but p0-p2 share ONLY a corner point (1,1). With the correct
    # edge-only predicate the reachable path is p0->p1->p2, so p2 is layer 3. If
    # corner touches were (wrongly) counted as adjacency (e.g. `.intersects()`
    # instead of `shared.length > 0`), p2 would be a direct neighbour of p0 and
    # drop to layer 2 -- so this test's value FLIPS on that predicate, unlike the
    # symmetric grid tests where diagonal links never change the answer.
    p0 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])   # seeded by the left-edge street
    p1 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])   # shares edge x=1 with p0
    p2 = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])   # shares edge y=1 with p1; corner w/ p0
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1, 2]}, geometry=[p0, p1, p2], crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    block = Block(block_id="L", crs=UTM, boundary=cast(Polygon, parcels.geometry.union_all()),
                  parcels=parcels, streets=streets,
                  source_content_hash=None, building_geometries=no_buildings(UTM),
                  building_tier=SpacingDiscs)
    layers = parcel_access_layers(block, None)
    assert layers.loc[0] == 1
    assert layers.loc[1] == 2
    assert layers.loc[2] == 3        # 3 via the edge chain; would be 2 if corners counted
    assert layers.max() == 3


def test_access_before_unchanged_by_connectivity_change() -> None:
    # roads=None: seed_geom is exactly the streets, so before-layers are identical
    # to the pre-change behaviour (2x2 -> all layer 1, 3x3 -> centre layer 2).
    assert parcel_access_layers(_grid_block(2), None).max() == 1
    assert parcel_access_layers(_grid_block(3), None).max() == 2


def test_disconnected_road_gives_no_credit() -> None:
    # 1x5 strip, street on the left edge only -> depths 1..5. A road segment
    # floating in the interior (NOT touching the street) must confer NO access:
    # k stays 5, exactly as with no roads.
    polys = [Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)]) for i in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(5))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    block = Block(block_id="s", crs=UTM, boundary=cast(Polygon, parcels.geometry.union_all()),
                  parcels=parcels, streets=streets,
                  source_content_hash=None, building_geometries=no_buildings(UTM),
                  building_tier=SpacingDiscs)
    floating = gpd.GeoDataFrame(geometry=[LineString([(3, 0.5), (4, 0.5)])], crs=UTM)  # interior
    assert parcel_access_layers(block, floating).max() == 5          # no unearned credit
    assert parcel_access_layers(block, None).max() == 5


def test_connected_road_reduces_depth() -> None:
    # Same strip, but a road running from the street (x=0) inward DOES connect,
    # so it grants access and reduces depth.
    polys = [Polygon([(i, 0), (i+1, 0), (i+1, 1), (i, 1)]) for i in range(5)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(5))}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    block = Block(block_id="s", crs=UTM, boundary=cast(Polygon, parcels.geometry.union_all()),
                  parcels=parcels, streets=streets,
                  source_content_hash=None, building_geometries=no_buildings(UTM),
                  building_tier=SpacingDiscs)
    connected = gpd.GeoDataFrame(geometry=[LineString([(0, 0.5), (4, 0.5)])], crs=UTM)  # touches
    assert parcel_access_layers(block, connected).max() == 1


def test_disconnected_parcel_gets_layer_past_deepest() -> None:
    # A parcel with no adjacency to anything and no street contact must not
    # silently read as layer 0 (which would look identical to "touches
    # street" under a naive off-by-one); it gets one layer past the
    # deepest *reached* layer instead, so it sorts last honestly.
    near = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    far = Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0, 1]}, geometry=[near, far], crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (0, 1)])], crs=UTM)
    hull = cast(Polygon, parcels.geometry.union_all().convex_hull)
    block = Block(block_id="d", crs=UTM, boundary=hull, parcels=parcels, streets=streets,
                  source_content_hash=None, building_geometries=no_buildings(UTM),
                  building_tier=SpacingDiscs)
    layers = parcel_access_layers(block, None)
    assert layers.loc[0] == 1
    assert layers.loc[1] == 2


def test_street_adjacency_is_not_decided_by_sub_nanometre_noise() -> None:
    """A road placed at exactly `STREET_TOL` must be street-connected, and stay that way under
    perturbations far below any physical scale.

    Distances are differences of coordinates, so the error floor is set by COORDINATE magnitude,
    not by the tolerance. At a UTM northing of ~6.24e6 m one float64 ULP is 9.3e-10 m, so a
    geometry pipeline delivers a nominal 0.5 m gap as 0.5 +/- ~1e-9 -- and a bare `d <= 0.5` then
    decides street-adjacency on the 16th significant digit. `euclidean_grid` trims to
    `street_buffer: 0.5`, exactly `STREET_TOL`, so it parks every grid road on that knife edge.

    The property is STABILITY, not acceptance: whatever the answer is, moving a road by a
    nanometre must not change it. A road genuinely past the tolerance is still refused, so this
    cannot be satisfied by accepting everything.

    FAULT INJECTION: dropping the absolute epsilon from `STREET_TOL` (i.e. `= 0.5`) makes the
    verdicts differ across the perturbations and this fails with both True and False observed.
    """
    import math

    from reblock.derive.access import STREET_TOL, street_connectivity

    base_y = 6_237_723.123456789                      # a real Cape Town UTM 34S northing
    assert math.ulp(base_y) > 1e-10, "fixture is vacuous: pick coordinates where an ULP is coarse"
    street = gpd.GeoDataFrame(
        geometry=[LineString([(260_000.0, base_y), (260_100.0, base_y)])], crs=UTM)

    def connected(gap: float) -> bool:
        y = base_y + gap
        road = gpd.GeoDataFrame(
            geometry=[LineString([(260_010.0, y), (260_090.0, y)])], crs=UTM)
        return street_connectivity(street, road, STREET_TOL).connected_frac == 1.0

    nominal = 0.5
    verdicts = {connected(nominal + d) for d in (-1e-9, -5e-10, 0.0, 5e-10, 1e-9)}
    assert verdicts == {True}, (
        f"street-adjacency at the tolerance flipped under nanometre perturbation: {verdicts}")
    assert not connected(0.6), "a road genuinely past the tolerance must still be refused"
