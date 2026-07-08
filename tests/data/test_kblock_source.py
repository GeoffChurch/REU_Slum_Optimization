from pathlib import Path
from typing import cast

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.data.kblock import KblockSource
from reblock.derive.access import parcel_access_layers
from reblock.derive.geometric_access import geometric_access_distances

ROOT = Path(__file__).resolve().parents[1]
DJI_BLOCKS = str(ROOT / "data" / "kblock" / "blocks_dji_sample.parquet")
DJI_BLD = str(ROOT / "data" / "kblock" / "buildings_dji_sample.parquet")
CT_BLOCKS = str(ROOT / "data" / "kblock" / "blocks_capetown_sample.parquet")
CT_BLD = str(ROOT / "data" / "kblock" / "buildings_capetown_sample.parquet")


def test_yields_wellformed_blocks_from_fixture() -> None:
    blocks = list(KblockSource(DJI_BLOCKS, DJI_BLD, region_id="dji").region().blocks)
    assert len(blocks) >= 5
    b = blocks[0]
    assert isinstance(b, Block) and isinstance(b.boundary, Polygon) and b.crs.is_projected
    assert not b.parcels.empty and b.parcels["parcel_id"].is_unique
    assert all(g.geom_type == "Polygon" for g in b.parcels.geometry)     # exploded, no MultiPolygon
    assert "kblock_k" in b.attrs


def test_voronoi_parcels_tile_a_synthetic_block() -> None:
    # 3x3 grid of building points in a unit-ish block -> 9 tiling parcels, centre at peel-depth 2.
    utm = CRS.from_epsg(32638)
    poly = box(0, 0, 30, 30)
    pts = [Point(5 + 10 * i, 5 + 10 * j) for i in range(3) for j in range(3)]
    blocks = gpd.GeoDataFrame({"block_id": ["b"], "k_complexity": [0.0]}, geometry=[poly], crs=utm)
    bld = gpd.GeoDataFrame(geometry=pts, crs=utm)
    src = KblockSource("unused", "unused", region_id="t", min_buildings=4)
    # test helper: (blocks_gdf, bld_gdf) -> Iterator[Block]
    block = next(src._blocks_from(blocks, bld))
    assert len(block.parcels) == 9
    assert parcel_access_layers(block, None).max() == 2


def test_all_parcels_single_polygon_on_concave_block() -> None:
    # Explode invariant: on a concave ("plus"-shaped) block, every yielded parcel is a single
    # Polygon even when a clipped Voronoi cell splits into disjoint lobes across the concavity.
    # (Real MultiPolygon splitting is exercised on real data by
    # test_yields_wellformed_blocks_from_fixture, which asserts all-Polygon on the dense
    # informal fixture blocks -- concave, and verified this session to produce MultiPolygon
    # cells pre-explode.)
    utm = CRS.from_epsg(32638)
    # a "+" (single Polygon)
    poly = cast(Polygon, unary_union([box(10, 0, 20, 30), box(0, 10, 30, 20)]))
    # all points inside the "+"
    pts = [Point(15, 5), Point(15, 15), Point(15, 25), Point(5, 15), Point(25, 15)]
    block = next(KblockSource("u", "u", region_id="t", min_buildings=4)._blocks_from(
        gpd.GeoDataFrame({"block_id": ["b"], "k_complexity": [0.0]}, geometry=[poly], crs=utm),
        gpd.GeoDataFrame(geometry=pts, crs=utm)))
    assert all(g.geom_type == "Polygon" for g in block.parcels.geometry)
    assert block.parcels["parcel_id"].is_unique


def test_streets_are_full_boundary_including_holes() -> None:
    # KblockSource sets block.streets = poly.boundary -- the WHOLE boundary including
    # interior rings (a courtyard seeds the peel), deliberately unlike ShapefileSource's
    # `.exterior`. No committed fixture has a hole, so a regression to `.exterior` would
    # pass every other test; lock it with a synthetic block that has one.
    utm = CRS.from_epsg(32638)
    poly = cast(Polygon, box(0, 0, 30, 30).difference(box(10, 10, 20, 20)))
    assert len(poly.interiors) == 1  # sanity: genuinely has a hole
    # building points in the ring around the hole (avoid it), one per side
    pts = [Point(2, 2), Point(28, 2), Point(2, 28), Point(28, 28)]
    blocks = gpd.GeoDataFrame({"block_id": ["b"], "k_complexity": [0.0]}, geometry=[poly], crs=utm)
    bld = gpd.GeoDataFrame(geometry=pts, crs=utm)
    src = KblockSource("unused", "unused", region_id="t", min_buildings=4)
    block = next(src._blocks_from(blocks, bld))
    streets_len = float(block.streets.geometry.length.sum())
    assert abs(streets_len - poly.boundary.length) < 1e-6
    assert streets_len > poly.exterior.length + 1e-6  # the hole's ring adds length


def test_pinned_capetown_block_morphology() -> None:
    # A deep, dense CapeTown block (force-included in the fixture) with real peel signal --
    # pins exact morphology values read off the committed fixture (stable: the fixture's
    # building set is fixed), replacing a vacuous "peel-k >= 2" assertion.
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown", min_buildings=10)
    block = next(b for b in src.region().blocks if b.block_id == "ZAF.9.3.1_1_44882")
    layers = parcel_access_layers(block, None)
    geo = geometric_access_distances(block, None)
    # Pinned by running this test once against the committed fixture and reading the
    # values off the (deterministic) assertion failure -- not invented.
    assert int(layers.max()) == 7
    assert abs(float(geo.max()) - 62.34) < 1.0
    assert list(layers.value_counts().sort_index().values[:3]) == [167, 195, 168]


def test_block_ids_filters_to_requested_block() -> None:
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown",
                       block_ids=["ZAF.9.3.1_1_44882"])
    blocks = list(src.region().blocks)
    assert [b.block_id for b in blocks] == ["ZAF.9.3.1_1_44882"]
    # UTM is estimated from the full frame, so filtering to one block can't shift the
    # CRS: the block reproduces its full-region morphology (pinned peel-k == 7).
    assert int(parcel_access_layers(blocks[0], None).max()) == 7


def test_block_ids_selects_exactly_the_listed_blocks() -> None:
    ids = ["ZAF.9.3.1_1_44882", "ZAF.9.3.1_1_44571"]
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown", block_ids=ids)
    got = [b.block_id for b in src.region().blocks]
    assert got == sorted(ids)   # _blocks_from yields in sorted block_id order


def test_block_ids_unknown_raises() -> None:
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown", block_ids=["NOPE"])
    with pytest.raises(ValueError, match="NOPE"):
        src.region()


def test_capetown_fixture_has_density_columns() -> None:
    import geopandas as gpd
    cols = set(gpd.read_parquet(CT_BLOCKS).columns)
    assert {"building_count", "block_area_m2"} <= cols   # the Screen's cheap-pass signals
