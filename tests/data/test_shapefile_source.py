from pathlib import Path

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import Polygon, box

from reblock.contracts import Block
from reblock.data.shapefile import ShapefileSource

ROOT = Path(__file__).resolve().parents[2]
PHULE = ROOT / "ext" / "topology" / "examples" / "data" / "phule_nagar_v6.shp"
EPWORTH = ROOT / "ext" / "topology" / "Data" / "Epworth_Before.shp"


def test_source_yields_metric_blocks() -> None:
    # Phule Nagar ships with no .prj sidecar, so a CRS assumption must be
    # supplied explicitly -- no more silent EPSG:3857 guessing.
    region = ShapefileSource(PHULE, region_id="phule", assumed_crs=3857).region()
    blocks = list(region.blocks)
    assert len(blocks) >= 1
    b = blocks[0]
    assert isinstance(b, Block) and b.crs.is_projected
    assert not b.parcels.empty and "parcel_id" in b.parcels.columns
    assert b.boundary.area > 0 and b.block_id.startswith("phule_")


def test_streets_excludes_interior_gap_rings() -> None:
    # 8 unit squares tiling a 3x3 grid with the CENTRE removed -> the dissolved
    # block boundary is a donut with one interior ring (the central hole). That
    # hole is a gap between parcels, not a street, so block.streets must be the
    # OUTER frontage only (a single ring), never the interior gap-ring. Real
    # data has many such gaps (CapeTown's dissolved union has 169); seeding the
    # peel or marking roads from them is wrong and crashes the road-builder.
    utm = CRS.from_epsg(32643)
    squares = [box(x, y, x + 1, y + 1) for x in range(3) for y in range(3)
               if not (x == 1 and y == 1)]
    raw = gpd.GeoDataFrame(geometry=squares, crs=utm)
    block = next(ShapefileSource("unused", region_id="donut")._iter_blocks(
        raw, utm, source_content_hash=""))
    assert isinstance(block.boundary, Polygon)    # a single block is a Polygon (holes and all)
    assert len(block.boundary.interiors) == 1     # the central hole really exists
    assert len(block.streets) == 1                # ...but streets = the outer ring only


def test_missing_crs_without_assumed_crs_raises() -> None:
    # Phule Nagar has no .prj; without an explicit assumed_crs, guessing a CRS
    # (e.g. defaulting to Web Mercator) can silently land parcels on Null
    # Island. Fail loud instead.
    with pytest.raises(ValueError, match="CRS"):
        ShapefileSource(PHULE, region_id="phule").region()


def test_epworth_full_drain_is_non_fatal_and_skips_unloadable_components() -> None:
    # The Goal: one malformed record must not crash a whole dataset. Epworth
    # exercises two independent defects at once:
    #   * two native MultiPolygon records -- fixed structurally by exploding
    #     multi-part rows to single Polygons before component grouping (their
    #     exploded pieces dissolve cleanly to single Polygons);
    #   * a handful of components (empirically ~4 of ~584) whose parcels are
    #     genuinely overlapping-sliver geometry: pairwise-adjacent yet their
    #     whole-component unary_union resolves to a MultiPolygon of two
    #     disjoint, substantial-area parts. That's a real source-data defect,
    #     not multi-part-record noise and not floating-point point-touch
    #     noise (raising the _components touch-length threshold doesn't change
    #     which components are bad).
    # A component that can't be expressed as a single Block.boundary Polygon
    # is skipped with a warning (visible, logged data loss) rather than
    # raising -- so a FULL drain of the entire region completes, yields every
    # loadable block, and drops+logs only the genuinely-unloadable ones. This
    # is the fail-loud-but-non-fatal behaviour the Goal wants at scale.
    # assumed_crs=3857 is passed to exercise the same call shape as CRS-less
    # sources; it's harmlessly ignored here since Epworth ships a real .prj
    # (EPSG:32736).
    region = ShapefileSource(EPWORTH, region_id="epworth", assumed_crs=3857).region()
    with pytest.warns(UserWarning, match="skipping component"):
        blocks = list(region.blocks)

    # ~584 components, ~4 unloadable -> comfortably >500 loadable blocks, and
    # the full drain completed without raising (the whole point).
    assert len(blocks) > 500
    for b in blocks:
        assert isinstance(b, Block) and b.crs.is_projected
        assert isinstance(b.boundary, Polygon) and b.boundary.is_valid and b.boundary.area > 0


def test_shapefile_blocks_carry_source_content_hash() -> None:
    region = ShapefileSource(PHULE, region_id="phule", assumed_crs=3857).region()
    blocks = list(region.blocks)
    assert blocks, "expected at least one built block"
    h = blocks[0].source_content_hash
    assert h and all(b.source_content_hash == h for b in blocks)  # same hash for all blocks


def test_shapefile_building_points_empty_and_block_geometries_present() -> None:
    # Phule Nagar has no .prj sidecar (see test_missing_crs_without_assumed_crs_raises).
    src = ShapefileSource(PHULE, region_id="phule", assumed_crs=3857)
    assert src.building_points().empty          # no point cloud -- honest, not a stub
    bg = src.block_geometries()
    assert not bg.empty and set(bg.columns) >= {"block_id", "geometry"}
    assert (bg.geometry.geom_type == "Polygon").all()


def test_shapefile_block_has_empty_building_points() -> None:
    # A parcel shapefile has no point cloud -- Block.building_points is honestly empty (the
    # dataclass default), not a stub or a throwing accessor.
    block = next(iter(ShapefileSource(PHULE, region_id="phule", assumed_crs=3857).region().blocks))
    assert block.building_points.empty
