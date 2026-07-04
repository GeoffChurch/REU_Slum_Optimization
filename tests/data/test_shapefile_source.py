from pathlib import Path

import pytest
from shapely.geometry import Polygon

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
