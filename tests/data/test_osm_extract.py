from pathlib import Path

import geopandas as gpd
import pytest
import yaml
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.data.osm_extract import (
    FOOTPATH_TAGS,
    TOLERANCES,
    PbfDesireLines,
    assert_zone_fit,
    census_rows,
    interiority_row,
    utm_zone_epsg,
)

CRS_M = CRS.from_epsg(32734)
BOUNDARY = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])


def _lines(*geoms: LineString) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=list(geoms), crs=CRS_M)


def test_interiority_row_reports_count_and_length_at_every_tolerance() -> None:
    row = interiority_row(
        "b1", BOUNDARY, _lines(LineString([(10, 50), (90, 50)])), _lines(), CRS_M)
    assert row["block_id"] == "b1"
    for tol in TOLERANCES:
        assert row[f"n_interior_segments_{tol}"] == 1
        assert row[f"interior_length_m_{tol}"] == pytest.approx(80.0)


def test_interiority_row_count_gate_is_robust_where_length_is_not() -> None:
    """A path crossing the interior but touching the edge: length is trimmed by tolerance,
    the count is not. This is the spike's central finding and the reason both are reported."""
    row = interiority_row(
        "b2", BOUNDARY, _lines(LineString([(0, 50), (90, 50)])), _lines(), CRS_M)
    assert row["n_interior_segments_0.5"] == row["n_interior_segments_5.0"] == 1
    len_low: float = row["interior_length_m_0.5"]  # type: ignore[assignment]
    len_high: float = row["interior_length_m_5.0"]  # type: ignore[assignment]
    assert len_low > len_high


def test_interiority_row_keeps_near_miss_separate() -> None:
    row = interiority_row(
        "b3", BOUNDARY,
        _lines(LineString([(10, 50), (90, 50)])),
        _lines(LineString([(10, 20), (90, 20)])),
        CRS_M)
    assert row["n_interior_segments_0.5"] == 1
    assert row["n_near_miss_segments_0.5"] == 1
    assert row["interior_length_m_0.5"] == pytest.approx(80.0)


def test_interiority_row_uncovered_block_is_all_zero() -> None:
    row = interiority_row("b4", BOUNDARY, _lines(), _lines(), CRS_M)
    assert row["n_interior_segments_0.5"] == 0
    assert row["interior_length_m_0.5"] == 0.0


def test_config_tag_list_matches_python_definition() -> None:
    """conf/ and Python must not be able to drift: one list, one place."""
    shared = yaml.safe_load(Path("conf/desire_source/_footpath_tags.yaml").read_text())
    assert tuple(shared["footpath_tags"]) == FOOTPATH_TAGS

    osm_cfg = yaml.safe_load(Path("conf/desire_source/osm.yaml").read_text())
    assert osm_cfg["tags"] == "${footpath_tags}", (
        "osm.yaml must interpolate the shared list, not re-declare it")


def test_pbf_identity_is_stable_and_keys_on_content_and_tags(tmp_path: Path) -> None:
    """Unlike OSMDesireLines (identity None when live), a PBF source is cacheable -- which is
    what flips osm_footpaths from uncacheable to cacheable, so the identity must be content-keyed.
    """
    pbf = tmp_path / "x.osm.pbf"
    pbf.write_bytes(b"not-a-real-pbf-but-hashable")
    a = PbfDesireLines(pbf)
    b = PbfDesireLines(pbf)
    assert a.identity == b.identity
    assert a.identity is not None

    pbf2 = tmp_path / "y.osm.pbf"
    pbf2.write_bytes(b"different-content")
    assert PbfDesireLines(pbf2).identity != a.identity
    assert PbfDesireLines(pbf, tags=("footway",)).identity != a.identity


def test_pbf_conforms_to_desire_line_source_protocol() -> None:
    """Structural conformance is enforced STATICALLY by this annotated binding -- mypy --strict
    fails if PbfDesireLines does not satisfy the Protocol. Do NOT rewrite this as
    `isinstance(..., DesireLineSource)`: DesireLineSource is a bare Protocol, not
    @runtime_checkable, so isinstance raises TypeError rather than returning False."""
    from reblock.methods.desire_lines import DesireLineSource

    source: DesireLineSource = PbfDesireLines(Path("nonexistent.osm.pbf"))
    assert callable(source.desire_lines)


def test_utm_zone_epsg_picks_hemisphere_and_zone() -> None:
    assert utm_zone_epsg(18.5, -33.9) == 32734      # Cape Town, zone 34 south
    assert utm_zone_epsg(36.8, -1.3) == 32737       # Nairobi, zone 37 south
    assert utm_zone_epsg(36.8, 1.3) == 32637        # just north of the equator


def test_assert_zone_fit_is_loud_about_a_forgotten_batch() -> None:
    """A single country-wide UTM does not crash -- it silently biases lengths by up to 3.5%.
    The assertion is what makes a missed batch loud instead of a quiet drift."""
    assert_zone_fit(18.5, 32734)                    # zone 34 central meridian is 21E
    with pytest.raises(ValueError, match="outside UTM zone"):
        assert_zone_fit(41.9, 32734)


def test_census_rows_emits_one_row_per_block() -> None:
    blocks = gpd.GeoDataFrame(
        {"block_id": ["a", "b"]},
        geometry=[
            Polygon([(18.50, -33.95), (18.51, -33.95), (18.51, -33.94), (18.50, -33.94)]),
            Polygon([(18.52, -33.95), (18.53, -33.95), (18.53, -33.94), (18.52, -33.94)]),
        ],
        crs=CRS.from_epsg(4326))
    empty = gpd.GeoDataFrame(geometry=[], crs=CRS.from_epsg(4326))
    rows = census_rows(blocks, empty, empty, 32734)
    assert [r["block_id"] for r in rows] == ["a", "b"]
    assert all(r["n_interior_segments_0.5"] == 0 for r in rows)
    for r in rows:
        area: float = r["area_m2"]  # type: ignore[assignment]
        assert area > 0


@pytest.mark.network
def test_pbf_and_overpass_agree_on_a_pinned_bbox() -> None:
    """Two sources for the same data WILL disagree (Geofabrik extract timestamp vs live Overpass;
    GDAL `lines` layer vs Overpass `out geom`). Without this test, two sources is accommodation
    rather than a Strategy. Tolerance is loose because the snapshots differ in date, not content."""
    from reblock.methods.desire_lines import OSMDesireLines

    pbf = Path.home() / ".cache" / "reblock" / "osm_pbf" / "south-africa-latest.osm.pbf"
    if not pbf.exists():
        pytest.skip("run scripts/osm_census.py --fetch first")

    bbox = (18.55, -33.99, 18.58, -33.96)   # a Cape Flats window with dense footpath mapping
    crs = CRS.from_epsg(32734)
    a = PbfDesireLines(pbf).desire_lines(bbox, crs)
    b = OSMDesireLines(timeout_s=180.0).desire_lines(bbox, crs)
    assert a.geometry.length.sum() == pytest.approx(b.geometry.length.sum(), rel=0.25)
