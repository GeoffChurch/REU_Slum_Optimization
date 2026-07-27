import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.data.osm_extract import TOLERANCES, interiority_row

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
