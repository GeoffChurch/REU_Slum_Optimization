import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block, Metrics, Proposal

UTM = CRS.from_epsg(32643)  # WGS84 / UTM 43N (metres)


def _parcels() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"parcel_id": [0]},
                            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs=UTM)


def _streets() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=UTM)


def test_block_constructs() -> None:
    b = Block(block_id="phule_0", crs=UTM, boundary=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
              parcels=_parcels(), streets=_streets())
    assert b.block_id == "phule_0" and b.crs.is_projected


def test_block_rejects_geographic_crs() -> None:
    with pytest.raises(ValueError, match="projected"):
        Block(block_id="x", crs=CRS.from_epsg(4326),
              boundary=Polygon([(0, 0), (1, 0), (1, 1)]),
              parcels=_parcels().to_crs(4326), streets=_streets().to_crs(4326))


def test_block_rejects_missing_parcel_id() -> None:
    with pytest.raises(ValueError, match="parcel_id"):
        Block(block_id="x", crs=UTM, boundary=Polygon([(0, 0), (1, 0), (1, 1)]),
              parcels=_parcels().drop(columns=["parcel_id"]), streets=_streets())


def test_block_rejects_missing_geometry_column() -> None:
    with pytest.raises(ValueError, match="geometry"):
        Block(block_id="x", crs=UTM, boundary=Polygon([(0, 0), (1, 0), (1, 1)]),
              parcels=_parcels().rename_geometry("geom"), streets=_streets())


def test_block_rejects_empty_parcels() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Block(block_id="x", crs=UTM, boundary=Polygon([(0, 0), (1, 0), (1, 1)]),
              parcels=_parcels().head(0), streets=_streets())


def test_metrics_and_proposal_records() -> None:
    m = Metrics(block_id="x", method="topology", eval="kcomplexity",
                values={"k_before": 3.0, "k_after": 1.0})
    assert m.values["k_before"] == 3.0
    assert Proposal(block_id="x", crs=UTM, method="topology").roads is None
