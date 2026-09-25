import geopandas as gpd
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.buildings import SpacingDiscs
from reblock.contracts import Block, Metrics, Proposal, Result
from tests.block_fixtures import no_buildings

UTM = CRS.from_epsg(32643)  # WGS84 / UTM 43N (metres)


def _parcels() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"parcel_id": [0]},
                            geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])], crs=UTM)


def _streets() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs=UTM)


@pytest.fixture
def _block() -> Block:
    return Block(block_id="b", crs=UTM, boundary=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                 parcels=_parcels(), streets=_streets(),
                 source_content_hash=None, building_geometries=no_buildings(UTM),
                 building_tier=SpacingDiscs)


@pytest.fixture
def _proposal() -> Proposal:
    return Proposal(block_id="b", crs=UTM, roads=None, edges=None, proposal_id="test",
                    method="test", params={})


def test_block_constructs() -> None:
    b = Block(block_id="phule_0", crs=UTM, boundary=Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
              parcels=_parcels(), streets=_streets(),
              source_content_hash=None, building_geometries=no_buildings(UTM),
              building_tier=SpacingDiscs)
    assert b.block_id == "phule_0" and b.crs.is_projected


def test_block_rejects_geographic_crs() -> None:
    with pytest.raises(ValueError, match="projected"):
        Block(block_id="x", crs=CRS.from_epsg(4326),
              boundary=Polygon([(0, 0), (1, 0), (1, 1)]),
              parcels=_parcels().to_crs(4326), streets=_streets().to_crs(4326),
              source_content_hash=None, building_geometries=no_buildings(CRS.from_epsg(4326)),
              building_tier=SpacingDiscs)


def test_block_rejects_missing_parcel_id() -> None:
    with pytest.raises(ValueError, match="parcel_id"):
        Block(block_id="x", crs=UTM, boundary=Polygon([(0, 0), (1, 0), (1, 1)]),
              parcels=_parcels().drop(columns=["parcel_id"]), streets=_streets(),
              source_content_hash=None, building_geometries=no_buildings(UTM),
              building_tier=SpacingDiscs)


def test_block_rejects_missing_geometry_column() -> None:
    with pytest.raises(ValueError, match="geometry"):
        Block(block_id="x", crs=UTM, boundary=Polygon([(0, 0), (1, 0), (1, 1)]),
              parcels=_parcels().rename_geometry("geom"), streets=_streets(),
              source_content_hash=None, building_geometries=no_buildings(UTM),
              building_tier=SpacingDiscs)


def test_block_rejects_empty_parcels() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Block(block_id="x", crs=UTM, boundary=Polygon([(0, 0), (1, 0), (1, 1)]),
              parcels=_parcels().head(0), streets=_streets(),
              source_content_hash=None, building_geometries=no_buildings(UTM),
              building_tier=SpacingDiscs)


def test_metrics_and_proposal_records() -> None:
    m = Metrics(block_id="x", method="topology", eval="kcomplexity",
                values={"k_before": 3.0, "k_after": 1.0}, fields={})
    assert m.values["k_before"] == 3.0
    assert Proposal(block_id="x", crs=UTM, roads=None, edges=None, proposal_id="topology",
                    method="topology", params={}).roads is None


def test_proposal_rejects_an_empty_id() -> None:
    # An empty id would share its rendered filename with every other empty one.
    with pytest.raises(ValueError, match="proposal_id"):
        Proposal(block_id="x", crs=UTM, roads=None, edges=None, proposal_id="",
                 method="topology", params={})


def test_metrics_carries_per_parcel_fields() -> None:
    s = pd.Series([1, 2], index=pd.Index([0, 1], name="parcel_id"))
    m = Metrics(block_id="b", method="topology", eval="kcomplexity",
                values={"k_before": 2.0}, fields={"access_after": s})
    assert m.fields["access_after"].loc[1] == 2


def test_result_metric_lookup(_block: Block, _proposal: Proposal) -> None:
    m = Metrics(block_id="b", method="topology", eval="kcomplexity",
                values={"delta_k": 3.0}, fields={})
    r = Result(block=_block, proposal=_proposal, metrics=(m,))
    assert r.metric("kcomplexity", "delta_k") == 3.0
