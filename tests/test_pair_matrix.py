"""scripts/pair_matrix.py: `--analyze` reads only the parquet, and a matrix row is computed the way
the committed matrix's rows were."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reblock.buildings import SpacingDiscs
from reblock.contracts import Block
from reblock.transplant.transport import fit_transport, parcel_xy
from scripts import pair_matrix

UTM = CRS.from_epsg(32734)


def _synthetic_matrix() -> pd.DataFrame:
    # 2 recipients x 3 donors each -- small, but enough to exercise every column
    # `analyze_fidelity_vs_distance` touches without needing the real matrix.
    return pd.DataFrame(
        {
            "recipient": ["r0", "r0", "r0", "r1", "r1", "r1"],
            "donor": ["d0", "d1", "d2", "d3", "d4", "d5"],
            "real_gw_dist": [0.010, 0.020, 0.030, 0.011, 0.021, 0.031],
            "perm_gap": [0.10, 0.05, -0.05, -0.05, -0.10, -0.15],
            "feature_dist": [0.5, 1.0, 1.5, 0.6, 1.1, 1.6],
            "road_len_m": [10.0, 20.0, 30.0, 10.0, 20.0, 30.0],
        }
    )


def test_analyze_does_no_pool_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--analyze` re-derives the committed headline from the parquet alone: no pool, no GW, no
    OSM. Building a pool raises here, so reaching one fails the test."""
    def refuse(*_: object) -> None:
        raise AssertionError("--analyze built a pool")

    monkeypatch.setattr(pair_matrix, "donor_pool", refuse)
    out = tmp_path / "matrix.parquet"
    _synthetic_matrix().to_parquet(out)
    monkeypatch.setattr(sys, "argv", ["pair_matrix", "--analyze", "--out", str(out)])
    pair_matrix.main()


def test_analyze_matches_direct_call_to_analyze_fidelity_vs_distance() -> None:
    result = pair_matrix.analyze_fidelity_vs_distance(_synthetic_matrix())
    assert result.n == 6
    assert result.n_recipients == 2
    assert result.within_recipient.dof == 6 - 2 - 1


def _slab(w: int, h: int, block_id: str) -> Block:
    """`w` x `h` 10 m parcels, one building each, street frontage along the bottom edge only."""
    polys = [Polygon([(10 * i, 10 * j), (10 * i + 10, 10 * j), (10 * i + 10, 10 * j + 10),
                      (10 * i, 10 * j + 10)]) for j in range(h) for i in range(w)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    return Block(block_id=block_id, crs=UTM, boundary=cast(Polygon, unary_union(polys)),
                 parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (10 * w, 0)])], crs=UTM),
                 building_geometries=gpd.GeoDataFrame(
                     geometry=[Point(p.centroid.x, p.centroid.y) for p in polys], crs=UTM),
                 source_content_hash=None, building_tier=SpacingDiscs)


def test_a_row_carries_the_fits_own_distance_and_a_consistent_gap() -> None:
    recipient, donor = _slab(8, 7, "r"), _slab(7, 8, "d")        # 56 parcels: signable
    lines = gpd.GeoDataFrame(geometry=[LineString([(30, 0), (30, 55), (5, 55)])], crs=UTM)
    scorer = pair_matrix.pair_scorer()
    timings = pair_matrix.StageTimings(osm_fetch=0.0, gw=0.0, transplant=0.0, clearance=0.0,
                                       permeability=0.0)
    row = scorer.score(recipient, donor, lines, timings)
    assert set(row) == set(pair_matrix.PairRow.__annotations__)
    assert (row["recipient"], row["donor"]) == ("r", "d")
    assert row["real_gw_dist"] == fit_transport(parcel_xy(donor), parcel_xy(recipient),
                                                scorer.transport).gw_dist
    assert row["road_len_m"] > 0
    assert row["perm_gap"] == row["perm_proposal"] - row["perm_direct"]
    assert timings.gw > 0 and timings.permeability > 0
