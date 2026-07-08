from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Block, Metrics, Proposal, Result
from reblock.emit import RenderConfig, render_results

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _kc(block: Block) -> Metrics:
    layers = pd.Series([1] * len(block.parcels),
                       index=pd.Index(block.parcels["parcel_id"], name="parcel_id"))
    return Metrics(block_id=block.block_id, method="x", eval="kcomplexity",
                   values={"delta_k": 0.0},
                   fields={"access_before": layers, "access_after": layers})


def test_render_results_after_filenames_unique_for_empty_proposal_ids(tmp_path: Path) -> None:
    # Two proposals for one block that both leave proposal_id="" must not collide
    # onto one filename -- the emitter falls back to a per-proposal index.
    block = _grid_block(3)
    results = [
        Result(block=block, proposal=Proposal(block_id="g", crs=UTM, proposal_id=""),
               metrics=(_kc(block),)),
        Result(block=block, proposal=Proposal(block_id="g", crs=UTM, proposal_id=""),
               metrics=(_kc(block),)),
    ]
    render_results(results, tmp_path, RenderConfig(enabled=True))
    afters = sorted(p.name for p in tmp_path.glob("*_after.png"))
    assert afters == ["g_proposal0_after.png", "g_proposal1_after.png"]
    assert (tmp_path / "g_before.png").exists()


def test_render_results_skips_block_without_kcomplexity(tmp_path: Path) -> None:
    block = _grid_block(3)
    other = Metrics(block_id="g", method="x", eval="weakdual_k", values={"k": 1.0})
    result = Result(block=block, proposal=Proposal(block_id="g", crs=UTM), metrics=(other,))
    render_results([result], tmp_path, RenderConfig(enabled=True))
    assert list(tmp_path.glob("*.png")) == []


def test_render_results_rejects_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        render_results([], tmp_path, RenderConfig(enabled=True, format="webpage"))
