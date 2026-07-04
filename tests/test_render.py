from pathlib import Path
from typing import cast

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block, Metrics, Proposal
from reblock.derive.access import parcel_access_layers
from reblock.render import render_after, render_before, save_render

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _connector_proposal(block: Block) -> Proposal:
    # 3x3 grid: interior road from boundary node (1,0) to the centre's corner
    # (1,1) reaches the centre parcel -> k drops from 2 to 1 (mirrors
    # tests/eval/test_kcomplexity.py's fixture).
    connector = gpd.GeoDataFrame(geometry=[LineString([(1, 0), (1, 1)])], crs=UTM)
    return Proposal(block_id=block.block_id, crs=UTM, roads=connector, method="topology")


def test_render_before_returns_figure_with_axes() -> None:
    block = _grid_block(3)
    layers = parcel_access_layers(block, None)

    fig = render_before(block, layers, vmax=2)

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_render_after_returns_figure_with_axes() -> None:
    block = _grid_block(3)
    proposal = _connector_proposal(block)
    layers = parcel_access_layers(block, proposal.roads)

    fig = render_after(block, proposal, layers, vmax=2)

    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_render_after_adds_a_roads_artist_over_render_before() -> None:
    # render_before's axes hold the parcel-fill collection + the boundary
    # outline collection; render_after adds a third collection for
    # proposal.roads -- so with a non-empty proposal it has strictly more.
    block = _grid_block(3)
    proposal = _connector_proposal(block)
    layers_before = parcel_access_layers(block, None)
    layers_after = parcel_access_layers(block, proposal.roads)

    fig_before = render_before(block, layers_before, vmax=2)
    fig_after = render_after(block, proposal, layers_after, vmax=2)

    assert len(fig_after.axes[0].collections) > len(fig_before.axes[0].collections)


def test_render_after_with_no_roads_adds_no_extra_artist() -> None:
    # Contract: a proposal with roads=None must not blow up, and (since
    # there's nothing new to draw) shouldn't add a roads collection either.
    block = _grid_block(3)
    proposal = Proposal(block_id=block.block_id, crs=UTM, roads=None, method="topology")
    layers = parcel_access_layers(block, None)

    fig_before = render_before(block, layers, vmax=2)
    fig_after = render_after(block, proposal, layers, vmax=2)

    assert len(fig_after.axes[0].collections) == len(fig_before.axes[0].collections)


def test_render_before_and_after_share_the_passed_vmax() -> None:
    # A before/after pair for the same block must be on the same colour
    # scale, so a viewer can compare shading across the two figures directly.
    block = _grid_block(3)
    proposal = _connector_proposal(block)
    layers_before = parcel_access_layers(block, None)
    layers_after = parcel_access_layers(block, proposal.roads)

    fig_before = render_before(block, layers_before, vmax=2)
    fig_after = render_after(block, proposal, layers_after, vmax=2)

    before_fill = cast(PatchCollection, fig_before.axes[0].collections[0])
    after_fill = cast(PatchCollection, fig_after.axes[0].collections[0])
    assert before_fill.get_clim() == (1, 2)
    assert after_fill.get_clim() == (1, 2)


def test_render_after_accepts_optional_metrics() -> None:
    block = _grid_block(3)
    proposal = _connector_proposal(block)
    layers = parcel_access_layers(block, proposal.roads)
    metrics = Metrics(
        block_id=block.block_id, method="topology", eval="kcomplexity",
        values={"k_before": 2.0, "k_after": 1.0, "delta_k": 1.0, "added_road_length_m": 1.0},
    )

    fig = render_after(block, proposal, layers, vmax=2, metrics=metrics)

    assert isinstance(fig, Figure)


def test_save_render_writes_a_nonempty_file(tmp_path: Path) -> None:
    block = _grid_block(3)
    layers = parcel_access_layers(block, None)
    fig = render_before(block, layers, vmax=2)
    out = tmp_path / "b.png"

    save_render(fig, out)

    assert out.exists()
    assert out.stat().st_size > 0
