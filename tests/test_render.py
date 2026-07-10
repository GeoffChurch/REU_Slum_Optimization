from pathlib import Path
from typing import cast

import geopandas as gpd
import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import Block, Metrics, Proposal
from reblock.derive.access import parcel_access_layers
from reblock.render import frame_bbox, render_after, render_before, save_render

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


def test_frame_bbox_is_square_centred_and_padded() -> None:
    # A non-square input (wide rectangle) must still produce a square frame,
    # centred on the input's own centre, with the half-width strictly larger
    # than the input's half-extent (padding applied).
    geoms = gpd.GeoSeries([Polygon([(0, 0), (10, 0), (10, 2), (0, 2)])], crs=UTM)

    minx, miny, maxx, maxy = frame_bbox(geoms, pad_frac=0.6)

    width, height = maxx - minx, maxy - miny
    assert width == pytest.approx(height)
    assert (minx + maxx) / 2 == pytest.approx(5.0)
    assert (miny + maxy) / 2 == pytest.approx(1.0)
    # half-extent of the input's longer side is 5.0 (+1.0 base pad) -> 6.0,
    # then padded by pad_frac=0.6 -> 9.6, so the frame is well beyond the input.
    assert width / 2 == pytest.approx(6.0 * 1.6)


def test_draw_heatmap_with_context_and_own_points_renders_without_error(
    tmp_path: Path,
) -> None:
    # Small synthetic context/own-point layers, disjoint from the block itself,
    # exercise the full draw order (context outlines/points + own points) and
    # must not raise, and must still produce a written, non-empty file.
    block = _grid_block(3)
    layers = parcel_access_layers(block, None)
    context_outlines = gpd.GeoDataFrame(
        {"block_id": ["neighbour"]},
        geometry=[Polygon([(5, 5), (7, 5), (7, 7), (5, 7)])],
        crs=UTM,
    )
    context_points = gpd.GeoDataFrame(geometry=[Point(6, 6)], crs=UTM)
    own_points = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5), Point(1.5, 1.5)], crs=UTM)

    fig = render_before(
        block, layers, vmax=2,
        context_outlines=context_outlines, context_points=context_points,
        own_points=own_points,
    )
    out = tmp_path / "with_context.png"
    save_render(fig, out)

    assert out.exists()
    assert out.stat().st_size > 0


def test_render_before_uses_the_passed_frame_verbatim() -> None:
    # The caller (emit.py) computes frame_bbox ONCE and threads it through as the `frame`
    # kwarg, so the context query and the axes view never drift apart. Passing an explicit
    # frame must set the view to exactly that bbox, not recompute one internally.
    block = _grid_block(3)
    layers = parcel_access_layers(block, None)
    frame = (-100.0, -100.0, 200.0, 200.0)

    fig = render_before(block, layers, vmax=2, frame=frame)

    ax = fig.axes[0]
    assert ax.get_xlim() == pytest.approx((frame[0], frame[2]))
    assert ax.get_ylim() == pytest.approx((frame[1], frame[3]))
