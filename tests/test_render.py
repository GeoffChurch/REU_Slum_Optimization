from pathlib import Path
from typing import cast

import geopandas as gpd
import matplotlib
import pandas as pd
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
    assert len(fig.axes) == 1


def test_render_after_returns_figure_with_axes() -> None:
    block = _grid_block(3)
    proposal = _connector_proposal(block)
    layers = parcel_access_layers(block, proposal.roads)

    fig = render_after(block, proposal, layers, vmax=2)

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


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


def _potentials(block: Block) -> pd.Series:
    # A stand-in for permeability.parcel_potentials: a continuous, non-integer per-parcel series
    # indexed by parcel_id, monotonically increasing so vmax = its max is unambiguous.
    return pd.Series(
        [0.1 * i for i in range(len(block.parcels))],
        index=pd.Index(block.parcels["parcel_id"], name="parcel_id"))


def test_render_before_defaults_to_depth_field() -> None:
    # Backward-friendly: an existing caller that never passes `field=` still gets the depth
    # coloring (vmin=1, YlOrRd), unchanged from before the perm coloring was added.
    block = _grid_block(3)
    layers = parcel_access_layers(block, None)

    fig = render_before(block, layers, vmax=2)

    fill = cast(PatchCollection, fig.axes[0].collections[0])
    assert fill.get_clim() == (1, 2)
    assert fill.get_cmap().name == "YlOrRd"


def test_render_before_perm_field_uses_perm_cmap_and_zero_vmin() -> None:
    # field="perm" colors by the continuous egress-potential series (Task 5 supplies it from
    # permeability.parcel_potentials), normalized to [0, series.max()] -- a different vmin/cmap
    # from the depth coloring.
    block = _grid_block(3)
    potentials = _potentials(block)

    fig = render_before(block, potentials, vmax=float(potentials.max()), field="perm")

    fill = cast(PatchCollection, fig.axes[0].collections[0])
    assert fill.get_clim() == pytest.approx((0.0, float(potentials.max())))
    assert fill.get_cmap().name != "YlOrRd"    # visually distinct from the depth coloring


def test_render_after_perm_field_renders_and_writes_a_file(tmp_path: Path) -> None:
    block = _grid_block(3)
    proposal = _connector_proposal(block)
    potentials = _potentials(block)

    fig = render_after(block, proposal, potentials, vmax=float(potentials.max()), field="perm")
    out = tmp_path / "perm_after.png"
    save_render(fig, out)

    assert out.exists() and out.stat().st_size > 0


def test_render_before_and_after_have_no_title() -> None:
    # Global cleanup: bare heatmaps, no decorative title (matches the already-bare screen map).
    block = _grid_block(3)
    proposal = _connector_proposal(block)
    layers = parcel_access_layers(block, proposal.roads)

    fig_before = render_before(block, layers, vmax=2)
    fig_after = render_after(block, proposal, layers, vmax=2)

    assert fig_before.axes[0].get_title() == ""
    assert fig_after.axes[0].get_title() == ""


def test_draw_heatmap_uses_poster_figsize() -> None:
    # Bumped to (16, 16) so the saved PNG clears ~3500-4000 px on the long edge at
    # save_render's dpi=300 after bbox_inches="tight"/pad_inches=0 cropping -- sharp at
    # 3-4 ft poster scale; see the comment at the figsize call site.
    block = _grid_block(3)
    layers = parcel_access_layers(block, None)

    fig = render_before(block, layers, vmax=2)

    assert tuple(fig.get_size_inches()) == (16.0, 16.0)


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


def test_render_after_marks_displaced_points_and_writes_a_file(tmp_path: Path) -> None:
    block = _grid_block(3)
    proposal = _connector_proposal(block)
    layers = parcel_access_layers(block, proposal.roads)
    # sits on the connector; c/radius are the columns _displaced_points (emit.py) now attaches.
    displaced = gpd.GeoDataFrame(
        {"c": [0.8], "radius": [1.0]}, geometry=[Point(1.0, 0.5)], crs=UTM)

    fig_after = render_after(block, proposal, layers, vmax=2, displaced_points=displaced)
    out = tmp_path / "displaced.png"
    save_render(fig_after, out)

    assert out.exists() and out.stat().st_size > 0


def test_render_after_displaced_points_add_an_artist_over_own_points_alone() -> None:
    # The displaced-point disk is drawn on top of own_points as its own artist, so a call with
    # displaced_points must have strictly more collections than the same call without.
    block = _grid_block(3)
    proposal = _connector_proposal(block)
    layers = parcel_access_layers(block, proposal.roads)
    own_points = gpd.GeoDataFrame(geometry=[Point(0.5, 0.5), Point(1.0, 0.5)], crs=UTM)
    displaced = gpd.GeoDataFrame(
        {"c": [0.8], "radius": [1.0]}, geometry=[Point(1.0, 0.5)], crs=UTM)

    fig_own_only = render_after(block, proposal, layers, vmax=2, own_points=own_points)
    fig_with_displaced = render_after(
        block, proposal, layers, vmax=2, own_points=own_points, displaced_points=displaced)

    assert len(fig_with_displaced.axes[0].collections) > len(fig_own_only.axes[0].collections)


def test_render_after_with_empty_displaced_points_adds_no_extra_artist() -> None:
    # Guard-empty: an empty displaced_points frame (e.g. a method whose corridor hits no site)
    # must not add anything or raise.
    block = _grid_block(3)
    proposal = _connector_proposal(block)
    layers = parcel_access_layers(block, proposal.roads)
    empty_displaced = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=UTM)

    fig_without = render_after(block, proposal, layers, vmax=2)
    fig_with_empty = render_after(block, proposal, layers, vmax=2, displaced_points=empty_displaced)

    assert len(fig_with_empty.axes[0].collections) == len(fig_without.axes[0].collections)


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


def test_displaced_points_carry_fraction_and_radius(tmp_path):
    # a proposal with roads over a couple of building points -> _displaced_points has c in (0,1]
    import geopandas as gpd
    from shapely.geometry import LineString, Point, Polygon

    from reblock.contracts import Block, Proposal
    from reblock.emit import _displaced_points
    crs = CRS.from_epsg(32734)
    boundary = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    parcels = gpd.GeoDataFrame({"parcel_id": [0]}, geometry=[boundary], crs=crs)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (20, 0)])], crs=crs)
    pts = gpd.GeoDataFrame(geometry=[Point(10, 10), Point(10, 12)], crs=crs)
    block = Block(block_id="b", crs=crs, boundary=boundary, parcels=parcels,
                  streets=streets, building_points=pts)
    roads = gpd.GeoDataFrame(geometry=[LineString([(0, 10), (20, 10)])], crs=crs)
    prop = Proposal(block_id="b", crs=crs, roads=roads, edges=None,
                    proposal_id="x", method="m", params={"corridor_m": 1.0}, block_identity=None)
    disp = _displaced_points(block, prop)
    assert "c" in disp.columns and "radius" in disp.columns
    assert (disp["c"] > 0).any() and (disp["c"] <= 1).all()

