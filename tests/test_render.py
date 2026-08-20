from pathlib import Path
from typing import cast

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.colors import to_hex, to_rgba
from matplotlib.figure import Figure
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.budget import building_radii
from reblock.contracts import Block, Metrics, Proposal
from reblock.derive.access import parcel_access_layers
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width
from reblock.render import (
    _DISPLACED_PT,
    field_contributions,
    frame_bbox,
    render_after,
    render_before,
    render_field,
    save_render,
)

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
    connector = with_width(gpd.GeoDataFrame(geometry=[LineString([(1, 0), (1, 1)])], crs=UTM),
                           DEFAULT_ROAD_WIDTH_M)
    return Proposal(block_id=block.block_id, crs=UTM, roads=connector, method="topology")


def _field_block(n: int = 3, cell: float = 20.0) -> Block:
    """A grid block WITH building points -- `_grid_block` has none, and render_field draws them.

    20 m cells rather than `_grid_block`'s 1 m: a 7 m road (DEFAULT_ROAD_WIDTH_M, and
    permeability.py:205 RAISES below it) buffers to 3.5 m, which on unit parcels swallows the whole
    block and leaves every building fully displaced -- so there would be no zero-cost disk to
    assert on, which is exactly the case these tests exist to check.
    """
    polys, ids, pts = [], [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(i * cell, j * cell), ((i + 1) * cell, j * cell),
                                  ((i + 1) * cell, (j + 1) * cell), (i * cell, (j + 1) * cell)]))
            ids.append(i * n + j)
            pts.append(Point((i + 0.5) * cell, (j + 0.5) * cell))
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    return Block(block_id="f", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM),
                 building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))


def _mid_road(block: Block) -> gpd.GeoDataFrame:
    """A 7 m road up the middle column of `_field_block`: grazes the two nearest columns and misses
    the far one entirely, so both disk collections are non-empty and the shading is partial."""
    x0, y0, x1, y1 = block.parcels.total_bounds
    x = float(x0 + (x1 - x0) / 3.0)
    return with_width(gpd.GeoDataFrame(
        geometry=[LineString([(x, y0), (x, y1)])], crs=UTM), DEFAULT_ROAD_WIDTH_M)


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
    # permeability.parcel_potentials), normalized to [0, series.max()] -- a different vmin
    # from the depth coloring, but the same red YlOrRd cmap.
    block = _grid_block(3)
    potentials = _potentials(block)

    fig = render_before(block, potentials, vmax=float(potentials.max()), field="perm")

    fill = cast(PatchCollection, fig.axes[0].collections[0])
    assert fill.get_clim() == pytest.approx((0.0, float(potentials.max())))
    assert fill.get_cmap().name == "YlOrRd"    # same red scale as the depth coloring


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
    roads = with_width(
        gpd.GeoDataFrame(geometry=[LineString([(0, 10), (20, 10)])], crs=crs),
        DEFAULT_ROAD_WIDTH_M)
    prop = Proposal(block_id="b", crs=crs, roads=roads, edges=None,
                    proposal_id="x", method="m", params={"corridor_m": 1.0}, block_identity=None)
    disp = _displaced_points(block, prop)
    assert "c" in disp.columns and "radius" in disp.columns
    assert (disp["c"] > 0).any() and (disp["c"] <= 1).all()


def test_render_graph_returns_figure_with_axes() -> None:
    from reblock.perm_graph import permeability_graph
    from reblock.render import render_graph

    block = _grid_block(6)
    fig_data = permeability_graph(block, None)
    fig = render_graph(fig_data, block, layer="conductance",
                       vmax=float(fig_data.potential.max()),
                       width_norm=float(fig_data.conductance.max()))

    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_render_graph_draws_upgraded_edges_in_the_road_colour() -> None:
    """Road-raised edges are drawn in the road blue, and only when a road actually raised one.

    Asserted on COLOUR, not on a count of collections: on this unit-cell fixture a 7 m road
    blankets the whole mesh, so the grey collection can legitimately be absent and a count would
    read the wrong way round.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba, to_rgba_array

    from reblock.perm_graph import permeability_graph
    from reblock.render import _ROAD_COLOR, _UPGRADED_LW, render_graph

    block = _grid_block(6)
    roads = with_width(
        gpd.GeoDataFrame(geometry=[LineString([(3.0, 0.0), (3.0, 5.0)])], crs=UTM),
        DEFAULT_ROAD_WIDTH_M)

    plain = permeability_graph(block, None)
    roaded = permeability_graph(block, roads)
    assert roaded.upgraded.any(), "fixture road must upgrade an edge or the test is vacuous"

    def _edge_colours(f: Figure) -> set[tuple[float, ...]]:
        # `LineCollection.get_colors()` is typed `ColorType | Sequence[ColorType]` (it can, in
        # principle, return one bare color rather than a sequence); at runtime it always hands
        # back an (N, 4) RGBA array here, and to_rgba_array is the identity on that shape -- so
        # routing through it gives mypy a concrete `np.ndarray` to iterate without changing what
        # gets compared.
        return {tuple(round(float(v), 4) for v in colour)
                for coll in f.axes[0].collections if isinstance(coll, LineCollection)
                for colour in to_rgba_array(coll.get_colors())}

    def _blue_collections(f: Figure, blue: tuple[float, ...]) -> list[LineCollection]:
        return [coll for coll in f.axes[0].collections if isinstance(coll, LineCollection)
                for colour in to_rgba_array(coll.get_colors())
                if tuple(round(float(v), 4) for v in colour) == blue]

    blue = tuple(round(float(v), 4) for v in to_rgba(_ROAD_COLOR))
    a = render_graph(plain, block, layer="current", vmax=1.0, width_norm=1.0)
    b = render_graph(roaded, block, layer="current", vmax=1.0, width_norm=1.0, roads=roads)
    assert blue not in _edge_colours(a)
    assert blue in _edge_colours(b)

    def _linewidths(coll: LineCollection) -> list[float]:
        # `LineCollection.get_linewidth()` is typed `float | Sequence[float]` for the same reason
        # `get_colors()` is above -- at runtime it hands back a sequence here, but the isinstance
        # check keeps this correct (and mypy --strict happy) even if a single bare float ever came
        # back.
        lw = coll.get_linewidth()
        return [float(lw)] if isinstance(lw, (int, float)) else [float(v) for v in lw]

    # The road-raised (blue) collection draws at the fixed `_UPGRADED_LW`, never a width derived
    # from conductance/current -- this is exactly what the site caption's exception (Important 1,
    # 2026-08-14 fix wave) rests on, so a regression back to a variable width must fail here.
    for coll in _blue_collections(b, blue):
        widths = _linewidths(coll)
        assert widths == [_UPGRADED_LW] * len(widths)


def _disk_paths(ax: Axes) -> int:
    """Total paths across the disk collections, found by COLOUR rather than by count.

    Both disk layers draw in `_DISPLACED_PT` -- filled for grazed buildings, outlined for untouched
    ones -- while the parcel wireframe is `_CONTEXT_OUTLINE` and the corridor is `_ROAD_COLOR`. A
    path COUNT cannot separate them: parcels are Voronoi cells of the building points
    (src/reblock/mesh.py:59), so a block has exactly as many parcels as buildings (263 and 263 on
    the pinned block), and the first version of this test matched the parcel wireframe and passed
    while the untouched disks were not drawn at all.
    """
    want = to_rgba(_DISPLACED_PT)[:3]

    def matches(arr: object) -> bool:
        rows = np.atleast_2d(np.asarray(arr, dtype=float))
        return any(row.size >= 3 and np.allclose(row[:3], want, atol=1.0 / 255.0) for row in rows)

    return sum(len(c.get_paths()) for c in ax.collections
               if matches(c.get_facecolor()) or matches(c.get_edgecolor()))


def test_render_field_draws_every_building_not_only_the_displaced_ones():
    """The point of the figure: a reader must be able to see that a road THREADED a gap, which means
    seeing the disks it missed. `render_after` draws only the displaced ones."""
    block = _field_block()
    radii = building_radii(block.building_points)
    fig = render_field(block, _mid_road(block), radii)
    n = len(block.building_points)
    assert _disk_paths(fig.axes[0]) == n, (
        f"{_disk_paths(fig.axes[0])} disks drawn for {n} buildings -- one of the two disk layers is "
        f"missing, so a road that threaded a gap would look like a road that merely missed")


def test_render_field_shades_by_c_and_uses_the_named_constant():
    block = _field_block()
    roads = _mid_road(block)
    radii = building_radii(block.building_points)
    c = field_contributions(block, roads, radii)
    fig = render_field(block, roads, radii)
    alphas = sorted({round(float(a), 6) for coll in fig.axes[0].collections
                     for a in np.atleast_1d(coll.get_alpha() or 1.0)})
    assert any(a not in (0.0, 1.0) for a in alphas), (
        "no partial alpha anywhere: the disks are not shaded by c")
    assert _DISPLACED_PT in {to_hex(col) for coll in fig.axes[0].collections
                             for col in np.atleast_2d(coll.get_facecolor())}, (
        f"the grazed disks must use the named constant {_DISPLACED_PT}, not an inline literal")
    assert 0.0 < c.max() <= 1.0


def test_render_field_never_fills_parcels():
    """Piece B's finding, restated: filling parcels states one quantity twice and drowns the
    subject. The parcel collection must be face-transparent."""
    block = _field_block()
    fig = render_field(block, _mid_road(block), building_radii(block.building_points))
    parcel_faces = [coll.get_facecolor() for coll in fig.axes[0].collections
                    if len(coll.get_paths()) == len(block.parcels)]
    assert parcel_faces, "no collection matches the parcel count"
    assert all(np.asarray(f).size == 0 or float(np.asarray(f)[0][3]) == 0.0
               for f in parcel_faces), "parcels are filled"

