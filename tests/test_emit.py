import colorsys
from dataclasses import replace
from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import pytest
from matplotlib.figure import Figure
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import BBox, Block, Metrics, Proposal, Region, Result
from reblock.emit import (
    RenderConfig,
    _displaced_points,
    _member_ids,
    _method_colors,
    region_map,
    render_results,
)
from reblock.render import save_render as _real_save_render

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(n) for j in range(n)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_member_ids_parses_region_id_and_passes_through_plain_id() -> None:
    # The own/context split keys on this: a region block_id is "region:" + "+"-joined sorted
    # members (region.region_block); a plain block is itself. A regression here would mis-split
    # own vs surrounding context.
    assert _member_ids("region:DJI.3_1_1808+DJI.3_1_1809") == ["DJI.3_1_1808", "DJI.3_1_1809"]
    assert _member_ids("DJI.3_1_1808") == ["DJI.3_1_1808"]


def _kc(block: Block) -> Metrics:
    layers = pd.Series([1] * len(block.parcels),
                       index=pd.Index(block.parcels["parcel_id"], name="parcel_id"))
    return Metrics(block_id=block.block_id, method="x", eval="kcomplexity",
                   values={"delta_k": 0.0},
                   fields={"access_before": layers, "access_after": layers})


class _FakeSource:
    """A minimal `Source` for emit tests: fixed `block_geometries`/`building_points` GeoData-
    Frames, ignoring `bbox`. Windowing itself is a Source-implementation behaviour already
    covered by KblockSource/ShapefileSource's own tests; these tests exercise only how the
    emitters (region_map, render_results) consume the two accessors -- context-outline
    dropping, own/context point splitting, and the empty-building_points guard path."""

    def __init__(self, blocks: gpd.GeoDataFrame, points: gpd.GeoDataFrame) -> None:
        self._blocks = blocks
        self._points = points

    def region(self) -> Region:
        raise NotImplementedError("not used by render_results/region_map")

    def block_geometries(self, bbox: BBox | None = None) -> gpd.GeoDataFrame:
        return self._blocks

    def building_points(self, bbox: BBox | None = None) -> gpd.GeoDataFrame:
        return self._points


def _source_with_neighbour_and_points() -> _FakeSource:
    # block "g" is the [0,3]x[0,3] square _grid_block(3) builds; "neighbour" sits just east of
    # it, disjoint -- exercises both the own-block-outline drop and the context point dimming.
    blocks = gpd.GeoDataFrame(
        {"block_id": ["g", "neighbour"]},
        geometry=[
            Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
            Polygon([(4, 0), (6, 0), (6, 3), (4, 3)]),
        ],
        crs=UTM,
    )
    points = gpd.GeoDataFrame(
        # one inside "g" (own), one inside "neighbour" (context):
        geometry=[Point(1, 1), Point(5, 1.5)], crs=UTM,
    )
    return _FakeSource(blocks, points)


def _empty_points_source() -> _FakeSource:
    # Mirrors ShapefileSource: block outlines but no building-point cloud.
    blocks = gpd.GeoDataFrame(
        {"block_id": ["g"]},
        geometry=[Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])],
        crs=UTM,
    )
    points = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=UTM)
    return _FakeSource(blocks, points)


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
    render_results(results, tmp_path, RenderConfig(enabled=True),
                   _source_with_neighbour_and_points())
    afters = sorted(p.name for p in tmp_path.glob("*_after.png"))
    assert afters == ["g_proposal0_after.png", "g_proposal1_after.png"]
    assert (tmp_path / "g_before.png").exists()


def test_render_results_skips_block_without_kcomplexity(tmp_path: Path) -> None:
    block = _grid_block(3)
    other = Metrics(block_id="g", method="x", eval="weakdual_k", values={"k": 1.0})
    result = Result(block=block, proposal=Proposal(block_id="g", crs=UTM), metrics=(other,))
    render_results([result], tmp_path, RenderConfig(enabled=True),
                   _source_with_neighbour_and_points())
    assert list(tmp_path.glob("*.png")) == []


def test_render_results_rejects_unsupported_format(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError):
        render_results([], tmp_path, RenderConfig(enabled=True, format="webpage"),
                       _source_with_neighbour_and_points())


def test_render_results_draws_context_outlines_and_points(tmp_path: Path) -> None:
    # A fake Source with a neighbouring block + points inside AND outside the selection's
    # boundary exercises the full context wiring -- windowed query, own-block-outline drop,
    # own/context point split -- end to end, without erroring, and still writes both PNGs.
    block = _grid_block(3)
    result = Result(block=block, proposal=Proposal(block_id="g", crs=UTM), metrics=(_kc(block),))

    render_results([result], tmp_path, RenderConfig(enabled=True),
                   _source_with_neighbour_and_points())

    before = tmp_path / "g_before.png"
    assert before.exists() and before.stat().st_size > 0
    afters = list(tmp_path.glob("g_*_after.png"))
    assert len(afters) == 1 and afters[0].stat().st_size > 0


def test_render_results_guards_empty_building_points(tmp_path: Path) -> None:
    # A source whose building_points is empty (e.g. ShapefileSource) must still render --
    # the guard-empty path, exercised end to end through the emitter.
    block = _grid_block(3)
    result = Result(block=block, proposal=Proposal(block_id="g", crs=UTM), metrics=(_kc(block),))

    render_results([result], tmp_path, RenderConfig(enabled=True), _empty_points_source())

    assert (tmp_path / "g_before.png").stat().st_size > 0


def test_displaced_points_only_keeps_sites_with_positive_displacement_fraction() -> None:
    # Two points: one on the road corridor (c > 0, kept), one far off it (c == 0, dropped) --
    # _displaced_points no longer gates on proposal.params["cost"] (Task 5 replaced the binary
    # within-corridor mark with a continuous disk-shading fraction, for every method).
    block = replace(_grid_block(3),
                    building_points=gpd.GeoDataFrame(
                        geometry=[Point(1.0, 0.5), Point(2.9, 2.9)], crs=UTM))
    roads = gpd.GeoDataFrame(geometry=[LineString([(1.0, 0.0), (1.0, 1.0)])], crs=UTM)
    proposal = Proposal(block_id="g", crs=UTM, roads=roads, params={"corridor_m": 0.5})

    displaced = _displaced_points(block, proposal)

    assert list(displaced.geometry) == [Point(1.0, 0.5)]   # the far point's c == 0, dropped
    assert "c" in displaced.columns and "radius" in displaced.columns
    assert (displaced["c"] > 0).all() and (displaced["c"] <= 1).all()
    assert (displaced["radius"] > 0).all()


def test_displaced_points_defaults_corridor_m_when_absent_from_params() -> None:
    # 1.5m from the road at x=1, well inside the default 3.0m corridor -- a single point, so its
    # radius falls back to corridor_m itself (building_radii, n < 2).
    block = replace(_grid_block(3),
                    building_points=gpd.GeoDataFrame(geometry=[Point(1.0, 2.5)], crs=UTM))
    roads = gpd.GeoDataFrame(geometry=[LineString([(1.0, 0.0), (1.0, 1.0)])], crs=UTM)
    proposal = Proposal(block_id="g", crs=UTM, roads=roads)   # corridor_m defaults to 3.0

    displaced = _displaced_points(block, proposal)

    assert len(displaced) == 1
    assert displaced["radius"].iloc[0] == pytest.approx(3.0)
    assert displaced["c"].iloc[0] == pytest.approx(1.0)


def test_displaced_points_empty_without_building_points_or_roads() -> None:
    block = _grid_block(3)   # building_points defaults to empty
    roads = gpd.GeoDataFrame(geometry=[LineString([(1.0, 0.0), (1.0, 1.0)])], crs=UTM)
    assert _displaced_points(block, Proposal(block_id="g", crs=UTM, roads=roads)).empty

    pts_block = replace(block, building_points=gpd.GeoDataFrame(
        geometry=[Point(1.0, 0.5)], crs=UTM))
    assert _displaced_points(pts_block, Proposal(block_id="g", crs=UTM, roads=None)).empty


def test_render_results_marks_displaced_points(tmp_path: Path) -> None:
    # End-to-end: a block with real building_points + a proposal whose roads corridor covers
    # one of them must still render the after-heatmap without error.
    block = replace(_grid_block(3),
                    building_points=gpd.GeoDataFrame(geometry=[Point(1.0, 0.5)], crs=UTM))
    roads = gpd.GeoDataFrame(geometry=[LineString([(1.0, 0.0), (1.0, 1.0)])], crs=UTM)
    proposal = Proposal(block_id="g", crs=UTM, roads=roads, params={"corridor_m": 0.5})
    result = Result(block=block, proposal=proposal, metrics=(_kc(block),))

    render_results([result], tmp_path, RenderConfig(enabled=True),
                   _source_with_neighbour_and_points())

    afters = list(tmp_path.glob("g_*_after.png"))
    assert len(afters) == 1 and afters[0].stat().st_size > 0


def test_method_colors_are_stable_when_a_method_is_dropped() -> None:
    # The bug: the length pass (with topology) and the displacement pass (topology dropped) drew
    # the same method in different colours, because matplotlib's default cycle assigns by plot
    # order. The fix: colour is a method's index in the FULL registry, which both passes hand in
    # identically -- so a method keeps its colour even when another is absent from the run.
    registry = ["topology", "greedy_arterial_buildable", "greedy_arterial_aspirational",
                "greedy_arterial_displacement", "clearance", "clearance_grid", "osm_footpaths"]
    colors = _method_colors(registry)
    # Every method in the registry gets a distinct colour (evenly spaced hues, no wrap collision).
    assert len(set(colors.values())) == len(registry)
    # Deterministic, and dropping a method from the *plotted subset* never touches the map, because
    # the map is keyed on the registry, not the subset.
    assert _method_colors(registry) == colors


def test_method_colors_hues_are_evenly_spaced_from_zero() -> None:
    # Contract: hue of method i is exactly i/N (N points from [0, 1), so the wheel's wrap -- hue 0
    # == hue 1 -- never lands the last method on the first's colour).
    registry = ["a", "b", "c", "d", "e"]
    colors = _method_colors(registry)
    for i, name in enumerate(registry):
        h, s, v = colorsys.rgb_to_hsv(*colors[name])
        assert h == pytest.approx(i / len(registry))


def test_region_map_draws_member_and_context_points(tmp_path: Path) -> None:
    # region_map now takes the typed Source (no pre-read GeoDataFrame): it reads all
    # candidate outlines itself and windows the building-points query to the region's frame.
    out = region_map(_source_with_neighbour_and_points(), [["g"]], [["g"]], tmp_path)
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_region_map_guards_empty_building_points(tmp_path: Path) -> None:
    out = region_map(_empty_points_source(), [["g"]], [["g"]], tmp_path)
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_screen_proxy_helper_is_gone() -> None:
    # migrate-not-accommodate: the squared-proxy coloring helper is deleted, not retained.
    import reblock.emit as emit
    assert not hasattr(emit, "_screen_proxy")


def test_region_map_colors_by_depth_and_blanks_deselected(tmp_path: Path) -> None:
    # With a selection + depths map, screen.png colors only the selected block ("g") by its true
    # depth; the deselected block ("neighbour") is blanked. Written without error.
    src = _source_with_neighbour_and_points()
    out = region_map(src, [["g"]], [["g"]], tmp_path,
                     selection=["g"], depths={"g": 7.0})
    assert out is not None and out.exists()
    assert (tmp_path / "screen.png").stat().st_size > 0


def test_region_map_screen_has_no_colorbar_title_or_member_outline(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # screen.png must show only the coloured fills + the thick black bounding-box locator: no
    # colorbar (a second Axes matplotlib appends via fig.colorbar), no title, and no per-member
    # outline collection (the old `edgecolor="black"` trace over every member's own fill, which
    # occluded the metric colours it was meant to show). Intercept save_render to inspect the
    # Figure/Axes before it's written+closed.
    import reblock.emit as emit
    captured: dict[str, Figure] = {}

    def spy(fig: Figure, path: str | Path) -> None:
        name = Path(path).name
        if name not in captured:
            captured[name] = fig
        _real_save_render(fig, path)

    monkeypatch.setattr(emit, "save_render", spy)
    region_map(_source_with_neighbour_and_points(), [["g"]], [["g"]], tmp_path,
              selection=["g"], depths={"g": 7.0})

    ax_s = captured["screen.png"].axes[0]
    assert ax_s.get_title() == ""
    assert len(captured["screen.png"].axes) == 1     # no colorbar axes appended
    # blanked ("neighbour") + flagged ("g") = 2 fill collections; a 3rd would be the removed
    # per-member outline.
    assert len(ax_s.collections) == 2
    assert len(ax_s.patches) == 1                     # only the bounding-box Rectangle survives

    # region.png is now bare too: no member-count/seed-count title.
    assert captured["region.png"].axes[0].get_title() == ""


def test_region_map_without_depths_still_writes(tmp_path: Path) -> None:
    # depths=None (a non-DenseCompact screen) falls back to a flat located map -- no proxy coloring.
    out = region_map(_source_with_neighbour_and_points(), [["g"]], [["g"]], tmp_path)
    assert out is not None and out.exists()


def test_region_map_scores_unflagged_members_by_metric_not_depth(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A member the screen didn't flag ("neighbour", not in `depths`) is coloured by the SAME
    # metric's fine, not peeled for depth. With a geometry-only metric (needs_peel=False),
    # block_depths is never called for the fallback -- the region map stays single-metric.
    import reblock.region as region_mod
    from reblock.metric import Compactness
    calls = {"n": 0}
    real = region_mod.block_depths

    def spy(*a: object, **k: object) -> object:
        calls["n"] += 1
        return real(*a, **k)   # type: ignore[arg-type]

    monkeypatch.setattr(region_mod, "block_depths", spy)
    blocks = gpd.GeoDataFrame(
        {"block_id": ["g", "neighbour"], "building_count": [5.0, 9.0]},
        geometry=[Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),
                  Polygon([(4, 0), (6, 0), (6, 3), (4, 3)])], crs=UTM)
    points = gpd.GeoDataFrame(geometry=[Point(1, 1), Point(5, 1.5)], crs=UTM)
    out = region_map(_FakeSource(blocks, points), [["g", "neighbour"]], [["g"]], tmp_path,
                     selection=["g"], depths={"g": 0.5}, metric=Compactness())
    assert out is not None and out.exists()
    assert calls["n"] == 0     # geometry-only metric -> no depth peel for the unflagged member


def _permeability_and_displacement_curves():
    from reblock.budget import Curve
    from reblock.compare import MethodCurve
    return [
        MethodCurve("clearance", "B", "permeability",
                   Curve([0.0, 100.0], [0.0, 0.8]), 0.1, 0.2),
        MethodCurve("clearance", "B", "displacement", Curve([0.0, 100.0], [0.0, 0.42]), 0.1, 0.2),
    ]


def test_compare_report_emits_one_permeability_vs_displacement_frontier(tmp_path):
    from reblock.emit import compare_report
    compare_report(_permeability_and_displacement_curves(), tmp_path, method_order=["clearance"])
    assert (tmp_path / "frontier_permeability.csv").exists()
    text = (tmp_path / "frontier_permeability.csv").read_text()
    assert "displacement" in text and "permeability" in text
    assert (tmp_path / "frontier_B.png").exists()
    # the old three-curve/displacement-table surface is gone -- ONE frontier now, not several:
    assert not (tmp_path / "frontier_external_connectivity.csv").exists()
    assert not (tmp_path / "frontier_internal_connectivity.csv").exists()
    assert not (tmp_path / "displacement_vs_length.csv").exists()
    assert not (tmp_path / "displacement_table.csv").exists()
    assert not (tmp_path / "displacement_B.png").exists()
    assert not list(tmp_path.glob("curve_*_B.png"))
    assert not list(tmp_path.glob("tradeoff_table_*.csv"))


def test_compare_report_frontier_has_no_title_and_percent_axes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # x/y axes are "displacement"/"permeability" (both [0,1) fractions, PercentFormatter'd), and
    # the curve carries no title -- no block_id, no method-comparison boilerplate.
    import reblock.emit as emit
    captured: dict[str, Figure] = {}

    def spy(fig: Figure, path: str | Path) -> None:
        captured["fig"] = fig
        _real_save_render(fig, path)

    monkeypatch.setattr(emit, "save_render", spy)
    emit.compare_report(_permeability_and_displacement_curves(), tmp_path,
                        method_order=["clearance"])

    ax = captured["fig"].axes[0]
    assert ax.get_title() == ""
    assert ax.get_xlabel() == "displacement"
    assert ax.get_ylabel() == "permeability"
    assert "B" not in ax.get_title()
    assert tuple(captured["fig"].get_size_inches()) == (12.0, 9.0)


def test_compare_report_without_permeability_rows_writes_nothing(tmp_path: Path) -> None:
    # A results list with only "displacement" (no benefit metric) has no frontier to plot.
    from reblock.budget import Curve
    from reblock.compare import MethodCurve
    from reblock.emit import compare_report
    curves = [MethodCurve("clearance", "B", "displacement", Curve([0.0, 100.0], [0.0, 0.42]))]
    compare_report(curves, tmp_path, method_order=["clearance"])
    assert list(tmp_path.iterdir()) == []


def test_compare_report_has_no_cost_param():
    import inspect

    from reblock.emit import compare_report
    assert "cost" not in inspect.signature(compare_report).parameters
