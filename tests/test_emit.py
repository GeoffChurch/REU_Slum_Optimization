from pathlib import Path
from typing import cast

import geopandas as gpd
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import Point, Polygon

from reblock.contracts import BBox, Block, Metrics, Proposal, Region, Result
from reblock.emit import RenderConfig, _member_ids, region_map, render_results

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


def test_region_map_draws_member_and_context_points(tmp_path: Path) -> None:
    # region_map now takes the typed Source (no pre-read GeoDataFrame): it reads all
    # candidate outlines itself and windows the building-points query to the region's frame.
    out = region_map(_source_with_neighbour_and_points(), [["g"]], [["g"]], tmp_path)
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_region_map_guards_empty_building_points(tmp_path: Path) -> None:
    out = region_map(_empty_points_source(), [["g"]], [["g"]], tmp_path)
    assert out is not None and out.exists() and out.stat().st_size > 0
