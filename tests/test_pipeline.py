from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Source
from reblock.pipeline import RunOutput, _reachable_blocks, _region_depth_map

_UTM = CRS.from_epsg(32643)


def _chain_gdf() -> gpd.GeoDataFrame:
    # A 4-block chain s-a-b-c of unit squares, 10 buildings each (adjacent left-to-right).
    polys = {"s": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
             "a": Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
             "b": Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
             "c": Polygon([(3, 0), (4, 0), (4, 1), (3, 1)])}
    return gpd.GeoDataFrame({"block_id": list(polys), "building_count": [10.0, 10.0, 10.0, 10.0]},
                            geometry=list(polys.values()), crs=_UTM)


def test_runoutput_holds_selection_and_results() -> None:
    out = RunOutput(selection=["a", "b", "c"], results=[])
    assert out.selection == ["a", "b", "c"] and out.results == []


def test_reachable_blocks_bfs_bounds_by_building_count() -> None:
    # BFS from the seed accumulates building_count to the bound: bound=25 covers s (10) + a (20) + b
    # (30, which crosses 25 so BFS stops), but not the farther c. So the batched peel stays local.
    out = set(_reachable_blocks(_chain_gdf(), [["s"]], 25.0))
    assert {"s", "a"} <= out          # the seed and its near neighbourhood are covered
    assert "c" not in out             # a block beyond the bound is left un-peeled (defaults 0.0)


def test_reachable_blocks_covers_whole_component_with_generous_bound() -> None:
    # A generous bound reaches the whole connected chain -- the real use (~3x the growth budget).
    assert set(_reachable_blocks(_chain_gdf(), [["s"]], 10_000.0)) == {"s", "a", "b", "c"}


def test_region_depth_map_empty_for_non_peelable_source() -> None:
    # No blocks_path -> not peel-capable -> {} (the builder falls back to its proxy).
    class _Bare:
        pass

    assert _region_depth_map(cast(Source, _Bare()), _chain_gdf(), [["s"]], 100.0) == {}
