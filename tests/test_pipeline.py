from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import Source
from reblock.pipeline import RunOutput, _reachable_blocks, _region_score_map

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


class _MetricScreen:
    """A minimal Screen stand-in that just carries a metric, mirrors DenseCompactScreen."""

    def __init__(self, m):
        self.metric = m

    def select(self, s):
        return []


def test_region_score_map_empty_for_non_peelable_source() -> None:
    # No blocks_path -> not peel-capable -> {} (the builder falls back to its proxy).
    from reblock.metric import Depth

    class _Bare:
        pass

    assert _region_score_map(cast(Source, _Bare()), _MetricScreen(Depth()), _chain_gdf(),
                             [["s"]], 100.0) == {}


def test_region_score_map_empty_when_screen_has_no_metric() -> None:
    # IdentityScreen (no `.metric`) -> {} (the builder falls back to its proxy), even for a
    # peel-capable source.
    class _Src:
        blocks_path = "x"

    class _NoMetricScreen:
        def select(self, s):
            return []

    assert _region_score_map(cast(Source, _Src()), _NoMetricScreen(), _chain_gdf(),
                             [["s"]], 100.0) == {}


def test_region_score_map_uses_metric_fine_and_skips_peel_when_geometry_only() -> None:
    # A density_compactness metric (needs_peel=False) -> _region_score_map must NOT call
    # block_depths; scores come from columns. A depth metric (needs_peel=True) -> block_depths
    # supplies the depth.
    import reblock.pipeline as pl
    from reblock.metric import Compactness, Density, Depth, Product

    class _Screen:            # carries the metric, mirrors DenseCompactScreen
        def __init__(self, m): self.metric = m
        def select(self, s): return []

    class _Src:
        blocks_path = "x"

    calls = {"n": 0}
    real = pl.block_depths  # type: ignore[attr-defined]
    pl.block_depths = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or {})  # type: ignore
    try:
        gdf = _chain_gdf()
        pl._region_score_map(cast(Source, _Src()), _Screen(Product([Density(), Compactness()])),
                             gdf, [["s"]], 100.0)
        assert calls["n"] == 0        # geometry-only: no peel
        pl._region_score_map(cast(Source, _Src()), _Screen(Depth()), gdf, [["s"]], 100.0)
        assert calls["n"] == 1        # depth: one batched block_depths call
    finally:
        pl.block_depths = real        # type: ignore
