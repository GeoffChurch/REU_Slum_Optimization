from typing import cast

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import Polygon

from reblock.contracts import ScoringScreen, Source
from reblock.data.counts import BuildingCount, KblockCount
from reblock.data.kblock import KblockSource
from reblock.metric import BlockMetric
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
    out = RunOutput(selection=["a", "b", "c"], results=[], regions=[], seed_groups=[])
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
    """A minimal `ScoringScreen` stand-in, mirroring DenseCompactScreen: it carries a metric."""

    def __init__(self, m: BlockMetric) -> None:
        self.metric = m
        self.counts: BuildingCount = KblockCount()

    def select(self, s: Source) -> list[str]:
        return []

    def selection_scores(self, s: Source) -> dict[str, float]:
        return {}


def _peelable() -> Source:
    # I/O-free at construction, so placeholder paths are fine: nothing here reads them.
    return KblockSource("blocks.parquet", "buildings.parquet", member_buildings=None)


def test_capabilities_are_answered_by_the_shipped_types() -> None:
    """The pipeline asks a screen whether it scores, and a region builder whether it grows, with
    `isinstance` against a Protocol -- which is structural, so renaming `metric` on
    DenseCompactScreen or `max_buildings` on a growing builder would quietly flip the answer to
    "no" (growth ranked by the proxy, no region-map colours) with nothing else failing."""
    from reblock.metric import AbsoluteGate, Depth
    from reblock.region import (
        ConvexHullRegionBuilder,
        DenseClusterRegionBuilder,
        GrowingRegionBuilder,
        IdentityRegionBuilder,
        ShapeStandardizingRegionBuilder,
    )
    from reblock.screen.dense_compact import DenseCompactScreen
    from reblock.screen.identity import IdentityScreen
    assert isinstance(DenseCompactScreen(Depth(), AbsoluteGate(2.0)), ScoringScreen)
    assert isinstance(_MetricScreen(Depth()), ScoringScreen)
    assert not isinstance(IdentityScreen(), ScoringScreen)
    assert isinstance(DenseClusterRegionBuilder(), GrowingRegionBuilder)
    assert isinstance(ShapeStandardizingRegionBuilder(), GrowingRegionBuilder)
    assert not isinstance(IdentityRegionBuilder(), GrowingRegionBuilder)
    assert not isinstance(ConvexHullRegionBuilder(), GrowingRegionBuilder)


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
    class _NoMetricScreen:
        def select(self, s):
            return []

    assert _region_score_map(_peelable(), _NoMetricScreen(), _chain_gdf(),
                             [["s"]], 100.0) == {}


def test_region_score_map_uses_metric_fine_and_skips_peel_when_geometry_only() -> None:
    # A density_compactness metric (needs_peel=False) -> _region_score_map must NOT call
    # block_depths; scores come from columns. A depth metric (needs_peel=True) -> block_depths
    # supplies the depth.
    import reblock.pipeline as pl
    from reblock.metric import Compactness, Density, Depth, Product

    calls = {"n": 0}
    real = pl.block_depths  # type: ignore[attr-defined]
    pl.block_depths = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1) or {})  # type: ignore
    try:
        gdf = _chain_gdf()
        pl._region_score_map(_peelable(), _MetricScreen(Product([Density(), Compactness()])),
                             gdf, [["s"]], 100.0)
        assert calls["n"] == 0        # geometry-only: no peel
        pl._region_score_map(_peelable(), _MetricScreen(Depth()), gdf, [["s"]], 100.0)
        assert calls["n"] == 1        # depth: one batched block_depths call
    finally:
        pl.block_depths = real        # type: ignore


def test_region_score_map_leaves_out_a_block_the_peel_could_not_build(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """`block_depths` omits a block it cannot build. That block has no depth, so it gets no score
    -- not `fine(0.0, ...)`, a depth nobody measured. Growth then ranks it with every other
    unscored candidate."""
    import reblock.pipeline as pl
    from reblock.metric import Depth
    monkeypatch.setattr(pl, "block_depths", lambda source, ids: {"s": 3.0, "a": 2.0})
    scores = _region_score_map(_peelable(), _MetricScreen(Depth()), _chain_gdf(), [["s"]], 1e4)
    assert scores == {"s": 3.0, "a": 2.0}


def test_reachable_blocks_expands_per_group_and_bounds_only_the_expansion() -> None:
    """`_reachable_blocks` must return a STRICT superset of its seeds. The neighbourhood is the
    whole point: `_region_score_map` peels exactly what this returns, and every growth candidate
    outside it is ranked by the cheap geometric proxy instead of its true peel depth.

    On the s-a-b-c chain (10 buildings each):

    `_seed_groups` wraps a screen selection as one singleton group per flagged block, so this is
    called with many groups. Seeding ONE BFS from their union meant the bound was compared against
    the whole selection's count -- ~413,806 buildings against 9,000 in the Cape Town run -- so the
    loop body never executed and this returned exactly the seed set.

    A seed's own count still counts toward its group's bound; that is deliberate and is pinned by
    `test_reachable_blocks_bfs_bounds_by_building_count` above, which is what caught an attempt to
    change it here.

    FAULT INJECTION: folding the per-group loop back into one pass over the union makes the two
    seeds (20 buildings) exceed the bound of 15 immediately, collapsing reach to {s, c}.
    """
    gdf = _chain_gdf()                       # s-a-b-c, adjacent left-to-right, 10 buildings each

    # Two groups at opposite ends. Their counts (20) already exceed the bound (15), which is
    # exactly the production shape -- and under the old arithmetic that alone killed expansion.
    reach = set(_reachable_blocks(gdf, [["s"], ["c"]], bound_buildings=15.0))
    assert reach > {"s", "c"}, "each group must expand from its own seed"
    assert reach == {"s", "a", "b", "c"}, reach

    # One group, same bound: s (10) is under 15, a (20) crosses it, so expansion stops at `a`.
    assert set(_reachable_blocks(gdf, [["s"]], bound_buildings=15.0)) == {"s", "a"}
