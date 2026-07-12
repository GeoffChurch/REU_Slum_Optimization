from collections.abc import Iterator
from pathlib import Path

import geopandas as gpd
import joblib
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import Point, box

import reblock.derive_graph as dg
from reblock.contracts import Block
from reblock.data.kblock import KblockSource
from reblock.derivations import access_before
from reblock.screen.dense_compact import DenseCompactScreen, _cheap_survivors

ROOT = Path(__file__).resolve().parents[1]
CT_BLOCKS = str(ROOT / "data" / "kblock" / "blocks_capetown_sample.parquet")
CT_BLD = str(ROOT / "data" / "kblock" / "buildings_capetown_sample.parquet")
UTM = CRS.from_epsg(32734)     # Cape Town UTM: valid metric coords, KblockSource reprojects cleanly
EX, NY = 3.0e5, 6.25e6          # a realistic easting/northing base


def _write_synth(tmp: Path) -> tuple[str, str]:
    # A: dense + DEEP (5x5 grid in a 50x50 m block -> ring depths 1/2/3, mean depth 1.4);
    # B: dense but SHALLOW (30 buildings in two rows of a 30x2 m block -> all front a
    #    street, mean 1.0);
    # C: SPARSE (density 22/ha -> fails the cheap gate outright).
    a = box(EX, NY, EX + 50, NY + 50)
    b = box(EX + 70, NY, EX + 100, NY + 2)
    c = box(EX + 120, NY, EX + 150, NY + 30)
    blocks = gpd.GeoDataFrame({
        "block_id": ["A", "B", "C"], "k_complexity": [3.0, 2.0, 1.0],
        "building_count": [25, 30, 2], "block_area_m2": [2500.0, 60.0, 900.0],
    }, geometry=[a, b, c], crs=UTM)
    pts = [Point(EX + 5 + 10 * i, NY + 5 + 10 * j) for i in range(5) for j in range(5)]  # A: 5x5
    pts += [Point(EX + 71 + 2 * i, NY + row) for i in range(15) for row in (0.5, 1.5)]   # B: 2 rows
    pts += [Point(EX + 125, NY + 5), Point(EX + 140, NY + 20)]                            # C: 2
    bld = gpd.GeoDataFrame(geometry=pts, crs=UTM)
    bp, dp = tmp / "b.parquet", tmp / "d.parquet"
    blocks.to_parquet(bp)
    bld.to_parquet(dp)
    return str(bp), str(dp)


def test_cheap_survivors_gate(tmp_path: Path) -> None:
    bp, _ = _write_synth(tmp_path)
    # density/ha: A=25/(2500/1e4)=100, B=30/(60/1e4)=5000, C=2/(900/1e4)=22
    assert _cheap_survivors(gpd.read_parquet(bp), density_min=50.0, k_min=None) == ["A", "B"]


def test_select_two_tier_drops_shallow(tmp_path: Path) -> None:
    bp, dp = _write_synth(tmp_path)
    # cheap keeps A,B; fine gate mean_depth_min=1.2 keeps A (deep, ~1.4), drops B (strip, ~1.0)
    s = DenseCompactScreen(density_min=50.0, mean_depth_min=1.2, min_buildings=10)
    src = KblockSource(bp, dp, region_id="test", min_buildings=10)
    assert s.select(src) == ["A"]


def test_select_flags_flagship_on_real_fixture() -> None:
    # density_min=35.0 clears the flagship's real column-based density (~35.6/ha over
    # the free building_count/block_area_m2 columns). Returned order is now max-access-
    # depth descending (not alphabetical), so assert membership, not sort.
    s = DenseCompactScreen(density_min=35.0, mean_depth_min=1.3)
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown")
    ids = s.select(src)
    assert ids is not None and "ZAF.9.3.1_1_44882" in ids


def test_max_depth_min_gate_drops_blocks_without_a_deep_parcel(tmp_path: Path) -> None:
    bp, dp = _write_synth(tmp_path)
    # A: max-depth 3; B: max-depth 1. mean_depth_min=1.0 passes BOTH on the mean gate,
    # so max_depth_min=3 is the deciding gate -> only A (has a parcel at depth 3) survives.
    s = DenseCompactScreen(density_min=50.0, mean_depth_min=1.0,
                           max_depth_min=3.0, min_buildings=10)
    src = KblockSource(bp, dp, region_id="test", min_buildings=10)
    assert s.select(src) == ["A"]


def _write_sort_fixture(tmp: Path) -> tuple[str, str]:
    # "aaa": shallow — buildings all fronting the block edge (two rows, max-depth 1).
    # "zzz": deep — a 5x5 grid in a compact block (ring depths 1/2/3 -> max-depth 3).
    shallow = box(EX, NY, EX + 30, NY + 2)
    deep = box(EX + 60, NY, EX + 110, NY + 50)
    blocks = gpd.GeoDataFrame({
        "block_id": ["aaa", "zzz"], "k_complexity": [2.0, 3.0],
        "building_count": [15, 25], "block_area_m2": [60.0, 2500.0],
    }, geometry=[shallow, deep], crs=UTM)
    pts = [Point(EX + 1 + 2 * i, NY + row)                            # aaa: 2 rows
           for i in range(15) for row in (0.5, 1.5)]
    pts += [Point(EX + 65 + 10 * i, NY + 5 + 10 * j)                  # zzz: 5x5
            for i in range(5) for j in range(5)]
    bld = gpd.GeoDataFrame(geometry=pts, crs=UTM)
    bp, dp = tmp / "b.parquet", tmp / "d.parquet"
    blocks.to_parquet(bp)
    bld.to_parquet(dp)
    return str(bp), str(dp)


def test_select_ranks_by_max_depth_descending(tmp_path: Path) -> None:
    bp, dp = _write_sort_fixture(tmp_path)
    s = DenseCompactScreen(density_min=50.0, mean_depth_min=1.0, min_buildings=10)
    src = KblockSource(bp, dp, region_id="test", min_buildings=10)
    # deep "zzz" (max-depth 3) outranks shallow "aaa" (max-depth 1) -> reverse-alphabetical,
    # which alphabetical sorted() could never produce -> proves the severity sort.
    assert s.select(src) == ["zzz", "aaa"]


@pytest.fixture
def _isolate(tmp_path_factory: pytest.TempPathFactory,
             monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # Repoint derive_graph's L2 disk cache to a fresh dir + clear L1, so this test's
    # cache-HIT assertion is not silently served by an entry a sibling test (same
    # synthetic source content -> same source_hash + gates -> same key) already wrote.
    loc = tmp_path_factory.mktemp("l2")
    monkeypatch.setattr(dg, "memory", joblib.Memory(location=str(loc), verbose=0))
    monkeypatch.setattr(dg, "_l2", dg.memory.cache(dg._l2_impl, ignore=["fn", "inputs"]))
    dg.clear_l1()
    yield
    dg.clear_l1()


def test_select_result_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 _isolate: None) -> None:
    # The whole ranked selection is memoized: a rerun with the same source + gates must NOT
    # re-walk the survivors (no further access_before calls) and must return the same ids.
    bp, dp = _write_synth(tmp_path)
    box_ = {"n": 0}

    def spy(blk: Block) -> pd.Series:
        box_["n"] += 1
        return access_before(blk)
    monkeypatch.setattr("reblock.screen.dense_compact.access_before", spy)

    s = DenseCompactScreen(density_min=50.0, mean_depth_min=1.2, min_buildings=10)
    src = KblockSource(bp, dp, region_id="test", min_buildings=10)
    first = s.select(src)
    after_first = box_["n"]
    second = s.select(src)                 # (source_hash + gates)-keyed cache hit
    assert first == second == ["A"]
    assert after_first > 0                  # first run walked the survivors (computed depths)
    assert box_["n"] == after_first         # second run added zero -> whole selection cached
