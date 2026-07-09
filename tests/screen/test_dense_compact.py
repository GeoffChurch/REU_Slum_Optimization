from pathlib import Path

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import Point, box

from reblock.data.kblock import KblockSource
from reblock.screen.dense_compact import DenseCompactScreen

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
    s = DenseCompactScreen(density_min=50.0, min_buildings=10)
    # density/ha: A=25/(2500/1e4)=100, B=30/(60/1e4)=5000, C=2/(900/1e4)=22
    assert s._cheap_survivors(gpd.read_parquet(bp)) == ["A", "B"]   # C (22) fails; sorted


def test_select_two_tier_drops_shallow(tmp_path: Path) -> None:
    bp, dp = _write_synth(tmp_path)
    # cheap keeps A,B; fine gate mean_depth_min=1.2 keeps A (deep, ~1.4), drops B (strip, ~1.0)
    s = DenseCompactScreen(density_min=50.0, mean_depth_min=1.2, min_buildings=10)
    src = KblockSource(bp, dp, region_id="test", min_buildings=10)
    assert s.select(src) == ["A"]


def test_select_flags_flagship_on_real_fixture() -> None:
    # density_min=35.0 is the smallest round threshold that clears the flagship's real
    # column-based density (~35.6/ha over the free building_count/block_area_m2 columns;
    # see git history / PROVENANCE for why it is not the ~108/ha spatial-join figure).
    s = DenseCompactScreen(density_min=35.0, mean_depth_min=1.3)
    src = KblockSource(CT_BLOCKS, CT_BLD, region_id="capetown")
    ids = s.select(src)
    assert ids is not None and "ZAF.9.3.1_1_44882" in ids and ids == sorted(ids)
