from pathlib import Path

import geopandas as gpd

from scripts.crossblock_probe import enumerate_adjacent_pairs, probe_cluster

ROOT = Path(__file__).resolve().parents[1]
CT_BLOCKS = str(ROOT / "tests" / "data" / "kblock" / "blocks_capetown_sample.parquet")
CT_BLD = str(ROOT / "tests" / "data" / "kblock" / "buildings_capetown_sample.parquet")


def test_enumerate_adjacent_pairs_finds_neighbours() -> None:
    blocks = gpd.read_parquet(CT_BLOCKS, columns=["block_id", "geometry"])
    pairs = enumerate_adjacent_pairs(blocks.to_crs(blocks.estimate_utm_crs()))
    assert len(pairs) > 0
    assert all(a < b for a, b in pairs)              # canonical ordering, no dup/self pairs


def test_probe_cluster_returns_baseline_and_reference_metrics() -> None:
    # a real adjacent Cape Town pair including the flagship's neighbour
    row = probe_cluster(["ZAF.9.3.1_1_44882", "ZAF.9.3.1_1_44673"], CT_BLOCKS, CT_BLD)
    assert row["base_n_cross_block_streets"] == 0.0    # block-local baseline never crosses
    # probe_cluster's values are `float | str` (a "block_ids" identifier column rides along
    # with the float metrics), so a numeric comparison needs a float() narrowing for mypy.
    assert float(row["base_circuity"]) >= 1.0
    assert "ref_circuity" in row                        # the spine-merge reference was scored too
    assert float(row["base_added_road_length_per_parcel"]) > 0.0   # baseline genuinely has roads
    # (catches the empty-generator bug: a drained region.blocks silently yields an
    # empty-roads baseline, which would make this and the assertion below vacuously pass)
    assert float(row["ref_n_cross_block_streets"]) > 0.0   # spine-merge reference genuinely fires
