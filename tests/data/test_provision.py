import shutil
from pathlib import Path

from reblock.data.provision import cached_kblock_source, ensure_city_data

ROOT = Path(__file__).resolve().parents[1]
CT_BLOCKS = ROOT / "data" / "kblock" / "blocks_capetown_sample.parquet"
CT_BLD = ROOT / "data" / "kblock" / "buildings_capetown_sample.parquet"


def _seed(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy(CT_BLOCKS, cache / "blocks_capetown_full.parquet")
    shutil.copy(CT_BLD, cache / "buildings_capetown_full.parquet")


def test_ensure_city_data_uses_cache_no_download(tmp_path: Path) -> None:
    _seed(tmp_path)
    bp, dp = ensure_city_data("capetown", cache_dir=tmp_path)   # must NOT hit the network
    assert bp.exists() and dp.exists()
    assert bp == tmp_path / "blocks_capetown_full.parquet"


def test_cached_kblock_source_builds_from_cache(tmp_path: Path) -> None:
    _seed(tmp_path)
    src = cached_kblock_source("capetown", block_ids=["ZAF.9.3.1_1_44882"], cache_dir=tmp_path)
    blocks = list(src.region().blocks)
    assert [b.block_id for b in blocks] == ["ZAF.9.3.1_1_44882"]
