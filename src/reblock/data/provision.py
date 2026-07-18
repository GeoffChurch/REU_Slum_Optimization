"""On-demand cached full-city data: download the kblock blocks + Open Buildings for a
city (retaining building_count/block_area_m2), cache under ~/.cache/reblock, return paths.
Plain file cache (check-if-exists), not joblib. The large data is never committed.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd

from reblock.data.kblock import KblockSource
from scripts.fetch_kblock_fixtures import (
    CT_BBOX,
    download_capetown_buildings,
    download_dataverse_blocks,
)

DEFAULT_CACHE = Path.home() / ".cache" / "reblock"
# Central Nairobi metro (lon_min, lat_min, lon_max, lat_max) -- covers the core city incl. the major
# informal settlements (Kibera, Mathare, Mukuru); clips the country-wide KEN geodata.
NAIROBI_BBOX = (36.75, -1.35, 36.95, -1.20)
_ISO3 = {"capetown": "ZAF", "nairobi": "KEN"}
_BBOX = {"capetown": CT_BBOX, "nairobi": NAIROBI_BBOX}
_BLOCK_COLS = ["block_id", "k_complexity", "building_count", "block_area_m2", "geometry"]


def ensure_city_data(city: str, *, cache_dir: Path = DEFAULT_CACHE) -> tuple[Path, Path]:
    if city not in _ISO3:
        raise ValueError(f"unknown city {city!r}; known: {sorted(_ISO3)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = cache_dir / f"blocks_{city}_full.parquet"
    buildings_path = cache_dir / f"buildings_{city}_full.parquet"
    if not blocks_path.exists():
        raw = cache_dir / f"{_ISO3[city]}_geodata.parquet"
        if not raw.exists():
            download_dataverse_blocks(_ISO3[city], raw)
        bbox = _BBOX[city]
        blocks = gpd.read_parquet(raw, columns=_BLOCK_COLS)
        blocks.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].reset_index(drop=True).to_parquet(blocks_path)
    if not buildings_path.exists():
        download_capetown_buildings(_BBOX[city], buildings_path)
    return blocks_path, buildings_path


def cached_kblock_source(city: str, *, block_ids: list[str] | None = None,
                         min_buildings: int = 10, cache_dir: Path = DEFAULT_CACHE) -> KblockSource:
    blocks_path, buildings_path = ensure_city_data(city, cache_dir=cache_dir)
    return KblockSource(blocks_path, buildings_path, region_id=city,
                        min_buildings=min_buildings, block_ids=block_ids)
