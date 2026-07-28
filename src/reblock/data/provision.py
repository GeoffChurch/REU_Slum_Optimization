"""On-demand cached full-city data: download the kblock blocks + Open Buildings for a
city (retaining building_count/block_area_m2), cache under ~/.cache/reblock, return paths.
Plain file cache (check-if-exists), not joblib. The large data is never committed.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

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


def tiles_for(shortlist: gpd.GeoDataFrame, tiles: gpd.GeoDataFrame) -> list[str]:
    """Open Buildings point-tile URLs whose S2 cell intersects any shortlist block.

    Measured: tiles.geojson has 333 features, 20 of which cover ZAF+KEN (3.78 GB gzipped as
    points; the polygon variants are 14.09 GB). The existing single-centroid-tile lookup is
    correct only for a bbox smaller than one cell.
    """
    joined = gpd.sjoin(tiles.to_crs(shortlist.crs or "EPSG:4326"), shortlist,
                       how="inner", predicate="intersects")
    return sorted(set(joined["tile_url"]))


def filter_to_shortlist(
    points: gpd.GeoDataFrame, shortlist: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Keep only building points falling inside a shortlist block POLYGON.

    Load-bearing: filtering to the shortlist's bounding rectangle would, for a ZAF+KEN shortlist,
    retain essentially every Open Buildings row in the download and make the targeted provisioning
    country-wide by accident. De-duplicate on the left index so a point inside multiple shortlist
    polygons is returned at most once.
    """
    joined = gpd.sjoin(points, shortlist, how="inner", predicate="within")
    # De-duplicate: keep first occurrence of each left index (point)
    deduped = joined[~joined.index.duplicated(keep="first")]
    return cast(gpd.GeoDataFrame, deduped.drop(columns=["index_right"]))
