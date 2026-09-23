"""Shared helpers for Source implementations under reblock.data."""
from __future__ import annotations

from collections.abc import Sequence

import geopandas as gpd

from reblock.contracts import BBox


def _window(gdf: gpd.GeoDataFrame, bbox: BBox | None) -> gpd.GeoDataFrame:
    """Return only the rows of `gdf` intersecting `bbox` (source-CRS `.cx` window),
    or `gdf` unchanged when `bbox` is None (the whole-metro / RegionBuilder case)."""
    if bbox is None:
        return gdf
    return gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]


def _narrowed(region_id: str, current: tuple[str, ...] | None,
              requested: Sequence[str]) -> list[str]:
    """`requested` as the block_ids of a `restricted` copy of a source currently narrowed to
    `current` (None: not narrowed). An id outside `current` raises: the copy would otherwise yield
    a block the source it came from does not."""
    if current is not None:
        outside = set(requested) - set(current)
        if outside:
            raise ValueError(f"{region_id}: {sorted(outside)} are outside this source's block_ids")
    return list(requested)
