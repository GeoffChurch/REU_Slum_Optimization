"""Shared helpers for Source implementations under reblock.data."""
from __future__ import annotations

import geopandas as gpd

from reblock.contracts import BBox


def _window(gdf: gpd.GeoDataFrame, bbox: BBox | None) -> gpd.GeoDataFrame:
    """Return only the rows of `gdf` intersecting `bbox` (source-CRS `.cx` window),
    or `gdf` unchanged when `bbox` is None (the whole-metro / RegionBuilder case)."""
    if bbox is None:
        return gdf
    return gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]
