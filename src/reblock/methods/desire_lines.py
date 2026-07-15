"""Desire-line sources for the dream_come_true reblocker: pull the real informal circulation
network (worn footpaths) for a region instead of synthesizing one. `DesireLineSource` is the
pluggable seam (like a routing Substrate); `OSMDesireLines` (Phase 1) reads OpenStreetMap via
Overpass. A later imagery detector becomes another DesireLineSource behind the same interface.
"""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Protocol

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString


class DesireLineSource(Protocol):
    def desire_lines(
        self, bbox_wgs84: tuple[float, float, float, float], crs: CRS
    ) -> gpd.GeoDataFrame: ...
    @property
    def identity(self) -> Hashable: ...


def _overpass_query(bbox_wgs84: tuple[float, float, float, float], tags: Sequence[str]) -> str:
    """Overpass QL for every `highway` way of the given tag classes in the bbox. `bbox_wgs84` is
    (min_lon, min_lat, max_lon, max_lat) (geopandas total_bounds order); Overpass wants
    (south,west,north,east). Tags are `^(...)$`-anchored so `path` doesn't match `pathway`."""
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    tag_re = "|".join(tags)
    return (
        "[out:json][timeout:60];"
        f'way["highway"~"^({tag_re})$"]({min_lat},{min_lon},{max_lat},{max_lon});'
        "out geom;"
    )


def _parse_overpass_geom(payload: dict[str, Any], target_crs: CRS) -> gpd.GeoDataFrame:
    """Overpass `out geom` JSON -> a GeoDataFrame of LineStrings in `target_crs`. Each `way` carries
    `geometry: [{lat, lon}, ...]`; ways with < 2 nodes are dropped. Coordinates are (lon, lat) =
    (x, y) in EPSG:4326, then reprojected to `target_crs`."""
    lines: list[LineString] = []
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(coords) < 2:
            continue
        lines.append(LineString(coords))
    gdf = gpd.GeoDataFrame(geometry=lines, crs=CRS.from_epsg(4326))
    return gdf.to_crs(target_crs)
