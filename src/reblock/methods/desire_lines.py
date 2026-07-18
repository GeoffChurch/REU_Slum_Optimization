"""Desire-line sources for the osm_footpaths reblocker: pull the real informal circulation
network (worn footpaths) for a region instead of synthesizing one. `DesireLineSource` is the
pluggable seam (like a routing Substrate); `OSMDesireLines` (Phase 1) reads OpenStreetMap via
Overpass. A later imagery detector becomes another DesireLineSource behind the same interface.
"""
from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
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
    `geometry: [{lat, lon}, ...]`; ways with < 2 nodes are dropped, as are ways with
    `geometry: null` (nodes weren't downloaded). Coordinates are (lon, lat) = (x, y) in
    EPSG:4326, then reprojected to `target_crs`."""
    lines: list[LineString] = []
    for el in payload.get("elements", []):
        if el.get("type") != "way":
            continue
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry") or []]
        if len(coords) < 2:
            continue
        lines.append(LineString(coords))
    gdf = gpd.GeoDataFrame(geometry=lines, crs=CRS.from_epsg(4326))
    return gdf.to_crs(target_crs)


_USER_AGENT = "reblock-osm-footpaths/0.1 (informal-settlement research)"
_DEFAULT_TAGS = ("path", "footway", "track", "steps", "pedestrian", "living_street")


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "reblock" / "osm"


@dataclass
class OSMDesireLines:
    """A DesireLineSource backed by OpenStreetMap. Fetch precedence: a committed `snapshot`
    GeoJSON (byte-stable, no network) -> a disk cache under `cache_dir` (default
    ~/.cache/reblock/osm; offline after first fetch) -> a live Overpass query. `identity` is None
    when live (uncacheable, so the derivation cache bypasses and never serves stale OSM), and a
    stable tuple keyed on the snapshot's content hash when a snapshot is pinned."""

    tags: Sequence[str] = _DEFAULT_TAGS
    endpoint: str = "https://overpass-api.de/api/interpreter"
    cache_dir: str | None = None
    snapshot: str | None = None
    timeout_s: float = 60.0            # client read timeout; raise for a large region's bbox

    @property
    def identity(self) -> Hashable:
        if self.snapshot is None:
            return None                                   # live: uncacheable (data can drift)
        digest = hashlib.sha256(Path(self.snapshot).read_bytes()).hexdigest()[:16]
        return ("osm", tuple(sorted(self.tags)), digest)   # sorted: tag order is not meaningful

    def _cache_path(self, bbox_wgs84: tuple[float, float, float, float]) -> Path:
        root = Path(self.cache_dir) if self.cache_dir else _default_cache_dir()
        key = f"{'|'.join(sorted(self.tags))}@{','.join(f'{c:.5f}' for c in bbox_wgs84)}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return root / f"{digest}.geojson"

    def _fetch(self, query: str) -> dict[str, Any]:
        """POST the Overpass query and return the parsed JSON. A real User-Agent is required
        (default UA -> HTTP 406)."""
        data = urllib.parse.urlencode({"data": query}).encode()
        req = urllib.request.Request(
            self.endpoint, data=data, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(  # noqa: S310 (trusted endpoint)
                req, timeout=self.timeout_s) as resp:
            payload: dict[str, Any] = json.loads(resp.read().decode())
            return payload

    def desire_lines(
        self, bbox_wgs84: tuple[float, float, float, float], crs: CRS
    ) -> gpd.GeoDataFrame:
        if self.snapshot is not None:
            return gpd.read_file(self.snapshot).to_crs(crs)
        cache_path = self._cache_path(bbox_wgs84)
        if cache_path.exists():
            return gpd.read_file(cache_path).to_crs(crs)
        payload = self._fetch(_overpass_query(bbox_wgs84, self.tags))
        gdf_4326 = _parse_overpass_geom(payload, CRS.from_epsg(4326))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        gdf_4326.to_file(cache_path, driver="GeoJSON")
        return gdf_4326.to_crs(crs)
