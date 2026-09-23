"""Desire lines: where people already walk, as a DEMAND FIELD a reblocker can route toward.

`DesireLineSource` is the pluggable seam (like a routing Substrate): for a block it returns a
`DesireField`, weighted groups of lines, where the demand at a point is the total weight of the
groups passing within a corridor of it (`demand_greedy.demand_edge_weights` reads it).

- A mapped network is ONE group at weight 1 -- the block's own footpaths (`mapped_field`), so its
  demand is the binary "inside the corridor or not". `OSMDesireLines` reads OpenStreetMap via
  Overpass; `reblock.data.osm_extract.PbfDesireLines` a local extract. Both are also
  `osm_footpaths.FootpathSource`s: that reblocker proposes the same lines as the roads themselves.
- `NoDesire` is no group at all: every edge costs its own length.
- A consensus of GW-transported donor networks is one group per donor, weighted by how good and how
  close each is (`reblock.transplant.consensus`), and is injected by configuration only, so that
  research code stays out of this module's import closure.
"""
from __future__ import annotations

import hashlib
import json
import math
import urllib.parse
import urllib.request
from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString

from reblock.contracts import Block
from reblock.methods.osm_footpaths import block_footpaths


@dataclass(frozen=True, eq=False)
class WeightedLines:
    """One group of desire lines: a point within the corridor of any of them gains `weight`."""

    lines: gpd.GeoDataFrame     # in the block's CRS
    weight: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.weight) and self.weight >= 0.0):
            raise ValueError(f"a desire-line weight must be finite and >= 0, got {self.weight}")


@dataclass(frozen=True, eq=False)
class DesireField:
    """Demand over a block: the total weight of the groups whose corridors hold a point. Groups
    are summed in order, and none are normalized here -- a source that wants demand in [0, 1]
    weights its groups to sum 1."""

    groups: tuple[WeightedLines, ...]

    @property
    def n_lines(self) -> int:
        return sum(len(g.lines) for g in self.groups)


def mapped_field(lines: gpd.GeoDataFrame) -> DesireField:
    """A mapped network as a field: one group, weight 1."""
    return DesireField(groups=(WeightedLines(lines=lines, weight=1.0),))


@runtime_checkable
class DesireLineSource(Protocol):
    def desire_field(self, block: Block) -> DesireField: ...
    @property
    def identity(self) -> Hashable | None: ...


@dataclass(frozen=True)
class NoDesire:
    """No desire lines at all: a uniform field, under which `demand_greedy` reduces to a pure
    shortest-path drainage tree -- the honest ablation for "how much is the prior worth?"."""

    @property
    def identity(self) -> Hashable:
        return ("no_desire",)

    def desire_field(self, block: Block) -> DesireField:
        del block
        return DesireField(groups=())


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
    EPSG:4326, then reprojected to `target_crs`.

    Raises on a failed query rather than returning what it got: whatever this returns is cached to
    disk as the region's footpaths, for good."""
    # Overpass reports a failed query -- a timeout, running out of memory -- as an ordinary
    # response whose optional `remark` says so, carrying whatever elements it had reached.
    remark = payload.get("remark")
    if remark is not None and "error" in remark:
        raise RuntimeError(f"Overpass query failed: {remark}")
    lines: list[LineString] = []
    # Indexed: every Overpass response carries `elements` and every element its `type`. A default
    # would read a payload that is not one as "no footpaths here".
    for el in payload["elements"]:
        if el["type"] != "way":
            continue
        coords = [(p["lon"], p["lat"]) for p in el.get("geometry") or []]
        if len(coords) < 2:
            continue
        lines.append(LineString(coords))
    gdf = gpd.GeoDataFrame(geometry=lines, crs=CRS.from_epsg(4326))
    return gdf.to_crs(target_crs)


_USER_AGENT = "reblock-osm-footpaths/0.1 (informal-settlement research)"


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "reblock" / "osm"


@dataclass
class OSMDesireLines:
    """A FootpathSource and DesireLineSource backed by OpenStreetMap. Fetch precedence: a committed
    `snapshot` GeoJSON (byte-stable, no network) -> a disk cache under `cache_dir` (default
    ~/.cache/reblock/osm; offline after first fetch) -> a live Overpass query. `identity` is None
    when live (uncacheable, so the derivation cache bypasses and never serves stale OSM), and a
    stable tuple keyed on the snapshot's content hash when a snapshot is pinned."""

    tags: Sequence[str]
    endpoint: str
    cache_dir: str | None
    snapshot: str | None
    timeout_s: float                   # client read timeout; raise for a large region's bbox

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

    def desire_field(self, block: Block) -> DesireField:
        return mapped_field(block_footpaths(self, block))

    def footpaths(
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
