"""Regions whose only exit is their own boundary (owner 2026-10-10: "just the boundary of the
region would be the sink, so we still wouldn't use roads at all").

A kblock's exits are its whole outline (kblock.py: streets = every ring), so kblocks are
independent problems. A region here is one polygon instead: the face of an OSM road network that
holds a kblock's interior point. Every building anchored inside it is a candidate, its outer ring
is the only street, and the roads inside it are open ground like any other.

An id `<kblock id>@<network>` names one (e.g. ZAF.9.3.1_1_5810@major); common.build_blocks and
common.write_bank resolve it here, once, into a FaceRegion. The face comes from the country's
Geofabrik extract (reblock.data.pools.pbf_path), polygonized inside a window around the seed
that doubles until the face no longer touches it; a face the network does not close within
MAX_WINDOW_M is an error. The buildings are the kblock source's, from every kblock whose interior
point the face holds, at any building count (the source's own floor dropped), kept where their
published anchor is inside the face, each with its kblock (`member`) and its row there
(`member_row`).

Its OSM features (gap_exits.Osm, attrs["osm"]: the extract's ways, railway nodes and water areas
meeting the face, their tags read once here, `osm_of`) are what the region model's barriers and
street boost read (gap_exits: NOTES, "Exits as leaks: the region model on the complex
(5810@major)").
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from gap_exits import Osm
from reblock.buildings import ANCHOR_COL
from reblock.contracts import Block
from reblock.data.pools import pbf_path
from reblock.region import shared_tier

FIRST_WINDOW_M = 5000.0
MAX_WINDOW_M = 40000.0
OSM_PAD_M = 100.0        # the OSM features are read in the face's box grown by this (m)


@dataclass(frozen=True)
class RoadNetwork:
    """The OSM highway classes whose ways bound a face (each with its `_link` ways)."""
    name: str
    classes: tuple[str, ...]


MAJOR = RoadNetwork("major", ("motorway", "trunk", "primary", "secondary", "tertiary"))
NETWORKS = (MAJOR,)


@dataclass(frozen=True)
class FaceRegion:
    seed: str              # the kblock whose interior point the face holds
    network: RoadNetwork

    @property
    def block_id(self) -> str:
        return f"{self.seed}@{self.network.name}"


def parse(block_id: str) -> FaceRegion | None:
    """The region an id names, or None for a plain kblock id; an unknown network raises."""
    if "@" not in block_id:
        return None
    seed, name = block_id.split("@")
    by_name = {n.name: n for n in NETWORKS}
    if name not in by_name:
        raise ValueError(f"{block_id}: unknown road network {name!r}; known: {sorted(by_name)}")
    return FaceRegion(seed, by_name[name])


def face(seed: Polygon, network: RoadNetwork, crs, iso: str) -> Polygon:
    """The smallest face of `network` holding `seed`'s interior point."""
    p = seed.representative_point()
    r = FIRST_WINDOW_M
    while r <= MAX_WINDOW_M:
        win = p.buffer(r)
        bbox = tuple(gpd.GeoSeries([win.buffer(100)], crs=crs).to_crs(4326).total_bounds)
        ways = pyogrio.read_dataframe(pbf_path(iso), layer="lines", bbox=bbox,
                                      columns=["highway"]).to_crs(crs)
        ways = ways[ways.highway.fillna("").str.removesuffix("_link").isin(network.classes)]
        # unclipped ways crossing the window's ring node with it, so its faces close
        held = [f for f in polygonize(unary_union([*ways.geometry, win.exterior]))
                if f.contains(p)]
        if held:
            f = min(held, key=lambda x: x.area)
            if f.boundary.distance(win.exterior) > 1.0:
                return Polygon(f.exterior)       # a face has no holes; drop any sliver ring
        r *= 2
    raise ValueError(f"the {network.name} network does not close a face around this seed "
                     f"within {MAX_WINDOW_M:g} m")


_TAG = re.compile(r'"((?:[^"\\]|\\.)*)"=>"((?:[^"\\]|\\.)*)"')


def _tags(other: pd.Series) -> list[dict[str, str]]:
    """GDAL's `other_tags` (an hstore string, or None) per feature, as a dict."""
    return [dict(_TAG.findall(t)) if isinstance(t, str) else {} for t in other]


def _tag(tags: list[dict[str, str]], key: str) -> list[str]:
    return [t[key] if key in t else "" for t in tags]


def osm_of(f: Polygon, crs: Any, iso: str) -> Osm:
    """The OSM features of the country's extract meeting the face `f` (module docstring)."""
    bbox = tuple(gpd.GeoSeries([shapely.box(*f.buffer(OSM_PAD_M).bounds)], crs=crs)
                 .to_crs(4326).total_bounds)
    path = pbf_path(iso)

    def read(layer: str, columns: list[str]) -> gpd.GeoDataFrame:
        g = pyogrio.read_dataframe(path, layer=layer, bbox=bbox, columns=columns).to_crs(crs)
        return cast(gpd.GeoDataFrame,
                    g[shapely.intersects(g.geometry.to_numpy(), f)].reset_index(drop=True))

    w = read("lines", ["highway", "railway", "waterway", "other_tags"])
    wt = _tags(w.other_tags)
    ways = gpd.GeoDataFrame(
        {"highway": w.highway.fillna("").to_numpy(), "railway": w.railway.fillna("").to_numpy(),
         "waterway": w.waterway.fillna("").to_numpy(),
         "bridge": [b not in ("", "no") for b in _tag(wt, "bridge")],
         "tunnel": [b not in ("", "no") for b in _tag(wt, "tunnel")]},
        geometry=w.geometry.to_numpy(), crs=crs)
    p = read("points", ["other_tags"])
    nodes = gpd.GeoDataFrame({"railway": _tag(_tags(p.other_tags), "railway")},
                             geometry=p.geometry.to_numpy(), crs=crs)
    a = read("multipolygons", ["natural", "landuse", "other_tags"])
    areas = gpd.GeoDataFrame(
        {"natural": a.natural.fillna("").to_numpy(), "landuse": a.landuse.fillna("").to_numpy(),
         "waterway": _tag(_tags(a.other_tags), "waterway")},
        geometry=a.geometry.to_numpy(), crs=crs)
    return Osm(ways=ways, nodes=nodes, areas=areas)


def build(region: FaceRegion, blocks_path, members_of: Callable[[list[str]], list[Block]]
          ) -> Block:
    """The region's Block. `blocks_path`: the kblock source's blocks parquet; `members_of`: the
    source's blocks for these ids at any building count."""
    allb = gpd.read_parquet(blocks_path, columns=["block_id", "geometry"])
    allb["block_id"] = allb.block_id.astype(str)
    crs = allb.estimate_utm_crs()             # the source's own: the full frame's
    allb = allb.to_crs(crs)
    [seed] = allb.geometry[allb.block_id == region.seed]
    f = face(seed, region.network, crs, region.seed.split(".", 1)[0])
    pts = allb.geometry.representative_point()
    members = members_of(sorted(allb.block_id[shapely.contains(f, pts.to_numpy())]))
    # each building's kblock and its row there (the per-block model's home: region_leaks)
    bld = pd.concat([m.building_geometries.assign(member=m.block_id,
                                                   member_row=range(len(m.building_geometries)))
                     for m in members], ignore_index=True)
    anchors = gpd.GeoSeries(bld[ANCHOR_COL].to_numpy(), crs=crs)
    bld = gpd.GeoDataFrame(bld[shapely.contains(f, anchors.to_numpy())].reset_index(drop=True),
                           geometry="geometry", crs=crs)
    parcels = gpd.GeoDataFrame(pd.concat([m.parcels for m in members], ignore_index=True),
                               geometry="geometry", crs=crs)
    parcels["parcel_id"] = range(len(parcels))
    return Block(block_id=region.block_id, crs=crs, boundary=f, parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString(f.exterior.coords)], crs=crs),
                 source_content_hash=None, building_geometries=bld,
                 building_tier=shared_tier(members),
                 attrs={"members": len(members), "network": region.network.name,
                        "osm": osm_of(f, crs, region.seed.split(".", 1)[0])})
