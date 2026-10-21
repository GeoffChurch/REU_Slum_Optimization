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
meeting the face, their tags read once, `with_osm`) are what the metric's barriers and street
boost read (gap_exits: NOTES, "Exits as leaks: the region model on the complex (5810@major)"); every
kblock carries its own since the consolidated pass (common.build_blocks), all from one cached read
of the extract over the kblock source's bounds (`Extract`).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import shapely
from pyproj import CRS
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

import gap_exits as gx
from gap_exits import Osm
from reblock.buildings import ANCHOR_COL
from reblock.contracts import Block
from reblock.data.pools import pbf_path
from reblock.region import shared_tier

FIRST_WINDOW_M = 5000.0
MAX_WINDOW_M = 40000.0
OSM_PAD_M = 100.0        # the OSM features are read in the face's box grown by this (m)
EXTRACT_PAD_M = 10000.0  # the cached extract covers the kblock source's blocks grown by this (m):
                         # every face of the major network about a kblock (checked at each use)
EXTRACT_REV = 3          # what a cached extract holds (its key): the layers, columns and rows of
                         # _read (2: the features no reader reads dropped; 3: the ways' surface)


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


class Extract(NamedTuple):
    """The country's extract's features (gap_exits.Osm's columns) whose envelope meets `box` (the
    kblock source's blocks' bounds grown by EXTRACT_PAD_M, in `crs`): read from the extract once
    and cached beside it (`of`), so a block's features are a query, not a 40 s scan of the
    extract."""
    box: tuple[float, float, float, float]
    osm: Osm

    @staticmethod
    def _dir(iso: str, crs: Any, box: tuple[float, float, float, float]) -> Path:
        """The cache's directory: keyed by the extract's file (name, size, mtime), the CRS, the box
        and what a cached extract holds (EXTRACT_REV)."""
        pbf = pbf_path(iso)
        st = pbf.stat()
        key = json.dumps(dict(pbf=pbf.name, size=st.st_size, mtime_ns=st.st_mtime_ns,
                              crs=CRS.from_user_input(crs).to_string(),
                              box=[round(v, 3) for v in box], rev=EXTRACT_REV), sort_keys=True)
        return pbf.parent / f"{pbf.name}.osm-{hashlib.sha256(key.encode()).hexdigest()[:16]}"

    @classmethod
    def of(cls, iso: str, crs: Any, box: tuple[float, float, float, float]) -> Extract:
        d = cls._dir(iso, crs, box)
        parts = ("ways", "nodes", "areas")
        if all((d / f"{k}.parquet").exists() for k in parts):
            return cls(box=box, osm=Osm(*(gpd.read_parquet(d / f"{k}.parquet") for k in parts)))
        osm = _read(iso, crs, box)
        d.mkdir(parents=True, exist_ok=True)
        for k, g in zip(parts, osm, strict=True):
            tmp = d / f"{k}.parquet.{os.getpid()}.tmp"
            g.to_parquet(tmp)
            os.replace(tmp, d / f"{k}.parquet")
        return cls(box=box, osm=osm)

    def meeting(self, f: BaseGeometry) -> Osm:
        """The features meeting `f` (a face, a block's polygon), in the extract's order; `f`'s
        bounds grown by OSM_PAD_M must lie inside the box (or a feature could be missing)."""
        x0, y0, x1, y1 = f.buffer(OSM_PAD_M).bounds
        bx0, by0, bx1, by1 = self.box
        if not (bx0 <= x0 and by0 <= y0 and x1 <= bx1 and y1 <= by1):
            raise ValueError(f"a face's bounds {f.bounds} lie outside the cached extract's box "
                             f"{self.box}: grow EXTRACT_PAD_M")

        def meet(g: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            return cast(gpd.GeoDataFrame,
                        g[shapely.intersects(g.geometry.to_numpy(), f)].reset_index(drop=True))
        return Osm(*(meet(g) for g in self.osm))


def _read(iso: str, crs: Any, box: tuple[float, float, float, float]) -> Osm:
    """The extract's features whose envelope meets `box` (in `crs`), their tags read once, less
    those no reader reads (the barriers and the boost select by these tags alone, gap_exits; a
    region's formal and deep homes by the highway tag, region_leaks): the lines with no highway,
    railway or waterway tag, the points with no railway tag, the areas that are not water (the
    province's and the city's boundaries among them, which met every block and filled its bank)."""
    bbox = tuple(gpd.GeoSeries([shapely.box(*box)], crs=crs).to_crs(4326).total_bounds)
    path = pbf_path(iso)

    def read(layer: str, columns: list[str]) -> gpd.GeoDataFrame:
        return pyogrio.read_dataframe(path, layer=layer, bbox=bbox, columns=columns).to_crs(crs)

    w = read("lines", ["highway", "railway", "waterway", "other_tags"])
    wt = _tags(w.other_tags)
    ways = gpd.GeoDataFrame(
        {"highway": w.highway.fillna("").to_numpy(), "railway": w.railway.fillna("").to_numpy(),
         "waterway": w.waterway.fillna("").to_numpy(), "surface": _tag(wt, "surface"),
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

    def keep(g: gpd.GeoDataFrame, mask: Any) -> gpd.GeoDataFrame:
        return cast(gpd.GeoDataFrame, g[np.asarray(mask)].reset_index(drop=True))
    return Osm(ways=keep(ways, (ways.highway != "") | (ways.railway != "")
                         | (ways.waterway != "")),
               nodes=keep(nodes, nodes.railway != ""),
               areas=keep(areas, areas.natural.isin(gx.WATER_NATURAL)
                          | areas.landuse.isin(gx.WATER_LANDUSE)
                          | areas.waterway.isin(gx.WATER_WATERWAY)))


def extract_of(blocks_path: Any, iso: str) -> Extract:
    """The cached extract covering every kblock of the source's `blocks_path` (its bounds grown by
    EXTRACT_PAD_M), in the source's frame (its blocks' estimated UTM CRS, as `build` takes it)."""
    allb = gpd.read_parquet(blocks_path, columns=["geometry"])
    crs = allb.estimate_utm_crs()
    x0, y0, x1, y1 = allb.to_crs(crs).total_bounds
    pad = EXTRACT_PAD_M
    return Extract.of(iso, crs, (float(x0 - pad), float(y0 - pad), float(x1 + pad),
                                 float(y1 + pad)))


def with_osm(b: Block, extract: Extract) -> Block:
    """`b` with its OSM features (attrs["osm"]: the extract's features meeting its polygon), as
    the metric's barriers and street boost read them (gap_exits.osm_in): every block, a kblock as
    a region (owner 2026-10-16: one metric for every block and region; NOTES, "The consolidated
    pass")."""
    if CRS.from_user_input(b.crs) != CRS.from_user_input(extract.osm.ways.crs):
        raise ValueError(f"{b.block_id}: its CRS {b.crs} is not the extract's")
    return dataclasses.replace(b, attrs={**b.attrs, "osm": extract.meeting(b.boundary)})


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
    return with_osm(Block(block_id=region.block_id, crs=crs, boundary=f, parcels=parcels,
                          streets=gpd.GeoDataFrame(geometry=[LineString(f.exterior.coords)],
                                                   crs=crs),
                          source_content_hash=None, building_geometries=bld,
                          building_tier=shared_tier(members),
                          attrs={"members": len(members), "network": region.network.name}),
                    extract_of(blocks_path, region.seed.split(".", 1)[0]))
