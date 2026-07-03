"""ShapefileSource: read a parcel shapefile into a Region of Blocks (geopandas)."""
from __future__ import annotations

from pathlib import Path
from typing import cast

import geopandas as gpd
import networkx as nx
from shapely import STRtree
from shapely.geometry import LineString, MultiLineString, Polygon

from reblock.contracts import Block, Region


def _components(gdf: gpd.GeoDataFrame) -> list[list[int]]:
    geoms = list(gdf.geometry)
    tree = STRtree(geoms)
    graph: nx.Graph = nx.Graph()
    graph.add_nodes_from(range(len(geoms)))
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            jj = int(j)
            # "Edge-adjacent" means sharing a boundary segment, not merely a
            # point. shapely's touches() is looser than that: it is also True
            # for parcels that meet at a single corner vertex. On the real
            # Phule Nagar data, grouping by touches() puts point-touching
            # parcels in the same connected component, and the dissolved
            # union of such a component is a MultiPolygon (shapely can't
            # express two regions meeting at one point as a single Polygon) —
            # 133 of 377 components, empirically — which violates
            # Block.boundary: Polygon. Requiring the shared intersection to
            # have positive length (a real edge segment, or genuine overlap)
            # excludes point-only contact and yields a single Polygon per
            # component for every component in this dataset (verified: 0/370).
            if i < jj and g.intersection(geoms[jj]).length > 0:
                graph.add_edge(i, jj)
    return [sorted(c) for c in nx.connected_components(graph)]


def _boundary_lines(boundary: object) -> list[LineString]:
    if isinstance(boundary, MultiLineString):
        return list(boundary.geoms)
    if isinstance(boundary, LineString):
        return [boundary]
    return []


class ShapefileSource:
    def __init__(self, path: str | Path, region_id: str = "region") -> None:
        self.path = Path(path)
        self.region_id = region_id

    def region(self) -> Region:
        raw = gpd.read_file(self.path)
        mask = raw.geometry.notna() & ~raw.geometry.is_empty
        raw = cast(gpd.GeoDataFrame, raw[mask])
        if raw.crs is None:
            # Some shapefiles (e.g. topology's Phule Nagar fixture) ship without a
            # .prj sidecar, so geopandas reads them with crs=None and
            # estimate_utm_crs() has nothing to work from. Assume Web Mercator
            # (EPSG:3857) — the common default for shapefiles exported without
            # projection info — as the least-surprising fallback so the UTM
            # estimate below has a CRS to reproject through. For Phule Nagar this
            # resolves to ~19.05N/72.93E (Thane, India) and EPSG:32643, matching
            # the UTM zone already assumed elsewhere in this project.
            raw = raw.set_crs(3857)
        utm = raw.estimate_utm_crs()
        raw = raw.to_crs(utm).reset_index(drop=True)

        blocks: list[Block] = []
        for k, idx in enumerate(_components(raw)):
            geoms = list(raw.iloc[idx].geometry)
            parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(geoms)))},
                                       geometry=geoms, crs=utm)
            boundary_poly = parcels.geometry.union_all()
            if not isinstance(boundary_poly, Polygon):
                raise ValueError(
                    f"{self.region_id}_{k}: dissolved component is a "
                    f"{type(boundary_poly).__name__}, not a single Polygon"
                )
            streets = gpd.GeoDataFrame(geometry=_boundary_lines(boundary_poly.boundary), crs=utm)
            blocks.append(Block(block_id=f"{self.region_id}_{k}", crs=utm,
                                boundary=boundary_poly, parcels=parcels, streets=streets))
        return Region(region_id=self.region_id, crs=utm, blocks=tuple(blocks))
