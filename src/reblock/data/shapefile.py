"""ShapefileSource: read a parcel shapefile into a Region of Blocks (geopandas)."""
from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import geopandas as gpd
import networkx as nx
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import LineString, Polygon

from reblock.cache import source_hash
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


class ShapefileSource:
    def __init__(self, path: str | Path, region_id: str = "region", *,
                 assumed_crs: CRS | int | None = None) -> None:
        self.path = Path(path)
        self.region_id = region_id
        self.assumed_crs = assumed_crs

    def region(self) -> Region:
        raw = gpd.read_file(self.path)
        mask = raw.geometry.notna() & ~raw.geometry.is_empty
        raw = cast(gpd.GeoDataFrame, raw[mask])
        if raw.crs is None:
            # Some shapefiles (e.g. topology's Phule Nagar fixture) ship without a
            # .prj sidecar, so geopandas reads them with crs=None and
            # estimate_utm_crs() has nothing to work from. Guessing a CRS here
            # (e.g. defaulting to Web Mercator) risks silently landing real
            # parcels on Null Island if the guess is wrong, so require the
            # caller to state the assumption explicitly instead.
            if self.assumed_crs is None:
                raise ValueError(
                    f"{self.path}: shapefile has no CRS (.prj missing); pass "
                    "assumed_crs=... to ShapefileSource to state the assumption "
                    "explicitly (e.g. assumed_crs=3857)"
                )
            raw = raw.set_crs(self.assumed_crs)
        utm = raw.estimate_utm_crs()
        raw = raw.to_crs(utm).reset_index(drop=True)

        # Explode multi-part records (e.g. a native MultiPolygon row) into
        # their constituent single-part geometries at the row level, before
        # component grouping. Without this, one native multi-part parcel can
        # dissolve its whole connected component into a MultiPolygon, which
        # violates Block.boundary: Polygon (see Epworth_Before.shp: 2 native
        # MultiPolygon rows out of 5918). Then keep only non-empty Polygons:
        # explode can leave empty Polygon parts, which still report
        # geom_type == "Polygon" but carry no geometry, so mirror the
        # top-of-region() ~is_empty filter here too.
        raw = raw.explode(index_parts=False, ignore_index=True)
        keep = (raw.geometry.geom_type == "Polygon") & ~raw.geometry.is_empty
        raw = cast(gpd.GeoDataFrame, raw[keep].reset_index(drop=True))

        sch = source_hash(self.path)
        return Region(region_id=self.region_id, crs=utm,
                      blocks=self._iter_blocks(raw, utm, sch))

    def _iter_blocks(self, raw: gpd.GeoDataFrame, utm: CRS,
                     source_content_hash: str = "") -> Iterator[Block]:
        for k, idx in enumerate(_components(raw)):
            geoms = list(raw.iloc[idx].geometry)
            parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(geoms)))},
                                       geometry=geoms, crs=utm)
            boundary_poly = parcels.geometry.union_all()
            if not isinstance(boundary_poly, Polygon):
                # Real-defect backstop, now non-fatal: exploding multi-part
                # rows above means a native multi-part record can no longer
                # land here, but genuine source-data defects still can -- e.g.
                # overlapping-sliver parcels whose whole-component union
                # resolves to two disjoint parts (Epworth: ~4 of ~584
                # components). Such a component can't be expressed as a single
                # Block.boundary Polygon, so drop it with a visible warning
                # (logged data loss) rather than crashing the entire load. The
                # Goal: one malformed record must not take down the dataset.
                warnings.warn(
                    f"{self.region_id}_{k}: skipping component; dissolve is "
                    f"{type(boundary_poly).__name__}, not a Polygon "
                    f"({len(geoms)} parcels dropped)",
                    stacklevel=2,
                )
                continue
            # streets = the block's OUTER frontage only. `boundary_poly.boundary`
            # would also include every interior ring, and on real data the
            # dissolved parcel union has many -- sliver gaps between imperfectly
            # tiling parcels (CapeTown: 169). Those holes are digitization gaps,
            # not streets: seeding the BFS peel from them falsely reads
            # gap-adjacent interior parcels as street frontage, and marking their
            # edges as roads paints stray interior road segments that break
            # topology's greedy builder. The exterior ring is exactly the outer
            # frontage, matching topology's own outer-face define_roads().
            streets = gpd.GeoDataFrame(
                geometry=[LineString(boundary_poly.exterior.coords)], crs=utm)
            yield Block(block_id=f"{self.region_id}_{k}", crs=utm,
                        boundary=boundary_poly, parcels=parcels, streets=streets,
                        source_content_hash=source_content_hash)
