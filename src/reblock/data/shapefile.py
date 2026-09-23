"""ShapefileSource: read a parcel shapefile into a Region of Blocks (geopandas)."""
from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final, cast

import geopandas as gpd
import networkx as nx
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import LineString, Polygon

from reblock.buildings import SpacingDiscs
from reblock.contracts import BBox, Block, Region
from reblock.data._util import _narrowed, _window
from reblock.derive_graph import source_hash


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
    def __init__(self, path: str | Path, region_id: str, *,
                 assumed_crs: CRS | int | None, block_ids: Sequence[str] | None) -> None:
        self.path = Path(path)
        self.region_id = region_id
        self.assumed_crs = assumed_crs
        # A flat filter over the component ids (`{region_id}_{k}`), None for every block. Final
        # for the reason `KblockSource.block_ids` is: a narrower source is a new one.
        self.block_ids: Final = tuple(block_ids) if block_ids is not None else None

    def restricted(self, block_ids: Sequence[str]) -> ShapefileSource:
        return ShapefileSource(self.path, self.region_id, assumed_crs=self.assumed_crs,
                               block_ids=_narrowed(self.region_id, self.block_ids, block_ids))

    def _ids(self, components: list[list[int]]) -> list[str]:
        """Each component's block_id. Numbered over EVERY component, so an id names the same
        block whatever the filter."""
        return [f"{self.region_id}_{k}" for k in range(len(components))]

    def _kept(self, components: list[list[int]]) -> list[tuple[str, list[int]]]:
        """The (block_id, parcel rows) of each component the filter keeps, in component order.
        An id the shapefile does not have raises."""
        ids = self._ids(components)
        if self.block_ids is None:
            return list(zip(ids, components, strict=True))
        missing = set(self.block_ids) - set(ids)
        if missing:
            raise ValueError(f"{self.region_id}: block_ids not found in source: {sorted(missing)}")
        wanted = set(self.block_ids)
        return [(bid, idx) for bid, idx in zip(ids, components, strict=True) if bid in wanted]

    def _prepared(self) -> tuple[gpd.GeoDataFrame, CRS]:
        """Shared read + CRS + explode prep -- the frame `region()`'s blocks are built
        from, and that `block_geometries()` dissolves into block polygons. Factored out
        of `region()` so both share exactly one read+reproject+explode path (DRY)."""
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
        return raw, utm

    def region(self) -> Region:
        raw, utm = self._prepared()
        sch = source_hash(self.path)
        # Components, and so the id check, eagerly: an unknown id raises here, like
        # `KblockSource.region`, not partway through the iteration.
        kept = self._kept(_components(raw))
        return Region(region_id=self.region_id, crs=utm,
                      blocks=self._iter_blocks(raw, utm, sch, kept), roads=None)

    def block_geometries(self, bbox: BBox | None = None) -> gpd.GeoDataFrame:
        """block_id + dissolved connected-component geometry (it genuinely has block
        polygons, unlike `building_geometries`), windowed to `bbox` (in `utm`)."""
        raw, utm = self._prepared()
        ids: list[str] = []
        polys: list[Polygon] = []
        for bid, idx in self._kept(_components(raw)):
            poly = gpd.GeoSeries(list(raw.iloc[idx].geometry), crs=utm).union_all()
            if isinstance(poly, Polygon):
                ids.append(bid)
                polys.append(poly)
        out = gpd.GeoDataFrame({"block_id": ids}, geometry=polys, crs=utm)
        return _window(out, bbox)

    def building_geometries(self, bbox: BBox | None = None) -> gpd.GeoDataFrame:
        """A parcel shapefile has no building-point cloud, so this is honestly empty --
        a correct total implementation, not a throwing stub. `bbox` is accepted for
        protocol conformance; there is nothing to window against."""
        _, utm = self._prepared()
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=utm)

    def _iter_blocks(self, raw: gpd.GeoDataFrame, utm: CRS, source_content_hash: str | None,
                     kept: list[tuple[str, list[int]]]) -> Iterator[Block]:
        for bid, idx in kept:
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
                    f"{bid}: skipping component; dissolve is "
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
            # A parcel shapefile has no buildings (see `building_geometries` above), so the frame
            # is empty and the tier, which has nothing to model, is the point tier.
            yield Block(block_id=bid, crs=utm,
                        boundary=boundary_poly, parcels=parcels, streets=streets,
                        source_content_hash=source_content_hash,
                        building_geometries=gpd.GeoDataFrame(
                            {"geometry": []}, geometry="geometry", crs=utm),
                        building_tier=SpacingDiscs)
