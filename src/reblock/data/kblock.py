"""KblockSource: real kblock street-bounded blocks + a building-point layer -> Blocks.

Parcels are the Voronoi cells of a block's building points, clipped to the block and
exploded to single polygons (standard Voronoi-clip, implemented independently).
streets = the block boundary (a kblock block is a street-bounded face). Agnostic to
the building source (reads whatever points GeoParquet fixture-prep produced).
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import cast

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from pyproj import CRS
from shapely import make_valid, voronoi_polygons
from shapely.geometry import GeometryCollection, MultiPoint, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry

from reblock.buildings import Extents, SpacingDiscs
from reblock.contracts import BBox, Block, Region
from reblock.data._util import _window
from reblock.data.counts import RAW_COUNT
from reblock.derivations import VoronoiInput, voronoi
from reblock.derive_graph import source_hash


def _voronoi_parcels(poly: Polygon, points: list[Point], crs: CRS) -> gpd.GeoDataFrame | None:
    seen: set[tuple[float, float]] = set()
    sites: list[Point] = []
    for p in points:
        key = (round(p.x, 3), round(p.y, 3))          # mm dedupe: voronoi needs distinct sites
        if key not in seen:
            seen.add(key)
            sites.append(p)
    if len(sites) < 4:
        return None
    geoms: list[Polygon] = []
    for cell in voronoi_polygons(MultiPoint(sites), extend_to=poly.envelope).geoms:
        clipped = make_valid(cell).intersection(poly)
        parts: list[BaseGeometry]
        if isinstance(clipped, (MultiPolygon, GeometryCollection)):
            parts = list(clipped.geoms)
        else:
            parts = [clipped]
        for part in parts:                             # explode lobes -> one parcel_id per polygon
            if isinstance(part, Polygon) and not part.is_empty and part.area > 0:
                geoms.append(part)
    if not geoms:
        return None
    return gpd.GeoDataFrame({"parcel_id": list(range(len(geoms)))}, geometry=geoms, crs=crs)


class KblockSource:
    def __init__(self, blocks_path: str | Path, buildings_path: str | Path,
                 region_id: str = "kblock", *, min_buildings: int = 10,
                 block_ids: list[str] | None = None,
                 building_tier: Callable[[gpd.GeoDataFrame], Extents] = SpacingDiscs) -> None:
        self.blocks_path = Path(blocks_path)
        self.buildings_path = Path(buildings_path)
        self.region_id = region_id
        self.min_buildings = min_buildings
        self.block_ids = list(block_ids) if block_ids is not None else None
        self.building_tier = building_tier
        self._utm: CRS | None = None

    def _target_utm(self) -> CRS:
        """The UTM `region()`/the accessors reproject to, computed once from the blocks
        parquet's geometry column (cheap: no buildings, no Voronoi) and cached."""
        if self._utm is None:
            self._utm = gpd.read_parquet(
                self.blocks_path, columns=["geometry"]).estimate_utm_crs()
        return self._utm

    def block_geometries(self, bbox: BBox | None = None) -> gpd.GeoDataFrame:
        """Cheap block_id + building_count + geometry accessor for a RegionBuilder: reads only
        the blocks parquet (no buildings, no Voronoi), reprojected to the same UTM `region()`
        uses. `building_count` is a per-block building count (present for kblock sources),
        e.g. for budgeting region growth on buildings as a parcel proxy. Applies
        `self.block_ids` as a flat filter if set (the region CLI passes a flat set of
        candidate ids here, not the nested seed groups). `bbox` (in the target UTM) windows
        the result via `.cx`; `bbox=None` returns everything."""
        blocks = gpd.read_parquet(
            self.blocks_path, columns=["block_id", "building_count", "geometry"])
        # Renamed at the boundary: this is the VENDOR count, and only `counts.resolved()` may turn
        # it into the `building_count` every metric and region builder reads.
        blocks = blocks.rename(columns={"building_count": RAW_COUNT})
        blocks["block_id"] = blocks["block_id"].astype(str)
        if self.block_ids is not None:
            wanted = {str(b) for b in self.block_ids}
            blocks = cast(gpd.GeoDataFrame, blocks[blocks["block_id"].isin(wanted)])
        out = cast(gpd.GeoDataFrame, blocks.to_crs(self._target_utm())[
            ["block_id", RAW_COUNT, "geometry"]])
        return _window(out, bbox)

    def building_geometries(self, bbox: BBox | None = None) -> gpd.GeoDataFrame:
        """The buildings parquet -- points or polygons, with every column it carries (notably
        `area_in_meters`) -- reprojected to the same UTM as `block_geometries()`/`region()` so
        overlays align. `bbox` (in the target UTM) windows via `.cx`."""
        bld = gpd.read_parquet(self.buildings_path).to_crs(self._target_utm())
        return _window(bld, bbox)

    def region(self) -> Region:
        blocks = gpd.read_parquet(
            self.blocks_path, columns=["block_id", "k_complexity", "geometry"])
        blocks["block_id"] = blocks["block_id"].astype(str)
        utm = blocks.estimate_utm_crs()   # full frame => CRS is stable under block_ids filtering
        if self.block_ids is not None:
            wanted = {str(b) for b in self.block_ids}
            missing = wanted - set(blocks["block_id"])
            if missing:
                raise ValueError(
                    f"{self.region_id}: block_ids not found in source: {sorted(missing)}")
            blocks = cast(gpd.GeoDataFrame, blocks[blocks["block_id"].isin(wanted)])
        # Every column, not just geometry: `columns=["geometry"]` is what dropped area_in_meters.
        bld = gpd.read_parquet(self.buildings_path)
        # Fail at LOAD, not deep inside a run: build the tier on the rows just read, so AreaDiscs
        # on a parquet without `area_in_meters`, or Footprints on point geometry, raises HERE --
        # naming the file -- before any block is yielded or any method runs, rather than on the
        # first lazy `block.buildings` access hours later. Not in __init__: the class is
        # deliberately I/O-free at construction (see the lazy `_target_utm`), and tests build it
        # on placeholder paths.
        try:
            self.building_tier(bld.head(32))
        except ValueError as e:
            raise ValueError(f"{self.region_id}: {self.buildings_path} cannot supply the "
                             f"configured building tier: {e}") from e
        sch = source_hash(self.blocks_path, self.buildings_path)
        return Region(region_id=self.region_id, crs=utm,
                      blocks=self._blocks_from(blocks.to_crs(utm), bld.to_crs(utm), sch))

    def _blocks_from(self, blocks: gpd.GeoDataFrame, bld: gpd.GeoDataFrame,
                     source_content_hash: str) -> Iterator[Block]:
        utm = blocks.crs
        if utm is None:
            raise ValueError(f"{self.region_id}: blocks GeoDataFrame has no CRS")
        # Assign and tessellate on CENTROIDS, whatever the geometry: that keeps parcels IDENTICAL
        # across tiers (a polygon straddling a block edge would fail `within` and vanish), and the
        # point tier's centroids are its points, so nothing moves. Only the building MODEL changes.
        anchor = bld.assign(_row=np.arange(len(bld))).set_geometry(bld.geometry.centroid)
        joined = gpd.sjoin(anchor, blocks, predicate="within", how="inner")
        by_block: dict[object, tuple[list[Point], NDArray[np.int64]]] = {
            bid: (cast(list[Point], list(grp.geometry)), grp["_row"].to_numpy(dtype=np.int64))
            for bid, grp in joined.groupby("block_id")
        }
        empty: tuple[list[Point], NDArray[np.int64]] = ([], np.empty(0, dtype=np.int64))
        for _, row in blocks.sort_values("block_id").iterrows():
            pts, rows = by_block.get(row["block_id"], empty)
            if len(pts) < self.min_buildings:
                continue
            poly = make_valid(row["geometry"])
            if not isinstance(poly, Polygon):
                warnings.warn(f"{self.region_id}:{row['block_id']}: dissolve is "
                              f"{poly.geom_type}, not Polygon; skipping", stacklevel=2)
                continue
            parcels = voronoi(VoronoiInput(
                source_id=source_content_hash, block_id=str(row["block_id"]),
                poly=poly, points=pts, crs=utm))
            if parcels is None:
                continue
            streets = gpd.GeoDataFrame(  # all rings (incl. holes)
                geometry=[poly.boundary], crs=utm)
            yield Block(block_id=str(row["block_id"]), crs=utm, boundary=poly,
                        parcels=parcels, streets=streets,
                        source_content_hash=source_content_hash,
                        attrs={"kblock_k": float(row["k_complexity"])},
                        building_geometries=cast(
                            gpd.GeoDataFrame, bld.iloc[rows].reset_index(drop=True)),
                        building_tier=self.building_tier)
