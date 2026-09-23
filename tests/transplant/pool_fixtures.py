"""A donor pool small enough to fit in a test: slabs of 10 m parcels laid out along one street,
with fixed footpaths, at a Cape Town UTM origin so the blocks' WGS84 bboxes are ordinary ones."""
from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reblock.buildings import SpacingDiscs
from reblock.contracts import Block
from reblock.data.pools import DonorPool, Pools
from reblock.methods.osm_footpaths import FootpathSource
from reblock.transplant.gw import GWParams
from reblock.transplant.signature import SignatureParams
from reblock.transplant.transport import TransportParams

UTM = CRS.from_epsg(32734)
X0, Y0 = 260_000.0, 6_240_000.0     # Cape Town, in UTM 34S

# Cheap enough to fit a handful of donors per test; the shipped values are in conf/donors/.
SIGNATURE = SignatureParams(n_sub=12, n_boot=2, seed=0)
TRANSPORT = TransportParams(gw=GWParams(eps=0.01, tau=1.0, outer_iters=5, inner_iters=20),
                            idw_k=4)


def slab(w: int, h: int, block_id: str, *, x: float, source_hash: str | None = None) -> Block:
    """`w` x `h` 10 m parcels with a building each, `x` metres east of the origin, fronting a
    street along its south edge only."""
    x0 = X0 + x
    polys = [Polygon([(x0 + 10 * i, Y0 + 10 * j), (x0 + 10 * i + 10, Y0 + 10 * j),
                      (x0 + 10 * i + 10, Y0 + 10 * j + 10), (x0 + 10 * i, Y0 + 10 * j + 10)])
             for j in range(h) for i in range(w)]
    return Block(block_id=block_id, crs=UTM, boundary=cast(Polygon, unary_union(polys)),
                 parcels=gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))},
                                          geometry=polys, crs=UTM),
                 streets=gpd.GeoDataFrame(geometry=[LineString([(x0, Y0), (x0 + 10 * w, Y0)])],
                                          crs=UTM),
                 building_geometries=gpd.GeoDataFrame(
                     geometry=[Point(p.centroid.x + 2, p.centroid.y - 1) for p in polys],
                     crs=UTM),
                 source_content_hash=source_hash, building_tier=SpacingDiscs)


def path_up(block: Block) -> LineString:
    """A footpath from the block's street up through its middle, stopping short of the top."""
    minx, miny, maxx, maxy = block.boundary.bounds
    mid = minx + 10 * ((maxx - minx) // 20) + 5
    return LineString([(mid, miny), (mid, maxy - 5), (minx + 5, maxy - 5)])


class Footpaths:
    """A FootpathSource holding fixed lines in UTM, ignoring the bbox: the interior filter clips
    each block's share out of them. Counts its reads."""

    def __init__(self, lines: Sequence[LineString], identity: Hashable = ("test", "footpaths")):
        self._lines, self._identity, self.reads = list(lines), identity, 0

    @property
    def identity(self) -> Hashable:
        return self._identity

    def footpaths(self, bbox_wgs84: tuple[float, float, float, float],
                  crs: CRS) -> gpd.GeoDataFrame:
        del bbox_wgs84
        self.reads += 1
        return gpd.GeoDataFrame(geometry=self._lines, crs=UTM).to_crs(crs)


@dataclass(frozen=True, eq=False)
class FixedPool(DonorPool):
    """Every block a donor, and every block a recipient."""

    blocks: tuple[Block, ...]
    source: Footpaths
    content: Hashable | None

    def pools(self) -> Pools:
        blocks = sorted(self.blocks, key=lambda b: b.block_id)
        idx = list(range(len(blocks)))
        return Pools(blocks=blocks, blocks_gdf=gpd.GeoDataFrame(
            {"block_id": [b.block_id for b in blocks]},
            geometry=[b.boundary for b in blocks], crs=UTM), recipients=idx, donors=idx)

    def footpaths(self) -> FootpathSource:
        return self.source

    @property
    def identity(self) -> Hashable | None:
        return self.content

    def load(self) -> None:
        """Nothing to read: the blocks and footpaths are held already."""


def pool(blocks: Sequence[Block], *, with_paths: Sequence[Block],
         content: Hashable | None = None) -> FixedPool:
    """`blocks` as a pool where only `with_paths` carry a footpath."""
    return FixedPool(blocks=tuple(blocks), source=Footpaths([path_up(b) for b in with_paths]),
                     content=content)
