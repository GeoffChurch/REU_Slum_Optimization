"""FAULT INJECTION for the tier ladder: these calls MUST be mypy errors.

A type-safety claim nobody has watched fail guards nothing.
"""
import geopandas as gpd

from reblock.buildings import AreaDiscs, Extents, Footprints, Positions, Shapes, SpacingDiscs


def needs_extent(b: Extents) -> int:
    return len(b.radii)


def needs_shape(b: Shapes) -> int:
    return len(b.polygons)


def ok(pts: gpd.GeoDataFrame, polys: gpd.GeoDataFrame) -> None:
    needs_extent(SpacingDiscs(pts))     # fine: discs have a size
    needs_extent(AreaDiscs(pts))        # fine
    needs_extent(Footprints(polys))     # fine: footprints have a size too
    needs_shape(Footprints(polys))      # fine: only footprints have outlines


def must_fail(pts: gpd.GeoDataFrame, bare: Positions) -> None:
    needs_extent(bare)                  # ERROR: Positions has no size
    needs_shape(SpacingDiscs(pts))      # ERROR: a disc has no outline
    needs_shape(AreaDiscs(pts))         # ERROR: nor does an equal-area disc
