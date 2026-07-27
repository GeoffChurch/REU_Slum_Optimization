"""Country-wide OSM footpath census (tier T1): per-block interior-footpath coverage over the
whole ZAF+KEN block corpus, computed from the blocks parquet + OSM linework alone -- no
building points, no Voronoi parcels, no Block. See
docs/superpowers/specs/2026-07-27-ot-retrieval-substrate-phase1-design.md.
"""
from __future__ import annotations

import geopandas as gpd
from pyproj import CRS
from shapely.geometry.base import BaseGeometry

from reblock.methods.osm_footpaths import interior_desire_lines

# The shipped osm_footpaths tag set, imported by conf/desire_source/_footpath_tags.yaml so the
# census and the method can never disagree about what a footpath is.
FOOTPATH_TAGS: tuple[str, ...] = (
    "path", "footway", "track", "steps", "pedestrian", "living_street")
# Tags informal paths are SOMETIMES mapped under. Counted separately and never mixed into the
# primary columns, so the cost of widening the filter is visible before anyone re-extracts.
NEAR_MISS_TAGS: tuple[str, ...] = ("service", "residential", "unclassified")
# OSM ways are digitized against different imagery than the kblock outlines, so a boundary-running
# path more than STREET_TOL off the outline reads as interior. Measured: the count gate moves only
# 2.6 points across this range while total length drops ~18%, so tolerance matters for
# donor-quality ranking, not for the coverage census -- but report both and let the data say so.
TOLERANCES: tuple[float, ...] = (0.5, 2.0, 5.0)


def interiority_row(
    block_id: str,
    boundary: BaseGeometry,
    footpaths: gpd.GeoDataFrame,
    near_miss: gpd.GeoDataFrame,
    crs: CRS,
) -> dict[str, object]:
    """One census row: interior segment count and length at every tolerance, for the primary
    footpath tags and (separately) the near-miss tags.

    For a kblock block `streets` IS the outline, so the street corridor is `boundary.boundary`.
    """
    streets = boundary.boundary
    row: dict[str, object] = {"block_id": block_id, "boundary_length_m": float(streets.length)}
    for label, lines in (("interior", footpaths), ("near_miss", near_miss)):
        for tol in TOLERANCES:
            kept = interior_desire_lines(lines, boundary, streets, crs, tol=tol)
            row[f"n_{label}_segments_{tol}"] = int(len(kept))
            row[f"{label}_length_m_{tol}"] = (
                float(kept.geometry.length.sum()) if len(kept) else 0.0)
    return row
