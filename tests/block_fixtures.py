"""What a test-built `Block` without buildings states in place of them.

`Block` defaults none of its fields, so a fixture with no buildings has to say so. This is that
saying, spelled once rather than as a bare empty frame at every site.
"""
from __future__ import annotations

import geopandas as gpd
from pyproj import CRS


def no_buildings(crs: CRS) -> gpd.GeoDataFrame:
    """The building frame of a fixture block that has no buildings."""
    return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
