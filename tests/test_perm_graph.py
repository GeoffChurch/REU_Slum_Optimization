import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block
from reblock.perm_graph import GraphFigure, permeability_graph
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width

UTM = CRS.from_epsg(32734)


def _grid_block(k: int = 6, cell: float = 10.0, street: bool = True) -> Block:
    """k x k `cell`-sized parcels; the south edge (y=0) is the street unless `street` is False,
    in which case the street sits far away and NO parcel is grounded."""
    polys, ids = [], []
    for r in range(k):
        for c in range(k):
            x0, x1, y0, y1 = c * cell, (c + 1) * cell, r * cell, (r + 1) * cell
            polys.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
            ids.append(r * k + c)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    line = (LineString([(0.0, 0.0), (k * cell, 0.0)]) if street
            else LineString([(0.0, -1e5), (k * cell, -1e5)]))
    streets = gpd.GeoDataFrame(geometry=[line], crs=UTM)
    boundary = Polygon([(0, 0), (k * cell, 0), (k * cell, k * cell), (0, k * cell)])
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def _roads(lines: list[LineString]) -> gpd.GeoDataFrame:
    return with_width(gpd.GeoDataFrame(geometry=lines, crs=UTM), DEFAULT_ROAD_WIDTH_M)


SPINE = [LineString([(15.0, 0.0), (15.0, 55.0)])]


def test_shapes_are_consistent():
    fig = permeability_graph(_grid_block(), _roads(SPINE))
    assert isinstance(fig, GraphFigure)
    assert fig.n == 36
    for arr in (fig.cx, fig.cy, fig.potential, fig.ground_g):
        assert arr.shape == (36,)
    m = len(fig.rows)
    for arr in (fig.cols, fig.conductance, fig.footpath_g, fig.upgraded, fig.current):
        assert arr.shape == (m,)
    assert m > 0


def test_upgraded_is_empty_without_roads_and_nonempty_with_a_road():
    """`upgraded` means the road RAISED this edge -- so it is vacuous with no roads, and a spine
    road through the grid must raise at least one edge."""
    assert not permeability_graph(_grid_block(), None).upgraded.any()
    assert permeability_graph(_grid_block(), _roads(SPINE)).upgraded.any()


def test_ground_g_is_g_street_on_street_fronting_parcels_only():
    fig = permeability_graph(_grid_block(), None)
    grounded = fig.ground_g > 0.0
    assert grounded.sum() == 6                      # the south row of a 6x6 grid
    assert np.allclose(fig.ground_g[grounded], 20.0)   # PermeabilityParams.g_street


def test_ungrounded_block_raises():
    """A figure of an ungrounded block would be a picture of no flow anywhere -- absent is fine,
    silently zero is not."""
    with pytest.raises(ValueError, match="ungrounded"):
        permeability_graph(_grid_block(street=False), None)
