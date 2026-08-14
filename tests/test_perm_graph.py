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


def _node_net_current(fig: GraphFigure) -> np.ndarray:
    """Signed current leaving each node, plus what it sheds to ground. Equals (L v)_i, which is
    b_i = 1 for every node: one unit of escape current injected per parcel."""
    out = np.zeros(fig.n, dtype=np.float64)
    np.add.at(out, fig.rows, fig.current)     # leaves rows[k]
    np.add.at(out, fig.cols, -fig.current)    # arrives at cols[k]
    return out + fig.ground_g * fig.potential


@pytest.mark.parametrize("roads", [None, _roads(SPINE)], ids=["no_roads", "spine"])
def test_energy_identity(roads):
    """Dissipated power recomputed from the DRAWN quantities equals the solver's own P, which for
    b = ones also equals the sum of potentials:

        sum_edges g (dphi)^2 + sum_nodes ground_g phi^2  ==  p  ==  sum(phi)

    because v = L^-1 b makes v^T L v = v^T b. Exact up to solver residual.
    """
    fig = permeability_graph(_grid_block(), roads)
    dphi = fig.potential[fig.rows] - fig.potential[fig.cols]
    drawn = float((fig.conductance * dphi**2).sum() + (fig.ground_g * fig.potential**2).sum())
    assert drawn == pytest.approx(fig.p, rel=1e-9)
    assert float(fig.potential.sum()) == pytest.approx(fig.p, rel=1e-9)


@pytest.mark.parametrize("roads", [None, _roads(SPINE)], ids=["no_roads", "spine"])
def test_per_node_kirchhoff(roads):
    """Every node injects exactly one unit. Catches indexing and sign errors the aggregate energy
    identity can absorb -- a globally-flipped current still squares to the same power."""
    fig = permeability_graph(_grid_block(), roads)
    assert np.allclose(_node_net_current(fig), 1.0, rtol=1e-9, atol=1e-9)


def test_current_is_zero_when_every_parcel_fronts_the_street():
    """A sanity anchor for the sign convention: with every parcel grounded and the fabric
    symmetric, no unit has any reason to cross the mesh -- each leaves through its own ground edge,
    so the potentials are equal and every dphi is 0.

    Built with the full boundary ring as street, not the shared south-edge fixture, and asserted to
    have edges: on a 1x1 block `current` is empty and `allclose` would pass vacuously.
    """
    base = _grid_block(2, 10.0)
    ring = gpd.GeoDataFrame(geometry=[base.boundary.boundary], crs=UTM)
    block = Block(block_id="ring", crs=UTM, boundary=base.boundary,
                  parcels=base.parcels, streets=ring)

    fig = permeability_graph(block, None)
    assert len(fig.rows) > 0                       # not vacuous
    assert np.all(fig.ground_g > 0.0)              # every parcel fronts the ring
    assert np.allclose(fig.current, 0.0)
