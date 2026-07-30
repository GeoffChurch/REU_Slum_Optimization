"""ResistanceGreedyReblocker: select on the reported objective, stop when it stops paying.

Those two are what distinguish it from clearance, which selects on access depth and runs until
every parcel is served regardless of what the metric says.
"""
from __future__ import annotations

from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.methods.clearance import ClearanceReblocker
from reblock.methods.resistance_greedy import ResistanceGreedyReblocker
from reblock.permeability import permeability

UTM = CRS.from_epsg(32734)


def _slab(w: int, h: int) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for j in range(h) for i in range(w)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    pts = [Point(p.centroid.x, p.centroid.y) for p in polys]
    return Block(block_id="slab", crs=UTM, boundary=cast(Polygon, unary_union(polys)),
                 parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (w, 0)])], crs=UTM),
                 building_points=gpd.GeoDataFrame(geometry=pts, crs=UTM))


def test_the_first_road_is_the_ARGMAX_over_candidates_by_gain_per_metre() -> None:
    """The selection rule itself, checked against the candidate set it chose from.

    Two weaker versions of this test were tried and both were VACUOUS, caught by fault injection:
      * "permeability rises as roads are added" -- monotone BY CONSTRUCTION for any road set at
        all, since roads only add conductance, so it passes for a method choosing at random.
      * "beats what clearance would have picked" -- clearance's deepest-parcel road is usually the
        long one, so picking the LONGEST candidate passes too.
    So enumerate the candidates the method saw and require its pick to be the argmax.
    """
    import numpy as np
    import shapely
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from scipy.spatial import cKDTree

    from reblock.methods.resistance_greedy import _path_road
    from reblock.methods.substrates import ChordSubstrate

    block = _slab(6, 6)
    empty = gpd.GeoDataFrame(geometry=[], crs=block.crs)
    base = permeability(block, empty)

    graph = ChordSubstrate().build(block)
    street = unary_union(list(block.streets.geometry))
    net = np.flatnonzero(
        shapely.dwithin(shapely.points(graph.pts), street, graph.net_tol)).tolist()
    reps = np.array([[g.representative_point().x, g.representative_point().y]
                     for g in block.parcels.geometry])
    starts = cKDTree(graph.pts).query(reps)[1]
    w = np.concatenate([graph.edist, graph.edist])
    csr = csr_matrix((w, (np.concatenate([graph.rows, graph.cols]),
                          np.concatenate([graph.cols, graph.rows]))),
                     shape=(len(graph.pts), len(graph.pts)))
    _d, pred, _s = dijkstra(csr, indices=net, return_predecessors=True, min_only=True)

    best_rate = 0.0
    for i in range(len(reps)):
        if pred[starts[i]] < 0:
            continue
        road = _path_road(graph, pred, int(starts[i]), reps[i], street)
        if road is None or road.length <= 0:
            continue
        rate = (permeability(block, gpd.GeoDataFrame(geometry=[road], crs=block.crs)) - base)
        best_rate = max(best_rate, rate / road.length)

    chosen = ResistanceGreedyReblocker(max_roads=1, shortlist=999).propose(block).roads
    assert chosen is not None and len(chosen) == 1
    got = (permeability(block, chosen) - base) / float(chosen.geometry.length.sum())
    assert got >= best_rate - 1e-12, f"not the argmax: chose {got}, best available {best_rate}"


def test_it_stops_when_the_gain_stops_paying() -> None:
    """Unlike a drainage tree, which runs until every parcel is served, this has a floor: a high
    min_gain_per_m must stop it early and say so."""
    block = _slab(6, 6)
    greedy = ResistanceGreedyReblocker(max_roads=400, shortlist=8, min_gain_per_m=1e9)
    proposal = greedy.propose(block)

    assert proposal.params["stopped"] == "gain below floor"
    assert cast(int, proposal.params["roads"]) == 0


def test_it_selects_differently_from_clearance() -> None:
    """Same substrate, same candidate generation, different choice rule -- so the networks must
    differ. If they matched, the objective would be doing nothing."""
    block = _slab(6, 6)
    theirs = ClearanceReblocker(depth_target=1).propose(block).roads
    ours = ResistanceGreedyReblocker(max_roads=30, shortlist=10).propose(block).roads
    assert theirs is not None and ours is not None

    assert list(ours.geometry.astype(str)) != list(theirs.geometry.astype(str))


def test_a_block_with_no_street_frontage_is_reported_not_crashed() -> None:
    """No frontage means no ground node, so the objective is undefined -- it must say so rather
    than return an empty network that reads as 'nothing worth building'."""
    block = _slab(4, 4)
    floating = Block(block_id="floating", crs=block.crs, boundary=block.boundary,
                     parcels=block.parcels,
                     streets=gpd.GeoDataFrame(geometry=[LineString([(99, 99), (100, 100)])],
                                              crs=block.crs),
                     building_points=block.building_points)
    proposal = ResistanceGreedyReblocker(max_roads=4, shortlist=4).propose(floating)
    assert proposal.params["stopped"] == "no street frontage"
