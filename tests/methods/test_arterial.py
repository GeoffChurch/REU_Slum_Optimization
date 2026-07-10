from typing import cast

import geopandas as gpd
from pyproj import CRS
from shapely import STRtree
from shapely.geometry import LineString, Point, Polygon

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, street_connectivity
from reblock.derive.adjacency import parcel_adjacency
from reblock.methods.arterial import (
    GreedyArterialReblocker,
    _anchor_points,
    _candidate_chords,
    _deep_targets,
    _greedy_arterials,
    _planarize,
    _snap,
)
from reblock.methods.dijkstra import _boundary_graph

UTM = CRS.from_epsg(32643)


def _grid_block(n: int) -> Block:
    polys, ids = [], []
    for i in range(n):
        for j in range(n):
            polys.append(Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]))
            ids.append(i * n + j)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="grid", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_anchor_points_sample_the_network_and_include_vertices() -> None:
    net = [LineString([(0.0, 0.0), (4.0, 0.0)])]
    pts = _anchor_points(net, n=4)
    assert (0.0, 0.0) in pts and (4.0, 0.0) in pts        # endpoints/vertices kept
    assert len(pts) >= 4 and pts == sorted(pts)           # sampled + deterministic order


def test_deep_targets_are_the_deepest_parcels() -> None:
    block = _grid_block(5)                       # center parcel is deepest, full-boundary streets
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    targets = _deep_targets(block, None, k=1, adj=adj)
    assert len(targets) == 1
    assert Point(targets[0]).distance(Point(2.5, 2.5)) < 1.0   # near the 5x5 center


def test_candidate_chords_include_through_roads_and_spurs() -> None:
    anchors = [(0.0, 0.0), (4.0, 0.0)]
    targets = [(2.0, 2.0)]
    chords = _candidate_chords(anchors, targets)
    assert LineString([(0.0, 0.0), (4.0, 0.0)]) in chords            # through-road
    assert any(t == (2.0, 2.0) for c in chords for t in c.coords)    # a spur reaches the target
    assert chords == sorted(chords, key=lambda ls: ls.wkt)           # deterministic order


def test_snap_returns_a_boundary_following_street_connected_path() -> None:
    block = _grid_block(5)
    g = _boundary_graph(block.parcels)
    nodes = list(g.nodes)
    tree = STRtree([Point(nd) for nd in nodes])
    chord = LineString([(0.0, 2.5), (5.0, 2.5)])          # a horizontal cut across the grid
    path = _snap(chord, g, tree, nodes, lam=2.0)
    assert path is not None
    # every vertex of the snapped path is a boundary-graph node (buildable)
    assert all(_round(c) in set(g.nodes) for c in path.coords)


def _round(c: tuple[float, ...]) -> tuple[float, float]:
    return (round(c[0], 2), round(c[1], 2))


def test_planarize_nodes_two_crossing_chords() -> None:
    a = LineString([(0.0, 1.0), (2.0, 1.0)])
    b = LineString([(1.0, 0.0), (1.0, 2.0)])             # crosses a at (1, 1)
    gdf = _planarize([a, b], UTM)
    coords = {c for geom in gdf.geometry for c in geom.coords}
    assert (1.0, 1.0) in coords                    # crossing became a shared vertex
    assert len(gdf) == 4                           # each chord split into two at the crossing


def test_greedy_first_arterial_cuts_the_deep_block() -> None:
    # A long 3x9 block with street frontage on ONE short end only (not full-boundary: with
    # streets on all 4 sides every interior parcel is at most 2 hops from a street, so a
    # length-1 stub that merely touches the shared corner of two such parcels already "unlocks"
    # them for network_efficiency's per-pair entry-reachability test -- under the greedy's
    # gain-per-meter ranking that corner-touch stub always beats a length-9 spine, no matter how
    # many parcels the spine would serve, because the spine's benefit is diluted by its length.
    # A real deep pocket -- depth growing down the spine, only reachable by routing its length --
    # is what actually forces a spanning arterial to be the best first move.
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(9)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(0.0, 0.0), (0.0, 9.0)])], crs=UTM)
    block = Block(block_id="long", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
    roads = _greedy_arterials(block, mode="buildable", objective="directness", max_roads=3,
                              n_anchors=12)
    assert len(roads) >= 1
    assert roads.geometry.length.max() >= 6.0                        # a real lengthwise arterial
    conn = street_connectivity(block.streets, roads, STREET_TOL)
    assert conn.connected_frac == 1.0                                # buildable + street-connected


def test_greedy_is_deterministic() -> None:
    block = _grid_block(5)
    r1 = _greedy_arterials(block, mode="buildable", objective="directness", max_roads=4,
                           n_anchors=12)
    r2 = _greedy_arterials(block, mode="buildable", objective="directness", max_roads=4,
                           n_anchors=12)
    assert [g.wkt for g in r1.geometry] == [g.wkt for g in r2.geometry]


def test_greedy_roads_carry_drainage_and_slice_into_a_curve() -> None:
    from reblock.budget import cost_benefit_curve, road_drainage
    block = _grid_block(6)
    roads = _greedy_arterials(block, mode="buildable", objective="directness",
                              max_roads=5, n_anchors=12)
    assert len(roads) >= 1
    assert list(roads["drain"]) == road_drainage(block, roads)   # drain IS the actual drainage
    curve = cost_benefit_curve(block, roads)                     # integrates with budget machinery
    assert len(curve.cost) >= 2                                  # multiple budget points, not stub
    assert curve.benefit[-1] >= curve.benefit[0]                 # benefit doesn't regress w/ budget


def test_aspirational_planarizes_crossings_into_true_intersections() -> None:
    block = _grid_block(6)
    roads = _greedy_arterials(block, mode="aspirational", objective="directness", max_roads=6,
                              n_anchors=12)
    from reblock.budget import _road_street_graph
    g = _road_street_graph(block, roads, STREET_TOL)
    # at least one interior node has degree >= 4 -> a real crossroads, not overlapping lines
    interior = [nd for nd in g.nodes if Point(nd).distance(block.boundary.boundary) > STREET_TOL]
    assert any(g.degree[nd] >= 4 for nd in interior)


def test_identity_and_proposal_metadata() -> None:
    m = GreedyArterialReblocker(mode="buildable", objective="directness")
    assert m.identity == ("greedy_arterial", "buildable", "directness")
    proposal = m.propose(_grid_block(5))
    assert proposal.method == "greedy_arterial"
    assert proposal.proposal_id == "greedy_arterial_buildable_directness"
    assert proposal.roads is not None and len(proposal.roads) > 0
    assert proposal.block_identity == _grid_block(5).identity


def test_both_modes_produce_roads() -> None:
    block = _grid_block(6)
    for mode in ("buildable", "aspirational"):
        p = GreedyArterialReblocker(mode=mode, objective="directness", max_roads=4).propose(block)
        assert p.roads is not None and len(p.roads) > 0


def test_config_and_derivation_wiring() -> None:
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from reblock.derive_graph import _DERIVATION_MODULES
    assert any(p.name == "arterial.py" for p in _DERIVATION_MODULES)
    conf_dir = str(Path("conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=conf_dir):
        cfg = compose(config_name="compare_config",
                      overrides=["shapefile=x", "methods=[greedy_arterial_buildable]"])
    m = instantiate(cfg.all_methods["greedy_arterial_buildable"])
    assert m.identity == ("greedy_arterial", "buildable", "directness")
