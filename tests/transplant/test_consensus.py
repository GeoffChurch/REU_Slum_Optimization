"""The weighted consensus: its field, its weights, its extraction, and the length matching every
consensus comparison is made at."""
from __future__ import annotations

from typing import cast

import geopandas as gpd
import numpy as np
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from reblock.budget import max_access_depth
from reblock.buildings import SpacingDiscs
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL, ParcelAdjacency
from reblock.methods.demand_greedy import demand_edge_weights
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from reblock.transplant.consensus import (
    ConsensusField,
    ConsensusParams,
    DonorFit,
    consensus_edge_weights,
    consensus_weights,
    extract_consensus,
    fit_donors,
    length_matched_prefix,
)
from reblock.transplant.gw import GWParams
from reblock.transplant.transport import TransportParams, fit_transport, parcel_xy

UTM = CRS.from_epsg(32734)
PARAMS = ConsensusParams(substrate=ChordSubstrate(), buffer_m=0.6, eps=0.05, gamma=1.0,
                         depth_target=1, max_roads=400, road_width_m=DEFAULT_ROAD_WIDTH_M)


def _slab(w: int, h: int, block_id: str = "slab") -> Block:
    """`w` x `h` unit parcels, one building each, street frontage along the bottom edge only."""
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for j in range(h) for i in range(w)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(len(polys)))}, geometry=polys, crs=UTM)
    return Block(block_id=block_id, crs=UTM, boundary=cast(Polygon, unary_union(polys)),
                 parcels=parcels,
                 streets=gpd.GeoDataFrame(geometry=[LineString([(0, 0), (w, 0)])], crs=UTM),
                 building_geometries=gpd.GeoDataFrame(
                     geometry=[Point(p.centroid.x, p.centroid.y) for p in polys], crs=UTM),
                 source_content_hash=None, building_tier=SpacingDiscs)


def _line(y: float, x0: float = 0.0, x1: float = 5.0) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(geometry=[LineString([(x0, y), (x1, y)])], crs=UTM)


def _fit(block_id: str, lines: gpd.GeoDataFrame, gw: float) -> DonorFit:
    return DonorFit(block_id=block_id, transported=lines, gw_dist=gw)


def test_one_donor_is_demand_greedys_own_corridor() -> None:
    """With a single donor the consensus field IS the binary corridor `demand_edge_weights` reads,
    so the two must price every edge identically. The corridor (y in [1.65, 2.85]) keeps every
    sample point off its boundary, where `contains` and `intersects` would disagree.

    FAULT INJECTION: sampling only the endpoints (dropping the midpoint) fails this.
    """
    graph = ChordSubstrate().build(_slab(5, 5))
    lines = _line(2.25)
    field = ConsensusField.of([lines], [0.3], PARAMS.buffer_m)
    ours = consensus_edge_weights(graph, field, eps=PARAMS.eps, gamma=PARAMS.gamma)
    theirs = demand_edge_weights(graph, lines, buffer_m=PARAMS.buffer_m, eps=PARAMS.eps,
                                 gamma=PARAMS.gamma)
    np.testing.assert_allclose(ours, theirs, rtol=1e-12)
    assert (ours < graph.edist - 1e-9).any(), "the corridor priced no edge below its length"


def test_weights_normalize_and_an_empty_network_adds_nothing() -> None:
    empty = gpd.GeoDataFrame(geometry=[], crs=UTM)
    field = ConsensusField.of([_line(2.25), empty], [2.0, 6.0], PARAMS.buffer_m)
    assert field.weights == (0.25, 0.75)
    np.testing.assert_allclose(field.sample(np.array([[2.0, 2.2], [2.0, 4.0]])), [0.25, 0.0])


def test_all_zero_weights_leave_the_plain_union() -> None:
    """No donor's own network is worth anything: every corridor counts equally."""
    field = ConsensusField.of([_line(1.25), _line(3.25)], [0.0, 0.0], PARAMS.buffer_m)
    assert field.weights == (0.5, 0.5)


def test_closeness_is_relative_to_the_median_distance() -> None:
    """tau is the MEDIAN, which a skewed ladder tells apart from the mean (2 here, not 3)."""
    fits = [_fit("a", _line(1.0), 1.0), _fit("b", _line(2.0), 2.0), _fit("c", _line(3.0), 6.0)]
    quality = {"a": 1.0, "b": 0.5, "c": 1.0}
    np.testing.assert_allclose(consensus_weights(fits, quality),
                               [np.exp(-0.5), 0.5 * np.exp(-1.0), np.exp(-3.0)])


def test_weights_refuse_what_they_cannot_rank() -> None:
    with pytest.raises(ValueError, match="tau > 0"):
        consensus_weights([_fit("a", _line(1.0), 0.0)], {"a": 1.0})
    with pytest.raises(ValueError, match="non-finite quality"):
        consensus_weights([_fit("a", _line(1.0), 1.0)], {"a": float("nan")})


def test_extraction_serves_every_parcel_and_stamps_its_width() -> None:
    """The consensus is a drainage tree: a single corridor steers it but cannot leave a parcel
    behind, which is what separates it from snapping the donor's own geometry."""
    block = _slab(5, 5)
    roads = extract_consensus(block, [_fit("a", _line(2.25), 1.0)], {"a": 1.0}, PARAMS)
    assert len(roads)
    assert (roads["width_m"] == PARAMS.road_width_m).all()
    assert max_access_depth(ParcelAdjacency.of(block, STREET_TOL), roads) <= PARAMS.depth_target


def test_a_length_prefix_reaches_its_target_by_at_most_one_road() -> None:
    block = _slab(5, 5)
    roads = extract_consensus(block, [_fit("a", _line(2.25), 1.0)], {"a": 1.0}, PARAMS)
    total = float(roads.geometry.length.sum())
    target = total / 2
    prefix = length_matched_prefix(block, roads, target)
    lengths = prefix.geometry.length.to_numpy()
    assert lengths.sum() >= target > lengths[:-1].sum()
    assert len(length_matched_prefix(block, roads, 10 * total)) == len(roads)
    assert length_matched_prefix(block, cast(gpd.GeoDataFrame, roads.iloc[:0]), target).empty


def test_fit_donors_keeps_donor_order_and_each_fits_own_distance() -> None:
    """The k-sweep takes `fits[:k]`, so order is the nesting guarantee."""
    recipient, a, b = _slab(8, 8, "r"), _slab(10, 6, "a"), _slab(7, 9, "b")
    params = TransportParams(gw=GWParams(eps=0.01, tau=1.0, outer_iters=5, inner_iters=20),
                             idw_k=4)
    networks = {"a": _line(3.5, x1=10.0), "b": _line(4.5, x1=7.0)}
    fits = fit_donors(recipient, [b, a], networks, params)
    assert [f.block_id for f in fits] == ["b", "a"]
    for f, donor in zip(fits, [b, a], strict=True):
        assert f.gw_dist == fit_transport(parcel_xy(donor), parcel_xy(recipient), params).gw_dist
        assert f.transported.crs == recipient.crs and len(f.transported) == 1
