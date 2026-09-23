"""The weighted consensus: its weights, its desire field, and the network `demand_greedy` extracts
from it."""
from __future__ import annotations

from typing import cast

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from reblock.budget import max_access_depth
from reblock.derive.access import STREET_TOL, ParcelAdjacency
from reblock.methods.demand_greedy import DemandGreedyReblocker, demand_edge_weights
from reblock.methods.desire_lines import mapped_field
from reblock.methods.substrates import ChordSubstrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M
from reblock.transplant.consensus import ConsensusDesireSource, consensus_weights, normalized
from reblock.transplant.donors import DonorFit, Donors
from tests.permeability_fixtures import SHIPPED
from tests.transplant.pool_fixtures import SIGNATURE, TRANSPORT, UTM, pool, slab

RECIPIENT = slab(6, 5, "r", x=0.0)
DONORS = (slab(6, 6, "a", x=3000.0), slab(7, 5, "b", x=4000.0), slab(5, 6, "c", x=5000.0))


def _source(k: int) -> ConsensusDesireSource:
    blocks = (RECIPIENT, *DONORS)
    return ConsensusDesireSource(
        donors=Donors(pool=pool(blocks, with_paths=blocks), exclusion_radius_m=2000.0, k=k,
                      min_donors=1, signature=SIGNATURE, transport=TRANSPORT),
        permeability=SHIPPED, road_width_m=DEFAULT_ROAD_WIDTH_M)


def _fit(block_id: str, gw: float) -> DonorFit:
    lines = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs=UTM)
    return DonorFit(donor=slab(4, 3, block_id, x=0.0), footpaths=lines, footpaths_identity=None,
                    transported=lines, gw_dist=gw)


def test_closeness_is_relative_to_the_median_distance() -> None:
    """tau is the MEDIAN, which a skewed ladder tells apart from the mean (2 here, not 3)."""
    fits = [_fit("a", 1.0), _fit("b", 2.0), _fit("c", 6.0)]
    quality = {"a": 1.0, "b": 0.5, "c": 1.0}
    np.testing.assert_allclose(consensus_weights(fits, quality),
                               [np.exp(-0.5), 0.5 * np.exp(-1.0), np.exp(-3.0)])


def test_weights_refuse_what_they_cannot_rank() -> None:
    with pytest.raises(ValueError, match="tau > 0"):
        consensus_weights([_fit("a", 0.0)], {"a": 1.0})
    with pytest.raises(ValueError, match="non-finite quality"):
        consensus_weights([_fit("a", 1.0)], {"a": float("nan")})


def test_weights_normalize_and_all_zero_leaves_the_plain_union() -> None:
    """No donor's own network is worth anything: every corridor counts equally."""
    assert normalized([2.0, 6.0]) == (0.25, 0.75)
    assert normalized([0.0, 0.0]) == (0.5, 0.5)


def test_the_field_is_one_group_per_donor_at_its_normalized_weight() -> None:
    """Each donor's transported footpaths, nearest first, weighted quality x closeness and
    normalized -- so demand lies in [0, 1].

    FAULT INJECTION: weighting by closeness alone (quality 1 for every donor) moves every weight.
    """
    source = _source(3)
    fits = source.donors.fits(RECIPIENT)
    quality = {f.donor.block_id: source.quality(f) for f in fits}
    assert all(0.0 < q <= 1.0 for q in quality.values())
    field = source.desire_field(RECIPIENT)
    assert [g.weight for g in field.groups] == list(normalized(consensus_weights(fits, quality)))
    assert sum(g.weight for g in field.groups) == pytest.approx(1.0)
    for group, fit in zip(field.groups, fits, strict=True):
        assert [g.wkb for g in group.lines.geometry] == [g.wkb for g in fit.transported.geometry]


def test_one_donor_is_the_binary_corridor_of_its_transplant() -> None:
    """With k=1 the weight normalizes to 1, so the consensus prices edges exactly as a mapped
    network lying where the transplant landed would."""
    source = _source(1)
    field = source.desire_field(RECIPIENT)
    (only,) = source.donors.fits(RECIPIENT)
    graph = ChordSubstrate().build(RECIPIENT)
    ours = demand_edge_weights(graph, field, buffer_m=3.0, eps=0.05, gamma=1.0)
    theirs = demand_edge_weights(graph, mapped_field(only.transported), buffer_m=3.0, eps=0.05,
                                 gamma=1.0)
    np.testing.assert_array_equal(ours, theirs)


def test_extraction_serves_every_parcel_and_stamps_its_width() -> None:
    """The consensus is a drainage tree: the donors steer it but cannot leave a parcel behind,
    which is what separates it from snapping a donor's own geometry."""
    method = DemandGreedyReblocker(desire_source=_source(3), substrate=ChordSubstrate(),
                                   buffer_m=3.0, eps=0.05, gamma=1.0, depth_target=1,
                                   max_roads=400, road_width_m=DEFAULT_ROAD_WIDTH_M)
    proposal = method.propose(RECIPIENT)
    roads = cast(gpd.GeoDataFrame, proposal.roads)
    assert len(roads) and (roads["width_m"] == DEFAULT_ROAD_WIDTH_M).all()
    assert proposal.params["demand_groups"] == 3
    assert max_access_depth(ParcelAdjacency.of(RECIPIENT, STREET_TOL), roads) <= 1
