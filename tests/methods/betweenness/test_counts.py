from __future__ import annotations

import dataclasses

import numpy as np

from reblock.buildings import AreaDiscs
from reblock.contracts import Block
from reblock.methods.betweenness.counts import (
    NO_BUILDINGS,
    CountParams,
    CountsInput,
    observed,
    prior,
)
from reblock.methods.betweenness.routing import build_graph
from tests.scoring_fixtures import _block_1808

P = CountParams(res_m=1.0, r0_m=2.0, bend_lambda=50.0, max_sources=400, seed=0)


def _keyed() -> Block:
    # A real source hash, so Block.identity (and every key built on it) is not None.
    return dataclasses.replace(_block_1808(), source_content_hash="fixture")


def test_with_no_buildings_the_observed_graph_is_the_prior_graph() -> None:
    # r0 = 2 over clearance 1e12 is density 1 + (2/1e12)^2 == 1.0 exactly: the prior's own graph,
    # so on an empty block observed and prior must agree bit for bit.
    inside = np.zeros((40, 60), bool)
    inside[1:-1, 1:-1] = True
    clear = np.where(inside, NO_BUILDINGS, np.nan)
    a, b = build_graph(inside, clear, 0.5, 2.0), build_graph(inside, clear, 0.5, 0.0)
    assert np.array_equal(a.cost, b.cost) and np.array_equal(a.nbr, b.nbr)


def test_counts_share_the_raster_and_are_nan_outside() -> None:
    block = _block_1808()
    o, e = observed(block, P, 1), prior(block, P, 1)
    assert o.egress.shape == e.pairs.shape == o.raster.shape
    assert np.isnan(o.pairs[~o.raster.inside]).all()
    assert np.nanmax(o.pairs) > 0 and np.nanmax(e.pairs) > 0


def test_the_prior_actually_drops_the_buildings() -> None:
    b = _block_1808()
    assert not np.array_equal(observed(b, P, 1).pairs, prior(b, P, 1).pairs, equal_nan=True)


def test_the_prior_ignores_clearance_and_r0() -> None:
    b = _block_1808()
    p0, p5 = prior(b, P, 1), prior(b, dataclasses.replace(P, r0_m=5.0), 1)
    assert np.array_equal(p0.egress, p5.egress, equal_nan=True)
    assert np.array_equal(p0.pairs, p5.pairs, equal_nan=True)
    o0, o5 = observed(b, P, 1), observed(b, dataclasses.replace(P, r0_m=5.0), 1)
    assert not np.array_equal(o0.pairs, o5.pairs, equal_nan=True)


def test_a_block_with_no_buildings_has_zero_counts_and_a_finite_raw_share() -> None:
    from reblock.methods.betweenness.contrast import RawShare

    b = dataclasses.replace(_block_1808(),
                            building_geometries=_block_1808().building_geometries.head(0))
    o, e = observed(b, P, 1), prior(b, P, 1)
    for c in (o, e):
        assert np.isnan(c.egress[~c.raster.inside]).all()
        assert np.isnan(c.pairs[~c.raster.inside]).all()
        assert (c.egress[c.raster.inside] == 0.0).all()
        assert (c.pairs[c.raster.inside] == 0.0).all()
        assert c.egress.dtype == np.float32 and c.pairs.dtype == np.float32
    _raster, field = RawShare().field(b, P, 1)
    assert np.isfinite(field[o.raster.inside]).all()


def test_counts_are_keyed_on_the_building_tier() -> None:
    spacing = _keyed()
    area = dataclasses.replace(spacing, building_tier=AreaDiscs)
    assert CountsInput(spacing, P, 1).identity != CountsInput(area, P, 1).identity


def test_workers_do_not_enter_the_cache_key() -> None:
    b = _keyed()
    assert CountsInput(b, P, 1).identity == CountsInput(b, P, 8).identity
