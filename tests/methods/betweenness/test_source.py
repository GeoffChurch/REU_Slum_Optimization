from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import pytest

from reblock.derive_graph import closure_hash
from reblock.methods.betweenness import BetweennessDesire, PriorDeviance, RawShare
from reblock.methods.desire_lines import DesireLineSource
from tests.block_fixtures import no_buildings
from tests.scoring_fixtures import _block_1808

Q = (0.80, 0.85, 0.90, 0.95, 0.98)

BASE = BetweennessDesire(res_m=1.0, r0_m=2.0, bend_lambda=50.0, max_sources=400, seed=0,
                         quantiles=Q, contrast=RawShare(), workers=1)


def test_it_is_a_desire_line_source_with_one_group_per_quantile() -> None:
    assert isinstance(BASE, DesireLineSource)
    field = BASE.desire_field(_block_1808())
    assert len(field.groups) == len(Q)
    assert all(abs(g.weight - 1 / len(Q)) < 1e-12 for g in field.groups)
    assert field.n_lines > 0
    # nested levels: the top level's ridge is shorter than the bottom level's
    lens = [float(g.lines.length.sum()) for g in field.groups]
    assert lens[0] > lens[-1] > 0


def test_ridges_lie_within_the_block_boundary() -> None:
    b = _block_1808()
    boundary = b.boundary.buffer(1e-6)
    field = BASE.desire_field(b)
    assert field.groups
    for g in field.groups:
        assert all(boundary.contains(line) for line in g.lines.geometry)


def test_a_block_with_no_buildings_has_no_desire_instead_of_crashing() -> None:
    b = _block_1808()
    # source_content_hash=None too: under the real hash this is a block production can never
    # construct (same content address, different buildings), and derive() would serve it the
    # real block's cached counts instead of recomputing on empty buildings.
    empty = replace(b, building_geometries=no_buildings(b.crs), source_content_hash=None)
    assert BASE.desire_field(empty).groups == ()


def test_the_contrast_strategy_runs_too() -> None:
    deviance = replace(BASE, contrast=PriorDeviance(floor=1e-9))
    b = _block_1808()
    field = deviance.desire_field(b)
    assert len(field.groups) == 5
    # a desire_field that ignored self.contrast would return the raw-share ridges here too.
    base_top = float(BASE.desire_field(b).groups[0].lines.length.sum())
    deviance_top = float(field.groups[0].lines.length.sum())
    assert base_top != deviance_top


def test_identity_leads_with_the_packages_code_hash() -> None:
    """The source reaches `derive` only as a field of a Method (demand_greedy, loop_closure), and
    that Method's key hashes its own import closure, which holds no file of this package. The
    code enters the key through this hash instead, riding the Method's identity.

    FAULT INJECTION: returning `config_identity(self, exempt=...)` alone from
    `BetweennessDesire.identity` fails this; an edit to the raster, the routing or the ridges
    would then serve the old cached roads.
    """
    identity = BASE.identity
    assert isinstance(identity, tuple)
    assert identity[0] == closure_hash("reblock.methods.betweenness.source")


def test_identity_covers_every_setting_but_workers() -> None:
    assert replace(BASE, workers=8).identity == BASE.identity
    for other in (replace(BASE, res_m=0.5), replace(BASE, r0_m=1.0),
                 replace(BASE, bend_lambda=5.0), replace(BASE, max_sources=100),
                 replace(BASE, seed=1), replace(BASE, quantiles=(0.8, 0.9)),
                 replace(BASE, contrast=PriorDeviance(floor=1e-9))):
        assert other.identity != BASE.identity, other


_BAD: tuple[Callable[[], BetweennessDesire], ...] = (
    lambda: replace(BASE, quantiles=()),
    lambda: replace(BASE, quantiles=(0.9, 0.8)),
    lambda: replace(BASE, quantiles=(0.8, 0.8)),
    lambda: replace(BASE, quantiles=(0.0, 0.5)),
    lambda: replace(BASE, quantiles=(0.5, 1.0)),
    lambda: replace(BASE, quantiles=cast("tuple[float, ...]", [0.8, 0.9])),
    lambda: replace(BASE, res_m=0.0),
    lambda: replace(BASE, r0_m=-1.0),
    lambda: replace(BASE, bend_lambda=-1.0),
    lambda: replace(BASE, max_sources=0),
    lambda: replace(BASE, workers=0),
)


@pytest.mark.parametrize("make", _BAD)
def test_invalid_settings_raise_at_construction(make: Callable[[], BetweennessDesire]) -> None:
    with pytest.raises(ValueError):
        make()
