from __future__ import annotations

import dataclasses

import pytest

from reblock.methods.betweenness import BetweennessDesire, PriorDeviance, RawShare
from reblock.methods.desire_lines import DesireLineSource
from tests.block_fixtures import no_buildings
from tests.scoring_fixtures import _block_1808

Q = (0.80, 0.85, 0.90, 0.95, 0.98)

BASE = BetweennessDesire(res_m=1.0, r0_m=2.0, bend_lambda=50.0, max_sources=400, seed=0,
                         quantiles=Q, contrast=RawShare(), workers=1)


def _src(**kw: object) -> BetweennessDesire:
    return dataclasses.replace(BASE, **kw)  # type: ignore[arg-type]


def test_it_is_a_desire_line_source_with_one_group_per_quantile() -> None:
    src = _src()
    assert isinstance(src, DesireLineSource)
    field = src.desire_field(_block_1808())
    assert len(field.groups) == len(Q)
    assert all(abs(g.weight - 1 / len(Q)) < 1e-12 for g in field.groups)
    assert field.n_lines > 0
    # nested levels: the top level's ridge is shorter than the bottom level's
    lens = [float(g.lines.length.sum()) for g in field.groups]
    assert lens[0] > lens[-1] > 0


def test_a_block_with_no_buildings_has_no_desire_instead_of_crashing() -> None:
    b = _block_1808()
    empty = dataclasses.replace(b, building_geometries=no_buildings(b.crs))
    assert _src().desire_field(empty).groups == ()


def test_the_contrast_strategy_runs_too() -> None:
    assert len(_src(contrast=PriorDeviance(floor=1e-9)).desire_field(_block_1808()).groups) == 5


def test_identity_covers_every_setting_but_workers() -> None:
    a = _src()
    assert a.identity == _src(workers=8).identity
    for change in (dict(res_m=0.5), dict(r0_m=1.0), dict(bend_lambda=5.0), dict(max_sources=100),
                   dict(seed=1), dict(quantiles=(0.8, 0.9)), dict(contrast=PriorDeviance(1e-9))):
        assert _src(**change).identity != a.identity, change


@pytest.mark.parametrize("bad", [dict(quantiles=()), dict(quantiles=(0.9, 0.8)),
                                 dict(quantiles=(0.0, 0.5)), dict(res_m=0.0),
                                 dict(max_sources=0), dict(workers=0)])
def test_invalid_settings_raise_at_construction(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _src(**bad)
