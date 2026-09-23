"""scripts/real_vs_synthetic.py scores every network -- the real one too -- at one budget, cut by
the lenses' own displacement truncation and nothing else. The consensus study's guard
(`tests/test_consensus_matrix.py`), applied to this study's one truncation."""
from __future__ import annotations

import math
from typing import cast

import geopandas as gpd
import pytest

import scripts.real_vs_synthetic as study
from reblock import budget
from reblock.emit import pct_displaced
from tests.test_consensus_matrix import BLOCK, _arms


def _same(a: object, b: object) -> bool:
    """Equal, with NaN equal to NaN (an empty network's statistics are NaN)."""
    return a == b or (isinstance(a, float) and isinstance(b, float)
                      and math.isnan(a) and math.isnan(b))


def test_every_network_is_scored_through_the_one_truncation_and_nothing_else(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A sentinel replaces the truncation; every statistic of every network must be the
    sentinel prefix's, the real network's included, and the budget must be the real network's own
    displacement.

    FAULT INJECTION: scoring the real network whole (`metrics_for(block, roads)` for
    `name == "real"`) fails this -- its row is not the sentinel's.
    """
    arms = _arms()
    real, synthetic = arms["own"], {"clearance": arms["clearance"], "repelled": arms["repelled"]}
    every = {"real": real, **synthetic}
    seen: list[str] = []

    def sentinel(block: object, roads: gpd.GeoDataFrame, d_frac: float) -> gpd.GeoDataFrame:
        del block
        seen.append(next(name for name, r in every.items() if r is roads))
        assert d_frac == pct_displaced(real, BLOCK.buildings)
        return cast(gpd.GeoDataFrame, roads.iloc[:2])

    monkeypatch.setattr(study, "prefix_to_displacement", sentinel)
    rows = study.score_block(BLOCK, real, synthetic)
    assert sorted(seen) == sorted(every)
    for row in rows:
        name = cast(str, row["network"])
        want = study.metrics_for(BLOCK, cast(gpd.GeoDataFrame, every[name].iloc[:2]))
        assert all(_same(row[k], v) for k, v in want.items()), name


def test_the_truncation_is_the_one_the_lenses_use() -> None:
    """Not a copy: each network is cut exactly where `budget.prefix_to_displacement` -- Lens A's
    function and the consensus study's prediction lens -- cuts it, at the real network's own
    displacement.

    FAULT INJECTION: restoring a truncation of the study's own under the same name (here, one that
    keeps every road) fails this.
    """
    arms = _arms()
    real, synthetic = arms["own"], {"clearance": arms["clearance"], "repelled": arms["repelled"]}
    target = pct_displaced(real, BLOCK.buildings)
    rows = {cast(str, r["network"]): r for r in study.score_block(BLOCK, real, synthetic)}
    cut = {name: budget.prefix_to_displacement(BLOCK, roads, target)
           for name, roads in {"real": real, **synthetic}.items()}
    assert any(len(cut[n]) < len(r) for n, r in synthetic.items()), "nothing was truncated"
    for name, prefix in cut.items():
        want = study.metrics_for(BLOCK, prefix)
        assert all(_same(rows[name][k], v) for k, v in want.items()), name
