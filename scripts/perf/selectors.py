"""How a greedy step reduces thousands of candidates to the few it scores exactly.

`CandidateSelector`, `FirstOrder`, `RankContext`, `CHUNK`, `RANK_RADIUS` and `first_order_score` now
live in `reblock.methods.arterial.shortlist` -- production owns the seam, since
`engines.ShortlistEngine` always injects `FirstOrder` through the SAME step loop
(`engines._greedy_shortlist`) these harnesses call. Re-exported here so callers that were already
importing them from this module keep working. What stays here are the arms that exist ONLY to
answer "does tier 2's RANKING earn its place" and have no production use:

  ScoreAll              score every candidate exactly -- the shipped greedy, the control
  RandomSample          take k uniformly at random -- the NULL MODEL
  StochasticFirstOrder  draw k from the top `pool` by score -- best-of-R restarts

`RandomSample` is the one that matters. The exact greedy's argmax flips under a 1e-10 perturbation,
the candidate gains are densely near-tied, and best-of-k on a near-tied distribution is close to
best-of-everything by order statistics alone. If picking k at random does as well as ranking k, the
estimate is decoration and the real finding is that this greedy only ever needed a subsample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

# Plain (not `X as X`) re-export: unlike src/, this module is outside mypy's --strict scan (see
# pyproject.toml's [tool.mypy] `files` list), so the --no-implicit-reexport workaround doesn't
# apply here -- and it would fragment into one import statement per name (ruff/isort keeps every
# `as`-aliased name in its own statement when combine-as-imports is off, the repo default). `CHUNK`,
# `RANK_RADIUS`, `CandidateSelector` and `FirstOrder` have no local use, only this re-export, so
# ruff's unused-import check needs `__all__` below -- the same convention the now-deleted
# shortlist_greedy.py used for its own single export.
from reblock.methods.arterial.shortlist import (
    CHUNK,
    RANK_RADIUS,
    CandidateSelector,
    FirstOrder,
    RankContext,
    first_order_score,
)

__all__ = ["CHUNK", "RANK_RADIUS", "CandidateSelector", "FirstOrder", "RandomSample",
          "RankContext", "ScoreAll", "StochasticFirstOrder", "first_order_score"]


@dataclass(frozen=True)
class ScoreAll:
    """The shipped behaviour: no reduction at all."""

    def select(self, chords: list[LineString], ctx: RankContext) -> list[LineString]:
        del ctx
        return chords


@dataclass(frozen=True)
class StochasticFirstOrder:
    """Rank, then draw k at random from the top `pool` rather than taking the top k outright.

    Exists because the two findings compose. The ranking earns its place (it lands at the top of the
    uniform-random spread, not the middle), and the greedy's outcome scatters widely and
    bidirectionally between arbitrary choices. Deterministic top-k harvests the first and ignores
    the second: every restart returns the same network, so restarts buy nothing.

    Drawing k from the top `pool` keeps the ranking's signal -- every candidate considered is
    already in the best `pool` by score -- while making runs genuinely independent, so best-of-R
    becomes available. `pool = k` degenerates to `FirstOrder(k)`; larger `pool` trades per-run
    quality for diversity across runs.
    """
    k: int
    pool: int
    seed: int

    def select(self, chords: list[LineString], ctx: RankContext) -> list[LineString]:
        if self.k <= 0 or len(chords) <= self.k:
            return chords
        pool = min(max(self.pool, self.k), len(chords))
        score = first_order_score(chords, ctx)
        top = np.argpartition(-score, pool - 1)[:pool]
        rng = np.random.default_rng((self.seed, ctx.step))
        keep = rng.choice(top, size=self.k, replace=False)
        return [chords[i] for i in sorted(keep.tolist())]


@dataclass(frozen=True)
class RandomSample:
    """The null model: k candidates chosen uniformly, ignoring every geometric signal.

    Seeded per STEP as well as per run, so a block's steps are not all the same draw and two runs
    with the same seed reproduce exactly.
    """
    k: int
    seed: int

    def select(self, chords: list[LineString], ctx: RankContext) -> list[LineString]:
        if self.k <= 0 or len(chords) <= self.k:
            return chords
        rng = np.random.default_rng((self.seed, ctx.step))
        keep = rng.choice(len(chords), size=self.k, replace=False)
        return [chords[i] for i in sorted(keep.tolist())]
