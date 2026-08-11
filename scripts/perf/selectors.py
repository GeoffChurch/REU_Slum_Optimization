"""How a greedy step reduces thousands of candidates to the few it scores exactly.

Pulled out as a Protocol because the interesting question is no longer "is tier 2 fast" (it is,
~320x per step at region scale) but "does tier 2's RANKING earn its place". Answering that needs
two reducers that differ only in how they choose, run through identical machinery -- which is
exactly what an injected strategy buys, and what a `rank_mode` string threaded through the greedy
would not.

The implementations here are the arms of that comparison:

  ScoreAll      score every candidate exactly -- the shipped greedy, the control
  FirstOrder    rank by (sum of d^2-1 over fronted parcels) / (buildings in corridor), take top k
  RandomSample  take k uniformly at random -- the NULL MODEL

`RandomSample` is the one that matters. The exact greedy's argmax flips under a 1e-10 perturbation,
the candidate gains are densely near-tied, and best-of-k on a near-tied distribution is close to
best-of-everything by order statistics alone. If picking k at random does as well as ranking k, the
estimate is decoration and the real finding is that this greedy only ever needed a subsample.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from shapely import STRtree
from shapely.geometry import LineString


@dataclass(frozen=True)
class RankContext:
    """Everything a reducer may look at, built once per step by the greedy.

    `depths` is positional over `parcel_tree`'s geometries -- the greedy reindexes the id-indexed
    peel before handing it over, so a reducer can never mis-align weights against parcels.
    """
    depths: np.ndarray
    parcel_tree: STRtree
    building_tree: STRtree
    half_width_m: float
    step: int


class CandidateSelector(Protocol):
    def select(self, chords: list[LineString], ctx: RankContext) -> list[LineString]: ...


@dataclass(frozen=True)
class ScoreAll:
    """The shipped behaviour: no reduction at all."""

    def select(self, chords: list[LineString], ctx: RankContext) -> list[LineString]:
        del ctx
        return chords


RANK_RADIUS = 0.5             # == STREET_TOL: "fronts" means what the peel's seeding means
CHUNK = 20_000                # chords per bulk query -- bounds the pair array, not the total work


def first_order_score(chords: list[LineString], ctx: RankContext,
                      threads: int = 1) -> np.ndarray:
    """First-order gain per building displaced, for every chord, in two bulk queries per chunk.

    `STRtree.query` over an ARRAY returns the flat (chord_index, tree_index) hit list, so
    `np.bincount` reduces it per chord with no Python loop and no buffering. Weights are
    `d^2 - 1` because the greedy optimizes `budget.access_burden` = sum d^2 and a fronted parcel
    drops to depth 1 -- NOT the `(d-1)^2` of the reported metric, which is a different function.

    Chunks are independent, and `STRtree.query` releases the GIL, so `threads > 1` spreads them over
    a thread pool. Measured on the 468,968-candidate region step: 354.9 s at 1 thread, 120.8 s at 4,
    **104.3 s at 8**, and 134.0 s at 16 -- it saturates early and then DEGRADES, the query being
    memory-bandwidth bound rather than compute bound. Hence 8, not "as many cores as exist".

    At block scale this is a no-op by construction: a few thousand candidates is a single chunk, so
    every block-scale result stands whatever `threads` says.
    """
    weights = ctx.depths ** 2 - 1.0
    out = np.zeros(len(chords))
    los = list(range(0, len(chords), CHUNK))

    def work(lo: int) -> tuple[int, np.ndarray]:
        arr = np.asarray(chords[lo:lo + CHUNK], dtype=object)
        src, tgt = ctx.parcel_tree.query(arr, predicate="dwithin", distance=RANK_RADIUS)
        gain = np.bincount(src, weights=weights[tgt], minlength=len(arr))
        bsrc, _ = ctx.building_tree.query(arr, predicate="dwithin", distance=ctx.half_width_m)
        nb = np.bincount(bsrc, minlength=len(arr)).astype(float)
        # Floor at one building. A chord displacing none is not free -- dividing by zero would rank
        # it infinitely ahead of every real candidate. See the note's open question: the exact
        # scorer DOES treat that class as infinite gain, and this cannot express it.
        return lo, gain / np.maximum(nb, 1.0)

    if threads <= 1 or len(los) <= 1:
        for lo in los:
            _, vals = work(lo)
            out[lo:lo + len(vals)] = vals
        return out
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for lo, vals in ex.map(work, los):
            out[lo:lo + len(vals)] = vals
    return out


@dataclass(frozen=True)
class FirstOrder:
    k: int
    threads: int = 1          # >1 only matters at region scale; see `first_order_score`

    def select(self, chords: list[LineString], ctx: RankContext) -> list[LineString]:
        if self.k <= 0 or len(chords) <= self.k:
            return chords
        score = first_order_score(chords, ctx, self.threads)
        keep = np.argpartition(-score, self.k)[:self.k]
        # back into candidate order: `_best_candidate`'s tie-break is order-independent, but a
        # stable order keeps an exact/shortlist diff readable
        return [chords[i] for i in sorted(keep.tolist())]


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
