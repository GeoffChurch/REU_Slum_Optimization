"""Tier 2: rank every candidate by a cheap first-order estimate, score only the top `k` exactly.

Needed because CELF (`engines.LazyEngine`) is valid only for submodular objectives and does not
hold for access-burden reduction, and the exact greedy (`engines.ExactEngine`) is too slow at
region scale once enumeration itself grows across steps. `CandidateSelector` is the seam:
`engines.ShortlistEngine` always injects `FirstOrder`, and the research harnesses under
scripts/perf compare it against a uniform-random null model and stochastic draws for best-of-R
restarts through the SAME step loop (`engines._greedy_shortlist`) -- a selector is the only thing
that differs between production and the research arms, so there is one implementation of the step
loop rather than a production copy and a research copy.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from shapely import STRtree
from shapely.geometry import LineString


@dataclass(frozen=True)
class RankContext:
    """Everything a reducer may look at, built once per step by the greedy.

    `depths` is positional over `parcel_tree`'s geometries -- the greedy reindexes the id-indexed
    peel before handing it over, so a reducer can never mis-align weights against parcels.
    """
    depths: NDArray[np.float64]
    parcel_tree: STRtree
    building_tree: STRtree
    half_width_m: float
    step: int


@runtime_checkable
class CandidateSelector(Protocol):
    """Which of a step's candidates get scored exactly. Production always uses `FirstOrder`; the
    seam exists because the research harnesses in scripts/perf compare selectors against each
    other (uniform-random as a null model, stochastic draws for best-of-R restarts) and would
    otherwise need their own copy of the step loop -- a duplicate that would silently drift from
    production, which is exactly the hazard that kept arterial_incremental.py untracked."""

    def select(self, chords: list[LineString], ctx: RankContext) -> list[LineString]: ...


RANK_RADIUS = 0.5             # == STREET_TOL: "fronts" means what the peel's seeding means
CHUNK = 20_000                # chords per bulk query -- bounds the pair array, not the total work


def first_order_score(chords: list[LineString], ctx: RankContext,
                      threads: int = 1) -> NDArray[np.float64]:
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
    out: NDArray[np.float64] = np.zeros(len(chords))
    los = list(range(0, len(chords), CHUNK))

    def work(lo: int) -> tuple[int, NDArray[np.float64]]:
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
