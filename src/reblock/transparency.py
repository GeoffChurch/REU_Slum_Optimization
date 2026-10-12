"""Transparency: how cheap each building is for a road method to displace and to route through.

One factor per building, `f_b` in (0, 1]: 1 is the building as every method treats it today, and a
smaller factor makes it cheaper. A `Transparency` gives a block's factors, aligned with
`block.buildings` row order (the order `IncrementalOverlap` and every displacement use), and an
identity. It is a Strategy: built once where configuration is read, held by a method or a
substrate as a field, and never asked which one it is.

It enters a method in two independent places, so either can be switched on alone:

* PRICE: the displacement a method uses to CHOOSE counts building b's displaced share times `f_b`
  (`cycle_native`'s gain per unit displacement, `resistance_lp`'s budget row and the rounding that
  enforces it, `clearance`'s building term at the nearest building's factor), through each
  method's `price` field and `PricedOverlap`. Everything reported stays TRUE displacement, and so
  do the caps that decide how far a method builds where they are not its allocation
  (`cycle_native`'s `max_displacement`).
* ROUTE: `TransparentSubstrate`, which makes an edge cheaper by its length inside the buildings,
  each at its factor.

`Opaque` (every factor 1) is today's behaviour, bit for bit. A factor of 0 is refused: it makes a
cost vanish, and zero-weight edges are a trap in scipy's csgraph.

Where this sits. Not in `reblock.buildings`: that module is in every Block producer's import
closure (`derive_graph._closure_paths` of `data.kblock`, `region`, `derivations`, `contracts`), so
an edit there misses every cached Block; this module is in the closures of the methods that import
it and of nothing else. And `TransparentSubstrate` is here rather than in `methods/` beside
`DesireWarpedSubstrate`: every module in `methods/` must sit in some configured method's closure
(tests/test_derive_graph.py), and no preset routes on it, while the methods it serves import this
module for their price anyway.
"""
from __future__ import annotations

import hashlib
from collections.abc import Hashable
from dataclasses import dataclass, replace
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from reblock.buildings import Extents, IncrementalOverlap
from reblock.contracts import Block
from reblock.derive_graph import closure_hash, config_identity
from reblock.methods.substrates import RoutingGraph, Substrate


@runtime_checkable
class Transparency(Protocol):
    @property
    def identity(self) -> Hashable | None:
        """The cache key of every network built with these factors. It must change whenever the
        factors can: a set read from stored rows covers what it READ, not where it read it from,
        since rows are rewritten in place. None makes those networks uncacheable."""

    def factors(self, block: Block) -> NDArray[np.float64]:
        """`f_b` per building of `block`, in `block.buildings` row order. Read it through
        `factors_of`, which checks it."""


def factors_of(transparency: Transparency, block: Block) -> NDArray[np.float64]:
    """`transparency`'s factors for `block`, checked: one per building, each in (0, 1]."""
    f = np.asarray(transparency.factors(block), dtype=np.float64)
    n = len(block.buildings)
    if f.shape != (n,):
        raise ValueError(f"{block.block_id}: factors of shape {f.shape} for {n} buildings")
    bad = ~((f > 0.0) & (f <= 1.0))          # NaN fails both comparisons
    if bad.any():
        raise ValueError(f"{block.block_id}: factors must lie in (0, 1]; buildings "
                         f"{np.flatnonzero(bad)[:5].tolist()} have {f[bad][:5].tolist()}")
    return f


@dataclass(frozen=True)
class Opaque:
    """Every building at factor 1: what every method did before transparency existed."""

    @property
    def identity(self) -> Hashable | None:
        return config_identity(self)

    def factors(self, block: Block) -> NDArray[np.float64]:
        return np.ones(len(block.buildings), dtype=np.float64)


# ------------------------------------------------------------------------------------------------
# The price


class Increment(NamedTuple):
    """What a piece of corridor adds to displacement: `true` as `IncrementalOverlap.delta` counts
    it, `priced` with each building's added share times its factor."""

    true: float
    priced: float


class PricedOverlap(IncrementalOverlap):
    """`IncrementalOverlap` that also prices what it holds, building b's share times `f_b`.

    A subclass beside `buildings.py` rather than an edit to it (see the module docstring). `true`
    repeats `delta`'s arithmetic exactly, and at factor 1 `priced` equals it bit for bit, so a
    method that chooses by `priced` under `Opaque` chooses exactly as it did by `delta`.
    """

    def __init__(self, buildings: Extents, *, factors: NDArray[np.float64]) -> None:
        super().__init__(buildings)
        if factors.shape != (len(buildings),):
            raise ValueError(f"{factors.shape} factors for {len(buildings)} buildings")
        self._factors = factors

    def increment(self, piece: BaseGeometry) -> Increment:
        """What `piece` would add, true and priced, without committing it."""
        hit, taken_after, _ = self._pieces(piece)
        true = 0.0
        priced = 0.0
        if len(hit):
            a = self._area[hit]
            pos = a > 0.0
            with np.errstate(divide="ignore", invalid="ignore"):
                before = np.clip(np.where(pos, self._taken[hit] / a, 0.0), 0.0, 1.0)
                after = np.clip(np.where(pos, taken_after / a, 0.0), 0.0, 1.0)
            step = after - before
            true += float(step[pos].sum())
            priced += float((self._factors[hit] * step)[pos].sum())
        newly = self._zero_newly(piece)
        return Increment(true=true + float(newly.sum()),
                         priced=priced + float(self._factors[self._zero_idx][newly].sum()))

    def priced_total(self) -> float:
        """Priced displacement under everything added so far: `sum_b f_b c_b`."""
        return float((self._factors * self.c()).sum())


# ------------------------------------------------------------------------------------------------
# The route


def route_multiplier(graph: RoutingGraph, outlines: NDArray[np.object_],
                     factors: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per entry of `graph`'s symmetric COO, the mean of the factor field along its edge: 1 outside
    every building, `factors[b]` inside building b (`outlines[b]`), the smallest where several
    overlap.

    That is (length outside buildings + sum_b f_b * length inside b) / length wherever footprints
    do not overlap. They do -- hundreds of pairs on a large block -- and over an overlap that sum
    counts a stretch once per building, so that a stretch inside two opaque buildings would read
    dearer than open ground; the smallest factor keeps the mean in [min f, 1], and exactly 1 on
    every edge when every factor is 1.

    By levels: with the distinct factors below 1 sorted, v_1 < ... < v_k, and U_j the union of the
    outlines whose factor is at most v_j, the field is v_j on U_j less U_(j-1) and 1 outside U_k,
    so the mean along an edge of length L is 1 - sum_j (v_(j+1) - v_j) len(edge & U_j) / L, with
    v_(k+1) = 1. A union's parts are disjoint, so an edge's length inside U_j sums over the parts
    it crosses."""
    mask = graph.rows < graph.cols                     # one direction per undirected edge
    ui, uj = graph.rows[mask], graph.cols[mask]
    lines = shapely.linestrings(np.stack([graph.pts[ui], graph.pts[uj]], axis=1))
    length = shapely.length(lines)
    discount = np.zeros(len(lines), dtype=np.float64)
    levels = np.unique(factors[factors < 1.0])
    for v, step in zip(levels, np.diff(np.append(levels, 1.0)), strict=True):
        parts = shapely.get_parts(shapely.union_all(outlines[factors <= v]))
        e, p = STRtree(parts).query(lines, predicate="intersects")
        inside = shapely.length(shapely.intersection(lines[e], parts[p]))
        np.add.at(discount, e, step * inside / length[e])
    per_edge = 1.0 - discount
    if not np.all((per_edge > 0.0) & (per_edge <= 1.0)):  # a zero-length edge in a building
        raise ValueError("route multipliers outside (0, 1]: an edge of zero length?")
    # Back onto both directed entries, in the COO order the graph hands out.
    n = len(graph.pts)
    key = (np.minimum(graph.rows, graph.cols).astype(np.int64) * n
           + np.maximum(graph.rows, graph.cols).astype(np.int64))
    ukey = ui.astype(np.int64) * n + uj.astype(np.int64)
    order = np.argsort(ukey)
    return np.asarray(per_edge[order][np.searchsorted(ukey[order], key)], dtype=np.float64)


@dataclass(frozen=True, kw_only=True)
class TransparentSubstrate:
    """Any substrate, its edges cheaper where they cross transparent buildings: the base's `edist`
    times `route_multiplier`. A method routes its shortest paths in its substrate's `edist`, so
    this reroutes any method that does -- `cycle_native`, `resistance_lp` and `clearance` (whose
    cost field multiplies `edist` rather than recomputing a length).

    As a multiplier on the base's weight it composes with any base, `DesireWarpedSubstrate`
    included; the length it divides by is the edge's own, not the base's warped weight."""

    base: Substrate
    transparency: Transparency

    @property
    def identity(self) -> Hashable:
        # A method holding this need not import this module (one with no `price`), so this
        # module's code reaches the key through its own closure hash.
        config = config_identity(self)
        return None if config is None else (closure_hash(__name__), config)

    @property
    def tag(self) -> str:
        """The base's tag and a digest of the configuration: a label names render files, and two
        transparencies over one base must not share one."""
        digest = hashlib.sha256(repr(config_identity(self)).encode()).hexdigest()[:8]
        return f"{self.base.tag}+transparent:{digest}"

    def build(self, block: Block) -> RoutingGraph:
        graph = self.base.build(block)
        outlines = np.asarray(block.buildings.outlines.to_numpy(), dtype=object)
        factors = factors_of(self.transparency, block)
        return replace(graph, edist=graph.edist * route_multiplier(graph, outlines, factors))
