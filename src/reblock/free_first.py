"""Free first: roads a method commits before it builds anything of its own.

Under B, the roadless metric's order (owner 2026-10-11, "B SGTM"), a network is judged first by the
population it leaves ENCLOSED -- homes with no open ground within a metre of their walls that
reaches the street -- and only then by how well everyone else gets out. A road method frees such
homes only in passing. A `FreeFirst` strategy hands a method roads to start from, the HELD roads:
the method commits them before its own first move, so its displacement accounting, its network
for routing and its output all include them, and then it builds the rest as it does today.

A Strategy, resolved where configuration is read and held by each method as the required
keyword-only field `free_first`, beside its `price` (`reblock.transparency`):

* `HoldNothing`, in every preset: nothing held, today's behaviour bit for bit -- the output frame
  included, which gains no column.
* `FreeEnclosed` (research/roadless/free_roads.py), which routes, on the method's own substrate, a
  road to each home enclosed at baseline and checks it frees the home by the metric's own test.

The NETWORK says which of its roads are held: they are its leading rows, marked in the `HELD_COL`
column (`mark_held`), and a frame that holds nothing has no such column. So `held_count` reads any
road frame -- of a method with this seam or without -- and a consumer that cuts a network into
prefixes puts the held roads first because the network says they are held, not because some
ordering happens to (research/roadless/common.canonical_order).

Where this sits. Beside `reblock.transparency` and for the same reason: `reblock.buildings` and
`reblock.contracts` are in every Block producer's import closure (`derive_graph._closure_paths` of
`data.kblock`, `region`, `derivations`), so an edit there misses every cached Block; this module
is in the closures of the methods that import it and of nothing else.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
from geopandas import GeoDataFrame
from shapely.geometry import LineString

from reblock.contracts import Block
from reblock.derive_graph import config_identity
from reblock.methods.substrates import RoutingGraph

HELD_COL = "held"     # the road frame's column marking the held roads (`mark_held`, `held_count`)


class HeldRoad(NamedTuple):
    """One held road: its line, street (or an earlier held road) end first, and the nodes of the
    method's routing graph it runs through, in the same order -- the graph `FreeFirst.held` was
    given, so a method can join them to its routing network by index."""

    geometry: LineString
    nodes: tuple[int, ...]


@runtime_checkable
class FreeFirst(Protocol):
    @property
    def identity(self) -> Hashable | None:
        """The cache key of every network built from these held roads. It must change whenever
        they can, and cover what they are computed from (code included) that no method's import
        closure reaches. None makes those networks uncacheable."""

    @property
    def tag(self) -> str:
        """What this adds to the label of a method it holds roads for: '' for `HoldNothing`, so
        every existing label (and render filename) stays as it was."""

    def held(self, block: Block, graph: RoutingGraph, *,
             width_m: float) -> tuple[HeldRoad, ...]:
        """The roads to commit first on `block`, in order, on `graph` (the method's routing graph,
        built once by the method), for roads of total width `width_m`."""


@dataclass(frozen=True)
class HoldNothing:
    """Nothing held: what every method did before free first existed."""

    @property
    def identity(self) -> Hashable | None:
        return config_identity(self)

    @property
    def tag(self) -> str:
        return ""

    def held(self, block: Block, graph: RoutingGraph, *,
             width_m: float) -> tuple[HeldRoad, ...]:
        del block, graph, width_m
        return ()


def mark_held(roads: GeoDataFrame, n: int) -> GeoDataFrame:
    """`roads` with its first `n` rows marked held in `HELD_COL`. With nothing held the frame is
    returned as it is, no column added: that absence is how a frame says it holds nothing, so a
    preset's output is today's frame exactly."""
    if not 0 <= n <= len(roads):
        raise ValueError(f"{n} held roads in a frame of {len(roads)}")
    if n == 0:
        return roads
    out = roads.copy()
    out[HELD_COL] = np.arange(len(out)) < n
    return out


def held_count(roads: GeoDataFrame) -> int:
    """How many leading rows of `roads` are held. A frame without `HELD_COL` holds none: the
    column is written only when something is held (`mark_held`), and a method without this seam
    never writes it. A frame with it must hold a non-empty LEADING block, as `mark_held` writes:
    anything else raises rather than be read as some other count."""
    if HELD_COL not in roads.columns:
        return 0
    col = roads[HELD_COL].to_numpy()
    if col.dtype != np.bool_:
        raise TypeError(f"{HELD_COL} must be bool, not {col.dtype}")
    n = int(col.sum())
    if n == 0 or not bool(col[:n].all()):
        raise ValueError(f"{HELD_COL} must mark a non-empty leading block of rows: "
                         f"{np.flatnonzero(col)[:10].tolist()} of {len(col)}")
    return n
