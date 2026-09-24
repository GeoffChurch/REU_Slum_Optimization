"""Route counts per cell: observed on the block, and the prior -- the same computation with no
buildings at all. Both memoized through `derive`, keyed on the block (tier included) and the
counting parameters; `workers` changes nothing and is not in the key."""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from reblock.contracts import Block
from reblock.derive_graph import config_identity, derive
from reblock.methods.betweenness.raster import BlockRaster, home_cells
from reblock.methods.betweenness.routing import (
    bend_table,
    build_graph,
    egress_counts,
    pair_counts,
    to_grid,
)

NO_BUILDINGS = 1e12      # clearance with no buildings: 1 + (0/1e12)^2 == 1.0 exactly


@dataclass(frozen=True)
class CountParams:
    res_m: float
    r0_m: float
    bend_lambda: float
    max_sources: int
    seed: int

    @property
    def identity(self) -> Hashable:
        return config_identity(self)


@dataclass(frozen=True, eq=False)
class Counts:
    raster: BlockRaster
    egress: NDArray[np.float32]      # homes per cell on their best route to the street
    pairs: NDArray[np.float32]       # (source, target) pairs per cell, scaled to all homes


@dataclass(frozen=True, eq=False)
class CountsInput:
    block: Block
    params: CountParams
    workers: int                     # not in `identity`: it changes nothing

    @property
    def identity(self) -> Hashable | None:
        return config_identity(self, exempt=frozenset({"workers"}))


def _count(inp: CountsInput, *, with_buildings: bool) -> Counts:
    p = inp.params
    raster = BlockRaster.of(inp.block, p.res_m)
    node = -np.ones(raster.shape, dtype=np.int64)
    node[raster.inside] = np.arange(int(raster.inside.sum()))
    rc = home_cells(raster, inp.block)
    empty = np.full(raster.shape, np.nan, dtype=np.float32)
    empty[raster.inside] = 0.0
    band_mask = raster.inside & (np.nan_to_num(raster.edge, nan=np.inf) <= 1.5 * p.res_m)
    if len(rc) == 0:
        return Counts(raster=raster, egress=empty, pairs=empty.copy())
    homes = node[rc[:, 0], rc[:, 1]]
    band = node[band_mask]
    src, scale = homes, 1.0
    if p.max_sources < len(homes):
        src = homes[np.random.default_rng(p.seed).choice(len(homes), p.max_sources,
                                                          replace=False)]
        scale = len(homes) / p.max_sources
    if with_buildings:
        g = build_graph(raster.inside, raster.clearance, p.res_m, p.r0_m)
    else:
        g = build_graph(raster.inside, np.where(raster.inside, NO_BUILDINGS, np.nan),
                        p.res_m, 0.0)
    bend = bend_table(p.bend_lambda)
    c1 = egress_counts(g, bend, homes, band)
    c2 = pair_counts(g, bend, homes, src, inp.workers)
    return Counts(raster=raster,
                  egress=to_grid(c1, g, raster.inside).astype(np.float32),
                  pairs=to_grid(c2 * scale, g, raster.inside).astype(np.float32))


def _observed_impl(inp: CountsInput) -> Counts:
    return _count(inp, with_buildings=True)


def _prior_impl(inp: CountsInput) -> Counts:
    return _count(inp, with_buildings=False)


def observed(block: Block, params: CountParams, workers: int) -> Counts:
    return derive(_observed_impl, CountsInput(block, params, workers))


def prior(block: Block, params: CountParams, workers: int) -> Counts:
    return derive(_prior_impl, CountsInput(block, params, workers))
