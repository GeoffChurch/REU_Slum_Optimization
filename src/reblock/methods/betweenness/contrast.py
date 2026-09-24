"""How route counts become a desire field: a Strategy, chosen in config.

RawShare  -- egress and all-pairs counts, each as a share of its own maximum, summed.
PriorDeviance -- the signed root Poisson deviance of the observed total against the no-buildings
                 prior: what the buildings CHANNEL, with centrality and street approaches divided
                 out. Ranks worse as a picture, generates better (it spreads the network to the
                 detours the buildings force)."""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from reblock.contracts import Block
from reblock.derive_graph import config_identity
from reblock.methods.betweenness.counts import CountParams, observed, prior
from reblock.methods.betweenness.raster import BlockRaster


def raw_share_field(o1: NDArray[np.float32], o2: NDArray[np.float32]) -> NDArray[np.float64]:
    a, b = o1.astype(np.float64), o2.astype(np.float64)
    # nanmax: the grids are NaN outside the block, and .max() would make every cell NaN.
    return a / max(float(np.nanmax(a)), 1e-12) + b / max(float(np.nanmax(b)), 1e-12)


def deviance_field(o1: NDArray[np.float32], o2: NDArray[np.float32], e1: NDArray[np.float32],
                   e2: NDArray[np.float32], floor: float) -> NDArray[np.float64]:
    o = o1.astype(np.float64) + o2.astype(np.float64)
    e = np.maximum(e1.astype(np.float64) + e2.astype(np.float64), floor)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2.0 * (np.where(o > 0, o * np.log(o / e), 0.0) - (o - e))
    out: NDArray[np.float64] = np.sign(o - e) * np.sqrt(np.maximum(dev, 0.0))
    return out


@runtime_checkable
class FieldContrast(Protocol):
    @property
    def identity(self) -> Hashable: ...

    def field(self, block: Block, params: CountParams,
              workers: int) -> tuple[BlockRaster, NDArray[np.float64]]: ...


@dataclass(frozen=True)
class RawShare:
    @property
    def identity(self) -> Hashable:
        return config_identity(self)

    def field(self, block: Block, params: CountParams,
              workers: int) -> tuple[BlockRaster, NDArray[np.float64]]:
        o = observed(block, params, workers)
        return o.raster, raw_share_field(o.egress, o.pairs)


@dataclass(frozen=True)
class PriorDeviance:
    floor: float                     # the measured value is 1e-9

    @property
    def identity(self) -> Hashable:
        return config_identity(self)

    def field(self, block: Block, params: CountParams,
              workers: int) -> tuple[BlockRaster, NDArray[np.float64]]:
        o, e = observed(block, params, workers), prior(block, params, workers)
        return o.raster, deviance_field(o.egress, o.pairs, e.egress, e.pairs, self.floor)
