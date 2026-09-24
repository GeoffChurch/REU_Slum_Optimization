"""`BetweennessDesire`: a `DesireLineSource` from curvature-aware repelled betweenness.

Where many homes' best routes -- to the street, and to each other -- run, with routes that pay for
turning and are repelled by buildings. Measured as a road GENERATOR (demand_greedy toward its
ridges; docs/superpowers/notes/2026-09-24-roadless-fields-and-the-betweenness-generator.md): it
beats clearance on both lenses, and looped it beats clearance_looped, on 220 small and 36 large
blocks; cycle_native still leads Lens A on large blocks."""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from itertools import pairwise

from reblock.contracts import Block
from reblock.derive_graph import config_identity
from reblock.methods.betweenness.contrast import FieldContrast
from reblock.methods.betweenness.counts import CountParams
from reblock.methods.betweenness.ridges import ridge_desire
from reblock.methods.desire_lines import DesireField


@dataclass(frozen=True)
class BetweennessDesire:
    res_m: float                     # raster resolution; 1.0 is the large-block operating point
    r0_m: float                      # repulsion length: cost density 1 + (r0 / clearance)^2
    bend_lambda: float               # turning cost lam * dtheta^2 / 2 m
    max_sources: int                 # all-pairs sources sampled when homes exceed this
    seed: int                        # the sample's seed
    quantiles: tuple[float, ...]     # the nested ridge levels
    contrast: FieldContrast          # counts -> field: RawShare or PriorDeviance
    workers: int                     # fork pool for the all-pairs pass; changes nothing

    def __post_init__(self) -> None:
        q = self.quantiles
        if not (isinstance(q, tuple) and q and all(0.0 < a < 1.0 for a in q)
                and all(a < b for a, b in pairwise(q))):
            raise ValueError(
                f"quantiles must be a non-empty strictly increasing tuple in (0, 1), got {q!r}")
        if self.res_m <= 0 or self.r0_m < 0 or self.bend_lambda < 0:
            raise ValueError(f"res_m > 0, r0_m >= 0, bend_lambda >= 0 required: {self!r}")
        if self.max_sources < 1 or self.workers < 1:
            raise ValueError(f"max_sources and workers must be >= 1: {self!r}")

    @property
    def identity(self) -> Hashable | None:
        return config_identity(self, exempt=frozenset({"workers"}))

    def desire_field(self, block: Block) -> DesireField:
        params = CountParams(res_m=self.res_m, r0_m=self.r0_m, bend_lambda=self.bend_lambda,
                             max_sources=self.max_sources, seed=self.seed)
        raster, field = self.contrast.field(block, params, self.workers)
        return ridge_desire(raster, field, self.quantiles, block.crs)
