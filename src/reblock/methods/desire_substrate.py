"""`DesireWarpedSubstrate`: any substrate, its edges shortened along desire lines.

A method routes its shortest paths in whatever metric its substrate's `edist` defines. This
wrapper warps that metric toward a desire field: each edge costs `length / (eps + demand)^gamma`,
demand_greedy's attraction, where demand is the total weight of the desire groups whose corridors
(`buffer_m`) hold the edge's endpoints and midpoint. What a method BUILDS is still its own
choice -- cycle_native still scores each candidate loop by permeability gain per unit
displacement -- only the shape of each candidate moves toward where the field says people route.

Measured with the curvature-aware betweenness field around cycle_native on 36 large blocks: it
beats plain cycle_native on both lenses (docs/superpowers/notes/
2026-09-24-roadless-fields-and-the-betweenness-generator.md).

Not for `demand_greedy`, which applies the same attraction to the edges it is given: wrapped, its
field would count twice.
"""
from __future__ import annotations

import hashlib
from collections.abc import Hashable
from dataclasses import dataclass, replace

from reblock.contracts import Block
from reblock.derive_graph import closure_hash, config_identity
from reblock.methods.demand_greedy import demand_edge_weights
from reblock.methods.desire_lines import DesireLineSource
from reblock.methods.substrates import RoutingGraph, Substrate


@dataclass(frozen=True)
class DesireWarpedSubstrate:
    base: Substrate
    desire_source: DesireLineSource
    buffer_m: float
    eps: float
    gamma: float

    def __post_init__(self) -> None:
        if self.buffer_m < 0 or self.eps <= 0 or self.gamma <= 0:
            raise ValueError(f"DesireWarpedSubstrate needs buffer_m >= 0, eps > 0 and gamma > 0; "
                             f"got {self.buffer_m}, {self.eps}, {self.gamma}")

    @property
    def identity(self) -> Hashable:
        # A method holding this imports `substrates`, not this module, so this module's code
        # reaches the key only through its own closure hash.
        config = config_identity(self)
        return None if config is None else (closure_hash(__name__), config)

    @property
    def tag(self) -> str:
        """The base's tag and a digest of the configuration: a label names render files, and two
        desire sources over one base must not share one."""
        digest = hashlib.sha256(repr(config_identity(self)).encode()).hexdigest()[:8]
        return f"{self.base.tag}+desire:{digest}"

    def build(self, block: Block) -> RoutingGraph:
        graph = self.base.build(block)
        weights = demand_edge_weights(graph, self.desire_source.desire_field(block),
                                      buffer_m=self.buffer_m, eps=self.eps, gamma=self.gamma)
        return replace(graph, edist=weights)
