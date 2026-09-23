"""Barycenter consensus: route the recipient's own substrate toward where several donors agree.

`ConsensusDesireSource` is a `DesireLineSource`. For a recipient it fits its donors (`donors`),
and returns each one's transported footpaths as one group of a desire field, weighted
`quality_i * exp(-gw_i / tau)` and normalized to sum 1, so demand lies in [0, 1]. The network is
then extracted by `demand_greedy.DemandGreedyReblocker` like any other field: worst-served parcel
first on the recipient's substrate, each edge costing `length / (eps + demand)^gamma`. Every road is
a substrate edge, so none crosses a building, and the tree still serves every parcel wherever the
donors are silent. `conf/method/consensus.yaml` is the operating point every published consensus
result was measured at; `conf/donors/` holds its held-out and leaky donor draws.

It reaches `demand_greedy` only by configuration, never by import, so this package stays out of
every shipped module's code closure (`tests/transplant/test_isolation.py`). What keeps a cached
consensus proposal honest instead is `identity`, which carries this module's own closure hash.

What the k-sweep found this buys (docs/superpowers/notes/2026-07-28-consensus-k-sweep-and-
displacement.md): at k=1 the extraction beats gap-snapping the same donor by +0.303 permeability
in 95% of blocks, and k=30 adds nothing over k=1. The gain is the extraction, not the averaging.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import geopandas as gpd
import numpy as np

from reblock.contracts import Block
from reblock.derive_graph import closure_hash, config_identity, derive
from reblock.emit import pct_displaced
from reblock.methods.desire_lines import DesireField, WeightedLines
from reblock.permeability import EgressContext, PermeabilityParams, permeability, with_width
from reblock.transplant.donors import DonorFit, Donors


def consensus_weights(fits: Sequence[DonorFit], quality: Mapping[str, float]) -> list[float]:
    """`quality_i * exp(-gw_i / tau)`, tau the donors' median GW distance: a donor counts for
    more the better its own network and the closer its shape. Unnormalized."""
    tau = float(np.median([f.gw_dist for f in fits]))
    if not tau > 0.0:
        raise ValueError(f"median donor GW distance is {tau}; closeness exp(-gw/tau) needs tau > 0")
    weights: list[float] = []
    for f in fits:
        q = quality[f.donor.block_id]
        if not np.isfinite(q):
            raise ValueError(f"donor {f.donor.block_id} has non-finite quality {q}")
        weights.append(q * float(np.exp(-f.gw_dist / tau)))
    return weights


def normalized(weights: Sequence[float]) -> tuple[float, ...]:
    """Weights over their sum. If every one is zero -- no donor's own network is worth anything --
    they fall to uniform, which leaves the plain union of corridors."""
    total = sum(weights)
    return (tuple(w / total for w in weights) if total > 0
            else tuple(1.0 / len(weights) for _ in weights))


@dataclass(frozen=True, eq=False)
class _QualityInput:
    """Identified carrier: how good one donor's own network is on its own block."""

    donor: Block
    footpaths: gpd.GeoDataFrame
    footpaths_identity: Hashable | None
    params: PermeabilityParams
    road_width_m: float

    @property
    def identity(self) -> Hashable | None:
        params = config_identity(self.params)
        if self.footpaths_identity is None or params is None:
            return None
        return ("donor_quality", self.footpaths_identity, params, self.road_width_m)


def _quality_impl(inp: _QualityInput) -> float:
    """`own_perm * (1 - own_disp)`, the donor's footpaths scored as the streets `osm_footpaths`
    would build on them. A donor with an excellent network is worth more in the consensus than a
    close-but-poorly-served one, which is why weight is quality x proximity, not proximity alone."""
    roads = with_width(inp.footpaths, inp.road_width_m)
    perm = permeability(EgressContext.of(inp.donor, inp.params), roads)
    return float(perm * (1.0 - pct_displaced(roads, inp.donor.buildings)))


@dataclass(frozen=True)
class ConsensusDesireSource:
    """A recipient's desire field: its donors' transported footpaths, one weighted group each."""

    donors: Donors
    permeability: PermeabilityParams    # the metric a donor's own network is judged by
    road_width_m: float                 # the street a donor's footpath is judged as

    @property
    def identity(self) -> Hashable | None:
        config = config_identity(self)
        return None if config is None else (closure_hash(__name__), config)

    def quality(self, fit: DonorFit) -> float:
        return derive(_quality_impl, _QualityInput(
            donor=fit.donor, footpaths=fit.footpaths, footpaths_identity=fit.footpaths_identity,
            params=self.permeability, road_width_m=self.road_width_m))

    def desire_field(self, block: Block) -> DesireField:
        fits = self.donors.fits(block)
        quality = {f.donor.block_id: self.quality(f) for f in fits}
        weights = normalized(consensus_weights(fits, quality))
        return DesireField(groups=tuple(WeightedLines(lines=f.transported, weight=w)
                                        for f, w in zip(fits, weights, strict=True)))
