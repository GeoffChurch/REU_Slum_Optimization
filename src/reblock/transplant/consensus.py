"""Barycenter consensus: route the recipient's own substrate toward where several donors agree.

Each donor's network is transported onto the recipient (`transport`), buffered into a corridor,
and the corridors are summed under per-donor weights into one demand field on [0, 1]. The network
is then extracted by `clearance.greedy_drainage` on the recipient's substrate -- the same
worst-served-parcel-first drainage tree every shipped greedy method builds -- with each edge
costing `length / (eps + demand)^gamma`. Every road is a substrate edge, so none crosses a
building, and the tree still serves every parcel wherever the donors are silent.

This is `methods.demand_greedy` with a WEIGHTED field. `demand_greedy.demand_edge_weights` reads a
single set of lines as a binary corridor and so cannot carry per-donor weights; with one donor the
two are the same function (`tests/transplant/test_consensus.py` pins that). Generalizing it in
place would move the demand_greedy derivation's code hash, so the weighted form lives here.

What the k-sweep found this buys (docs/superpowers/notes/2026-07-28-consensus-k-sweep-and-
displacement.md): at k=1 the extraction beats gap-snapping the same donor by +0.303 permeability
in 95% of blocks, and k=30 adds nothing over k=1. The gain is the extraction, not the averaging.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.budget import street_first_ordered
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.methods.clearance import greedy_drainage
from reblock.methods.substrates import RoutingGraph, Substrate
from reblock.permeability import with_width
from reblock.transplant.gw import Arr
from reblock.transplant.transport import TransportParams, fit_transport, parcel_xy, transport_lines


@dataclass(frozen=True)
class ConsensusParams:
    substrate: Substrate    # the recipient graph the consensus is routed on
    buffer_m: float         # corridor radius around each transported network
    eps: float              # demand floor: an edge no corridor reaches costs length / eps^gamma
    gamma: float            # demand exponent
    depth_target: int       # stop once every parcel is within this many parcels of a road
    max_roads: int
    road_width_m: float     # stamped on every extracted road


@dataclass(frozen=True, eq=False)
class DonorFit:
    """One donor's network carried into the recipient's frame (pre-snap), and how far apart the
    two blocks' shapes are."""

    block_id: str
    transported: gpd.GeoDataFrame
    gw_dist: float


def fit_donors(recipient: Block, donors: Sequence[Block],
               networks: Mapping[str, gpd.GeoDataFrame],
               params: TransportParams) -> list[DonorFit]:
    """One GW fit per donor, in `donors` order -- the expensive half. Kept apart from extraction
    so a k-sweep fits once at the largest k and reuses the first k at every smaller one, which
    also nests the rungs: k=3's donors are a subset of k=15's."""
    r_xy = parcel_xy(recipient)
    fits: list[DonorFit] = []
    for donor in donors:
        fit = fit_transport(parcel_xy(donor), r_xy, params)
        fits.append(DonorFit(
            block_id=donor.block_id,
            transported=transport_lines(networks[donor.block_id], fit, crs=recipient.crs),
            gw_dist=fit.gw_dist))
    return fits


def consensus_weights(fits: Sequence[DonorFit], quality: Mapping[str, float]) -> list[float]:
    """`quality_i * exp(-gw_i / tau)`, tau the donors' median GW distance: a donor counts for
    more the better its own network and the closer its shape. Unnormalized."""
    tau = float(np.median([f.gw_dist for f in fits]))
    if not tau > 0.0:
        raise ValueError(f"median donor GW distance is {tau}; closeness exp(-gw/tau) needs tau > 0")
    weights: list[float] = []
    for f in fits:
        q = quality[f.block_id]
        if not np.isfinite(q):
            raise ValueError(f"donor {f.block_id} has non-finite quality {q}")
        weights.append(q * float(np.exp(-f.gw_dist / tau)))
    return weights


@dataclass(frozen=True, eq=False)
class ConsensusField:
    """Weighted corridors: the demand at a point is the total weight of the corridors holding it.
    Weights sum to 1, so demand lies in [0, 1]."""

    corridors: tuple[BaseGeometry, ...]
    weights: tuple[float, ...]

    @classmethod
    def of(cls, networks: Sequence[gpd.GeoDataFrame], weights: Sequence[float],
           buffer_m: float) -> ConsensusField:
        """Each network buffered by `buffer_m`, an empty one to an empty corridor. Weights are
        normalized to sum 1; if every weight is zero -- no donor's own network is worth anything
        -- they fall to uniform, which leaves the plain union of corridors."""
        corridors = tuple(unary_union(list(n.geometry)).buffer(buffer_m) if len(n) else Polygon()
                          for n in networks)
        total = sum(weights)
        norm = (tuple(w / total for w in weights) if total > 0
                else tuple(1.0 / len(weights) for _ in weights))
        return cls(corridors=corridors, weights=norm)

    def sample(self, pts: Arr) -> Arr:
        out = np.zeros(len(pts), dtype=np.float64)
        for corridor, w in zip(self.corridors, self.weights, strict=True):
            if w <= 0.0 or corridor.is_empty:
                continue
            out[shapely.contains_xy(corridor, pts[:, 0], pts[:, 1])] += w
        return out


def consensus_edge_weights(graph: RoutingGraph, field: ConsensusField, *, eps: float,
                           gamma: float) -> Arr:
    """`length / (eps + demand)^gamma` per edge, in the graph's symmetric COO order, with demand
    the mean of the field at both endpoints and the midpoint -- `demand_edge_weights`' three-point
    convention, so an edge whose middle leaves every corridor does not read as cheap."""
    pts, rows, cols, edist = graph.pts, graph.rows, graph.cols, graph.edist
    n = len(pts)
    mask = rows < cols                                  # one direction per undirected edge
    ui, uj, ulen = rows[mask], cols[mask], edist[mask]
    e = len(ui)
    if e == 0:
        return np.zeros(0, dtype=np.float64)
    dem = field.sample(np.vstack([pts[ui], pts[uj], (pts[ui] + pts[uj]) / 2.0]))
    mean_dem = (dem[:e] + dem[e:2 * e] + dem[2 * e:]) / 3.0
    uw = ulen / (eps + mean_dem) ** gamma
    key = np.minimum(rows, cols).astype(np.int64) * n + np.maximum(rows, cols).astype(np.int64)
    ukey = ui.astype(np.int64) * n + uj.astype(np.int64)
    order = np.argsort(ukey)
    return uw[order][np.searchsorted(ukey[order], key)]


def extract_consensus(recipient: Block, fits: Sequence[DonorFit], quality: Mapping[str, float],
                      params: ConsensusParams) -> gpd.GeoDataFrame:
    """The consensus network over `fits`, in greedy construction order, widths stamped."""
    field = ConsensusField.of([f.transported for f in fits], consensus_weights(fits, quality),
                              params.buffer_m)
    graph = params.substrate.build(recipient)
    roads, _ = greedy_drainage(
        recipient, graph, consensus_edge_weights(graph, field, eps=params.eps, gamma=params.gamma),
        depth_target=params.depth_target, max_roads=params.max_roads)
    return with_width(roads, params.road_width_m)


def length_matched_prefix(block: Block, roads: gpd.GeoDataFrame,
                          target_m: float) -> gpd.GeoDataFrame:
    """The shortest street-first prefix (`budget.street_first_ordered`, the canonical order every
    lens truncates in) whose length reaches `target_m`, so it overshoots by at most one road; all
    of `roads` if even that falls short."""
    if len(roads) == 0:
        return roads
    ordered = street_first_ordered(block, roads, STREET_TOL)
    cum = ordered.geometry.length.to_numpy().cumsum()
    m = min(int(np.searchsorted(cum, target_m)) + 1, len(ordered))
    return cast(gpd.GeoDataFrame, ordered.iloc[:m].reset_index(drop=True))
