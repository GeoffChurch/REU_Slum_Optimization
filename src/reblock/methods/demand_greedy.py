"""DemandGreedyReblocker: a drainage tree routed toward where people already walk.

Same greedy drainage-tree construction as `ClearanceReblocker` -- take the worst-served parcel,
Dijkstra it to the growing network, repeat -- but the edge cost is *attraction to a demand field*
rather than *repulsion from buildings*:

    clearance      edge = length * mean(node cost),  node cost = (1-t) + t / clearance
    demand_greedy  edge = length / (eps + demand)^gamma

The demand field is a set of real desire lines, buffered. Where people already walk is cheap to
route through; everywhere else costs full length. Both sample cost at three points (endpoints and
midpoint) so a long edge whose middle strays does not read as cheap.

Why this is not just `osm_footpaths` again. That method proposes the real footpaths *as the roads*,
which inherits their coverage gaps -- a sparse skeleton that leaves the interior unserved (access
0.026 on the deep region). This one uses them only as a PRIOR: the drainage tree still guarantees
every parcel reaches a road, still routes on the substrate so it can never cut a building, and
still terminates on the street. It is "build a buildable, complete network, biased toward the lines
people have already worn."

Provenance: the mechanism is `demand_greedy_reblock` from the 2026-07-23 OT-transplant spike, where
the demand field came from GW-transported donor networks. That arc closed -- transplanted networks
do not beat a direct clearance solve at matched displacement -- but the extraction step was the one
thing in it that clearly worked: +0.303 permeability over gap-snapping the donor geometry, in 95%
of blocks (p<0.0001). Crucially that gain did not depend on the donors: one donor scored as well as
thirty (k=1 to k=30 was worth -0.009, p=0.064), which is what suggests the field carries very
little information and almost any reasonable prior would do. See
docs/superpowers/notes/2026-07-28-consensus-k-sweep-and-displacement.md.

With `desire_source=None` the field is uniform, every edge costs its own length, and this reduces
to a pure shortest-path drainage tree -- the honest ablation for "how much is the prior worth?".
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
from numpy.typing import NDArray
from shapely import STRtree
from shapely.geometry import Point
from shapely.ops import unary_union

from reblock.contracts import Block, Proposal
from reblock.methods.clearance import greedy_drainage
from reblock.methods.desire_lines import DesireLineSource
from reblock.methods.osm_footpaths import interior_desire_lines
from reblock.methods.substrates import ChordSubstrate, RoutingGraph, Substrate

# Guards the divide when an edge sits entirely outside the demand field, and sets how much cheaper
# a fully-in-demand edge is than a fully-outside one: at gamma=1 the ratio is (eps+1)/eps = 11x.
DEMAND_EPS = 0.1
DEMAND_GAMMA = 1.0
BUFFER_M = 3.0


def demand_edge_weights(
    graph: RoutingGraph, demand: gpd.GeoDataFrame, *,
    buffer_m: float = BUFFER_M, eps: float = DEMAND_EPS, gamma: float = DEMAND_GAMMA,
) -> NDArray[np.float64]:
    """`length / (eps + demand)^gamma` per edge, demand sampled at both endpoints and the midpoint.

    Three-point sampling mirrors `clearance._edge_weights`' own convention: an edge whose endpoints
    happen to touch the corridor but whose middle leaves it must not read as cheap.
    """
    pts, rows, cols, edist = graph.pts, graph.rows, graph.cols, graph.edist
    n = len(pts)
    mask = rows < cols                                  # one direction per undirected edge
    ui, uj, ulen = rows[mask], cols[mask], edist[mask]
    if len(ui) == 0:
        return np.zeros(0, dtype=np.float64)

    if demand.empty:
        dem = np.zeros(3 * len(ui), dtype=np.float64)
    else:
        corridor = unary_union(list(demand.geometry)).buffer(buffer_m)
        tree = STRtree([corridor])
        mid = (pts[ui] + pts[uj]) / 2.0
        sample = np.vstack([pts[ui], pts[uj], mid])
        hit = np.zeros(len(sample), dtype=np.float64)
        idx = tree.query(np.array([Point(x, y) for x, y in sample]), predicate="intersects")
        hit[np.unique(idx[0])] = 1.0
        dem = hit
    e = len(ui)
    mean_dem = (dem[:e] + dem[e:2 * e] + dem[2 * e:]) / 3.0
    uw = ulen / (eps + mean_dem) ** gamma

    # Re-expand to the symmetric COO order the graph hands out.
    key = np.minimum(rows, cols).astype(np.int64) * n + np.maximum(rows, cols).astype(np.int64)
    ukey = ui.astype(np.int64) * n + uj.astype(np.int64)
    order = np.argsort(ukey)
    return np.asarray(uw[order][np.searchsorted(ukey[order], key)], dtype=np.float64)


@dataclass(frozen=True)
class DemandGreedyIdentity:
    """Cache-key identity. `demand` is the desire source's own identity, or None when the source
    is live/uncacheable -- which propagates None upward exactly as ClearanceReblocker does for an
    uncacheable substrate, so a live OSM fetch can never serve a stale memoized proposal."""

    substrate: Hashable
    demand: Hashable
    buffer_m: float
    eps: float
    gamma: float
    depth_target: int
    max_roads: int


@dataclass
class DemandGreedyReblocker:
    """Greedy drainage tree whose routing is attracted to a desire-line demand field."""

    desire_source: DesireLineSource | None = None
    substrate: Substrate = field(default_factory=ChordSubstrate)
    buffer_m: float = BUFFER_M
    eps: float = DEMAND_EPS
    gamma: float = DEMAND_GAMMA
    depth_target: int = 2
    max_roads: int = 400

    @property
    def identity(self) -> DemandGreedyIdentity | None:
        if self.substrate.identity is None:
            return None
        demand_id: Hashable = "uniform"
        if self.desire_source is not None:
            src_id = self.desire_source.identity
            if src_id is None:                    # live fetch: uncacheable, propagate up
                return None
            demand_id = src_id
        return DemandGreedyIdentity(
            substrate=self.substrate.identity, demand=demand_id, buffer_m=float(self.buffer_m),
            eps=float(self.eps), gamma=float(self.gamma), depth_target=int(self.depth_target),
            max_roads=int(self.max_roads))

    def _demand(self, block: Block) -> gpd.GeoDataFrame:
        if self.desire_source is None:
            return gpd.GeoDataFrame(geometry=[], crs=block.crs)
        b = gpd.GeoSeries([block.boundary], crs=block.crs).to_crs(4326).total_bounds
        bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        lines = self.desire_source.desire_lines(bbox, block.crs)
        streets = unary_union(list(block.streets.geometry))
        return interior_desire_lines(lines, block.boundary, streets, block.crs)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; routing is block-only
        demand = self._demand(block)
        graph = self.substrate.build(block)
        weights = demand_edge_weights(graph, demand, buffer_m=self.buffer_m, eps=self.eps,
                                      gamma=self.gamma)
        roads, params = greedy_drainage(block, graph, weights, depth_target=self.depth_target,
                                        max_roads=self.max_roads)
        pid = (f"demand_greedy:{self.substrate.tag}:b{self.buffer_m:g}:e{self.eps:g}"
               f":g{self.gamma:g}:d{self.depth_target}:mr{self.max_roads}")
        return Proposal(
            block_id=block.block_id, crs=block.crs, roads=roads, edges=None,
            proposal_id=pid, method="demand_greedy",
            params={**params, "substrate": self.substrate.tag, "buffer_m": self.buffer_m,
                    "eps": self.eps, "gamma": self.gamma, "depth_target": self.depth_target,
                    "demand_segments": int(len(demand))},
            block_identity=block.identity if self.identity is not None else None)
