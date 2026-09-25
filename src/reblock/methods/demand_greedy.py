"""DemandGreedyReblocker: a drainage tree routed toward where people already walk.

Same greedy drainage-tree construction as `ClearanceReblocker` -- take the worst-served parcel,
Dijkstra it to the growing network, repeat -- but the edge cost is *attraction to a demand field*
rather than *repulsion from buildings*:

    clearance      edge = length * mean(node cost),  node cost = (1-t) + t / clearance
    demand_greedy  edge = length / (eps + demand)^gamma

The demand field is a `DesireField` -- weighted groups of desire lines, each buffered into a
corridor -- and the demand at a point is the total weight of the corridors holding it. Real OSM
footpaths are one group at weight 1, so their demand is binary; a consensus of transplanted donor
networks is one group per donor. Where people already walk is cheap to route through; everywhere
else costs full length. Both costs sample at three points (endpoints and midpoint) so a long edge
whose middle strays does not read as cheap.

Why this is not just `osm_footpaths` again. That method proposes the real footpaths *as the roads*,
which inherits their coverage gaps -- a sparse skeleton that leaves the interior unserved (access
0.026 on the deep region). This one uses them only as a PRIOR: the drainage tree still guarantees
every parcel reaches a road, still routes on the substrate so it can never cut a building, and
still terminates on the street. It is "build a buildable, complete network, biased toward the lines
people have already worn."

Provenance: the mechanism is `demand_greedy_reblock` from the 2026-07-23 OT-transplant spike, where
the demand field came from GW-transported donor networks -- which is still how the consensus preset
drives it (`conf/method/consensus.yaml`). That arc closed -- transplanted networks do not beat a
direct clearance solve at matched displacement -- but the extraction step was the one thing in it
that clearly worked: +0.303 permeability over gap-snapping the donor geometry, in 95% of blocks
(p<0.0001). Crucially that gain did not depend on the donors: one donor scored as well as thirty
(k=1 to k=30 was worth -0.009, p=0.064), which is what suggests the field carries very little
information and almost any reasonable prior would do. See
docs/superpowers/notes/2026-07-28-consensus-k-sweep-and-displacement.md.

With `desire_source=NoDesire()` the field is uniform, every edge costs its own length, and this
reduces to a pure shortest-path drainage tree -- the honest ablation for "how much is the prior
worth?".
"""
from __future__ import annotations

import hashlib
from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.ops import unary_union

from reblock.contracts import Block, Proposal
from reblock.derive_graph import config_identity
from reblock.methods.clearance import greedy_drainage
from reblock.methods.desire_lines import DesireField, DesireLineSource
from reblock.methods.substrates import RoutingGraph, Substrate
from reblock.permeability import with_width


def field_demand(field: DesireField, xy: NDArray[np.float64],
                 buffer_m: float) -> NDArray[np.float64]:
    """The demand at each point: the total weight of the groups whose lines pass within
    `buffer_m` of it (the corridor's own boundary counts as within). Groups add in field order."""
    out = np.zeros(len(xy), dtype=np.float64)
    for group in field.groups:
        if group.lines.empty:
            continue
        corridor = unary_union(list(group.lines.geometry)).buffer(buffer_m)
        out[shapely.intersects_xy(corridor, xy[:, 0], xy[:, 1])] += group.weight
    return out


def demand_edge_weights(
    graph: RoutingGraph, field: DesireField, *, buffer_m: float, eps: float, gamma: float,
) -> NDArray[np.float64]:
    """`length / (eps + demand)^gamma` per edge, demand the mean of `field_demand` at both
    endpoints and the midpoint.

    Three-point sampling mirrors `clearance._edge_weights`' own convention: an edge whose endpoints
    happen to touch the corridor but whose middle leaves it must not read as cheap. `eps` guards the
    divide where no corridor reaches, and prices demand: at gamma=1 an edge at demand 1 is
    (eps+1)/eps times cheaper than one outside every corridor.
    """
    pts, rows, cols, edist = graph.pts, graph.rows, graph.cols, graph.edist
    n = len(pts)
    mask = rows < cols                                  # one direction per undirected edge
    ui, uj, ulen = rows[mask], cols[mask], edist[mask]
    if len(ui) == 0:
        return np.zeros(0, dtype=np.float64)

    e = len(ui)
    dem = field_demand(field, np.vstack([pts[ui], pts[uj], (pts[ui] + pts[uj]) / 2.0]), buffer_m)
    mean_dem = (dem[:e] + dem[e:2 * e] + dem[2 * e:]) / 3.0
    uw = ulen / (eps + mean_dem) ** gamma

    # Re-expand to the symmetric COO order the graph hands out.
    key = np.minimum(rows, cols).astype(np.int64) * n + np.maximum(rows, cols).astype(np.int64)
    ukey = ui.astype(np.int64) * n + uj.astype(np.int64)
    order = np.argsort(ukey)
    return np.asarray(uw[order][np.searchsorted(ukey[order], key)], dtype=np.float64)


@dataclass
class DemandGreedyReblocker:
    """Greedy drainage tree whose routing is attracted to a desire-line demand field."""

    desire_source: DesireLineSource
    substrate: Substrate
    buffer_m: float
    eps: float
    gamma: float
    depth_target: int
    max_roads: int
    # Total width of the roads this method emits; stamped on every one. The metric has no
    # global corridor to fall back on.
    road_width_m: float

    @property
    def identity(self) -> Hashable | None:
        return config_identity(self)

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior  # accepted for Method conformance; routing is block-only
        field = self.desire_source.desire_field(block)
        graph = self.substrate.build(block)
        weights = demand_edge_weights(graph, field, buffer_m=self.buffer_m, eps=self.eps,
                                      gamma=self.gamma)
        roads, params = greedy_drainage(block, graph, weights, depth_target=self.depth_target,
                                        max_roads=self.max_roads)
        # The source is hashed into the label the way `osm_footpaths` hashes its own: the label
        # names render files, and two sources on one block propose different roads.
        source = self.desire_source.identity
        src = "" if source is None else f":{hashlib.sha256(str(source).encode()).hexdigest()[:8]}"
        pid = (f"demand_greedy{src}:{self.substrate.tag}:b{self.buffer_m:g}:e{self.eps:g}"
               f":g{self.gamma:g}:d{self.depth_target}:mr{self.max_roads}:w{self.road_width_m:g}")
        return Proposal(
            block_id=block.block_id, crs=block.crs, edges=None,
            roads=with_width(roads, self.road_width_m),
            proposal_id=pid, method="demand_greedy",
            params={**params, "substrate": self.substrate.tag, "buffer_m": self.buffer_m,
                    "eps": self.eps, "gamma": self.gamma, "depth_target": self.depth_target,
                    "demand_segments": field.n_lines, "demand_groups": len(field.groups)})
