"""FlowPathsReblocker: paths where trips concentrate, not paths that serve everyone.

Every other reblocker here is a **drainage tree**: take the worst-served parcel, connect it, repeat
until nobody is left out. That objective cannot produce a footpath network, because real footpaths
are not trying to serve everyone. Measured against 25 blocks' real networks at matched
displacement, they leave people walking *twice as far* as a clearance tree (7.85 m vs 3.83 m mean)
and give a third fewer parcels frontage. They are a **circulation** network, not an access one.

The generative story that fits is trail formation: each person walks roughly the least-effort route
to where they are going, ground gets easier where it is trodden, walkers prefer trodden ground, and
paths emerge by positive feedback where many trips coincide (Helbing's active-walker model). This
implements that directly:

  1. route every origin-destination trip by cheapest path on the substrate
  2. accumulate how much traffic each edge carries
  3. optionally REINFORCE -- make used edges cheaper and re-route, so trips coalesce onto shared
     corridors instead of each finding its own private line
  4. keep the edges above a flow threshold

Steps 1-3 are flow accumulation; step 4 is what makes a network out of it. Because the threshold is
on *volume*, the same field yields a hierarchy: a low cut gives footpaths, a high cut gives the
arterial skeleton those footpaths feed. That is what makes this the natural candidate for street
networks at multiblock scale as well, where a drainage tree has no notion of a road being more
important than another.

Three properties follow from the construction rather than being tuned in, and all three are ones
the drainage methods lack: the network is SPARSE (only where flow concentrates), it can contain
LOOPS (different trips take different routes between the same places), and it does not reach every
parcel. Routing is on the substrate, so displacement stays low for the same reason it does
everywhere else here -- edges run between parcels, never through one.

Scored against real networks with `eval.agreement`, not permeability: mimicry is the goal, and on
permeability the real networks lose.
"""
from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Literal

import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import unary_union

from reblock.contracts import Block, Proposal
from reblock.methods.substrates import ChordSubstrate, RoutingGraph, Substrate
from reblock.permeability import DEFAULT_ROAD_WIDTH_M, with_width

Destination = Literal["gateway", "all_pairs"]


def _gateway_nodes(block: Block, graph: RoutingGraph) -> list[int]:
    """Substrate nodes on the block's street frontage -- where trips leave and enter."""
    street = unary_union(list(block.streets.geometry))
    on = shapely.dwithin(shapely.points(graph.pts), street, graph.net_tol)
    return [int(i) for i in np.flatnonzero(on)]


def accumulate_flow(
    block: Block, graph: RoutingGraph, *, destination: Destination = "gateway",
    iterations: int = 3, reinforcement: float = 0.5, max_sources: int = 400,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Traffic per undirected edge, from routing every parcel's trip and counting.

    `destination="gateway"` sends each parcel to the nearest street frontage -- the egress trip
    every household makes. `"all_pairs"` additionally routes parcel-to-parcel trips against a
    random sample of other parcels, which is what creates the *interior* through-routes real
    footpath networks have and a pure egress model cannot.

    `reinforcement` in (0, 1) is the trail-formation feedback: after each round, an edge's cost is
    multiplied by `1 - reinforcement * (its share of the peak flow)`, so trodden ground gets
    cheaper and later trips coalesce onto it. At `reinforcement=0` this degenerates to independent
    shortest paths, which is the honest ablation for whether the feedback matters at all.
    """
    pts, rows, cols, edist = graph.pts, graph.rows, graph.cols, graph.edist
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    mask = rows < cols
    ui, uj, ulen = rows[mask], cols[mask], edist[mask]
    n_edges = len(ui)
    if n_edges == 0:
        return np.zeros(0, dtype=np.float64)

    tree = cKDTree(pts)
    reps = np.array([[g.representative_point().x, g.representative_point().y]
                     for g in block.parcels.geometry])
    if len(reps) == 0:
        return np.zeros(n_edges, dtype=np.float64)
    sources = np.unique(tree.query(reps)[1])
    rng = np.random.default_rng(seed)
    if len(sources) > max_sources:
        sources = rng.choice(sources, max_sources, replace=False)
    gateways = _gateway_nodes(block, graph)
    if not gateways:
        return np.zeros(n_edges, dtype=np.float64)

    # Undirected edge key -> index, so a traced path can be counted.
    key_to_edge = {(int(min(a, b)), int(max(a, b))): e
                   for e, (a, b) in enumerate(zip(ui, uj, strict=True))}
    cost = ulen.astype(np.float64).copy()
    flow = np.zeros(n_edges, dtype=np.float64)

    for _ in range(max(iterations, 1)):
        flow = np.zeros(n_edges, dtype=np.float64)
        w = np.concatenate([cost, cost])
        r2 = np.concatenate([ui, uj])
        c2 = np.concatenate([uj, ui])
        csr = csr_matrix((w, (r2, c2)), shape=(n, n))

        targets: list[list[int]] = [gateways]
        if destination == "all_pairs":
            # A random peer for each source: interior trips, which egress-only routing never makes.
            targets.append(rng.choice(sources, size=len(sources), replace=True).tolist())

        for target in targets:
            _d, pred, _src = dijkstra(csr, indices=target, return_predecessors=True,
                                      min_only=True)
            for s in sources:
                node = int(s)
                while pred[node] >= 0:
                    nxt = int(pred[node])
                    e = key_to_edge.get((min(node, nxt), max(node, nxt)))
                    if e is not None:
                        flow[e] += 1.0
                    node = nxt

        peak = float(flow.max())
        if peak <= 0 or reinforcement <= 0:
            break
        cost = ulen * (1.0 - reinforcement * (flow / peak))
        cost = np.maximum(cost, ulen * 1e-3)      # never free: keeps Dijkstra well-posed
    return flow


@dataclass(frozen=True)
class FlowPathsIdentity:
    substrate: Hashable
    destination: str
    iterations: int
    reinforcement: float
    flow_quantile: float
    max_sources: int
    seed: int


@dataclass
class FlowPathsReblocker:
    """Keep the edges carrying the most traffic. `flow_quantile` is the hierarchy knob: a low cut
    keeps the worn footpath web, a high cut keeps only the arterial skeleton it feeds."""

    substrate: Substrate = field(default_factory=ChordSubstrate)
    destination: Destination = "all_pairs"
    iterations: int = 3
    reinforcement: float = 0.5
    flow_quantile: float = 0.90
    max_sources: int = 400
    seed: int = 0
    # Total width of the roads this method emits; stamped on every one. The metric has no
    # global corridor to fall back on.
    road_width_m: float = DEFAULT_ROAD_WIDTH_M

    @property
    def identity(self) -> FlowPathsIdentity | None:
        if self.substrate.identity is None:
            return None
        return FlowPathsIdentity(
            substrate=self.substrate.identity, destination=str(self.destination),
            iterations=int(self.iterations), reinforcement=float(self.reinforcement),
            flow_quantile=float(self.flow_quantile), max_sources=int(self.max_sources),
            seed=int(self.seed))

    def propose(self, block: Block, prior: Proposal | None = None) -> Proposal:
        del prior
        graph = self.substrate.build(block)
        flow = accumulate_flow(
            block, graph, destination=self.destination, iterations=self.iterations,
            reinforcement=self.reinforcement, max_sources=self.max_sources, seed=self.seed)
        mask = graph.rows < graph.cols
        ui, uj = graph.rows[mask], graph.cols[mask]
        used = flow > 0
        roads: list[LineString] = []
        if used.any():
            cut = float(np.quantile(flow[used], self.flow_quantile))
            keep = np.flatnonzero(flow >= max(cut, 1.0))
            # Highest-traffic first, so a budget prefix keeps the busiest corridors -- the
            # ordering every length/displacement-matched comparison in this repo consumes.
            for e in keep[np.argsort(-flow[keep])]:
                a, b = graph.pts[ui[e]], graph.pts[uj[e]]
                roads.append(LineString([(a[0], a[1]), (b[0], b[1])]))
        gdf = gpd.GeoDataFrame(geometry=roads, crs=block.crs)
        pid = (f"flow_paths:{self.substrate.tag}:{self.destination}:i{self.iterations}"
               f":r{self.reinforcement:g}:q{self.flow_quantile:g}")
        return Proposal(
            block_id=block.block_id, crs=block.crs, edges=None,
            roads=with_width(gdf, self.road_width_m),
            proposal_id=pid, method="flow_paths",
            params={"roads": len(roads), "substrate": self.substrate.tag,
                    "destination": self.destination, "iterations": self.iterations,
                    "reinforcement": self.reinforcement, "flow_quantile": self.flow_quantile},
            block_identity=block.identity if self.identity is not None else None)
