"""Strong orientation: which roads can be one-way, and which way they run.

Robbins' theorem: a connected graph has a strongly-connected orientation **iff it is bridgeless**.
That is the whole rule. A bridge must stay two-way -- orienting it strands everything beyond it --
while every edge in a 2-edge-connected component can be made one-way with the component still
reachable from itself.

## Why the street is one node

A method's loop runs from the street, out through the block, and back to the street at a different
point. Without the street in the graph that reads as a simple PATH -- every edge a bridge, nothing
orientable. But the two ends are joined, by the street itself: you can always drive along it. So
every street-touching node is contracted into a single super-node, which is exactly the model the
metric already uses (ground is one eliminated node at potential 0, not a set of separate nodes).
With that contraction a street-to-street loop is a genuine cycle and orients.

## Which direction

DFS from the street super-node: tree edges away from it, non-tree edges back toward it (descendant
-> ancestor). That is the textbook Robbins construction and it is strongly connected exactly when
the component is bridgeless.

Orienting for EGRESS specifically -- every edge along the direction traffic wants to go -- would
score better, and it is what an earlier probe did. It is also the out-tree that serves egress
perfectly and fails ingress completely, which is why `permeability` scores directed road sets as
egress and ingress halved. There is no free lunch to collect by choosing a self-serving
orientation, so this takes the canonical one.

## Granularity: rows are SPLIT, not refused

Orientation is decided per graph EDGE, but a road ROW is what carries `width_m` and what the lenses
walk in prefix order. A method emits whole Dijkstra paths, and one of those can cross several
2-edge-connected components, so its segments do not all orient the same way.

Refusing such a row outright is the obvious rule and it is far too lossy: MEASURED on 8 real blocks
it oriented 39% of road length against a Robbins ceiling of 95%, leaving 56 points of the discount
unclaimed and understating one-way's case badly (`scratchpad/width/orient_coverage.py`). So a row is
instead split into maximal runs of consistently-oriented segments, each emitted as its own road.

That changes row granularity, and finer rows let a method stop closer to a lens target. `oneway=
False` therefore performs the IDENTICAL split and leaves everything two-way at full width -- it is
the granularity-matched control arm, so a comparison against it varies direction and width only.
"""
from __future__ import annotations

from typing import cast

import networkx as nx
from geopandas import GeoDataFrame
from shapely.geometry import LineString

from reblock.budget import _rnd, _road_net
from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.permeability import ONEWAY_COL, WIDTH_COL, PermeabilityParams

_Node = tuple[float, float]
STREET_NODE: _Node = (float("inf"), float("inf"))
"""The contracted street super-node. Not a coordinate -- no road vertex can collide with it."""


def one_way_width(params: PermeabilityParams, two_way_width_m: float) -> float:
    """Width at which a one-way road gives its permitted direction the conductance a two-way road of
    `two_way_width_m` gives each of its two.

    Solving `lane_width(W_one, oneway=True) == lane_width(W_two, oneway=False)` gives
    `W_one = (W_two + margin) / 2` -- 3.5 m for a 6 m two-way road at the default 1 m margin, NOT
    3.0 m. Both roads pay the margin once, so a one-way street is wider than half a two-way one.
    That is derived from the margin, not asserted.
    """
    return (two_way_width_m + params.road_margin_m) / 2.0


def _oriented_edges(g: nx.Graph) -> dict[frozenset[_Node], tuple[_Node, _Node]]:
    """Robbins orientation per component: tree edges away from the root, back edges toward it.

    Bridges are omitted -- they have no strong orientation and their rows stay two-way.
    """
    out: dict[frozenset[_Node], tuple[_Node, _Node]] = {}
    for comp in nx.connected_components(g):
        sub = g.subgraph(comp)
        if sub.number_of_edges() == 0:
            continue
        bridges = {frozenset(e) for e in nx.bridges(sub)}
        root = STREET_NODE if STREET_NODE in comp else next(iter(sorted(comp)))
        rank = {node: i for i, node in enumerate(nx.dfs_preorder_nodes(sub, source=root))}
        tree = {frozenset((u, v)) for u, v in nx.dfs_edges(sub, source=root)}
        for u, v in sub.edges:
            key = frozenset((u, v))
            if key in bridges or u not in rank or v not in rank:
                continue
            # tree edge: parent -> child (away from root). back edge: descendant -> ancestor.
            out[key] = (u, v) if (rank[u] < rank[v]) == (key in tree) else (v, u)
    return out


def strong_orientation(block: Block, roads: GeoDataFrame, *, params: PermeabilityParams,
                       oneway: bool = True, tol: float = STREET_TOL) -> GeoDataFrame:
    """Split every road at its orientation boundaries and, with `oneway`, point each run one way.

    Returns a frame with one row per maximal run of consistently-oriented segments, carrying every
    non-geometry column of the row it came from. Each one-way run's coordinate order IS its
    permitted direction (which is how `edge_conductances` reads direction), and its `width_m` is
    reduced to `one_way_width`. Bridges keep their original width and stay two-way.

    `oneway=False` performs the identical split and orients nothing -- the granularity-matched
    control described in the module docstring.
    """
    if WIDTH_COL not in roads.columns:
        raise ValueError(f"roads must carry a '{WIDTH_COL}' column before they can be oriented")
    if len(roads) == 0:
        out = roads.copy()
        out[ONEWAY_COL] = False
        return out

    net = _road_net(block, roads, tol)
    g: nx.Graph = nx.Graph(net.graph)
    for node in net.street_nodes:                     # contract the street into one super-node
        for nbr in list(g.neighbors(node)):
            if nbr != STREET_NODE:
                g.add_edge(STREET_NODE, nbr, weight=g[node][nbr].get("weight", 0.0))
        g.remove_node(node)
    street_set = set(net.street_nodes)
    oriented = _oriented_edges(g)

    def contracted(node: _Node) -> _Node:
        return STREET_NODE if node in street_set else node

    src, geoms, widths, flags = [], [], [], []
    for i, geom in enumerate(roads.geometry):
        base_w = float(roads[WIDTH_COL].iloc[i])
        for part in (list(geom.geoms) if hasattr(geom, "geoms") else [geom]):
            cs = list(part.coords)
            # per segment: True = the orientation runs ALONG this row's coordinate order, False =
            # against it, None = unorientable (a bridge, or contracted to a street self-loop)
            states: list[bool | None] = []
            for a, b in zip(cs, cs[1:], strict=False):
                na, nb = contracted(_rnd(a)), contracted(_rnd(b))
                d = None if na == nb else oriented.get(frozenset((na, nb)))
                states.append(None if d is None else d == (na, nb))
            for lo, hi, state in _runs(states):
                run = LineString(cs[lo:hi + 2])
                one = oneway and state is not None
                src.append(i)
                geoms.append(run.reverse() if (one and state is False) else run)
                widths.append(one_way_width(params, base_w) if one else base_w)
                flags.append(one)

    out = roads.iloc[src].reset_index(drop=True).set_geometry(geoms, crs=roads.crs)
    out[WIDTH_COL] = widths
    out[ONEWAY_COL] = flags
    return cast(GeoDataFrame, out)


def _runs(states: list[bool | None]) -> list[tuple[int, int, bool | None]]:
    """Maximal runs of equal state as `(first index, last index, state)`."""
    out: list[tuple[int, int, bool | None]] = []
    for k, s in enumerate(states):
        if out and out[-1][2] is s:
            out[-1] = (out[-1][0], k, s)
        else:
            out.append((k, k, s))
    return out


def bridge_fraction(block: Block, roads: GeoDataFrame, *, tol: float = STREET_TOL) -> float:
    """Fraction of road LENGTH that is a bridge -- i.e. that cannot be made one-way.

    0.0 means every metre is orientable (the cycle end); 1.0 is a pure tree.
    """
    if roads is None or len(roads) == 0:
        return 0.0
    net = _road_net(block, roads, tol)
    g: nx.Graph = nx.Graph(net.graph)
    for node in net.street_nodes:
        for nbr in list(g.neighbors(node)):
            if nbr != STREET_NODE:
                g.add_edge(STREET_NODE, nbr, weight=g[node][nbr].get("weight", 0.0))
        g.remove_node(node)
    total = bridged = 0.0
    for comp in nx.connected_components(g):
        sub = g.subgraph(comp)
        if sub.number_of_edges() == 0:
            continue
        bridges = {frozenset(e) for e in nx.bridges(sub)}
        for u, v, data in sub.edges(data=True):
            w = float(data.get("weight", LineString([u, v]).length))
            total += w
            if frozenset((u, v)) in bridges:
                bridged += w
    return bridged / total if total > 0 else 0.0
