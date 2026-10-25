"""The damage term for vehicles, on the road graph alone (BACKLOG, "Try everything": expected P
over single blockages from one solve and each road piece's leverage; agenda item 5).

The graph: the block's streets and a network's roads (snapped to the streets and to each other
within budget.STREET_TOL), noded where they cross and cut into pieces of at most `step_m`, each
piece an edge of conductance 1 / its length; every node on a street is the ground (a vehicle on
the street is out). Pieces apart from the ground (a network's floating parts) are dropped. Each
home (a standing building, its population common.AreaPopulation's times the share the roads leave
standing, common.MEASURE) is served by the nearest node within `carry_m` of its outline (a
stretcher's carry to where a vehicle reaches); its cost is that node's effective resistance to
the ground (metres of road, parallel routes lowering it), and the network's P_v the
population-weighted mean over the served homes. A home served from the street itself costs 0.

One blockage is one road piece removed, each with probability its share of the network's length.
Removing piece e = (a, b) of conductance g changes the inverse G of the grounded Laplacian by
Sherman-Morrison, g G[:, a-b] G[a-b, :] / (1 - g R_e), R_e = G[a-b, a-b] its effective resistance
and g R_e its leverage, so every home's cost rises by g (G[h, a] - G[h, b])^2 / (1 - g R_e): one
inverse for every piece. A bridge (leverage 1) cuts the homes behind it off instead; the others'
costs are unchanged (the part behind hangs from the bridge, holding no ground). Two numbers, no
price on being cut off:
    cut:    the expected share of the served population a blockage cuts off;
    detour: the expected relative rise of P_v over blockages that cut no one off.

    PYTHONPATH=. uv run python research/roadless/damage.py check <block> <network.parquet>
        <step_m> <carry_m> [none|no-denominator|no-bridges]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import geopandas as gpd
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import shapely
from numpy.typing import NDArray
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402
from reblock.derive.access import STREET_TOL  # noqa: E402

BRIDGE = 1e-9        # 1 - leverage under this is a bridge (float dust on an exact 0)
NEAR_NODES = 8       # nodes searched per home (by centroid) for the one nearest its outline


class Graph(NamedTuple):
    """The road graph: per edge its ends (`a`, `b`: node indices, the ground -1) and length;
    per home its node (-1 the ground, -2 unserved) and weight."""
    a: NDArray[np.int64]
    b: NDArray[np.int64]
    length: NDArray[np.float64]
    n: int
    home: NDArray[np.int64]
    w: NDArray[np.float64]

    @staticmethod
    def of(block, roads: gpd.GeoDataFrame, *, step_m: float,  # type: ignore[no-untyped-def]
           carry_m: float) -> Graph:
        street = shapely.union_all(block.streets.geometry.to_numpy())
        lines = shapely.snap(roads.geometry.to_numpy(), street, STREET_TOL)
        mesh = shapely.union_all(np.concatenate([shapely.get_parts(street), lines]))
        mesh = shapely.segmentize(mesh, step_m)
        segs = []
        for part in shapely.get_parts(mesh):
            xy = shapely.get_coordinates(part)
            segs.append(np.stack([xy[:-1], xy[1:]], axis=1))
        seg = np.concatenate(segs)
        seg = seg[np.hypot(*(seg[:, 1] - seg[:, 0]).T) > 1e-9]
        pts, inv = np.unique(np.round(seg.reshape(-1, 2), 6), axis=0, return_inverse=True)
        inv = inv.reshape(-1, 2)
        ground = shapely.distance(shapely.points(pts), street) <= STREET_TOL
        # the ground merged into one node; edges inside it dropped
        g = nx.Graph()
        GROUND = -1
        node = np.where(ground, GROUND, np.arange(len(pts)))
        for (i, j), s in zip(inv, seg, strict=True):
            u, v = int(node[i]), int(node[j])
            if u != v:
                L = float(np.hypot(*(s[1] - s[0])))
                if g.has_edge(u, v):         # parallel pieces: conductances add
                    g[u][v]["g"] += 1.0 / L
                    g[u][v]["length"] += L
                else:
                    g.add_edge(u, v, g=1.0 / L, length=L)
        g.add_node(GROUND)
        keep = nx.node_connected_component(g, GROUND)
        g = g.subgraph(keep).copy()
        ids = sorted(v for v in g.nodes if v != GROUND)
        index = {v: k for k, v in enumerate(ids)} | {GROUND: -1}
        a = np.array([index[u] for u, _ in g.edges], dtype=np.int64)
        b = np.array([index[v] for _, v in g.edges], dtype=np.int64)
        length = np.array([1.0 / g[u][v]["g"] for u, v in g.edges])
        # homes: the standing population, served by the nearest node within carry_m
        outlines = np.asarray(block.buildings.outlines)
        c = (common.MEASURE.c(block.buildings, road_corridor(roads)) if len(roads)
             else np.zeros(len(outlines)))
        w = common.AreaPopulation().weights(outlines) * (1.0 - c)
        xy = np.array([pts[v] for v in ids]) if ids else np.zeros((0, 2))
        home = np.full(len(outlines), -2, dtype=np.int64)
        d_street = shapely.distance(outlines, street)
        home[d_street <= carry_m] = -1
        if len(ids):
            tree = cKDTree(xy)
            cen = shapely.get_coordinates(shapely.centroid(outlines))
            _, k = tree.query(cen, k=min(NEAR_NODES, len(ids)))
            k = k.reshape(len(outlines), -1)
            d = shapely.distance(outlines[:, None], shapely.points(xy[k]))
            best = d.argmin(axis=1)
            dn = d[np.arange(len(outlines)), best]
            road = (dn <= carry_m) & (dn < d_street)
            home[road] = k[np.arange(len(outlines)), best][road]
        return Graph(a=a, b=b, length=length, n=len(ids), home=home, w=w)

    def laplacian(self, drop: int | None = None) -> sp.csc_matrix:
        """The grounded Laplacian (the ground's row and column removed), edge `drop` left out."""
        keep = np.ones(len(self.a), dtype=bool)
        if drop is not None:
            keep[drop] = False
        a, b, g = self.a[keep], self.b[keep], 1.0 / self.length[keep]
        rows, cols, vals = [], [], []
        for u, v, gv in zip(a, b, g, strict=True):
            for p, q, s in ((u, u, gv), (v, v, gv), (u, v, -gv), (v, u, -gv)):
                if p >= 0 and q >= 0:
                    rows.append(p)
                    cols.append(q)
                    vals.append(s)
        return sp.csc_matrix((vals, (rows, cols)), shape=(self.n, self.n))


class Damage(NamedTuple):
    P: float          # the served homes' mean cost (m of road)
    served: float     # the served share of the standing population
    cut: float        # the expected share of the served population one blockage cuts off
    detour: float     # the expected relative rise of P over blockages that cut no one off
    bridges: float    # the share of the network's length on bridges


class Pieces(NamedTuple):
    """Per edge: its leverage, whether it is a bridge, the served cost's rise its removal makes
    (Sherman-Morrison; bridges' meaningless) and the served population behind it (bridges'; a
    share of the served population)."""
    lev: NDArray[np.float64]
    bridge: NDArray[np.bool_]
    rise: NDArray[np.float64]
    behind: NDArray[np.float64]
    cost: NDArray[np.float64]       # per home its cost (0 unless served from a road node)
    w: NDArray[np.float64]          # per home its weight if served (by a road or the street)


def pieces(gr: Graph, *, fault: str) -> Pieces:
    """Every piece's leverage, rise and population behind (module docstring) from one inverse;
    `fault` breaks it for the check (none | no-denominator | no-bridges)."""
    w = np.where(gr.home >= -1, gr.w, 0.0)
    hs = gr.home >= 0
    G = np.linalg.inv(gr.laplacian().toarray())
    idx = np.arange(len(gr.a))
    Da = (np.where(gr.a[None, :] >= 0, G[:, np.maximum(gr.a, 0)], 0.0)
          - np.where(gr.b[None, :] >= 0, G[:, np.maximum(gr.b, 0)], 0.0))   # G[:, a] - G[:, b]
    R = (np.where(gr.a >= 0, Da[np.maximum(gr.a, 0), idx], 0.0)
         - np.where(gr.b >= 0, Da[np.maximum(gr.b, 0), idx], 0.0))
    lev = R / gr.length
    bridge = 1.0 - lev < BRIDGE
    if fault == "no-bridges":
        bridge = np.zeros_like(bridge)
    denom = np.ones_like(lev) if fault == "no-denominator" else np.maximum(1.0 - lev, BRIDGE)
    rise = (w[hs] @ Da[gr.home[hs]] ** 2) / gr.length / denom
    behind = np.zeros(len(gr.a))
    if bridge.any():
        nxg = nx.Graph()
        nxg.add_edges_from(zip(gr.a.tolist(), gr.b.tolist(), strict=True))
        pop = np.zeros(gr.n)
        np.add.at(pop, gr.home[hs], w[hs])
        for e in np.flatnonzero(bridge):
            u, v = int(gr.a[e]), int(gr.b[e])
            nxg.remove_edge(u, v)
            side = v if u == -1 or nx.has_path(nxg, u, -1) else u
            part = [k for k in nx.node_connected_component(nxg, side) if k >= 0]
            behind[e] = pop[part].sum() / w.sum()
            nxg.add_edge(u, v)
    cost = np.where(hs, G[np.maximum(gr.home, 0), np.maximum(gr.home, 0)], 0.0)
    return Pieces(lev=lev, bridge=bridge, rise=rise, behind=behind, cost=cost, w=w)


def damage(gr: Graph, *, fault: str = "none") -> Damage:
    """The two numbers (module docstring)."""
    W = float(np.where(gr.home >= -1, gr.w, 0.0).sum())
    served = W / float(gr.w.sum())
    if gr.n == 0:
        return Damage(P=0.0, served=served, cut=0.0, detour=0.0, bridges=0.0)
    pc = pieces(gr, fault=fault)
    total = float(pc.w @ pc.cost)
    pi = gr.length / gr.length.sum()
    ok = ~pc.bridge
    detour = (float(pi[ok] @ pc.rise[ok]) / float(pi[ok].sum()) / total
              if ok.any() and total > 0 else 0.0)
    return Damage(P=total / W, served=served, cut=float(pi[pc.bridge] @ pc.behind[pc.bridge]),
                  detour=detour, bridges=float(pi[pc.bridge].sum()))


def exact(gr: Graph, e: int) -> tuple[int, float, float]:
    """Edge e removed by fresh solves: (the nodes it parts from the ground, the served population
    they hold as a share, the rise of the rest's summed cost), the check's reference."""
    w = np.where(gr.home >= -1, gr.w, 0.0)
    keep = np.arange(len(gr.a)) != e
    nxg = nx.Graph()
    nxg.add_nodes_from(range(-1, gr.n))
    nxg.add_edges_from(zip(gr.a[keep].tolist(), gr.b[keep].tolist(), strict=True))
    live = np.array(sorted(v for v in nx.node_connected_component(nxg, -1) if v >= 0),
                    dtype=np.int64)
    on = np.zeros(gr.n, dtype=bool)
    on[live] = True
    served = gr.home >= 0
    lost = served & ~on[np.maximum(gr.home, 0)]
    hs = served & ~lost
    cols = gr.home[hs]
    before = spla.splu(gr.laplacian()).solve(np.eye(gr.n)[:, cols])
    pos = np.full(gr.n, -1)
    pos[live] = np.arange(len(live))
    sub = sp.csc_matrix(gr.laplacian(drop=e)[live][:, live])
    after = spla.splu(sub).solve(np.eye(len(live))[:, pos[cols]])
    k = np.arange(len(cols))
    rise = float(w[hs] @ (after[pos[cols], k] - before[cols, k]))
    return int(gr.n - len(live)), float(w[lost].sum() / w.sum()), rise


def check(block_id: str, network: Path, *, step_m: float, carry_m: float, fault: str) -> None:
    """pieces' rise and population behind against fresh solves on 40 edges (20 bridges and 20
    others, drawn at random); a bridge must part nodes from the ground and the population behind
    it must match, any other piece must part none and match the rise to 1e-6."""
    [b] = common.build_blocks([block_id])
    gr = Graph.of(b, gpd.read_parquet(network), step_m=step_m, carry_m=carry_m)
    if gr.n > 4000:
        raise SystemExit(f"{gr.n} nodes: too many for the dense check")
    pc = pieces(gr, fault=fault)
    rng = np.random.default_rng(0)
    pick = np.concatenate([rng.permutation(np.flatnonzero(pc.bridge))[:20],
                           rng.permutation(np.flatnonzero(~pc.bridge))[:20]])
    worst = 0.0
    for e in pick:
        parted, lost, d = exact(gr, int(e))
        if pc.bridge[e]:
            err = 1.0 if parted == 0 else abs(lost - pc.behind[e])
        else:
            err = 1.0 if parted else abs(pc.rise[e] - d) / max(abs(d), 1e-12)
        worst = max(worst, err)
        if err > 1e-6:
            print(f"edge {e}: bridge {bool(pc.bridge[e])}, leverage {pc.lev[e]:.6f}, parted "
                  f"{parted}: rise {pc.rise[e]:.6e} against {d:.6e}, behind "
                  f"{pc.behind[e]:.4f} against {lost:.4f}")
    dm = damage(gr, fault=fault)
    print(f"{block_id} {network.name}: {gr.n} nodes, {len(gr.a)} pieces, "
          f"{int(pc.bridge.sum())} bridges; P_v {dm.P:.1f} m, served {dm.served:.3f}, cut "
          f"{dm.cut:.4f}, detour {dm.detour:.4f}; {len(pick)} pieces checked, worst "
          f"{worst:.2e} -> {'PASS' if worst <= 1e-6 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if len(argv) in (5, 6) and argv[0] == "check":
        check(argv[1], Path(argv[2]), step_m=float(argv[3]), carry_m=float(argv[4]),
              fault=argv[5] if len(argv) == 6 else "none")
    else:
        raise SystemExit(__doc__)
