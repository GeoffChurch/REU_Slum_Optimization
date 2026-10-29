"""Escape time under blockage (owner 2026-10-28: does it separate loop networks from dead-end ones
by a margin worth building into a greedy?). A measurement on block 5810's stored frames, in their
world: the kappa-10 one of the 5810 build-out page (atlas/roads5810/compute.carved, the frames'
own carved meshes, rebuilt from the same networks so the cache meets them): carved roads
conducting to the street at ten times open ground, the street ring the ground, rail and water cut
out. Not the grounded leak of growth.py and ears.py, where every road point is an exit and a
blockage does nothing.

Networks (NETWORKS, each a Shown: how its frames are rebuilt, at which D): Loop roads (ears.paths
v1), the partition SIMP's spectral cut (rev 4), partition plus spurs, the lazy ΔP greedy, the
kappa-10 growth and clearance_looped, each at its frames nearest D 0.05 / 0.10 / 0.15 (per budget
the shortest prefix reaching it, as the page cuts; an optimizer's answer per budget).

Blockage of one road stretch (`stretches`, `blocked`): the network's corridor skeletonized on a 1 m
raster (partition.skeleton, which also counts its dead ends) and its lines cut into stretches of
about stretch_m, every `every`-th measured (a systematic quadrature along the skeleton, its error
half the gap between its two interleaved halves); a stretch blocked makes its cross-section
impassable: the corridor's chord square to the line at the stretch's middle, every pair of cells
whose centroids it separates (a passage of the flow model's graph: gap_clear.Reach.W) at zero
conductance; the ground around still conducts. Its exact P is a re-solve on the one pattern (the
state's reach with those pairs at 0, the operator's values again, cuDSS's refactorization). Checked
on `checks` stretches per network-D against the rank-k Woodbury update with the base factorization
(dP = d' (G^-1 - B' A^-1 B)^-1 d, d = B' u, k solves). Per network-D: E[dP]/P0 uniform per metre of
road (the stretches weighted by their lengths), L_P^block = 1 - E[P_blocked]/P0, the worst stretch,
and the share of road length at leverage > `lever`, a stretch's leverage 1 - sum_q g_q du_q^2 / dP
(the share of the drop the cut pairs' own power does not explain: g R of the rank-one case, near 1
where nothing else carries the current, a dead end's).

A fire at a home (`fire`): every cell whose centroid lies within fire_r of a home's centroid
impassable (its passages at zero, off the unknowns), and the homes whose centroids lie within it
carry no load (they are the fire; the others escape). P_roads(x) on the network's state, P_none(x)
on the same mesh's plain state under the same fire; L_P^fire = 1 - E_x[P_roads] / E_x[P_none] over a
deterministic quadrature: the homes in Morton order of their centroids, every k-th (fire_n of them),
its error stated as half the gap between its two interleaved halves. Cells the fire seals from the
street (a courtyard whose one way out burns: buildings conduct nothing in this metric) are found
locally (`sealed`, within seal_m) and taken off the unknowns with their homes' loads (counted per
fire, with roads and without); a pocket it misses makes the solve fail, and that fire is dropped
from both means (counted).

    B=research/roadless/blockage.py                    # from the repo root, PYTHONPATH=.
    uv run python $B run|block|fire|probe <network>    # both, the blockage, the fires, a few
    uv run python $B table

Rows in OUT/<network>_<DDD>_stretches.parquet and _fires.parquet, summaries in
OUT/summary_<network>_<parts>.json.
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from numpy.typing import NDArray
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "atlas" / "roads5810"))
import networks as nw  # noqa: E402  (the 5810 page's sources; compute, its carved models)
import partition as pt  # noqa: E402  (skeleton, its dead ends)

cp, gcl, common, metrics = nw.cp, nw.gcl, nw.common, nw.metrics
ROOT = HERE.parents[1]
OUT = ROOT / ".superpowers/sdd/2026-10-28-blockage"
FLOW = ROOT / ".superpowers/sdd/2026-10-22-flow-roads"     # the lazy ΔP greedy's build
F = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class Params:
    """The study's settings, set once at the edge (main)."""
    stretch_m: float          # a road stretch's length
    chord_m: float            # the cross-section's half-length before it is clipped to the corridor
    lever: float              # leverage above which a stretch is dead-end-like
    checks: int               # stretches per network-D checked against Woodbury
    fire_r: float             # m: a fire's radius
    fire_n: int               # the fire quadrature's homes
    seal_m: float             # m: the window searched for the cells a fire seals (`sealed`)
    shell_m: float            # m: its outer shell, assumed to reach on
    targets: tuple[float, ...]
    every: int                # the stretches measured: every `every`-th along the skeleton
                              # (a systematic quadrature; its error from the two interleaved halves)
    limit: int                # stretches per network-D at most (0: every one; a probe's few)


# ------------------------------------------------------------------ the networks

class Frames(NamedTuple):
    """One carved model's frames: the Network, its frames' pass and mesh cache, and per target D
    (the study's) its stop in that model."""
    net: Any
    frames: Any
    meshes: Path
    at: dict[float, int]      # target -> the frame's stop (a prefix length of `net`)


@dataclass(frozen=True)
class Shown:
    """A network of the study: its name, and its frames rebuilt (one or more carved models)."""
    name: str
    build: Callable[[Any, Params], list[Frames]]


def grown(plan: str) -> Callable[[Any, Params], list[Frames]]:
    def build(b: Any, p: Params) -> list[Frames]:
        src = nw.GrownSource(plan, budget=0.15, site=nw.PAGE)
        net = nw.Grown(b, src.dir, targets=nw.SLIDER)
        full = max(nw.SLIDER) not in net.stop_of
        at = {t: net.prefix_len(t) for t in p.targets}
        return [Frames(net=net, frames=cp.Pass(targets=nw.SLIDER, full=full),
                       meshes=nw.PAGE.networks / ".meshes", at=at)]
    return build


def answers(argv: tuple[str, ...]) -> Callable[[Any, Params], list[Frames]]:
    def build(b: Any, p: Params) -> list[Frames]:
        src = nw.family_of(list(argv)).source()
        out = []
        for budget in src.design.budgets():
            a = src.design.answer(b, budget)
            out.append(Frames(net=nw.One(b, a), frames=cp.Pass(targets=(), full=True),
                              meshes=src.design.meshes(), at={budget: 1}))
        return out
    return build


def roads(path: Path, frames: Any, meshes: Path) -> Callable[[Any, Params], list[Frames]]:
    def build(b: Any, p: Params) -> list[Frames]:
        net = cp.Roads(b, gpd.read_parquet(path))
        return [Frames(net=net, frames=frames, meshes=meshes,
                       at={t: net.prefix_len(t) for t in p.targets})]
    return build


R3 = "pt.K5.R10.band.t0.2,0.1,0.05,0.03,0.02.b1,2,4,8,16.q1,2,3,3,3.e0.2,0.1,0.05,0.02,0.i40"
NETWORKS = {s.name: s for s in (
    Shown("loop", grown("ears.paths.b20.f2000.e30.gl0.25Croad.n2s0.1.M8.K4.D0.15.v1")),
    Shown("cut", answers(("partition", "cut", f"{R3}.D{{D}}.s0.1.spec{{n}}s0.p1e0.0001v4"))),
    Shown("spurs", answers(("spurs", "ps.lazy5.D{D}.spec{n}s0.x{x}.v4"))),
    Shown("lazy", roads(FLOW / "dp_greedy_lineup_lazy5_D" / "built.parquet",
                        cp.Pass(targets=tuple(round(0.01 * i, 2) for i in range(1, 16)),
                                full=False), FLOW / ".meshes")),
    Shown("k10", grown("gr.k10.att.greedy.N48.sep14.n2s0.1.B21m8k2.K0.D0.15.v2")),
    Shown("clearance_looped", roads(
        cp.LINEUP / "scratch/networks" / nw.PAGE.bid / "clearance_looped.ordered.parquet",
        cp.PASSES["coarse"], nw.PAGE.data / ".meshes")),
)}


# ------------------------------------------------------------------ the flow model's graph

class Graph(NamedTuple):
    """A model's pattern read once: per off-diagonal pair its cells, the cells' centroids and a
    tree over them."""
    pi: NDArray[np.int64]
    pj: NDArray[np.int64]
    xy: F
    tree: cKDTree


def graph_of(m: Any) -> Graph:
    g = m.sys.graph
    p = g.pattern
    rows = np.repeat(np.arange(p.n), np.diff(p.indptr))
    q = g.epair
    off = q >= 0
    npair = int(q.max()) + 1
    pi = np.full(npair, -1, dtype=np.int64)
    pj = np.full(npair, -1, dtype=np.int64)
    lo = off & (rows < p.indices)
    pi[q[lo]] = rows[lo]
    pj[q[lo]] = p.indices[lo]
    xy = np.asarray(p.xy, dtype=np.float64)
    return Graph(pi=pi, pj=pj, xy=xy, tree=cKDTree(xy))


class Blocked(NamedTuple):
    """A blocked state's P (the loads of the homes it seals excluded) and the homes it seals."""
    P: float
    sealed_homes: int
    sealed_load: float        # their share of the demand
    sealed_cells: int
    fallback: bool            # the local search missed a pocket: the global one found it


class Solver:
    """P of a state with some pairs closed, some cells off the unknowns and some homes' loads
    removed: the reach's W and unknown changed, the operator's values again, the plan refactored
    (one factorization at a time: each solve's own). Cells the change seals from the street are
    taken off the unknowns with their loads (`sealed` in a window around the change; where that
    misses a pocket the system is singular, its P not finite, and cuDSS's plan is left unusable:
    the plan is made again and the pocket found over the whole graph, `sealed_all`)."""

    def __init__(self, m: Any, st: Any, gr: Graph, *, replan: Callable[[], None]) -> None:
        self.m, self.st, self.gr = m, st, gr
        self.r = st.reach
        self.replan = replan

    def _P(self, closed: NDArray[np.int64], off: NDArray[np.int64], inj: F) -> float:
        r = self.r
        W = r.W.copy()
        W[closed] = 0.0
        unknown = r.unknown.copy()
        unknown[off] = False
        data = self.st.sys.graph.values(r._replace(W=W, unknown=unknown))
        try:
            f = self.st.sys.planned.factor(data, unknown)
            rhs = np.asarray(self.st.sys.Pc @ inj) * unknown
            P = float(rhs @ f.solve(rhs))
        except RuntimeError:          # the refinement refused: a singular system
            return float("nan")
        return P if np.isfinite(P) else float("nan")

    def P(self, *, closed: NDArray[np.int64], off: NDArray[np.int64], inj: F, at: F,
          p: Params) -> Blocked:
        sl = sealed(self.gr, self.r, at, closed=closed, off=off, reach_m=p.seal_m,
                    shell_m=p.shell_m)
        P = self._P(closed, np.concatenate([off, sl]), inj)
        fallback = not np.isfinite(P)
        if fallback:
            self.replan()
            sl = sealed_all(self.gr, self.r, closed=closed, off=off)
            P = self._P(closed, np.concatenate([off, sl]), inj)
            if not np.isfinite(P):
                self.replan()
                raise RuntimeError("no finite P with every sealed pocket off the unknowns")
        w = np.asarray(self.st.sys.Pc[sl].sum(axis=0)).ravel()
        homes = w > 0
        return Blocked(P=P, sealed_homes=int(homes.sum()),
                       sealed_load=float(inj[homes].sum() / inj.sum()), sealed_cells=len(sl),
                       fallback=fallback)


# ------------------------------------------------------------------ stretches

class Stretch(NamedTuple):
    mid: tuple[float, float]
    length: float
    pairs: NDArray[np.int64]


def stretches(corridor: Any, lines: Any, gr: Graph, p: Params) -> list[Stretch]:
    """The skeleton `lines` cut into stretches of about stretch_m, each with the pairs its
    cross-section (the corridor's chord square to the line at its middle) separates."""
    out = []
    shapely.prepare(corridor)
    for line in shapely.get_parts(lines):
        L = float(line.length)
        if L <= 0:
            continue
        n = max(1, int(round(L / p.stretch_m)))
        for k in range(n):
            s = (k + 0.5) * L / n
            a = np.asarray(shapely.get_coordinates(line.interpolate(max(s - 0.5, 0.0))))[0]
            c = np.asarray(shapely.get_coordinates(line.interpolate(min(s + 0.5, L))))[0]
            mid = np.asarray(shapely.get_coordinates(line.interpolate(s)))[0]
            t = c - a
            if np.hypot(*t) == 0:
                continue
            t /= np.hypot(*t)
            nrm = np.array([-t[1], t[0]])
            chord = shapely.LineString([mid - p.chord_m * nrm, mid + p.chord_m * nrm])
            piece = shapely.intersection(chord, corridor)
            parts = [q for q in shapely.get_parts(piece)
                     if q.geom_type == "LineString" and q.length > 0]
            if not parts:
                continue
            seg = min(parts, key=lambda q: q.distance(shapely.Point(mid)))
            (x0, y0), (x1, y1) = np.asarray(seg.coords)[[0, -1]]
            cells = np.asarray(gr.tree.query_ball_point(mid, seg.length / 2 + 3.0), dtype=np.int64)
            out.append(Stretch(mid=(float(mid[0]), float(mid[1])), length=L / n,
                               pairs=crossing(gr, cells, (x0, y0), (x1, y1))))
    return out


def crossing(gr: Graph, cells: NDArray[np.int64], a: tuple[float, float],
             b: tuple[float, float]) -> NDArray[np.int64]:
    """The pairs among `cells` (both ends in it) whose centroids' segment crosses a--b."""
    if len(cells) == 0:
        return np.zeros(0, dtype=np.int64)
    inside = np.zeros(len(gr.xy), dtype=bool)
    inside[cells] = True
    q = np.flatnonzero(inside[np.maximum(gr.pi, 0)] & inside[np.maximum(gr.pj, 0)]
                       & (gr.pi >= 0))
    P, Q = gr.xy[gr.pi[q]], gr.xy[gr.pj[q]]
    A, B = np.asarray(a), np.asarray(b)

    def side(o: F, p1: F, p2: F) -> F:
        return (p1[..., 0] - o[..., 0]) * (p2[..., 1] - o[..., 1]) - \
            (p1[..., 1] - o[..., 1]) * (p2[..., 0] - o[..., 0])
    d1, d2 = side(A[None], B[None], P), side(A[None], B[None], Q)
    d3, d4 = side(P, Q, A[None]), side(P, Q, B[None])
    return q[(d1 * d2 < 0) & (d3 * d4 < 0)]


def woodbury(m: Any, st: Any, u: F, pairs: NDArray[np.int64], gr: Graph) -> float:
    """dP of closing `pairs` by the rank-k Woodbury update on the base factorization."""
    r = st.reach
    g = r.W[pairs]
    keep = g > 0
    pairs, g = pairs[keep], g[keep]
    if not len(pairs):
        return 0.0
    n = len(u)
    cols = []
    for q in pairs:
        e = np.zeros(n)
        if r.unknown[gr.pi[q]]:
            e[gr.pi[q]] = 1.0
        if r.unknown[gr.pj[q]]:
            e[gr.pj[q]] = -1.0
        cols.append(e)
    Bm = np.stack(cols, 1)
    AinvB = np.stack([st.factor.solve(Bm[:, k]) for k in range(Bm.shape[1])], 1)
    S = np.diag(1.0 / g) - Bm.T @ AinvB
    d = Bm.T @ u
    return float(d @ np.linalg.solve(S, d))


# ------------------------------------------------------------------ the run

def morton(xy: F) -> NDArray[np.int64]:
    lo, hi = xy.min(0), xy.max(0)
    q = ((xy - lo) / np.maximum(hi - lo, 1e-9) * 65535).astype(np.int64)
    key = np.zeros(len(xy), dtype=np.int64)
    for bit in range(16):
        key |= ((q[:, 0] >> bit) & 1) << (2 * bit)
        key |= ((q[:, 1] >> bit) & 1) << (2 * bit + 1)
    return key


def one(name: str, b: Any, fr: Frames, k: int, D_target: float, p: Params, *, block: bool,
        fire: bool, cv: Any, gr: Graph, factorization: Any,
        log: Callable[[str], None]) -> dict[str, Any]:
    """One network-D: its stop k (1-based in the carving), its blockage and its fires."""
    m = cv.m

    def replan() -> None:
        """A fresh plan of the pattern (a singular system leaves cuDSS's unusable)."""
        m.sys.planned = factorization.plan(m.sys.graph.pattern)

    mm = cv.stops[k - 1]
    st = m.road_state(gcl.Prefix(net=0, stop=k))
    u = m.solve(st, m.pop)
    rhs = np.asarray(st.sys.Pc @ m.pop) * st.reach.unknown
    P = float(rhs @ u)
    P0 = m.base.P
    corridor = m.carving.corridors[0][k - 1]
    adapter = SimpleNamespace(streets=gpd.GeoDataFrame(geometry=[m.geo.streets]))
    lines, dead, _ = pt.skeleton(adapter, corridor)
    W = common.POPULATIONS["area"].weights(np.asarray(b.buildings.outlines))
    D = float((fr.net.c(mm) * W).sum() / W.sum())
    n_roads, road_m = fr.net.roads(mm)
    out = dict(network=name, target=D_target, D=D, stop=int(mm), n_roads=int(n_roads),
               road_km=road_m / 1000, skeleton_km=float(shapely.length(lines)) / 1000,
               dead_ends=int(dead), P0=P0, P=P, L_P=1 - P / P0)
    log(f"{name} D {D:.4f} (stop {mm}): L_P {1 - P / P0:.4f}, "
        f"{out['skeleton_km']:.2f} km skeleton, {dead} dead ends")
    tag = f"{int(round(D_target * 1000)):03d}"
    OUT.mkdir(parents=True, exist_ok=True)
    if block:
        out |= blocked(m, st, u, P, corridor, lines, gr, p, replan=replan, log=log, name=name,
                       tag=tag)
    if fire:
        out |= fires(m, st, gr, p, replan=replan, log=log, name=name, tag=tag)
    return out


def blocked(m: Any, st: Any, u: F, P: float, corridor: Any, lines: Any, gr: Graph, p: Params, *,
            replan: Callable[[], None], log: Callable[[str], None], name: str,
            tag: str) -> dict[str, Any]:
    """The single-stretch blockage measures (module docstring)."""
    P0 = m.base.P
    t0 = time.time()
    every = stretches(corridor, lines, gr, p)
    ss = every[p.every // 2::p.every]
    if p.limit:
        ss = ss[:: max(1, len(ss) // p.limit)][: p.limit]
    # the Woodbury check, on the base factorization (before any refactorization)
    pick = np.linspace(0, len(ss) - 1, min(p.checks, len(ss))).astype(int)
    wb = {int(i): woodbury(m, st, u, ss[i].pairs, gr) for i in pick}
    solver = Solver(m, st, gr, replan=replan)
    none_i = np.zeros(0, dtype=np.int64)
    rows = []
    for i, s in enumerate(ss):
        bl = solver.P(closed=s.pairs, off=none_i, inj=m.pop, at=np.asarray(s.mid), p=p)
        Pb = bl.P
        g = st.reach.W[s.pairs]
        du = u[gr.pi[s.pairs]] - u[gr.pj[s.pairs]]          # u is 0 off the unknowns
        own = float((g * du ** 2).sum())
        dP = Pb - P
        rows.append(dict(x=s.mid[0], y=s.mid[1], length=s.length, k=len(s.pairs), dP=dP,
                         own=own, lever=1.0 - own / dP if dP > 0 else np.nan,
                         woodbury=wb.get(i, np.nan), sealed_homes=bl.sealed_homes,
                         sealed_load=bl.sealed_load, sealed_cells=bl.sealed_cells,
                         fallback=bl.fallback))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f"{name}_{tag}_stretches.parquet")
    chk = df.dropna(subset=["woodbury"])
    chk = chk[chk.sealed_cells == 0]          # Woodbury's is the unsealed system's own
    rel = float((np.abs(chk.dP - chk.woodbury) / np.maximum(np.abs(chk.dP), 1e-300)).max()) \
        if len(chk) else np.nan
    if df.dP.isna().any():
        raise AssertionError(f"{name} {tag}: a stretch with no P")
    w = df.length.to_numpy()
    EdP = float((w * df.dP).sum() / w.sum())
    seals = (df.sealed_homes > 0).to_numpy()
    halves = [float((w[h::2] * df.dP.to_numpy()[h::2]).sum() / w[h::2].sum()) for h in (0, 1)]
    worst = df.loc[df.dP.idxmax()]
    out = dict(stretches=len(ss), stretches_all=len(every), EdP_P0=EdP / P0,
               L_P_block=1 - (P + EdP) / P0, L_P_block_err=abs(halves[0] - halves[1]) / 2 / P0,
               worst_dP_P0=float(worst.dP) / P0, worst_x=float(worst.x), worst_y=float(worst.y),
               lever_share=float(w[(df.lever > p.lever).to_numpy()].sum() / w.sum()),
               sealing_share=float(w[seals].sum() / w.sum()),
               sealed_homes=float((w * df.sealed_homes).sum() / w.sum()),
               sealed_load=float((w * df.sealed_load).sum() / w.sum()),
               fallbacks=int(df.fallback.sum()), woodbury_rel=rel, t_block=time.time() - t0)
    log(f"{name}: L_P^block {out['L_P_block']:.4f} (E[dP]/P0 {out['EdP_P0']:.2e}, {len(ss)} of "
        f"{len(every)} stretches), worst {out['worst_dP_P0']:.2e}, lever>{p.lever} "
        f"{out['lever_share']:.3f}; sealing {out['sealing_share']:.3f} of the length, "
        f"{out['sealed_homes']:.2f} homes a blockage ({out['fallbacks']} global searches); "
        f"Woodbury rel {rel:.1e} ({out['t_block']:.0f}s)")
    return out


def sealed(gr: Graph, r: Any, at: F, *, closed: NDArray[np.int64], off: NDArray[np.int64],
           reach_m: float, shell_m: float) -> NDArray[np.int64]:
    """The unknown cells a fire seals from the street: within reach_m of its centre, among the
    unknowns (less `off`) joined by conducting pairs (less `closed`), the components with no cell
    on the street (G > 0), none joined to a cell off the unknowns that conducts (a ground cell)
    and none in the window's outer shell_m (assumed to reach on: a pocket larger than the window
    is not found, and its solve is refused by `guarded`)."""
    win = np.asarray(gr.tree.query_ball_point(at, reach_m), dtype=np.int64)
    unk = r.unknown.copy()
    unk[off] = False
    W = r.W.copy()
    W[closed] = 0.0
    loc = np.full(len(gr.xy), -1, dtype=np.int64)
    loc[win] = np.arange(len(win))
    a, b = np.maximum(gr.pi, 0), np.maximum(gr.pj, 0)
    live = (gr.pi >= 0) & (W > 0)
    both = live & (loc[a] >= 0) & (loc[b] >= 0) & unk[a] & unk[b]
    n = len(win)
    A = sp.coo_matrix((np.ones(int(both.sum())), (loc[a[both]], loc[b[both]])), shape=(n, n))
    _, comp = connected_components(A, directed=False)
    grounded = (r.G[win] > 0) | ~unk[win]
    # a cell joined by a conducting pair to a ground cell (off the unknowns: the fire's pairs are
    # closed, so not its cells) or out of the window
    edge = live & (unk[a] != unk[b])
    for x, y in ((a, b), (b, a)):
        hit = edge & (loc[x] >= 0) & unk[x]
        grounded[loc[x[hit]]] = True
        out = live & (loc[x] >= 0) & (loc[y] < 0)
        grounded[loc[x[out]]] = True
    far = np.hypot(*(gr.xy[win] - at).T) > reach_m - shell_m
    ok = np.zeros(comp.max() + 1, dtype=bool)
    ok[np.unique(comp[grounded | far])] = True
    bad = unk[win] & ~ok[comp]
    return win[bad]


def sealed_all(gr: Graph, r: Any, *, closed: NDArray[np.int64],
               off: NDArray[np.int64]) -> NDArray[np.int64]:
    """`sealed` over the whole graph: the unknown cells (less `off`) in conducting components
    (pairs less `closed`) with no street face and no conducting pair to a ground cell."""
    unk = r.unknown.copy()
    unk[off] = False
    W = r.W.copy()
    W[closed] = 0.0
    a, b = np.maximum(gr.pi, 0), np.maximum(gr.pj, 0)
    live = (gr.pi >= 0) & (W > 0)
    both = live & unk[a] & unk[b]
    n = len(gr.xy)
    A = sp.coo_matrix((np.ones(int(both.sum())), (a[both], b[both])), shape=(n, n))
    _, comp = connected_components(A, directed=False)
    grounded = (r.G > 0) & unk
    edge = live & (unk[a] != unk[b])
    grounded[a[edge & unk[a]]] = True
    grounded[b[edge & unk[b]]] = True
    ok = np.zeros(comp.max() + 1, dtype=bool)
    ok[np.unique(comp[grounded])] = True
    return np.flatnonzero(unk & ~ok[comp])


def fires(m: Any, st: Any, gr: Graph, p: Params, *, replan: Callable[[], None],
          log: Callable[[str], None], name: str, tag: str) -> dict[str, Any]:
    """L_P^fire over the quadrature (module docstring)."""
    t0 = time.time()
    polys = m.geo.polys
    c = shapely.centroid(polys)
    hxy = np.column_stack([shapely.get_x(c), shapely.get_y(c)])
    homes = np.flatnonzero((m.pop > 0) & ~m.nodem)
    order = homes[np.argsort(morton(hxy[homes]), kind="stable")]
    step = max(1, len(order) // p.fire_n)
    xs = order[step // 2::step][: p.fire_n]
    st0 = m.state(np.zeros(m.n))
    s_road, s_none = Solver(m, st, gr, replan=replan), Solver(m, st0, gr, replan=replan)
    htree = cKDTree(hxy)
    Pr: list[float] = []
    Pn: list[float] = []
    Sr: list[int] = []
    Sn: list[int] = []
    fb = 0
    for x in xs:
        cells = np.asarray(gr.tree.query_ball_point(hxy[x], p.fire_r), dtype=np.int64)
        inside = np.zeros(len(gr.xy), dtype=bool)
        inside[cells] = True
        closed = np.flatnonzero((gr.pi >= 0) & (inside[np.maximum(gr.pi, 0)]
                                                | inside[np.maximum(gr.pj, 0)]))
        gone = np.asarray(htree.query_ball_point(hxy[x], p.fire_r), dtype=np.int64)
        inj = m.pop.copy()
        inj[gone] = 0.0
        for s, P_, seal in ((s_road, Pr, Sr), (s_none, Pn, Sn)):
            bl = s.P(closed=closed, off=cells, inj=inj, at=hxy[x], p=p)
            P_.append(bl.P)
            seal.append(bl.sealed_homes)
            fb += bl.fallback
    Pr_a, Pn_a = np.asarray(Pr), np.asarray(Pn)
    pd.DataFrame(dict(home=xs, x=hxy[xs, 0], y=hxy[xs, 1], P_roads=Pr_a, P_none=Pn_a,
                      sealed_roads=Sr, sealed_none=Sn)).to_parquet(
        OUT / f"{name}_{tag}_fires.parquet")
    ok = np.isfinite(Pr_a) & np.isfinite(Pn_a)
    lp = 1 - Pr_a[ok].mean() / Pn_a[ok].mean()
    idx = np.flatnonzero(ok)
    half = [1 - Pr_a[idx[h::2]].mean() / Pn_a[idx[h::2]].mean() for h in (0, 1)]
    out = dict(L_P_fire=lp, L_P_fire_err=abs(half[0] - half[1]) / 2, fire_n=int(ok.sum()),
               fire_fallbacks=fb, sealed_homes_roads=float(np.mean(Sr)),
               sealed_homes_none=float(np.mean(Sn)), t_fire=time.time() - t0)
    log(f"{name}: L_P^fire {lp:.4f} +- {out['L_P_fire_err']:.4f} over {int(ok.sum())} homes "
        f"({int((~ok).sum())} refused); homes sealed per fire {np.mean(Sr):.2f} with roads, "
        f"{np.mean(Sn):.2f} without ({out['t_fire']:.0f}s)")
    return out


def run(name: str, p: Params, *, block: bool, fire: bool) -> None:
    sh = NETWORKS[name]
    metric = metrics.metric_of(metrics.RUNTIME, device="gpu")
    [b] = common.build_blocks([nw.PAGE.bid])
    res = []
    for fr in sh.build(b, p):
        cv = cp.carved(b, fr.net, frames=fr.frames, metric=metric, meshes=fr.meshes)
        gr = graph_of(cv.m)
        for t, mm in fr.at.items():
            if mm not in cv.stops:
                raise AssertionError(f"{name}: stop {mm} at {t} is not a frame's")
            res.append(one(name, b, fr, cv.stops.index(mm) + 1, t, p, block=block, fire=fire,
                           cv=cv, gr=gr, factorization=metric.factorization,
                           log=lambda s: print(s, flush=True)))
        del cv, gr
    OUT.mkdir(parents=True, exist_ok=True)
    part = "_".join(k for k, on in (("block", block), ("fire", fire)) if on)
    (OUT / f"summary_{name}_{part}.json").write_text(json.dumps(res, indent=1) + "\n")


def table() -> None:
    """One row per network-D from the summaries (a network's parts merged by its target)."""
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    for f in sorted(OUT.glob("summary_*.json")):
        for r in json.loads(f.read_text()):
            key = (r["network"], r["target"])
            rows[key] = rows.get(key, {}) | {k: v for k, v in r.items()
                                             if not (isinstance(v, float) and np.isnan(v))}
    df = pd.DataFrame(list(rows.values()))
    df["order"] = df.network.map({n: i for i, n in enumerate(NETWORKS)})
    df = df.sort_values(["order", "target"]).drop(columns="order")
    df.to_csv(OUT / "table.csv", index=False)
    cols = ["network", "D", "L_P", "L_P_block", "L_P_block_err", "L_P_fire", "L_P_fire_err",
            "dead_ends", "lever_share", "sealing_share", "sealed_homes", "road_km", "skeleton_km",
            "worst_dP_P0", "sealed_homes_roads", "sealed_homes_none", "woodbury_rel"]
    print(df[[c for c in cols if c in df]].to_string(index=False,
                                                    float_format=lambda v: f"{v:.4g}"))


PARAMS = Params(stretch_m=10.0, chord_m=25.0, lever=0.9, checks=3, fire_r=15.0, fire_n=100,
                seal_m=80.0, shell_m=10.0,
                targets=(0.05, 0.10, 0.15), every=2, limit=0)
PROBE = Params(stretch_m=10.0, chord_m=25.0, lever=0.9, checks=2, fire_r=15.0, fire_n=4,
               seal_m=80.0, shell_m=10.0,
               targets=(0.05,), every=1, limit=12)          # the cost probe: a few of each


if __name__ == "__main__":
    match sys.argv[1:]:
        case ["run", name]:
            OUT.mkdir(parents=True, exist_ok=True)
            try:                      # two processes share the networks: the first claims one
                (OUT / f"claim_{name}").open("x").close()
            except FileExistsError:
                raise SystemExit(f"{name}: claimed by another run") from None
            run(name, PARAMS, block=True, fire=True)
        case ["block", name]:
            run(name, PARAMS, block=True, fire=False)
        case ["fire", name]:
            run(name, PARAMS, block=False, fire=True)
        case ["probe", name]:
            run(name, PROBE, block=True, fire=True)
        case ["table"]:
            table()
        case _:
            raise SystemExit(__doc__)
