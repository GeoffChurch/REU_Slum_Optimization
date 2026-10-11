"""The private crossing: two simpler ways out for the walled-in homes, against the door link.

The owner (2026-10-11), of the door link (insulator.py, "The door link"): "Is there a way to
simplify your suggested method? The if-else feels unnatural"; then "Yes, test that. But, it
would be nice to also test a version that can account for traffic/contention, if that would fit
into the same framework, since it would reward multiple points of egress".

Both models leave the open network O exactly as today (lifted.System on pockets_of's open field,
no extra edge), so no open home's own escape can leak through a wall. The walled-in homes
(insulator.walled: tiers 1, 3, 4, 5 under the placement fix) get out by a private crossing:
  toll    (`Toll`) a home's demand in each pocket it injects into enters O at the street end b
          of its thinnest crossing (insulator.HomeDoors' site) and pays a toll there: u is the
          share-weighted mean of u_O(b) + d T / g1 (per metre: T metres of footprint crossed)
          or u_O(b) + d / g (per crossing: the door's physics with no edge in the operator), d
          the demand crossing. One solve per block; the report evaluates any g1 or g.
  medium  (`Medium`) a private copy M of the block's geometry (isotropic, 4-connected, on the
          host) in which footprint conducts eps relative to open ground, C0 per square. The
          walkers pick their way through M's walls to its exits (fully open street-connected
          cells, held at 0), then continue in O from where they come out: u = toll (M) + beyond
          (O's potential at the exits, averaged over where M takes the walker). Homes share M's
          walls (contention) and several thin walls conduct in parallel (multiple egress).

    S=research/roadless/crossing.py      # from the repo root, PYTHONPATH=., single-threaded
    REBLOCK_BLOCK_BANK=<bank> uv run python $S small <procs> toll <ids,>
    REBLOCK_BLOCK_BANK=<bank> uv run python $S small <procs> medium <window_m> <eps,> <ids,>
    CUDA_PATH=/usr uv run python $S large toll <ids,>
    CUDA_PATH=/usr uv run python $S large medium <window_m> <eps,> <eps for the clearings,> \\
        <ids,>
    uv run python $S report      # the small blocks at matched medians: summary_crossing_*.csv

Rows in insulator_rows/ (parquet, ignored): crossing_toll_h0.5_ell3_K8/ (small blocks
<block>.parquet) and crossing_medium_w<window_m>_h0.5_ell3_K8/ (<block>__eps<eps>.parquet), with
<block>__sites.parquet; large blocks <block>__large_homes.parquet and <block>__large.parquet.

Stopped on the small blocks (2026-10-11): the owner then dropped the criterion these models were
built for ("I am fine with having fully enclosed buildings be much worse off than exposed
neighbors") in favour of one wall resistance for everyone (walls.py). Run: the toll and the
medium (window 25 m) on the 31 small blocks, the window check (50 m) on 12 of them, the toll on
9 of the 12 large blocks with walled-in homes; NOTES, "The insulator world".
"""
from __future__ import annotations

import gc
import math
import os
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import factorized

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import insulator  # noqa: E402
import lifted  # noqa: E402

# ------------------------------------------------------------------ the run's parameters
# Everything that can change a number is here or on the command line, and in the rows' paths
# (with insulator's H, ELL_M, K, RING_M, NEIGHBOURS, RTOL, RTOL_LARGE, STORED, CLEARING_POWER).
G1_GRID = (0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 1.0)     # toll per metre
G_GRID = (0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3)  # per crossing (door's)
EPS_GRID = (1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 3e-2, 1e-1)        # the medium's walls
C0 = 1.0 / 1.618         # M's conductance per square of open ground: O's resistance per metre of
#                          1 m-wide straight corridor (summary_door_link_corridor.csv), as a sheet
NEAR_B_M = 5.0           # near_b_share: a home's exit current landing this close to its main b
TARGET_R = 1.797907      # the door's median r at g 0.01 (count, small blocks, walled-in homes)
DOOR = "link_home_heading_open_k0"   # the door link the models are compared with, at DOOR_G
DOOR_G = 0.01
FIELDS_PER_SOLVE = 4     # O's right-hand sides per CG call, for device memory (the door's large
#                          run solved 4 at once on 20543). Not bit-identical: a column's CG runs
#                          on past rtol while the others in its call finish; within rtol.
ROWS = insulator.ROWS
SUFFIX = f"h{insulator.H:g}_ell{insulator.ELL_M:g}_K{insulator.K}"


# ------------------------------------------------------------------ the block

@dataclass(frozen=True, eq=False)
class Block:
    """One block's baseline as insulator.py's door-link runs build it: the placement fix, the
    walled-in homes `wl`, their pockets (open ends), HomeDoors' sites and, per site, `share`:
    the part of its home's injection that lies in its pocket."""
    bid: str
    p: lifted.Params
    grid: lifted.Grid
    polys: np.ndarray
    pl: insulator.Placement
    wl: np.ndarray
    pk: insulator.Pockets
    sites: insulator.DoorSites
    share: np.ndarray
    W: dict
    t_setup: float

    @classmethod
    def of(cls, bid: str, p: lifted.Params) -> Block:
        t0 = time.perf_counter()
        [b] = common.build_blocks([bid])
        grid = insulator.mesh(insulator.H).build(b, p, None)
        polys = np.asarray(b.buildings.outlines)
        pl = insulator.Placement.of(grid, polys, p, open_side=True)
        wl = insulator.walled(pl)
        pk = insulator.pockets_of(grid, pl, p, ends=insulator.OpenEnds())
        sites = insulator.HomeDoors().sites(pk, pl)
        Sc = pl.S.tocsc()
        share = np.zeros(len(sites.a))
        for k, (j, q) in enumerate(zip(sites.home, sites.pocket, strict=True)):
            sl = slice(Sc.indptr[j], Sc.indptr[j + 1])
            share[k] = Sc.data[sl][pk.label[Sc.indices[sl]] == q].sum()
        tot = np.bincount(sites.home, weights=share, minlength=len(polys))[wl]
        assert np.abs(tot - 1.0).max(initial=0.0) <= 1e-9, "a walled-in home's shares"
        return cls(bid=bid, p=p, grid=grid, polys=polys, pl=pl, wl=wl, pk=pk, sites=sites,
                   share=share,
                   W={name: pop.weights(polys) for name, pop in common.POPULATIONS.items()},
                   t_setup=time.perf_counter() - t0)

    @property
    def shape(self) -> tuple[int, int]:
        return self.grid.inside.shape

    def homes(self, name: str) -> dict:
        """The columns every per-home row carries, for population `name`."""
        cen = insulator._centroids(self.polys)
        pc = insulator._pocket_cols(self.pl, self.pk, self.W)
        return dict(block=self.bid, pop=name, home=np.arange(len(self.polys)),
                    tier=self.pl.tier, barrier_m=self.pl.barrier_m, shadowed=self.pl.shadowed,
                    w=self.W[name], cx=cen[:, 0], cy=cen[:, 1], pocket=pc["pocket"],
                    pocket_homes=pc["pocket_homes"], pocket_w=pc[f"pocket_w_{name}"])

    def u_open(self, uc: np.ndarray) -> np.ndarray:
        """Per home from the tier-0 demand's cell potentials: tier 0's u (nan elsewhere)."""
        return np.where(self.pl.tier == 0, self.pl.home_u(uc), np.nan)

    def at(self, cells: np.ndarray, vals: np.ndarray) -> np.ndarray:
        """An injection field: `vals` on the flat `cells`, in the grid's shape."""
        return np.bincount(cells, weights=vals, minlength=self.grid.inside.size).reshape(
            self.shape)


class OSolve(NamedTuple):
    """O's solution for several injection fields: per-cell potentials (host, grid cells flat x
    fields; insulator._cell_u's angular-share mean, 0 on ground and closed cells), P = B.U per
    field, unknowns, the worst field's relative residual, the fields, seconds (system build
    and solves)."""
    ucell: np.ndarray
    P: np.ndarray
    unknowns: int
    resid: float
    fields: int
    t: float


def o_solve(bk: Block, open_: np.ndarray, fields: list[np.ndarray], *, rtol: float) -> OSolve:
    """The open network O in the world `open_` (lifted.System, no extra edge) for `fields`."""
    t0 = time.perf_counter()
    p = bk.p
    xp = p.solver.xp
    sy = lifted.System(bk.grid, open_, p)
    ucell = np.zeros((bk.grid.inside.size, len(fields)))
    P = np.zeros(len(fields))
    res = 0.0
    for lo in range(0, len(fields), FIELDS_PER_SOLVE):
        part = fields[lo:lo + FIELDS_PER_SOLVE]
        B = xp.stack([sy.load(f, p) for f in part], axis=1)
        U = sy.solve(B, rtol=rtol)
        P[lo:lo + len(part)] = p.solver.to_host((B * U).sum(axis=0))
        for k in range(len(part)):
            b = B[sy.keep, k]
            res = max(res, float(xp.linalg.norm(b - sy.A @ U[sy.keep, k])
                                 / max(float(xp.linalg.norm(b)), 1e-300)))
        ucell[:, lo:lo + len(part)] = insulator._cell_u(U, sy.unk_cell, open_, p)
        del B, U
    n = sy.n
    del sy
    p.solver.release()
    return OSolve(ucell=ucell, P=P, unknowns=n, resid=res, fields=len(fields),
                  t=time.perf_counter() - t0)


class Clearing(NamedTuple):
    """A stored clearing of a large block as insulator.link_large reads it: its open field (the
    door world's opened cells kept fully open), buildings cleared, its stored Lens A."""
    name: str
    open_: np.ndarray
    n_cleared: int
    stored: float


def clearings(bk: Block) -> list[Clearing]:
    lab = bk.grid.label_sub(bk.polys)
    full = bk.grid.isub.mean(axis=-1)
    opened = bk.pk.open_ != bk.grid.ff0
    out = []
    for cname, (rel, col, perm_col) in insulator.STORED.items():
        r = pd.read_parquet(HERE / rel / f"{bk.bid}_p{insulator.CLEARING_POWER:g}.parquet").iloc[0]
        x = np.zeros(len(bk.polys), dtype=bool)
        x[np.asarray(r[col], dtype=np.int64)] = True
        cleared = (lab >= 0) & x[np.maximum(lab, 0)]
        out.append(Clearing(name=cname, open_=np.where(opened, full, (
            bk.grid.isub & (~bk.grid.bsub | cleared)).mean(axis=-1)), n_cleared=int(x.sum()),
            stored=float(r[perm_col])))
    return out


def j2_row(bk: Block, w: np.ndarray, u: np.ndarray) -> dict:
    """One scenario's J_2 over every home, the tier-0 homes and the walled-in homes."""
    t0 = bk.pl.tier == 0
    return dict(J2=float((w * u ** 2).sum()), J2_open=float((w[t0] * u[t0] ** 2).sum()),
                J2_walled=float((w[bk.wl] * u[bk.wl] ** 2).sum()),
                u_walled=",".join(f"{v:.4g}" for v in u[bk.wl]))


def _write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, out)


# ------------------------------------------------------------------ the crossings

class Crossing(Protocol):
    """How the walled-in homes' walkers get out (resolved once, from the command line)."""

    @property
    def rows(self) -> Path: ...

    def small_missing(self, bid: str) -> bool: ...

    def small(self, bk: Block) -> None: ...

    def large(self, bk: Block) -> None: ...


def crossing_paths(bk: Block) -> tuple[np.ndarray, ...]:
    """Per site, its baseline crossing's cells from the street end b (excluded) to the pocket
    end a: _crossing_graph's Dijkstra from the open street-side cells, as pockets_of runs it."""
    o = bk.pk.open_.ravel()
    _d, pred, _src = dijkstra(insulator._crossing_graph(bk.grid, bk.pk.open_),
                              indices=np.flatnonzero(insulator.OpenEnds().sources(
                                  o, bk.pk.full, bk.pk.reach)),
                              min_only=True, return_predecessors=True)
    out = []
    for a, b in zip(bk.sites.a, bk.sites.b, strict=True):
        cells, x = [], int(a)
        while pred[x] >= 0:
            cells.append(x)
            x = int(pred[x])
        assert x == b, "a crossing's path does not start at its site's b"
        out.append(np.asarray(cells, dtype=np.int64))
    return tuple(out)


def footprint_m(bk: Block, paths: tuple[np.ndarray, ...], o: np.ndarray) -> np.ndarray:
    """Per site, the metres of footprint standing on its crossing in the world `o` (flat):
    sum over the path's cells of h (1 - o)."""
    return np.array([bk.grid.h * float((1.0 - o[c]).sum()) for c in paths])


def toll_cols(bk: Block, w: np.ndarray, uc: np.ndarray, T: np.ndarray) -> dict:
    """Per home, from O's cell potentials `uc` of the full field and the crossings' footprint
    metres T: u_base (walled-in: the share-weighted u_O(b); others: u), toll_metre (the
    share-weighted d T: u = u_base + toll_metre / g1) and toll_fixed (d: + toll_fixed / g)."""
    n, s = len(bk.polys), bk.sites
    d = w[s.home] * bk.share
    u_base = np.where(bk.wl, np.bincount(s.home, weights=bk.share * uc[s.b], minlength=n),
                      bk.pl.home_u(uc))
    return dict(u_base=u_base,
                toll_metre=np.bincount(s.home, weights=bk.share * d * T, minlength=n),
                toll_fixed=np.bincount(s.home, weights=bk.share * d, minlength=n))


def toll_power_err(bk: Block, w: np.ndarray, tc: dict, T: np.ndarray, P_O: float) -> float:
    """The power identity sum_i w_i u_i = P_O + sum_s d_s^2 T_s / g1 at G1_GRID's ends, and
    + sum_s d_s^2 / g at G_GRID's: the worst relative error."""
    d = w[bk.sites.home] * bk.share
    err = []
    for col, R, gs in (("toll_metre", T, (G1_GRID[0], G1_GRID[-1])),
                       ("toll_fixed", np.ones(len(T)), (G_GRID[0], G_GRID[-1]))):
        for g in gs:
            lhs = float((w * (tc["u_base"] + tc[col] / g)).sum())
            err.append(abs(lhs - (P_O + float((d * d * R).sum()) / g)) / lhs)
    return max(err)


@dataclass(frozen=True)
class Toll:
    """A toll per crossing site, paid after the solve: no parameter at run time."""

    @property
    def rows(self) -> Path:
        return ROWS / f"crossing_toll_{SUFFIX}"

    def small_missing(self, bid: str) -> bool:
        return not (self.rows / f"{bid}.parquet").exists()

    def _full(self, bk: Block, w: np.ndarray) -> np.ndarray:
        """The full demand: the open homes' on their cells, the walled-in homes' at b."""
        return bk.pl.field(w, ~bk.wl, bk.shape) + bk.at(bk.sites.b,
                                                         w[bk.sites.home] * bk.share)

    def _sites(self, bk: Block, paths: tuple[np.ndarray, ...], o: np.ndarray) -> pd.DataFrame:
        """Per site: its ends, share, barrier_m and T, the footprint metres on its baseline
        crossing in the world `o` (flat)."""
        xy = bk.grid.xy.reshape(-1, 2)
        s = bk.sites
        return pd.DataFrame(dict(block=bk.bid, pocket=s.pocket, home=s.home, a=s.a, b=s.b,
                                 ax=xy[s.a, 0], ay=xy[s.a, 1], bx=xy[s.b, 0], by=xy[s.b, 1],
                                 barrier_m=s.barrier_m, share=bk.share,
                                 T=footprint_m(bk, paths, o),
                                 path_cells=[len(c) for c in paths],
                                 b_ground=bk.grid.ground.ravel()[s.b]))

    def small(self, bk: Block) -> None:
        t0 = time.perf_counter()
        paths = crossing_paths(bk)
        sites = self._sites(bk, paths, bk.pk.open_.ravel())
        t_paths = time.perf_counter() - t0
        T = bk.sites.barrier_m
        assert np.abs(sites["T"].to_numpy() - T).max(initial=0.0) <= 1e-6, "T_s(0) != barrier_m"
        fields = []
        for w in bk.W.values():
            fields += [bk.pl.field(w, bk.pl.tier == 0, bk.shape), self._full(bk, w)]
        so = o_solve(bk, bk.pk.open_, fields, rtol=insulator.RTOL)
        rows = []
        for k, (name, w) in enumerate(bk.W.items()):
            tc = toll_cols(bk, w, so.ucell[:, 2 * k + 1], T)
            err = toll_power_err(bk, w, tc, T, float(so.P[2 * k + 1]))
            assert err <= 1e-6, f"toll power identity {err:.1e}"
            rows.append(pd.DataFrame(dict(
                **bk.homes(name), **tc, u_open=bk.u_open(so.ucell[:, 2 * k]),
                sites=len(bk.sites.a), P_O=so.P[2 * k + 1], power_err=err,
                unknowns=so.unknowns, resid=so.resid, t_setup=bk.t_setup, t_paths=t_paths,
                t_solve=so.t)))
        _write(sites, self.rows / f"{bk.bid}__sites.parquet")
        _write(pd.concat(rows, ignore_index=True), self.rows / f"{bk.bid}.parquet")
        print(f"{time.strftime('%H:%M:%S')} {bk.bid} toll: {int(bk.wl.sum())} walled-in, "
              f"{len(bk.sites.a)} sites, setup {bk.t_setup:.1f}s, paths {t_paths:.1f}s, O "
              f"{so.unknowns} unknowns {so.fields} fields {so.t:.1f}s, resid {so.resid:.1e}",
              flush=True)

    def large(self, bk: Block) -> None:
        """Nothing cleared (both populations, per home: u_base, toll_metre, toll_fixed and
        u_open) and the stored clearings (area, every home: per scenario and home, T of that
        world on the baseline crossing paths; b stays)."""
        paths = crossing_paths(bk)
        sites = self._sites(bk, paths, bk.pk.open_.ravel()).assign(clearing="none",
                                                                   pocket_open=False)
        T = bk.sites.barrier_m
        assert np.abs(sites["T"].to_numpy() - T).max(initial=0.0) <= 1e-6, "T_s(0) != barrier_m"
        fields = []
        for w in bk.W.values():
            fields += [bk.pl.field(w, bk.pl.tier == 0, bk.shape), self._full(bk, w)]
        so = o_solve(bk, bk.pk.open_, fields, rtol=insulator.RTOL_LARGE)
        homes, scen, allsites = [], [], [sites]
        k_area = list(bk.W).index("area")   # the stored clearings' population (link_large's)
        for k, (name, w) in enumerate(bk.W.items()):
            tc = toll_cols(bk, w, so.ucell[:, 2 * k + 1], T)
            err = toll_power_err(bk, w, tc, T, float(so.P[2 * k + 1]))
            assert err <= 1e-6, f"toll power identity {err:.1e}"
            homes.append(pd.DataFrame(dict(
                **bk.homes(name), world="toll", **tc, u_open=bk.u_open(so.ucell[:, 2 * k]),
                sites=len(bk.sites.a), P_O=so.P[2 * k + 1], power_err=err,
                unknowns=so.unknowns, resid=so.resid, t_solve=so.t)))
            if k == k_area:
                scen.append(pd.DataFrame(dict(**bk.homes(name), clearing="none", n_cleared=0,
                                              stored_perm=np.nan, **tc, P_O=so.P[2 * k + 1],
                                              unknowns=so.unknowns, resid=so.resid,
                                              t_solve=so.t)))
        print(f"  {bk.bid} toll none: {so.unknowns} unknowns, resid {so.resid:.1e}, "
              f"{so.t:.0f}s", flush=True)
        w = bk.W["area"]
        for cl in clearings(bk):
            Tc = footprint_m(bk, paths, cl.open_.ravel())
            reach = lifted.grounded(bk.grid, cl.open_, bk.p).ravel()
            lab = bk.pk.label[reach & (bk.pk.label >= 0)]
            opened = (np.bincount(lab, minlength=len(bk.pk.homes)) > 0)[bk.sites.pocket]
            allsites.append(self._sites(bk, paths, cl.open_.ravel()).assign(
                clearing=cl.name, pocket_open=opened))
            s1 = o_solve(bk, cl.open_, [self._full(bk, w)], rtol=insulator.RTOL_LARGE)
            tc = toll_cols(bk, w, s1.ucell[:, 0], Tc)
            scen.append(pd.DataFrame(dict(**bk.homes("area"), clearing=cl.name,
                                          n_cleared=cl.n_cleared, stored_perm=cl.stored, **tc,
                                          P_O=s1.P[0], unknowns=s1.unknowns, resid=s1.resid,
                                          t_solve=s1.t)))
            print(f"  {bk.bid} toll {cl.name}: {s1.unknowns} unknowns, resid {s1.resid:.1e}, "
                  f"{s1.t:.0f}s, crossings shortened {int((Tc < T - 1e-9).sum())}, pockets "
                  f"opened {int(opened.sum())}", flush=True)
        _write(pd.concat(allsites, ignore_index=True), self.rows / f"{bk.bid}__sites.parquet")
        _write(pd.concat(homes, ignore_index=True), self.rows / f"{bk.bid}__large_homes.parquet")
        _write(pd.concat(scen, ignore_index=True), self.rows / f"{bk.bid}__large.parquet")


class Walls(NamedTuple):
    """M in one world at one eps: its unknown cells `unk` (I) and exits E (flat; E held at 0),
    the cells dropped (I components with no edge to E), L_II factorized, Bc (I x E): each I-E
    edge's conductance."""
    unk: np.ndarray
    E: np.ndarray
    dropped: np.ndarray
    solve: Callable
    Bc: sp.csr_matrix


def window(bk: Block, *, window_m: float) -> np.ndarray:
    """The cells M lives on (flat bool): the union over the walled-in homes of the bounding box
    of the home's injection cells, padded by ceil(window_m / h) cells, clipped to the grid."""
    ny, nx = bk.shape
    pad = math.ceil(window_m / bk.grid.h)
    Sc = bk.pl.S.tocsc()
    win = np.zeros((ny, nx), dtype=bool)
    for j in np.flatnonzero(bk.wl):
        r, c = np.divmod(Sc.indices[Sc.indptr[j]:Sc.indptr[j + 1]], nx)
        win[max(r.min() - pad, 0):r.max() + pad + 1, max(c.min() - pad, 0):c.max() + pad + 1] = True
    return win.ravel()


def walls(grid: lifted.Grid, o: np.ndarray, full: np.ndarray, exits: np.ndarray,
          win: np.ndarray, *, eps: float) -> Walls:
    """M in the world `o` (flat open fractions; `full` the inside fractions) at `eps`, on the
    inside cells of `win`: per cell sigma = (o + eps (full - o)) / full, between 4-neighbours a
    conductance C0 2 sigma_a sigma_b / (sigma_a + sigma_b); `exits` (flat bool) held at 0."""
    nx, size = grid.inside.shape[1], o.size
    inm = win & (full > 0)
    cells = np.flatnonzero(inm)
    sig = (o[cells] + eps * (full[cells] - o[cells])) / full[cells]
    right, down = cells + 1, cells + nx
    okr = (cells % nx != nx - 1) & inm[np.minimum(right, size - 1)]
    okd = (down < size) & inm[np.minimum(down, size - 1)]
    a = np.concatenate([cells[okr], cells[okd]])
    b = np.concatenate([right[okr], down[okd]])
    sa, sb = sig[np.searchsorted(cells, a)], sig[np.searchsorted(cells, b)]
    g = C0 * 2.0 * sa * sb / (sa + sb)
    xa, xb = exits[a], exits[b]
    unk_all, E = cells[~exits[cells]], cells[exits[cells]]
    ii, ie, ei = ~xa & ~xb, ~xa & xb, xa & ~xb
    ia, ib, gi = np.searchsorted(unk_all, a[ii]), np.searchsorted(unk_all, b[ii]), g[ii]
    ri = np.concatenate([np.searchsorted(unk_all, a[ie]), np.searchsorted(unk_all, b[ei])])
    re = np.concatenate([np.searchsorted(E, b[ie]), np.searchsorted(E, a[ei])])
    ge = np.concatenate([g[ie], g[ei]])
    nI = len(unk_all)
    nc, comp = connected_components(sp.coo_matrix((np.ones(len(ia)), (ia, ib)), shape=(nI, nI)),
                                    directed=False)
    has = np.zeros(nc, dtype=bool)
    has[comp[ri]] = True
    keep = has[comp]
    new = np.cumsum(keep) - 1
    nk = int(keep.sum())
    diag = (np.bincount(ia, gi, nI) + np.bincount(ib, gi, nI) + np.bincount(ri, ge, nI))[keep]
    m = keep[ia]
    L = sp.coo_matrix((np.concatenate([-gi[m], -gi[m], diag]),
                       (np.concatenate([new[ia[m]], new[ib[m]], np.arange(nk)]),
                        np.concatenate([new[ib[m]], new[ia[m]], np.arange(nk)]))),
                      shape=(nk, nk)).tocsc()
    me = keep[ri]
    return Walls(unk=unk_all[keep], E=E, dropped=unk_all[~keep], solve=factorized(L),
                 Bc=sp.csr_matrix((ge[me], (new[ri[me]], re[me])), shape=(nk, len(E))))


class Through(NamedTuple):
    """One population's walled-in demand through M: v on I, exit currents c on E, the demand
    on exits f_E (a cleared world's opened pocket: it enters O there), sum(f_I) . v, the flux
    error |sum c - sum f_I| / sum f_I and c's minimum."""
    v: np.ndarray
    c: np.ndarray
    f_E: np.ndarray
    fv: float
    flux_err: float
    c_min: float


def through(m: Walls, f_W: np.ndarray) -> Through:
    on = np.flatnonzero(f_W)
    assert np.isin(on, np.concatenate([m.unk, m.E])).all(), "walled-in demand off M's I and E"
    f_I = f_W[m.unk]
    v = m.solve(f_I)
    c = m.Bc.T @ v
    flux_err = abs(float(c.sum() - f_I.sum())) / float(f_I.sum()) if f_I.sum() > 0 else 0.0
    c_min = float(c.min(initial=0.0))
    assert flux_err <= 1e-9 and c_min >= -1e-12, f"M's exit currents: {flux_err:.1e} {c_min:.1e}"
    return Through(v=v, c=c, f_E=f_W[m.E], fv=float(f_I @ v), flux_err=flux_err, c_min=c_min)


def medium_u(bk: Block, m: Walls, th: Through, uc: np.ndarray) -> dict:
    """Per home: u (walled-in: through M, then O from its exits; the rest: O), toll (M's part,
    exits at 0) and beyond (O's potential where M lets the walker out; nan off the walled-in)."""
    H = m.solve(m.Bc @ uc[m.E])
    SI = bk.pl.S[m.unk]
    toll = SI.T @ th.v
    beyond = SI.T @ H + bk.pl.S[m.E].T @ uc[m.E]
    return dict(u=np.where(bk.wl, toll + beyond, bk.pl.home_u(uc)),
                toll=np.where(bk.wl, toll, np.nan), beyond=np.where(bk.wl, beyond, np.nan))


def egress(bk: Block, m: Walls) -> dict:
    """Per walled-in home, from its injection alone (count population) through M: the share of
    its exit current landing within NEAR_B_M of b of its main site (its site with the largest
    share), and the RMS distance of its landing cells from their centroid (current-weighted);
    nan off the walled-in."""
    n = len(bk.polys)
    hs = np.flatnonzero(bk.wl)
    w1 = common.CountPopulation().weights(bk.polys)
    C = m.Bc.T @ m.solve(bk.pl.S[m.unk][:, hs].toarray() * w1[hs][None, :])
    xy = bk.grid.xy.reshape(-1, 2)
    xe = xy[m.E]
    near, rms = np.full(n, np.nan), np.full(n, np.nan)
    for k, j in enumerate(hs):
        ks = np.flatnonzero(bk.sites.home == j)
        b = bk.sites.b[ks[np.argmax(bk.share[ks])]]
        c = C[:, k]
        tot = c.sum()
        near[j] = c[np.hypot(*(xe - xy[b]).T) <= NEAR_B_M].sum() / tot
        cen = (c[:, None] * xe).sum(axis=0) / tot
        rms[j] = math.sqrt(float((c * ((xe - cen) ** 2).sum(axis=1)).sum() / tot))
    return dict(near_b_share=near, exit_rms_m=rms)


@dataclass(frozen=True)
class Medium:
    """Private walls conducting `eps` (each value a run) on the window `window_m` around the
    walled-in homes; on the large blocks the stored clearings at `eps_clear` (each in eps)."""
    window_m: float
    eps: tuple[float, ...]
    eps_clear: tuple[float, ...]

    @property
    def rows(self) -> Path:
        return ROWS / f"crossing_medium_w{self.window_m:g}_{SUFFIX}"

    def _path(self, bid: str, eps: float) -> Path:
        return self.rows / f"{bid}__eps{eps:g}.parquet"

    def small_missing(self, bid: str) -> bool:
        return any(not self._path(bid, e).exists() for e in self.eps)

    def _baseline(self, bk: Block, eps_list: tuple[float, ...], *, rtol: float
                  ) -> tuple[list[pd.DataFrame], OSolve]:
        """Nothing cleared, both populations, per eps in `eps_list`: one frame of rows per eps,
        and O's solution (fields: tier-0 demand per population, then the full demand per eps
        and population)."""
        o, full = bk.pk.open_.ravel(), bk.pk.full
        exits = insulator.OpenEnds().sources(o, full, bk.pk.reach)
        win = window(bk, window_m=self.window_m)
        through_of, fields = {}, [bk.pl.field(w, bk.pl.tier == 0, bk.shape)
                                  for w in bk.W.values()]
        for eps in eps_list:
            t0 = time.perf_counter()
            m = walls(bk.grid, o, full, exits, win, eps=eps)
            ths = {name: through(m, bk.pl.S @ np.where(bk.wl, w, 0.0))
                   for name, w in bk.W.items()}
            assert not any(th.f_E.any() for th in ths.values()), "walled-in demand on an exit"
            through_of[eps] = (m, ths, egress(bk, m), time.perf_counter() - t0)
            fields += [bk.pl.field(w, ~bk.wl, bk.shape) + bk.at(m.E, ths[name].c)
                       for name, w in bk.W.items()]
        so = o_solve(bk, bk.pk.open_, fields, rtol=rtol)
        npop = len(bk.W)
        out = []
        for e, eps in enumerate(eps_list):
            m, ths, eg, t_m = through_of[eps]
            rows = []
            for k, (name, w) in enumerate(bk.W.items()):
                t0 = time.perf_counter()
                col = npop + e * npop + k
                mu = medium_u(bk, m, ths[name], so.ucell[:, col])
                P = float(so.P[col]) + ths[name].fv
                err = abs(float((w * mu["u"]).sum()) - P) / P
                assert err <= 1e-6, f"medium power identity {err:.1e}"
                t_m += time.perf_counter() - t0
                rows.append(dict(**bk.homes(name), eps=eps, window_m=self.window_m, **mu,
                                 u_open=bk.u_open(so.ucell[:, k]), **eg,
                                 unknowns=so.unknowns, m_unknowns=len(m.unk), m_exits=len(m.E),
                                 m_dropped=len(m.dropped), flux_err=ths[name].flux_err,
                                 c_min=ths[name].c_min, P_O=so.P[col], P=P, power_err=err,
                                 resid=so.resid, o_fields=len(fields), t_setup=bk.t_setup,
                                 t_solve=so.t))
            out.append(pd.concat([pd.DataFrame(r).assign(t_medium=t_m) for r in rows],
                                 ignore_index=True))
        return out, so

    def _none(self, bk: Block, eps_list: tuple[float, ...], *, rtol: float
              ) -> tuple[list[pd.DataFrame], OSolve]:
        """A block with no walled-in home: one O solve, the same rows for every eps."""
        fields = []
        for w in bk.W.values():
            fields += [bk.pl.field(w, bk.pl.tier == 0, bk.shape), bk.pl.field(w, ~bk.wl,
                                                                            bk.shape)]
        so = o_solve(bk, bk.pk.open_, fields, rtol=rtol)
        out = []
        for eps in eps_list:
            rows = []
            for k, (name, w) in enumerate(bk.W.items()):
                u = bk.pl.home_u(so.ucell[:, 2 * k + 1])
                P = float(so.P[2 * k + 1])
                nan = np.full(len(bk.polys), np.nan)
                rows.append(pd.DataFrame(dict(
                    **bk.homes(name), eps=eps, window_m=self.window_m, u=u, toll=nan,
                    beyond=nan, u_open=bk.u_open(so.ucell[:, 2 * k]), near_b_share=nan,
                    exit_rms_m=nan, unknowns=so.unknowns, m_unknowns=0, m_exits=0,
                    m_dropped=0, flux_err=0.0, c_min=0.0, P_O=P, P=P,
                    power_err=abs(float((w * u).sum()) - P) / P, resid=so.resid,
                    o_fields=len(fields), t_setup=bk.t_setup, t_solve=so.t, t_medium=0.0)))
            out.append(pd.concat(rows, ignore_index=True))
        return out, so

    def small(self, bk: Block) -> None:
        todo = tuple(e for e in self.eps if not self._path(bk.bid, e).exists())
        run = self._baseline if bk.wl.any() else self._none
        frames, so = run(bk, todo, rtol=insulator.RTOL)
        for eps, df in zip(todo, frames, strict=True):
            _write(df, self._path(bk.bid, eps))
        tm = " ".join(f"{e:g}:{df.t_medium.iloc[0]:.1f}s/{df.m_unknowns.iloc[0]}"
                      for e, df in zip(todo, frames, strict=True))
        print(f"{time.strftime('%H:%M:%S')} {bk.bid} medium w{self.window_m:g}: "
              f"{int(bk.wl.sum())} walled-in, setup {bk.t_setup:.1f}s, O {so.unknowns} "
              f"unknowns {so.fields} fields {so.t:.1f}s, resid {so.resid:.1e}; M (eps: "
              f"s/unknowns) {tm}", flush=True)

    def large(self, bk: Block) -> None:
        """Nothing cleared at every eps (both populations, per home), and the stored clearings
        at each of eps_clear (area, every home): sigma and the exits from the cleared field; the
        walled-in demand landing on an exit there enters O at that cell."""
        assert set(self.eps_clear) <= set(self.eps), "a clearing eps not in the eps list"
        frames, so = self._baseline(bk, self.eps, rtol=insulator.RTOL_LARGE)
        homes = pd.concat(frames, ignore_index=True).assign(world="medium")
        w = bk.W["area"]
        full = bk.pk.full
        scen = []
        for eps in self.eps_clear:
            h0 = homes[(homes.eps == eps) & (homes["pop"] == "area")]
            scen.append(dict(block=bk.bid, eps=eps, clearing="none", n_cleared=0,
                             stored_perm=np.nan, **j2_row(bk, w, h0.u.to_numpy()),
                             P=float(h0.P.iloc[0]), P_sum=float((w * h0.u.to_numpy()).sum()),
                             unknowns=so.unknowns, m_unknowns=int(h0.m_unknowns.iloc[0]),
                             walled_on_exit=0.0, resid=so.resid,
                             power_err=float(h0.power_err.iloc[0]), t_solve=so.t,
                             t_medium=float(h0.t_medium.iloc[0])))
        win = window(bk, window_m=self.window_m)
        for cl in clearings(bk):
            o = cl.open_.ravel()
            exits = lifted.grounded(bk.grid, cl.open_, bk.p).ravel() & (o >= full - 1e-12)
            for eps in self.eps_clear:
                t0 = time.perf_counter()
                m = walls(bk.grid, o, full, exits, win, eps=eps)
                th = through(m, bk.pl.S @ np.where(bk.wl, w, 0.0))
                t_m = time.perf_counter() - t0
                s1 = o_solve(bk, cl.open_, [bk.pl.field(w, ~bk.wl, bk.shape)
                                            + bk.at(m.E, th.c + th.f_E)],
                             rtol=insulator.RTOL_LARGE)
                t0 = time.perf_counter()
                mu = medium_u(bk, m, th, s1.ucell[:, 0])
                t_m += time.perf_counter() - t0
                P = float(s1.P[0]) + th.fv
                P_sum = float((w * mu["u"]).sum())
                err = abs(P_sum - P) / P
                assert err <= 1e-6, f"medium power identity {err:.1e}"
                scen.append(dict(block=bk.bid, eps=eps, clearing=cl.name,
                                 n_cleared=cl.n_cleared, stored_perm=cl.stored,
                                 **j2_row(bk, w, mu["u"]), P=P, P_sum=P_sum,
                                 unknowns=s1.unknowns, m_unknowns=len(m.unk),
                                 walled_on_exit=float(th.f_E.sum()), resid=s1.resid,
                                 power_err=err, t_solve=s1.t, t_medium=t_m))
                print(f"  {bk.bid} medium eps {eps:g} {cl.name}: {s1.unknowns} unknowns, resid "
                      f"{s1.resid:.1e}, O {s1.t:.0f}s, M {t_m:.1f}s", flush=True)
        _write(homes, self.rows / f"{bk.bid}__large_homes.parquet")
        _write(pd.DataFrame(scen), self.rows / f"{bk.bid}__large.parquet")


def crossing_of(args: list[str], *, large: bool) -> tuple[Crossing, list[str]]:
    """The model a command line names (the one place it is a string), and the arguments left:
    toll | medium <window_m> <eps,> (`large`: and <eps for the clearings,>)."""
    floats = lambda s: tuple(float(x) for x in s.split(","))    # noqa: E731
    if args[0] == "toll":
        return Toll(), args[1:]
    if args[0] == "medium":
        n = 4 if large else 3
        return (Medium(window_m=float(args[1]), eps=floats(args[2]),
                       eps_clear=floats(args[3]) if large else ()), args[n:])
    raise SystemExit(f"unknown crossing {args[0]!r}: toll | medium\n{__doc__}")


# ------------------------------------------------------------------ runs

def small_one(bid: str, crossing: Crossing) -> None:
    if crossing.small_missing(bid):
        crossing.small(Block.of(bid, insulator.params("cpu")))


def large(ids: list[str], crossing: Crossing) -> None:
    p = insulator.params("gpu")
    for bid in ids:
        if (crossing.rows / f"{bid}__large.parquet").exists():
            continue
        try:
            t0 = time.perf_counter()
            bk = Block.of(bid, p)
            print(f"{time.strftime('%H:%M:%S')} {bid} n={len(bk.polys)} cells="
                  f"{int(bk.grid.inside.sum())} walled-in {int(bk.wl.sum())} sites "
                  f"{len(bk.sites.a)} setup {bk.t_setup:.0f}s", flush=True)
            if not bk.wl.any():
                print(f"{bid}: no walled-in home under the fix", flush=True)
                continue
            crossing.large(bk)
            print(f"{time.strftime('%H:%M:%S')} {bid} done in {time.perf_counter() - t0:.0f}s",
                  flush=True)
            del bk
        except Exception as e:
            print(f"{time.strftime('%H:%M:%S')} {bid} FAILED {type(e).__name__}: {e}"[:400],
                  flush=True)
            traceback.print_exc()
        gc.collect()
        p.solver.release()


# ------------------------------------------------------------------ report (small blocks)

def _walled_r(rows: pd.DataFrame, key: str) -> pd.DataFrame:
    """insulator.ratio_rows over the walled-in homes (tiers 1, 3, 4, 5)."""
    rat = insulator.ratio_rows(insulator._first_g(rows), rows, key)
    return rat[(rat.tier == 1) | (rat.tier >= 3)]


def _toll_world(toll: pd.DataFrame, g: float, col: str) -> pd.DataFrame:
    """The toll's rows at g (g1 for toll_metre): u = u_base + toll / g."""
    return toll.assign(u=toll.u_base + toll[col] / g, g=g)


def _calibrate(toll: pd.DataFrame, col: str) -> float:
    """The g (or g1) at which the walled-in homes' median r (count) is TARGET_R: bisection in
    log g (u is affine in 1 / g)."""
    lo, hi = math.log(1e-4), math.log(10.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        rat = _walled_r(_toll_world(toll, math.exp(mid), col), "g")
        if float(np.median(rat[rat["pop"] == "count"].r)) > TARGET_R:
            lo = mid
        else:
            hi = mid
    return math.exp(0.5 * (lo + hi))


def report() -> None:
    """The small blocks at matched medians (the door at DOOR_G, each toll at its calibrated g,
    the medium on its grid), r against pocket size, and the medium's egress, as markdown, with
    the CSV summaries summary_crossing_small.csv and summary_crossing_size.csv."""
    door = insulator._read(ROWS / f"door_{DOOR}_{SUFFIX}", f"ZAF*__g{DOOR_G:g}.parquet")
    toll = insulator._read(Toll().rows, "ZAF*[0-9].parquet")
    med = insulator._read(ROWS / f"crossing_medium_w25_{SUFFIX}", "*__eps*.parquet")
    mr = _walled_r(med, "eps")
    grid = np.array(sorted(mr.eps.unique()))
    meds = np.array([np.median(mr[(mr.eps == e) & (mr["pop"] == "count")].r) for e in grid])
    eps_star = insulator.eps_star(meds, grid, TARGET_R)
    g_fixed, g1 = _calibrate(toll, "toll_fixed"), _calibrate(toll, "toll_metre")
    eps_m = float(grid[np.argmin(np.abs(np.log(grid / eps_star)))])   # the grid value nearest
    worlds = {f"door g {DOOR_G:g}": _walled_r(door, "g"),
              f"toll per crossing g {g_fixed:.3g}": _walled_r(_toll_world(toll, g_fixed,
                                                                         "toll_fixed"), "g"),
              f"toll per metre g1 {g1:.3g}": _walled_r(_toll_world(toll, g1, "toll_metre"), "g"),
              f"private walls eps {eps_m:g}": mr[mr.eps == eps_m]}
    pk = insulator._first_g(toll)[["block", "pop", "home", "pocket_homes"]]
    a, size = [], []
    for model, rat in worlds.items():
        for pop, g in rat.groupby("pop"):
            r = g.r.to_numpy()
            a.append(dict(model=model, pop=pop, homes=len(r), r_med=float(np.median(r)),
                          r_p90=float(np.quantile(r, 0.9)), r_max=float(r.max()),
                          share_r_le_1=float((r <= 1).mean()), share_r_gt_3=float((r > 3).mean())))
        m = rat[rat["pop"] == "count"].merge(pk, on=["block", "pop", "home"])
        for cls, g in m.groupby(pd.cut(m.pocket_homes, [0, 1, 3, 100], labels=["1", "2-3", "4+"]),
                                observed=True):
            size.append(dict(model=model, pocket_homes=str(cls), homes=len(g),
                             r_med=float(g.r.median()), r_p90=float(g.r.quantile(0.9)),
                             r_max=float(g.r.max())))
    a, size = pd.DataFrame(a), pd.DataFrame(size)
    a.to_csv(ROWS / "summary_crossing_small.csv", index=False)
    size.to_csv(ROWS / "summary_crossing_size.csv", index=False)
    f3 = lambda cols: {c: "{:.3g}" for c in cols}           # noqa: E731
    print(f"calibrated to median r {TARGET_R} (count): toll per crossing g* {g_fixed:.4g}, per "
          f"metre g1* {g1:.4g}; private walls eps* {eps_star:.3g} (grid {eps_m:g})\n")
    print(insulator._md(a, f3(a.columns)))
    print()
    print(insulator._md(size, f3(size.columns)))
    eg = med[(med.eps == eps_m) & (med["pop"] == "count") & ((med.tier == 1) | (med.tier >= 3))]
    print(f"\nprivate walls at {eps_m:g}: near_b_share median {eg.near_b_share.median():.2f} "
          f"(p10 {eg.near_b_share.quantile(0.1):.2f}, p90 {eg.near_b_share.quantile(0.9):.2f}), "
          f"exit_rms_m median {eg.exit_rms_m.median():.1f}")


def main(argv: list[str]) -> None:
    cmd = argv[0]
    insulator._check_single_threaded()
    if cmd == "small":
        crossing, rest = crossing_of(argv[2:], large=False)
        ids = insulator._by_size(rest[0].split(","))
        insulator._pool(int(argv[1]), small_one, [(i, crossing) for i in ids])
    elif cmd == "large":
        crossing, rest = crossing_of(argv[1:], large=True)
        large(rest[0].split(","), crossing)
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
