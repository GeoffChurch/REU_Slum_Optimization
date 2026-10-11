"""The insulator world: buildings as imperfect insulators, so that no home is excluded.

The owner (2026-10-11): "We shouldn't exclude homes. Instead, we could maybe have buildings be
imperfect insulators, which is perhaps related to transparency. That way, escape time would be
finite. They should be fairly good insulators but we should get realistic results (e.g. the
worst escape time should be worse, but not massively worse, than any of its neighbors that have
open paths)".

Today (common.Scorer) a home whose ring has no path to the street through open space is
STRANDED: it injects nothing and is left out of P and J_2. Here every cell's open fraction is
o_eps = o + eps (full - o): a footprint sub-sample conducts eps relative to an open one (the eps
world clear.py and relax.py search in, made the metric), so every inside cell is an unknown and
every home drains. Clearing raises a cell from eps toward 1, so the score stays monotone. Where
each home injects (`Placement`, by tier):
  0  a ring cell (a free cell within RING_M whose nearest building it is) reaches the street
     through open space: today's placement (lifted.demand), on the open side;
  1  none does (today's stranded): uniformly over all its ring cells, sealed as they are;
  a stranded home with no ring cell at all (the nearest-building rule gave every free cell
  near it to a neighbour, or footprint all round) uses the free cells within RING_M of its own
  outline, whoever's ring they are in:
  2  those reaching the street, if any (its door on the open side: not walled in);
  3  else all of them, sealed;
  4  no free cell within RING_M: the cells its footprint covers, by the sub-samples it holds;
  5  a footprint holding no sub-sample: the inside cell nearest its centroid.
eps 0 is today's rule (tier 0 only, the rest excluded): the reference. lifted.py and common.py
are used as they are (a long run imports them); everything new is here.

    S=research/roadless/insulator.py      # from the repo root, PYTHONPATH=., single-threaded
    REBLOCK_BLOCK_BANK=<bank> uv run python $S small <procs> <ids,>      # eps grid, h 0.5, CPU
    REBLOCK_BLOCK_BANK=<bank> uv run python $S fine <procs> <h> <ids,>   # eps 0 at a finer h
    REBLOCK_BLOCK_BANK=<bank> CUDA_PATH=/usr uv run python $S large <ids,>   # GPU, stored clearings
    REBLOCK_BLOCK_BANK=<bank> uv run python $S census <procs> <ids,>     # stranded share, no solve
    REBLOCK_BLOCK_BANK=<bank> uv run python $S door <procs> <kind> <ids,>   # confined leaks
    REBLOCK_BLOCK_BANK=<bank> CUDA_PATH=/usr uv run python $S large-door <kind> <ids,>
    uv run python $S report       # every table of insulator_report.md, and the CSV summaries

<kind>: metre | leaf | straight | local (DoorKind; the section "the door"). Rows in
insulator_rows/ (parquet, ignored); the report's CSV summaries sit beside them; the findings
in insulator_report.md.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402

# ------------------------------------------------------------------ the run's parameters
# Everything that can change a number is here or on the command line, and in the rows' paths.
EPS_GRID = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)   # 0: today's rule (the reference)
H = 0.5                  # the metric's cell size (m)
ELL_M = 3.0              # turning length (m)
K = 8                    # headings
RING_M = 1.0             # a home's ring: free cells within this of it (lifted.demand's, today's)
NEIGHBOURS = 6           # N(i): the nearest open-access homes by centroid distance
RTOL = 1e-8              # CG relative residual: per-home u to ~1e-7 (P, quadratic, far better)
RTOL_LARGE = 1e-6        # large blocks (GPU): Lens A and per-home u; the stored scores used 1e-5
SINGLE = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
              NUMBA_NUM_THREADS="1")
ROWS = HERE / "insulator_rows"
# road_rescore.STORED's clearings optimized and scored on the uniform h 0.5 mesh (the a5x8 one
# is on another mesh): (rows, cleared column, Lens A column)
STORED = {
    "simp_default": ("relax_rows/uni/fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2/D0.05",
                     "simp_cleared", "simp_perm"),
    "cheap": ("polish_rows/uni/S0.01cat.P64w8/D0.05", "cleared", "perm"),
}
CLEARING_POWER = 2.0     # the stored clearings' objective and Lens A: J_2 under area population


def _check_single_threaded() -> None:
    bad = {k: os.environ.get(k) for k, v in SINGLE.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f"run single-threaded: set {bad} to {SINGLE}")


def params(device: str) -> lifted.Params:
    return lifted.Params(ell_m=ELL_M, K=K, along=lifted.Uniform(),
                         solver=lifted.solver_of(device))


def mesh(h: float) -> lifted.UniformMesh:
    return lifted.UniformMesh(h, offset=lifted.OFFSET)


def insulator_open(open_: np.ndarray, full: np.ndarray, eps: float) -> np.ndarray:
    """o_eps = o + eps (full - o): footprint sub-samples conduct eps (eps 0: the field itself)."""
    return open_ if eps == 0.0 else open_ + eps * (full - open_)


# ------------------------------------------------------------------ demand

@dataclass(frozen=True, eq=False)
class Placement:
    """Where each home injects: S (grid cells flat x homes) holds home j's share of its
    injection per cell (each column sums to 1), so a population w injects S w and home j's
    escape time is (S^T u_cell)_j, the injection-weighted mean potential over its cells
    (common.Scorer.home_u_of's). `tier` per home (the module docstring). `barrier_m` per home:
    the fewest metres of footprint a 4-connected path from its cells to open space with a path
    to the street must cross (0 for tier 0). `shadowed` per home: a free cell within RING_M of
    its outline reaches the street (whoever's ring it is in): a stranded home so flagged is
    stranded by the nearest-building rule, not walled in."""
    S: sp.csr_matrix
    tier: np.ndarray
    barrier_m: np.ndarray
    shadowed: np.ndarray

    @classmethod
    def of(cls, grid: lifted.Grid, polys: np.ndarray, p: lifted.Params) -> Placement:
        n = len(polys)
        ncell = grid.inside.size
        reach = lifted.grounded(grid, grid.ff0, p)
        _f, stranded, own = lifted.demand(grid, polys, reach, np.ones(n), ring_m=RING_M)
        tier = np.where(stranded, 1, 0)
        shadowed = np.zeros(n, dtype=bool)
        on = np.flatnonzero(own.ravel() >= 0)
        cells, homes, vals = [on], [own.ravel()[on]], [np.ones(len(on))]

        def add(at: np.ndarray, j: int, v: np.ndarray | None = None) -> None:
            cells.append(at)
            homes.append(np.full(len(at), j))
            vals.append(np.ones(len(at)) if v is None else v)

        if stranded.any():
            # tier 1: every ring cell, sealed ones included; the same nearest-building rule
            sealed = (grid.ff0 > 0) & (grid.dist_b <= RING_M + 1e-9) & ~reach
            at = np.flatnonzero(sealed.ravel())
            have = np.zeros(n, dtype=bool)
            if len(at):
                xy = grid.xy.reshape(-1, 2)[at]
                idx = shapely.STRtree(polys).query_nearest(shapely.points(xy[:, 0], xy[:, 1]),
                                                           all_matches=False)
                owner = np.full(len(at), -1)
                owner[idx[0]] = idx[1]
                mine = (owner >= 0) & stranded[np.maximum(owner, 0)]
                cells.append(at[mine])
                homes.append(owner[mine])
                vals.append(np.ones(int(mine.sum())))
                have[owner[mine]] = True
            reach_flat = reach.ravel()
            footprint = []
            for j in np.flatnonzero(stranded):
                near = _near_outline(grid, polys[j])
                open_near = near[reach_flat[near]]
                shadowed[j] = len(open_near) > 0
                if have[j]:
                    continue
                if len(open_near):          # tier 2: its door on the open side
                    tier[j] = 2
                    add(open_near, j)
                elif len(near):             # tier 3: every free cell by its outline, sealed
                    tier[j] = 3
                    add(near, j)
                else:
                    footprint.append(j)
            if footprint:
                # tier 4: the cells its footprint covers, by sub-samples held
                lab = grid.label_sub(polys).reshape(ncell, -1)
                c, s_ = np.nonzero(np.isin(lab, footprint))
                m = sp.coo_matrix((np.ones(len(c)), (c, lab[c, s_])), shape=(ncell, n)).tocsc()
                m.sum_duplicates()
                ins = np.flatnonzero(grid.inside.ravel())
                tree = cKDTree(grid.xy.reshape(-1, 2)[ins])
                for j in footprint:
                    rows = m.indices[m.indptr[j]:m.indptr[j + 1]]
                    if len(rows):
                        tier[j] = 4
                        add(rows, j, m.data[m.indptr[j]:m.indptr[j + 1]])
                    else:                   # tier 5: the inside cell nearest its centroid
                        tier[j] = 5
                        _d, k = tree.query(shapely.get_coordinates(shapely.centroid(polys[j]))[0])
                        add(np.array([ins[k]]), j)
        c = np.concatenate(cells)
        h_ = np.concatenate(homes)
        v = np.concatenate(vals)
        S = sp.coo_matrix((v, (c, h_)), shape=(ncell, n)).tocsc()
        S.sum_duplicates()
        col = np.asarray(S.sum(axis=0)).ravel()
        assert (col > 0).all(), "a home with nowhere to inject"
        S = (S @ sp.diags(1.0 / col)).tocsr()
        barrier = np.zeros(n)
        if stranded.any():
            barrier = _barrier(grid, reach, S, stranded)
        return cls(S=S, tier=tier, barrier_m=barrier, shadowed=shadowed)

    def field(self, w: np.ndarray, homes: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """The injection of population w by the homes in mask `homes`, in the grid's shape."""
        return (self.S @ np.where(homes, w, 0.0)).reshape(shape)

    def home_u(self, u_cell: np.ndarray) -> np.ndarray:
        """Per home, from per-cell potentials (flat; columns for several solves)."""
        return np.asarray(self.S.T @ u_cell)


def _near_outline(grid: lifted.Grid, poly) -> np.ndarray:
    """Flat indices of the free cells (open fraction > 0) whose centres lie within RING_M of
    `poly`, whichever building is nearest them."""
    xs, ys = grid.xy[0, :, 0], grid.xy[:, 0, 1]
    x0, y0, x1, y1 = poly.bounds
    pad = RING_M + grid.h
    c0, c1 = np.searchsorted(xs, x0 - pad), np.searchsorted(xs, x1 + pad, side="right")
    r0, r1 = np.searchsorted(ys, y0 - pad), np.searchsorted(ys, y1 + pad, side="right")
    rr, cc = np.nonzero(grid.ff0[r0:r1, c0:c1] > 0)
    rr, cc = rr + r0, cc + c0
    ok = shapely.dwithin(poly, shapely.points(grid.xy[rr, cc, 0], grid.xy[rr, cc, 1]),
                         RING_M + 1e-9)
    return (rr * grid.inside.shape[1] + cc)[ok]


def _crossing_graph(grid: lifted.Grid) -> sp.csr_matrix:
    """4-connected steps between inside cells, a step costing the metres of footprint it
    enters, h (1 - o) at its target (plus 1e-9: no zero-weight edges)."""
    ny, nx = grid.inside.shape
    o = grid.ff0.ravel()
    ins = grid.inside.ravel()
    idx = np.arange(ny * nx).reshape(ny, nx)
    src, dst = [], []
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1, :], idx[1:, :])):
        a, b = a.ravel(), b.ravel()
        ok = ins[a] & ins[b]
        src += [a[ok], b[ok]]
        dst += [b[ok], a[ok]]
    s, d = np.concatenate(src), np.concatenate(dst)
    return sp.csr_matrix((grid.h * (1.0 - o[d]) + 1e-9, (s, d)), shape=(ny * nx, ny * nx))


def _barrier(grid: lifted.Grid, reach: np.ndarray, S: sp.csr_matrix,
             stranded: np.ndarray) -> np.ndarray:
    """Per home: the fewest metres of footprint crossed on a 4-connected path through inside
    cells from open space with a path to the street to the nearest of its cells
    (_crossing_graph's; 0 where reached already; tier 0 homes: 0)."""
    dist = dijkstra(_crossing_graph(grid), indices=np.flatnonzero(reach.ravel()), min_only=True)
    out = np.zeros(len(stranded))
    Sc = S.tocsc()
    for j in np.flatnonzero(stranded):
        rows = Sc.indices[Sc.indptr[j]:Sc.indptr[j + 1]]
        out[j] = float(dist[rows].min())
    return out


# ------------------------------------------------------------------ solves

def home_potentials(grid: lifted.Grid, open_: np.ndarray, fields: list[np.ndarray],
                    pl: Placement, p: lifted.Params, *, rtol: float
                    ) -> tuple[np.ndarray, np.ndarray, int]:
    """(home potentials (homes x fields), P = b.u per field, unknowns) of each injection field
    in the world `open_`: one system, the fields solved together."""
    xp = p.solver.xp
    sy = lifted.System(grid, open_, p)
    B = xp.stack([sy.load(f, p) for f in fields], axis=1)
    U = sy.solve(B, rtol=rtol)
    P = p.solver.to_host((B * U).sum(axis=0))
    share = xp.asarray(lifted.axes(p.K)[2] / np.pi)
    o = xp.asarray(open_).ravel()
    free = xp.flatnonzero(o > 0)
    live = sy.unk_cell >= 0
    ub = np.zeros((o.size, len(fields)))
    for j in range(len(fields)):
        uk = xp.zeros((len(free), p.K))
        uk[live] = U[:, j].reshape(-1, p.K)
        col = xp.zeros(o.size)
        col[free] = uk @ share
        ub[:, j] = p.solver.to_host(col)
    n = sy.n
    del sy, B, U
    p.solver.release()
    return pl.home_u(ub), P, n


def _centroids(polys: np.ndarray) -> np.ndarray:
    return shapely.get_coordinates(shapely.centroid(polys))


# ------------------------------------------------------------------ small blocks (CPU)

def small_dir() -> Path:
    return ROWS / f"small_h{H:g}_ell{ELL_M:g}_K{K}"


def fine_dir(h: float) -> Path:
    return ROWS / f"fine_h{h:g}_ell{ELL_M:g}_K{K}"


def _guard(fn, *args) -> None:
    try:
        fn(*args)
    except Exception as e:                       # one bad block must not kill the pool
        print(f"{time.strftime('%H:%M:%S')} {args} FAILED {type(e).__name__}: {e}"[:400],
              flush=True)
        traceback.print_exc()


def small_one(bid: str) -> None:
    """The eps grid at h 0.5 on one block, both populations: per home and eps, u (all homes
    injecting; eps 0: tier 0 only) and its part due to the tier-0 demand alone (u_open, the
    leakage-only world)."""
    out = small_dir() / f"{bid}.parquet"
    if out.exists():
        return
    t0 = time.perf_counter()
    [b] = common.build_blocks([bid])
    p = params("cpu")
    grid = mesh(H).build(b, p, None)
    polys = np.asarray(b.buildings.outlines)
    pl = Placement.of(grid, polys, p)
    W = {name: pop.weights(polys) for name, pop in common.POPULATIONS.items()}
    reach = lifted.grounded(grid, grid.ff0, p)
    open_home = pl.tier == 0
    for w in W.values():                        # tier 0 is today's demand, exactly
        f_today = lifted.demand(grid, polys, reach, w)[0]
        assert np.allclose(pl.field(w, open_home, grid.inside.shape), f_today, atol=1e-12)
    full = grid.isub.mean(axis=-1)
    cen = _centroids(polys)
    any_str = bool((~open_home).any())
    rows, meta = [], []
    for eps in EPS_GRID:
        op = insulator_open(grid.ff0, full, eps)
        fields, keys = [], []
        for name, w in W.items():
            fields.append(pl.field(w, open_home, grid.inside.shape))
            keys.append((name, "open"))
            if eps > 0 and any_str:
                fields.append(pl.field(w, ~open_home, grid.inside.shape))
                keys.append((name, "str"))
        ts = time.perf_counter()
        hu, P, n_unk = home_potentials(grid, op, fields, pl, p, rtol=RTOL)
        t_solve = time.perf_counter() - ts
        for name, w in W.items():
            u_open = hu[:, keys.index((name, "open"))]
            u_str = (hu[:, keys.index((name, "str"))] if (name, "str") in keys
                     else np.zeros(len(polys)))
            u = u_open + u_str
            if eps == 0.0:
                u = np.where(open_home, u, np.nan)
                u_open = np.where(open_home, u_open, np.nan)
            rows.append(pd.DataFrame(dict(
                block=bid, eps=eps, pop=name, home=np.arange(len(polys)), tier=pl.tier,
                barrier_m=pl.barrier_m, shadowed=pl.shadowed, w=w, cx=cen[:, 0],
                cy=cen[:, 1], u=u, u_open=u_open)))
        meta.append(dict(block=bid, eps=eps, n=len(polys), cells=int(grid.inside.sum()),
                         unknowns=n_unk, t_solve=t_solve,
                         P=",".join(f"{k[0]}-{k[1]}:{v:.10g}" for k, v in zip(keys, P,
                                                                               strict=True))))
    df = pd.concat(rows, ignore_index=True)
    small_dir().mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta).to_parquet(small_dir() / f"{bid}__meta.parquet")
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, out)
    n_t = np.bincount(pl.tier, minlength=4)
    print(f"{time.strftime('%H:%M:%S')} {bid} n={len(polys)} tiers {n_t.tolist()} "
          f"{time.perf_counter() - t0:.0f}s", flush=True)


def fine_one(bid: str, h: float) -> None:
    """eps 0 (today's rule) at a finer h, both populations: per home its tier there and u."""
    out = fine_dir(h) / f"{bid}.parquet"
    if out.exists():
        return
    t0 = time.perf_counter()
    [b] = common.build_blocks([bid])
    p = params("cpu")
    grid = mesh(h).build(b, p, None)
    polys = np.asarray(b.buildings.outlines)
    pl = Placement.of(grid, polys, p)
    W = {name: pop.weights(polys) for name, pop in common.POPULATIONS.items()}
    open_home = pl.tier == 0
    fields = [pl.field(w, open_home, grid.inside.shape) for w in W.values()]
    ts = time.perf_counter()
    hu, P, n_unk = home_potentials(grid, grid.ff0, fields, pl, p, rtol=RTOL)
    t_solve = time.perf_counter() - ts
    cen = _centroids(polys)
    df = pd.concat([pd.DataFrame(dict(
        block=bid, h=h, pop=name, home=np.arange(len(polys)), tier=pl.tier,
        barrier_m=pl.barrier_m, shadowed=pl.shadowed, w=w, cx=cen[:, 0], cy=cen[:, 1],
        u=np.where(open_home, hu[:, j], np.nan), P=P[j], cells=int(grid.inside.sum()),
        unknowns=n_unk, t_solve=t_solve)) for j, (name, w) in enumerate(W.items())],
        ignore_index=True)
    fine_dir(h).mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, out)
    print(f"{time.strftime('%H:%M:%S')} {bid} h={h:g} n={len(polys)} tiers "
          f"{np.bincount(pl.tier, minlength=4).tolist()} {time.perf_counter() - t0:.0f}s",
          flush=True)


def _pool(procs: int, fn, jobs: list) -> None:
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        for _ in pool.starmap(_guard, [(fn, *j) for j in jobs], chunksize=1):
            pass


def _by_size(ids: list[str]) -> list[str]:
    """Largest first (the long poles start early)."""
    n = {b.block_id: len(b.buildings) for b in common.build_blocks(ids)}
    return sorted(ids, key=lambda i: -n[i])


# ------------------------------------------------------------------ large blocks (GPU)

def large_dir() -> Path:
    return ROWS / f"large_h{H:g}_ell{ELL_M:g}_K{K}"


def _large_path(bid: str, eps: float, what: str) -> Path:
    """A large block's rows at one eps: `what` is scenarios or homes."""
    return large_dir() / f"{bid}__eps{eps:g}__{what}.parquet"


def large_one(bid: str, p: lifted.Params, eps_list: tuple[float, ...]) -> None:
    """The eps worlds `eps_list` on one large block: the baseline (both populations, per home)
    and the stored clearings (area population: J_2 for Lens A). One file pair per eps (a run
    resumes). No leakage-only decomposition here: these blocks hold 0 -- 2 stranded homes."""
    todo = [e for e in eps_list if not _large_path(bid, e, "homes").exists()]
    if not todo:
        return
    t0 = time.perf_counter()
    [b] = common.build_blocks([bid])
    grid = mesh(H).build(b, p, None)
    polys = np.asarray(b.buildings.outlines)
    pl = Placement.of(grid, polys, p)
    W = {name: pop.weights(polys) for name, pop in common.POPULATIONS.items()}
    open_home = pl.tier == 0
    full = grid.isub.mean(axis=-1)
    lab = grid.label_sub(polys)
    clear = {}
    for cname, (rel, col, perm_col) in STORED.items():
        r = pd.read_parquet(HERE / rel / f"{bid}_p{CLEARING_POWER:g}.parquet").iloc[0]
        x = np.zeros(len(polys), dtype=bool)
        x[np.asarray(r[col], dtype=np.int64)] = True
        clear[cname] = (x, float(r[perm_col]))
    print(f"{time.strftime('%H:%M:%S')} {bid} n={len(polys)} cells={int(grid.inside.sum())} "
          f"tiers {np.bincount(pl.tier, minlength=4).tolist()} setup "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    cen = _centroids(polys)
    for eps in todo:
        inj = open_home if eps == 0.0 else np.ones(len(polys), dtype=bool)
        scen = []
        for cname in ("none", *clear):
            if cname == "none":
                op0 = grid.ff0
                pops = list(W)
            else:
                cleared = (lab >= 0) & clear[cname][0][np.maximum(lab, 0)]
                op0 = (grid.isub & (~grid.bsub | cleared)).mean(axis=-1)
                pops = ["area"]
            op = insulator_open(op0, full, eps)
            fields = [pl.field(W[name], inj, grid.inside.shape) for name in pops]
            ts = time.perf_counter()
            hu, P, n_unk = home_potentials(grid, op, fields, pl, p, rtol=RTOL_LARGE)
            t_solve = time.perf_counter() - ts
            for j, name in enumerate(pops):
                w, u = W[name], hu[:, j]
                scen.append(dict(block=bid, eps=eps, clearing=cname, pop=name,
                                 J2=float((w[inj] * u[inj] ** 2).sum()),
                                 P=float((w[inj] * u[inj]).sum()),
                                 J2_open=float((w[open_home] * u[open_home] ** 2).sum()),
                                 P_open=float((w[open_home] * u[open_home]).sum()),
                                 stored_perm=np.nan if cname == "none" else clear[cname][1],
                                 n_cleared=0 if cname == "none" else int(clear[cname][0].sum()),
                                 unknowns=n_unk, t_solve=t_solve))
                if cname == "none":
                    homes = pd.DataFrame(dict(
                        block=bid, eps=eps, pop=name, home=np.arange(len(polys)), tier=pl.tier,
                        barrier_m=pl.barrier_m, shadowed=pl.shadowed, w=w, cx=cen[:, 0],
                        cy=cen[:, 1], u=np.where(inj, u, np.nan)))
                    homes.to_parquet(large_dir() / f"{bid}__eps{eps:g}__{name}.tmp.parquet")
            print(f"  eps {eps:g} {cname}: {n_unk} unknowns, {t_solve:.0f}s", flush=True)
        pd.DataFrame(scen).to_parquet(_large_path(bid, eps, "scenarios"))
        parts = [large_dir() / f"{bid}__eps{eps:g}__{name}.tmp.parquet" for name in W]
        pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True).to_parquet(
            _large_path(bid, eps, "homes"))
        for f in parts:
            f.unlink()
    print(f"{time.strftime('%H:%M:%S')} {bid} {todo} done in {time.perf_counter() - t0:.0f}s",
          flush=True)


# The order the large run takes the eps grid in (no result depends on it): a coarse pass over
# every block first, so a run cut short still spans the grid
LARGE_PASSES = ((0.0, 1e-4, 1e-3, 1e-2, 1e-1), (3e-4, 3e-3, 3e-2))
assert sorted(e for ps in LARGE_PASSES for e in ps) == sorted(EPS_GRID)


def large(ids: list[str]) -> None:
    import gc
    p = params("gpu")
    large_dir().mkdir(parents=True, exist_ok=True)
    for eps_list in LARGE_PASSES:
        for bid in ids:
            try:
                large_one(bid, p, eps_list)
            except Exception as e:
                print(f"{time.strftime('%H:%M:%S')} {bid} FAILED {type(e).__name__}: {e}"[:400],
                      flush=True)
                traceback.print_exc()
            gc.collect()
            p.solver.release()


# ------------------------------------------------------------------ census (no solve)

def census_dir() -> Path:
    return ROWS / f"census_h{H:g}"


def census_one(bid: str) -> None:
    out = census_dir() / f"{bid}.parquet"
    if out.exists():
        return
    t0 = time.perf_counter()
    [b] = common.build_blocks([bid])
    p = params("cpu")
    grid = mesh(H).build(b, p, None)
    polys = np.asarray(b.buildings.outlines)
    pl = Placement.of(grid, polys, p)
    W = {name: pop.weights(polys) for name, pop in common.POPULATIONS.items()}
    st = pl.tier > 0
    row = dict(block=bid, n=len(polys), cells=int(grid.inside.sum()),
               area_km2=float(b.boundary.area / 1e6),
               **{f"n_tier{t}": int((pl.tier == t).sum()) for t in range(1, 6)},
               n_shadowed=int((st & pl.shadowed).sum()),
               barrier_med_m=float(np.median(pl.barrier_m[st])) if st.any() else np.nan,
               barrier_max_m=float(pl.barrier_m[st].max()) if st.any() else np.nan,
               t=time.perf_counter() - t0)
    for name, w in W.items():
        row[f"stranded_{name}"] = float(w[st].sum() / w.sum())
    census_dir().mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(out)
    print(f"{time.strftime('%H:%M:%S')} {bid} n={len(polys)} stranded {int(st.sum())} "
          f"({row['stranded_count']:.4f}) {row['t']:.0f}s", flush=True)


# ------------------------------------------------------------------ the door (an alternative)
# The eps world leaks everywhere. A door leaks only where the open graph does not reach: each
# walled-in pocket's cheapest crossing (_crossing_graph's path from street-connected open space
# to the stranded cell nearest it) is opened, every other footprint stays closed. The door hangs
# off the open network, so an open-access home's u moves only by the demand the stranded homes
# add. The kinds (DoorKind):
#   metre  every crossing cell conducts door_eps (a cost per metre of footprint crossed);
#          door_eps 1: an open corridor one cell wide through the wall;
#   leaf   the crossing is an open passage but for one cell per wall it passes (the most closed
#          cell of each run of footprint-bearing cells), which conducts door_eps (a cost per
#          wall crossed);
#   straight  not the cheapest path but the thinnest straight wall: from the pocket cell nearest
#          (Euclidean) to street-connected open space, a STRAIGHT_HALF_M capsule to that open
#          cell, its footprint conducting door_eps (per metre, as `metre`). The cheapest path
#          counts footprint metres only, so it winds along sealed gaps (77 cells on 38486), and
#          a one-cell staircase pays a 90-degree turn per step at ell 3 m;
#   local  no door: the eps world confined to each walled-in pocket's surroundings, footprint
#          within (the pocket's largest barrier_m + LOCAL_MARGIN_M) of its cells conducting
#          door_eps; the pocket drains through all its walls, as in the eps world, and nothing
#          far from a pocket leaks.
DOOR_EPS = (1e-2, 3e-2, 1e-1, 3e-1, 1.0)     # metre, leaf, straight: a door's conductance
LOCAL_EPS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)   # local: the walls' (the eps world's range)
STRAIGHT_HALF_M = 0.5    # the straight door's half-width: a 1 m doorway
LOCAL_MARGIN_M = 1.0     # local: how far past the pocket's barrier its walls conduct


class Door(NamedTuple):
    cells: np.ndarray     # (ny, nx) bool: the door's cells, and the closed cells tiers 4-5 use
    leaves: np.ndarray    # (ny, nx) bool: per crossing, the most closed cell of each wall


class DoorKind(Protocol):
    """How walled-in pockets are let out (resolved once, from the command line)."""
    name: str
    eps_grid: tuple[float, ...]

    def build(self, grid: lifted.Grid, reach: np.ndarray, pl: Placement) -> Door: ...

    def open(self, base: np.ndarray, full: np.ndarray, door: Door,
             door_eps: float) -> np.ndarray: ...


def _conduct(base: np.ndarray, full: np.ndarray, door: Door, door_eps: float) -> np.ndarray:
    """Every door cell's footprint conducts door_eps."""
    return base + door_eps * (full - base) * door.cells


@dataclass(frozen=True)
class MetreDoor:
    name: str = "metre"
    eps_grid: tuple[float, ...] = DOOR_EPS

    def build(self, grid, reach, pl):
        return door_of(grid, reach, pl)

    def open(self, base, full, door, door_eps):
        return _conduct(base, full, door, door_eps)


@dataclass(frozen=True)
class LeafDoor:
    name: str = "leaf"
    eps_grid: tuple[float, ...] = DOOR_EPS

    def build(self, grid, reach, pl):
        return door_of(grid, reach, pl)

    def open(self, base, full, door, door_eps):
        op = base + (full - base) * door.cells
        return np.where(door.leaves, base + door_eps * (full - base), op)


@dataclass(frozen=True)
class StraightDoor:
    name: str = "straight"
    eps_grid: tuple[float, ...] = DOOR_EPS

    def build(self, grid, reach, pl):
        return straight_door_of(grid, reach, pl)

    def open(self, base, full, door, door_eps):
        return _conduct(base, full, door, door_eps)


@dataclass(frozen=True)
class LocalWalls:
    name: str = "local"
    eps_grid: tuple[float, ...] = LOCAL_EPS

    def build(self, grid, reach, pl):
        return local_door_of(grid, reach, pl)

    def open(self, base, full, door, door_eps):
        return _conduct(base, full, door, door_eps)


DOOR_KINDS: tuple[DoorKind, ...] = (MetreDoor(), LeafDoor(), StraightDoor(), LocalWalls())


def door_kind(name: str) -> DoorKind:
    """The kind a command line names (the one place a kind is a string)."""
    for k in DOOR_KINDS:
        if k.name == name:
            return k
    raise SystemExit(f"unknown door kind {name!r}: {[k.name for k in DOOR_KINDS]}")


def door_dir(kind: DoorKind) -> Path:
    return ROWS / f"door_{kind.name}_h{H:g}_ell{ELL_M:g}_K{K}"


def door_of(grid: lifted.Grid, reach: np.ndarray, pl: Placement) -> Door:
    """The door. The closed cells a formerly stranded home injects on (tiers 4-5, inside
    footprints) are door cells; then, while any cell a home injects on has no path to the street
    through open or door cells, the cheapest crossing from the one nearest open space is added
    (one per sealed pocket a home touches)."""
    from scipy import ndimage
    shape = grid.inside.shape
    _d, pred, _s = dijkstra(_crossing_graph(grid), indices=np.flatnonzero(reach.ravel()),
                            min_only=True, return_predecessors=True)
    o = grid.ff0.ravel()
    full = grid.isub.mean(axis=-1).ravel()
    want = np.asarray(pl.S[:, pl.tier > 0].sum(axis=1)).ravel() > 0
    mask = want & (o == 0)
    leaves = np.zeros_like(mask)
    ground = grid.ground.ravel()
    while True:
        lab, n = ndimage.label(((o > 0) | mask).reshape(shape))
        lab = lab.ravel()
        ok = np.zeros(n + 1, dtype=bool)
        ok[lab[ground & ((o > 0) | mask)]] = True
        ok[0] = False
        bad = np.flatnonzero(want & ~ok[lab])
        if len(bad) == 0:
            return Door(cells=mask.reshape(shape), leaves=leaves.reshape(shape))
        c = int(bad[np.argmin(_d[bad])])
        assert np.isfinite(_d[c]), "a stranded cell no crossing reaches"
        path = []
        while c >= 0:
            path.append(c)
            c = int(pred[c])
        path = np.asarray(path)
        mask[path] = True
        # leaf: per run of footprint-bearing cells along the crossing, its most closed cell (a
        # crossing passes at least one fully closed cell: the pocket is sealed)
        wall = o[path] < full[path]
        run = np.cumsum(wall & ~np.r_[False, wall[:-1]])
        for r_ in np.unique(run[wall]):
            at = path[wall & (run == r_)]
            leaves[at[np.argmin(o[at] / full[at])]] = True


def straight_door_of(grid: lifted.Grid, reach: np.ndarray, pl: Placement) -> Door:
    """The straight door: the closed cells tiers 4-5 inject on, then, while a cell a formerly
    stranded home injects on has no path to the street, the capsule (half-width
    STRAIGHT_HALF_M, inside cells only) from the one nearest street-connected open space
    (Euclidean, between cell centres) to that open cell. A capsule that connects nothing (it
    left the block) falls back to door_of's crossing for that cell."""
    from scipy import ndimage
    shape = grid.inside.shape
    o = grid.ff0.ravel()
    want = np.asarray(pl.S[:, pl.tier > 0].sum(axis=1)).ravel() > 0
    mask = want & (o == 0)
    ground = grid.ground.ravel()
    inside = grid.inside.ravel()
    dist, (iy, ix) = ndimage.distance_transform_edt(~reach, return_indices=True)
    dist, iy, ix = dist.ravel(), iy.ravel(), ix.ravel()
    xy = grid.xy.reshape(-1, 2)
    fallback = None
    while True:
        lab, n = ndimage.label(((o > 0) | mask).reshape(shape))
        lab = lab.ravel()
        ok = np.zeros(n + 1, dtype=bool)
        ok[lab[ground & ((o > 0) | mask)]] = True
        ok[0] = False
        bad = np.flatnonzero(want & ~ok[lab])
        if len(bad) == 0:
            return Door(cells=mask.reshape(shape), leaves=np.zeros(shape, dtype=bool))
        c = int(bad[np.argmin(dist[bad])])
        t = int(iy[c]) * shape[1] + int(ix[c])
        seg = shapely.LineString([xy[c], xy[t]])
        near = np.flatnonzero(inside & shapely.dwithin(
            seg, shapely.points(xy[:, 0], xy[:, 1]), STRAIGHT_HALF_M + 1e-9))
        kept = mask.copy()
        mask[near] = True
        lab2, n2 = ndimage.label(((o > 0) | mask).reshape(shape))
        lab2 = lab2.ravel()
        ok2 = np.zeros(n2 + 1, dtype=bool)
        ok2[lab2[ground & ((o > 0) | mask)]] = True
        ok2[0] = False
        if not ok2[lab2[c]]:                    # the capsule connects nothing: the crossing
            if fallback is None:
                fallback = door_of(grid, reach, pl)
            mask = kept | fallback.cells.ravel()
            assert int(mask.sum()) > int(kept.sum()), "no door reaches a stranded cell"


def local_door_of(grid: lifted.Grid, reach: np.ndarray, pl: Placement) -> Door:
    """local: per walled-in pocket (a connected component of the cells formerly stranded homes
    inject on, through open or those closed cells), every cell within its largest barrier_m +
    LOCAL_MARGIN_M (Euclidean, from its cells); then door_of's crossing for any injected cell
    that still has no path to the street."""
    from scipy import ndimage
    shape = grid.inside.shape
    o = grid.ff0.ravel()
    Sst = pl.S[:, pl.tier > 0].tocsc()
    want = np.asarray(Sst.sum(axis=1)).ravel() > 0
    mask = want & (o == 0)
    lab, n = ndimage.label((((o > 0) & ~reach.ravel()) | mask).reshape(shape))
    lab = lab.ravel()
    homes = np.flatnonzero(pl.tier > 0)
    reach_of = np.zeros(n + 1)                  # per pocket: its largest barrier
    for k, j in enumerate(homes):
        rows = Sst.indices[Sst.indptr[k]:Sst.indptr[k + 1]]
        for c in np.unique(lab[rows]):
            reach_of[c] = max(reach_of[c], pl.barrier_m[j])
    reach_of[0] = 0.0                            # label 0: not a pocket
    inside = grid.inside.ravel()
    for c in np.flatnonzero(reach_of > 0):
        d = ndimage.distance_transform_edt((lab != c).reshape(shape)).ravel() * grid.h
        mask |= inside & (d <= reach_of[c] + LOCAL_MARGIN_M)
    ground = grid.ground.ravel()
    lab2, n2 = ndimage.label(((o > 0) | mask).reshape(shape))
    lab2 = lab2.ravel()
    ok = np.zeros(n2 + 1, dtype=bool)
    ok[lab2[ground & ((o > 0) | mask)]] = True
    ok[0] = False
    if (want & ~ok[lab2]).any():
        mask |= door_of(grid, reach, pl).cells.ravel()
    return Door(cells=mask.reshape(shape), leaves=np.zeros(shape, dtype=bool))


def door_one(bid: str, kind: DoorKind) -> None:
    out = door_dir(kind) / f"{bid}.parquet"
    if out.exists():
        return
    t0 = time.perf_counter()
    [b] = common.build_blocks([bid])
    p = params("cpu")
    grid = mesh(H).build(b, p, None)
    polys = np.asarray(b.buildings.outlines)
    pl = Placement.of(grid, polys, p)
    W = {name: pop.weights(polys) for name, pop in common.POPULATIONS.items()}
    open_home = pl.tier == 0
    if open_home.all():
        return
    reach = lifted.grounded(grid, grid.ff0, p)
    door = kind.build(grid, reach, pl)
    full = grid.isub.mean(axis=-1)
    cen = _centroids(polys)
    rows = []
    for de in kind.eps_grid:
        op = kind.open(grid.ff0, full, door, de)
        fields, keys = [], []
        for name, w in W.items():
            fields += [pl.field(w, open_home, grid.inside.shape),
                       pl.field(w, ~open_home, grid.inside.shape)]
            keys += [(name, "open"), (name, "str")]
        ts = time.perf_counter()
        hu, P, n_unk = home_potentials(grid, op, fields, pl, p, rtol=RTOL)
        t_solve = time.perf_counter() - ts
        for name, w in W.items():
            u_open = hu[:, keys.index((name, "open"))]
            u = u_open + hu[:, keys.index((name, "str"))]
            rows.append(pd.DataFrame(dict(
                block=bid, kind=kind.name, door_eps=de, pop=name, home=np.arange(len(polys)),
                tier=pl.tier, barrier_m=pl.barrier_m, shadowed=pl.shadowed, w=w,
                cx=cen[:, 0], cy=cen[:, 1], u=u, u_open=np.where(open_home, u_open, np.nan),
                door_cells=int(door.cells.sum()), door_leaves=int(door.leaves.sum()),
                unknowns=n_unk, t_solve=t_solve)))
    door_dir(kind).mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    pd.concat(rows, ignore_index=True).to_parquet(tmp)
    os.replace(tmp, out)
    print(f"{time.strftime('%H:%M:%S')} {bid} door {kind.name}: {int(door.cells.sum())} cells, "
          f"{int(door.leaves.sum())} leaves, {time.perf_counter() - t0:.0f}s", flush=True)


def large_door(ids: list[str], kind: DoorKind) -> None:
    """The door on large blocks (GPU): per door_eps, the baseline and the stored clearings, area
    population (J_2, Lens A). A block with no stranded home is today's world exactly (skipped)."""
    import gc
    p = params("gpu")
    door_dir(kind).mkdir(parents=True, exist_ok=True)
    for bid in ids:
        out = door_dir(kind) / f"{bid}__large.parquet"
        if out.exists():
            continue
        t0 = time.perf_counter()
        [b] = common.build_blocks([bid])
        grid = mesh(H).build(b, p, None)
        polys = np.asarray(b.buildings.outlines)
        pl = Placement.of(grid, polys, p)
        if (pl.tier == 0).all():
            print(f"{bid}: no stranded home, the door world is today's", flush=True)
            continue
        w = common.POPULATIONS["area"].weights(polys)
        reach = lifted.grounded(grid, grid.ff0, p)
        door = kind.build(grid, reach, pl)
        full = grid.isub.mean(axis=-1)
        lab = grid.label_sub(polys)
        rows = []
        for cname in ("none", *STORED):
            if cname == "none":
                op0, n_cl = grid.ff0, 0
            else:
                rel, col, _perm = STORED[cname]
                r = pd.read_parquet(HERE / rel / f"{bid}_p{CLEARING_POWER:g}.parquet").iloc[0]
                x = np.zeros(len(polys), dtype=bool)
                x[np.asarray(r[col], dtype=np.int64)] = True
                cleared = (lab >= 0) & x[np.maximum(lab, 0)]
                op0, n_cl = (grid.isub & (~grid.bsub | cleared)).mean(axis=-1), int(x.sum())
            for de in kind.eps_grid:
                op = kind.open(op0, full, door, de)
                f = pl.field(w, np.ones(len(polys), dtype=bool), grid.inside.shape)
                hu, P, n_unk = home_potentials(grid, op, [f], pl, p, rtol=RTOL_LARGE)
                u = hu[:, 0]
                rows.append(dict(block=bid, kind=kind.name, door_eps=de, clearing=cname,
                                 n_cleared=n_cl,
                                 J2=float((w * u ** 2).sum()),
                                 J2_open=float((w[pl.tier == 0] * u[pl.tier == 0] ** 2).sum()),
                                 u_stranded=",".join(f"{v:.4g}" for v in u[pl.tier > 0]),
                                 unknowns=n_unk))
                print(f"  {bid} door {de:g} {cname}: {n_unk} unknowns", flush=True)
        pd.DataFrame(rows).to_parquet(out)
        print(f"{time.strftime('%H:%M:%S')} {bid} door done in {time.perf_counter() - t0:.0f}s",
              flush=True)
        del grid
        gc.collect()
        p.solver.release()


# ------------------------------------------------------------------ report

def _read(d: Path, pattern: str) -> pd.DataFrame:
    fs = sorted(d.glob(pattern))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def neighbours(homes: pd.DataFrame) -> dict[int, np.ndarray]:
    """One block's formerly stranded homes -> N(i), their NEIGHBOURS nearest tier-0 homes by
    centroid distance (`homes`: one row per home, with home, tier, cx, cy)."""
    op = homes[homes.tier == 0]
    tree = cKDTree(op[["cx", "cy"]].to_numpy())
    out = {}
    for r in homes[homes.tier > 0].itertuples():
        _d, k = tree.query([r.cx, r.cy], k=NEIGHBOURS)
        out[int(r.home)] = op.home.to_numpy()[np.atleast_1d(k)]
    return out


def ratio_rows(small: pd.DataFrame, worlds: pd.DataFrame, key: str) -> pd.DataFrame:
    """(a) Per formerly stranded home, population and world (`key`: eps, or door_eps):
    r = u_i / max u_j over N(i), and u_i over the block's worst tier-0 home (r_block). Tiers,
    centroids and N(i) from the eps-0 rows of `small`; u from `worlds`."""
    rows = []
    base_all = small[small.eps == 0.0]
    for (bid, pop), g in worlds.groupby(["block", "pop"]):
        base = base_all[(base_all.block == bid) & (base_all["pop"] == pop)].set_index("home")
        nb = neighbours(base.reset_index())
        if not nb:
            continue
        for val, ge in g.groupby(key):
            u = ge.set_index("home").u
            worst = u[base.tier == 0].max()
            for i, js in nb.items():
                rows.append({"block": bid, "pop": pop, key: val, "home": i,
                             "tier": int(base.tier[i]), "shadowed": bool(base.shadowed[i]),
                             "barrier_m": float(base.barrier_m[i]),
                             "u": float(u[i]), "u_nbr_max": float(u[js].max()),
                             "r": float(u[i] / u[js].max()), "r_block": float(u[i] / worst)})
    return pd.DataFrame(rows)


def _q(x: pd.Series, q: float) -> float:
    return float(np.quantile(x.to_numpy(), q)) if len(x) else np.nan


def table_a(rat: pd.DataFrame, key: str) -> pd.DataFrame:
    """Per population, world and class: r over the walled-in formerly stranded homes (tier 2,
    whose door is on the open side and whose u does not depend on the world, is left out).
    Classes: `own ring` (tier 1, sealed in its own ring), `in footprint` (tiers 3-5: no free
    cell of its own, footprint all round), `walled` (both)."""
    out = []
    cls = {"own ring": rat.tier == 1, "in footprint": rat.tier >= 3,
           "walled": rat.tier != 2}
    for name, mask in cls.items():
        for (pop, val), g in rat[mask].groupby(["pop", key]):
            out.append({"pop": pop, key: val, "class": name} | dict(
                homes=len(g), blocks=g.block.nunique(), r_med=_q(g.r, 0.5),
                r_p90=_q(g.r, 0.9), r_max=float(g.r.max()),
                share_r_le_1=float((g.r <= 1.0).mean()),
                share_r_gt_3=float((g.r > 3.0).mean()),
                r_block_med=_q(g.r_block, 0.5)))
    return pd.DataFrame(out).sort_values(["class", "pop", key]).reset_index(drop=True)


def fine_rows(small: pd.DataFrame, h: float) -> pd.DataFrame:
    """(b) The formerly stranded homes at a finer h (eps 0): open there or not, and, if open,
    r_fine = u_i / max u_j over the same N(i) (dropping any neighbour not open at h)."""
    fine = _read(fine_dir(h), "*.parquet")
    if fine.empty:
        return pd.DataFrame()
    rows = []
    for (bid, pop), g in small[small.eps == 0.0].groupby(["block", "pop"]):
        f = fine[(fine.block == bid) & (fine["pop"] == pop)].set_index("home")
        if f.empty:
            continue
        for i, js in neighbours(g).items():
            js_ok = [j for j in js if f.tier[j] == 0]
            open_fine = bool(f.tier[i] == 0)
            rows.append(dict(block=bid, pop=pop, h=h, home=i, tier=int(g.set_index("home").tier[i]),
                             open_fine=open_fine,
                             nbrs_kept=len(js_ok),
                             r_fine=(float(f.u[i] / f.u[js_ok].max()) if open_fine and js_ok
                                     else np.nan),
                             barrier_m=float(g.set_index("home").barrier_m[i]),
                             barrier_fine_m=float(f.barrier_m[i])))
    return pd.DataFrame(rows)


def eps_star(r_of: np.ndarray, vals: np.ndarray, target: float) -> float:
    """The parameter at which r(param) (falling in param) equals `target`, log-log interpolated
    along the grid `vals` (0 below its smallest value, inf above its largest)."""
    lr, t, lv = np.log(r_of), np.log(target), np.log(vals)
    if t > lr[0]:
        return 0.0
    if t < lr[-1]:
        return np.inf
    for k in range(len(vals) - 1):
        a, b = lr[k], lr[k + 1]
        if (a - t) * (b - t) <= 0:
            s_ = 0.0 if a == b else (t - a) / (b - a)
            return float(np.exp(lv[k] + s_ * (lv[k + 1] - lv[k])))
    return np.nan


def table_b(fr: pd.DataFrame, rat: pd.DataFrame, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per h and population, over the walled-in formerly stranded homes (tier 2 left out, as in
    table_a): how many open at h, their r_fine, and the parameter (`key`) at which each one's r
    matches it; per parameter value the log ratio log(r / r_fine) over the homes with an
    r_fine: median, quartiles, p10, p90."""
    piv = rat.pivot_table(index=["block", "pop", "home"], columns=key, values="r")
    vals = np.array(sorted(piv.columns))
    opened, match = [], []
    for (h, pop), g in fr[fr.tier != 2].groupby(["h", "pop"]):
        ok = g[g.open_fine & g.r_fine.notna()].set_index(["block", "pop", "home"])
        ok = ok[ok.index.isin(piv.index)]
        R = piv.loc[ok.index, vals].to_numpy() if len(ok) else np.zeros((0, len(vals)))
        es = pd.Series([eps_star(R[k], vals, t) for k, t in enumerate(ok.r_fine)], dtype=float)
        opened.append(dict(h=h, pop=pop, stranded_at_0_5=len(g),
                           open_at_h=int(g.open_fine.sum()), with_r_fine=len(ok),
                           blocks=ok.index.get_level_values(0).nunique(),
                           r_fine_med=_q(ok.r_fine, 0.5), r_fine_min=float(ok.r_fine.min()),
                           r_fine_max=float(ok.r_fine.max()),
                           star_p10=_q(es, 0.1), star_med=_q(es, 0.5), star_p90=_q(es, 0.9)))
        for k, v in enumerate(vals):
            lr = pd.Series(np.log(R[:, k] / ok.r_fine.to_numpy()))
            match.append({"h": h, "pop": pop, key: v, "homes": len(ok),
                          "logratio_med": _q(lr, 0.5), "logratio_p25": _q(lr, 0.25),
                          "logratio_p75": _q(lr, 0.75), "logratio_p10": _q(lr, 0.1),
                          "logratio_p90": _q(lr, 0.9), "abs_logratio_med": _q(lr.abs(), 0.5)})
    return pd.DataFrame(opened), pd.DataFrame(match)


def table_c(small: pd.DataFrame, controls: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(c) Tier-0 homes: u(eps)/u(0) pooled (the full insulator world, and u_open/u(0), the
    leakage alone); per block P and J_2 over the same homes against eps 0."""
    t0 = small[small.tier == 0]
    base = t0[t0.eps == 0.0].set_index(["block", "pop", "home"])
    homes, blocks = [], []
    for (pop, eps), g in t0[t0.eps > 0].groupby(["pop", "eps"]):
        g = g.set_index(["block", "pop", "home"])
        u0 = base.u.reindex(g.index)
        ok = u0 > 0
        ratio = (g.u / u0)[ok]
        leak = (g.u_open / u0)[ok]
        for subset, mask in (("all41", np.ones(len(ratio), dtype=bool)),
                             ("controls", ratio.index.get_level_values(0).isin(controls)),
                             ("stranded31", ~ratio.index.get_level_values(0).isin(controls))):
            rr, ll = ratio[mask], leak[mask]
            homes.append(dict(pop=pop, eps=eps, subset=subset, homes=len(rr),
                              ratio_med=_q(rr, 0.5), ratio_p5=_q(rr, 0.05),
                              ratio_p95=_q(rr, 0.95), ratio_min=float(rr.min()),
                              leak_med=_q(ll, 0.5), leak_p5=_q(ll, 0.05),
                              leak_min=float(ll.min())))
        for bid in g.index.get_level_values(0).unique():
            gb, bb = g.loc[bid], base.loc[bid]
            bb = bb.loc[gb.index]
            P0, J0 = float((bb.w * bb.u).sum()), float((bb.w * bb.u ** 2).sum())
            blocks.append(dict(pop=pop, eps=eps, block=bid, control=bid in controls,
                               P_ratio=float((gb.w * gb.u).sum()) / P0,
                               J2_ratio=float((gb.w * gb.u ** 2).sum()) / J0,
                               P_leak_ratio=float((gb.w * gb.u_open).sum()) / P0,
                               J2_leak_ratio=float((gb.w * gb.u_open ** 2).sum()) / J0))
    bl = pd.DataFrame(blocks)
    summ = []
    for (pop, eps), g in bl.groupby(["pop", "eps"]):
        for subset, gg in (("all41", g), ("controls", g[g.control]), ("stranded31", g[~g.control])):
            summ.append(dict(pop=pop, eps=eps, subset=subset, blocks=len(gg),
                             P_ratio_med=_q(gg.P_ratio, 0.5), P_ratio_min=float(gg.P_ratio.min()),
                             J2_ratio_med=_q(gg.J2_ratio, 0.5),
                             J2_ratio_min=float(gg.J2_ratio.min()),
                             shift_P_med=1 - _q(gg.P_ratio, 0.5),
                             shift_J2_med=1 - _q(gg.J2_ratio, 0.5) ** 0.5,
                             shift_J2_max=1 - float(gg.J2_ratio.min()) ** 0.5))
    return pd.DataFrame(homes), pd.DataFrame(summ)


SLIVER_BINS = (0.0, 2.0, 4.0, 8.0, np.inf)    # z = u / block median at h 0.5, eps 0


def table_sliver(small: pd.DataFrame, h: float) -> pd.DataFrame:
    """(c) against the finer raster: tier-0 homes (open at h 0.5 and at h) by z = u / the
    block's median u at h 0.5, eps 0 (z >> 1: reached only through a sliver). Per bin: the
    median of z_h / z (does the finer raster keep the long escape?) and, per eps, of
    z(eps) / z_h (how much the eps world shortens it against the finer raster)."""
    fine = _read(fine_dir(h), "*.parquet")
    out = []
    for pop in ("count", "area"):
        s_ = small[(small["pop"] == pop)]
        f = fine[fine["pop"] == pop][["block", "home", "tier", "u"]].rename(
            columns={"u": "u_h", "tier": "tier_h"})
        m = s_[s_.eps == 0.0][["block", "home", "tier", "u"]].merge(f, on=["block", "home"])
        m = m[(m.tier == 0) & (m.tier_h == 0)].copy()
        m["z"] = m.u / m.groupby("block").u.transform("median")
        m["z_h"] = m.u_h / m.groupby("block").u_h.transform("median")
        for e in (e for e in EPS_GRID if e > 0):
            ue = s_[s_.eps == e][["block", "home", "u"]].rename(columns={"u": "u_e"})
            m = m.merge(ue, on=["block", "home"])
            m[f"z@{e:g}"] = m.u_e / m.groupby("block").u_e.transform("median")
            m = m.drop(columns=["u_e"])
        for lo, hi in zip(SLIVER_BINS[:-1], SLIVER_BINS[1:], strict=True):
            g = m[(m.z >= lo) & (m.z < hi)]
            row = dict(pop=pop, h=h, z_from=lo, z_to=hi, homes=len(g),
                       blocks=g.block.nunique(), zh_over_z=_q(g.z_h / g.z, 0.5))
            for e in (e for e in EPS_GRID if e > 0):
                row[f"zeps_over_zh@{e:g}"] = _q(g[f"z@{e:g}"] / g.z_h, 0.5)
            out.append(row)
    return pd.DataFrame(out)


def table_d(sc: pd.DataFrame) -> pd.DataFrame:
    """(d) Per large block and eps: Lens A (area, J_2) of each stored clearing, against the same
    eps world with nothing cleared (over every home, and over the tier-0 homes alone: the
    formerly stranded homes' J_2 is a floor no stored clearing touches); and the baseline shift,
    1 - (J_2 of the tier-0 homes at eps / J_2 at eps 0)^(1/2) (area) and 1 - P/P0 (count)."""
    out = []
    for (bid, eps), g in sc.groupby(["block", "eps"]):
        a = g[g["pop"] == "area"].set_index("clearing")
        c = g[g["pop"] == "count"].set_index("clearing")
        a0 = sc[(sc.block == bid) & (sc.eps == 0.0) & (sc["pop"] == "area")].set_index("clearing")
        c0 = sc[(sc.block == bid) & (sc.eps == 0.0) & (sc["pop"] == "count")].set_index("clearing")
        row = dict(block=bid, eps=eps,
                   shift_area_J2=1 - (a.J2_open["none"] / a0.J2["none"]) ** 0.5,
                   shift_count_P=1 - c.P_open["none"] / c0.P["none"])
        for cname in STORED:
            if cname in a.index:
                row[f"lensA_{cname}"] = 1 - (a.J2[cname] / a.J2["none"]) ** 0.5
                row[f"lensA_open_{cname}"] = 1 - (a.J2_open[cname] / a.J2_open["none"]) ** 0.5
                row[f"stored_{cname}"] = a.stored_perm[cname]
        out.append(row)
    return pd.DataFrame(out)


def _md(df: pd.DataFrame, fmt: dict[str, str]) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, r, strict=True):
            f = fmt.get(c)
            cells.append(f.format(v) if f and isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def report() -> None:
    """Every table of insulator_report.md, printed as markdown, and the CSV summaries."""
    small = _read(small_dir(), "ZAF*[0-9].parquet")
    controls = set((ROWS / "control_ids.txt").read_text().split(","))
    f3 = lambda cols: {c: "{:.3g}" for c in cols}           # noqa: E731
    if not small.empty:
        rat = ratio_rows(small, small[small.eps > 0], "eps")
        rat.to_csv(ROWS / "stranded_homes.csv", index=False)
        ta = table_a(rat, "eps")
        ta.to_csv(ROWS / "summary_a_ratio.csv", index=False)
        print("## (a) eps world: r = u_i / max u_j over N(i)\n")
        print(_md(ta, f3(ta.columns)))
        frs = [fr for h in (0.35, 0.25) if not (fr := fine_rows(small, h)).empty]
        fr = pd.concat(frs, ignore_index=True) if frs else pd.DataFrame()
        if not fr.empty:
            fr.to_csv(ROWS / "fine_homes.csv", index=False)
            tb_open, tb_match = table_b(fr, rat, "eps")
            tb_open.to_csv(ROWS / "summary_b_open.csv", index=False)
            tb_match.to_csv(ROWS / "summary_b_match.csv", index=False)
            print("\n## (b) eps world against the finer raster\n")
            print(_md(tb_open, f3(tb_open.columns)))
            print()
            print(_md(tb_match, f3(tb_match.columns)))
        ts_ = pd.concat([table_sliver(small, h) for h in (0.35, 0.25)], ignore_index=True)
        ts_.to_csv(ROWS / "summary_c_sliver.csv", index=False)
        print("\n## (c) slivered homes against the finer raster\n")
        print(_md(ts_, f3(ts_.columns)))
        th, tbl = table_c(small, controls)
        th.to_csv(ROWS / "summary_c_homes.csv", index=False)
        tbl.to_csv(ROWS / "summary_c_blocks.csv", index=False)
        print("\n## (c) eps world: leakage, tier-0 homes\n")
        print(_md(th, f3(th.columns)))
        print()
        print(_md(tbl, f3(tbl.columns)))
        for kind in DOOR_KINDS:
            door = _read(door_dir(kind), "ZAF*[0-9].parquet")
            if door.empty:
                continue
            drat = ratio_rows(small, door, "door_eps")
            drat.to_csv(ROWS / f"door_{kind.name}_stranded_homes.csv", index=False)
            dta = table_a(drat, "door_eps")
            dta.to_csv(ROWS / f"summary_door_{kind.name}_a.csv", index=False)
            print(f"\n## the door ({kind.name}): (a)\n")
            print(_md(dta, f3(dta.columns)))
            if not fr.empty:
                dbo, dbm = table_b(fr, drat, "door_eps")
                dbo.to_csv(ROWS / f"summary_door_{kind.name}_b_open.csv", index=False)
                dbm.to_csv(ROWS / f"summary_door_{kind.name}_b_match.csv", index=False)
                print(f"\n## the door ({kind.name}): (b)\n")
                print(_md(dbo, f3(dbo.columns)))
                print()
                print(_md(dbm, f3(dbm.columns)))
            dl = door_leak(small, door)
            dl.to_csv(ROWS / f"summary_door_{kind.name}_c.csv", index=False)
            print(f"\n## the door ({kind.name}): (c)\n")
            print(_md(dl, f3(dl.columns)))
            big = _read(door_dir(kind), "*__large.parquet")
            if not big.empty:
                dd = door_large(big)
                dd.to_csv(ROWS / f"summary_door_{kind.name}_d.csv", index=False)
                print(f"\n## the door ({kind.name}): (d)\n")
                print(_md(dd, {c: "{:.4f}" for c in dd.columns if c not in ("block",)}
                          | {"door_eps": "{:.3g}"}))
    sc = _read(large_dir(), "*__eps*__scenarios.parquet")
    if not sc.empty:
        td = table_d(sc)
        td.to_csv(ROWS / "summary_d_large.csv", index=False)
        print("\n## (d) large blocks\n")
        print(_md(td, {c: "{:.4f}" for c in td.columns if c not in ("eps", "block")}
                  | {"eps": "{:.3g}"}))
    cen = _read(census_dir(), "*.parquet")
    if not cen.empty:
        cen = cen.sort_values("stranded_count", ascending=False)
        cen.to_csv(ROWS / "summary_census.csv", index=False)
        print("\n## census\n")
        print(_md(cen.drop(columns=["t"]), {c: "{:.4g}" for c in cen.columns}))


def door_large(big: pd.DataFrame) -> pd.DataFrame:
    """The door's (d): Lens A (area, J_2) of the stored clearings in the door world, over every
    home and over the tier-0 homes alone, and the tier-0 J_2 against today's (eps 0)."""
    out = []
    for (bid, de), g in big.groupby(["block", "door_eps"]):
        a = g.set_index("clearing")
        sc0 = pd.read_parquet(_large_path(bid, 0.0, "scenarios"))
        j0 = float(sc0[(sc0["pop"] == "area") & (sc0.clearing == "none")].J2.iloc[0])
        row = dict(block=bid, door_eps=de, shift_area_J2=1 - (a.J2_open["none"] / j0) ** 0.5)
        for cname in STORED:
            row[f"lensA_{cname}"] = 1 - (a.J2[cname] / a.J2["none"]) ** 0.5
            row[f"lensA_open_{cname}"] = 1 - (a.J2_open[cname] / a.J2_open["none"]) ** 0.5
        out.append(row)
    return pd.DataFrame(out)


def door_leak(small: pd.DataFrame, door: pd.DataFrame) -> pd.DataFrame:
    """The door's (c): tier-0 homes' u / u(eps 0), with the stranded demand (u) and without
    (u_open, which a door that hangs off the open network leaves exactly as it was)."""
    base = small[(small.eps == 0.0) & (small.tier == 0)].set_index(["block", "pop", "home"]).u
    out = []
    for (pop, de), g in door[door.tier == 0].groupby(["pop", "door_eps"]):
        g = g.set_index(["block", "pop", "home"])
        u0 = base.reindex(g.index)
        ok = u0 > 0
        r, lk = (g.u / u0)[ok], (g.u_open / u0)[ok]
        out.append(dict(pop=pop, door_eps=de, homes=len(r), ratio_med=_q(r, 0.5),
                        ratio_p95=_q(r, 0.95), ratio_max=float(r.max()), ratio_min=float(r.min()),
                        leak_min=float(lk.min()), leak_max=float(lk.max())))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ main

def main(argv: list[str]) -> None:
    cmd = argv[0]
    _check_single_threaded()
    if cmd == "small":
        ids = _by_size(argv[2].split(","))
        _pool(int(argv[1]), small_one, [(i,) for i in ids])
    elif cmd == "fine":
        h = float(argv[2])
        ids = _by_size(argv[3].split(","))
        _pool(int(argv[1]), fine_one, [(i, h) for i in ids])
    elif cmd == "large":
        large(argv[1].split(","))
    elif cmd == "census":
        ids = _by_size(argv[2].split(","))
        _pool(int(argv[1]), census_one, [(i,) for i in ids])
    elif cmd == "large-door":
        large_door(argv[2].split(","), door_kind(argv[1]))
    elif cmd == "door":
        kind = door_kind(argv[2])
        ids = _by_size(argv[3].split(","))
        _pool(int(argv[1]), door_one, [(i, kind) for i in ids])
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
