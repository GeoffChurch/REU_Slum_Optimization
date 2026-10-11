"""Uniform walls: every footprint an imperfect insulator for everyone, one number against two.

The owner (2026-10-11): "Actually, I think I am fine with having fully enclosed buildings be much
worse off than exposed neighbors, so we can try to find a uniform resistance that buildings can
have, such that we strike some sort of nice, ideally canonical/motivated balance." Then, on a
single eps of 1e-4 against ordering by two numbers: "I'm not sure between a single 1e-4 and order
via two numbers, so maybe we can test both".

Both candidates place demand with the placement fix (insulator.Placement.of(open_side=True)).
The ENCLOSED homes are insulator.walled's: tiers 1, 3, 4 and 5, no free cell within RING_M of
the outline reaching the street.
  A  one number: every footprint sub-sample conducts eps relative to open ground
     (insulator.insulator_open; EPS_A, the strength SIMP's default already searches at). Every
     home injects, the enclosed included, and Lens A is over every home.
  B  two numbers, the enclosed homes first: walls are walls (eps 0, the exact world). A clearing
     is scored first by (i) the population still enclosed after it (no injection cell open and
     street-connected in its exact field), then by (ii) Lens A over the homes not enclosed at
     baseline: today's metric with the placement fix. The enclosed homes inject nothing in B,
     whatever the clearing, so their freeing shows in (i) alone. Optimizers would search at
     eps_gate: the largest eps at which the walls carry at most 1 - GATE of any open home's
     escape (u_open(eps) / u_open(0) >= GATE for every tier-0 home; u_open the potential due to
     the tier-0 demand alone).

    S=research/roadless/walls.py      # from the repo root, PYTHONPATH=., single-threaded
    REBLOCK_BLOCK_BANK=<bank> uv run python $S small <procs> <eps,> <ids,>   # CPU: the gate curve
    REBLOCK_BLOCK_BANK=<bank> uv run python $S small-exact <procs> <ids,>    # CPU: its repairs
    uv run python $S clearings <out.parquet> <ids,>       # the stored clearings, for `large`
    CUDA_PATH=/usr uv run python $S large <clearings.parquet> <eps,> <ids,>   # GPU
    uv run python $S report    # markdown to stdout; CSVs insulator_rows/summary_walls_*.csv

small (h 0.5, both populations, one file per block and eps): per home u, the full demand (eps 0:
the non-enclosed homes, which are tiers 0 and 2; eps > 0: every home; NaN for the enclosed homes
at eps 0), and u_open, the tier-0 demand alone (NaN off tier 0). Also the solve's unknowns, time,
P and CG residual per field, and the solver. `small` solves with lifted.CpuAMG (pyamg's CG to
insulator.RTOL, which stops at 500 iterations and accepts 10x the tolerance); `small-exact`
re-solves with SuperLU (Exact) each block and eps where that CG failed or stopped above
insulator.RTOL in some field.
large (GPU, the ids in the order given; on the cluster one block a task; one file pair per block
and eps): the scenarios none, simp_default and cheap (the stored D 0.05 clearings), free and
free+simp. `free` clears every building with a footprint sub-sample on any enclosed home's
cheapest crossing (insulator.HomeDoors with OpenEnds: one per home and pocket). Per scenario:
J_2 over every home, over the non-enclosed and over the enclosed (area population; count too
with nothing cleared), D, and the enclosed population left (B (i)). Per home: u, whether it is
reached and whether its building is cleared.
clearings.parquet: block, name (simp_default, cheap), cleared (building indices), stored_perm:
insulator.STORED's clearings, read as insulator.large_one reads them. `large` reads it from the
cluster's inputs ($CLUSTER_SUBMIT_INPUTS/<its name>) or a local path, since a task's checkout
has no relax_rows/ or polish_rows/ (parquet is not committed); `free` and `free+simp` are made
in the task. `large` exits non-zero when a block failed, after running the rest.
Rows in insulator_rows/ (parquet, ignored); the report's CSV summaries sit beside them. A CG
that does not converge, or a GPU out of memory, is recorded under failed/ beside the rows and
reported, never loosened; that block and eps are then done for `small` and `large`.
"""
from __future__ import annotations

import dataclasses
import gc
import os
import sys
import time
import traceback
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import factorized

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import insulator  # noqa: E402
import lifted  # noqa: E402

# ------------------------------------------------------------------ the run's parameters
# eps is on the command line and in every row's path; h, ell, K, the ring, the neighbours and the
# CG tolerances are insulator's (H, ELL_M, K, RING_M, NEIGHBOURS, RTOL, RTOL_LARGE).
GATE = 0.95              # eps_gate: the walls carry at most 1 - GATE of any open home's escape
EPS_A = 1e-4             # A's eps: the strength SIMP's default searches at
GATE_SIG = 2             # eps_gate's significant figures (the large run's eps)
Z_GATED = 8.0            # a gated home: u(0) over its block's median tier-0 u(0), at least this
GATE_Q = (0.001, 0.01, 0.05, 0.5)    # the gate curve's quantiles (besides its min)
WORST = 5                # the worst homes named at EPS_A
FIT_BOUNDS = (1e-12, 1.0)    # where the fit looks for eps0 (a fit on a bound is reported)
SCENARIOS = ("none", "simp_default", "cheap", "free", "free+simp")
CLEARINGS = SCENARIOS[1:]
SENTINELS = ("ZAF.9.3.1_1_22422", "ZAF.9.3.1_1_30848", "ZAF.9.3.1_1_5810", "ZAF.9.5.4_1_9712")
NOT_CONVERGED = "AMG-CG did not converge"   # lifted.CpuAMG's and lifted._pcg's RuntimeError


def small_dir() -> Path:
    return insulator.ROWS / f"walls_small_h{insulator.H:g}_ell{insulator.ELL_M:g}_K{insulator.K}"


def large_dir() -> Path:
    return insulator.ROWS / f"walls_large_h{insulator.H:g}_ell{insulator.ELL_M:g}_K{insulator.K}"


def _small_path(bid: str, eps: float) -> Path:
    return small_dir() / f"{bid}__eps{eps:g}.parquet"


def _large_path(bid: str, eps: float, what: str) -> Path:
    """A large block's rows at one eps: `what` is scenarios or homes."""
    return large_dir() / f"{bid}__eps{eps:g}__{what}.parquet"


def _write(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, out)


def _failed(d: Path, bid: str, eps: float | None, stage: str, e: BaseException) -> None:
    """Record a failed solve (or a block's setup, eps None) under d/failed/."""
    tag = "setup" if eps is None else f"eps{eps:g}"
    _write(pd.DataFrame([dict(block=bid, eps=np.nan if eps is None else eps, stage=stage,
                              error=type(e).__name__, message=str(e)[:400],
                              at=time.strftime("%Y-%m-%d %H:%M:%S"))]),
           d / "failed" / f"{bid}__{tag}__{stage}.parquet")
    print(f"{time.strftime('%H:%M:%S')} {bid} {tag} {stage} FAILED {type(e).__name__}: {e}"[:400],
          flush=True)


def _done(out: Path, d: Path, bid: str, eps: float) -> bool:
    """A block and eps are done when their rows exist or a failure was recorded for them."""
    return out.exists() or any((d / "failed").glob(f"{bid}__eps{eps:g}__*.parquet"))


def _not_converged(e: RuntimeError) -> bool:
    return str(e).startswith(NOT_CONVERGED)


def _injecting(enc: np.ndarray, eps: float) -> np.ndarray:
    """The homes that inject: eps 0 (B) the non-enclosed, whatever the clearing; eps > 0 (A)
    every home."""
    return ~enc if eps == 0.0 else np.ones(len(enc), dtype=bool)


# ------------------------------------------------------------------ solves

class Solve(NamedTuple):
    """One system's solve: u per home and field (homes x fields), P = b . u per field, the
    unknowns, the relative residual ||b - A u|| / ||b|| per field on the unknowns the solver
    kept (its own system), and the seconds of setup and solve."""
    u: np.ndarray
    P: np.ndarray
    unknowns: int
    resid: np.ndarray
    t_solve: float


def potentials(grid: lifted.Grid, open_: np.ndarray, fields: list[np.ndarray],
               pl: insulator.Placement, p: lifted.Params, *, rtol: float) -> Solve:
    """insulator.home_potentials, with each field's CG residual and the solve time."""
    xp = p.solver.xp
    ts = time.perf_counter()
    sy = lifted.System(grid, open_, p)
    B = xp.stack([sy.load(f, p) for f in fields], axis=1)
    U = sy.solve(B, rtol=rtol)
    t_solve = time.perf_counter() - ts
    P = p.solver.to_host((B * U).sum(axis=0))
    resid = np.array([_resid(sy, B[:, j], U[:, j]) for j in range(B.shape[1])])
    ub = insulator._cell_u(U, sy.unk_cell, open_, p)
    n = sy.n
    del sy, B, U
    p.solver.release()
    return Solve(u=pl.home_u(ub), P=P, unknowns=n, resid=resid, t_solve=t_solve)


def _resid(sy: lifted.System, b, u) -> float:
    xp = sy.xp
    bk = b[sy.keep]
    return float(xp.linalg.norm(bk - sy.A @ u[sy.keep])) / float(xp.linalg.norm(bk))


def _check_P(s: Solve, j: int, w: np.ndarray, inj: np.ndarray) -> None:
    """P = b . u is the injecting homes' sum of w u (S's columns sum to 1, and u_i is the
    injection-weighted mean of its cells' potentials)."""
    u = s.u[:, j]
    assert np.isclose(s.P[j], float((w[inj] * u[inj]).sum()), rtol=1e-9, atol=0.0), \
        f"P {s.P[j]} against sum w u {(w[inj] * u[inj]).sum()}"


@dataclasses.dataclass(frozen=True)
class Exact(lifted.CpuAMG):
    """lifted.CpuAMG's device with an exact solve in place of its CG: SuperLU (scipy's
    factorized) and one step of iterative refinement with the same factors. At eps 1e-6 the
    contrast is 1e6 and the plain factorization leaves the all-homes field's residual near 1e-8;
    the step takes it to about 2e-9 (38486: 8.8e-9 to 2.2e-9, its u within 6e-9 of pyamg's)."""

    def prepare(self, A, coarsening):
        lu = factorized(A.tocsc())

        def solve1(b):
            x = lu(b)
            return x + lu(b - A @ x)

        def solve(b, rtol, x0=None):
            if b.ndim == 1:
                return solve1(b)
            return np.stack([solve1(b[:, i]) for i in range(b.shape[1])], axis=1)
        return solve


class SmallSolver(NamedTuple):
    """How a small block is solved (resolved by the command): the rows' name for it, and
    lifted's solver."""
    name: str
    solver: lifted.Solver


AMG = SmallSolver(name="amg", solver=lifted.CpuAMG())         # insulator.params("cpu")'s
EXACT = SmallSolver(name="superlu", solver=Exact())


# ------------------------------------------------------------------ small blocks (CPU)

def _todo_amg(bid: str, eps_list: tuple[float, ...]) -> tuple[float, ...]:
    """`small`: the eps with neither rows nor a recorded failure."""
    return tuple(e for e in eps_list if not _done(_small_path(bid, e), small_dir(), bid, e))


def _todo_exact(bid: str) -> tuple[float, ...]:
    """`small-exact`: the eps where the CG failed (a record and no rows) or stopped above
    insulator.RTOL in some field."""
    todo = set()
    for f in (small_dir() / "failed").glob(f"{bid}__eps*__*.parquet"):
        e = float(pd.read_parquet(f).eps.iloc[0])
        if not _small_path(bid, e).exists():
            todo.add(e)
    for f in small_dir().glob(f"{bid}__eps*.parquet"):
        r = pd.read_parquet(f, columns=["eps", "resid", "resid_open", "solver"])
        if (r.solver == AMG.name).all() and float(
                r[["resid", "resid_open"]].to_numpy().max()) > insulator.RTOL:
            todo.add(float(r.eps.iloc[0]))
    return tuple(sorted(todo))


def small_one(bid: str, todo: tuple[float, ...], how: SmallSolver) -> None:
    """The gate curve and the enclosed homes on one small block: per eps in `todo` one solve of
    four fields (per population the tier-0 demand, u_open, and the full demand, u) by `how`."""
    if not todo:
        return
    t0 = time.perf_counter()
    [b] = common.build_blocks([bid])
    p = dataclasses.replace(insulator.params("cpu"), solver=how.solver)
    grid = insulator.mesh(insulator.H).build(b, p, None)
    polys = np.asarray(b.buildings.outlines)
    pl = insulator.Placement.of(grid, polys, p, open_side=True)
    enc = insulator.walled(pl)
    W = {name: pop.weights(polys) for name, pop in common.POPULATIONS.items()}
    full = grid.isub.mean(axis=-1)
    shape = grid.inside.shape
    tier0 = pl.tier == 0
    cen = insulator._centroids(polys)
    n = len(polys)
    for eps in todo:
        inj = _injecting(enc, eps)
        fields = []
        for w in W.values():
            fields += [pl.field(w, tier0, shape), pl.field(w, inj, shape)]
        try:
            s = potentials(grid, insulator.insulator_open(grid.ff0, full, eps), fields, pl, p,
                           rtol=insulator.RTOL)
        except RuntimeError as e:
            if not _not_converged(e):
                raise
            _failed(small_dir(), bid, eps, "none", e)
            continue
        rows = []
        for j, (name, w) in enumerate(W.items()):
            _check_P(s, 2 * j, w, tier0)
            _check_P(s, 2 * j + 1, w, inj)
            rows.append(pd.DataFrame(dict(
                block=bid, eps=eps, pop=name, home=np.arange(n), tier=pl.tier, enclosed=enc,
                shadowed=pl.shadowed, barrier_m=pl.barrier_m, w=w, cx=cen[:, 0], cy=cen[:, 1],
                u=np.where(inj, s.u[:, 2 * j + 1], np.nan),
                u_open=np.where(tier0, s.u[:, 2 * j], np.nan),
                P_open=s.P[2 * j], P=s.P[2 * j + 1], resid_open=s.resid[2 * j],
                resid=s.resid[2 * j + 1], unknowns=s.unknowns, t_solve=s.t_solve,
                cells=int(grid.inside.sum()), solver=how.name)))
        _write(pd.concat(rows, ignore_index=True), _small_path(bid, eps))
        print(f"{time.strftime('%H:%M:%S')} {bid} eps {eps:g} ({how.name}): {s.unknowns} "
              f"unknowns, resid {s.resid.max():.1e}, {s.t_solve:.0f}s", flush=True)
    print(f"{time.strftime('%H:%M:%S')} {bid} n={n} enclosed {int(enc.sum())} tiers "
          f"{np.bincount(pl.tier, minlength=6).tolist()} {time.perf_counter() - t0:.0f}s",
          flush=True)


# ------------------------------------------------------------------ large blocks (GPU)

class Scenario(NamedTuple):
    """A clearing: its name (the rows'), the buildings it clears (bool per building), and its
    stored Lens A under today's metric (nan for the clearings made here)."""
    name: str
    x: np.ndarray
    stored_perm: float


class Cleared(NamedTuple):
    """A scenario in the block's geometry: its exact open field and, per home, whether it is
    reached there (`reached`)."""
    s: Scenario
    op0: np.ndarray
    reached: np.ndarray


class FreePaths(NamedTuple):
    """`free`'s crossings: per door site its home, pocket, ends (a in the pocket, b street
    side), footprint metres, path cells and the buildings with a sub-sample on the path (`own`:
    the home's own among them); x, their union (bool per building)."""
    sites: pd.DataFrame
    x: np.ndarray


def write_clearings(out: Path, ids: list[str]) -> None:
    """`clearings`: the stored D 0.05 clearings for `large` (insulator.STORED, read as
    insulator.large_one reads them), one row per block and clearing."""
    rows = []
    for bid in ids:
        for name, (rel, col, perm_col) in insulator.STORED.items():
            r = pd.read_parquet(insulator.HERE / rel
                                / f"{bid}_p{insulator.CLEARING_POWER:g}.parquet").iloc[0]
            rows.append(dict(block=bid, name=name, cleared=np.asarray(r[col], dtype=np.int64),
                             stored_perm=float(r[perm_col])))
    _write(pd.DataFrame(rows), out)
    print(f"{out}: {len(rows)} clearings of {len(ids)} blocks", flush=True)


def _stored(rows: pd.DataFrame, n: int, name: str) -> Scenario:
    """The stored clearing `name` from a block's rows of the clearings file."""
    r = rows[rows["name"] == name]
    if len(r) != 1:
        raise ValueError(f"{len(r)} stored clearings named {name!r}")
    x = np.zeros(n, dtype=bool)
    x[np.asarray(r.cleared.iloc[0], dtype=np.int64)] = True
    return Scenario(name=name, x=x, stored_perm=float(r.stored_perm.iloc[0]))


def free_paths(grid: lifted.Grid, pl: insulator.Placement, p: lifted.Params,
               lab: np.ndarray) -> FreePaths:
    """Every enclosed home's cheapest crossing per pocket it injects into (insulator.HomeDoors
    with OpenEnds), followed back by the predecessors of the crossing Dijkstra pockets_of runs,
    and the buildings with a footprint sub-sample on any of their cells."""
    ends = insulator.OpenEnds()
    pk = insulator.pockets_of(grid, pl, p, ends=ends)
    sites = insulator.HomeDoors().sites(pk, pl)
    dist, pred, src = dijkstra(insulator._crossing_graph(grid, pk.open_),
                               indices=np.flatnonzero(ends.sources(pk.open_.ravel(), pk.full,
                                                                  pk.reach)),
                               min_only=True, return_predecessors=True)
    o = pk.open_.ravel()
    labf = lab.reshape(grid.inside.size, -1)
    x = np.zeros(len(pl.tier), dtype=bool)
    rows = []
    for a, b, bar, q, j in zip(sites.a, sites.b, sites.barrier_m, sites.pocket, sites.home,
                               strict=True):
        path = [int(a)]
        while pred[path[-1]] >= 0:
            path.append(int(pred[path[-1]]))
        assert path[-1] == b == src[a], \
            f"home {j}: the crossing from {a} ends at {path[-1]}, not {b}"
        # the path's cost (each step pays h (1 - o) at its target, plus 1e-9) is the site's
        cost = float((grid.h * (1.0 - o[path[:-1]]) + 1e-9).sum())
        assert np.isclose(cost, bar, rtol=1e-9, atol=1e-9) and np.isclose(dist[a], bar), \
            f"home {j}: path cost {cost} against {bar}"
        sub = labf[path]
        hit = np.unique(sub[sub >= 0])
        x[hit] = True
        rows.append(dict(home=int(j), pocket=int(q), a=int(a), b=int(b), barrier_m=float(bar),
                         cells=len(path), n_buildings=len(hit),
                         buildings=",".join(str(k) for k in hit), own=bool(np.isin(j, hit))))
    return FreePaths(sites=pd.DataFrame(rows), x=x)


def _exact(grid: lifted.Grid, lab: np.ndarray, x: np.ndarray) -> np.ndarray:
    """The exact open field with the buildings x cleared (insulator.large_one's; none: ff0)."""
    cleared = (lab >= 0) & x[np.maximum(lab, 0)]
    return (grid.isub & (~grid.bsub | cleared)).mean(axis=-1)


def reached(grid: lifted.Grid, op0: np.ndarray, pl: insulator.Placement,
            p: lifted.Params) -> np.ndarray:
    """Per home: some injection cell (pl.S's nonzeros in its column) open and street-connected
    in the exact field op0."""
    g = lifted.grounded(grid, op0, p).ravel().astype(float)
    return np.asarray(pl.S.T @ g).ravel() > 0


def large_one(bid: str, p: lifted.Params, eps_list: tuple[float, ...],
              stored: pd.DataFrame) -> bool:
    """The scenarios on one large block, per eps in `eps_list` (one file pair each); `stored`:
    its rows of the clearings file. True when a scenario's CG did not converge (recorded)."""
    todo = [e for e in eps_list
            if not _done(_large_path(bid, e, "scenarios"), large_dir(), bid, e)]
    if not todo:
        return False
    t0 = time.perf_counter()
    unconverged = False
    try:
        [b] = common.build_blocks([bid])
        grid = insulator.mesh(insulator.H).build(b, p, None)
        polys = np.asarray(b.buildings.outlines)
        n = len(polys)
        pl = insulator.Placement.of(grid, polys, p, open_side=True)
        enc = insulator.walled(pl)
        W = {name: pop.weights(polys) for name, pop in common.POPULATIONS.items()}
        wa = W["area"]
        full = grid.isub.mean(axis=-1)
        shape = grid.inside.shape
        lab = grid.label_sub(polys)
        none = Scenario(name="none", x=np.zeros(n, dtype=bool), stored_perm=np.nan)
        assert (reached(grid, _exact(grid, lab, none.x), pl, p) == ~enc).all(), \
            "enclosed and unreached at baseline differ"
        simp = _stored(stored, n, "simp_default")
        scen = [none, simp, _stored(stored, n, "cheap")]
        if enc.any():
            fp = free_paths(grid, pl, p, lab)
            _write(fp.sites.assign(block=bid), large_dir() / f"{bid}__free.parquet")
            scen += [Scenario(name="free", x=fp.x, stored_perm=np.nan),
                     Scenario(name="free+simp", x=fp.x | simp.x, stored_perm=np.nan)]
        worlds = []
        for s in scen:
            op0 = _exact(grid, lab, s.x)
            worlds.append(Cleared(s=s, op0=op0, reached=reached(grid, op0, pl, p)))
        del lab
    except Exception as e:
        _failed(large_dir(), bid, None, "setup", e)
        raise
    cen = insulator._centroids(polys)
    print(f"{time.strftime('%H:%M:%S')} {bid} n={n} cells={int(grid.inside.sum())} tiers "
          f"{np.bincount(pl.tier, minlength=6).tolist()} enclosed {int(enc.sum())} "
          + " ".join(f"{c.s.name}:{int(c.s.x.sum())}/{wa[c.s.x].sum() / wa.sum():.4f}"
                     f"/left {int((enc & ~c.reached).sum())}" for c in worlds)
          + f" setup {time.perf_counter() - t0:.0f}s", flush=True)
    for eps in todo:
        inj = _injecting(enc, eps)
        srows, hrows = [], []
        for c in worlds:
            s = c.s
            pops = list(W) if s.name == "none" else ["area"]
            fields = [pl.field(W[k], inj, shape) for k in pops]
            left = enc & ~c.reached
            base = dict(block=bid, eps=eps, scenario=s.name, n_cleared=int(s.x.sum()),
                        D=float(wa[s.x].sum() / wa.sum()), enc_left=int(left.sum()),
                        enc_left_area=float(wa[left].sum() / wa.sum()),
                        enc_left_count=float(left.sum() / n), stored_perm=s.stored_perm)
            try:
                sv = potentials(grid, insulator.insulator_open(c.op0, full, eps), fields,
                                pl, p, rtol=insulator.RTOL_LARGE)
            except RuntimeError as e:
                if not _not_converged(e):
                    _failed(large_dir(), bid, eps, s.name, e)
                    raise
                _failed(large_dir(), bid, eps, s.name, e)
                srows += [dict(base, pop=k, error=str(e)[:400]) for k in pops]
                unconverged = True
                continue
            except MemoryError as e:      # the GPU is shared: recorded, the block abandoned
                _failed(large_dir(), bid, eps, s.name, e)
                raise
            for j, k in enumerate(pops):
                w, u = W[k], sv.u[:, j]
                _check_P(sv, j, w, inj)
                srows.append(dict(base, pop=k, J2=float((w[inj] * u[inj] ** 2).sum()),
                                  J2_open=float((w[~enc] * u[~enc] ** 2).sum()),
                                  J2_enc=(float((w[enc] * u[enc] ** 2).sum()) if eps > 0
                                          else np.nan),
                                  P=float(sv.P[j]), unknowns=sv.unknowns, t_solve=sv.t_solve,
                                  resid=float(sv.resid[j]), error=""))
                hrows.append(pd.DataFrame(dict(
                    block=bid, eps=eps, scenario=s.name, pop=k, home=np.arange(n),
                    tier=pl.tier, enclosed=enc, shadowed=pl.shadowed, barrier_m=pl.barrier_m,
                    w=w, cx=cen[:, 0], cy=cen[:, 1], u=np.where(inj, u, np.nan),
                    reached=c.reached, cleared=s.x)))
            print(f"  {bid} eps {eps:g} {s.name}: {sv.unknowns} unknowns, resid "
                  f"{sv.resid.max():.1e}, {sv.t_solve:.0f}s", flush=True)
        if hrows:
            _write(pd.concat(hrows, ignore_index=True), _large_path(bid, eps, "homes"))
        _write(pd.DataFrame(srows), _large_path(bid, eps, "scenarios"))
    print(f"{time.strftime('%H:%M:%S')} {bid} {todo} done in {time.perf_counter() - t0:.0f}s",
          flush=True)
    return unconverged


def large(clearings: str, eps_list: tuple[float, ...], ids: list[str]) -> None:
    """`large`: the blocks in the order given. A block that fails is recorded and the next one
    runs; the command then exits non-zero."""
    src = (Path(os.environ["CLUSTER_SUBMIT_INPUTS"]) / clearings
           if "CLUSTER_SUBMIT_INPUTS" in os.environ else Path(clearings))
    stored = pd.read_parquet(src)
    p = insulator.params("gpu")
    large_dir().mkdir(parents=True, exist_ok=True)
    failed = []
    for bid in ids:
        try:
            if large_one(bid, p, eps_list, stored[stored.block == bid]):
                failed.append(bid)
        except Exception as e:            # recorded by large_one; on to the next block
            print(f"{time.strftime('%H:%M:%S')} {bid} FAILED {type(e).__name__}: {e}"[:400],
                  flush=True)
            traceback.print_exc()
            failed.append(bid)
        gc.collect()
        p.solver.release()
    if failed:
        raise SystemExit(f"failed: {failed} (recorded in {large_dir() / 'failed'})")


# ------------------------------------------------------------------ report: small blocks

def gate_rows(small: pd.DataFrame) -> pd.DataFrame:
    """Per tier-0 home with u_open(0) > 0, population and eps > 0: ratio = u_open(eps) /
    u_open(0), and z = u(0) / its block's median tier-0 u(0) (u the full demand at eps 0: B
    (ii)'s world). A tier-0 home whose cells are all ground has u 0 in every world and no
    ratio."""
    t0 = small[small.tier == 0]
    base = t0[t0.eps == 0.0][["block", "pop", "home", "u", "u_open"]].rename(
        columns={"u": "u0", "u_open": "uo0"})
    base["z"] = base.u0 / base.groupby(["block", "pop"]).u0.transform("median")
    m = t0[t0.eps > 0][["block", "pop", "eps", "home", "u_open"]].merge(
        base, on=["block", "pop", "home"], validate="many_to_one")
    m = m[m.uo0 > 0].copy()
    m["ratio"] = m.u_open / m.uo0
    return m


def gate_table(m: pd.DataFrame) -> pd.DataFrame:
    """Per population and eps, the ratio over the tier-0 homes and over the gated ones (z >=
    Z_GATED): min, quantiles, and how many are below GATE."""
    out = []
    for (pop, eps), g in m.groupby(["pop", "eps"]):
        for subset, gg in (("tier0", g), ("gated", g[g.z >= Z_GATED])):
            r = gg.ratio
            row = dict(pop=pop, eps=eps, subset=subset, blocks=gg.block.nunique(), homes=len(gg),
                       min=float(r.min()) if len(r) else np.nan)
            for q in GATE_Q:
                row[f"p{100 * q:g}"] = insulator._q(r, q)
            row["below"] = int((r < GATE).sum())
            row["share_below"] = float((r < GATE).mean()) if len(r) else np.nan
            out.append(row)
    return pd.DataFrame(out)


def crossing_eps(eps: np.ndarray, v: np.ndarray, level: float) -> float:
    """The largest eps at which v (falling with eps) is still >= level: linear in log eps
    between the last grid point at or above it and the first below. nan if the smallest eps is
    already below; inf if no grid point is."""
    o = np.argsort(eps)
    e, v = eps[o], v[o]
    below = np.flatnonzero(v < level)
    if not len(below):
        return np.inf
    k = int(below[0])
    if k == 0:
        return np.nan
    t = (v[k - 1] - level) / (v[k - 1] - v[k])
    return float(np.exp(np.log(e[k - 1]) + t * (np.log(e[k]) - np.log(e[k - 1]))))


def gate_eps(gt: pd.DataFrame) -> pd.DataFrame:
    """Per population: the eps at which the tier-0 min, p0.1 and p1 cross GATE, and whether
    each curve falls monotonically over the grid."""
    out = []
    for pop, g in gt[gt.subset == "tier0"].groupby("pop"):
        g = g.sort_values("eps")
        e = g.eps.to_numpy()
        row = dict(pop=pop)
        for col in ("min", "p0.1", "p1"):
            v = g[col].to_numpy()
            row[f"eps_{col}"] = crossing_eps(e, v, GATE)
            row[f"monotone_{col}"] = bool((np.diff(v) <= 0).all())
        out.append(row)
    return pd.DataFrame(out)


def round_sig(x: float, sig: int) -> float:
    return float(f"{x:.{sig}g}")


def fit_eps0(eps: np.ndarray, r: np.ndarray) -> float:
    """eps0 of r = 1 / (1 + eps / eps0), least squares in log r."""
    le, lr = np.log(eps), np.log(r)
    res = minimize_scalar(lambda t: float(((lr + np.log1p(np.exp(le - t))) ** 2).sum()),
                          bounds=tuple(np.log(FIT_BOUNDS)), method="bounded",
                          options=dict(xatol=1e-10))
    return float(np.exp(res.x))


def worst_fit(m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per population, the tier-0 home with the lowest ratio at EPS_A: eps0 of
    1 / (1 + eps / eps0) fitted to its curve, the eps that puts it at GATE, whether it is the
    worst at every eps; and its curve, with the model and the eps0 each point alone gives."""
    fits, curves = [], []
    for pop, g in m.groupby("pop"):
        a = g[g.eps == EPS_A]
        w = a.loc[a.ratio.idxmin()]
        c = g[(g.block == w.block) & (g.home == w.home)].sort_values("eps")
        e, r = c.eps.to_numpy(), c.ratio.to_numpy()
        eps0 = fit_eps0(e, r)
        worst_all = all(gg.loc[gg.ratio.idxmin(), "block"] == w.block
                        and gg.loc[gg.ratio.idxmin(), "home"] == w.home
                        for _, gg in g.groupby("eps"))
        fits.append(dict(pop=pop, block=w.block, home=int(w.home), u0=float(w.u0), z=float(w.z),
                         eps0=eps0, on_bound=bool(np.isclose(eps0, FIT_BOUNDS, rtol=1e-3).any()),
                         eps_at_gate=eps0 * (1.0 / GATE - 1.0),
                         rms_log=float(np.sqrt(np.mean((np.log(r) + np.log1p(e / eps0)) ** 2))),
                         worst_at_every_eps=worst_all))
        curves.append(pd.DataFrame(dict(pop=pop, eps=e, ratio=r, model=1.0 / (1.0 + e / eps0),
                                        eps0_point=e * r / (1.0 - r))))
    return pd.DataFrame(fits), pd.concat(curves, ignore_index=True)


def worst_homes(m: pd.DataFrame) -> pd.DataFrame:
    """Per population, the WORST tier-0 homes at EPS_A by ratio."""
    a = m[m.eps == EPS_A]
    return pd.concat([g.nsmallest(WORST, "ratio") for _, g in a.groupby("pop")],
                     ignore_index=True)[["pop", "block", "home", "ratio", "u0", "z"]]


def enclosed_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Per enclosed home, population and eps > 0: r = u_i / max u_j over N(i), its NEIGHBOURS
    nearest tier-0 homes (insulator.neighbours), and z = u_i / its block's median tier-0 u (same
    eps). `df`: one row per block, population, eps and home (u, tier, enclosed, cx, cy)."""
    out = []
    for (bid, pop), g in df[df.eps > 0].groupby(["block", "pop"]):
        if not g.enclosed.any():
            continue
        one = g[g.eps == g.eps.iloc[0]]
        nb = insulator.neighbours(one[["home", "tier", "cx", "cy"]])
        tier = one.set_index("home").tier
        encl = one.home[one.enclosed].to_numpy()
        for eps, ge in g.groupby("eps"):
            u = ge.set_index("home").u
            med = float(u[tier == 0].median())
            for i in encl:
                out.append(dict(block=bid, pop=pop, eps=eps, home=int(i), tier=int(tier[i]),
                                u=float(u[i]), u_nbr_max=float(u[nb[i]].max()),
                                r=float(u[i] / u[nb[i]].max()), z=float(u[i] / med)))
    return pd.DataFrame(out)


def enclosed_table(er: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (pop, eps), g in er.groupby(["pop", "eps"]):
        out.append(dict(pop=pop, eps=eps, homes=len(g), blocks=g.block.nunique(),
                        r_med=insulator._q(g.r, 0.5), r_p90=insulator._q(g.r, 0.9),
                        r_max=float(g.r.max()), z_med=insulator._q(g.z, 0.5),
                        z_p90=insulator._q(g.z, 0.9), z_max=float(g.z.max())))
    return pd.DataFrame(out)


def enclosed_share(small: pd.DataFrame) -> pd.DataFrame:
    """Per eps > 0, over the blocks with enclosed homes: their share of J_2 (area)."""
    a = small[(small["pop"] == "area") & (small.eps > 0)].copy()
    a["j"] = a.w * a.u ** 2
    a["j_enc"] = np.where(a.enclosed, a.j, 0.0)
    per = a.groupby(["eps", "block"]).agg(j=("j", "sum"), j_enc=("j_enc", "sum"),
                                          enclosed=("enclosed", "sum")).reset_index()
    per = per[per.enclosed > 0]
    per["share"] = per.j_enc / per.j
    return per.groupby("eps").agg(blocks=("block", "nunique"), share_med=("share", "median"),
                                  share_max=("share", "max")).reset_index()


def old_rows_check(small: pd.DataFrame) -> pd.DataFrame:
    """Tier 0 is the same under both placements, so tier-0 u_open here equals the eps world's
    (insulator_rows/small_h0.5_ell3_K8): its eps-0 u, and its EPS_A u_open."""
    old = insulator._read(insulator.small_dir(), "ZAF*[0-9].parquet")
    out = []
    for eps, col in ((0.0, "u"), (EPS_A, "u_open")):
        o = old[old.eps == eps][["block", "pop", "home", "tier", col]].rename(
            columns={col: "old", "tier": "tier_old"})
        nw = small[small.eps == eps][["block", "pop", "home", "tier", "u_open"]]
        m = nw.merge(o, on=["block", "pop", "home"], validate="one_to_one")
        m = m[(m.tier == 0) | (m.tier_old == 0)]
        same_tier0 = bool(((m.tier == 0) == (m.tier_old == 0)).all())
        m = m[m.tier == 0]
        z = m.old == 0
        rel = (m.u_open[~z] - m.old[~z]).abs() / m.old[~z].abs()
        out.append(dict(eps=eps, old=col, blocks=m.block.nunique(), homes=len(m),
                        same_tier0=same_tier0, rel_max=float(rel.max()),
                        rel_med=insulator._q(rel, 0.5), zero_homes=int(z.sum()),
                        zero_abs_max=float(m.u_open[z].abs().max()) if z.any() else 0.0))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ report: large blocks

class LargeTables(NamedTuple):
    blocks: pd.DataFrame
    a: pd.DataFrame
    a_gate: pd.DataFrame
    b: pd.DataFrame
    order: pd.DataFrame
    free: pd.DataFrame
    leak: pd.DataFrame


def _area(sc: pd.DataFrame, bid: str, eps: float) -> pd.DataFrame:
    """A block's area rows at eps, one per scenario (absent ones nan)."""
    g = sc[(sc.block == bid) & (sc.eps == eps) & (sc["pop"] == "area")]
    return g.set_index("scenario").reindex(list(SCENARIOS))


def _lens(J2: pd.Series) -> pd.Series:
    """Lens A per scenario: 1 - (J_2 / J_2 with nothing cleared)^(1/2)."""
    return 1.0 - (J2 / J2["none"]) ** 0.5


def _world_cols(a: pd.DataFrame, tag: str) -> dict:
    """A's columns at one eps: Lens A over every home and over the non-enclosed, and the
    enclosed homes' share of J_2, per scenario."""
    la, lo, sh = _lens(a.J2), _lens(a.J2_open), a.J2_enc / a.J2
    return ({f"{tag}_{c}": la[c] for c in CLEARINGS}
            | {f"{tag}_open_{c}": lo[c] for c in CLEARINGS}
            | {f"{tag}_encJ2_{c}": sh[c] for c in SCENARIOS})


def _leader(d: float) -> str:
    """Which stored clearing a difference simp_default - cheap (higher better) puts first."""
    if not np.isfinite(d):
        return "n/a"
    return "simp" if d > 0 else "cheap" if d < 0 else "tie"


def _geometry(sc: pd.DataFrame, bid: str) -> pd.DataFrame:
    """A block's per-scenario columns that no eps changes (D, the buildings cleared, the
    enclosed population left, the stored Lens A), checked equal across its eps rows."""
    cols = ["n_cleared", "D", "enc_left", "enc_left_area", "stored_perm"]
    g = sc[(sc.block == bid) & (sc["pop"] == "area")]
    for c in cols:
        spread = g.groupby("scenario")[c].nunique(dropna=False)
        assert (spread <= 1).all(), f"{bid}: {c} differs between eps"
    return g.drop_duplicates("scenario").set_index("scenario")[cols].reindex(list(SCENARIOS))


def large_tables(sc: pd.DataFrame, homes: pd.DataFrame, gate: float) -> LargeTables:
    blocks, A, Ag, B, order, free, leak = [], [], [], [], [], [], []
    for bid in sorted(sc.block.unique()):
        h = homes[(homes.block == bid) & (homes["pop"] == "area")]
        h = h[h.eps == h.eps.min()]
        h0 = h[h.scenario == "none"]
        enc = h0[h0.enclosed]
        tiers = np.bincount(enc.tier, minlength=6)
        geo = _geometry(sc, bid)
        a0, aA, aG = _area(sc, bid, 0.0), _area(sc, bid, EPS_A), _area(sc, bid, gate)
        own = h[(h.scenario == "free") & h.enclosed & h.cleared]
        blocks.append(dict(block=bid, homes=len(h0), enclosed=len(enc),
                           tiers=" ".join(f"{t}:{k}" for t, k in enumerate(tiers) if k),
                           tier2=int((h0.tier == 2).sum()),
                           enc_area=float(enc.w.sum() / h0.w.sum()),
                           **{f"D_{c}": geo.D[c] for c in CLEARINGS},
                           n_free=geo.n_cleared["free"], own_free=len(own)))
        A.append(dict(block=bid, **_world_cols(aA, "A")))
        Ag.append(dict(block=bid, **_world_cols(aG, "G")))
        lb = _lens(a0.J2_open)
        B.append(dict(block=bid, **{f"Bi_{c}": geo.enc_left_area[c] for c in SCENARIOS},
                      **{f"Bii_{c}": lb[c] for c in CLEARINGS},
                      today_simp=geo.stored_perm["simp_default"],
                      today_cheap=geo.stored_perm["cheap"],
                      Bii_minus_today_simp=lb["simp_default"] - geo.stored_perm["simp_default"],
                      Bii_minus_today_cheap=lb["cheap"] - geo.stored_perm["cheap"]))
        la, lg = _lens(aA.J2), _lens(aG.J2)
        bi = geo.enc_left_area["simp_default"] - geo.enc_left_area["cheap"]
        d = dict(today=geo.stored_perm["simp_default"] - geo.stored_perm["cheap"],
                 A=la["simp_default"] - la["cheap"], A_gate=lg["simp_default"] - lg["cheap"],
                 Bii=lb["simp_default"] - lb["cheap"])
        order.append(dict(block=bid, **{f"d_{k}": v for k, v in d.items()}, d_Bi=bi,
                          **{f"lead_{k}": _leader(v) for k, v in d.items()},
                          lead_B=_leader(-bi) if bi != 0 else _leader(d["Bii"])))
        if len(enc):
            free.append(dict(block=bid, D_free=geo.D["free"], D_simp=geo.D["simp_default"],
                             n_free=geo.n_cleared["free"], A_free=la["free"],
                             A_simp=la["simp_default"],
                             A_free_over_simp=la["free"] / la["simp_default"],
                             G_free=lg["free"], G_simp=lg["simp_default"],
                             A_open_free=_lens(aA.J2_open)["free"], Bii_free=lb["free"],
                             Bi_free=geo.enc_left_area["free"]))
        if bid in SENTINELS:
            for c in ("simp_default", "cheap"):
                l0 = lb[c]
                leak.append(dict(block=bid, clearing=c, eps0=l0, gate=lg[c], A=la[c],
                                 d_gate=lg[c] - l0, d_A=la[c] - l0))
    return LargeTables(blocks=pd.DataFrame(blocks), a=pd.DataFrame(A), a_gate=pd.DataFrame(Ag),
                       b=pd.DataFrame(B), order=pd.DataFrame(order), free=pd.DataFrame(free),
                       leak=pd.DataFrame(leak))


def large_gate(homes: pd.DataFrame) -> pd.DataFrame:
    """The gate on the large blocks with no enclosed home, where every world's injecting homes
    are the same, so u(eps) / u(0) with nothing cleared is the walls' share alone: per block,
    population and eps > 0, its min over the homes with u(0) > 0, how many fall below GATE, the
    worst home and its z (over the block's tier-0 median u(0)), and the eps at which
    1 / (1 + eps / eps0) puts that home at GATE (eps0 from this eps alone)."""
    h = homes[homes.scenario == "none"]
    out = []
    for (bid, pop), g in h.groupby(["block", "pop"]):
        if g.enclosed.any() or not (g.eps == 0.0).any():
            continue
        g0 = g[g.eps == 0.0].set_index("home")
        base = g0.u
        for eps, ge in g[g.eps > 0].groupby("eps"):
            r = (ge.set_index("home").u / base)[base > 0]
            i = r.idxmin()
            out.append(dict(block=bid, pop=pop, eps=eps, homes=len(r), min=float(r[i]),
                            below=int((r < GATE).sum()), home=int(i),
                            z=float(base[i] / base[g0.tier == 0].median()),
                            eps_at_gate=float(eps * r[i] / (1.0 - r[i]) * (1.0 / GATE - 1.0))))
    return pd.DataFrame(out)


def solver_table(small: pd.DataFrame, sc: pd.DataFrame, fails: pd.DataFrame) -> pd.DataFrame:
    """Per world (set, eps and solver): the residuals (every field) and solve times (every
    system), median and max; the fields above the tolerance asked for (insulator.RTOL on the
    small blocks, RTOL_LARGE on the large); and the CG failures recorded (the iterative
    solver's rows)."""
    def n_failed(set_: str, eps: float) -> int:
        return 0 if fails.empty else int(((fails["set"] == set_) & (fails.eps == eps)).sum())

    out = []
    if not small.empty:
        s = small.drop_duplicates(["block", "eps", "pop"])
        for (eps, solver), g in s.groupby(["eps", "solver"]):
            res = pd.concat([g.resid_open, g.resid])
            t = g.drop_duplicates("block").t_solve
            out.append(dict(set="small", eps=eps, solver=solver, systems=len(t),
                            resid_med=insulator._q(res, 0.5), resid_max=float(res.max()),
                            above_rtol=int((res > insulator.RTOL).sum()),
                            t_med=insulator._q(t, 0.5), t_max=float(t.max()),
                            failed=n_failed("small", eps) if solver == AMG.name else 0))
    if not sc.empty:
        ok = sc[sc.error == ""]
        for eps, g in ok.groupby("eps"):
            t = g.drop_duplicates(["block", "scenario"]).t_solve
            out.append(dict(set="large", eps=eps, solver="gpu", systems=len(t),
                            resid_med=insulator._q(g.resid, 0.5), resid_max=float(g.resid.max()),
                            above_rtol=int((g.resid > insulator.RTOL_LARGE).sum()),
                            t_med=insulator._q(t, 0.5), t_max=float(t.max()),
                            failed=n_failed("large", eps)))
    return pd.DataFrame(out)


def report() -> None:
    """The tables, as markdown, and the CSV summaries (insulator_rows/summary_walls_*.csv)."""
    small = insulator._read(small_dir(), "*.parquet")
    sc = insulator._read(large_dir(), "*__scenarios.parquet")
    homes = insulator._read(large_dir(), "*__homes.parquet")
    fails = pd.concat([insulator._read(d / "failed", "*.parquet").assign(set=name)
                       for name, d in (("small", small_dir()), ("large", large_dir()))
                       if (d / "failed").exists()] or [pd.DataFrame()], ignore_index=True)
    rows = insulator.ROWS
    f3 = lambda cols: {c: "{:.3g}" for c in cols}           # noqa: E731
    f4 = lambda cols: {c: "{:.4f}" for c in cols}           # noqa: E731
    gate = np.nan
    if not small.empty:
        print(f"# Uniform walls: small blocks ({small.block.nunique()} blocks, eps "
              f"{', '.join(f'{e:g}' for e in sorted(small.eps.unique()))})\n")
        chk = old_rows_check(small)
        chk.to_csv(rows / "summary_walls_check_old.csv", index=False)
        print("## Check: tier-0 u_open against the eps world's rows\n")
        print(insulator._md(chk, f3(chk.columns)))
        m = gate_rows(small)
        gt = gate_table(m)
        gt.to_csv(rows / "summary_walls_gate.csv", index=False)
        print(f"\n## The gate curve: u_open(eps) / u_open(0) over tier-0 homes (gated: z >= "
              f"{Z_GATED:g}); below = under {GATE}\n")
        print(insulator._md(gt, f3(gt.columns) | {"share_below": "{:.2%}"}))
        ge = gate_eps(gt)
        ge.to_csv(rows / "summary_walls_eps_gate.csv", index=False)
        gate = round_sig(float(ge.eps_min.min()), GATE_SIG)
        print(f"\n## eps_gate: where each curve crosses {GATE} (linear in log eps)\n")
        print(insulator._md(ge, f3(ge.columns)))
        print(f"\neps_gate (the min over populations, {GATE_SIG} significant figures): "
              f"{gate:g}")
        fit, curve = worst_fit(m)
        fit.to_csv(rows / "summary_walls_worst_fit.csv", index=False)
        curve.to_csv(rows / "summary_walls_worst_curve.csv", index=False)
        print(f"\n## The worst home at eps {EPS_A:g}: 1 / (1 + eps / eps0) fitted to its curve\n")
        print(insulator._md(fit, f3(fit.columns)))
        print()
        print(insulator._md(curve, f3(curve.columns)))
        wh = worst_homes(m)
        wh.to_csv(rows / "summary_walls_worst_homes.csv", index=False)
        print(f"\n## The {WORST} worst tier-0 homes at eps {EPS_A:g}\n")
        print(insulator._md(wh, f3(wh.columns)))
        er = enclosed_rows(small)
        if not er.empty:
            er.to_csv(rows / "walls_enclosed_homes.csv", index=False)
            et = enclosed_table(er)
            et.to_csv(rows / "summary_walls_enclosed.csv", index=False)
            print("\n## Enclosed homes: r = u_i / max u_j over the 6 nearest tier-0 homes, "
                  "z = u_i / block median\n")
            print(insulator._md(et, f3(et.columns)))
            es = enclosed_share(small)
            es.to_csv(rows / "summary_walls_enclosed_share.csv", index=False)
            print("\n## Enclosed homes' share of J_2 (area), per block with enclosed homes\n")
            print(insulator._md(es, f3(es.columns)))
    if not sc.empty:
        print(f"\n# Large blocks ({sc.block.nunique()} blocks, eps "
              f"{', '.join(f'{e:g}' for e in sorted(sc.eps.unique()))}; eps_gate {gate:g})\n")
        if gate not in set(sc.eps):
            print(f"**The large rows have no eps_gate {gate:g}**: its columns are nan.\n")
        lt = large_tables(sc, homes, gate)
        for name, df in lt._asdict().items():
            df.to_csv(rows / f"summary_walls_large_{name}.csv", index=False)
        print("## Blocks: enclosed homes (tiers), D of each clearing (area share), `free`'s "
              "buildings and the enclosed homes' own among them\n")
        print(insulator._md(lt.blocks, f4(lt.blocks.columns) | {"enc_area": "{:.5f}"}))
        for tag, df, e in (("A", lt.a, EPS_A), ("G", lt.a_gate, gate)):
            print(f"\n## A at eps {e:g}: Lens A (area, J_2) over every home ({tag}_<clearing>) "
                  f"and over the non-enclosed ({tag}_open_<clearing>)\n")
            print(insulator._md(df[["block", *[c for c in df.columns if c.startswith(tag)
                                                and "encJ2" not in c]]], f4(df.columns)))
            print(f"\nThe enclosed homes' share of J_2 at eps {e:g}:\n")
            print(insulator._md(df[["block", *[c for c in df.columns if "encJ2" in c]]],
                                f4(df.columns)))
        print("\n## B: (i) the enclosed population left (area share), (ii) Lens A over the "
              "non-enclosed at eps 0; today's stored Lens A\n")
        print(insulator._md(lt.b, f4(lt.b.columns) | {f"Bi_{c}": "{:.5f}" for c in SCENARIOS}
                            | {"Bii_minus_today_simp": "{:.1e}",
                               "Bii_minus_today_cheap": "{:.1e}"}))
        print("\n## simp_default against cheap: d = simp - cheap (Lens A; d_Bi: enclosed "
              "share left, lower first); lead: which comes first (B: (i), then (ii))\n")
        print(insulator._md(lt.order, f4(lt.order.columns) | {"d_Bi": "{:.5f}"}))
        for k in ("today", "A", "A_gate", "Bii", "B"):
            print(f"\n{k}: " + ", ".join(f"{who} {n}" for who, n in
                                         lt.order[f"lead_{k}"].value_counts().items()))
        if not lt.free.empty:
            print(f"\n## `free` against simp_default: D, and Lens A under A (eps {EPS_A:g}) and "
                  f"at eps_gate (G)\n")
            print(insulator._md(lt.free, f4(lt.free.columns)))
        if not lt.leak.empty:
            print("\n## The sentinels' leak: Lens A over every home at eps 0, eps_gate and "
                  f"{EPS_A:g}\n")
            print(insulator._md(lt.leak, f4(lt.leak.columns)))
        lg_ = large_gate(homes)
        if not lg_.empty:
            lg_.to_csv(rows / "summary_walls_large_gate.csv", index=False)
            print(f"\n## The gate on the large blocks with no enclosed home: u(eps) / u(0) with "
                  f"nothing cleared (below = under {GATE}; eps_at_gate: where the worst home "
                  f"would reach {GATE})\n")
            print(insulator._md(lg_, f3(lg_.columns)))
        hn = homes[homes.scenario == "none"]
        ler = enclosed_rows(hn)
        if not ler.empty:
            let = enclosed_table(ler)
            let.to_csv(rows / "summary_walls_large_enclosed.csv", index=False)
            print("\n## Large blocks' enclosed homes: r and z (as the small blocks')\n")
            print(insulator._md(let, f3(let.columns)))
    st = solver_table(small, sc, fails)
    if not st.empty:
        st.to_csv(rows / "summary_walls_solver.csv", index=False)
        print("\n## CG residuals and solve times per world (s)\n")
        print(insulator._md(st, f3(st.columns)))
    if not fails.empty:
        # a small block's failed eps is recovered when `small-exact` has written its rows
        fails["recovered"] = [s_ == "small" and _small_path(b_, e_).exists()
                              for s_, b_, e_ in zip(fails["set"], fails.block, fails.eps,
                                                    strict=True)]
        fails.to_csv(rows / "summary_walls_failed.csv", index=False)
        print("\n## Failures (recorded; `recovered`: re-solved by small-exact)\n")
        print(insulator._md(fails, {"eps": "{:g}"}))


# ------------------------------------------------------------------ main

def _eps_list(s: str) -> tuple[float, ...]:
    eps = tuple(float(x) for x in s.split(","))
    assert all(e >= 0 for e in eps), eps
    return eps


def main(argv: list[str]) -> None:
    cmd = argv[0]
    insulator._check_single_threaded()
    if cmd == "small":
        eps = _eps_list(argv[2])
        ids = insulator._by_size(argv[3].split(","))
        insulator._pool(int(argv[1]), small_one, [(i, _todo_amg(i, eps), AMG) for i in ids])
    elif cmd == "small-exact":
        ids = insulator._by_size(argv[2].split(","))
        insulator._pool(int(argv[1]), small_one,
                        [(i, t, EXACT) for i in ids if (t := _todo_exact(i))])
    elif cmd == "clearings":
        write_clearings(Path(argv[1]), argv[2].split(","))
    elif cmd == "large":
        large(argv[1], _eps_list(argv[2]), argv[3].split(","))
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
