"""Road-free clearing: remove whole buildings, greedily, by the roadless score.

Tension. Solve once with the remaining buildings as a poor conductor (open fraction eps
wherever a footprint is), so the potential is defined inside them. dP/dw_e = -(u_a - u_b)^2, so
a building's first-order gain from clearing is the sum over the edges it covers of
(Delta u)^2 x (the weight gained). That is the TENSION: how hard the escape flow presses on it.

Greedy. Each step: one tension solve (two for power != 1), then a Picker clears: Screened (exact
solves for the top M by tension, clear the best) or Batched (a spaced set of the top buildings up
to the next multiple of delta of population, one exact solve). `loo` checks the screen against
exact leave-one-out on one step.

    PYTHONPATH=. pixi run python research/roadless/clear.py loo <block idx> [h]
    PYTHONPATH=. pixi run python research/roadless/clear.py run <workers> <M4|B0.01g3> <h> <d_max> [pop] [p]
    PYTHONPATH=. pixi run python research/roadless/clear.py one <block id> <picker> <h> <d_max> <pop> <p>
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402

EPS = 0.01


class Clearing:
    def __init__(self, block, h: float, p: lifted.Params, population=None):
        self.sc = common.Scorer(block, h, p, population=population)
        self.cost = self.sc.w / self.sc.w.sum()        # population share of each building
        self.p = p
        g = self.sc.grid
        # sub-sample -> building index (-1 = none); overlaps go to the later footprint
        self.lab = lab = g.label_sub(self.sc.polys)
        self.n = len(self.sc.polys)
        # per building: its cells and the fraction of each it covers
        ny, nx, S2 = lab.shape
        flat = lab.reshape(-1, S2)
        cells, subs = np.nonzero(flat >= 0)
        b = flat[cells, subs]
        df = pd.DataFrame(dict(b=b, c=cells)).value_counts().reset_index(name="k")
        self.bcells = {int(k): (g2.c.to_numpy(), g2.k.to_numpy() / S2)
                       for k, g2 in df.groupby("b")}
        self.removed = np.zeros(self.n, dtype=bool)

    def open_(self, removed: np.ndarray) -> np.ndarray:
        g = self.sc.grid
        cleared = np.isin(self.lab, np.nonzero(removed)[0])
        return (g.isub & (~g.bsub | cleared)).mean(axis=-1)

    def P(self, removed: np.ndarray) -> float:
        return self.sc.P_free(self.open_(removed))

    def tension(self, power: float = 1.0) -> np.ndarray:
        """First-order gain in J_power of clearing each (remaining) building, from one
        eps-material solve (power 1: the potential is its own adjoint) or two (power != 1: the
        adjoint L lam = dJ/du = power u_i^(power-1) x injection). dJ/dw_e = -(du_e)(dlam_e)."""
        g = self.sc.grid
        op = self.open_(self.removed)
        full = g.isub.mean(axis=-1)                       # everything open
        op_eps = op + EPS * (full - op)
        sol = lifted.solve(g, op_eps, self.sc.f, self.p)
        K = self.p.K
        nc = int((op_eps > 0).sum())
        uk = np.zeros((nc, K))
        live = sol.unk_cell >= 0
        uk[live] = sol.u.reshape(-1, K)
        if power == 1.0:
            lk = uk
        else:
            uh = self.sc.home_u_of(sol, op_eps)
            mult = np.zeros_like(self.sc.f)
            on = self.sc.owner >= 0
            mult[on] = power * np.nan_to_num(uh[self.sc.owner[on]]) ** (power - 1.0)
            lam = lifted.solve(g, op_eps, self.sc.f * mult, self.p)
            lk = np.zeros((nc, K))
            lk[live] = lam.u.reshape(-1, K)
        # per cell: sum over incident edges of (du)^2 x (weight fully open - weight now), halved
        T = np.zeros(nc)
        cur = {k: (a, b, w) for k, a, b, w in lifted.along_edges(op_eps, self.p)}
        for k, a, b, wf in lifted.along_edges(full, self.p):
            ca, cb, wc = cur[k]
            assert len(ca) == len(a) and (ca == a).all()
            gain = (uk[a, k] - uk[b, k]) * (lk[a, k] - lk[b, k]) * (wf - wc)
            np.add.at(T, a, 0.5 * gain)
            np.add.at(T, b, 0.5 * gain)
        _v, _th, _m, gap = lifted.axes(K)
        free_ids = np.arange(nc)
        dvol = (full - op_eps)[op_eps > 0]
        for k in range(K):
            k2 = (k + 1) % K
            T += (uk[free_ids, k] - uk[free_ids, k2]) * (lk[free_ids, k] - lk[free_ids, k2]) * dvol * (
                g.h * g.h / (self.p.ell_m ** 2 * gap[k]))
        # grid cell (row-major over all cells) -> node id
        node = -np.ones(g.inside.size, dtype=np.int64)
        node[np.flatnonzero(op_eps > 0)] = np.arange(nc)
        out = np.zeros(self.n)
        for j, (cells, frac) in self.bcells.items():
            if self.removed[j]:
                continue
            nd = node[cells]
            ok = nd >= 0
            # the cell's tension is shared by the buildings covering it, by covered fraction
            out[j] = float((T[nd[ok]] * frac[ok] / np.maximum(1 - op[np.unravel_index(
                cells[ok], op.shape)], 1e-9)).sum())
        out[self.removed] = -np.inf
        return out


def loo(i: int, h: float, pop: str = "count", power: float = 1.0) -> None:
    from scipy.stats import spearmanr
    blocks = common.build_blocks(common.recipients())
    b = blocks[i]
    c = Clearing(b, h, lifted.Params(ell_m=3.0, K=8), population=common.POPULATIONS[pop])
    sc = c.sc
    P0 = sc.J(sc.u0, power)
    t = time.time()
    T = c.tension(power)
    tt = time.time() - t
    t = time.time()
    exact = np.zeros(c.n)
    for j in range(c.n):
        r = c.removed.copy()
        r[j] = True
        op = c.open_(r)
        exact[j] = P0 - sc.J(sc.home_u_of(lifted.solve(sc.grid, op, sc.f, c.p), op), power)
    te = time.time() - t
    rho = spearmanr(T, exact).statistic
    top = np.argsort(-exact)
    rankT = np.argsort(np.argsort(-T))
    print(f"{b.block_id} n={c.n} h={h} pop={pop} p={power:g}: tension {tt:.0f}s, exact LOO {te:.0f}s "
          f"({te / c.n:.1f}s each)", flush=True)
    print(f"  Spearman(tension, exact gain) {rho:+.3f}; tension rank of the exact top 5: "
          f"{rankT[top[:5]].tolist()}; exact gain top 5 (share of P0): "
          f"{np.round(exact[top[:5]] / P0, 4).tolist()}", flush=True)
    for M in (1, 4, 8, 16):
        pick = np.argsort(-T)[:M]
        print(f"  M={M:2d}: best exact gain among tension top-M = "
              f"{exact[pick].max() / exact.max():.3f} of the true best", flush=True)


class Picked(NamedTuple):
    cleared: list[int]
    J: float          # J_power after clearing them
    P1: float         # P (= J_1) after clearing them


class Picker(Protocol):
    """What one greedy step clears, given the tension per unit population `T` and J_power now."""
    name: str

    def pick(self, c: Clearing, T: np.ndarray, J: float, power: float) -> Picked: ...


def _exact(c: Clearing, cleared: list[int], power: float) -> tuple[float, float]:
    r = c.removed.copy()
    r[cleared] = True
    op = c.open_(r)
    sol = lifted.solve(c.sc.grid, op, c.sc.f, c.p)
    return c.sc.J(c.sc.home_u_of(sol, op), power), sol.P


@dataclass(frozen=True)
class Screened:
    """Exact-solve the top M by tension, clear the one with the best exact gain per unit
    population. M + 1 (or 2) solves per building."""
    M: int

    @property
    def name(self) -> str:
        return f"M{self.M}"

    def pick(self, c: Clearing, T: np.ndarray, J: float, power: float) -> Picked:
        best = Picked([], np.inf, np.inf)
        bestv = -np.inf
        for j in (int(j) for j in np.argsort(-T)[:self.M] if not c.removed[j]):
            Jj, P1 = _exact(c, [j], power)
            v = (J - Jj) / c.cost[j]
            if v > bestv:
                best, bestv = Picked([j], Jj, P1), v
        return best


@dataclass(frozen=True)
class Batched:
    """Clear, by tension per unit population, every building not within `gap_m` of one already
    taken this round, until D reaches the next multiple of `delta` (so a lens threshold on that
    lattice is overshot by at most one building, as in the one-at-a-time greedy). One exact
    solve per round, for the record; no screen."""
    delta: float
    gap_m: float

    @property
    def name(self) -> str:
        return f"B{self.delta:g}g{self.gap_m:g}"

    def pick(self, c: Clearing, T: np.ndarray, J: float, power: float) -> Picked:
        D = float(c.cost[c.removed].sum())
        target = (np.floor(D / self.delta + 1e-9) + 1) * self.delta
        taken: list[int] = []
        near: set[int] = set()
        for j in (int(j) for j in np.argsort(-T) if not c.removed[j]):
            if D >= target - 1e-12:
                break
            if j in near:
                continue
            taken.append(j)
            D += float(c.cost[j])
            near.update(int(k) for k in c.sc.tree.query(c.sc.polys[j], predicate="dwithin",
                                                         distance=self.gap_m))
        return Picked(taken, *_exact(c, taken, power))


def greedy_block(b, picker: Picker, h: float, d_max: float, out: Path, population,
                 power: float = 1.0) -> None:
    """Each step: rank remaining buildings by tension (in J_power) per unit of population and
    let `picker` clear some. Records perm (in J_power) and perm1 (the p = 1 score) at every
    step; `cleared` lists the step's buildings in pick order."""
    c = Clearing(b, h, lifted.Params(ell_m=3.0, K=8), population=population)
    sc = c.sc
    J0 = sc.J(sc.u0, power)
    P0 = sc.P0
    J = J0
    rows = [dict(block=b.block_id, n=c.n, step=0, D=0.0, perm=0.0, perm1=0.0, cleared=[],
                 P0=P0, t=0.0)]
    step = 0
    t0 = time.time()
    while c.cost[c.removed].sum() < d_max - 1e-12 and not c.removed.all():
        pk = picker.pick(c, c.tension(power) / c.cost, J, power)
        c.removed[pk.cleared] = True
        J = pk.J
        step += 1
        rows.append(dict(block=b.block_id, n=c.n, step=step, D=float(c.cost[c.removed].sum()),
                         perm=1 - (pk.J / J0) ** (1 / power), perm1=1 - pk.P1 / P0,
                         cleared=pk.cleared, P0=P0, t=time.time() - t0))
        if c.n > 1000:
            print(f"  {b.block_id} step {step} D {rows[-1]['D']:.3f} perm' {rows[-1]['perm']:.3f}"
                  f" {rows[-1]['t']:.0f}s", flush=True)
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    pd.DataFrame(rows).to_parquet(tmp)
    os.replace(tmp, out)
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} n={c.n} {step} steps, perm' at end "
          f"{rows[-1]['perm']:.3f}  {time.time() - t0:.0f}s", flush=True)


_BLOCKS: list = []
_CFG: dict = {}


def _one(i: int) -> None:
    b = _BLOCKS[i]
    out = _CFG["dir"] / f"{b.block_id}.parquet"
    if out.exists():
        return
    try:
        greedy_block(b, _CFG["picker"], _CFG["h"], _CFG["d_max"], out,
                     common.POPULATIONS[_CFG["pop"]], _CFG["power"])
    except Exception as e:
        print(f"{b.block_id} FAILED {type(e).__name__}: {e}"[:300], flush=True)


def cleared_through(g: pd.DataFrame, step: int) -> np.ndarray:
    """Buildings cleared in steps 1..step of one block's greedy rows, in clearing order."""
    g = g[(g.step >= 1) & (g.step <= step)].sort_values("step")
    return np.array([j for js in g.cleared for j in js], dtype=np.int64)


def picker_of(spec: str) -> Picker:
    """`M4` -> Screened(4); `B0.01g3` -> Batched(delta 0.01, gap 3 m)."""
    if spec.startswith("M"):
        return Screened(int(spec[1:]))
    if spec.startswith("B"):
        delta, gap = spec[1:].split("g")
        return Batched(float(delta), float(gap))
    raise ValueError(f"unknown picker {spec!r}")


def rows_dir(picker: str, h: float, pop: str, power: float = 1.0) -> Path:
    return HERE / (f"clear_rows_{picker}_h{h:g}" + ("" if pop == "count" else f"_{pop}")
                   + ("" if power == 1.0 else f"_p{power:g}"))


def run(workers: int, picker: Picker, h: float, d_max: float, pop: str, power: float) -> None:
    import multiprocessing
    global _BLOCKS, _CFG
    d = rows_dir(picker.name, h, pop, power)
    d.mkdir(exist_ok=True)
    _CFG = dict(picker=picker, h=h, d_max=d_max, dir=d, pop=pop, power=power)
    _BLOCKS = common.build_blocks(common.recipients())
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for _ in pool.imap_unordered(_one, list(range(len(_BLOCKS)))[::-1]):
            pass


if __name__ == "__main__":
    if sys.argv[1] == "loo":
        loo(int(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.5,
            sys.argv[4] if len(sys.argv) > 4 else "count",
            float(sys.argv[5]) if len(sys.argv) > 5 else 1.0)
    elif sys.argv[1] == "one":
        pk = picker_of(sys.argv[3])
        pop, power = sys.argv[6], float(sys.argv[7])
        d = rows_dir(pk.name, float(sys.argv[4]), pop, power)
        d.mkdir(exist_ok=True)
        [blk] = common.build_blocks([sys.argv[2]])
        greedy_block(blk, pk, float(sys.argv[4]), float(sys.argv[5]),
                     d / f"{blk.block_id}.parquet", common.POPULATIONS[pop], power)
    elif sys.argv[1] == "run":
        run(int(sys.argv[2]), picker_of(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]),
            sys.argv[6] if len(sys.argv) > 6 else "count",
            float(sys.argv[7]) if len(sys.argv) > 7 else 1.0)
