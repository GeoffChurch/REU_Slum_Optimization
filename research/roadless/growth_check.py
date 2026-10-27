"""growth's guard, run both ways (as is, and under a fault that must fail it).

bridge: the bridge's rank-one gain (growth.rank_one: g (u_a - u_b)^2 / (1 + g R_ab), R_ab by
Growth.resistance's dipole solve) against a re-solve, on the block's eps world with nothing grown,
g = 1 / R_ab so the denominator is 2. Two pairs: the two cells of the passage carrying the largest
potential difference between open cells (the pattern holds the pair: the modified operator's values written into the
pattern and refactored by the plan's factorization) and two open cells 15 m apart, the first at
the highest potential among open cells (the pattern does not hold them: the modified system A + g
e e^T re-solved by conjugate gradients preconditioned by A's factor to CG_TOL, which a rank-one
change converges in two iterations). P's drop from the formula within TOL of the re-solve's at
both. Fault: `no-denominator` (g du^2, the R 0 bound the screen ranks by: off by a factor 2).

    C=research/roadless/growth_check.py                # from the repo root, PYTHONPATH=.
    uv run python $C bridge <block> [no-denominator]
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import growth as gr  # noqa: E402
import metrics  # noqa: E402

TOL = 1e-6               # relative: the re-solves' own error is about 1e-10
CG_TOL = 1e-13           # the conjugate gradients' relative residual
CG_ITERS = 50
FAR_M = 15.0             # m: the far pair's distance
F: TypeAlias = NDArray[np.float64]


def _slot(indptr: NDArray[np.int32], indices: NDArray[np.int32], i: int, j: int) -> int:
    """The position of entry (i, j) in the pattern."""
    row = indices[indptr[i]:indptr[i + 1]]
    k = np.flatnonzero(row == j)
    if len(k) != 1:
        raise AssertionError(f"({i}, {j}) not in the pattern")
    return int(indptr[i] + k[0])


def bridge(bid: str, fault: str | None) -> None:
    if fault not in (None, "no-denominator"):
        raise SystemExit(f"{fault}: the fault is no-denominator")
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    g = gr.growth_of(b, metric, contrast=gr.Road(kappa=10.0))
    m = g.m
    st = g.state(np.zeros(g.nt, dtype=bool))
    u, P = g.solve(st)
    assert st.A is not None and st.factor is not None
    unk = st.reach.unknown
    rhs = m._rhs(st, m.pop)
    c = m.c
    ea, eb = c.ea.astype(np.int64), c.eb.astype(np.int64)
    # open cells (an open triangle's): a drop well above P's cancellation (P - P2 at P's 1e-13)
    cell_open = np.bincount(c.cell, weights=g.kt0 > 0.5, minlength=c.n) > 0
    ok_cell = unk & cell_open
    cand = np.flatnonzero(ok_cell[ea] & ok_cell[eb] & (ea != eb))
    q = cand[np.argmax(np.abs(u[ea[cand]] - u[eb[cand]]))]
    open_t = np.flatnonzero((g.kt0 > 0.5) & unk[c.cell])
    t1 = open_t[np.argmax(u[c.cell[open_t]])]
    t2 = open_t[np.argmin(np.abs(np.hypot(*(g.cen[open_t] - g.cen[t1]).T) - FAR_M))]
    worst = 0.0
    for name, (i, j) in (("passage", (int(ea[q]), int(eb[q]))),
                         (f"{FAR_M:g} m", (int(c.cell[t1]), int(c.cell[t2])))):
        R = g.resistance(st, i, j)
        gab = 1.0 / R
        du = float(u[i] - u[j])
        est = gab * du * du if fault == "no-denominator" else gr.rank_one(gab, du, R)
        if name == "passage":
            p = m.sys.graph.pattern
            data = st.A.data.copy()
            if not (np.array_equal(st.A.indptr, p.indptr)
                    and np.array_equal(st.A.indices, p.indices)):
                raise AssertionError("the state's operator is not on the pattern")
            for (r, s), v in (((i, i), gab), ((j, j), gab), ((i, j), -gab), ((j, i), -gab)):
                data[_slot(p.indptr, p.indices, r, s)] += v
            x = m.sys.planned.factor(data, unk).solve(rhs)
        else:
            x = _cg(st, unk, rhs, i, j, gab)
        exact = P - float(rhs @ x)
        rel = abs(est - exact) / abs(exact)
        worst = max(worst, rel)
        print(f"bridge {name}: cells {i}, {j}, R {R:.6e}, du {du:.6e}: rank one {est:.9e}, "
              f"re-solve {exact:.9e}, rel {rel:.1e}", flush=True)
    ok = worst <= TOL
    print(f"bridge {bid} {fault or 'none'}: worst rel {worst:.1e} (tol {TOL:g}) -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        raise SystemExit(1)


def _cg(st: Any, unk: NDArray[np.bool_], rhs: F, i: int, j: int, gab: float) -> F:
    """(A + g e e^T) x = rhs on the unknowns, e = e_i - e_j, by conjugate gradients preconditioned
    by A's factor."""
    n = len(unk)
    Au = st.A[unk][:, unk]
    e = np.zeros(n)
    e[i], e[j] = 1.0, -1.0
    ee = e[unk]

    def op(v: F) -> F:
        return np.asarray(Au @ v + gab * ee * float(ee @ v), dtype=np.float64)

    def pre(v: F) -> F:
        z = np.zeros(n)
        z[unk] = v
        return np.asarray(st.factor.solve(z)[unk], dtype=np.float64)

    bb = rhs[unk]
    xk = pre(bb)
    r = bb - op(xk)
    z = pre(r)
    p = z.copy()
    for _ in range(CG_ITERS):
        if np.linalg.norm(r) <= CG_TOL * np.linalg.norm(bb):
            break
        Ap = op(p)
        al = float(r @ z) / float(p @ Ap)
        xk = xk + al * p
        r2 = r - al * Ap
        z2 = pre(r2)
        be = float(r2 @ z2) / float(r @ z)
        r, z, p = r2, z2, z2 + be * p
    else:
        raise AssertionError(f"conjugate gradients: residual {np.linalg.norm(r):.3e} after "
                             f"{CG_ITERS}")
    x = np.zeros(n)
    x[unk] = xk
    return x


if __name__ == "__main__":
    verb, bid, *fault = sys.argv[1:]
    {"bridge": bridge}[verb](bid, fault[0] if fault else None)
