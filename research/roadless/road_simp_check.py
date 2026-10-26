"""road_simp's guards, each run both ways (as is, and under a fault that must fail it).

grad: P's derivative in the design per pixel (the adjoint through the openness and the boost, the
projection and the filter) against central differences at the twelve pixels of largest |dP|, on a
random design at q 3, beta 4; within 1e-4 relative of the larger. Faults: `no-boost` (the
boost's term dropped), `no-open` (the openness's).
dD: D's derivative the same way (the ramp's slope through the covered shares), at the twelve
pixels of largest |dD| on a design that puts some buildings past the ramp's knee and some under,
with no clearing and with a real one (the LISTED largest footprints: saturated, their slope 0).
Fault: `flat` (the ramp's slope 1 / tau everywhere: the saturated buildings read as moving).
adaptive: Adaptive(gamma)'s update (road_simp._update with its paving and eta) iterated on a
planted road-dominated gain, -dP/dr = C / r^2 per pixel (C log-uniform over two decades, |j| its
square root), no budget binding: the density it settles on is the fixed point (C / (price gamma))^
(1/(1+gamma)) to TOL_FP and its log-log slope against |j| is 2 / (1 + gamma) to TOL_FP, at gamma 0.5
and 1.5. Fault: `cost-linear` (the paving cost's exponent dropped, OC's linear cost: slope 1).
clearing: the starting world: with the LISTED largest footprints cleared on the block, the eps
world's P at no road is the exact model's P of that clearing (gap_clear.Model.lenses) to TOL_CLEAR,
the clearing moving P by more than ten times that. Fault: `not-applied` (the eps world built
without it).

    PYTHONPATH=. uv run python research/roadless/road_simp_check.py grad|dD|adaptive|clearing <block> [fault]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import shapely
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import metrics  # noqa: E402
import road_simp as rs  # noqa: E402

Q, BETA, H, TOL, N = 3.0, 4.0, 1e-5, 1e-4, 12
TOL_FP = 1e-3            # adaptive: the fixed point's and the slope's tolerance
FP_STEPS = 200           # adaptive: updates (MOVE 0.2 from XMIN: a few dozen reach it)
TOL_CLEAR = 2e-2         # clearing: the eps world's P against the exact clearing's, relative
                         # (the eps world's own distance from the exact model: 1.6e-2 at no
                         # clearing on 19570, EPS through every standing footprint)
LISTED = 20              # clearing: the footprints cleared, the largest


def roads(bid: str, *, cleared: NDArray[np.bool_] | None = None,
          real: bool = False) -> rs.Roads:
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    m = metric.clearing(b, demand=common.PopulationDemand())
    return rs.Roads(m, radius=3.5, cleared=np.zeros(m.n, dtype=bool) if cleared is None
                    else cleared, real=real)


def _largest(rd: rs.Roads) -> NDArray[np.bool_]:
    big = np.argsort(-shapely.area(np.asarray(rd.m.geo.polys)))[:LISTED]
    out = np.zeros(rd.m.n, dtype=bool)
    out[big] = True
    return out


def design(rd: rs.Roads) -> NDArray[np.float64]:
    x = np.random.default_rng(0).uniform(0.2, 0.8, len(rd.raster.on))
    return np.where(rd.raster.on, x, rs.XMIN)


def grad(bid: str, fault: str | None) -> None:
    rd = roads(bid)
    x = design(rd)

    def P_of(x: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        xf, r = rd.chain(x, BETA)
        P, gs = rd.P_grad(r ** Q)
        if fault is not None:                # the term the fault drops, by the chain rule's split
            gk, gb = _split(rd, r ** Q)
            gs = gs - (gb * rd.db if fault == "no-boost" else gk * (1.0 - rd.kt0))
        return P, rd.back(xf, BETA, gs * Q * np.maximum(r, 1e-300) ** (Q - 1.0))

    _, g = P_of(x)
    _check(x, g, lambda y: P_of(y)[0], "dP")


def _split(rd: rs.Roads, s: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    m = rd.m
    st = rd.state(s)
    u = m.solve(st, m.pop)
    c, fm = st.sys.cells, st.fm
    gk, gb = rs._grad(c.pe, c.ea, c.eb, fm.pa, fm.pb, fm.T, st.kt, st.kp, rd.bw + s * rd.db,
                      c.pf, c.fc, fm.fa, fm.Tf, u, fm.nt)
    return gk, gb


def dD(bid: str, fault: str | None) -> None:
    rd0 = roads(bid)
    # with no clearing, then with a real one (its buildings saturated: their slope 0)
    for rd in (rd0, roads(bid, cleared=_largest(rd0), real=True)):
        # some buildings past the ramp's knee, some under: uniform on 0.1 .. 0.5 per 16 pixels
        nx, ny = rd.raster.shape
        coarse = np.random.default_rng(1).uniform(0.1, 0.5, (nx // 16 + 1, ny // 16 + 1))
        x = np.where(rd.raster.on, np.kron(coarse, np.ones((16, 16)))[:nx, :ny].ravel(), rs.XMIN)
        xf, r = rd.chain(x, BETA)
        X = rd.covered(r)
        print(f"{int(rd.cleared.sum())} cleared (real {rd.real}): buildings past the knee: "
              f"{(X >= common.MEASURE.tau).mean():.2f}")
        gr = rd.dD(r)
        if fault == "flat":
            gr = np.asarray(rd.cover.T @ rd.w) / common.MEASURE.tau
        g = rd.back(xf, BETA, gr)
        _check(x, g, lambda y, rd=rd: rd.D(rd.chain(y, BETA)[1]), "dD")


def adaptive(_bid: str, fault: str | None) -> None:
    rng = np.random.default_rng(2)
    n, price = 400, 1.0
    C = 10.0 ** rng.uniform(-3.0, -1.0, n)           # -dP/dr = C / r^2: |j| ~ sqrt(C)
    on = np.ones(n, dtype=bool)
    worst = 0.0
    for gamma in (0.5, 1.5):
        up = rs.Adaptive(gamma=gamma)
        paving = (lambda p, a, r: p * a) if fault == "cost-linear" else up.paving
        x = np.full(n, rs.XMIN)
        for _ in range(FP_STEPS):
            x, _nu = rs._update(x, on, C / x ** 2, paving(price, np.ones(n), x), np.zeros(n),
                                eta=up.eta, D_of=lambda y: 0.0, budget=1.0, nu0=0.0)
        want = np.clip((C / (price * gamma)) ** (1.0 / (1.0 + gamma)), rs.XMIN, 1.0)
        inner = (want > 2 * rs.XMIN) & (want < 0.5)
        fp = float(np.max(np.abs(x[inner] / want[inner] - 1.0)))
        slope = float(np.polyfit(np.log(np.sqrt(C[inner])), np.log(x[inner]), 1)[0])
        err = abs(slope - 2.0 / (1.0 + gamma))
        worst = max(worst, fp, err)
        print(f"adaptive gamma {gamma:g}: {int(inner.sum())} pixels inside the clip; off the fixed "
              f"point by {fp:.1e}, slope {slope:.4f} (want {2.0 / (1.0 + gamma):.4f})")
    ok = worst <= TOL_FP
    print(f"adaptive: worst {worst:.1e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def clearing(bid: str, fault: str | None) -> None:
    rd0 = roads(bid)
    m = rd0.m
    cleared = _largest(rd0)
    rd = roads(bid, cleared=None if fault == "not-applied" else cleared)
    s0 = np.zeros(m.fx.fm.nt)
    P_eps, P_eps0 = rd.P_grad(s0)[0], rd0.P_grad(s0)[0]
    P_exact = m.lenses(cleared).P
    off = abs(P_eps / P_exact - 1.0)
    moved = abs(P_exact / m.base.P - 1.0)
    ok = off <= TOL_CLEAR and moved > 10 * TOL_CLEAR
    print(f"clearing {bid}: {LISTED} footprints cleared; the exact P moves {moved:.2e} from the "
          f"baseline's, the eps world's {abs(P_eps0 / m.base.P - 1.0):.1e} off it at none; with the "
          f"clearing the eps world's P is the exact's to {off:.1e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def _check(x: NDArray[np.float64], g: NDArray[np.float64], f, what: str) -> None:  # type: ignore[no-untyped-def]
    worst = 0.0
    for j in np.argsort(-np.abs(g))[:N]:
        hi, lo = x.copy(), x.copy()
        hi[j] += H
        lo[j] -= H
        fd = (f(hi) - f(lo)) / (2 * H)
        rel = abs(fd - g[j]) / max(abs(fd), abs(g[j]), 1e-300)
        worst = max(worst, rel)
        print(f"{what} pixel {j}: adjoint {g[j]:.6e}, differences {fd:.6e}, rel {rel:.1e}")
    ok = worst <= TOL
    print(f"{what}: worst rel {worst:.1e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    verb, bid, *fault = sys.argv[1:]
    {"grad": grad, "dD": dD, "adaptive": adaptive, "clearing": clearing}[verb](
        bid, fault[0] if fault else None)
