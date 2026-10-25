"""road_simp's guards, each run both ways (as is, and under a fault that must fail it).

grad: P's derivative in the design per pixel (the adjoint through the openness and the boost, the
projection and the filter) against central differences at the twelve pixels of largest |dP|, on a
random design at q 3, beta 4; within 1e-4 relative of the larger. Faults: `no-boost` (the
boost's term dropped), `no-open` (the openness's).
dD: D's derivative the same way (the ramp's slope through the covered shares), at the twelve
pixels of largest |dD| on a design that puts some buildings past the ramp's knee and some under.
Fault: `flat` (the ramp's slope 1 / tau everywhere: the saturated buildings read as moving).

    PYTHONPATH=. uv run python research/roadless/road_simp_check.py grad|dD <block> [fault]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import metrics  # noqa: E402
import road_simp as rs  # noqa: E402

Q, BETA, H, TOL, N = 3.0, 4.0, 1e-5, 1e-4, 12


def roads(bid: str) -> rs.Roads:
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    return rs.Roads(metric.clearing(b, demand=common.PopulationDemand()), radius=3.5)


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
    rd = roads(bid)
    # some buildings past the ramp's knee, some under: a field uniform on 0.1 .. 0.5 per 16 pixels
    nx, ny = rd.raster.shape
    coarse = np.random.default_rng(1).uniform(0.1, 0.5, (nx // 16 + 1, ny // 16 + 1))
    x = np.where(rd.raster.on, np.kron(coarse, np.ones((16, 16)))[:nx, :ny].ravel(), rs.XMIN)
    xf, r = rd.chain(x, BETA)
    X = rd.cover @ r
    print(f"buildings past the knee: {(X >= common.MEASURE.tau).mean():.2f}")
    gr = rd.dD(r)
    if fault == "flat":
        gr = np.asarray(rd.cover.T @ rd.w) / common.MEASURE.tau
    g = rd.back(xf, BETA, gr)
    _check(x, g, lambda y: rd.D(rd.chain(y, BETA)[1]), "dD")


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
    {"grad": grad, "dD": dD}[verb](bid, fault[0] if fault else None)
