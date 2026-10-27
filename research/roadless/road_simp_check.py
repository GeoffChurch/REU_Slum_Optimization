"""road_simp's guards, each run both ways (as is, and under a fault that must fail it).

grad: P's derivative in the design per pixel (the adjoint through the openness and the boost, the
projection and the filter) against central differences at the twelve pixels of largest |dP|, on a
random design at q 3, beta 4, no clearing, at step H_P; within 1e-4 relative of the larger.
Faults: `no-boost` (the boost's term dropped), `no-open` (the openness's).
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

The joint clearing's (road_simp.Joint: x per pixel, y per building, every building free), each on a
random design (x and y uniform on 0.2 .. 0.8) at q 3, beta 4, against central differences at the
twelve pixels and the twelve buildings of largest |derivative|, within 1e-4 relative:
jgrad: dP/dx and dP/dy (x at step H_P). Faults: `no-clear-s` (the clearing path's (1 - s)
dropped), `kt0` (the paving path's (1 - kt_c) read as (1 - kt0), the openness of the world with
nothing cleared).
jdD: dD/dx and dD/dy (the union of the roads' cover and the clearing), x as dD's (buildings past
the knee and under). Fault: `sum` (the union's derivative taken as a sum's: neither factor the
other leaves).
jdC: dC/dx and dC/dy (SheetPrice.charge at lambda 1). Fault: `no-kt-db` (dC/ds's kt db term
dropped).
jsheet: at a binary design (x 0 / 1 on 16-pixel squares, beta BIG so every triangle's density is
0 or 1; y 1 on the LISTED largest footprints, else 0), C from the eps world against Model.sheet of
the exact state (the clearing's Model.factors, its paved triangles open and boosted as a road's)
less the baseline's, within the eps world's own distance from the exact at nothing paved or
cleared; the clearing and the paving each moving C by more than ten times that. Fault:
`not-applied` (the eps world's design without the clearing).

    C=research/roadless/road_simp_check.py             # from the repo root, PYTHONPATH=.
    uv run python $C grad|dD|adaptive|clearing|jgrad|jdD|jdC|jsheet <block> [fault]
"""
from __future__ import annotations

import sys
from collections.abc import Callable
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
H_P = 1e-3               # grad, jgrad: P's step in x (P is solved to about 1e-12 relative, which
                         # at H is up to 2e-4 of a derivative in x on 19570; at H_P 1e-7, the
                         # truncation's); in y H (the width form's terms are a minimum of affine
                         # functions, whose kinks H_P straddles: 5e-3; y's derivatives are larger,
                         # the solve's noise 5e-6 of them at H)
TOL_FP = 1e-3            # adaptive: the fixed point's and the slope's tolerance
FP_STEPS = 200           # adaptive: updates (MOVE 0.2 from XMIN: a few dozen reach it)
TOL_CLEAR = 2e-2         # clearing: the eps world's P against the exact clearing's, relative
                         # (the eps world's own distance from the exact model: 1.6e-2 at no
                         # clearing on 19570, EPS through every standing footprint)
LISTED = 20              # clearing, jsheet: the footprints cleared, the largest
BIG = 1e6                # jsheet: the projection's beta (every density 0 or 1)
F = NDArray[np.float64]


def model(bid: str):  # type: ignore[no-untyped-def]
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    return metric.clearing(b, demand=common.PopulationDemand())


def roads(m, *, cleared: NDArray[np.bool_] | None = None,  # type: ignore[no-untyped-def]
          real: bool = False, joint: bool = False) -> rs.Roads:
    none = np.zeros(m.n, dtype=bool)
    return rs.Roads(m, radius=3.5, cleared=none if cleared is None else cleared,
                    real=real or joint, free=~none if joint else none)


def _largest(m) -> NDArray[np.bool_]:  # type: ignore[no-untyped-def]
    big = np.argsort(-shapely.area(np.asarray(m.geo.polys)))[:LISTED]
    out = np.zeros(m.n, dtype=bool)
    out[big] = True
    return out


def design(rd: rs.Roads) -> F:
    x = np.random.default_rng(0).uniform(0.2, 0.8, len(rd.raster.on))
    return np.where(rd.raster.on, x, rs.XMIN)


def coarse(rd: rs.Roads, lo: float, hi: float, seed: int) -> F:
    """x uniform on lo .. hi per 16 pixels (dD's: some buildings past the ramp's knee, some
    under)."""
    nx, ny = rd.raster.shape
    c = np.random.default_rng(seed).uniform(lo, hi, (nx // 16 + 1, ny // 16 + 1))
    return np.where(rd.raster.on, np.kron(c, np.ones((16, 16)))[:nx, :ny].ravel(), rs.XMIN)


def grad(bid: str, fault: str | None) -> None:
    rd = roads(model(bid))
    x, y = design(rd), np.full(rd.m.n, rs.XMIN)

    def P_of(x: F) -> tuple[float, F]:
        w = rd.world(x, y, Q, BETA)
        P, gk, gb = rd.split(w)
        gs = rd.P_grad(w)[1]
        if fault is not None:                # the term the fault drops, by the chain rule's split
            gs = gs - (gb * rd.db if fault == "no-boost" else gk * (1.0 - w.kt_c))
        return P, rd.back(w.xf, BETA, gs * Q * np.maximum(w.r, 1e-300) ** (Q - 1.0))

    _, g = P_of(x)
    _check(x, g, lambda z: P_of(z)[0], "dP", h=H_P)


def dD(bid: str, fault: str | None) -> None:
    m = model(bid)
    # with no clearing, then with a real one (its buildings saturated: their slope 0)
    for rd in (roads(m), roads(m, cleared=_largest(m), real=True)):
        x, y = coarse(rd, 0.1, 0.5, 1), np.full(m.n, rs.XMIN)
        w = rd.world(x, y, 1.0, BETA)
        X = np.maximum(rd.cover @ w.r, w.c)
        print(f"{int(rd.cleared.sum())} cleared (real {rd.real}): buildings past the knee: "
              f"{(X >= common.MEASURE.tau).mean():.2f}")
        gr = rd.dD(w)[0]
        if fault == "flat":
            gr = np.asarray(rd.cover.T @ rd.w) / common.MEASURE.tau
        g = rd.back(w.xf, BETA, gr)
        _check(x, g, lambda z, rd=rd, y=y: rd.D(rd.chain(z, BETA)[1], rd.share(y, BETA)), "dD",
               h=H)


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
    m = model(bid)
    rd0 = roads(m)
    cleared = _largest(m)
    rd = roads(m, cleared=None if fault == "not-applied" else cleared)
    x0, y0 = np.zeros(len(rd.raster.on)), np.full(m.n, rs.XMIN)       # x 0: r 0, no road
    P_eps, P_eps0 = rd.split(rd.world(x0, y0, 1.0, BETA))[0], rd0.split(
        rd0.world(x0, y0, 1.0, BETA))[0]
    P_exact = m.lenses(cleared).P
    off = abs(P_eps / P_exact - 1.0)
    moved = abs(P_exact / m.base.P - 1.0)
    ok = off <= TOL_CLEAR and moved > 10 * TOL_CLEAR
    print(f"clearing {bid}: {LISTED} footprints cleared; the exact P moves {moved:.2e} from the "
          f"baseline's, the eps world's {abs(P_eps0 / m.base.P - 1.0):.1e} off it at none; with "
          f"the clearing the eps world's P is the exact's to {off:.1e} -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


# -- the joint clearing's


def _joint(bid: str, x_of: Callable[[rs.Roads], F]) -> tuple[rs.Roads, F, F]:
    rd = roads(model(bid), joint=True)
    y = np.random.default_rng(3).uniform(0.2, 0.8, rd.m.n)
    return rd, x_of(rd), y


def _both(rd: rs.Roads, x: F, y: F, g: F, f: Callable[[F, F], float], what: str, *,
          hx: float, hy: float) -> None:
    """`_check` on the pixels (y held) and on the buildings (x held); fails if either does."""
    npx = len(x)
    ok = _check(x, g[:npx], lambda z: f(z, y), f"{what}/dx", h=hx, final=False)
    ok &= _check(y, g[npx:], lambda z: f(x, z), f"{what}/dy", h=hy, final=False)
    print(f"{what}: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def jgrad(bid: str, fault: str | None) -> None:
    rd, x, y = _joint(bid, design)
    w = rd.world(x, y, Q, BETA)
    _, gs, gkc = rd.P_grad(w)
    _, gk, _ = rd.split(w)
    if fault == "no-clear-s":
        gkc = gk
    elif fault == "kt0":
        kt0 = np.asarray(rd.m.terms.value(np.full(rd.m.n, rs.EPS)))
        gs = gs + gk * (w.kt_c - kt0)
    g_r = gs * Q * np.maximum(w.r, 1e-300) ** (Q - 1.0)
    g = np.concatenate([rd.back(w.xf, BETA, g_r), rd.to_y(w, gkc)])
    moved = np.abs(w.kt_c - rd.m.terms.value(np.full(rd.m.n, rs.EPS))) > 1e-6
    print(f"kt_c - kt0 on {int(moved.sum())} of {len(w.kt_c)} triangles")
    _both(rd, x, y, g, lambda x, y: rd.split(rd.world(x, y, Q, BETA))[0], "dP", hx=H_P,
          hy=H)


def jdD(bid: str, fault: str | None) -> None:
    rd, x, y = _joint(bid, lambda rd: coarse(rd, 0.1, 0.5, 1))
    w = rd.world(x, y, Q, BETA)
    X = rd.cover @ w.r
    print(f"buildings past the knee: {(X >= common.MEASURE.tau).mean():.2f}; clearing share "
          f"{w.c.min():.2f} .. {w.c.max():.2f}")
    dDr, dDy = rd.dD(w)
    if fault == "sum":
        dDr = np.asarray(rd.cover.T @ (rd.w * (X < common.MEASURE.tau) / common.MEASURE.tau))
        dDy = rd.w * w.dpy
    g = np.concatenate([rd.back(w.xf, BETA, dDr), dDy])
    _both(rd, x, y, g, lambda x, y: rd.D(rd.chain(x, BETA)[1], rd.share(y, BETA)), "dD", hx=H, hy=H)


def jdC(bid: str, fault: str | None) -> None:
    rd, x, y = _joint(bid, design)
    w = rd.world(x, y, Q, BETA)
    unit = rs.SheetPrice(price=rs.SHEET_M2 / rd.P0)          # lambda 1: the charge is C itself
    ch = unit.charge(rd, w, rs.OC())
    d_r = ch.d_r
    if fault == "no-kt-db":
        d_r = d_r - rd.area * w.kt * rd.db * Q * np.maximum(w.r, 1e-300) ** (Q - 1.0)
    print(f"C {rd.C(w):.1f} m^2 (charge {ch.cost:.1f}), the clearing's own {rd.C_clear(w):.1f}")
    g = np.concatenate([rd.back(w.xf, BETA, d_r), rd.to_y(w, ch.d_kt)])
    _both(rd, x, y, g, lambda x, y: rd.C(rd.world(x, y, Q, BETA)), "dC", hx=H, hy=H)


def jsheet(bid: str, fault: str | None) -> None:
    rd = roads(model(bid), joint=True)
    m = rd.m
    cleared = _largest(m)
    x = np.where(coarse(rd, 0.0, 1.0, 4) > 0.7, 1.0, 0.0)
    y = np.where(cleared & (fault != "not-applied"), 1.0, 0.0)
    w = rd.world(x, y, 1.0, BIG)
    grey = float(np.minimum(w.r, 1.0 - w.r).max())
    paved = w.r > 0.5
    # the exact state: the clearing's own factors (Model's path), its paved triangles a road's
    st = m.factors(m.k_of(cleared))
    st = st._replace(kt=np.where(paved, 1.0, st.kt),
                     b=np.where(paved, np.maximum(st.b, m.exits.boost.road), st.b))
    C_exact = m.sheet(st) - rd.base
    C_clear = m.sheet(m.factors(m.k_of(cleared))) - rd.base
    none = rd.world(np.zeros(len(x)), np.zeros(m.n), 1.0, BIG)
    own = abs(rd.C(none))                                    # the exact's C at nothing is 0
    off = abs(rd.C(w) - C_exact)
    ok = grey == 0.0 and off <= own and C_clear > 10 * own and C_exact - C_clear > 10 * own
    print(f"jsheet {bid}: {int(paved.sum())} of {len(paved)} triangles paved (grey {grey:.1e}), "
          f"{LISTED} footprints cleared: C exact {C_exact:.3f} m^2 (the clearing's {C_clear:.3f}), "
          f"eps world {rd.C(w):.3f}, off by {off:.3e}; the eps world's own at nothing {own:.3e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def _check(x: F, g: F, f: Callable[[F], float], what: str, *, h: float,
           final: bool = True) -> bool:
    worst = 0.0
    for j in np.argsort(-np.abs(g))[:N]:
        hi, lo = x.copy(), x.copy()
        hi[j] += h
        lo[j] -= h
        fd = (f(hi) - f(lo)) / (2 * h)
        rel = abs(fd - g[j]) / max(abs(fd), abs(g[j]), 1e-300)
        worst = max(worst, rel)
        print(f"{what} {j}: adjoint {g[j]:.6e}, differences {fd:.6e}, rel {rel:.1e}")
    ok = worst <= TOL
    print(f"{what}: worst rel {worst:.1e} -> {'PASS' if ok else 'FAIL'}")
    if final and not ok:
        raise SystemExit(1)
    return ok


if __name__ == "__main__":
    verb, bid, *fault = sys.argv[1:]
    {"grad": grad, "dD": dD, "adaptive": adaptive, "clearing": clearing, "jgrad": jgrad,
     "jdD": jdD, "jdC": jdC, "jsheet": jsheet}[verb](bid, fault[0] if fault else None)
