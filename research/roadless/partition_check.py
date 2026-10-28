"""partition's guards, each run both ways (as is, and under a fault that must fail it).

grad: d(P + lambda C)/dpsi through the whole chain (the filter, the softmax, the normalized
interface density and its clip, the projection, SIMP's share, the adjoint) against central
differences at the twelve entries of largest |derivative| and the twelve largest on pixels where
the density is clipped at 1 (8% of them), on a design of the Voronoi start (4 cells: 19570 is
16,600 m^2) with uniform noise of NOISE on every field, at T 1.5 T_w, beta 4, q 3, price 0.3;
step H_P (P is solved to about 1e-12 relative: road_simp_check's H_P); within TOL relative of
the larger. The FFT filter
equals road_simp.Raster.filter to FILTER_TOL first. Faults: `no-offdiag` (the softmax
Jacobian's -phi_k phi_j term dropped), `no-T` (its 1 / T dropped), `unclipped` (the derivative
carried through the clip).
dD: dD/dpsi the same way at step H (D is arithmetic), the noise NOISE_D (with NOISE every building
is past the ramp's knee and D's derivative 0; where the density is clipped its buildings are
covered, so the clip does not reach D's derivative). Fault: `no-norm` (the normalization's
1 / (g_pair - g_pure) dropped).
deadends: the cut answer of the Voronoi start (4 cells, no update) counted on its ground's
skeleton: no dead end and no floating line; the stubs the thinning leaves at junctions are not
dead ends (shorter than a road's width). Fault: `spur` (a SPUR_M road from the middle of the
longest boundary, square to it on the side away from the street, added: one dead end).
mma: partition.Mma on a separable convex problem with a known optimum, min sum c_j / x_j s.t.
sum a_j x_j <= V (x_j proportional to sqrt(c_j / a_j)): the iterate within TOL_MMA of it after
MMA_STEPS. Fault: `no-mu` (the constraint's multiplier held at 0).

    C=research/roadless/partition_check.py             # from the repo root, PYTHONPATH=.
    uv run python $C grad|dD|deadends|mma <block> [fault]
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import metrics  # noqa: E402
import partition as pt  # noqa: E402
import road_simp as rs  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402

N, TOL, H, H_P = 12, 1e-4, 1e-5, 1e-3
NOISE = 0.6              # grad: uniform noise on the start's one-hot fields
NOISE_D = 0.02           # dD: its noise
FILTER_TOL = 1e-10       # grad: the FFT filter against Raster.filter, absolute
SPUR_M = 15.0            # deadends: the planted spur's length
TOL_MMA, MMA_STEPS = 1e-3, 200
F = NDArray[np.float64]


def problem(bid: str):  # type: ignore[no-untyped-def]
    plan = pt.Plan.of("pt.K4.R10.t1.5.b4.q3.i1.D0.05.s0.3.vor4s0")
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    p = pt.partition_of(b, metric, plan)
    f, n = pt.fields_of(p, plan.start)
    return b, metric, plan, p, f


def noisy(p: pt.Partition, f: F, noise: float) -> F:
    g = f + np.random.default_rng(0).uniform(0.0, noise, f.shape)
    return np.where(p.rd.raster.on, np.clip(g, 0.0, 1.0), 0.0)


def grad(bid: str, fault: str | None) -> None:
    _, _, plan, p, f = problem(bid)
    f = noisy(p, f, NOISE)
    off = float(np.abs(p.lab.filter(f)[1] - p.rd.raster.filter(f[1])).max())
    print(f"filter: FFT against Raster.filter {off:.1e} -> "
          f"{'PASS' if off <= FILTER_TOL else 'FAIL'}")
    if off > FILTER_TOL:
        raise SystemExit(1)
    T, beta, q = 1.5 * p.T_w, 4.0, 3.0
    if fault is None:
        g = p.derivs(f, T, beta, q, plan.price)[3]
    else:
        lab = p.lab
        w, phi, raw = p.world(f, T, beta, q)
        _, gs, _ = p.rd.P_grad(w)
        ch = plan.price.charge(p.rd, w, rs.OC())
        g_rho = p.back_r(w, gs * q * np.maximum(w.r, 1e-300) ** (q - 1.0) + ch.d_r)
        lo, hi = pt.g_ends(T, p.K)
        live = (raw > 0.0) & ((raw < 1.0) | (fault == "unclipped"))
        g_phi = np.where(live, g_rho, 0.0) * (-2.0 * phi) / (hi - lo)
        g_L = phi * g_phi if fault == "no-offdiag" else phi * (g_phi - (g_phi * phi).sum(0))
        g = lab.filter_T(g_L if fault == "no-T" else g_L / T)

    def J(z: F) -> float:
        w = p.world(z.reshape(f.shape), T, beta, q)[0]
        return p.rd.P_grad(w)[0] + plan.price.charge(p.rd, w, rs.OC()).cost

    clipped = np.tile(p.world(f, T, beta, q)[2] >= 1.0, p.K)
    print(f"rho clipped at 1 on {clipped[: p.npx][p.rd.raster.on].mean():.3f} of the pixels")
    gv = g.ravel()
    top = np.argsort(-np.abs(gv))[:N]
    at_clip = np.flatnonzero(clipped)[np.argsort(-np.abs(gv[clipped]))[:N]]
    _check(f.ravel(), gv, J, "dJ", h=H_P, at=np.concatenate([top, at_clip]))


def dD(bid: str, fault: str | None) -> None:
    _, _, plan, p, f = problem(bid)
    f = noisy(p, f, NOISE_D)
    T, beta, q = 1.5 * p.T_w, 4.0, 3.0
    w, phi, raw = p.world(f, T, beta, q)
    on = p.rd.raster.on
    X = p.rd.cover @ w.r
    print(f"buildings under the ramp's knee {(X < common.MEASURE.tau).mean():.2f}; rho clipped "
          f"at 1 on {(raw[on] >= 1.0).mean():.3f} of the design pixels")
    g_rho = p.back_r(w, p.rd.dD(w)[0])
    g = p.lab.back(g_rho, phi, raw, T)
    if fault == "no-norm":
        lo, hi = pt.g_ends(T, p.K)
        g = g * (hi - lo)
    _check(f.ravel(), g.ravel(), lambda z: p.D_of(z.reshape(f.shape), T, beta), "dD", h=H)


def deadends(bid: str, fault: str | None) -> None:
    b, _, _, p, f = problem(bid)
    geo = p.rd.m.geo
    cut = shapely.intersection(geo.utm(pt.cut_lines(p, f, 1.0 * p.T_w)), b.boundary)
    parts = shapely.get_parts(shapely.line_merge(shapely.union_all(shapely.get_parts(cut))))
    if fault == "spur":
        L = parts[np.argmax(shapely.length(parts))]
        a = shapely.line_interpolate_point(L, 0.5, normalized=True)
        c = shapely.line_interpolate_point(L, 0.5 + 1.0 / shapely.length(L), normalized=True)
        (ax, ay), (cx, cy) = shapely.get_coordinates(a)[0], shapely.get_coordinates(c)[0]
        t = np.array([cx - ax, cy - ay]) / np.hypot(cx - ax, cy - ay)
        street = shapely.union_all(b.streets.geometry.to_numpy())
        tips = [np.array([ax, ay]) + sgn * SPUR_M * np.array([-t[1], t[0]]) for sgn in (1, -1)]
        tip = max(tips, key=lambda z: shapely.distance(shapely.Point(z), street))
        parts = np.append(parts, shapely.linestrings([[ax, ay], tip]))
    ground = shapely.intersection(
        road_corridor(gpd.GeoDataFrame(geometry=list(parts), crs=b.crs).assign(width_m=pt.ROAD_W)),
        b.boundary)
    _, ends, _ = pt.skeleton(b, ground)
    street = shapely.union_all(b.streets.geometry.to_numpy())
    floating = int((~rs.linked(parts, street)).sum())
    ok = ends == 0 and floating == 0
    print(f"deadends {bid}: the Voronoi start's cut, {len(parts)} lines, "
          f"{shapely.length(parts).sum():.0f} m: {ends} dead ends, {floating} floating -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def mma(_bid: str, fault: str | None) -> None:
    rng = np.random.default_rng(5)
    n = 300
    c, a = rng.uniform(0.5, 2.0, n), rng.uniform(0.5, 2.0, n)
    V = 0.3 * a.sum()
    want = np.sqrt(c / a)
    want *= V / (a @ want)
    x = np.full(n, 0.5)
    m = pt.Mma(n)
    free = np.ones(n, dtype=bool)
    for _ in range(MMA_STEPS):
        def D_of(z: F) -> float:
            return 0.0 if fault == "no-mu" else float(a @ z) - V
        x, _mu, _ = m.step(x, -c / x ** 2, a, free=free, D_of=D_of, budget=0.0, mu0=0.0)
    off = float(np.max(np.abs(x / want - 1.0)))
    ok = off <= TOL_MMA
    print(f"mma: {n} variables, {MMA_STEPS} steps: off the optimum by {off:.1e} (constraint "
          f"{a @ x - V:+.2e}) -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def _check(x: F, g: F, f: Callable[[F], float], what: str, *, h: float,
           at: NDArray[np.int64] | None = None) -> None:
    """Central differences at the entries `at` (the N of largest |g| if None)."""
    worst = 0.0
    for j in np.argsort(-np.abs(g))[:N] if at is None else at:
        hi, lo = x.copy(), x.copy()
        hi[j] += h
        lo[j] -= h
        fd = (f(hi) - f(lo)) / (2 * h)
        rel = abs(fd - g[j]) / max(abs(fd), abs(g[j]), 1e-300)
        worst = max(worst, rel)
        print(f"{what} {j}: adjoint {g[j]:.6e}, differences {fd:.6e}, rel {rel:.1e}")
    ok = worst <= TOL
    print(f"{what}: worst rel {worst:.1e} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    verb, bid, *fault = sys.argv[1:]
    {"grad": grad, "dD": dD, "deadends": deadends, "mma": mma}[verb](
        bid, fault[0] if fault else None)
