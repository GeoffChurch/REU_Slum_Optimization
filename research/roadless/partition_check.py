"""partition's guards, each run both ways (as is, and under a fault that must fail it).

grad: d(P + lambda C)/dpsi through the whole chain of rev 2's road (Mixed: the filter, the
softmax, the normalized interface density and its clip, the projection, SIMP's share, the
adjoint) against central differences at the twelve entries of largest |derivative| and the
twelve largest on pixels where the density is clipped at 1, on a design of the Voronoi start
(4 cells: 19570 is 16,600 m^2) with uniform noise of NOISE on every field, at T 1.5 T_w, beta 4,
q 3, price 0.3; step H_P (P is solved to about 1e-12 relative: road_simp_check's H_P); within TOL
relative of the larger. The FFT filter equals road_simp.Raster.filter to FILTER_TOL first.
Faults: `no-offdiag` (the softmax Jacobian's -phi_k phi_j term dropped), `no-T` (its 1 / T
dropped), `unclipped` (the derivative carried through the clip).
bgrad: the same through the banded road (Banded: the band's cone on phi between the softmax and
the density) at T BAND_T. Faults: those three, and `no-band` (the band's transpose dropped from
the way back).
dD: dD/dpsi the same way at step H (D is arithmetic), the noise NOISE_D (with NOISE every building
is past the ramp's knee and D's derivative 0; where the density is clipped its buildings are
covered, so the clip does not reach D's derivative), the true ramp. Fault: `no-norm` (the
normalization's 1 / (g_pair - g_pure) dropped).
edD: dD/dpsi with the ramp's tail EPS_D (Partition.D), banded, the noise NOISE: every building
past the knee, so the derivative is the tail's alone. Fault: `no-tail` (the tail's slope
dropped: the true ramp's derivative, 0 there).
deadends: the cut answer of the Voronoi start (4 cells, no update) counted on its ground's
skeleton: no dead end and no floating line; the stubs the thinning leaves at junctions are not
dead ends (shorter than a road's width). Fault: `spur` (a SPUR_M road from the middle of the
longest boundary, square to it on the side away from the street, added: one dead end).
mma: partition.Mma on a separable convex problem with a known optimum, min sum c_j / x_j s.t.
sum a_j x_j <= V (x_j proportional to sqrt(c_j / a_j)): the iterate within TOL_MMA of it after
MMA_STEPS. Fault: `no-mu` (the constraint's multiplier held at 0).
ridges: the banded road has no ridge on the answer the optimizer ends at (RIDGE_PLAN from the
Voronoi start): no pixel at rho > HALF whose argmax labels within the band's radius are one
label (a road inside one label, the rev 2 band answers' dead ends). Fault: `ridge` (a road
ROAD_W wide and RIDGE_M long planted at the pixel farthest inside a label: found).

    C=research/roadless/partition_check.py             # from the repo root, PYTHONPATH=.
    uv run python $C grad|bgrad|dD|edD|deadends|mma|ridges <block> [fault]
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray
from scipy import ndimage

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
MIXED_PLAN = "pt.K4.R10.t1.5.b4.q3.i1.D0.05.s0.3.vor4s0"   # grad, dD, deadends
BAND_T = 0.1             # bgrad, edD: the banded road's temperature
H_BAND = 4e-3            # bgrad: its step (J's evaluation noise over the sharp softmax: at H_P
                         # 1.8e-4, at 2.5e-4 4.8e-4, at 4e-3 4.6e-5)
EPS_D = 0.2              # edD: the ramp's tail
RIDGE_PLAN = "pt.K4.R10.band.t0.1,0.05,0.02.b1,4,16.q1,3,3.e0.2,0.05,0.i10.D0.05.s0.1.vor4s0"
RIDGE_M = 20.0           # ridges: the planted ridge's length
F = NDArray[np.float64]


def problem(bid: str, spec: str):  # type: ignore[no-untyped-def]
    plan = pt.Plan.of(spec)
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    p = pt.partition_of(b, metric, plan)
    f, n = pt.fields_of(b, p, plan.start)
    return b, metric, plan, p, f


def noisy(p: pt.Partition, f: F, noise: float) -> F:
    g = f + np.random.default_rng(0).uniform(0.0, noise, f.shape)
    return np.where(p.rd.raster.on, np.clip(g, 0.0, 1.0), 0.0)


def grad(bid: str, fault: str | None) -> None:
    _gradient(bid, fault, MIXED_PLAN, h=H_P)


def bgrad(bid: str, fault: str | None) -> None:
    _gradient(bid, fault, f"pt.K4.R10.band.t{BAND_T:g}.b4.q3.i1.D0.05.s0.3.vor4s0", h=H_BAND)


def _gradient(bid: str, fault: str | None, spec: str, *, h: float) -> None:
    _, _, plan, p, f = problem(bid, spec)
    f = noisy(p, f, NOISE)
    off = float(np.abs(p.lab.filter(f)[1] - p.rd.raster.filter(f[1])).max())
    print(f"filter: FFT against Raster.filter {off:.1e} -> "
          f"{'PASS' if off <= FILTER_TOL else 'FAIL'}")
    if off > FILTER_TOL:
        raise SystemExit(1)
    st = plan.stages[0]
    T, beta, q = p.lab.band.T(st.t), st.beta, st.q
    if fault is None:
        g = p.derivs(f, T, beta, q, 0.0, plan.price)[3]
    else:
        lab = p.lab
        w, ch = p.world(f, T, beta, q)
        _, gs, _ = p.rd.P_grad(w)
        cost = plan.price.charge(p.rd, w, rs.OC())
        g_rho = p.back_r(w, gs * q * np.maximum(w.r, 1e-300) ** (q - 1.0) + cost.d_r)
        lo, hi = lab.band.ends(T, p.K)
        live = (ch.raw > 0.0) & ((ch.raw < 1.0) | (fault == "unclipped"))
        g_mix = np.where(live, g_rho, 0.0) * (-2.0 * ch.mix) / (hi - lo)
        g_phi = g_mix if fault == "no-band" else lab.band.mix_T(g_mix)
        phi = ch.phi
        g_L = phi * g_phi if fault == "no-offdiag" else phi * (g_phi - (g_phi * phi).sum(0))
        g = lab.filter_T(g_L if fault == "no-T" else g_L / T)

    def J(z: F) -> float:
        w = p.world(z.reshape(f.shape), T, beta, q)[0]
        return p.rd.P_grad(w)[0] + plan.price.charge(p.rd, w, rs.OC()).cost

    clipped = np.tile(p.world(f, T, beta, q)[1].raw >= 1.0, p.K)
    print(f"{p.lab.band.name or 'mixed'}: T {T:.4f}; rho clipped at 1 on "
          f"{clipped[: p.npx][p.rd.raster.on].mean():.3f} of the pixels")
    gv = g.ravel()
    top = np.argsort(-np.abs(gv))[:N]
    at_clip = np.flatnonzero(clipped)[np.argsort(-np.abs(gv[clipped]))[:N]]
    _check(f.ravel(), gv, J, "dJ", h=h, at=np.concatenate([top, at_clip]))


def dD(bid: str, fault: str | None) -> None:
    _, _, plan, p, f = problem(bid, MIXED_PLAN)
    f = noisy(p, f, NOISE_D)
    T, beta, q = p.lab.band.T(1.5), 4.0, 3.0
    w, ch = p.world(f, T, beta, q)
    on = p.rd.raster.on
    X = p.rd.cover @ w.r
    print(f"buildings under the ramp's knee {(X < common.MEASURE.tau).mean():.2f}; rho clipped "
          f"at 1 on {(ch.raw[on] >= 1.0).mean():.3f} of the design pixels")
    g = p.lab.back(p.back_r(w, p.dD(w, 0.0))[None], ch, T)[0]
    if fault == "no-norm":
        lo, hi = p.lab.band.ends(T, p.K)
        g = g * (hi - lo)
    _check(f.ravel(), g.ravel(), lambda z: p.D_of(z.reshape(f.shape), T, beta, 0.0), "dD", h=H)


def edD(bid: str, fault: str | None) -> None:
    _, _, plan, p, f = problem(bid, f"pt.K4.R10.band.t{BAND_T:g}.b4.q3.i1.D0.05.s0.3.vor4s0")
    f = noisy(p, f, NOISE)
    T, beta, q = BAND_T, 4.0, 3.0
    w, ch = p.world(f, T, beta, q)
    X = p.rd.cover @ w.r
    print(f"buildings past the ramp's knee {(X > common.MEASURE.tau).mean():.2f}; D "
          f"{p.D(w.r, 0.0):.4f}, with the tail {EPS_D:g} {p.D(w.r, EPS_D):.4f}")
    g = p.lab.back(p.back_r(w, p.dD(w, 0.0 if fault == "no-tail" else EPS_D))[None], ch, T)[0]
    _check(f.ravel(), g.ravel(), lambda z: p.D_of(z.reshape(f.shape), T, beta, EPS_D), "dD",
           h=H)


def ridges(bid: str, fault: str | None) -> None:
    b, _, plan, p, f = problem(bid, RIDGE_PLAN)
    f, rows = pt.simp(p, f, plan, log=lambda s: None)
    last = plan.stages[-1]
    T = p.lab.band.T(last.t)
    r = rows[-1]
    print(f"{RIDGE_PLAN}: {len(rows)} updates, L_P {r.L_P:.4f}, D {r.D:.4f}, grey {r.grey:.3f}")
    band = p.lab.band
    assert isinstance(band, pt.Banded)
    ra = p.rd.raster
    rho = p.lab.chain(f, T).rho.reshape(ra.shape).copy()
    lab = p.lab.argmax(f).reshape(ra.shape)
    on = ra.on.reshape(ra.shape)
    if fault == "ridge":
        inner = np.zeros(ra.shape)
        for k in np.unique(lab[on]):
            d = ndimage.distance_transform_edt(lab == k)
            inner = np.maximum(inner, np.where(on, d, 0.0))
        i, j = np.unravel_index(int(np.argmax(inner)), ra.shape)
        h, half = int(RIDGE_M / pt.PIXEL) // 2, int(np.ceil(pt.ROAD_W / pt.PIXEL / 2))
        rho[max(i - h, 0):i + h, max(j - half, 0):j + half] = 1.0
        print(f"planted at {i}, {j}: {inner[i, j]:.1f} m inside label {lab[i, j]}")
    road, bad = one_label_roads(p, lab, rho, band.radius)
    ok = bad == 0
    print(f"ridges {bid}: {road} road pixels, {bad} with one label within "
          f"{band.radius:.2f} m -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


def one_label_roads(p: pt.Partition, lab: NDArray[np.int64], rho: F,
                    radius: float) -> tuple[int, int]:
    """(the design pixels at rho > HALF, those whose argmax labels `lab` within `radius` are one
    label), both on the raster's shape."""
    ra = p.rd.raster
    k = int(np.ceil(radius / pt.PIXEL))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    disk = np.hypot(yy, xx) <= radius / pt.PIXEL
    one = (ndimage.maximum_filter(lab, footprint=disk, mode="nearest")
           == ndimage.minimum_filter(lab, footprint=disk, mode="nearest"))
    road = ra.on.reshape(ra.shape) & (rho > pt.HALF)
    return int(road.sum()), int((road & one).sum())


def deadends(bid: str, fault: str | None) -> None:
    b, _, _, p, f = problem(bid, MIXED_PLAN)
    geo = p.rd.m.geo
    cut = shapely.intersection(geo.utm(pt.cut_lines(p, f)), b.boundary)
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
    for _ in range(MMA_STEPS):
        def D_of(z: F) -> float:
            return 0.0 if fault == "no-mu" else float(a @ z) - V
        x, _mu, _, _ = m.step(x, -c / x ** 2, a, D_of=D_of, budget=0.0, mu0=0.0)
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
    {"grad": grad, "bgrad": bgrad, "dD": dD, "edD": edD, "deadends": deadends, "mma": mma,
     "ridges": ridges}[verb](
        bid, fault[0] if fault else None)
