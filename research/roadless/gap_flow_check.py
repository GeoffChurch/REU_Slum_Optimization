"""Guards of the uniform law on the gap complex (gap_flow.py; NOTES, "The gap complex: passability,
enclosure and flow without the raster"), each with its fault injection (a patch in one process, no
file edited): a guard passes on the code as it is and must fail under its fault.

    S=research/roadless/gap_flow_check.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S angle - [fault]
    uv run python $S shift - [fault]
    uv run python $S channel - [fault]
    uv run python $S field - [fault]
    uv run python $S closed - [fault]

The corridor is the effective-width study's walled channel (W x L_C, walls footprints WALL_M
thick, a dead end behind the demand, the street the far edge; NOTES, "Uniform conductance with an
effective width"): its demand the open ground within 1 m of the dead end,
uniform by area, summing to 1, sigma 1. Beyond the dead end's metre the channel carries all of it
to the band (u = 0 from L_C - 1), so P = (L_C - 2 + 1/3) / (sigma W) exactly in the continuum
(the last third the demand's own spread).

`angle`: P's max / min over ANGLES at the widths WIDTHS_ANGLE (R_CHECK): within ANGLE_TOL
of 1 (an isotropic scheme on a mesh of the geometry is rotation-invariant up to meshing; the
raster under Fws: 1.21 at 1 m).
`shift`: P's max / min over SHIFTS lattice offsets (the Steiner lattice moved k / SHIFTS of R on
both axes) at 0 deg, WIDTHS_SHIFT: within SHIFT_TOL of 1.
`channel`: P sigma W / (L_C - 5/3) within CHANNEL_TOL of 1 at WIDTHS_CHANNEL (cracks of 5 and
12.5 cm among them: a crack conducts by its width) at 0 and 30 deg.
`field`: an open disk (no footprint), the demand uniform over the disk inside the band: P = 1 / (8
pi sigma) for any radius; within FIELD_TOL.
Fault `centroids` (all four): the cells centred at their centroids, the two-point flux across a
passage over the centroids' distance along its normal: no longer orthogonal, so inconsistent.

`law`: each width form's law on the corridor (at w LAW_W, the ramp's s LAW_S): P(L 40) - P(L 20)
= 20 / (sigma G(W)), the ends cancelling: Closing G = W, Shy G = W - w, Ramp G = 2 int_0^{W/2}
clip((y - w/2) / s, 0, 1) dy (W - w - s from W = w + 2 s), within LAW_TOL at LAW_WIDTHS, 0 and
30 deg. Fault `no-grow`: the shy distance and the ramp without their zero zone (footprints not
grown, the ramp from the wall).

`convex` and `gradient` (the clearing model's) moved with it to gap_clear_check.py.

`closed`: a lane CLOSED_W wide conducts at a passable width below CLOSED_W and is walled in
above it (its demand reaches no street-connected cell), at R_CHECK. Fault `no-clearance`: the
Steiner points kept only R / 2 clear of the outlines whatever the width (a point in the lane makes
its crossing edges inf).
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap_complex as gc  # noqa: E402
import gap_flow as gf  # noqa: E402

S_M = 0.125
L_C = 40.0                       # the walled channel: its length (m) ...
WALL_M = 3.0                     # ... its walls' thickness (m) ...
ANGLES = (0, 2, 4, 6, 8, 10, 15, 20, 26.57, 30, 35, 40, 45)    # ... and its rotations (deg)
R_CHECK = 0.25
OPEN = gf.Closing(w=0.0)          # the uniform law, every passage open
SIGMA = 1.0
QUAD = 4
WIDTHS_ANGLE = (1.0, 1.5, 2.0, 4.0)
ANGLE_TOL = 0.01
SHIFTS = 8
WIDTHS_SHIFT = (0.75, 1.0, 1.5, 2.0)
SHIFT_TOL = 0.01
WIDTHS_CHANNEL = (0.05, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
CHANNEL_TOL = 0.01
FIELD_R = (6.0, 12.0)            # the disks' radii (m): the domain is a disk 1 m smaller
FIELD_TOL = 0.02                 # at R 0.5 on the smaller disk the error is 1.4%, second order
CLOSED_W = 0.6
CLOSED_AT = (0.5, 0.7)           # the passable widths tried: open at the first, walled at the second
CLOSED_AT_M = 3.0
CLOSED_SRC = 0.2


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def spec() -> gc.Spec:
    return gc.Spec(s=S_M, ring_m=1.0, band_m=1.0)


def fspec(R: float, offset: tuple[float, float] = gf.OFFSET) -> gf.FlowSpec:
    return gf.FlowSpec(R=R, levels=0, grade=0.0, sigma=SIGMA, quad=QUAD, offset=offset)


def corridor(alpha_deg: float, W: float, L: float = L_C):
    """The walled channel's geometry as a block (its length L), and its demand region (in its
    own frame)."""
    a = np.radians(alpha_deg)
    e, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])

    def pt(s: float, t: float) -> tuple[float, float]:
        x = s * e + t * n
        return float(x[0]), float(x[1])

    def rect(s0: float, s1: float, t0: float, t1: float) -> Polygon:
        return Polygon([pt(s0, t0), pt(s1, t0), pt(s1, t1), pt(s0, t1)])

    w2, T = W / 2, WALL_M
    block = rect(-T, L, -w2 - T, w2 + T)
    walls = [rect(-T, L, w2, w2 + T), rect(-T, L, -w2 - T, -w2), rect(-T, 0.0, -w2, w2)]
    street = LineString([pt(L, -w2 - T), pt(L, w2 + T)])
    src = LineString([pt(0.0, -w2), pt(0.0, w2)]).buffer(1.0)
    b = SimpleNamespace(block_id="corridor", boundary=block,
                        buildings=SimpleNamespace(outlines=walls),
                        streets=SimpleNamespace(geometry=[street]))
    return b, src


def P_of(b, src, fs: gf.FlowSpec, form: gf.Width) -> tuple[float, float]:
    """(P, the share of the demand region's area reached) of a uniform demand over `src`'s open
    ground (block frame) on the complex under the width form."""
    geo = gc.geometry(b)
    srcl = shapely.transform(src, lambda xy: xy - np.array(geo.origin))
    n = len(geo.polys)
    m = gf.mesh(geo, spec(), fs, form, np.ones(n, dtype=bool), carve=None)
    try:
        c = gf.cells(m)
    except ValueError as e:                  # a scheme with negative weights: no answer
        print(f"  {e}", flush=True)
        return np.nan, 0.0
    P_, wt, tt = gf._points_of(m.xy, m.tri, fs.quad)
    inside = shapely.contains_xy(srcl, P_[:, 0], P_[:, 1])
    if not inside.any():
        raise ValueError("no demand")
    cell = c.cell[tt]
    unk = -np.ones(c.n, dtype=np.int64)
    unk[~c.ground] = np.arange(int((~c.ground).sum()))
    uk = np.where(cell >= 0, unk[np.maximum(cell, 0)], -1)
    reached = float(wt[inside & (cell >= 0)].sum() / wt[inside].sum())
    ok = inside & (uk >= 0)
    if not ok.any():
        return np.inf, reached
    A = gf.laplacian(c)
    b_ = np.bincount(uk[ok], weights=wt[ok], minlength=A.shape[0])
    b_ = b_ / wt[inside & (cell >= 0)].sum()
    u = gf.factor(A).solve(b_)
    return float(b_ @ u), reached


def _exact(W: float) -> float:
    return (L_C - 2.0 + 1.0 / 3.0) / (SIGMA * W)


# ------------------------------------------------------------------ the guards

def angle(_ids: list[str]) -> list[str]:
    bad = []
    for W in WIDTHS_ANGLE:
        Ps = [P_of(*corridor(a, W), fspec(R_CHECK), OPEN)[0] for a in ANGLES]
        r = max(Ps) / min(Ps)
        ok = bool(r <= 1.0 + ANGLE_TOL)
        print(f"W {W:g}: P's max / min over {len(Ps)} angles {r:.4f} (P / exact "
              f"{min(Ps) / _exact(W):.4f} -- {max(Ps) / _exact(W):.4f}) -> "
              f"{'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(f"W {W:g}")
    return bad


def shift(_ids: list[str]) -> list[str]:
    bad = []
    for W in WIDTHS_SHIFT:
        Ps = []
        for k in range(SHIFTS):
            off = ((gf.OFFSET[0] + k / SHIFTS) % 1.0, (gf.OFFSET[1] + k / SHIFTS) % 1.0)
            Ps.append(P_of(*corridor(0.0, W), fspec(R_CHECK, off), OPEN)[0])
        r = max(Ps) / min(Ps)
        ok = bool(r <= 1.0 + SHIFT_TOL)
        print(f"W {W:g}: P's max / min over {SHIFTS} lattice shifts {r:.4f} -> "
              f"{'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(f"W {W:g}")
    return bad


def channel(_ids: list[str]) -> list[str]:
    bad = []
    for W in WIDTHS_CHANNEL:
        for a in (0.0, 30.0):
            P, _r = P_of(*corridor(a, W), fspec(R_CHECK), OPEN)
            x = P / _exact(W)
            ok = bool(abs(x - 1.0) <= CHANNEL_TOL)
            print(f"W {W:g} angle {a:g}: P / ((L - 5/3) / sigma W) {x:.4f} -> "
                  f"{'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"W {W:g} a {a:g}")
    return bad


def field(_ids: list[str]) -> list[str]:
    bad = []
    for rad in FIELD_R:
        disk = Point(0.0, 0.0).buffer(rad, quad_segs=64)
        b = SimpleNamespace(block_id="disk", boundary=disk,
                            buildings=SimpleNamespace(outlines=[]),
                            streets=SimpleNamespace(geometry=[disk.exterior]))
        for R in (0.5, R_CHECK):
            P, _r = P_of(b, disk, fspec(R), OPEN)
            x = P * 8.0 * np.pi * SIGMA
            ok = bool(abs(x - 1.0) <= FIELD_TOL)
            print(f"disk radius {rad:g} R {R:g}: P / (1 / 8 pi sigma) {x:.4f} -> "
                  f"{'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"r {rad:g} R {R:g}")
    return bad


LAW_W = 0.5
LAW_S = 0.25
LAW_WIDTHS = (0.75, 1.0, 1.5, 2.0, 3.0)
LAW_L = (40.0, 20.0)
LAW_TOL = 0.01


def _G(form: gf.Width, W: float) -> float:
    """The form's width law, a lane's conductance per unit sigma and length."""
    if isinstance(form, gf.Closing):
        return W if W >= form.w else 0.0
    if isinstance(form, gf.Shy):
        return max(0.0, W - form.w)
    a, s_ = form.w / 2.0, form.s
    h = W / 2.0
    if h <= a:
        return 0.0
    if h <= a + s_:
        return (h - a) ** 2 / s_
    return 2.0 * ((h - a - s_) + s_ / 2.0)


def law(_ids: list[str]) -> list[str]:
    """Each form's width law on the corridor: P(L_C 40) - P(20) = 20 / (sigma G(W)), the ends'
    effects cancelling, at w LAW_W (the ramp's s LAW_S), 0 deg and 30 deg."""
    bad = []
    for form in (gf.Closing(w=LAW_W), gf.Shy(w=LAW_W), gf.Ramp(w=LAW_W, s=LAW_S)):
        for W in LAW_WIDTHS:
            for a in (0.0, 30.0):
                P = [P_of(*corridor(a, W, L), fspec(R_CHECK), form)[0] for L in LAW_L]
                x = (P[0] - P[1]) * SIGMA * _G(form, W) / (LAW_L[0] - LAW_L[1])
                ok = bool(abs(x - 1.0) <= LAW_TOL)
                print(f"{form.name} W {W:g} angle {a:g}: dP sigma G(W) / dL {x:.4f} (G "
                      f"{_G(form, W):.4f}) -> {'ok' if ok else 'FAILS'}", flush=True)
                if not ok:
                    bad.append(f"{form.name} W {W:g} a {a:g}")
    return bad


def closed(_ids: list[str]) -> list[str]:
    """The demand a disk of radius CLOSED_SRC in the lane's middle, CLOSED_AT_M from the dead end
    (the dead end's corners are wedges a disk of width w cannot enter: walled in at w)."""
    bad = []
    for w in CLOSED_AT:
        b, _src = corridor(17.0, CLOSED_W)
        a = np.radians(17.0)
        src = Point(CLOSED_AT_M * np.cos(a), CLOSED_AT_M * np.sin(a)).buffer(CLOSED_SRC)
        _P, reached = P_of(b, src, fspec(R_CHECK), gf.Closing(w=w))
        want = w <= CLOSED_W
        ok = (reached == 1.0) if want else (reached == 0.0)
        print(f"a {CLOSED_W:g} m lane at passable width {w:g}: {reached:.3f} of the demand "
              f"reaches the street (want {'all' if want else 'none'}) -> "
              f"{'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(f"w {w:g}")
    return bad


# ------------------------------------------------------------------ the clearing model

# ------------------------------------------------------------------ the faults

_CELLS = gf.cells
_CIRC = gf.circumcentres


def _centroids(xy, tri):
    return xy[tri].mean(axis=1)


@contextmanager
def _fault_centroids() -> Iterator[None]:
    with _patched(gf, "circumcentres", _centroids):
        yield


_STEINER = gf.steiner_points


def _no_clearance(inner, fs, clear, *, box, lines):
    return _STEINER(inner, fs, 0.0, box=box, lines=lines)


_RAMP_KAPPA = gf.Ramp.kappa


def _ramp_from_zero(self, *a):
    import dataclasses
    return _RAMP_KAPPA(dataclasses.replace(self, w=0.0), *a)


@contextmanager
def _fault_no_grow() -> Iterator[None]:
    """The shy distance and the ramp without their zero zone (the footprints not grown, the ramp
    from the wall)."""
    zero = property(lambda self: 0.0)
    with _patched(gf.Shy, "grow", zero), _patched(gf.Ramp, "grow", zero), _patched(
            gf.Ramp, "kappa", _ramp_from_zero):
        yield


GUARDS: dict[str, Callable[[list[str]], list[str]]] = {
    "angle": angle, "shift": shift, "channel": channel, "field": field, "law": law,
    "closed": closed}
FAULTS: dict[str, dict[str, Callable]] = {
    "angle": {"centroids": _fault_centroids},
    "shift": {"centroids": _fault_centroids},
    "channel": {"centroids": _fault_centroids},
    "field": {"centroids": _fault_centroids},
    "law": {"no-grow": lambda: _fault_no_grow()},
    "closed": {"no-clearance": lambda: _patched(gf, "steiner_points", _no_clearance)},
}


def main(guard: str, ids: list[str], fault: str | None) -> None:
    if fault is None:
        bad = GUARDS[guard](ids)
    else:
        with FAULTS[guard][fault]():
            bad = GUARDS[guard](ids)
    verdict = "FAILS on " + ", ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) not in (2, 3) or a[0] not in GUARDS:
        raise SystemExit(__doc__)
    main(a[0], [] if a[1] == "-" else a[1].split(","), a[2] if len(a) == 3 else None)
