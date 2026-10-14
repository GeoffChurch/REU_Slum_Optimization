"""Guards of the complex as the runtime (gap_clear.py's fixed mesh; NOTES, "The complex as the
runtime: one fixed mesh"), each with its fault injection (a patch in one process, no file edited):
a guard passes on the code as it is and must fail under its fault. Every guard meshes afresh (a
cache of its own, in a temporary directory), so a fault in the meshing is seen.

    S=research/roadless/gap_clear_check.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S baseline <ids,> [fault]    # faults: interior, exits
    uv run python $S weights <ids,> [fault]     # fault: no-conform
    uv run python $S reach <ids,> [fault]       # fault: threshold
    uv run python $S gradient <ids,> [fault]    # fault: tie
    uv run python $S free <ids,> [fault]        # fault: drop-crossing
    uv run python $S identity - [fault]         # fault: collide
    uv run python $S convex <ids,> [fault]      # fault: multiply

`baseline`: the fixed mesh (the runtime's metric, METRIC) against the complex rebuilt
(gap_compare.ComplexModel, gap_flow's mesh under the shy distance at 0.5): P0, P_{1/2},0 and
J_2,0 within BASE_TOL (relative), the homes with no demand alike, each home's escape time within
HOME_TOL (relative), and the ONE_K most populous buildings each cleared alone, L_P within ONE_TOL.
Faults: `interior` (the most populous building's interior faces dropped from the mesh: its
clearing opens no ground), `exits` (the street line's vertices, moved by the linework's snap, read
at the unsnapped tolerance: the exits lost; the defect this guard caught while it was built, P0
22% high on 20269).
`weights`: no passage weight is negative or clipped (every face boundary two triangles share has
its centres in order), and at each of WEIGHT_STATES' states every conductance is non-negative and
every home with demand has u0 > 0. Fault `no-conform`: the conformity's splits skipped (a mesh
with face boundaries not Delaunay: refused).
`reach`: at the baseline, the ONE_K single clearings, REACH_RANDOM random clearings at area D
BUDGET and free first's state, the homes the flow model reaches (Model.live_homes_of) equal those
the complex grown by w / 2 reaches at width 0 (gap_complex's test), home for home. Fault
`threshold`: the flow model's reach of a home off by one comparison (a ring with no triangle
reaching the street counted as reaching it: gap_flow.Ring.live).
`gradient`: the model's gradient (power 1 and 2, q 1 and 3, eps GRAD_EPS) at a random x in [0.1,
0.4] and at SIMP's start (x = D everywhere: every triangle within w / 2 of two footprints a tie)
against the closer of the central differences at GRAD_STEPS on GRAD_BUILDINGS buildings (at the
start, those in the most two-way ties and no wider one, `_paired`), within GRAD_TOL of the
largest, under each of GRAD_FORMS. Fault `tie`: a tie's gradient broken to one building
(gap_flow.Terms.vjp).
`free`: free first on the complex (Model.free_first, budget 1) frees every walled-in home, and the
flow model's own reach agrees home for home (Model.free_first raises where it does not). Fault
`drop-crossing`: one crossing dropped from free first's answer (its claim kept).
`identity`: the complex's names spell back (gap_clear.complex_of), and a complex metric whose name
would be a raster row directory's (every directory under relax_rows/ and polish_rows/, and the
conductances' names) raises. Fault `collide`: gap_clear.check_disjoint disabled.
`convex` (the study's, from gap_flow_check.py): along CONVEX_LINES random segments of the
clearing box, every passage's factor concave and P (q 1, eps GRAD_EPS) convex, under each of
GRAD_FORMS. Fault `multiply`: a point's terms multiplied (transmittances in series: equal at every
0/1 state, not concave).
"""
from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_compare as gcmp  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_flow as gf  # noqa: E402

METRIC = gcl.complex_of("cx+shy0.5+s0.125r1b1+R0.5g4x2q4")    # the runtime's (owner 2026-10-14)
BUDGET = 0.05
BASE_TOL = 2e-3          # the baseline against the rebuild: the rebuild's own lattice, moved,
                         # moves P0 by up to 1.9e-3 and J_2,0 by 4.6e-3 (19570)
HOME_TOL = 5e-2          # ... each home's escape time (the rebuild's lattice moved: up to 9.8%)
ONE_K = 3
ONE_TOL = 2e-3           # a single clearing's L_P against the rebuild's
WEIGHT_STATES = 3        # random 0/1 clearings at BUDGET, besides the baseline, the eps world and
                         # a fractional state
REACH_RANDOM = 2
GRAD_FORMS = ("shy0.5", "close0.125", "ramp0.5")
GRAD_EPS = 1e-4          # SIMP base's eps world
GRAD_BUILDINGS = 6
GRAD_Q = (1.0, 3.0)
GRAD_STEPS = (1e-5, 1e-6)    # the closer central difference (the factors are piecewise linear)
GRAD_TOL = 1e-4
CONVEX_LINES = 4
CONVEX_POINTS = 7
CONVEX_TOL = 1e-9
SEED = 20261014
POP = common.PopulationDemand()


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _model(b, metric: gcl.ComplexMetric = METRIC) -> gcl.Model:
    """The block's model meshed afresh (a temporary cache: a fault in the meshing is seen)."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = gcl.Fixed.of(b, metric.spec, metric.fs, metric.form, lumped=metric.lumped,
                          cache=Path(tmp))
    return gcl.Model(b, fx, spec=metric.spec, fs=metric.fs, form=metric.form,
                     pop=common.POPULATIONS["area"].weights(np.asarray(b.buildings.outlines)),
                     demand=POP)


def _random(m: gcl.Model, rng: np.random.Generator) -> np.ndarray:
    order = rng.permutation(m.n)
    x = np.zeros(m.n, dtype=bool)
    x[order[np.cumsum(m.cost[order]) <= BUDGET]] = True
    return x


def _largest(m: gcl.Model, k: int) -> list[int]:
    return [int(j) for j in np.argsort(-m.pop, kind="stable")[:k]]


# ------------------------------------------------------------------ baseline

def baseline(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        try:
            m = _model(b)
        except Exception as e:      # noqa: BLE001 -- a fault may break the mesh: reported
            print(f"{bid}: the model fails to build ({e!r}) -> FAILS", flush=True)
            bad.append(bid)
            continue
        cm = gcmp.ComplexModel(b, METRIC.fs, METRIC.form)
        rel = {"P0": m.base.P / cm.P0 - 1.0, "Pq0": m.base.Pq / cm.Pq0 - 1.0,
               "J2_0": m.base.J2 / cm.J2_0 - 1.0}
        same_walled = bool(np.array_equal(m.nodem, cm.walled))
        live = ~m.nodem
        home = float(np.max(np.abs(m.u_pop0[live] / cm.u0[live] - 1.0))) if live.any() else 0.0
        ones = {}
        for j in _largest(m, ONE_K):
            x = np.zeros(m.n, dtype=bool)
            x[j] = True
            L = m.lenses(x)
            r, _u = cm.score(~x, None)
            ones[j] = (1.0 - L.P / m.base.P) - r["L_P"]
        worst_one = max(abs(v) for v in ones.values())
        ok = (max(abs(v) for v in rel.values()) <= BASE_TOL and same_walled
              and home <= HOME_TOL and worst_one <= ONE_TOL)
        print(f"{bid}: baseline against the rebuild "
              + ", ".join(f"{k} {v:+.1e}" for k, v in rel.items())
              + f"; no-demand homes alike {same_walled}; homes' u0 within {home:.1e}; single "
              f"clearings' L_P " + ", ".join(f"{j} {v:+.1e}" for j, v in ones.items())
              + f" -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


_FACES = gcl._faces


def _faces_less_interior(geo, spec, grow):
    """gap_clear._faces less the faces the most populous building covers (fault `interior`)."""
    faces, cover, zone, noded = _FACES(geo, spec, grow)
    j = int(np.argmax(common.POPULATIONS["area"].weights(geo.polys)))
    keep = np.array([j not in cv for cv in cover])
    return faces[keep], cover[keep], zone[keep], noded


# ------------------------------------------------------------------ weights

def weights(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        try:
            m = _model(b)
        except Exception as e:      # noqa: BLE001
            print(f"{bid}: the mesh is refused ({e!r}) -> FAILS", flush=True)
            bad.append(bid)
            continue
        fm, c = m.fm, m.c
        L = np.hypot(*(fm.xy[fm.pu] - fm.xy[fm.pv]).T)
        neg = int((fm.T < 0).sum())
        cross = fm.face[fm.pa] != fm.face[fm.pb]
        cross_inf = int((cross & np.isinf(fm.T)).sum())
        rng = np.random.default_rng(SEED)
        states = {"baseline": np.zeros(m.n), "eps": np.full(m.n, GRAD_EPS),
                  "fractional": rng.uniform(0.0, 1.0, m.n)}
        for k in range(WEIGHT_STATES):
            states[f"random{k}"] = _random(m, rng).astype(float)
        worst = np.inf
        for _name, k_ in states.items():
            st = m.factors(k_)
            worst = min(worst, float(st.reach.Tp.min()), float(st.reach.Gf.min()))
        u0_ok = bool((m.u_pop0[~m.nodem] > 0).all())
        ok = neg == 0 and cross_inf == 0 and worst >= 0 and u0_ok
        print(f"{bid}: {len(L)} passages, {neg} negative weights, {cross_inf} across faces "
              f"with one circumcentre, {fm.splits} conformity splits; the least conductance over "
              f"{len(states)} states {worst:.2e}; u0 > 0 for every home with demand {u0_ok} -> "
              f"{'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


def _no_conform(a):
    """gap_clear._conform without its splits (fault `no-conform`)."""
    return gcl.Conformed(arrays=a, unflat=0, flat_left=0, splits=0, rounds=0)


# ------------------------------------------------------------------ reach

def _complex_live(m: gcl.Model, removed: np.ndarray) -> np.ndarray:
    """The homes the complex grown by w / 2 reaches at width 0 with `removed` cleared."""
    cx = gc.build(m.geo, m.spec, ~removed, carve=None, grow=m.form.grow)
    return ~gc.walled(gc.home_exits(gc.homes_of(m.geo, cx), gc.exit_widths(cx)),
                      m.form.filtration_w)


def reach(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        try:
            m = _model(b)
        except Exception as e:      # noqa: BLE001 -- a fault may break the model: reported
            print(f"{bid}: the model fails to build ({e!r}) -> FAILS", flush=True)
            bad.append(bid)
            continue
        rng = np.random.default_rng(SEED + 1)
        states = {"baseline": np.zeros(m.n, dtype=bool)}
        for j in _largest(m, ONE_K):
            x = np.zeros(m.n, dtype=bool)
            x[j] = True
            states[f"single {j}"] = x
        for k in range(REACH_RANDOM):
            states[f"random{k}"] = _random(m, rng)
        fr = gc.free_first(m.geo, m.spec, w=m.form.filtration_w, grow=m.form.grow, budget=1.0,
                           cost=m.cost, pop=m.pop, held0=np.zeros(m.n, dtype=bool), cx0=None)
        states["free first"] = fr.held
        parts = []
        for name, x in states.items():
            flow = m.live_homes_of(m.factors(m.k_of(x)))
            cxl = _complex_live(m, x)
            d = np.flatnonzero(flow ^ cxl)
            parts.append(f"{name}: {int((~flow).sum())} walled in, {len(d)} apart")
            if len(d):
                bad.append(f"{bid} {name} {d[:5].tolist()}")
        print(f"{bid}: " + "; ".join(parts), flush=True)
    return bad


def _ring_off_by_one(self, live_tri):
    """gap_flow.Ring.live with its test `> 0` read `>= 0` (fault `threshold`): every home's ring
    reaches the street."""
    return (self.tri @ live_tri.astype(np.float64) >= 0) | self.band


# ------------------------------------------------------------------ gradient

def gradient(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for token in GRAD_FORMS:
            metric = gcl.complex_of(f"cx+{gcmp.form_of(token).name}+s0.125r1b1+R0.5g4x2q4")
            m = _model(b, metric)
            rng = np.random.default_rng(SEED)
            starts = {"random": (rng.uniform(0.1, 0.4, m.n),
                                 rng.choice(m.n, GRAD_BUILDINGS, replace=False)),
                      "start": (np.full(m.n, BUDGET), _paired(m.terms, m.n))}
            for where, (x0, pick) in starts.items():
                for power in (1.0, 2.0):
                    for q in GRAD_Q:
                        rel = m.relaxation(power, q=q, rtol=0.0, eps=GRAD_EPS)
                        _J, g = rel.value_grad(x0)
                        err_j = []
                        for j in pick:
                            best = np.inf
                            for h in GRAD_STEPS:
                                e = np.zeros(m.n)
                                e[j] = h
                                fd = (rel.value(x0 + e) - rel.value(x0 - e)) / (2 * h)
                                best = min(best, abs(fd - g[j]))
                            err_j.append(best)
                        err = float(np.max(err_j) / np.abs(g[pick]).max())
                        ok = err <= GRAD_TOL
                        print(f"{bid} {token} {where} J_{power:g} q {q:g}: gradient against "
                              f"central differences {err:.1e} of the largest -> "
                              f"{'ok' if ok else 'FAILS'}", flush=True)
                        if not ok:
                            bad.append(f"{bid} {token} {where} p{power:g} q{q:g}")
    return bad


def _paired(terms: gf.Terms, n: int) -> np.ndarray:
    """At SIMP's start (every k alike) the GRAD_BUILDINGS buildings in the most two-way ties (a
    point within w / 2 of it and one other footprint, phi 0) and in no wider one: there the
    central difference of the minimum is the shared gradient's 1/2 (of three tied, the central
    difference is still 1/2, the shared gradient 1/3: both supergradients)."""
    zero = terms.phi == 0.0
    row, b = terms.row[zero], terms.b[zero]
    size = np.bincount(row, minlength=terms.n * terms.G)[row]
    two = np.bincount(b[size == 2], minlength=n)
    wider = np.bincount(b[size > 2], minlength=n) > 0
    return np.argsort(-np.where(wider, -1, two), kind="stable")[:GRAD_BUILDINGS]


def _vjp_first(self, k, g):
    """gap_flow.Terms.vjp with a tie's gradient given wholly to its first building (fault
    `tie`)."""
    v, mins = self._mins(k)
    at = v <= mins[self.row] + gf.TIE_TOL
    idx = np.flatnonzero(at)
    first = np.ones(len(idx), dtype=bool)
    first[1:] = self.row[idx[1:]] != self.row[idx[:-1]]
    idx = idx[first]
    wt = (1.0 - self.phi[idx]) / self.G
    return np.bincount(self.b[idx], weights=g[self.row[idx] // self.G] * wt, minlength=len(k))


# ------------------------------------------------------------------ free first

def free(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        m = _model(b)
        try:
            held = m.free_first(budget=1.0, held=np.zeros(m.n, dtype=bool))
        except RuntimeError as e:
            print(f"{bid}: free first and the flow model part ({e}) -> FAILS", flush=True)
            bad.append(bid)
            continue
        left = m.walled0 & ~m.live_homes_of(m.factors(m.k_of(held)))
        ok = not left.any()
        print(f"{bid}: {int(m.walled0.sum())} walled in, free first holds "
              f"{np.flatnonzero(held).tolist()}, the flow model leaves "
              f"{np.flatnonzero(left).tolist()} walled in -> {'ok' if ok else 'FAILS'}",
              flush=True)
        if not ok:
            bad.append(bid)
    return bad


_FREE_FIRST = gc.free_first


def _drop_crossing(geo, spec, *, w, grow, budget, cost, pop, held0, cx0):
    """gap_complex.free_first with its last crossing's buildings dropped from the held set, its
    claim (`left`) kept (fault `drop-crossing`)."""
    fr = _FREE_FIRST(geo, spec, w=w, grow=grow, budget=budget, cost=cost, pop=pop, held0=held0,
                     cx0=cx0)
    if not fr.crossings:
        return fr
    held = fr.held.copy()
    held[fr.crossings[-1][1]] = False
    return fr._replace(held=held)


# ------------------------------------------------------------------ identity

def _raster_names() -> list[str]:
    names = {"uni+apwfs", "uni+b1+apwfs", "uni+d0.5+apwfs", "ss100k2n2r30+apwfs", "uni"}
    for d in ("relax_rows", "polish_rows"):
        if (HERE / d).is_dir():
            names |= {p.name for p in (HERE / d).iterdir() if p.is_dir()
                      and not p.name.startswith("cx+")}
    return sorted(names)


def identity(_ids: list[str]) -> list[str]:
    bad = []
    for name in ("cx+shy0.5+s0.125r1b1+R0.5g4x2q4", "cx+close0.125+s0.125r1b1+R0.5g4x2q4",
                 "cx+ramp0.5s0.5+s0.125r1b1+R0.25q4+lumped"):
        try:
            back = gcl.complex_of(name).name
        except ValueError as e:
            back = repr(e)
        if back != name:
            bad.append(f"{name} spells {back}")
    refused = 0
    names = _raster_names()
    for raster in names:
        class Colliding(gcl.ComplexMetric):
            @property
            def name(self) -> str:
                return raster
        try:
            Colliding(spec=METRIC.spec, fs=METRIC.fs, form=METRIC.form, lumped=False)
        except ValueError:
            refused += 1
            continue
        bad.append(f"a complex metric named {raster!r} was accepted")
    print(f"identity: names spelled back; {refused} of {len(names)} raster names refused as a "
          f"complex metric's ({', '.join(names[:6])}, ...) -> {'ok' if not bad else 'FAILS'}",
          flush=True)
    return bad


# ------------------------------------------------------------------ convexity (the study's)

def convex(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for token in GRAD_FORMS:
            m = _model(b, gcl.complex_of(f"cx+{gcmp.form_of(token).name}+s0.125r1b1+R0.5g4x2q4"))
            rel = m.relaxation(1.0, q=1.0, rtol=0.0, eps=GRAD_EPS)
            rng = np.random.default_rng(SEED + 2)
            worst_k, worst_P = -np.inf, np.inf
            for _ in range(CONVEX_LINES):
                a, z = rng.uniform(0.0, 1.0, m.n), rng.uniform(0.0, 1.0, m.n)
                ks, Ps = [], []
                for t in np.linspace(0.0, 1.0, CONVEX_POINTS):
                    x = a + t * (z - a)
                    ks.append(m.factors(rel.k_of(x)).kp)
                    Ps.append(rel.value(x))
                K2 = np.diff(np.array(ks), 2, axis=0)
                P2 = np.diff(np.array(Ps), 2)
                worst_k = max(worst_k, float(K2.max()))
                worst_P = min(worst_P, float((P2 / np.max(Ps)).min()))
            ok = worst_k <= CONVEX_TOL and worst_P >= -CONVEX_TOL
            print(f"{bid} {token}: on {CONVEX_LINES} lines the factors' largest second difference "
                  f"{worst_k:.1e}, P's smallest {worst_P:.1e} of P -> {'ok' if ok else 'FAILS'}",
                  flush=True)
            if not ok:
                bad.append(f"{bid} {token}")
    return bad


def _multiply(self, k):
    """gap_flow.Terms.value with a point's terms multiplied (transmittances in series), not their
    minimum: equal at every 0/1 state, not concave (fault `multiply`)."""
    v = self.phi + k[self.b] * (1.0 - self.phi)
    prod = np.ones(self.n * self.G)
    if len(self.row):
        np.multiply.at(prod, self.row, v)
    return prod.reshape(self.n, self.G).mean(axis=1)


# ------------------------------------------------------------------ the run

GUARDS: dict[str, Callable[[list[str]], list[str]]] = {
    "baseline": baseline, "weights": weights, "reach": reach, "gradient": gradient,
    "free": free, "identity": identity, "convex": convex}


def _fault_exits():
    """The band line's faces read at the unsnapped tolerance (gap_complex.TOL)."""
    return _patched(gcl, "BAND_TOL_M", gc.TOL)


FAULTS: dict[str, dict[str, Callable]] = {
    "baseline": {"interior": lambda: _patched(gcl, "_faces", _faces_less_interior),
                 "exits": _fault_exits},
    "weights": {"no-conform": lambda: _patched(gcl, "_conform", _no_conform)},
    "reach": {"threshold": lambda: _patched(gf.Ring, "live", _ring_off_by_one)},
    "gradient": {"tie": lambda: _patched(gf.Terms, "vjp", _vjp_first)},
    "free": {"drop-crossing": lambda: _patched(gc, "free_first", _drop_crossing)},
    "identity": {"collide": lambda: _patched(gcl, "check_disjoint", lambda name: None)},
    "convex": {"multiply": lambda: _patched(gf.Terms, "value", _multiply)},
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
