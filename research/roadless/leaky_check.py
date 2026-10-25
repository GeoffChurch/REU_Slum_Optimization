"""Guards of walls that leak (gap_walls; NOTES, "Walls that leak: every home's source inside it,
each wall priced as a length"), each with its fault injection (a patch in one process, no file
edited): a guard passes on the code as it is and must fail under its fault (gap_clear_check.py's
convention). Every guard meshes afresh (a cache of its own, in a temporary directory).

    S=research/roadless/leaky_check.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S direction - [fault]          # fault: in-out-swapped
    uv run python $S party - [fault]              # fault: one-side
    uv run python $S settles <ids,> [fault]       # fault: one-pass
    uv run python $S energy <ids,> [fault]        # fault: wrong-side
    uv run python $S monotone <ids,> [fault]      # fault: cleared-slow
    uv run python $S walled-count <ids,> [fault]  # fault: through-walls
    uv run python $S sources <ids,> [fault]       # fault: doors
    uv run python $S walls-identity - [fault]     # fault: part-dropped
    uv run python $S relax-gradient <ids,> [fault]  # faults: no-faces, forward-only, ground-zero
    uv run python $S relax-ends <ids,> [fault]    # fault: owners-reversed
    uv run python $S relax-monotone <ids,> [fault]  # fault: cleared-slow
    uv run python $S relax-warm <ids,> [fault]    # fault: one-pass
    uv run python $S relax-identity - [fault]     # faults: eps-accepted, part-dropped
    uv run python $S greedy-exact <ids,> [fault]  # faults: stale-gain, over-budget
    uv run python $S both-ways <guards,> <ids,|->

Each runs under the guards' walls, SYM (symmetric, ell 200 m) and DIODE (out 50 m, in 300 m), k_in
1, on the per-block metric (synthetic blocks: no leak, the street on one edge) or on UNIFORM (real
blocks: metrics.walls_on_uniform's base).

`direction` (DIRECTION_BLOCK: a strip 60 x 12 m, its east edge the street; a building B across it,
x 25 -- 35, between two alleys; a home H in the west alley, a pocket whose one way out is through
B): the current through each wall group, from the solved potentials and the definition alone
(each passage sigma |e| / (d_e + c), c ell_in where its current enters a building and ell_out
where it leaves one, d_e from the mesh's weight), equals what conservation says crosses it: H's
residents out through H's wall (they pay ell_out), H's current into B through its west wall and
H's and B's out through its east wall (through-current pays ell_in + ell_out, B's residents
ell_out), within CURRENT_TOL. Fault `in-out-swapped`: DiodeWalls charging ell_in to leave and
ell_out to enter.
`party` (PARTY_BLOCK: buildings A, x 5 -- 15, and B, x 15.5 -- 25.5, across a strip 40 x 10 m,
the gap between them w, so their grown footprints meet in a line: a party wall proper, the
passages from A's to B's): with the street on the east edge A's residents cross into B, with it on
the west B's into A; the party passages' current from the potentials and sigma |e| / (d_e +
ell_out + ell_in) equals the crossing population each way, within CURRENT_TOL. Fault `one-side`:
DiodeWalls charging a party passage only ell_out from its first triangle to its second.
`settles`: after the diode's loop, every wall passage's conductance is its current's side's (beyond
SIDE_REL), on DIRECTION_BLOCK and each block named, at the baseline (from the outward sides, the
loop taking more than one pass: the premise, directions flip) and at a random clearing of a
twentieth of the buildings, from each start (gap_walls.WarmStart, ColdStart). Fault `one-pass`:
the loop returning after its first pass.
`energy`: under each walls, at the baseline and a random clearing: P = q . u equals the
dissipation, each passage at its current's side's conductance (the current-voltage curve, not the
solver's record) plus the street faces' and the leaks', within ENERGY_TOL; and every unknown cell
conserves current, its imbalance under the same curve within ENERGY_TOL of the largest current
through a cell. Fault `wrong-side`: one asymmetric wall passage (the largest by potential
difference and asymmetry) settled on the wrong side.
`monotone`: under each walls, along MONO_ORDERS random orders of MONO_STEPS buildings cleared one
after another, P never rises (within MONO_TOL). Fault `cleared-slow`: a cleared building's ground
conducting CLEARED_SLOW of open ground's.
`walled-count` (B (i) under W is T's): the walled-in homes a clearing leaves read the same under W
as under T (RUNTIME) at the baseline, the ONE_K single clearings, REACH_RANDOM random clearings at
BUDGET and T's free first, on blocks with walled-in homes. Fault `through-walls`: B (i) read from
W's own reach (through walls: no home walled in).
`sources`: every home has demand under W (none walled out of P) but one with no footprint triangle
in the mesh (a footprint under a barrier's cut), its sources on its own footprint's cells only,
by area, against an oracle (shapely: the triangles whose centroid lies within its polygon) within
SOURCE_TOL, every source cell an unknown at baseline. Fault `doors`: W's sources put at the base's
demand points (its doors: a walled-in home's in its pocket, a home with no open ground none).
`walls-identity`: each walls on UNIFORM, RUNTIME and the per-block metric spells a name that
gap_walls.walls_of reads back to it and gap_clear.complex_of refuses; a name without walls is
refused by walls_of; a diode whose lengths are equal, a k_in over 1 and a zero wall are refused;
gap_walls's public constants are SIDE_REL alone (it moves scores at the floating-point level only,
so it is in no digest). Fault `part-dropped`: WallMetric's name without the walls' part.

The relaxation's guards (stage 2; NOTES, "Walls that leak", stage 2): `relax-gradient` (the
gradient against central differences at SIMP's start and a random design, q 1 and 3; at a grounded
exit's owner the limit against the gradient just below it), `relax-ends` (the relaxation at 0/1
clearings W's state bit for bit), `relax-monotone` (no table value falls as an owner clears, P along
designs never rises), `relax-warm` (the diode warm from the last design settles where the cold
start does), `relax-identity` (W's plan and arms read back, T's eps worlds refused).
`greedy-exact` (stage 3, walls_greedy): under each walls, each lazy greedy's (GREEDIES) every
step's gain is W's exact gain, J before less J after each scored afresh (WallModel.exact), within
GAIN_TOL of J0; it stops at the budget (D within it, no building left that fits what remains).
Faults `stale-gain` (a popped gain taken without re-scoring) and `over-budget` (a candidate's fit
ignoring its cost).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/leaky_check.py
"""
from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import digests  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_factor  # noqa: E402
import gap_walls as gw  # noqa: E402
import leaky_simp  # noqa: E402
import metrics  # noqa: E402
import region_leaks  # noqa: E402
import relax  # noqa: E402
import walls_greedy as wg  # noqa: E402

if TYPE_CHECKING:
    from reblock.contracts import Block

POP = common.PopulationDemand()
SYM = gw.SymmetricWalls(k_in=1.0, ell=200.0)
DIODE = gw.DiodeWalls(k_in=1.0, ell_out=50.0, ell_in=300.0, max_passes=32)
WALLS = (SYM, DIODE)
WARM = gw.WarmStart()    # every model's loops after the baseline (`settles` also runs them cold)
PER_BLOCK = gcl.complex_of(metrics.PER_BLOCK, factorization=gap_factor.SuperLU())
CURRENT_TOL = 1e-8       # a wall group's current from the definition against conservation's
                         # (relative: the refined solves agree to 1e-12 and better)
ENERGY_TOL = 1e-9        # P against the dissipation (relative), a cell's imbalance (of the
                         # largest current through a cell: gap_clear_check.FLUX_TOL)
SOURCE_TOL = 1e-6        # a home's sources against the oracle, of its population: a triangle in a
                         # sliver within the linework's snap (gap_clear.LINE_SNAP_M) of an outline
                         # reads either side by its centroid (8e-8 -- 1.8e-7 on 19570, 20269, 23597)
MONO_ORDERS = 3
MONO_STEPS = 8
MONO_TOL = 1e-10         # P rising by less than this (relative) is roundoff
CLEARED_SLOW = 1e-3      # fault `cleared-slow`: a cleared building's ground's factor
BUDGET = 0.05
ONE_K = 3
REACH_RANDOM = 2
SEED = 20261021
GRAD_STEPS = (1e-3, 1e-4)  # relax-gradient's steps h (each with h / 2, extrapolated: as
                           # gap_clear_check's)
GRAD_TOL = 1e-4          # the gradient against differences, of its largest (gap_clear_check's)
GROUND_STEPS = (1e-6, 5e-7)  # relax-gradient at a grounded exit's owner (x 1): the gradient
                         # just below, at 1 - t, extrapolated to t 0. The ground turns on over
                         # 1 - x ~ Gs / K (the exits' standing conductance over their cell's
                         # others'): on 19570 at 200 m the slope is -45.08 at x 1, -42.7 at
                         # 1 - 1e-4, -45.06 at 1 - 1e-6, so differences of J reach it only at
                         # steps whose roundoff (the refined solves', ~1e-13 of J) swamps them
GRAD_PICK = 6            # buildings checked per design: the largest |gradient| and the owners of
                         # shared zones and of inside street faces
ENDS_TOL = 1e-12         # the relaxation's J at a 0/1 clearing against W's scorer's (relative: the
                         # network is W's bit for bit; a diode's loop from another start ties)
WARM_ITERS = 6           # relax-warm: SIMP's first updates replayed warm and cold
WARM_TOL = 1e-10         # ... their potentials apart (of the largest)
GAIN_TOL = 1e-10         # greedy-exact: a step's gain against W's exact one, of J0 (a diode's
                         # loops from other starts settle to roundoff)
GREEDIES = (wg.Lazy(), wg.LazyLocal(radius_m=10.0))   # greedy-exact's: the lazy and the fit one


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _walls_model(b: Block, metric: gcl.ComplexMetric, walls: gw.SymmetricWalls | gw.DiodeWalls,
                 *, start: gw.ColdStart | gw.WarmStart) -> gw.WallModel:
    """W on `metric`'s mesh of `b`, meshed afresh (a temporary cache), its loops from `start`."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = metric.fixed(b, cache=Path(tmp), carving=gcl.Carving.none(gc.geometry(b)))
    return gw.WallModel(metric.model(b, fx, demand=POP), walls=walls, start=start)


def _uniform() -> gcl.ComplexMetric:
    return metrics.metric_of(metrics.UNIFORM, device="cpu")


# ------------------------------------------------------------------ synthetic blocks

def _strip(block_id: str, width: float, height: float, *, east: bool,
           homes: list[Any]) -> Block:
    """A strip `width` x `height` m (local metres), its east edge the street (`east`) or its west
    one, with `homes`."""
    x = width if east else 0.0
    return region_leaks.synthetic_block(block_id, shapely.box(0.0, 0.0, width, height),
                                        shapely.LineString([(x, 0.0), (x, height)]), homes, None)


H_BOX = (5.0, 4.0, 9.0, 8.0)          # DIRECTION_BLOCK's home H in the west alley
B_BOX = (25.0, 0.0, 35.0, 12.0)       # ... its building B across the strip


def _direction_block() -> Block:
    return _strip("synthetic-walls-direction", 60.0, 12.0, east=True,
                  homes=[shapely.box(*H_BOX), shapely.box(*B_BOX)])


def _party_block(*, east: bool) -> Block:
    return _strip(f"synthetic-walls-party-{'east' if east else 'west'}", 40.0, 10.0, east=east,
                  homes=[shapely.box(5.0, 0.0, 15.0, 10.0), shapely.box(15.5, 0.0, 25.5, 10.0)])


def _passage_current(wm: gw.WallModel, u: np.ndarray, q: np.ndarray, c_far: float,
                     c_back: float, far: np.ndarray) -> np.ndarray:
    """Per static passage `q` its current toward its far side (`far` True: its second cell),
    from the potentials `u` and the definition: sigma |e| / (d_e + c), d_e from the mesh's weight
    (k_in 1, no boost: the guards' blocks), c `c_far` where the current flows toward the far side
    and `c_back` where it flows back."""
    cells = wm.sys.cells
    du = np.where(far, u[cells.ea[q]] - u[cells.eb[q]], u[cells.eb[q]] - u[cells.ea[q]])
    sL = wm.sigma * wm.L[q]
    d = sL / wm.T0[q]
    c = np.where(du > 0, c_far, c_back)
    return du * sL / (d + c)


def _close(got: float, want: float, tol: float) -> bool:
    return abs(got - want) <= tol * max(abs(want), 1e-300)


def direction(_ids: list[str]) -> list[str]:
    b = _direction_block()
    bad = []
    for walls in WALLS:
        wm = _walls_model(b, PER_BLOCK, walls, start=WARM)
        ell_out, ell_in = walls.ell_out, walls.ell_in
        wl = wm.base_
        st, u = wl.state, wl.pop.u
        cells = wm.sys.cells
        hB = [i for i, p in enumerate(wm.base.polys) if shapely.area(p) > 50.0]
        hH = [i for i, p in enumerate(wm.base.polys) if shapely.area(p) <= 50.0]
        if len(hB) != 1 or len(hH) != 1 or not wm.walled0[hH[0]]:
            bad.append(f"{walls.name}: not one building B and one walled-in home H (the premise)")
            continue
        iB, iH = hB[0], hH[0]
        w = st.wall
        za = np.asarray(wm.zone[wm.pa[w]].todense())
        zb = np.asarray(wm.zone[wm.pb[w]].todense())
        mid = 0.5 * (wm.fm.xy[wm.fm.pu[cells.pe[w]]] + wm.fm.xy[wm.fm.pv[cells.pe[w]]])
        groups = {}
        # each group: (its passages, per passage True where its second cell is the far side,
        # the current conservation says crosses it toward the far side); toward the far side a
        # passage of an `in` group enters a building (ell_in), of an `out` group leaves one
        out_H = (za[:, iH] > 0) != (zb[:, iH] > 0)
        groups["H's wall, out"] = (out_H, za[out_H, iH] > 0, wm.pop[iH])
        crossB = (za[:, iB] > 0) != (zb[:, iB] > 0)
        west, east = crossB & (mid[:, 0] < 30.0), crossB & (mid[:, 0] > 30.0)
        groups["B's west wall, in"] = (west, zb[west, iB] > 0, wm.pop[iH])
        groups["B's east wall, out"] = (east, za[east, iB] > 0, wm.pop[iH] + wm.pop[iB])
        parts = []
        for name, (sel, into, want) in groups.items():
            c_far, c_back = (ell_in, ell_out) if name.endswith(", in") else (ell_out, ell_in)
            got = float(_passage_current(wm, u, w[sel], c_far, c_back, into).sum())
            ok = _close(got, float(want), CURRENT_TOL)
            parts.append(f"{name} {got:.6g} of {float(want):.6g}" + ("" if ok else " APART"))
            if not ok:
                bad.append(f"{walls.name}: {name} carries {got:.6g}, conservation {want:.6g}")
        print(f"direction {walls.name} (out {ell_out:g}, in {ell_in:g}; {wl.pop.passes} passes): "
              + "; ".join(parts), flush=True)
    return bad


def party(_ids: list[str]) -> list[str]:
    bad = []
    for walls in WALLS:
        ell_out, ell_in = walls.ell_out, walls.ell_in
        for east in (True, False):
            b = _party_block(east=east)
            wm = _walls_model(b, PER_BLOCK, walls, start=WARM)
            st, u = wm.base_.state, wm.base_.pop.u
            iA, iB = (0, 1) if wm.base.polys[0].bounds[0] < wm.base.polys[1].bounds[0] else (1, 0)
            w = st.wall
            za = np.asarray(wm.zone[wm.pa[w]].todense())
            zb = np.asarray(wm.zone[wm.pb[w]].todense())
            ab = (za[:, iA] > 0) & (zb[:, iB] > 0) & (za[:, iB] == 0) & (zb[:, iA] == 0)
            ba = (za[:, iB] > 0) & (zb[:, iA] > 0) & (za[:, iA] == 0) & (zb[:, iB] == 0)
            sel = ab | ba
            if sel.sum() < 10:
                bad.append(f"{b.block_id}: {int(sel.sum())} party passages (the premise)")
                continue
            # toward B: a passage from A's to B's has its second cell B's
            towards_b = ab[sel]
            cur = _passage_current(wm, u, w[sel], ell_out + ell_in, ell_out + ell_in, towards_b)
            got = float(cur.sum())
            want = float(wm.pop[iA] if east else -wm.pop[iB])
            ok = _close(got, want, CURRENT_TOL)
            print(f"party {walls.name} street {'east' if east else 'west'}: {int(sel.sum())} "
                  f"party passages carry {got:.6g} toward B at ell_out + ell_in, conservation "
                  f"{want:.6g} -> {'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"{walls.name} {b.block_id}: {got:.6g} against {want:.6g}")
    return bad


def _side_errors(wm: gw.WallModel, wl: gw.WallLenses) -> tuple[int, int]:
    """(wall passages whose conductance is not their current's side's beyond SIDE_REL, the
    asymmetric ones checked)."""
    st, s = wl.state, wl.pop
    c = wm.sys.cells
    w = st.wall
    asym = st.g_ab[w] != st.g_ba[w]
    du = s.u[c.ea[w]] - s.u[c.eb[w]]
    tol = gw.SIDE_REL * float(np.abs(s.u).max())
    want = np.where(du > tol, st.g_ab[w], np.where(du < -tol, st.g_ba[w], s.g[w]))
    return int(((s.g[w] != want) & asym).sum()), int(asym.sum())


def settles(ids: list[str]) -> list[str]:
    bad = []
    cases: list[tuple[str, Block, gcl.ComplexMetric]] = [
        ("direction block", _direction_block(), PER_BLOCK)]
    for bid in ids:
        [b] = common.build_blocks([bid])
        cases.append((bid, b, _uniform()))
    rng = np.random.default_rng(SEED + 3)
    for name, b, metric in cases:
        for start in (WARM, gw.ColdStart()):
            wm = _walls_model(b, metric, DIODE, start=start)
            # the baseline (from the outward sides) and a random clearing (from the start's)
            x = np.zeros(wm.n, dtype=bool)
            x[rng.permutation(wm.n)[:max(1, wm.n // 20)]] = True
            for label, wl in (("baseline", wm.base_), ("random", wm.lenses(x))):
                wrong, n = _side_errors(wm, wl)
                ok = wrong == 0 and (label != "baseline" or wl.pop.passes > 1)
                print(f"settles {name} {type(start).__name__} {label}: {wl.pop.passes} passes; "
                      f"{wrong} of {n} asymmetric wall passages off their current's side -> "
                      f"{'ok' if ok else 'FAILS'}", flush=True)
                if not ok:
                    bad.append(f"{name} {label}: {wrong} off side, {wl.pop.passes} passes")
    return bad


def _one_pass(self: gw.DiodeSolver, rhs: np.ndarray) -> gw.Solved:
    """DiodeSolver.solve returning after its first pass (fault `one-pass`)."""
    g = self.conductance(self.side0)
    return gw.Solved(u=self.wm.timed_solve(self.wm.factor(self.st, g), rhs), g=g, passes=1)


def _energy(wm: gw.WallModel, wl: gw.WallLenses) -> tuple[float, float]:
    """(P against the dissipation under the current-voltage curve, relative; the largest cell
    imbalance under the same curve, of the largest current through a cell)."""
    st, s = wl.state, wl.pop
    c = wm.sys.cells
    du = s.u[c.ea] - s.u[c.eb]
    g = np.where(du > 0, st.g_ab, st.g_ba)          # the curve: each side's conductance
    cur = g * du
    leak = wm.sys.graph.leak
    rhs = wm.rhs(st, wm.pop)
    P = float(rhs @ s.u)
    diss = float((cur * du).sum() + ((st.G + leak) * s.u ** 2)[st.unknown].sum())
    out = np.bincount(c.ea, weights=cur, minlength=c.n) - np.bincount(c.eb, weights=cur,
                                                                        minlength=c.n)
    imb = (rhs - out - (st.G + leak) * s.u)[st.unknown]
    through = np.bincount(c.ea, weights=np.abs(cur), minlength=c.n) + np.bincount(
        c.eb, weights=np.abs(cur), minlength=c.n)
    return abs(diss / P - 1.0), float(np.abs(imb).max() / through.max())


def energy(ids: list[str]) -> list[str]:
    bad = []
    cases: list[tuple[str, Block, gcl.ComplexMetric]] = [
        ("direction block", _direction_block(), PER_BLOCK)]
    for bid in ids:
        [b] = common.build_blocks([bid])
        cases.append((bid, b, _uniform()))
    rng = np.random.default_rng(SEED)
    for name, b, metric in cases:
        for walls in WALLS:
            wm = _walls_model(b, metric, walls, start=WARM)
            x = np.zeros(wm.n, dtype=bool)
            order = rng.permutation(wm.n)
            x[order[np.cumsum(wm.cost[order]) <= BUDGET]] = True
            for label, wl in (("baseline", wm.base_), ("random", wm.lenses(x))):
                e, imb = _energy(wm, wl)
                ok = e <= ENERGY_TOL and imb <= ENERGY_TOL
                print(f"energy {name} {walls.name} {label}: P {wl.lenses.P:.8g}, the dissipation "
                      f"{e:.1e} apart, imbalance {imb:.1e} -> {'ok' if ok else 'FAILS'}",
                      flush=True)
                if not ok:
                    bad.append(f"{name} {walls.name} {label}: {e:.1e}, {imb:.1e}")
    return bad


_SIDES = gw.DiodeSolver.sides


def _wrong_side(self: gw.DiodeSolver, u: np.ndarray, side: np.ndarray) -> np.ndarray:
    """DiodeSolver.sides with one asymmetric wall passage (the largest potential difference
    times asymmetry) put on the side its current is not on (fault `wrong-side`)."""
    new = _SIDES(self, u, side)
    w = self.st.wall
    weight = np.abs(u[self.ea] - u[self.eb]) * np.abs(self.st.g_ab[w] - self.st.g_ba[w])
    k = int(np.argmax(weight))
    du = u[self.ea[k]] - u[self.eb[k]]
    new[k] = -1 if du > 0 else 1
    return new


def monotone(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for walls in WALLS:
            wm = _walls_model(b, _uniform(), walls, start=WARM)
            rng = np.random.default_rng(SEED + 1)
            rises = []
            for k in range(MONO_ORDERS):
                order = rng.permutation(wm.n)[:MONO_STEPS]
                x = np.zeros(wm.n, dtype=bool)
                P = [wm.base_.lenses.P]
                for j in order:
                    x[j] = True
                    P.append(wm.lenses(x).lenses.P)
                d = np.diff(P) / np.asarray(P[:-1])
                rises.append(float(d.max()))
                print(f"monotone {bid} {walls.name} order {k}: P {P[0]:.6g} -> {P[-1]:.6g}, the "
                      f"largest step {d.max():+.2e} (relative)", flush=True)
            if max(rises) > MONO_TOL:
                bad.append(f"{bid} {walls.name}: P rises by {max(rises):.2e}")
    return bad


_OPEN_FACTOR = gw.WallModel.open_factor


def _cleared_slow(self: gw.WallModel, cleared: np.ndarray) -> np.ndarray:
    """WallModel.open_factor with a cleared building's ground (open, in a cleared building's
    grown footprint) at CLEARED_SLOW (fault `cleared-slow`)."""
    return np.where(cleared, CLEARED_SLOW, 1.0) * _OPEN_FACTOR(self, cleared)


def walled_count(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        with tempfile.TemporaryDirectory() as tmp:
            rt = metrics.metric_of(metrics.RUNTIME, device="cpu")
            fx = rt.fixed(b, cache=Path(tmp), carving=gcl.Carving.none(gc.geometry(b)))
            mT = rt.model(b, fx, demand=POP)
        if not mT.walled0.any():
            bad.append(f"{bid}: no home walled in (the premise)")
            continue
        wm = _walls_model(b, _uniform(), SYM, start=WARM)
        rng = np.random.default_rng(SEED + 2)
        states = {"baseline": np.zeros(mT.n, dtype=bool)}
        for j in np.argsort(-mT.pop, kind="stable")[:ONE_K]:
            x = np.zeros(mT.n, dtype=bool)
            x[j] = True
            states[f"{int(j)} cleared"] = x
        for k in range(REACH_RANDOM):
            order = rng.permutation(mT.n)
            x = np.zeros(mT.n, dtype=bool)
            x[order[np.cumsum(mT.cost[order]) <= BUDGET]] = True
            states[f"random {k}"] = x
        states["free first"] = mT.free_first(budget=BUDGET, held=np.zeros(mT.n, dtype=bool))
        ok, parts = bool(np.array_equal(wm.walled0, mT.walled0)), []
        for name, x in states.items():
            t = mT.walled0 & ~mT.live_homes_of(mT.factors(mT.k_of(x)))
            L = wm.lenses(x).lenses
            w = wm.walled0 & ~L.live
            same = bool(np.array_equal(t, w)) and L.enclosed == mT.enclosed_left(x)
            ok &= same
            parts.append(f"{name} {int(t.sum())} / {int(w.sum())}" + ("" if same else " APART"))
        print(f"walled-count {bid}: walled in left, T / W: " + ", ".join(parts)
              + f" -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(bid)
    return bad


def _through_walls(self: gw.WallModel, st: gw.WallState) -> np.ndarray:
    """WallModel.street_reach from W's own reach, through walls (fault `through-walls`)."""
    return np.asarray(self.base.fx.own.live(st.live[self.sys.cells.cell]))


def sources(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        wm = _walls_model(b, _uniform(), SYM, start=WARM)
        c, fm = wm.sys.cells, wm.fm
        xy = fm.xy[fm.tri]
        e1, e2 = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
        area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        cen = xy.mean(axis=1)
        # the oracle: each home's footprint (the model's geometry, in the mesh's local frame) and
        # its triangles by their centroids (shapely, not the mesh's cover), its population over
        # them by area
        t, h = shapely.STRtree(wm.base.geo.polys).query(shapely.points(cen[:, 0], cen[:, 1]),
                                                        predicate="within")
        oracle = sp.csr_matrix((area[t], (c.cell[t], h)), shape=(c.n, wm.n))
        tot = np.asarray(oracle.sum(axis=0)).ravel()
        oracle = (oracle @ sp.diags(np.where(tot > 0, 1.0 / np.maximum(tot, 1e-300), 0.0))).tocsr()
        diff = abs(oracle - wm.Pc).max()
        problems = []
        if diff > SOURCE_TOL:
            problems.append(f"placed {diff:.1e} off the footprints' areas")
        if not np.array_equal(wm.nodem, tot <= 0):
            problems.append(f"{int((tot <= 0).sum())} homes with no footprint triangle, but "
                            f"{int(wm.nodem.sum())} without demand")
        src = np.asarray(wm.Pc.sum(axis=1)).ravel() > 0
        if not wm.base_.state.unknown[src].all():
            problems.append(f"{int((src & ~wm.base_.state.unknown).sum())} source cells off "
                            "the unknowns")
        print(f"sources {bid}: {wm.n} homes ({int(wm.walled0.sum())} walled in under T), each on "
              f"its own footprint by area ({diff:.1e} from the oracle): {problems} -> "
              f"{'ok' if not problems else 'FAILS'}", flush=True)
        if problems:
            bad.append(bid)
    return bad


def _doors(self: gw.WallModel) -> tuple[Any, np.ndarray]:
    """WallModel.sources at the base's demand points, its doors (fault `doors`)."""
    return self.base.sys.Pc, np.asarray(self.base.nodem)


def walls_identity(_ids: list[str]) -> list[str]:
    bad = []
    names = []
    for base_name in (metrics.UNIFORM, metrics.RUNTIME, metrics.PER_BLOCK):
        base = gcl.complex_of(base_name, factorization=gap_factor.SuperLU())
        for walls in (*WALLS, gw.SymmetricWalls(k_in=0.5, ell=0.1 + 0.2),
                      gw.DiodeWalls(k_in=1e-05, ell_out=1500.0, ell_in=2.5, max_passes=7)):
            name = gw.WallMetric(base=base, walls=walls).name
            names.append(name)
            try:
                back = gw.walls_of(name, factorization=gap_factor.SuperLU())
                same = back.walls == walls and back.base.name == base_name
            except ValueError:
                same = False
            if not same:
                bad.append(f"{name} does not read back")
            try:
                gcl.complex_of(name, factorization=gap_factor.SuperLU())
                bad.append(f"complex_of accepted {name}")
            except ValueError:
                pass
        try:
            gw.walls_of(base_name, factorization=gap_factor.SuperLU())
            bad.append(f"walls_of accepted {base_name}")
        except ValueError:
            pass
    if len(set(names)) != len(names):
        bad.append("two walls share a name")
    refused: list[Callable[[], object]] = [
        lambda: gw.DiodeWalls(k_in=1.0, ell_out=100.0, ell_in=100.0, max_passes=8),
        lambda: gw.SymmetricWalls(k_in=2.0, ell=100.0),
        lambda: gw.SymmetricWalls(k_in=1.0, ell=0.0)]
    for make in refused:
        try:
            make()
            bad.append("an invalid walls accepted")
        except ValueError:
            pass
    consts = digests.constants_of((gw,), exempt={}, external={})
    if set(consts) != {"gap_walls.SIDE_REL"}:
        bad.append(f"gap_walls's constants {sorted(consts)}, not SIDE_REL alone")
    verdict = "ok" if not bad else "FAILS: " + "; ".join(bad[:3])
    print(f"walls-identity: {len(names)} names read back, e.g. {names[1]}; complex_of refuses "
          f"them; constants {sorted(consts)} -> {verdict}", flush=True)
    return bad


def _name_without_part(self: gw.WallMetric) -> str:
    """WallMetric.name without the walls' part (fault `part-dropped`)."""
    return self.base.name


# ------------------------------------------------------------------ stage 2: the relaxation

def _designs(wm: gw.WallModel, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """SIMP's start (x = D: relax.start_x) and a random design in [0.1, 0.9]."""
    return [("start", relax.start_x(0, wm.cost, BUDGET)), ("random", rng.uniform(0.1, 0.9, wm.n))]


def _pick(wm: gw.WallModel, g: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """GRAD_PICK buildings: the largest |gradient|, an owner of a passage with two or more owners
    and an owner of an inside street face, two of each where there are."""
    it = wm.interp
    shared = np.unique(np.concatenate([o.own.ravel() for o in it.passages if o.own.shape[1] > 1]
                                      + [np.zeros(0, dtype=np.int64)]))
    faced = np.unique(np.concatenate([o.own.ravel() for o in it.faces]
                                     + [np.zeros(0, dtype=np.int64)]))
    out = list(np.argsort(-np.abs(g), kind="stable")[:GRAD_PICK // 3])
    for pool in (shared, faced):
        pool = np.setdiff1d(pool, out)
        out += list(rng.choice(pool, min(GRAD_PICK // 3, len(pool)), replace=False))
    return np.asarray(out, dtype=np.int64)


def _central(rel: gw.WallRelaxation, x0: np.ndarray, g: np.ndarray, pick: np.ndarray) -> float:
    """The gradient `g` at x0 against central differences on `pick` (each the closest over
    GRAD_STEPS of 2 D(h / 2) - D(h)), of the largest |g| among them."""
    err = []
    for j in pick:
        best = np.inf
        for h in GRAD_STEPS:
            d = []
            for step in (h, h / 2.0):
                e = np.zeros(len(x0))
                e[j] = step
                d.append((rel.value(x0 + e) - rel.value(x0 - e)) / (2.0 * step))
            best = min(best, abs(2.0 * d[1] - d[0] - g[j]))
        err.append(best)
    return float(np.max(err) / np.abs(g[pick]).max())


def _limit(rel: gw.WallRelaxation, x0: np.ndarray, g: np.ndarray, j: int) -> float:
    """dJ / dx_j at x0 (x0_j 1: the drain currents' limit) against the gradient just below it,
    at x_j 1 - t for t in GROUND_STEPS (the formula the central differences check, the cell not
    ground), extrapolated to t 0 (2 g(t / 2) - g(t)); of |g_j|."""
    gs = []
    for t in GROUND_STEPS:
        x = x0.copy()
        x[j] = 1.0 - t
        gs.append(float(rel.value_grad(x)[1][j]))
    return float(abs(2.0 * gs[1] - gs[0] - g[j]) / abs(g[j]))


def relax_gradient(ids: list[str]) -> list[str]:
    bad = []
    rng = np.random.default_rng(SEED + 5)
    for bid in ids:
        [b] = common.build_blocks([bid])
        for walls in WALLS:
            wm = _walls_model(b, _uniform(), walls, start=WARM)
            rel = wm.relaxation(1.0, q=1.0, eps=clear.EPS)
            for q in (1.0, 3.0):
                rel.q = q
                for label, x0 in _designs(wm, rng):
                    _J, g = rel.value_grad(x0)
                    err = _central(rel, x0, g, _pick(wm, g, rng))
                    ok = err <= GRAD_TOL
                    print(f"relax-gradient {bid} {walls.name} q {q:g} {label}: {err:.1e} of the "
                          f"largest -> {'ok' if ok else 'FAILS'}", flush=True)
                    if not ok:
                        bad.append(f"{bid} {walls.name} q {q:g} {label}: {err:.1e}")
            # an owned obtuse exit's cell ground (every owner of every owned exit cleared, x 1):
            # the limit, against the gradient just below it
            owners = np.unique(np.concatenate([o.own.ravel() for o in wm.interp.exits]
                                              + [np.zeros(0, dtype=np.int64)]))
            if len(owners) == 0:
                bad.append(f"{bid}: no owned obtuse exit (the premise)")
                continue
            rel.q = 1.0
            x0 = rng.uniform(0.1, 0.9, wm.n)
            x0[owners] = 1.0
            _J, g = rel.value_grad(x0)
            j = int(owners[np.argmax(np.abs(g[owners]))])
            err = _limit(rel, x0, g, j)
            ok = err <= GRAD_TOL
            print(f"relax-gradient {bid} {walls.name} ground: building {j} at x 1, "
                  f"{int((wm.interp.ocell & wm.relaxed(rel.rho(x0)).ground).sum())} owned cells "
                  f"ground, {err:.1e} of its gradient -> {'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"{bid} {walls.name} ground: {err:.1e}")
    return bad


def _faces_dropped(self: gw.WallModel, net: gw.Relaxed, s: gw.Solved,
                   lam: np.ndarray) -> np.ndarray:
    """WallModel.grad_faces left out (fault `no-faces`)."""
    return np.zeros(self.n)


def _forward_only(self: gw.WallModel, net: gw.Relaxed, s: gw.Solved,
                  lam: np.ndarray) -> np.ndarray:
    """WallModel.grad_passages with every passage's slope its first-to-second table's, whatever
    its current's side (fault `forward-only`)."""
    c, u = self.sys.cells, s.u
    wp = -(u[c.ea] - u[c.eb]) * (lam[c.ea] - lam[c.eb])
    out = np.zeros(self.n)
    for o in self.interp.passages:
        sl = gw._slopes(o.ab, net.rho[o.own])
        out += np.bincount(o.own.ravel(), weights=(wp[o.pos][:, None] * sl).ravel(),
                           minlength=self.n)
    return out


def _drain_zero(self: gw.WallModel, g: np.ndarray, u: np.ndarray, b: np.ndarray) -> np.ndarray:
    """WallModel.drain nothing: a ground cell's term 0 (fault `ground-zero`)."""
    return np.zeros(self.sys.cells.n)


def _clearings(wm: gw.WallModel, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
    """The baseline, a random clearing within BUDGET, one owner of each shared passage of two
    owners cleared (shared zones half cleared) and every building cleared."""
    order = rng.permutation(wm.n)
    rand = np.zeros(wm.n, dtype=bool)
    rand[order[np.cumsum(wm.cost[order]) <= BUDGET]] = True
    half = np.zeros(wm.n, dtype=bool)
    for o in wm.interp.passages:
        if o.own.shape[1] == 2:
            half[o.own[:, 0]] = True
            half[o.own[:, 1]] = False
    return [("baseline", np.zeros(wm.n, dtype=bool)), ("random", rand), ("half", half),
            ("all", np.ones(wm.n, dtype=bool))]


def relax_ends(ids: list[str]) -> list[str]:
    bad = []
    rng = np.random.default_rng(SEED + 6)
    for bid in ids:
        [b] = common.build_blocks([bid])
        for walls in WALLS:
            wm = _walls_model(b, _uniform(), walls, start=WARM)
            rel = wm.relaxation(1.0, q=3.0, eps=clear.EPS)
            for label, r in _clearings(wm, rng):
                net, st = wm.relaxed(r.astype(np.float64)), wm.state(r)
                same = [k for k, a, z in (("g_ab", net.g_ab, st.g_ab), ("g_ba", net.g_ba, st.g_ba),
                                          ("Gf", net.Gf, st.Gf), ("G", net.G, st.G),
                                          ("ground", net.ground, st.ground),
                                          ("unknown", net.unknown, st.unknown))
                        if not np.array_equal(a, z)]
                Jr, Je = rel.value(r.astype(np.float64)), wm.exact(r, 1.0)[0]
                apart = abs(Jr / Je - 1.0)
                ok = not same and apart <= ENDS_TOL
                verdict = "W's bit for bit" if not same else "apart on " + ", ".join(same)
                print(f"relax-ends {bid} {walls.name} {label} ({int(r.sum())} cleared): network "
                      f"{verdict}, J {Jr:.10g} against W's {Je:.10g} ({apart:.1e}) -> "
                      f"{'ok' if ok else 'FAILS'}", flush=True)
                if not ok:
                    bad.append(f"{bid} {walls.name} {label}: {same} {apart:.1e}")
    return bad


_WEIGHTS = gw._weights


def _owners_reversed(r: np.ndarray) -> np.ndarray:
    """gap_walls._weights reading each element's owners in reverse order (fault
    `owners-reversed`)."""
    return _WEIGHTS(r[:, ::-1])


def relax_monotone(ids: list[str]) -> list[str]:
    bad = []
    rng = np.random.default_rng(SEED + 7)
    for bid in ids:
        [b] = common.build_blocks([bid])
        for walls in WALLS:
            wm = _walls_model(b, _uniform(), walls, start=WARM)
            it = wm.interp
            # every element's values rise as an owner clears (an obtuse exit's resistance falls)
            drops = 0
            for groups, sign in ((it.passages, 1.0), (it.faces, 1.0), (it.exits, -1.0)):
                for o in groups:
                    k = o.own.shape[1]
                    s = np.arange(1 << k)
                    for i in range(k):
                        lo = s[((s >> i) & 1) == 0]
                        for tab in (o.ab, o.ba):
                            d = sign * (tab[:, lo | (1 << i)] - tab[:, lo])
                            drops += int((d < 0).sum())
            # P along designs: MONO_STEPS buildings raised 0 -> 0.5 -> 1 one after another, the
            # rest at a random design
            rel = wm.relaxation(1.0, q=3.0, eps=clear.EPS)
            rises = []
            for _k in range(MONO_ORDERS):
                x = rng.uniform(0.0, 0.6, wm.n)
                order = rng.permutation(wm.n)[:MONO_STEPS]
                x[order] = 0.0
                P = [rel.value(x)]
                for j in order:
                    for v in (0.5, 1.0):
                        x[j] = v
                        P.append(rel.value(x))
                rises.append(float((np.diff(P) / np.asarray(P[:-1])).max()))
            ok = drops == 0 and max(rises) <= MONO_TOL
            print(f"relax-monotone {bid} {walls.name}: {drops} table steps down; P's largest step "
                  f"{max(rises):+.2e} (relative) -> {'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"{bid} {walls.name}: {drops} steps down, P rises {max(rises):.2e}")
    return bad


def relax_warm(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        wm = _walls_model(b, _uniform(), DIODE, start=WARM)
        # SIMP's first updates (q 1.5), their designs replayed: each solved warm (the relaxation
        # from the last design's sides) and cold (from the outward sides)
        xs: list[np.ndarray] = []

        def keep(x: np.ndarray, stage: relax.Stage, into: list[np.ndarray] = xs) -> None:
            into.append(x.copy())

        rel = wm.relaxation(1.0, q=1.0, eps=clear.EPS)
        relax.simp(rel, relax.start_x(0, wm.cost, BUDGET), BUDGET,
                   (relax.Stage(q=1.5, beta=0.0, eps=clear.EPS),), WARM_ITERS,
                   relax.DEFAULT_UPDATE, log=lambda s: None, watch=keep)
        rep = wm.relaxation(1.0, q=1.5, eps=clear.EPS)
        worst, wrong, passes = 0.0, 0, []
        for x in xs:
            net, warm, _uh = rep.solved(x)
            cold = wm.walls.solver(wm, net, np.zeros(len(wm.sys.cells.pe), dtype=np.int8)).solve(
                wm.rhs(net, wm.inj))
            worst = max(worst, float(np.abs(warm.u - cold.u).max() / np.abs(cold.u).max()))
            c = wm.sys.cells
            w = net.wall
            du = warm.u[c.ea[w]] - warm.u[c.eb[w]]
            tol = gw.SIDE_REL * float(np.abs(warm.u).max())
            want = np.where(du > tol, net.g_ab[w], np.where(du < -tol, net.g_ba[w], warm.g[w]))
            wrong += int(((warm.g[w] != want) & (net.g_ab[w] != net.g_ba[w])).sum())
            passes.append((warm.passes, cold.passes))
        ok = worst <= WARM_TOL and wrong == 0 and len(xs) > 0
        print(f"relax-warm {bid} {DIODE.name}: {len(xs)} of SIMP's designs; warm against cold "
              f"{worst:.1e} apart, {wrong} wall passages off their current's side; passes (warm, "
              f"cold) {passes} -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(f"{bid}: {worst:.1e} apart, {wrong} off side")
    return bad


_RELAXATION = gw.WallModel.relaxation


def _eps_accepted(self: gw.WallModel, power: float, *, q: float,
                  eps: float) -> gw.WallRelaxation:
    """WallModel.relaxation accepting any eps (fault `eps-accepted`)."""
    return _RELAXATION(self, power, q=q, eps=clear.EPS)


def relax_identity(_ids: list[str]) -> list[str]:
    bad = []
    plan = relax.plan_of(leaky_simp.W_PLAN)
    if plan.name != leaky_simp.W_PLAN or not isinstance(plan.free, relax.NoFree) or \
            plan.polish_scoring.suffix != "E" or plan.kscore != 0.0 or plan.eps != clear.EPS:
        bad.append(f"W's plan {leaky_simp.W_PLAN!r} reads back as {plan.name!r}")
    for label in leaky_simp.ARMS:
        arm = leaky_simp.w_arm(label)
        back = gw.walls_of(arm.metric.name, factorization=gap_factor.SuperLU())
        path = leaky_simp.arm_path(arm, "x")
        if back.walls != arm.metric.walls or back.name != arm.metric.name or \
                f"+{leaky_simp.WALLS[arm.walls]}+" not in arm.metric.name or \
                leaky_simp.optimizer_of(arm.opt.name) != arm.opt or \
                path.parts[-4:-1] != (arm.metric.name, arm.opt.name, "D0.05"):
            bad.append(f"{arm.label}: {arm.metric.name} / {path} does not round-trip")
    wm = _walls_model(_direction_block(), PER_BLOCK, SYM, start=WARM)
    refused: list[Callable[[], object]] = [
        lambda: wm.relaxation(1.0, q=1.0, eps=relax.plan_of(relax.SIMP_DEFAULT).eps),
        lambda: wm.relaxation(1.0, q=1.0, eps=clear.EPS).scorer(
            relax.plan_of(relax.SIMP_DEFAULT).kscore)]
    for k, make in enumerate(refused):
        try:
            make()
            bad.append(f"T's eps world {('`.e`', '`s`')[k]} accepted under W")
        except ValueError:
            pass
    sc = wm.relaxation(1.0, q=1.0, eps=clear.EPS).scorer(plan.kscore)
    if sc.q != 1.0 or sc.beta != 0.0:
        bad.append("W's scorer not at q 1, unprojected")
    verdict = "ok" if not bad else "FAILS: " + "; ".join(bad[:3])
    print(f"relax-identity: W's plan {plan.name}; {len(leaky_simp.ARMS)} arms' names and rows "
          f"read back; T's eps worlds refused -> {verdict}", flush=True)
    return bad


# ------------------------------------------------------------------ stage 3: the greedy

def greedy_exact(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for walls, rule in ((w, r) for w in WALLS for r in GREEDIES):
            wm = _walls_model(b, _uniform(), walls, start=WARM)
            res = wg.run(wm, rule, budget=BUDGET, power=1.0)
            x = np.zeros(wm.n, dtype=bool)
            J = wm.exact(x, 1.0)[0]
            worst = 0.0
            for st in res.steps:
                x[st.building] = True
                J1 = wm.exact(x, 1.0)[0]
                worst = max(worst, abs(st.gain - (J - J1)) / res.J0)
                J = J1
            D = float(wm.cost[x].sum())
            fits = int(((~x) & (wm.cost <= BUDGET - D)).sum())
            ok = len(res.steps) > 0 and worst <= GAIN_TOL and D <= BUDGET + 1e-12 and fits == 0
            print(f"greedy-exact {bid} {walls.name} {rule.name}: {len(res.steps)} steps, "
                  f"{res.scores} scores; "
                  f"gains against W's exact {worst:.1e} of J0; D {D:.5f}, {fits} buildings "
                  f"left that fit -> {'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"{bid} {walls.name} {rule.name}: gains {worst:.1e}, D {D:.5f}, "
                           f"{fits} fit")
    return bad


def _never_stale(self: wg.Lazy | wg.LazyLocal, scored_at: int, step: int) -> bool:
    """walls_greedy's lazy greedies' `stale` never: a popped gain taken as scored (fault
    `stale-gain`)."""
    return False


@contextmanager
def _stale_gain() -> Iterator[None]:
    with _patched(wg.Lazy, "stale", _never_stale), _patched(wg.LazyLocal, "stale", _never_stale):
        yield


def _fits_any(sc: wg.Scorer, j: int, left: float) -> bool:
    """walls_greedy._fits ignoring the cost (fault `over-budget`)."""
    return not bool(sc.removed[j])


GUARDS: dict[str, Callable[[list[str]], list[str]]] = {
    "direction": direction, "party": party, "settles": settles, "energy": energy,
    "monotone": monotone, "walled-count": walled_count, "sources": sources,
    "walls-identity": walls_identity, "relax-gradient": relax_gradient, "relax-ends": relax_ends,
    "relax-monotone": relax_monotone, "relax-warm": relax_warm, "relax-identity": relax_identity,
    "greedy-exact": greedy_exact}


def _diode_costs_swapped(self: gw.DiodeWalls, leave: np.ndarray, enter: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
    """DiodeWalls.costs charging ell_in to leave and ell_out to enter (fault `in-out-swapped`)."""
    lv, en = leave.astype(np.float64), enter.astype(np.float64)
    return self.ell_in * lv + self.ell_out * en, self.ell_in * en + self.ell_out * lv


def _diode_costs_one_side(self: gw.DiodeWalls, leave: np.ndarray, enter: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """DiodeWalls.costs charging a party passage only ell_out from its first triangle to its
    second (fault `one-side`)."""
    lv, en = leave.astype(np.float64), enter.astype(np.float64)
    return self.ell_out * lv + self.ell_in * en * (1.0 - lv), self.ell_out * en + self.ell_in * lv


FAULTS: dict[str, dict[str, Callable[[], AbstractContextManager[None]]]] = {
    "direction": {"in-out-swapped": lambda: _patched(gw.DiodeWalls, "costs",
                                                     _diode_costs_swapped)},
    "party": {"one-side": lambda: _patched(gw.DiodeWalls, "costs", _diode_costs_one_side)},
    "settles": {"one-pass": lambda: _patched(gw.DiodeSolver, "solve", _one_pass)},
    "energy": {"wrong-side": lambda: _patched(gw.DiodeSolver, "sides", _wrong_side)},
    "monotone": {"cleared-slow": lambda: _patched(gw.WallModel, "open_factor", _cleared_slow)},
    "walled-count": {"through-walls": lambda: _patched(gw.WallModel, "street_reach",
                                                       _through_walls)},
    "sources": {"doors": lambda: _patched(gw.WallModel, "sources", _doors)},
    "walls-identity": {"part-dropped": lambda: _patched(gw.WallMetric, "name",
                                                        property(_name_without_part))},
    "relax-gradient": {"no-faces": lambda: _patched(gw.WallModel, "grad_faces", _faces_dropped),
                       "forward-only": lambda: _patched(gw.WallModel, "grad_passages",
                                                        _forward_only),
                       "ground-zero": lambda: _patched(gw.WallModel, "drain", _drain_zero)},
    "relax-ends": {"owners-reversed": lambda: _patched(gw, "_weights", _owners_reversed)},
    "relax-monotone": {"cleared-slow": lambda: _patched(gw.WallModel, "open_factor",
                                                        _cleared_slow)},
    "relax-warm": {"one-pass": lambda: _patched(gw.DiodeSolver, "solve", _one_pass)},
    "relax-identity": {"eps-accepted": lambda: _patched(gw.WallModel, "relaxation",
                                                        _eps_accepted),
                       "part-dropped": lambda: _patched(gw.WallMetric, "name",
                                                        property(_name_without_part))},
    "greedy-exact": {"stale-gain": _stale_gain,
                     "over-budget": lambda: _patched(wg, "_fits", _fits_any)},
}


def main(guard: str, ids: list[str], fault: str | None) -> None:
    try:
        if fault is None:
            bad = GUARDS[guard](ids)
        else:
            with FAULTS[guard][fault]():
                bad = GUARDS[guard](ids)
    except (RuntimeError, ValueError) as e:
        bad = [f"raised {e!r}"]
    verdict = "FAILS on " + ", ".join(bad) if bad else "passes"
    print(f"{guard}{'' if fault is None else f' under the fault {fault}'}: {verdict}", flush=True)
    if bad and fault is None:
        raise SystemExit(1)
    if not bad and fault is not None:
        raise SystemExit(f"the fault {fault} went unseen: the guard guards nothing")


def both_ways(guards: list[str], ids: list[str]) -> None:
    wrong = []
    for guard in guards:
        for fault in (None, *FAULTS[guard]):
            try:
                main(guard, ids, fault)
            except SystemExit as e:
                wrong.append(f"{guard}{'' if fault is None else f' under {fault}'}: {e.code}")
    print(f"both ways, {', '.join(guards)} on {', '.join(ids)}: "
          + ("as expected" if not wrong else "NOT as expected: " + "; ".join(wrong)), flush=True)
    if wrong:
        raise SystemExit(1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "both-ways" and set(a[1].split(",")) <= set(GUARDS):
        both_ways(a[1].split(","), [] if a[2] == "-" else common.ids_of(a[2]))
        raise SystemExit(0)
    if len(a) not in (2, 3) or a[0] not in GUARDS:
        raise SystemExit(__doc__)
    main(a[0], [] if a[1] == "-" else common.ids_of(a[1]), a[2] if len(a) == 3 else None)
