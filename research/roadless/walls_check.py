"""Guards of walls that leak (gap_walls; NOTES, "Walls that leak: every home's source inside it,
each wall priced as a length"), each with its fault injection (a patch in one process, no file
edited): a guard passes on the code as it is and must fail under its fault (gap_clear_check.py's
convention). Every guard meshes afresh (a cache of its own, in a temporary directory).

    S=research/roadless/walls_check.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S direction - [fault]          # fault: in-out-swapped
    uv run python $S party - [fault]              # fault: one-side
    uv run python $S settles <ids,> [fault]       # fault: one-pass
    uv run python $S energy <ids,> [fault]        # fault: wrong-side
    uv run python $S monotone <ids,> [fault]      # fault: cleared-slow
    uv run python $S walled-count <ids,> [fault]  # fault: through-walls
    uv run python $S sources <ids,> [fault]       # fault: doors
    uv run python $S walls-identity - [fault]     # fault: part-dropped
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
SIDE_REL), on DIRECTION_BLOCK and each block named, the loop taking more than one pass (the premise:
directions flip). Fault `one-pass`: the loop returning after its first pass.
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
`sources`: every home has demand under W (none walled out of P), its sources on its own footprint's
cells only (by the cells' triangles' cover), summing to its population, by area within
CURRENT_TOL, every source cell an unknown at baseline. Fault `doors`: W's sources put at the base's
demand points (its doors: a walled-in home's in its pocket, a home with no open ground none).
`walls-identity`: each walls on UNIFORM, RUNTIME and the per-block metric spells a name that
gap_walls.walls_of reads back to it and gap_clear.complex_of refuses; a name without walls is
refused by walls_of; a diode whose lengths are equal, a k_in over 1 and a zero wall are refused;
gap_walls's public constants are SIDE_REL alone (it moves scores at the floating-point level only,
so it is in no digest). Fault `part-dropped`: WallMetric's name without the walls' part.

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/walls_check.py
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
import common  # noqa: E402
import digests  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_factor  # noqa: E402
import gap_walls as gw  # noqa: E402
import metrics  # noqa: E402
import region_leaks  # noqa: E402

if TYPE_CHECKING:
    from reblock.contracts import Block

POP = common.PopulationDemand()
SYM = gw.SymmetricWalls(k_in=1.0, ell=200.0)
DIODE = gw.DiodeWalls(k_in=1.0, ell_out=50.0, ell_in=300.0, max_passes=32)
WALLS = (SYM, DIODE)
PER_BLOCK = gcl.complex_of(metrics.PER_BLOCK, factorization=gap_factor.SuperLU())
CURRENT_TOL = 1e-8       # a wall group's current from the definition against conservation's
                         # (relative: the refined solves agree to 1e-12 and better)
ENERGY_TOL = 1e-9        # P against the dissipation (relative), a cell's imbalance (of the
                         # largest current through a cell: gap_clear_check.FLUX_TOL)
MONO_ORDERS = 3
MONO_STEPS = 8
MONO_TOL = 1e-10         # P rising by less than this (relative) is roundoff
CLEARED_SLOW = 1e-3      # fault `cleared-slow`: a cleared building's ground's factor
BUDGET = 0.05
ONE_K = 3
REACH_RANDOM = 2
SEED = 20261021


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _walls_model(b: Block, metric: gcl.ComplexMetric,
                 walls: gw.SymmetricWalls | gw.DiodeWalls) -> gw.WallModel:
    """W on `metric`'s mesh of `b`, meshed afresh (a temporary cache)."""
    with tempfile.TemporaryDirectory() as tmp:
        fx = metric.fixed(b, cache=Path(tmp), carving=gcl.Carving.none(gc.geometry(b)))
    return gw.WallModel(metric.model(b, fx, demand=POP), walls=walls)


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
        wm = _walls_model(b, PER_BLOCK, walls)
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
            wm = _walls_model(b, PER_BLOCK, walls)
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
    for name, b, metric in cases:
        wm = _walls_model(b, metric, DIODE)
        wl = wm.base_
        wrong, n = _side_errors(wm, wl)
        ok = wrong == 0 and wl.pop.passes > 1
        print(f"settles {name}: {wl.pop.passes} passes; {wrong} of {n} asymmetric wall passages "
              f"off their current's side -> {'ok' if ok else 'FAILS'}", flush=True)
        if not ok:
            bad.append(f"{name}: {wrong} off side, {wl.pop.passes} passes")
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
            wm = _walls_model(b, metric, walls)
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
            wm = _walls_model(b, _uniform(), walls)
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


def _cleared_slow(self: gw.WallModel, standing: np.ndarray) -> np.ndarray:
    """WallModel.open_factor with a cleared building's ground (open, in a cleared building's
    grown footprint) at CLEARED_SLOW (fault `cleared-slow`)."""
    cleared = np.asarray(self.zone @ (~standing).astype(np.float64)).ravel() > 0
    return np.where(cleared, CLEARED_SLOW, 1.0) * _OPEN_FACTOR(self, standing)


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
        wm = _walls_model(b, _uniform(), SYM)
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
    return np.asarray(self.base.fx.ring.live(st.live[self.sys.cells.cell]))


def sources(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        wm = _walls_model(b, _uniform(), SYM)
        c, fm = wm.sys.cells, wm.fm
        xy = fm.xy[fm.tri]
        e1, e2 = xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 0]
        area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        cen = xy.mean(axis=1)
        # the oracle: each home's footprint's triangles by their centroids (shapely, not the
        # mesh's cover), its population over them by area
        t, h = shapely.STRtree(wm.base.polys).query(shapely.points(cen[:, 0], cen[:, 1]),
                                                    predicate="within")
        oracle = sp.csr_matrix((area[t], (c.cell[t], h)), shape=(c.n, wm.n))
        tot = np.asarray(oracle.sum(axis=0)).ravel()
        oracle = (oracle @ sp.diags(np.where(tot > 0, 1.0 / np.maximum(tot, 1e-300), 0.0))).tocsr()
        diff = abs(oracle - wm.Pc).max()
        problems = []
        if diff > CURRENT_TOL:
            problems.append(f"placed {diff:.1e} off the footprints' areas")
        if (tot <= 0).any() or wm.nodem.any():
            problems.append(f"{int((tot <= 0).sum())} homes with no footprint triangle, "
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


GUARDS: dict[str, Callable[[list[str]], list[str]]] = {
    "direction": direction, "party": party, "settles": settles, "energy": energy,
    "monotone": monotone, "walled-count": walled_count, "sources": sources,
    "walls-identity": walls_identity}


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
