"""Guards of the effective-width study's conductances (lifted.CellConductance, width.py; NOTES,
"Uniform conductance with an effective width"), each with its fault injection (a patch in one
process, no file edited): a guard passes on the code as it is and must fail under its fault.

    S=research/roadless/width_check.py   # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S reference <ids,>    # (run once, at bd1704dc: writes REFERENCE)
    uv run python $S identity <ids,> [fault]
    uv run python $S operator <ids,> [fault]
    uv run python $S gradient <ids,> [fault]
    uv run python $S convex <ids,> [fault]

`identity`: U (lifted.OpenFraction) under the face aperture reproduces the code that adopted it
bit for bit. `numbers` computes, on the CPU with every system seeded alike (pyamg's setup draws
from numpy's global RNG), the scorer's baseline (P0, J_2's, every home's escape time, the
injection field, the placement), a random clearing's operator (the CSR arrays), P, J_2 and
P_{1/2}, the relaxation's value and gradient (SIMP q 3 on J_2 and q 1 on P, eps 1e-4), the
greedy's tension, current_map.flow's fields, and the sightline metric's (ss100k2n2r30) baseline
and clearing; each a float's repr or an array's SHA-256. REFERENCE holds them as computed at
bd1704dc, where the step rule was a Strategy (`reference`, run there with `_params` selecting
the face aperture, steps=lifted.steps_of("apwfs"), uni+apwfs). Fault `ulp`: OpenFraction's
factor moved up by one ulp.

`operator`: the operator under U is the one the open fractions give: every turning edge's weight
its cell's open fraction times h^2 / (ell^2 dtheta), assembled here (`_turning_edges`) and
compared with lifted.edges's turning families bit for bit, and every sub-sample's factor, which
the steps' tracks read, its openness (`_openness`: open 1, outside the block 0, on a footprint
its cell's cleared share of its footprint) against OpenFraction.sub_factor within OPENNESS_TOL;
on the baseline and a random clearing. Fault `b1`: OpenFraction's factor B_1's
(lifted.Blurred(1)).

`gradient`: the relaxation's gradient under B_r (relax.Relaxation.value_sgrad with Params.cells
Blurred(r) for r in GRAD_R), SIMP's q 1 and 3, P and J_2, eps 1e-4, at a random fractional
clearing, against central differences (per building the closer of those at the FD_STEPS, solves
to FD_RTOL from zero) on GRAD_BUILDINGS buildings: the largest difference within GRAD_TOL of the
largest |gradient| checked. Fault `no-blur-vjp`: Blurred's vjp without the disk's term (only the
sub-samples whose own openness binds the min).

`convex`: P under B_r is convex in the clearing fractions (q 1), because every cell's factor is
concave in them: along CONVEX_LINES random segments x(t) = a + t (z - a) across the box, at
CONVEX_POINTS evenly spaced t, every cell factor's second difference at most CONVEX_TOL and P's
at least -CONVEX_TOL of P (solves to FD_RTOL). Fault `product`: each sub-sample's openness
times B (a product of two increasing affine functions: convex along a line, not concave) in
place of their minimum (the two agree at every 0/1 state).
"""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import current_map  # noqa: E402
import lifted  # noqa: E402
import relax  # noqa: E402

REFERENCE = HERE / "width_rows" / "identity_reference_fws.csv"
SEED = 20261013
BUDGET = relax.D_LENS
EPS = 1e-4                # SIMP base's eps world (relax.plan_of(...e0.0001...))
SIGHTLINE = "ss100k2n2r30"
POP = common.PopulationDemand()


def _params(along: lifted.AlongConductance) -> lifted.Params:
    """The metric's physics on the CPU (relax._clearing's), U's cells. (At bd1704dc, for
    `reference`, the same with steps=lifted.steps_of("apwfs"): the step rule was a Strategy.)"""
    return lifted.Params(3.0, 8, along=along, solver=lifted.CpuAMG(), cells=lifted.OpenFraction())


@contextmanager
def _seeded_systems() -> Iterator[None]:
    """Every lifted.System set up after numpy's global RNG is seeded SEED: the same field gives
    the same solve, bit for bit."""
    init = lifted.System.__init__

    def seeded(self, *a, **k):
        np.random.seed(SEED)
        init(self, *a, **k)
    lifted.System.__init__ = seeded
    try:
        yield
    finally:
        lifted.System.__init__ = init


@contextmanager
def _patched(owner: object, name: str, value: object) -> Iterator[None]:
    old = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, old)


def _sha(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _random_clearing(c: clear.Clearing, rng: np.random.Generator) -> np.ndarray:
    """A random set of buildings within BUDGET (by Clearing.cost), in a random order."""
    x = np.zeros(c.n, dtype=bool)
    left = BUDGET
    for j in rng.permutation(c.n):
        if c.cost[j] <= left:
            x[j] = True
            left -= c.cost[j]
    return x


def _clearing(b, along: lifted.AlongConductance, demand: common.Demand) -> clear.Clearing:
    return clear.Clearing(b, lifted.UniformMesh(0.5, offset=lifted.OFFSET), _params(along),
                          population=common.POPULATIONS["area"], demand=demand)


def numbers(bid: str) -> dict[str, str]:
    """The U path's numbers on `bid` (the module's docstring, `identity`)."""
    out: dict[str, str] = {}
    [b] = common.build_blocks([bid])
    with _seeded_systems():
        c = _clearing(b, lifted.Uniform(), POP)
        sc, g, p = c.sc, c.sc.grid, c.p
        out.update(P0=repr(sc.P0), J2_0=repr(sc.J(sc.u0, 2.0)), u0=_sha(sc.u0), f=_sha(sc.f),
                   placement=_sha(sc.pl.S.data, sc.pl.S.indices, sc.pl.S.indptr, sc.enclosed))
        x = _random_clearing(c, np.random.default_rng(SEED))
        op = c.open_(x)
        sy = lifted.System(g, op, p)
        out["A_clearing"] = _sha(sy.A.data, sy.A.indices, sy.A.indptr)
        sol = lifted.solve(g, op, sc.f, p, system=sy, rtol=clear.RTOL_SCORE)
        u = sc.home_u_of(sol, op)
        out.update(P=repr(sol.P), J2=repr(sc.perm_p(u, 2.0)), u=_sha(u))
        sy0 = lifted.System(g, sc.free0, p)
        out["A_baseline"] = _sha(sy0.A.data, sy0.A.indices, sy0.A.indptr)
        cq = _clearing(b, lifted.Uniform(), common.PriorityDemand(0.5))
        out.update(Pq_P0=repr(cq.sc.P0), Pq=repr(cq.sc.P_free(cq.open_(x))))
        x0 = np.random.default_rng(SEED + 1).uniform(0.05, 0.3, c.n)
        for power, q in ((2.0, 3.0), (1.0, 1.0)):
            rel = relax.Relaxation(c, power, q=q, rtol=1e-9, eps=EPS)
            J, grad = rel.value_sgrad(x0)
            out[f"relax_p{power:g}_q{q:g}"] = repr(J) + " " + _sha(grad)
        c.removed[:] = x
        out["tension"] = _sha(c.tension(2.0, restore=False).g)
        c.removed[:] = False
        f = current_map.flow(sc, sc.free0, p)
        out["flow"] = repr(f.P) + " " + _sha(f.phi, f.power, f.u)
        cs = _clearing(b, lifted.along_of(SIGHTLINE, lifted.CpuScans()), POP)
        out.update(ss_P0=repr(cs.sc.P0), ss_P=repr(cs.sc.P_free(cs.open_(x))),
                   ss_layers=_sha(np.asarray(cs.p.along.layers(g, cs.sc.free0, 8))))
    return out


def reference(ids: list[str]) -> None:
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    rows = [(bid, k, v) for bid in ids for k, v in numbers(bid).items()]
    with open(REFERENCE, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(("block", "quantity", "value"))
        w.writerows(rows)
    print(f"{REFERENCE}: {len(rows)} numbers on {len(ids)} blocks", flush=True)


def identity(ids: list[str]) -> list[str]:
    with open(REFERENCE, newline="") as fh:
        ref: dict[str, dict[str, str]] = {}
        for r in csv.DictReader(fh):
            ref.setdefault(r["block"], {})[r["quantity"]] = r["value"]
    bad = []
    for bid in ids:
        got, want = numbers(bid), ref[bid]
        if set(got) != set(want):
            raise SystemExit(f"{bid}: quantities {sorted(set(got) ^ set(want))} on one side only")
        off = [k for k in want if got[k] != want[k]]
        print(f"{bid}: {len(want) - len(off)} of {len(want)} numbers as at bd1704dc, bit for bit"
              + (f"; differ: {off}" if off else ""), flush=True)
        if off:
            bad.append(bid)
    return bad


def _turning_edges(g: lifted.Grid, o: np.ndarray, p: lifted.Params):
    """The turning families as the open fractions define them, assembled here (not by lifted):
    in every open cell, (k, k+1)'s weight its open fraction times h^2 / (ell^2 dtheta)."""
    of = o.ravel()
    _v, _th, _m, gap = lifted.axes(p.K)
    vol = of[of > 0]
    for k in range(p.K):
        yield np.arange(len(vol)), np.arange(len(vol)), vol * (g.h * g.h / (p.ell_m ** 2 * gap[k]))


OPENNESS_TOL = 1e-15     # lifted computes a footprint sub-sample's openness as 1 - (1 - share)


def _openness(c: clear.Clearing, removed: np.ndarray) -> np.ndarray:
    """(cells, S*S) each sub-sample's openness at the 0/1 state `removed`, from the geometry (not
    lifted's): 1 inside the block and off every footprint, 0 outside it, and on a footprint its
    cell's cleared share of its footprint sub-samples (the cleared part of a cell spread evenly
    over its footprint: exact for a cell under one building)."""
    g = c.sc.grid
    n = g.inside.size
    isub = g.isub.reshape(n, -1)
    cov = isub & g.bsub.reshape(n, -1)
    lab = c.lab.reshape(n, -1)
    gone = cov & (lab >= 0) & removed[np.maximum(lab, 0)]
    share = gone.sum(axis=1) / np.maximum(cov.sum(axis=1), 1)
    return np.where(cov, share[:, None], (isub & ~cov).astype(float))


def operator(ids: list[str]) -> list[str]:
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        c = _clearing(b, lifted.Uniform(), POP)
        x = _random_clearing(c, np.random.default_rng(SEED))
        for label, removed in (("baseline", np.zeros(c.n, dtype=bool)), ("clearing", x)):
            o = c.open_(removed)
            turn = list(lifted.edges(c.sc.grid, o, c.p))[c.p.K:]
            ref = list(_turning_edges(c.sc.grid, o, c.p))
            same = len(turn) == len(ref) and all(
                np.array_equal(ca, ra) and np.array_equal(cb, rb) and np.array_equal(w, rw)
                for (ca, cb, _ka, _kb, w), (ra, rb, rw) in zip(turn, ref, strict=True))
            off = float(np.abs(c.p.cells.sub_factor(c.sc.grid, o, c.p.solver)
                               - _openness(c, removed)).max())
            print(f"{bid} {label}: U's {len(turn)} turning families against the open fractions' "
                  f"definition, bit for bit: {same}; every sub-sample's factor its openness, "
                  f"within {off:.1e}", flush=True)
            if not (same and off <= OPENNESS_TOL):
                bad.append(f"{bid} {label}")
    return bad


GRAD_R = (0.5, 1.0, 1.5)
GRAD_BUILDINGS = 6
GRAD_Q = (1.0, 3.0)
FD_STEPS = (1e-5, 1e-6)  # per building the closer of the central differences at these: B_r's
                         # factor is piecewise linear with many kinks (each sub-sample's openness
                         # against its disk's; under the face aperture each track's min too), and
                         # a step straddling one is off (38136 building 5: b0.5 q 1 at 1e-4,
                         # 1.1e-4; b1.5 J_2 q 1 at 1e-5, 3e-6, 1e-6, 3e-7: 1.1e-5, 9.4e-6, 4.4e-6,
                         # 1.0e-7), while a short step loses digits to cancellation (b1.5 J_2 q 3
                         # at 3e-7: 5.9e-5; at 1e-5, 7.9e-8)
FD_RTOL = 1e-9
GRAD_TOL = 1e-5


def _blurred_clearing(b, r: float) -> clear.Clearing:
    p = dataclasses.replace(_params(lifted.Uniform()), cells=lifted.Blurred(r))
    return clear.Clearing(b, lifted.UniformMesh(0.5, offset=lifted.OFFSET), p,
                          population=common.POPULATIONS["area"], demand=POP)


def _value(rel: relax.Relaxation, x: np.ndarray) -> float:
    rel._u = None                       # from zero: P second-order accurate in the solve
    return rel.value(x)


def gradient(ids: list[str]) -> list[str]:
    """The relaxation's gradient under B_r against central differences (the docstring)."""
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for r in GRAD_R:
            c = _blurred_clearing(b, r)
            rng = np.random.default_rng(SEED)
            x0 = rng.uniform(0.1, 0.4, c.n)
            pick = rng.choice(c.n, GRAD_BUILDINGS, replace=False)
            for power in (1.0, 2.0):
                for q in GRAD_Q:
                    rel = relax.Relaxation(c, power, q=q, rtol=FD_RTOL, eps=EPS)
                    rel._u = None
                    _, g = rel.value_grad(x0)
                    fd = []
                    for j in pick:
                        e = np.zeros(c.n)
                        near = []
                        for step in FD_STEPS:
                            e[j] = step
                            near.append((_value(rel, x0 + e) - _value(rel, x0 - e)) / (2 * step))
                        fd.append(min(near, key=lambda d, j=j: abs(d - g[j])))
                    err = float(np.abs(np.array(fd) - g[pick]).max() / np.abs(g[pick]).max())
                    ok = err <= GRAD_TOL
                    print(f"{bid} b{r:g} J_{power:g} SIMP q {q:g}: gradient against central "
                          f"differences on {GRAD_BUILDINGS} buildings, {err:.1e} of the largest "
                          f"-> {'ok' if ok else 'FAILS'}", flush=True)
                    if not ok:
                        bad.append(f"{bid} b{r:g} p{power:g} q{q:g}")
    return bad


CONVEX_LINES = 4
CONVEX_POINTS = 7
CONVEX_TOL = 1e-9


def convex(ids: list[str]) -> list[str]:
    """Under B_r, along random lines through the box: every cell's factor concave (second
    differences <= CONVEX_TOL), and P convex (second differences >= -CONVEX_TOL P)."""
    bad = []
    for bid in ids:
        [b] = common.build_blocks([bid])
        for r in GRAD_R:
            c = _blurred_clearing(b, r)
            rel = relax.Relaxation(c, 1.0, q=1.0, rtol=FD_RTOL, eps=EPS)
            rng = np.random.default_rng(SEED + 2)
            worst_k, worst_P = -np.inf, np.inf
            for _ in range(CONVEX_LINES):
                a, z = rng.uniform(0.0, 1.0, c.n), rng.uniform(0.0, 1.0, c.n)
                ts = np.linspace(0.0, 1.0, CONVEX_POINTS)
                kap, P = [], []
                for t in ts:
                    x = a + t * (z - a)
                    op = rel.open_of(x)
                    op_eps = op + EPS * (c.full - op)
                    kap.append(lifted.cell_factor(c.sc.grid, op_eps.ravel(), c.p))
                    P.append(_value(rel, x))
                K2 = np.diff(np.array(kap), 2, axis=0)
                P2 = np.diff(np.array(P), 2)
                worst_k = max(worst_k, float(K2.max()))
                worst_P = min(worst_P, float((P2 / np.max(P)).min()))
            ok = worst_k <= CONVEX_TOL and worst_P >= -CONVEX_TOL
            print(f"{bid} b{r:g}: on {CONVEX_LINES} lines, the cell factors' largest second "
                  f"difference {worst_k:.1e} (concave: <= {CONVEX_TOL:g}), P's smallest "
                  f"{worst_P:.1e} of P (convex: >= -{CONVEX_TOL:g}) -> "
                  f"{'ok' if ok else 'FAILS'}", flush=True)
            if not ok:
                bad.append(f"{bid} b{r:g}")
    return bad


def _ulp_factor(self, grid, open_, solver):
    return np.nextafter(open_, np.inf)


def _b1_factor(self, grid, open_, solver):
    return lifted.Blurred(1.0).factor(grid, open_, solver)


def _no_blur_vjp(self, grid, open_, A, solver):
    """Blurred.vjp without the disk's term: only the sub-samples whose own openness binds."""
    xp, S = solver.xp, grid.S
    ny, nx = grid.inside.shape
    Z, B, _disk = self._parts(grid, open_.reshape(ny, nx), xp)
    fw = lifted._fine_walls(grid, xp)
    a = fw.inside * xp.repeat(xp.repeat(A.reshape(ny, nx) / (S * S), S, axis=0), S, axis=1)
    H = xp.where((1.0 - Z) <= B, a, 0.0)
    return (lifted._cell_sum(fw.cov * H, S) * fw.spread).reshape(open_.shape)


def _product_factor(self, grid, open_, solver):
    """Each sub-sample's openness times B in place of their minimum."""
    xp, S = solver.xp, grid.S
    Z, B, _disk = self._parts(grid, open_.reshape(grid.inside.shape), xp)
    k = lifted._fine_walls(grid, xp).inside * (1.0 - Z) * B
    return (lifted._cell_sum(k, S) / (S * S)).reshape(open_.shape)


GUARDS: dict[str, Callable[[list[str]], list[str]]] = {
    "identity": identity, "operator": operator, "gradient": gradient, "convex": convex}
FAULTS: dict[str, dict[str, Callable]] = {
    "identity": {"ulp": lambda: _patched(lifted.OpenFraction, "factor", _ulp_factor)},
    "operator": {"b1": lambda: _patched(lifted.OpenFraction, "factor", _b1_factor)},
    "gradient": {"no-blur-vjp": lambda: _patched(lifted.Blurred, "vjp", _no_blur_vjp)},
    "convex": {"product": lambda: _patched(lifted.Blurred, "factor", _product_factor)},
}


def main(guard: str, ids: list[str], fault: str | None) -> None:
    if guard == "reference":
        reference(ids)
        return
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
    if len(a) not in (2, 3) or a[0] not in ("reference", *GUARDS):
        raise SystemExit(__doc__)
    main(a[0], a[1].split(","), a[2] if len(a) == 3 else None)
