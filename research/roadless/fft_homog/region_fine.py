"""The fine reference on 5810@major: ts's scalar 'uni' (face conductance min(o_a, o_b), the
metric's ground and demand) on the uniform h 0.5 lattice, solved by pyamg SA + CG.

FineBase holds the baseline system (A0, its SA hierarchy, u0). `delta` re-solves a clearing that
opens buildings locally: the new system A1 is A0 (padded with the newly reachable cells) plus a
correction dA on the faces near the change, built exactly; CG for d in A1 d = -dA x0 from the
baseline x0 = (u0, new cells 0) (the baseline's own residual left out, so u0's error cancels in
dJ), preconditioned by symmetric multiplicative Schwarz -- an exact solve on the unknowns within
`margin` px of the change, the baseline's V-cycle on what is left, the exact solve again -- to a
residual `rtol` times the initial one, so dJ_2 comes with its own relative accuracy. `full` is
a fresh solve (ts.fine_solve's system, a new hierarchy). Checked against fresh solves on 5810
(rtol 1e-9: dJ within 1.6e-6) and on the region (rtol 1e-8: within 1e-6 and 5.5e-5, the latter
the fresh solve's own precision).

Run (repo root):  ... uv run python research/roadless/fft_homog/region_fine.py
  the baseline (OUT/fine_u0.npy, fine_base.npz) and the two stored clearings' fresh solves
  (OUT/region_clearings_fine.csv).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import numpy as np  # noqa: E402
import pyamg  # noqa: E402
import region_common as rc  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import ts  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

BASE_TOL = 1e-11      # baseline residual / ||b||
DELTA_RTOL = 1e-9     # re-solve residual / its initial residual
MARGIN = 40           # px (20 m) of unknowns around the change solved exactly in the preconditioner


def pcg(Aop, b, M, tol: float, maxiter: int, x0=None):
    """CG preconditioned by M, from x0 (zero), to ||r|| <= tol ||b||. Returns (x, iterations,
    final relative residual)."""
    x = np.zeros_like(b) if x0 is None else x0.copy()
    r = b - Aop(x) if x0 is not None else b.copy()
    nb = float(np.linalg.norm(b))
    z = M(r)
    p = z.copy()
    rz = float(r @ z)
    it = 0
    res = float(np.linalg.norm(r)) / nb
    while res > tol and it < maxiter:
        Ap = Aop(p)
        alpha = rz / float(p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        it += 1
        res = float(np.linalg.norm(r)) / nb
        if res <= tol:
            break
        z = M(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    if res > tol:
        raise RuntimeError(f"PCG did not converge: {res:.2e} after {it}")
    return x, it, res


class FineBase:
    def __init__(self, fab: rc.Fabric, tol: float = BASE_TOL, u0: np.ndarray | None = None):
        """u0: a baseline solution saved by an earlier run (the hierarchy is rebuilt, the
        solve skipped)."""
        t = time.perf_counter()
        self.fab = fab
        o, gr, f = fab.o, fab.ground, fab.f
        self.A0, self.idx0 = ts.fine_matrix(o, gr)
        self.N0 = self.A0.shape[0]
        self.reach0 = ts.reach_of(o, gr)
        self.t_matrix = time.perf_counter() - t
        on = self.idx0 >= 0
        self.b0 = f[on]
        assert not (f[~on & ~gr] > 0).any(), "injection on a cell that is not an unknown"
        t = time.perf_counter()
        self.ml = pyamg.smoothed_aggregation_solver(self.A0, symmetry="symmetric",
                                                    max_coarse=2000)
        self.t_setup = time.perf_counter() - t
        t = time.perf_counter()
        if u0 is None:
            self.u0, self.iters, _ = pcg(lambda x: self.A0 @ x, self.b0, self.vcycle, tol, 2000)
        else:
            assert len(u0) == self.N0
            self.u0, self.iters = u0, 0
        self.t_solve = time.perf_counter() - t
        self.res = float(np.linalg.norm(self.b0 - self.A0 @ self.u0) / np.linalg.norm(self.b0))
        # the homes' readout: cells with injection, their unknown (-1 on ground), owner, f
        fc = np.flatnonzero(f.ravel() > 0)
        self.f_unk = self.idx0.ravel()[fc]
        self.f_own = fab.owner.ravel()[fc]
        self.f_val = f.ravel()[fc]
        assert (self.f_own >= 0).all()
        self.uh0 = self.home(self.u0)
        self.J0 = self.J(self.uh0)
        self.P0 = float(self.b0 @ self.u0)

    def vcycle(self, r: np.ndarray) -> np.ndarray:
        """One V-cycle of the baseline hierarchy from zero (pyamg's aspreconditioner without
        its two residual norms per call: the same arithmetic, two fewer fine matvecs)."""
        x = np.zeros_like(r)
        self.ml._MultilevelSolver__solve(0, x, np.ascontiguousarray(r), "V", 1)
        return x

    def home(self, x: np.ndarray) -> np.ndarray:
        """Per building: the injection-weighted mean of x (over the baseline unknowns) on its
        ring cells (ts.home_u); nan where stranded."""
        v = np.where(self.f_unk >= 0, x[np.maximum(self.f_unk, 0)], 0.0)
        num = np.bincount(self.f_own, weights=self.f_val * v, minlength=len(self.fab.w))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.fab.live, num / self.fab.w, np.nan)

    def J(self, uh: np.ndarray) -> float:
        lv = self.fab.live
        return float((self.fab.w[lv] * uh[lv] ** 2).sum())

    def u_cells(self, x: np.ndarray) -> np.ndarray:
        """Baseline-unknown vector -> (ny, nx) raster, 0 elsewhere."""
        u = np.zeros(self.fab.o.shape)
        u[self.idx0 >= 0] = x
        return u

    # ------------------------------------------------------------------ local re-solve
    def delta(self, oc: np.ndarray, rtol: float = DELTA_RTOL, margin: int = MARGIN,
              trace: bool = False) -> dict:
        """dJ_2 (and dP) of the clearing whose open fraction is oc (>= the baseline's)."""
        t0 = time.perf_counter()
        fab, o, gr = self.fab, self.fab.o, self.fab.ground
        changed = oc != o
        if (oc < o).any():
            raise ValueError("a clearing only opens")
        reach1 = ts.reach_of(oc, gr)
        new = reach1 & ~gr & (self.idx0 < 0)
        rows, cols = np.nonzero(changed | new)
        if len(rows) == 0:
            return dict(dJ=0.0, dP=0.0, iters=0, n_new=0, n_dA=0, n_loc=0, seconds=0.0)
        ny, nx = o.shape
        r_lo, r_hi = max(rows.min() - 1, 0), min(rows.max() + 2, ny)
        c_lo, c_hi = max(cols.min() - 1, 0), min(cols.max() + 2, nx)
        # local numbering: the box's unknowns after the clearing (old indices kept, new cells
        # appended); the box padded by `margin` for the preconditioner's exact solve
        R_lo, R_hi = max(r_lo - margin, 0), min(r_hi + margin, ny)
        C_lo, C_hi = max(c_lo - margin, 0), min(c_hi + margin, nx)
        BOX = (slice(R_lo, R_hi), slice(C_lo, C_hi))
        idx = self.idx0[BOX].copy()
        nb_ = new[BOX]
        n_new = int(new.sum())
        assert int(nb_.sum()) == n_new
        idx[nb_] = self.N0 + np.arange(n_new)
        N1 = self.N0 + n_new
        ob, ocb = o[BOX], oc[BOX]
        r0b, r1b = self.reach0[BOX], reach1[BOX]
        rr, cc_, vv, dd_i, dd_v = [], [], [], [], []
        for axis in (0, 1):
            a = [slice(None), slice(None)]
            b = [slice(None), slice(None)]
            a[axis], b[axis] = slice(0, -1), slice(1, None)
            a, b = tuple(a), tuple(b)
            cb = np.minimum(ob[a], ob[b]) * (r0b[a] & r0b[b])
            ca = np.minimum(ocb[a], ocb[b]) * (r1b[a] & r1b[b])
            dc = ca - cb
            m = dc != 0
            ia, ib, d = idx[a][m], idx[b][m], dc[m]
            both = (ia >= 0) & (ib >= 0)
            rr += [ia[both], ib[both]]
            cc_ += [ib[both], ia[both]]
            vv += [-d[both], -d[both]]
            dd_i += [ia[ia >= 0], ib[ib >= 0]]
            dd_v += [d[ia >= 0], d[ib >= 0]]
        di = np.concatenate(dd_i)
        dA = sp.coo_matrix((np.concatenate(vv + dd_v),
                            (np.concatenate(rr + [di]), np.concatenate(cc_ + [di]))),
                           shape=(N1, N1)).tocsr()
        dA.eliminate_zeros()
        A0, N0 = self.A0, self.N0

        def Aop(x):
            y = dA @ x
            y[:N0] += A0 @ x[:N0]
            return y

        # the preconditioner's exact block: the box's unknowns (old and new)
        ids = np.sort(idx[idx >= 0])
        old = ids[ids < N0]
        n_loc_new = len(ids) - len(old)
        A_oo = A0[old][:, old]
        A_loc = (sp.block_diag([A_oo, sp.csr_matrix((n_loc_new, n_loc_new))], format="csr")
                 + dA[ids][:, ids])
        lu = splu(A_loc.tocsc())
        vcycle = self.vcycle
        # rows of A1 = pad(A0) + dA at the local unknowns, for A1 z with z supported on them
        A_rows = (sp.vstack([A0[old], sp.csr_matrix((n_loc_new, N0))], format="csr")
                  if n_loc_new else A0[old])
        A_rows = sp.hstack([A_rows, sp.csr_matrix((len(ids), n_new))], format="csr") + dA[ids]

        def M(r):
            """Symmetric multiplicative Schwarz: the exact local solve, the baseline's V-cycle
            on what is left, the local solve again (2L - LAL + (I - LA) B (I - AL), SPD)."""
            z = np.zeros(N1)
            z[ids] = lu.solve(r[ids])
            q = r - A_rows.T @ z[ids]                  # r - A1 z (A1 symmetric, z local)
            z[:N0] += vcycle(q[:N0])
            q = r - Aop(z)
            z[ids] += lu.solve(q[ids])
            return z

        # the baseline's own residual is left out: u0's error then cancels in dJ
        x0 = np.zeros(N1)
        x0[:N0] = self.u0
        r0 = -(dA @ x0)
        t1 = time.perf_counter()
        d, it, res = pcg(Aop, r0, M, rtol, 1000)
        uh = self.home(d[:N0])
        lv = fab.live
        dJ = float((fab.w[lv] * (2 * self.uh0[lv] + uh[lv]) * uh[lv]).sum())
        dP = float(self.b0 @ d[:N0])
        out = dict(dJ=dJ, dP=dP, iters=it, res=res, n_new=n_new, n_dA=int(dA.nnz),
                   n_loc=len(ids), seconds=time.perf_counter() - t0,
                   solve_seconds=time.perf_counter() - t1)
        if trace:
            out["x"] = x0 + d
        return out

    def full(self, oc: np.ndarray, tol: float = BASE_TOL) -> dict:
        """A fresh solve of the cleared state (ts.fine_matrix, new hierarchy)."""
        t0 = time.perf_counter()
        A, idx = ts.fine_matrix(oc, self.fab.ground)
        on = idx >= 0
        b = self.fab.f[on]
        ml = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric", max_coarse=2000)
        res: list[float] = []
        x = ml.solve(b, tol=tol, accel="cg", maxiter=2000, residuals=res)
        rel = float(np.linalg.norm(b - A @ x) / np.linalg.norm(b))
        if not rel <= 10 * tol:
            raise RuntimeError(f"full solve: residual {rel:.1e}")
        u = np.zeros(oc.shape)
        u[on] = x
        uh = ts.home_u(u, self.fab.f, self.fab.owner, self.fab.w, self.fab.live)
        return dict(J=ts.J(uh, self.fab.w, self.fab.live), P=float(b @ x), uh=uh,
                    iters=len(res) - 1, seconds=time.perf_counter() - t0, u=u)


if __name__ == "__main__":
    import pandas as pd
    fab = rc.load_fabric()
    with rc.Monitor("fine: baseline (matrix, SA setup, CG to 1e-11)") as mon:
        fb = FineBase(fab)
    print(f"fine baseline: {fb.N0} unknowns; matrix {fb.t_matrix:.0f} s, SA setup "
          f"{fb.t_setup:.0f} s, CG {fb.iters} it {fb.t_solve:.0f} s, residual {fb.res:.1e}; "
          f"J2 {fb.J0:.8g} P {fb.P0:.8g}; levels "
          f"{[lv.A.shape[0] for lv in fb.ml.levels]}", flush=True)
    np.save(rc.OUT / "fine_u0.npy", fb.u0)
    np.savez(rc.OUT / "fine_base.npz", uh0=fb.uh0, J0=fb.J0, P0=fb.P0, N0=fb.N0,
             t_matrix=fb.t_matrix, t_setup=fb.t_setup, t_solve=fb.t_solve, iters=fb.iters)
    rows = []
    for name, ids in rc.stored_clearings().items():
        oc = rc.open_buildings(fab, ids)
        with rc.Monitor(f"fine: stored clearing {name}, fresh solve"):
            r = fb.full(oc)
        lensA = 1 - np.sqrt(r["J"] / fb.J0)
        print(f"{name}: {len(ids)} buildings, fine J2 {r['J']:.8g}, Lens A (fine scalar) "
              f"{lensA:.4f}, stored lifted {rc.STORED_LENS_A[name]:.4f}; {r['iters']} it "
              f"{r['seconds']:.0f} s", flush=True)
        np.save(rc.OUT / f"fine_uh_{name}.npy", r["uh"])
        rows.append(dict(clearing=name, n_buildings=len(ids), J_fine=r["J"], J0_fine=fb.J0,
                         lensA_fine_scalar=lensA, lensA_lifted_stored=rc.STORED_LENS_A[name],
                         P_fine=r["P"], seconds=r["seconds"], iters=r["iters"]))
        del r
    pd.DataFrame(rows).to_csv(rc.OUT / "region_clearings_fine.csv", index=False)
