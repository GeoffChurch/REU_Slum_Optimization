"""Prototype: a two-level composite operator built from a uniform fine Grid, against the uniform
operator's P0 and home escape times. Validation of the interface rule only (not memory-lean).

    cd ~/src/reblock && PYTHONPATH=. uv run python <this> <block> <r> <D_m> [<Dg_m>]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, "research/roadless")
import common  # noqa: E402
import lifted  # noqa: E402


def composite_P(sc, r: int, D: float, Dg: float, p: lifted.Params, field=None):
    """P and per-building home u under `field` (fine open fractions; baseline if None), on the
    mesh refined from the baseline geometry."""
    g = sc.grid
    h = g.h
    ny, nx = g.inside.shape
    open0 = sc.free0
    v, _th, m, gap = lifted.axes(p.K)
    # ---- tiles: refine near buildings / ground, or where not fully open --------------------
    Ty, Tx = -(-ny // r), -(-nx // r)
    pad = lambda a, fill: np.pad(a, ((0, Ty * r - ny), (0, Tx * r - nx)), constant_values=fill)  # noqa: E731
    op_p = pad(open0, 0.0)
    db_p = pad(g.dist_b, 0.0)
    gr = g.ground.astype(float)
    from scipy import ndimage
    dg = ndimage.distance_transform_edt(~g.ground) * h
    dg_p = pad(dg, 0.0)
    tiles = lambda a: a.reshape(Ty, r, Tx, r).transpose(0, 2, 1, 3).reshape(Ty, Tx, r * r)  # noqa: E731
    coarse_tile = ((tiles(op_p) >= 1.0 - 1e-12).all(-1) & (tiles(db_p) > D).all(-1)
                   & (tiles(dg_p) > Dg).all(-1))
    # ---- cells: fine cells of refined tiles (open > 0), coarse cells of coarse tiles -------
    fld0 = open0 if field is None else field
    rr, cc = np.nonzero(fld0 > 0)                 # the current field's open cells, as uniform
    fine_ok = ~coarse_tile[rr // r, cc // r]
    rr, cc = rr[fine_ok], cc[fine_ok]
    nf = len(rr)
    TY, TX = np.nonzero(coarse_tile)
    nc = len(TY)
    n = nf + nc
    fine_id = -np.ones((ny, nx), dtype=np.int64)
    fine_id[rr, cc] = np.arange(nf)
    coarse_id = -np.ones((Ty, Tx), dtype=np.int64)
    coarse_id[TY, TX] = nf + np.arange(nc)
    size = np.concatenate([np.full(nf, h), np.full(nc, r * h)])
    # centres in fine-index units (row, col)
    cy = np.concatenate([rr.astype(float), TY * r + (r - 1) / 2])
    cx = np.concatenate([cc.astype(float), TX * r + (r - 1) / 2])
    fld = open0 if field is None else field
    opn = np.concatenate([fld[rr, cc], np.ones(nc)])
    ground = np.concatenate([g.ground[rr, cc], np.zeros(nc, dtype=bool)])
    f = np.concatenate([sc.f[rr, cc], np.zeros(nc)])
    # every inside cell with open > 0 is one of ours? (open cells in coarse tiles are covered)

    def locate(py, px):
        """Cell id containing fine-index point (py, px) (floats), -1 outside."""
        iy, ix = np.floor(py + 0.5).astype(np.int64), np.floor(px + 0.5).astype(np.int64)
        out = -np.ones(len(iy), dtype=np.int64)
        ok = (iy >= 0) & (iy < ny) & (ix >= 0) & (ix < nx)
        ty, tx = iy[ok] // r, ix[ok] // r
        isc = coarse_tile[ty, tx]
        res = np.where(isc, coarse_id[ty, tx], fine_id[iy[ok], ix[ok]])
        out[ok] = res
        return out

    s_units = size / h                      # 1 for fine, r for coarse
    rows, cols, vals = [], [], []
    for k in range(p.K):
        dx, dy = int(v[k, 0]), int(v[k, 1])
        lines = lifted._line_cells(dx, dy)
        vn = np.hypot(dx, dy)
        for sign in (+1, -1):
            src = np.arange(n) if sign == 1 else np.arange(nf)      # coarse: forward only
            ty_ = cy[src] + sign * s_units[src] * dy
            tx_ = cx[src] + sign * s_units[src] * dx
            t = locate(ty_, tx_)
            ok = t >= 0
            src, t = src[ok], t[ok]
            frac = np.minimum(opn[src], opn[t])
            for ix, iy in lines:
                ci = locate(cy[src] + sign * s_units[src] * iy, cx[src] + sign * s_units[src] * ix)
                o = np.where(ci >= 0, opn[np.maximum(ci, 0)], 0.0)
                frac = np.minimum(frac, o)
            same = size[t] == size[src]
            bigger = size[t] > size[src]
            if sign == -1:
                keep = bigger                     # backward: only fine -> coarse
            else:
                keep = same | bigger              # forward: same size, or fine -> coarse
            keep &= frac > 0
            a, b, fr = src[keep], t[keep], frac[keep]
            # flux-consistent weight: m_k * strip width / projected distance (metres)
            dist = (sign * ((cx[b] - cx[a]) * dx + (cy[b] - cy[a]) * dy) / vn) * h
            strip = size[a] / vn
            w = fr * m[k] * strip / dist
            rows.append((a, b, w, k))
    # turning
    turn = []
    for k in range(p.K):
        turn.append(opn * (size ** 2) / (p.ell_m ** 2 * gap[k]))
    # ---- unknowns (cell, axis), ground removed -------------------------------------------
    K = p.K
    unk = -np.ones(n, dtype=np.int64)
    unk[~ground] = np.arange(int((~ground).sum()))
    N = int((~ground).sum()) * K
    R, C, V = [], [], []
    diag = np.zeros(N)
    for a, b, w, k in rows:
        ua, ub = unk[a], unk[b]
        ia, ib = ua * K + k, ub * K + k
        both = (ua >= 0) & (ub >= 0)
        R += [ia[both], ib[both]]; C += [ib[both], ia[both]]; V += [-w[both], -w[both]]
        diag += np.bincount(ia[ua >= 0], weights=w[ua >= 0], minlength=N)
        diag += np.bincount(ib[ub >= 0], weights=w[ub >= 0], minlength=N)
    live = np.flatnonzero(unk >= 0)
    for k in range(K):
        ia = unk[live] * K + k
        ib = unk[live] * K + (k + 1) % K
        w = turn[k][live]
        R += [ia, ib]; C += [ib, ia]; V += [-w, -w]
        diag += np.bincount(ia, weights=w, minlength=N) + np.bincount(ib, weights=w, minlength=N)
    L = sp.coo_matrix((np.concatenate(V + [diag]), (np.concatenate(R + [np.arange(N)]),
                                                     np.concatenate(C + [np.arange(N)]))),
                      shape=(N, N)).tocsr()
    # reach: components holding ground (cell graph of along edges)
    E = [(a, b) for a, b, _, _ in rows]
    Acell = sp.coo_matrix((np.ones(sum(len(a) for a, _ in E)),
                           (np.concatenate([a for a, _ in E]), np.concatenate([b for _, b in E]))),
                          shape=(n, n))
    ncomp, lab = connected_components(Acell, directed=False)
    good = np.zeros(ncomp, dtype=bool)
    good[lab[ground]] = True
    reach = good[lab]
    keepu = np.repeat(reach[unk >= 0], K)
    Lk = L[keepu][:, keepu]
    b = np.zeros(N)
    liveu = unk >= 0
    b[(unk[liveu][:, None] * K + np.arange(K)[None, :]).ravel()] = (f[liveu][:, None] * (m / np.pi)[None, :]).ravel()
    bk = b[keepu]
    import pyamg
    ml = pyamg.smoothed_aggregation_solver(Lk.tocsr(), symmetry="symmetric")
    u = ml.solve(bk, tol=1e-10, accel="cg", maxiter=2000)
    P = float(bk @ u)
    uu = np.zeros(N)
    uu[keepu] = u
    ucell = np.zeros(n)
    ucell[liveu] = uu.reshape(-1, K) @ (m / np.pi)
    ub = np.zeros((ny, nx))
    ub[rr, cc] = ucell[:nf]
    on = sc.owner >= 0
    num = np.bincount(sc.owner[on], weights=(sc.f * ub)[on], minlength=len(sc.polys))
    with np.errstate(invalid="ignore", divide="ignore"):
        home = np.where(sc.live_home, num / sc.w, np.nan)
    return P, home, dict(fine_cells=nf, coarse_cells=nc, unknowns=int(keepu.sum()), coarse_tiles=int(coarse_tile.sum()),
                   tiles=int(Ty * Tx))


if __name__ == "__main__":
    import pandas as pd
    import glob
    bid, r, D = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    Dg = float(sys.argv[4]) if len(sys.argv) > 4 else D
    p = lifted.Params(3.0, 8, along=lifted.Uniform(), solver=lifted.CpuAMG())
    [b] = common.build_blocks([bid])
    sc = common.Scorer(b, 0.5, p, population=common.POPULATIONS["area"])
    lab = sc.grid.label_sub(sc.polys)
    g = sc.grid
    cost = sc.w / sc.w.sum()
    P0c, home0c, info = composite_P(sc, r, D, Dg, p)
    J0u, J0c = sc.J(sc.u0, 2.0), sc.J(home0c, 2.0)
    print(f"{bid} r={r} D={D:g}: P0 rel {P0c / sc.P0 - 1:+.4%}, J0 rel {J0c / J0u - 1:+.4%}  {info}", flush=True)
    todo = pd.read_parquet("/tmp/claude-1641171234/-home-gchurchill-src-reblock/fe8870c0-fbd9-4712-ac98-aebcb951b199/scratchpad/res_clearings.parquet")
    todo = todo[todo.block == bid]
    exact = pd.concat(pd.read_parquet(f) for f in glob.glob(f"research/roadless/res_rows/uni/{bid}_p2_h0.5,*.parquet"))
    exact = exact[exact.h == 0.5].set_index("name").perm
    import relax
    for name, cleared in zip(todo.name, todo.cleared):
        if name.endswith("_steps"):
            name = name[:-6]
            cleared = np.flatnonzero(relax.cut_to_budget(list(cleared), cost, 0.05))
        removed = np.zeros(len(sc.polys), dtype=bool)
        removed[np.asarray(cleared, dtype=np.int64)] = True
        cl = (lab >= 0) & removed[np.maximum(lab, 0)]
        field = (g.isub & (~g.bsub | cl)).mean(axis=-1)
        _, home, _ = composite_P(sc, r, D, Dg, p, field)
        perm_c = 1 - (sc.J(home, 2.0) / J0c) ** 0.5
        print(f"  {name:8s} perm uniform {exact[name]:.4f}  composite {perm_c:.4f}  diff {perm_c - exact[name]:+.4f}", flush=True)
