"""Q4 test: does the two-scale model rank where clearing pays? For each fully-inside 50 m tile
of 5810 and two clearing rules, the change in J2 from the fine scalar solve vs (a) the two-scale
model re-solved with the tile's windows recomputed, (b) its first-order (adjoint) prediction from
one macro adjoint and the windows' tensor changes, and (c) a naive proxy (cleared area x the
tile's mean baseline potential).

Rules: R20 = buildings whose centre lies in the tile, in random order, until 20% of the tile's
footprint area; STRIP = every building with footprint in a 4 m wide strip through the tile centre
along the baseline macro gradient (a lane toward the exit), clipped to the tile.

Writes tiles_clear.csv.
"""
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import ts

H, K, FIELD = 0.5, 4, "fields_L50_s10.npz"
G: dict = {}


def init():
    d = np.load("fabric_5810.npz")
    C = np.load("clearings_5810.npz")
    G.update(o=d["ff0"].astype(float), ins=d["inside"], gr=d["ground"], f=d["f"],
             own=d["owner"], w=d["w"], live=~d["stranded"], area=d["area"])
    order = np.argsort(C["sub_bldg"], kind="stable")
    G.update(sc=C["sub_cell"][order], sb=C["sub_bldg"][order], sn=C["sub_count"][order])
    G["starts"] = np.searchsorted(G["sb"], np.arange(len(G["w"]) + 1))
    G["tf"] = ts.load_field(FIELD)
    cen = np.load("centroids.npy")
    G["cy"], G["cx"], G["cnt"] = cen


def init_worker():
    init()
    G.update(np.load("tiles_base.npz"))


def open_buildings(ids):
    o = G["o"].copy().ravel()
    for b in ids:
        a, z = G["starts"][b], G["starts"][b + 1]
        np.add.at(o, G["sc"][a:z], G["sn"][a:z] / 16.0)
    return np.minimum(o, 1.0).reshape(G["o"].shape)


def job(args):
    t0 = time.perf_counter()
    r0, c0, rule = args
    n = int(50 / H)
    o, f, own, w, live = G["o"], G["f"], G["own"], G["w"], G["live"]
    if rule == "R20":
        inn = ((G["cnt"] > 0) & (G["cy"] >= r0) & (G["cy"] < r0 + n)
               & (G["cx"] >= c0) & (G["cx"] < c0 + n))
        cand = np.flatnonzero(inn)
        rng = np.random.default_rng(r0 * 10000 + c0)
        cand = rng.permutation(cand)
        tot = G["area"][cand].sum()
        csum = np.cumsum(G["area"][cand])
        ids = cand[: int(np.searchsorted(csum, 0.2 * tot)) + 1] if len(cand) else cand
    else:
        rc, cc = r0 + n // 2, c0 + n // 2
        gx, gy = G["Ux"][rc, cc], G["Uy"][rc, cc]
        nrm = np.hypot(gx, gy)
        ux, uy = (gx / nrm, gy / nrm) if nrm > 0 else (0.0, 1.0)
        ii, jj = np.mgrid[r0:r0 + n, c0:c0 + n]
        dperp = np.abs((jj - cc) * uy - (ii - rc) * ux)
        cells = (ii * o.shape[1] + jj)[dperp <= 2.0 / H]
        ids = np.unique(G["sb"][np.isin(G["sc"], cells)])
    oc = open_buildings(ids)
    changed = oc != o
    fs = ts.fine_solve(oc, G["gr"], f)
    Jf = ts.J(ts.home_u(fs.u, f, own, w, live), w, live)
    tf = G["tf"]
    sig, nwin = ts.sigma_serial(tf, oc, G["ins"], f, changed)
    sb = ts.fill(ts.TensorField(sig, tf.inside, tf.have, tf.chi, tf.wf, tf.s, tf.n, 0, tf.iters),
                 None)
    mac = ts.macro_solve(sb, tf.s, K, G["ins"], G["gr"], f)
    U = ts.evaluate(mac, o.shape)[0]
    Jt = ts.J(ts.home_u(np.where(G["gr"], 0, U), f, own, w, live), w, live)
    # first order: dJ = -sum_e S_e . dk_e
    sbase = ts.fill(tf, None)
    NI, NJ = G["S"].shape[0] + 1, G["S"].shape[1] + 1
    dk = ts.element_tensors(sb - sbase, tf.s, K, NI, NJ)
    dlin = -float((G["S"][..., 0] * dk[..., 0, 0] + G["S"][..., 1] * dk[..., 1, 1]
                   + G["S"][..., 2] * dk[..., 0, 1]).sum())
    cleared_m2 = float(changed.sum() and ((oc - o)[changed]).sum() * H * H)
    tile_u = float(G["ufine"][r0:r0 + n, c0:c0 + n].mean())
    return dict(r0=r0, c0=c0, rule=rule, n_bldg=len(ids), cleared_m2=cleared_m2,
                dJ_fine=Jf - G["J0f"], dJ_twoscale=Jt - G["J0t"], dJ_linear=dlin,
                proxy=-cleared_m2 * tile_u, windows=nwin, seconds=time.perf_counter() - t0)


if __name__ == "__main__":
    init()
    o, f, own, w, live, ins, gr = (G[k] for k in ("o", "f", "own", "w", "live", "ins", "gr"))
    fs0 = ts.fine_solve(o, gr, f)
    J0f = ts.J(ts.home_u(fs0.u, f, own, w, live), w, live)
    tf = G["tf"]
    sb0 = ts.fill(tf, None)
    m0 = ts.macro_solve(sb0, tf.s, K, ins, gr, f)
    U0, Ux, Uy = ts.evaluate(m0, o.shape)
    ut = ts.home_u(np.where(gr, 0, U0), f, own, w, live)
    J0t = ts.J(ut, w, live)
    # adjoint of J2 (U-only readout): g = load of 2 u_owner f
    gcell = np.zeros(o.shape)
    on = own >= 0
    gcell[on] = 2 * np.nan_to_num(ut[own[on]]) * f[on]
    gvec = ts.load_vector(gcell, K, m0.unk, m0.n)
    lam, _, _ = ts.amg_solve(m0.A, gvec, 1e-10)
    NI, NJ = m0.U.shape
    Lam = np.zeros((NI, NJ))
    Lam[m0.unk >= 0] = lam
    Ir, Jn = np.meshgrid(np.arange(NI - 1), np.arange(NJ - 1), indexing="ij")
    ue = np.stack([m0.U[Ir, Jn], m0.U[Ir, Jn + 1], m0.U[Ir + 1, Jn], m0.U[Ir + 1, Jn + 1]], -1)
    le = np.stack([Lam[Ir, Jn], Lam[Ir, Jn + 1], Lam[Ir + 1, Jn], Lam[Ir + 1, Jn + 1]], -1)
    S = np.stack([np.einsum("...i,ij,...j->...", le, ts.Q1[k], ue) for k in ("xx", "yy", "xy")], -1)
    np.savez("tiles_base.npz", S=S, Ux=Ux.astype(np.float32), Uy=Uy.astype(np.float32),
             ufine=fs0.u.astype(np.float32), J0f=J0f, J0t=J0t)
    np.save("sens_map.npy", S)
    t = pd.read_csv("tiles.csv")
    t = t[t.L_m == 50]
    jobs = [(int(r), int(c), rule)
            for r, c in zip(t.r0, t.c0, strict=True) for rule in ("R20", "STRIP")]
    import os
    jobs = jobs[: int(os.environ.get("LIMIT", len(jobs)))]
    print(len(jobs), "jobs; J0 fine", J0f, "two-scale", J0t, flush=True)
    rows = []
    with ProcessPoolExecutor(8, initializer=init_worker) as ex:
        for i, row in enumerate(ex.map(job, jobs)):
            rows.append(row)
            if i % 20 == 0:
                print(i, row, flush=True)
    pd.DataFrame(rows).to_csv("tiles_clear.csv", index=False)
