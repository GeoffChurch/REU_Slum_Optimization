"""Q3: two-scale J vs the fine solve of the same scalar model on all of 5810.

Writes twoscale.csv (one row per variant), homes_<tag>.npz (per-home u, fine and two-scale) and
fields_L<L>.npz (tensor fields, reused by the clearing scripts).
"""
import sys

import numpy as np
import pandas as pd
import ts
from scipy import ndimage
from scipy.stats import spearmanr

WORKERS = 8
OUT = "twoscale.csv"
d = np.load("fabric_5810.npz")
o, ins, gr, f = d["ff0"].astype(float), d["inside"], d["ground"], d["f"]
own, w, live = d["owner"], d["w"], ~d["stranded"]
H = 0.5

fs = ts.fine_solve(o, gr, f, tol=1e-10)
u_f = ts.home_u(fs.u, f, own, w, live)
J_f, P_f = ts.J(u_f, w, live), float((f * fs.u).sum())
print(f"fine: {fs.n} unknowns {fs.seconds:.1f}s J2 {J_f:.6g} P {P_f:.6g}", flush=True)

# each home's distance to the exit (m): EDT from ground cells, at the home's ring cells
dist = ndimage.distance_transform_edt(~gr) * H
on = own >= 0
hd = np.bincount(own[on], weights=(f * dist)[on], minlength=len(w)) / np.where(w > 0, w, 1)
# main body vs the north-east strip (rows > 1700 or cols > 1550 px)
rr, cc = np.nonzero(on)
strip_cell = np.zeros(o.shape, dtype=bool)
strip_cell[(np.arange(o.shape[0])[:, None] > 1700) | (np.arange(o.shape[1])[None, :] > 1550)] = True
hs = (np.bincount(own[on], weights=(f * strip_cell)[on], minlength=len(w))
      / np.where(w > 0, w, 1) > 0.5)

rows = []
for L, s in [(int(a), int(b)) for a, b in (x.split(":") for x in sys.argv[1:])] or [(50, 10)]:
    n, sp_ = int(round(L / H)), int(round(s / H))
    import os
    if os.path.exists(f"fields_L{L}_s{s}.npz"):
        tf = ts.load_field(f"fields_L{L}_s{s}.npz")
        rows_t = dict(np.load(f"fields_L{L}_s{s}.npz").items())
        tf.seconds = float(rows_t.get("seconds", np.nan))
    else:
        tf = ts.tensor_field(o, ins, f, n, sp_, WORKERS)
    print(f"L {L} m stride {s} m: {int(tf.have.sum())} windows {tf.seconds:.0f}s "
          f"({WORKERS} workers), CG iters median {np.median(tf.iters[tf.have]):.0f} "
          f"max {tf.iters[tf.have].max()}", flush=True)
    if not os.path.exists(f"fields_L{L}_s{s}.npz"):
        np.savez_compressed(f"fields_L{L}_s{s}.npz", sigma=tf.sigma, sigma_in=tf.sigma_in,
                            inside=tf.inside,
                            have=tf.have, chi=tf.chi, wf=tf.wf, s=tf.s, n=tf.n, iters=tf.iters,
                            seconds=tf.seconds)
    for fill_name, mi, ia in (("open-outside", None, False), ("extrapolated", 0.9, False),
                              ("inside-avg", None, True)):
        sb = ts.fill(tf, mi, ia)
        for k_m in (2, 5, 10):
            k = int(round(k_m / H))
            mac = ts.macro_solve(sb, tf.s, k, ins, gr, f)
            U, Ux, Uy = ts.evaluate(mac, o.shape)
            for corr in ("U", "U+chi", "U+chi+w"):
                uc = U.copy()
                if corr != "U":
                    uc += tf.chi[0] * Ux + tf.chi[1] * Uy
                if corr == "U+chi+w":
                    uc += tf.wf
                uc = np.where(gr, 0.0, uc)
                u_t = ts.home_u(uc, f, own, w, live)
                J_t = ts.J(u_t, w, live)
                P_t = float((f * uc).sum())
                rel = (u_t - u_f) / u_f
                ok = live & (u_f > 0)
                row = dict(L_m=L, stride_m=s, fill=fill_name, H_m=k_m, corr=corr,
                           macro_unknowns=mac.n, macro_s=mac.seconds,
                           J2_rel_err=J_t / J_f - 1, P_rel_err=P_t / P_f - 1,
                           spearman=spearmanr(u_f[ok], u_t[ok]).statistic,
                           med_abs_rel=float(np.median(np.abs(rel[ok]))),
                           p90_abs_rel=float(np.percentile(np.abs(rel[ok]), 90)))
                for lo, hi in ((0, 10), (10, 30), (30, 100), (100, 1e9)):
                    m = ok & (hd >= lo) & (hd < hi)
                    row[f"medrel_d{lo}"] = float(np.median(rel[m])) if m.any() else np.nan
                    row[f"J2share_d{lo}"] = float((w[m] * u_f[m] ** 2).sum() / J_f)
                for nm, m in (("main", ok & ~hs), ("strip", ok & hs)):
                    row[f"J2err_{nm}"] = float((w[m] * u_t[m] ** 2).sum()
                                               / (w[m] * u_f[m] ** 2).sum() - 1)
                    row[f"J2share_{nm}"] = float((w[m] * u_f[m] ** 2).sum() / J_f)
                rows.append(row)
                print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()},
                      flush=True)
                if corr == "U+chi+w" and k_m == 2:
                    np.savez_compressed(f"homes_L{L}_{fill_name}.npz", u_f=u_f, u_t=u_t, hd=hd,
                                        strip=hs, w=w, live=live)
    pd.DataFrame(rows).to_csv(OUT, index=False)
