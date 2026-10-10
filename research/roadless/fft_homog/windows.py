"""Q2: effective tensors on square windows of 5810's fabric, FFT schemes vs a sparse direct
solve of the same periodic discretization; iterations and time; principal values/directions.

Writes windows.csv and schemes.csv.
"""
import sys

import fftk
import numpy as np
import pandas as pd

d = np.load("fabric_5810.npz")
ff, ins = d["ff0"].astype(float), d["inside"]
cen = np.load("centroids.npy")
cy, cx, cnt = cen
H = 0.5

CENTRES = {"A dense": (960, 1280), "B dense-SW": (840, 560), "C median-NW": (1240, 700),
           "D sparse": (680, 980)}
SIZES_M = [25, 50, 100, 200, 300]


def window(r, c, n):
    r0, c0 = r - n // 2, c - n // 2
    if r0 < 0 or c0 < 0 or r0 + n > ff.shape[0] or c0 + n > ff.shape[1]:
        return None, None
    i = ins[r0:r0 + n, c0:c0 + n]
    # cells outside the block are street: open
    return np.where(i, ff[r0:r0 + n, c0:c0 + n], 1.0), i


def density(r, c, n):
    r0, c0 = r - n // 2, c - n // 2
    m = (cnt > 0) & (cy >= r0) & (cy < r0 + n) & (cx >= c0) & (cx < c0 + n)
    return m.sum() / (n * n * H * H / 1e4)


rows = []
for name, (r, c) in CENTRES.items():
    for L in SIZES_M:
        n = int(round(L / H))
        o, i = window(r, c, n)
        if o is None:
            continue
        R = fftk.homogenize(o, "cg", tol=1e-8)
        S, sec, _ = fftk.direct(o)
        s1, s2, ang = fftk.principal(R.sigma)
        rows.append(dict(window=name, L_m=L, n=n, inside=float(i.mean()),
                         bldg_per_ha=density(r, c, n), open_frac=float(o.mean()),
                         sxx=R.sigma[0, 0], syy=R.sigma[1, 1], sxy=R.sigma[0, 1],
                         s_major=s1, s_minor=s2, aniso=s1 / max(s2, 1e-12), angle_deg=ang,
                         cg_iters=R.iters, cg_s=R.seconds, direct_s=sec,
                         max_abs_diff=float(np.abs(R.sigma - S).max()),
                         rel_diff=float(np.abs(R.sigma - S).max() / np.abs(S).max())))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv("windows.csv", index=False)
if "--no-schemes" in sys.argv:
    sys.exit(0)

# schemes: iterations / seconds to 1e-6 and 1e-8 at the c0 the sweep found best
SCHEMES = [("cg", 1.0), ("ms", 0.55), ("al", 0.15), ("em", 0.15)]
srows = []
for name in ("A dense", "C median-NW", "D sparse"):
    r, c = CENTRES[name]
    for L in (50, 100, 200):
        n = int(round(L / H))
        o, _ = window(r, c, n)
        S, _, _ = fftk.direct(o)
        for s, c0 in SCHEMES:
            for tol in (1e-6, 1e-8):
                R = fftk.homogenize(o, s, tol=tol, maxiter=20000, c0=c0)
                srows.append(dict(window=name, L_m=L, scheme=s, c0=c0, tol=tol, iters=R.iters,
                                  seconds=R.seconds, converged=R.resid < tol,
                                  rel_err_vs_direct=float(np.abs(R.sigma - S).max()
                                                          / np.abs(S).max())))
                print(srows[-1], flush=True)
pd.DataFrame(srows).to_csv("schemes.csv", index=False)
