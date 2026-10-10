"""Q2: the tensor over every fully-inside tile of 5810 at L = 12.5, 25, 50, 100 m (periodic FFT-CG),
plus (L = 25, 50 m) the oversampled variant (solve on the 2L window centred on the tile, average the
flux over the central L) and the lanes' orientation (structure tensor of the open fraction).

Writes tiles.csv.
"""
import time
from concurrent.futures import ProcessPoolExecutor

import fftk
import numpy as np
import pandas as pd
from scipy import ndimage

d = np.load("fabric_5810.npz")
ff, ins = d["ff0"].astype(float), d["inside"]
cen = np.load("centroids.npy")
H = 0.5


def integral(a):
    return np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))


Ii = integral(ins.astype(float))


def full(r0, c0, n):
    if r0 < 0 or c0 < 0 or r0 + n > ff.shape[0] or c0 + n > ff.shape[1]:
        return False
    return Ii[r0 + n, c0 + n] - Ii[r0, c0 + n] - Ii[r0 + n, c0] + Ii[r0, c0] == n * n


# structure tensor of the open fraction (derivative scale 1 px = 0.5 m)
gy = ndimage.gaussian_filter(ff, 1.0, order=(1, 0))
gx = ndimage.gaussian_filter(ff, 1.0, order=(0, 1))
Jxx, Jyy, Jxy = gx * gx, gy * gy, gx * gy
IJ = [integral(a) for a in (Jxx, Jyy, Jxy)]


def lanes(r0, c0, n):
    """(lane axis angle deg [0,180), coherence): the direction along which o varies least."""
    t = [integ[r0 + n, c0 + n] - integ[r0, c0 + n] - integ[r0 + n, c0] + integ[r0, c0]
         for integ in IJ]
    T = np.array([[t[0], t[2]], [t[2], t[1]]])
    w, v = np.linalg.eigh(T)
    ang = np.degrees(np.arctan2(v[1, 0], v[0, 0])) % 180.0      # smallest-eigenvalue vector
    coh = (w[1] - w[0]) / max(w[1] + w[0], 1e-300)
    return float(ang), float(coh)


def oversampled(r0, c0, n):
    """Periodic solve on the 2n window centred on the tile; flux averaged over the central n."""
    R0, C0, N = r0 - n // 2, c0 - n // 2, 2 * n
    if not full(R0, C0, N):
        return None
    o = ff[R0:R0 + N, C0:C0 + N]
    P = fftk.Problem.of(o)
    u, it, _, res = fftk.solve(P, np.eye(2), "cg", 1e-8, 20000, 1.0)
    gxx, gyy = fftk.grad(u)
    sl = (slice(n // 2, n // 2 + n), slice(n // 2, n // 2 + n))
    S = np.array([[np.mean((P.cx * (gxx[j] + (j == 0)))[sl]) for j in range(2)],
                  [np.mean((P.cy * (gyy[j] + (j == 1)))[sl]) for j in range(2)]])
    return 0.5 * (S + S.T)


def one(args):
    L, r0, c0 = args
    n = int(round(L / H))
    o = ff[r0:r0 + n, c0:c0 + n]
    R = fftk.homogenize(o, "cg", tol=1e-8)
    s1, s2, ang = fftk.principal(R.sigma)
    la, coh = lanes(r0, c0, n)
    m = (cen[2] > 0) & (cen[0] >= r0) & (cen[0] < r0 + n) & (cen[1] >= c0) & (cen[1] < c0 + n)
    row = dict(L_m=L, r0=r0, c0=c0, open_frac=float(o.mean()),
               bldg_per_ha=m.sum() / (n * n * H * H / 1e4),
               sxx=R.sigma[0, 0], syy=R.sigma[1, 1], sxy=R.sigma[0, 1], s_major=s1, s_minor=s2,
               angle_deg=ang, iters=R.iters, seconds=R.seconds, lane_deg=la, lane_coh=coh)
    if L in (25, 50):
        So = oversampled(r0, c0, n)
        if So is not None:
            o1, o2, oa = fftk.principal(So)
            row.update(os_sxx=So[0, 0], os_syy=So[1, 1], os_sxy=So[0, 1], os_major=o1,
                       os_minor=o2, os_angle=oa)
    return row


if __name__ == "__main__":
    jobs = []
    for L in (12.5, 25, 50, 100):
        n = int(round(L / H))
        for r0 in range(0, ff.shape[0] - n + 1, n):
            for c0 in range(0, ff.shape[1] - n + 1, n):
                if full(r0, c0, n):
                    jobs.append((L, r0, c0))
    print(len(jobs), "tiles", flush=True)
    t = time.time()
    with ProcessPoolExecutor(8) as ex:
        rows = list(ex.map(one, jobs, chunksize=8))
    print(f"{time.time() - t:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv("tiles.csv", index=False)
