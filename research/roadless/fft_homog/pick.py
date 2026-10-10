"""Pick window centres in 5810: fully inside the block; densest / median / sparsest by building
count at 200 m (building positions = mean of each building's ring cells)."""
import numpy as np

d = np.load("fabric_5810.npz")
ff, ins, own = d["ff0"].astype(float), d["inside"], d["owner"]
nb = int(d["n_buildings"])
rr, cc = np.nonzero(own >= 0)
k = own[rr, cc]
cnt = np.bincount(k, minlength=nb)
cy = np.bincount(k, weights=rr, minlength=nb) / np.maximum(cnt, 1)
cx = np.bincount(k, weights=cc, minlength=nb) / np.maximum(cnt, 1)
ok = cnt > 0
B = np.zeros(ff.shape)
np.add.at(B, (cy[ok].astype(int), cx[ok].astype(int)), 1)
cov = np.where(ins, 1 - ff, 0.0)
def integral(a):
    return np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
Ii, Ic, Ib = integral(ins.astype(float)), integral(cov), integral(B)
def box(integ, r0, c0, n):
    return integ[r0 + n, c0 + n] - integ[r0, c0 + n] - integ[r0 + n, c0] + integ[r0, c0]
for n in (400, 200):
    rows = []
    for r0 in range(0, ff.shape[0] - n, 20):
        for c0 in range(0, ff.shape[1] - n, 20):
            if box(Ii, r0, c0, n) == n * n:
                rows.append((box(Ib, r0, c0, n) / (n * n * 0.25 / 1e4), box(Ic, r0, c0, n) / n**2,
                             r0 + n // 2, c0 + n // 2))
    rows.sort()
    print(n // 2, "m:", len(rows),
          "fully-inside windows; (bldg/ha, coverage, row, col) min/median/max")
    for r in (rows[0], rows[len(rows) // 4], rows[len(rows) // 2], rows[3 * len(rows) // 4],
              rows[-1]):
        print("   ", np.round(r, 3))
np.save("centroids.npy", np.stack([cy, cx, cnt]))
