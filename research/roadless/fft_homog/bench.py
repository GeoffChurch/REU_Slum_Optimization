"""Per-iteration cost of FFT-CG (2 loads) on one core vs window size; iterations on real fabric."""
import time

import fftk
import numpy as np

d = np.load("fabric_5810.npz")
ff = d["ff0"].astype(float)
ins = d["inside"]
r, c = 1100, 900
for n in (50, 100, 200, 400, 800, 1200):
    o = np.where(ins[r - n//2:r + n//2, c - n//2:c + n//2],
                 ff[r - n//2:r + n//2, c - n//2:c + n//2], 1.0)
    t = time.perf_counter()
    R = fftk.homogenize(o, "cg", tol=1e-8)
    sec = time.perf_counter() - t
    print(f"n {n:5d} ({n*0.5:.0f} m): iters {R.iters:4d}  {sec:.3f}s  "
          f"{1e3*sec/R.iters:.3f} ms/iter  sigma_iso {np.trace(R.sigma)/2:.3f}", flush=True)
