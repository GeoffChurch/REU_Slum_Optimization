"""c0 sweep for the fixed-point schemes on one 50 m and one 100 m dense window (tol 1e-6)."""
import fftk
import numpy as np

d = np.load("fabric_5810.npz")
ff = d["ff0"].astype(float)
for (r, c), n in [((960, 1280), 100), ((1240, 700), 200)]:
    o = ff[r - n//2:r + n//2, c - n//2:c + n//2]
    for s, c0s in [("ms", [0.5, 0.55, 0.65, 0.8, 1.0]), ("al", [0.02, 0.05, 0.1, 0.2, 0.4, 1.0]),
                   ("em", [0.02, 0.05, 0.1, 0.2, 0.4, 1.0])]:
        out = []
        for c0 in c0s:
            R = fftk.homogenize(o, s, tol=1e-6, maxiter=6000, c0=c0)
            out.append(f"c0={c0}:{R.iters}{'' if R.resid < 1e-6 else f'(nc {R.resid:.0e})'}")
        print((r, c), n, s, " ".join(out), flush=True)
