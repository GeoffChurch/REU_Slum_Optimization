import fftk
import numpy as np

rng = np.random.default_rng(0)
# homogeneous
r = fftk.homogenize(np.ones((32, 40)))
print("uniform", r.sigma.round(12).tolist(), r.iters)
# laminate: rows 0..9 open, 10..15 closed -> sigma_xx = 10/16 (x-faces in open rows), sigma_yy = 0
o = np.ones((16, 20))
o[10:] = 0
r = fftk.homogenize(o)
print("laminate", r.sigma.round(10).tolist(), r.iters)
# random two-phase with partial fractions, compare direct
o = (rng.random((60, 50)) > 0.35).astype(float) * rng.choice([1.0, 0.5, 0.75], size=(60, 50))
for s in ["cg", "ms", "al", "em"]:
    r = fftk.homogenize(o, s, tol=1e-10, maxiter=20000)
    print(s, r.sigma.round(8).tolist(), r.iters,
          f"{r.seconds:.3f}s res {r.resid:.1e} asym {r.asym:.1e}")
S, sec, u = fftk.direct(o)
print("direct", S.round(8).tolist(), f"{sec:.3f}s")
