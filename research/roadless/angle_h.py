import sys; sys.path.insert(0, 'research/roadless')
import numpy as np, lifted
from checks import channel
for W in (2.0, 4.0):
    for ell in (3.0, 10.0):
        for h in (0.5, 0.25):
            ps = np.array([channel(a, W, 30.0, h, lifted.Params(ell_m=ell, K=16)) for a in (0, 10, 20, 30, 45)])
            print(f"W={W} ell={ell} h={h}: P {np.round(ps,2).tolist()} max/min {ps.max()/ps.min():.3f}", flush=True)
