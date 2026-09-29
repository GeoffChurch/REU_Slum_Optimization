import sys
sys.path.insert(0, 'research/roadless')
import numpy as np
from checks import channel

import lifted

for W in (2.0, 4.0):
    for h in (0.5, 0.25):
        ps = np.array([channel(a, W, 30.0, h, lifted.Params(ell_m=3.0, K=8))
                       for a in (0, 10, 20, 30, 45)])
        print(f"W={W} h={h}: P {np.round(ps, 2).tolist()} max/min {ps.max() / ps.min():.3f}",
              flush=True)
