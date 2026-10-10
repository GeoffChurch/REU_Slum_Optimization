"""The actual metric's baseline on 5810 ('uni': the lifted K = 8 graph, ell 3 m, h 0.5), on the
CPU: per-home u0, J2, P. Run from the reblock repo root. Writes lifted_ref_5810.npz."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "research/roadless")
import common  # noqa: E402
import lifted  # noqa: E402

HERE = Path(__file__).resolve().parent
t0 = time.time()
[b] = common.build_blocks(["ZAF.9.3.1_1_5810"])
p = lifted.Params(3.0, 8, solver=lifted.solver_of("cpu"))
sc = common.Scorer.__new__(common.Scorer)
g = lifted.UniformMesh(0.5, offset=lifted.OFFSET).build(b, p, common.POPULATIONS["area"])
polys = np.asarray(b.buildings.outlines)
w = common.POPULATIONS["area"].weights(polys)
reach = lifted.grounded(g, g.ff0, p)
f, stranded, owner = lifted.demand(g, polys, reach, w)
t1 = time.time()
sol = lifted.solve(g, g.ff0, f, p, rtol=1e-6)
t2 = time.time()
ub = lifted.cell_mean_u(sol, g.ff0, p)
on = owner >= 0
num = np.bincount(owner[on], weights=(f * ub)[on], minlength=len(polys))
u = np.where(~stranded, num / w, np.nan)
J2 = float(np.nansum(w * u ** 2))
np.savez_compressed(HERE / "lifted_ref_5810.npz", u=u, ucell=ub.astype(np.float32), P=sol.P,
                    J2=J2, n_unknowns=sol.n_unknowns, solve_s=t2 - t1)
print(f"unknowns {sol.n_unknowns} P {sol.P:.6g} J2 {J2:.6g} "
      f"setup {t1 - t0:.0f}s solve {t2 - t1:.0f}s", flush=True)
