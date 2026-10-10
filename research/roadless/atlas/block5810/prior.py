"""The no-buildings prior: the same homes' flow to the same edge with every building gone, under
each metric's conductance (the flux S = sum_k |phi_k| per cell, as the page's layers)."""
import dataclasses
import sys
from pathlib import Path

import numpy as np

HERE = Path("/home/gchurchill/src/reblock/research/roadless")
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
from current_map import flow  # noqa: E402

OUT = Path(sys.argv[1])
[b] = common.build_blocks(["ZAF.9.3.1_1_5810"])
base = lifted.Params(3.0, 8, solver=lifted.solver_of("gpu"))
sc = common.Scorer(b, lifted.UniformMesh(0.5, offset=lifted.OFFSET), base,
                   population=common.POPULATIONS["area"])
g = sc.grid
open_all = g.isub.mean(axis=-1)
out = {}
for key, along in (("uni", "uni"), ("ss", "ss100k2n2r30")):
    p = dataclasses.replace(base, along=lifted.along_of(along, lifted.scans_of("gpu")))
    f = flow(sc, open_all, p)
    out[f"open_{key}"] = np.abs(f.phi).sum(axis=0).astype("float32")
    print(key, "P open", f.P, flush=True)
np.savez_compressed(OUT / "raw_prior.npz", **out)
