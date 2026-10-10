"""The sightline arm for the 5810 page: the translucent-search greedy under the sightline metric
(ss100k2n2r30, search ss100k0.5n2r30) cut to D 0.05, its flow under its own conductance, and
every clearing's Lens A under both metrics."""
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path("/home/gchurchill/src/reblock/research/roadless")
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402
import relax  # noqa: E402
from clear import cleared_through  # noqa: E402
from current_map import flow  # noqa: E402

BID, BUDGET, SS = "ZAF.9.3.1_1_5810", 0.05, "ss100k2n2r30"
OUT = Path(sys.argv[1])
mesh = lifted.UniformMesh(0.5, offset=lifted.OFFSET)
[b] = common.build_blocks([BID])
scans = lifted.scans_of("gpu")
base = lifted.Params(3.0, 8, solver=lifted.solver_of("gpu"))
sc = common.Scorer(b, mesh, base, population=common.POPULATIONS["area"])
g = sc.grid
lab = g.label_sub(sc.polys)
cost = sc.w / sc.w.sum()
rows = pd.read_parquet(HERE / f"clear_rows_S0.01cat_h0.5_area_p2_{SS}@ss100k0.5n2r30"
                       / f"{BID}.parquet").sort_values("step")
take, left = [], BUDGET + 1e-12
for j in cleared_through(rows, int(rows.step.max())):
    if cost[j] <= left:
        take.append(j)
        left -= cost[j]
take = np.array(take, dtype=int)
print("sightline greedy:", len(take), "cleared, D", cost[take].sum(), flush=True)
p = dataclasses.replace(base, along=lifted.along_of(SS, scans))
f0 = flow(sc, g.ff0, p)
after = (g.isub & (~g.bsub | np.isin(lab, take))).mean(axis=-1)
f1 = flow(sc, after, p)
was = g.ff0.ravel() > 0
with np.errstate(divide="ignore", invalid="ignore"):
    ru = np.where(was & (f0.u > 0), (f1.u - f0.u) / f0.u, np.nan)
np.savez_compressed(OUT / "raw_ss.npz", s0=np.abs(f0.phi).sum(axis=0).astype("float32"),
                    s1_ss=np.abs(f1.phi).sum(axis=0).astype("float32"),
                    ru_ss=ru.astype("float32"), cleared=take)
print("escape median change", float(np.nanmedian(ru)), flush=True)
del sc, f0, f1
clear = pd.read_parquet(HERE / "fine_clearings.parquet")
sets = {n: np.asarray(c, dtype=int) for n, c in
        zip(clear[clear.block == BID].name, clear[clear.block == BID].cleared, strict=True)
        if n in ("cheap", "default")}
sets["sightline"] = take
cross = {}
for along in ("uni", SS):
    c = relax._clearing(BID, "gpu", along, mesh)
    rel = relax.Relaxation(c, 2.0)
    for name, cl in sets.items():
        r = np.zeros(c.n)
        r[cl] = 1.0
        cross[f"{name}@{along}"] = float(rel.perm(rel.exact(r)))
        print(name, along, cross[f"{name}@{along}"], flush=True)
    del rel, c
(OUT / "layers" / "cross.json").write_text(json.dumps(
    {"cross": cross, "sightline_n": int(len(take)), "sightline_D": float(cost[take].sum())},
    indent=1))
