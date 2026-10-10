"""Open fractions of 5810 after four stored clearings (union of a run's steps 1..k), plus the
per-cell majority building label (for tile clearings). Run from the reblock repo root.
Writes clearings_5810.npz beside this script."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "research/roadless")
import common  # noqa: E402
import lifted  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = {"simp": "clear_rows_SIMPfw0.q3.i10.t0.001.e0.0001.k1_h0.5_area_p2",
        "greedy": "clear_rows_S0.01cat_h0.5_area_p2"}
[b] = common.build_blocks(["ZAF.9.3.1_1_5810"])
p = lifted.Params(3.0, 8, solver=lifted.solver_of("cpu"))
g = lifted.UniformMesh(0.5, offset=lifted.OFFSET).build(b, p, common.POPULATIONS["area"])
polys = np.asarray(b.buildings.outlines)
lab = g.label_sub(polys)                       # (ny, nx, 16), -1 none
out = {}
for name, d in RUNS.items():
    rows = pd.read_parquet(Path("research/roadless") / d / "ZAF.9.3.1_1_5810.parquet")
    for step in (5, 10):
        s = rows[(rows.step >= 1) & (rows.step <= step)]
        ids = np.unique(np.concatenate([np.asarray(c, dtype=np.int64) for c in s.cleared]))
        removed = np.zeros(len(polys), dtype=bool)
        removed[ids] = True
        cleared = (lab >= 0) & removed[np.maximum(lab, 0)]
        ff = (g.isub & (~g.bsub | cleared)).mean(axis=-1)
        key = f"{name}_D{step / 100:.2f}"
        out[f"ff_{key}"] = ff.astype(np.float32)
        out[f"ids_{key}"] = ids
        out[f"perm_{key}"] = float(rows[rows.step == step].perm.iloc[0])
        print(key, len(ids), out[f"perm_{key}"], flush=True)
# majority label per cell (ties: any); -1 where no footprint sub-sample
L = lab.reshape(-1, lab.shape[-1])
maj = np.full(L.shape[0], -1, dtype=np.int32)
has = (L >= 0).any(axis=1)
for i in np.flatnonzero(has):
    v = L[i][L[i] >= 0]
    maj[i] = np.bincount(v).argmax()
out["label"] = maj.reshape(lab.shape[:2])
# per cell, the count of each building's sub-samples is needed to open a building exactly:
# store the (cell, building, count) triples
r = np.repeat(np.arange(L.shape[0]), L.shape[1])
k = L.ravel()
ok = k >= 0
pairs = np.unique(np.stack([r[ok], k[ok]]), axis=1, return_counts=True)
out["sub_cell"], out["sub_bldg"] = pairs[0][0].astype(np.int64), pairs[0][1].astype(np.int32)
out["sub_count"] = pairs[1].astype(np.int16)
np.savez_compressed(HERE / "clearings_5810.npz", **out)
print("done", flush=True)
