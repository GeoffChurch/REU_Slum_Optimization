"""Grid8 against the runs from before the Router Strategy (owner: its results bit for bit, FP
tolerance fine): a run's rows (crossings.py's) against a pre-router run's on the same world and
plan, every column but the wall times (steps.t, world.json's t) compared, floats within TOL.

    S=research/roadless/atlas/routers/same.py          # from the repo root, PYTHONPATH=.:src
    python $S <new run dir> <old run dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-9             # relative, floats


def same_frame(a: pd.DataFrame, b: pd.DataFrame, *, what: str, skip: tuple[str, ...]) -> bool:
    ok = list(a.columns) == list(b.columns) and len(a) == len(b)
    if not ok:
        print(f"  {what}: shape {a.shape} vs {b.shape}, "
              f"columns equal {list(a.columns) == list(b.columns)}")
        return False
    for c in a.columns:
        if c in skip:
            continue
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype == object:
            good = all(len(p) == len(q)
                       and np.allclose(np.asarray(p, dtype=float), np.asarray(q, dtype=float),
                                       rtol=TOL, atol=0)
                       for p, q in zip(x, y, strict=True))
        elif np.issubdtype(x.dtype, np.floating):
            good = bool(np.allclose(x, y, rtol=TOL, atol=1e-12, equal_nan=True))
        else:
            good = bool(np.array_equal(x, y))
        if not good:
            print(f"  {what}.{c}: differs")
        ok &= good
    print(f"  {what}: {len(a)} rows, {'same' if ok else 'DIFFERENT'}")
    return ok


def main(new: Path, old: Path) -> None:
    ok = same_frame(pd.read_parquet(new / "steps.parquet"), pd.read_parquet(old / "steps.parquet"),
                    what="steps", skip=("t",))
    ok &= same_frame(pd.read_parquet(new / "picks.parquet"), pd.read_parquet(old / "picks.parquet"),
                     what="picks", skip=())
    ok &= same_frame(pd.read_parquet(new / "cands.parquet"), pd.read_parquet(old / "cands.parquet"),
                     what="cands", skip=())
    ok &= same_frame(pd.read_parquet(new / "tri_step.parquet"),
                     pd.read_parquet(old / "tri_step.parquet"), what="tri_step", skip=())
    a, b = np.load(new / "states.npz"), np.load(old / "states.npz")
    for k in a.files:
        good = np.allclose(a[k], b[k], rtol=TOL, atol=0) if a[k].dtype.kind == "f" else \
            np.array_equal(a[k], b[k])
        ok &= bool(good)
        if not good:
            print(f"  states.{k}: differs")
    print(f"same: -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
