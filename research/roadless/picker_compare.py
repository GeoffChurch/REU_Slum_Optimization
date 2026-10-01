"""Two greedy pickers on the same objective, paired per block: Lens A (at D = 0.10, linear
between steps, so pickers with coarse steps are not scored past it) perm (J_power) and perm1 (P), Lens B least D reaching perm1 0.25 / 0.35, and time.

    PYTHONPATH=. pixi run python research/roadless/picker_compare.py <a> <b> [pop] [power]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clear import rows_dir  # noqa: E402
from study import _ci, _d_at  # noqa: E402


def _at(g: pd.DataFrame, col: str, d: float) -> float:
    """`col` at exactly D = d, linear between the steps either side (nan past the end)."""
    g = g.sort_values("D")
    return float(np.interp(d, g.D, g[col])) if g.D.max() >= d - 1e-9 else np.nan


def load(picker: str, pop: str, power: float) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(p) for p in rows_dir(picker, 0.5, pop, power)
                      .glob("*.parquet")], ignore_index=True)


def secs(picker: str, pop: str, power: float) -> pd.Series:
    log = HERE / f"clear_{picker}_{pop}_p{power:g}.log"
    rows = [re.match(r"\S+ (\S+) n=\d+ \d+ steps.* (\d+)s$", ln) for ln in log.read_text().split("\n")]
    return pd.Series({m.group(1): float(m.group(2)) for m in rows if m})


def main(a: str, b: str, pop: str, power: float) -> None:
    per = {}
    for pk in (a, b):
        g = load(pk, pop, power)
        cols = {col: pd.Series({blk: _at(gg, col, 0.10) for blk, gg in g.groupby("block")})
                for col in ("perm", "perm1")}
        cols = dict(A=cols["perm"], A1=cols["perm1"])
        for ps in (0.25, 0.35):
            cols[f"B{ps}"] = pd.Series({blk: _d_at(gg.rename(columns={"perm1": "P"}), "P", ps)
                                        for blk, gg in g.groupby("block")})
        cols["t"] = secs(pk, pop, power)
        per[pk] = pd.DataFrame(cols)
    x, y = per[a], per[b]
    blocks = x.index.intersection(y.index)
    print(f"{len(blocks)} blocks; {b} minus {a}: median [CI], share where {b} is better")
    for col, better in (("A", 1), ("A1", 1), ("B0.25", -1), ("B0.35", -1)):
        s = (y.loc[blocks, col] - x.loc[blocks, col]).dropna()
        print(f"  {col:6s} {a} {x.loc[blocks, col].median():.4f}  {b} {y.loc[blocks, col].median():.4f}"
              f"  diff {s.median():+.4f} {_ci(s)}  better {np.mean(better * s > 0):.2f} [{len(s)}]")
    t = (x.loc[blocks, "t"].sum(), y.loc[blocks, "t"].sum())
    print(f"  time: {a} {t[0] / 3600:.1f} h, {b} {t[1] / 3600:.1f} h (x{t[0] / t[1]:.1f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "count",
         float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
