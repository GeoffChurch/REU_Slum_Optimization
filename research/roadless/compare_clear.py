"""Road-free greedy clearing vs the road lineup, on the roadless (carve) score.

    PYTHONPATH=. uv run python research/roadless/compare_clear.py <M4|B0.01g3> [h] [pop] [power]

Greedy D counts whole buildings; the lineup's D is the overlap fraction its corridor takes. Lens A:
the first greedy step with D >= 0.10 (the lineup's prefix is likewise the first reaching it).
Lens B: least D reaching P*' (0.25, 0.35), linear between steps / prefixes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from study import SHORT, _ci, _d_at, rows_dir  # noqa: E402


def main(picker: str, h: float, pop: str, power: float) -> None:
    from clear import rows_dir as clear_dir
    g = pd.concat([pd.read_parquet(p) for p in clear_dir(picker, h, pop, power).glob("*.parquet")],
                  ignore_index=True)
    lin = pd.concat([pd.read_parquet(p) for p in rows_dir(h, 3.0, 8, pop).glob("*.parquet")],
                    ignore_index=True)
    lin["arm"] = lin["arm"].map(SHORT)
    blocks = sorted(set(g.block))
    lin = lin[lin.block.isin(blocks)]
    print(f"{len(blocks)} blocks with greedy {picker} rows\n")
    # Lens A
    gA = g[g.D >= 0.10 - 1e-9].sort_values("step").groupby("block").head(1).set_index("block")
    A = lin[(lin.budget == 0.10) & (lin.D >= 0.10 - 1e-9)].pivot_table(
        index="block", columns="arm", values="P_carve")
    A["clear_bldg"] = gA.perm
    print("LENS A (10% displaced), roadless perm: median, and greedy minus arm [CI], greedy wins")
    for arm in A.columns:
        print(f"  {arm:13s} {A[arm].median():.3f}", end="")
        if arm != "clear_bldg":
            s = (A.clear_bldg - A[arm]).dropna()
            print(f"   greedy {s.median():+.3f} {_ci(s)}  wins {np.mean(s > 0):.2f} [{len(s)}]",
                  end="")
        print()
    # Lens B
    for pstar in (0.25, 0.35):
        rows = []
        for blk, gg in g.groupby("block"):
            rows.append(dict(block=blk, arm="clear_bldg",
                             B=_d_at(gg.rename(columns={"perm": "P"}), "P", pstar)))
        for (blk, arm), gg in lin.groupby(["block", "arm"]):
            rows.append(dict(block=blk, arm=arm, B=_d_at(gg, "P_carve", pstar)))
        B = pd.DataFrame(rows).pivot_table(index="block", columns="arm", values="B",
                                           dropna=False)
        print(f"\nLENS B P*' {pstar}: reaching, median D, greedy minus arm [CI] (negative = "
              f"greedy displaces less), greedy wins")
        for arm in B.columns:
            print(f"  {arm:13s} {B[arm].notna().sum():4d} {B[arm].median():.4f}", end="")
            if arm != "clear_bldg":
                s = (B.clear_bldg - B[arm]).dropna()
                print(f"   {s.median():+.4f} {_ci(s)}  wins {np.mean(s < 0):.2f} [{len(s)}]",
                      end="")
            print()
        # frontier (A perm up, B displacement down)
        AB = A.stack().rename("A").reset_index().merge(
            B.stack().rename("B").reset_index(), on=["block", "arm"])
        tally: dict[str, int] = {}
        for _, gg in AB.groupby("block"):
            for _, r in gg.iterrows():
                dom = ((gg.A >= r.A) & (gg.B <= r.B) & ((gg.A > r.A) | (gg.B < r.B))).any()
                if not dom:
                    tally[r.arm] = tally.get(r.arm, 0) + 1
        print("  frontier: " + ", ".join(f"{a} {n}" for a, n in
                                         sorted(tally.items(), key=lambda t: -t[1])))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.5,
         sys.argv[3] if len(sys.argv) > 3 else "count",
         float(sys.argv[4]) if len(sys.argv) > 4 else 1.0)
