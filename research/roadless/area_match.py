"""Roadless permeability at matched FOOTPRINT AREA cleared (not homes): the greedy's size discount
removed. Lineup: the study's prefixes (first n_roads of the street-first order), area = footprint
area under the corridor. Greedy: cumulative area of the cleared buildings. Linear interpolation on
each curve through (0, 0); nan past its end.

    PYTHONPATH=. uv run python research/roadless/area_match.py [M]
"""
from __future__ import annotations

import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from clear import cleared_through  # noqa: E402
from study import SHORT, _ci, rows_dir  # noqa: E402

TARGETS = (0.10, 0.15, 0.20)
_B: dict = {}
_ARMS: dict = {}
_M = 4


def curves(bid: str) -> list[dict]:
    from reblock.budget import STREET_TOL, road_corridor, street_first_ordered
    from reblock.derivations import propose
    b = _B[bid]
    polys = np.asarray(b.buildings.outlines)
    area = shapely.area(polys)
    tot = area.sum()
    out = []
    g = pd.read_parquet(HERE / f"clear_rows_M{_M}_h0.5" / f"{bid}.parquet").sort_values("step")
    out += [dict(block=bid, arm="clear_bldg", A=0.0, P=0.0)]
    out += [dict(block=bid, arm="clear_bldg", A=float(area[cleared_through(g, k)].sum() / tot),
                 P=float(p))
            for k, p in zip(g.step.to_numpy()[1:], g.perm.to_numpy()[1:], strict=True)]
    lin = pd.read_parquet(rows_dir(0.5, 3.0, 8) / f"{bid}.parquet")
    for arm, gg in lin.groupby("arm"):
        roads = propose(_ARMS[arm], b).roads
        ordered = street_first_ordered(b, roads, STREET_TOL)
        out.append(dict(block=bid, arm=SHORT[arm], A=0.0, P=0.0))
        for n, p in gg[["n_roads", "P_carve"]].drop_duplicates().to_numpy():
            c = road_corridor(ordered.iloc[:int(n)])
            a = float(shapely.area(shapely.intersection(polys, c)).sum() / tot)
            out.append(dict(block=bid, arm=SHORT[arm], A=a, P=float(p)))
    return out


def at(g: pd.DataFrame, a: float) -> float:
    g = g.sort_values("A")
    A, P = g.A.to_numpy(), np.maximum.accumulate(g.P.to_numpy())
    if a > A.max():
        return np.nan
    return float(np.interp(a, A, P))


def main(M: int) -> None:
    global _B, _ARMS, _M
    _M = M
    done = sorted(p.stem for p in (HERE / f"clear_rows_M{M}_h0.5").glob("*.parquet"))
    _B = {b.block_id: b for b in common.build_blocks(common.recipients())
          if b.block_id in set(done)}
    _ARMS = common.arms()
    with multiprocessing.get_context("fork").Pool(16) as pool:
        rows = [r for rs in pool.imap_unordered(curves, done) for r in rs]
    d = pd.DataFrame(rows)
    d.to_parquet(HERE / f"area_curves_M{M}.parquet")
    print(f"{len(done)} blocks")
    for a in TARGETS:
        v = pd.DataFrame([dict(block=blk, arm=arm, P=at(g, a))
                          for (blk, arm), g in d.groupby(["block", "arm"])]).pivot_table(
            index="block", columns="arm", values="P", dropna=False)
        print(f"\nroadless perm at {a:.0%} of footprint AREA cleared: median, greedy minus arm")
        for arm in v.columns:
            print(f"  {arm:13s} {v[arm].notna().sum():4d} {v[arm].median():.3f}", end="")
            if arm != "clear_bldg":
                s = (v.clear_bldg - v[arm]).dropna()
                if len(s):
                    print(f"   {s.median():+.3f} {_ci(s)} wins {np.mean(s > 0):.2f}", end="")
            print()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
