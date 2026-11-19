"""The study's tables (results/): every run analysed (analyze.py), grouped by condition (the
world's kind and the plan's building cost), crossings pooled per condition against both nulls,
and the H1/H2 measures per condition.

    S=research/roadless/atlas/crossings/summary.py      # from the repo root, PYTHONPATH=.
    uv run python $S
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze as an  # noqa: E402

ROWS = HERE / "rows"
ALIGNED = 0.25     # a road's halves "aligned": their deepest spots closer along it than 3 in 4
                   # independent uniform draws (analyze.halves_of's pct)
OUT = HERE / "results"


def condition(world: str, plan: str) -> str:
    kind = world.split(":")[0]
    if kind == "fabric":
        kind = f"fabric-{world.split(':')[2]}"
    b = re.search(r"\.b([0-9.]+)\.", plan)[1]  # type: ignore[index]
    return f"{kind} b{b}"


def mechanism(d: Path) -> dict[str, float]:
    """The H1/H2 measures of one run (analyze's ends and halves)."""
    e = pd.read_parquet(d / "ends.parquet")
    h = pd.read_parquet(d / "halves.parquet")
    if "deep_pct" not in e:            # no end joined an ear: only the halves count
        e = e.reindex(columns=[*e.columns, "deep_pct", "ridge_pct", "lane_arm", "lane_far"])
    on = e[e.road >= 0]
    c, t = on[on.crossing], on[~on.crossing]
    p = pd.read_parquet(d / "pairs.parquet")
    crossed = set(p.road[p.ds <= an.TOL_M]) if len(p) else set()
    hit = h.road.isin(crossed) if len(h) else pd.Series(dtype=bool)
    al = h.pct <= ALIGNED if len(h) else pd.Series(dtype=bool)
    return dict(
        aligned_roads=int(al.sum()), aligned_crossed=int((al & hit).sum()),
        other_roads=int((~al).sum()), other_crossed=int((~al & hit).sum()),
        ends_on_ears=len(on), crossing_ends=len(c),
        deep_pct_sum=float(on.deep_pct.sum()), deep_n=int(on.deep_pct.notna().sum()),
        halves_pct_sum=float(h.pct.sum()) if len(h) else 0.0, halves_n=len(h),
        ridge_pct_sum=float(on.ridge_pct.sum()),
        lag_cross_sum=float(c.lag.sum()), lag_t_sum=float(t.lag.sum()), n_t=len(t),
        bld_cross_sum=float(c.bld_share.sum()), bld_t_sum=float(t.bld_share.sum()),
        lane_arm_cross_sum=float(c.lane_arm.sum()), lane_arm_t_sum=float(t.lane_arm.sum()),
        lane_far_t_open=int((t.lane_far >= 10).sum()))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    dirs = sorted(p.parent for p in ROWS.glob("*/*/states.npz"))
    rows, nulls, mech = [], [], []
    for d in dirs:
        r, n = an.analyze(d, budget=None)
        r["condition"] = condition(str(r["world"]), str(r["plan"]))
        rows.append(r)
        nulls.append(n)
        mech.append({"condition": r["condition"], **mechanism(d)})
    runs = pd.DataFrame(rows)
    pooled = []
    for cond, idx in runs.groupby("condition").groups.items():
        sel = list(idx)
        p = an.pooled([nulls[i] for i in sel], [rows[i] for i in sel])
        p["world"] = f"pooled: {cond}"
        p["condition"] = cond
        p["runs"] = len(sel)
        pooled.append(p)
    table = pd.concat([runs, pd.DataFrame(pooled)], ignore_index=True)
    table.drop(columns="plan").to_csv(OUT / "crossings.csv", index=False, float_format="%.4g")
    m = pd.DataFrame(mech).groupby("condition").sum()
    out = pd.DataFrame(dict(
        ends_on_ears=m.ends_on_ears, crossing_ends=m.crossing_ends,
        deep_pct_mean=m.deep_pct_sum / m.deep_n,
        halves_pct_mean=m.halves_pct_sum / m.halves_n, halves_n=m.halves_n,
        ridge_pct_mean=m.ridge_pct_sum / m.ends_on_ears,
        lag_crossing=m.lag_cross_sum / m.crossing_ends, lag_t=m.lag_t_sum / m.n_t,
        bld_share_crossing=m.bld_cross_sum / m.crossing_ends, bld_share_t=m.bld_t_sum / m.n_t,
        lane_arm_crossing=m.lane_arm_cross_sum / m.crossing_ends,
        lane_arm_t=m.lane_arm_t_sum / m.n_t,
        t_far_side_open_lane=m.lane_far_t_open / m.n_t,
        crossed_if_aligned=m.aligned_crossed / m.aligned_roads, aligned_roads=m.aligned_roads,
        crossed_otherwise=m.other_crossed / m.other_roads, other_roads=m.other_roads))
    out.to_csv(OUT / "mechanism.csv", float_format="%.4g")
    cols = ["world", "runs", "ears", "opposite_pairs"] + [
        f"{k}{x:g}" for x in an.TOLS for k in ("obs", "null", "p", "lnull", "lp")]
    with pd.option_context("display.width", 400, "display.max_columns", 60):
        print(pd.DataFrame(pooled)[cols].round(3).to_string(index=False))
        print(out.round(3).to_string())


if __name__ == "__main__":
    main()
