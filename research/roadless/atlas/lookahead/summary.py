"""The lookahead study's tables (results/): per block and search, L_P at matched D (BUDGETS) along
the frontier of the states it held (lookahead.frontier, linearly interpolated; `best`: the most
L_P held at D or less, no interpolation), and at greedy's cumulative cost there (the plan's cost,
nu P0 D + price P0 / SHEET_M2 C, over P0); each search less greedy per block, and pooled over the
blocks reaching the budget with a paired t interval; wall time to each budget and in all, as a
multiple of greedy's; and the crossings in each search's replayed network (crossings/analyze.py's
definition, TOL_M and ROAD_W), against greedy's.

    S=research/roadless/atlas/lookahead/summary.py      # from the repo root, PYTHONPATH=.
    uv run python $S
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "crossings"))
import analyze as an  # noqa: E402
import lookahead as la  # noqa: E402

BUDGETS = (0.05, 0.10, 0.15)
OUT = HERE / "results"


def interp(fr: pd.DataFrame, x: str, at: float) -> float:
    """L_P along the frontier `fr` (by x, increasing) at x = at; nan past its end."""
    xs, ys = fr[x].to_numpy(), fr.L_P.to_numpy()
    if at > xs[-1] + 1e-12:
        return np.nan
    return float(np.interp(at, xs, ys))


def readout(d: Path, *, nu: float, price: float) -> tuple[dict[str, float], pd.DataFrame]:
    tr = pd.read_parquet(d / "trace.parquet")
    tr["cost"] = nu * tr.D + price / la.gr.rs.SHEET_M2 * (tr.C - tr.C.iloc[0])
    fr = la.frontier(tr)
    row: dict[str, float] = dict(t=float(tr.t.iloc[-1]), states=len(tr),
                                 D_max=float(tr.D.max()), ears=len(tr.ears.iloc[-1]))
    top = BUDGETS[-1]
    grid = np.linspace(0.0, top, 61)
    row["auc"] = (float(np.mean([interp(fr, "D", x) for x in grid]))
                  if fr.D.iloc[-1] >= top else np.nan)    # mean L_P over D in [0, top]
    for b in BUDGETS:
        row[f"L{b:g}"] = interp(fr, "D", b)
        under = tr[tr.D <= b]
        row[f"best{b:g}"] = float(under.L_P.max())
        reach = tr[tr.D >= b]
        row[f"t{b:g}"] = float(reach.t.iloc[0]) if len(reach) else np.nan
    return row, fr.sort_values("cost")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rows = []
    fronts: dict[tuple[str, str], pd.DataFrame] = {}
    for d in sorted(la.ROWS.glob("ZAF*/*.ears.*")):
        if not (d / "trace.parquet").exists():
            continue
        search, plan_name = d.name.split(".", 1)
        plan = la.ears.Plan.of(plan_name.rsplit(".v", 1)[0])
        r, fr = readout(d, nu=plan.cost.nu, price=plan.cost.price)
        fronts[(d.parent.name, search)] = fr
        r.update(block=d.parent.name.rsplit("_", 1)[1], search=search)
        if (d / "replay" / "world.json").exists():
            c, _ = an.analyze(d / "replay", budget=None)
            r.update(x_ears=c["ears"], x_pairs=c["opposite_pairs"], x3_5=c["obs3.5"],
                     x7=c["obs7"], x10=c["obs10"])
        rows.append(r)
    t = pd.DataFrame(rows)
    # at greedy's cost at each budget
    for b in BUDGETS:
        vals = []
        for r in t.itertuples():
            g = t[(t.block == r.block) & (t.search == "greedy")]
            gfr = fronts.get((f"ZAF.9.3.1_1_{r.block}", "greedy"))
            fr = fronts[(f"ZAF.9.3.1_1_{r.block}", r.search)]
            if gfr is None or g.empty or np.isnan(getattr(g.iloc[0], f"L{b:g}")):
                vals.append(np.nan)
                continue
            c_at = float(np.interp(b, gfr.sort_values("D").D, gfr.sort_values("D").cost))
            vals.append(interp(fr, "cost", c_at))
        t[f"Lc{b:g}"] = vals
    t.to_csv(OUT / "runs.csv", index=False, float_format="%.5g")
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["block", "search", "ears", "D_max", "t", "auc"] + [f"L{b:g}" for b in BUDGETS] + [
            "x_ears", "x3_5", "x7"]
        print(t[[c for c in cols if c in t]].round(4).to_string(index=False))
    # pooled, paired against greedy
    out = []
    g = t[t.search == "greedy"].set_index("block")
    for s, grp in t[t.search != "greedy"].groupby("search"):
        grp = grp.set_index("block")
        blocks = grp.index.intersection(g.index)
        row: dict[str, object] = dict(search=s, blocks=len(blocks))
        for key in ["auc"] + [f"L{b:g}" for b in BUDGETS] + [f"best{b:g}" for b in BUDGETS] + [
                f"Lc{b:g}" for b in BUDGETS]:
            dd = (grp.loc[blocks, key] - g.loc[blocks, key]).dropna()
            n = len(dd)
            m = float(dd.mean()) if n else np.nan
            h = float(stats.t.ppf(0.975, n - 1) * dd.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            row[key] = (f"{m:+.4f} [{m - h:+.4f}, {m + h:+.4f}] n{n} "
                        f"({(dd > 0).sum()}+/{(dd < 0).sum()}-)")
        for key in [f"t{b:g}" for b in BUDGETS] + ["t"]:
            ratio = (grp.loc[blocks, key] / g.loc[blocks, key]).dropna()
            row[f"x{key}"] = (f"{np.exp(np.log(ratio).mean()):.2f} "
                              f"({ratio.min():.2f}-{ratio.max():.2f})")
        if "x3_5" in grp:
            for key in ("x_ears", "x3_5", "x7", "x10"):
                row[key] = f"{grp.loc[blocks, key].sum():g} vs {g.loc[blocks, key].sum():g}"
        out.append(row)
    p = pd.DataFrame(out)
    p.to_csv(OUT / "pooled.csv", index=False)
    for r in out:
        print()
        for k, v in r.items():
            print(f"  {k:>9}: {v}")


if __name__ == "__main__":
    main()
