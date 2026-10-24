"""Walls that leak: does the exit price shift P uniformly? (owner 2026-10-21: "if the current
sources are always the interiors of buildings, isn't the effect of varying the egress resistance
(close to) a uniform shift in total power dissipation P? If so, why not just use symmetric walls
since it would, I'd guess, not change which buildings are cleared?"; NOTES, "Walls that leak",
the single-clearing test.)

Each building j of a block cleared alone at baseline, its gain dP_j = P0 - P(j cleared) and its
gain per cost dP_j / D_j (D_j the metric's own cost of j: its population share) under:
SETTINGS (W on UNIFORM: A symmetric 3,000 m, the owner's proposal at setting 4's entry price; B
the diode out 10 m, in 3,000 m; C setting 4, out 100 m, in 3,000 m; S1 symmetric 200 m, setting 1)
and T (RUNTIME, P over the homes it scores). The diodes start each clearing's loop from the
baseline's directions (gap_walls.WarmStart).

    S=research/roadless/leaky_single.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S block <id> [--replace]       # ROWS/<id>.parquet: a row per building
    uv run python $S passes <id> <k>              # the diode B's passes per solve, cold and warm,
                                                  # on k clearings (ROWS/passes_<id>.parquet)
    uv run python $S report                       # ROWS/summary_{pairs,shift,drivers,frees}.csv

Per building (`block`): D_j and its population; dP_j under each setting and T, and each diode's
passes; the open wall its clearing adds to the other homes (their frontage on open ground gained,
m); its own wall's dissipation at baseline under each setting (the current leaving it through its
walls and street faces, g du^2: what clearing it removes from P, besides what it does for others);
the walled-in homes its clearing frees (T's B (i)) and their population.

`report`, by block and pooled over the blocks: (1) per pair of A, B, C, S1 and T, Spearman's
correlation of dP_j / D_j across the buildings and the Jaccard of the top TOP_K and of the top
TOP_SHARE by it; (2) the owner's claim in its exact form, how uniform the shift is: std_j[dP_j(X)
- dP_j(Y)] / std_j[dP_j(Y)] for A against B and C against B (the exit price's effect, the entry
price fixed); (3) what drives a shift that is not uniform: Spearman's correlation of dP_j(X) -
dP_j(B) with the frontage gained and with the difference of j's own-wall dissipation; (4) the
clearings that free walled-in homes: how many are in each setting's top TOP_K and top TOP_SHARE,
and the best rank among them. Over the blocks: each per-block measure's median, min and max; (1)
also pooled over all the blocks' buildings at once, each gain per cost normalised by its block's
P0 under that setting (so blocks are comparable); (4) also summed.

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/leaky_single.py
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_calib  # noqa: E402
import gap_walls as gw  # noqa: E402
import metrics  # noqa: E402

ROWS = HERE / "leaky_rows" / "single"
SETTINGS = {"A": "Wk1l3000", "B": "Wk1o10i3000n64", "C": "Wk1o100i3000n64", "S1": "Wk1l200"}
BLOCKS = ("ZAF.9.3.1_1_19570", "ZAF.9.3.1_1_41940", "ZAF.9.3.1_1_38486", "ZAF.9.3.1_1_63718",
          "ZAF.9.3.1_1_44837", "ZAF.9.3.1_1_21622")   # 63718, 44837, 21622 hold homes with no
                                                      # open ground within ring_m
POP = common.PopulationDemand()
TOP_K = 10
TOP_SHARE = 0.05
PAIRS = (("A", "B"), ("A", "C"), ("B", "C"), ("A", "T"), ("B", "T"), ("C", "T"), ("S1", "T"),
         ("S1", "A"))
SEED = 20261021


def frontage(wm: gw.WallModel, st: gw.WallState) -> NDArray[np.float64]:
    """Per home its frontage on open ground at the state (m): the wall passages one side of which
    is open, their length, to each standing home holding the other side."""
    w = st.wall
    ia, ib = st.inside[wm.pa[w]], st.inside[wm.pb[w]]
    one = ia ^ ib
    tri = np.where(ia, wm.pa[w], wm.pb[w])[one]
    M = (wm.zone[tri] @ sp.diags(st.standing.astype(np.float64))).tocsr()
    M.eliminate_zeros()
    M.data[:] = 1.0
    return np.asarray(M.T @ wm.L[w][one]).ravel()


def own_dissipation(wm: gw.WallModel, wl: gw.WallLenses) -> NDArray[np.float64]:
    """Per home the dissipation in the walls its current leaves it through at the state (each wall
    passage's g du^2 to the homes it leaves going its current's way; its inside street faces' and
    obtuse exits' to their homes)."""
    st, s = wl.state, wl.pop
    c = wm.sys.cells
    w = st.wall
    stand = sp.diags(st.standing.astype(np.float64))
    za = (wm.zone[wm.pa[w]] @ stand).tocsr()
    zb = (wm.zone[wm.pb[w]] @ stand).tocsr()
    for z in (za, zb):
        z.eliminate_zeros()
        z.data[:] = 1.0
    both = za.multiply(zb)
    du = s.u[c.ea[w]] - s.u[c.eb[w]]
    d = s.g[w] * du ** 2
    out = np.zeros(wm.n)
    for left, fwd in ((za - both, du > 0), (zb - both, du < 0)):
        coo = left.tocoo()
        keep = (coo.data > 0) & fwd[coo.row]
        out += np.bincount(coo.col[keep], weights=d[coo.row[keep]], minlength=wm.n)
    for tri, G, cell in ((wm.f_tri, st.Gf, c.fc), (wm.g_tri, st.Gg, c.gcell)):
        z = (wm.zone[tri] @ stand).tocoo()
        keep = z.data > 0
        out += np.bincount(z.col[keep], weights=(G * s.u[cell] ** 2)[z.row[keep]],
                           minlength=wm.n)
    return out


def block_run(bid: str, *, standing: clearings.Standing) -> None:
    out = ROWS / f"{bid}.parquet"
    if not standing.todo(out):
        return
    t_all = time.time()
    [b] = common.build_blocks([bid])
    mT = metrics.metric_of(metrics.RUNTIME, device="cpu").clearing(b, demand=POP)
    base = metrics.metric_of(metrics.UNIFORM, device="cpu").clearing(b, demand=POP)
    models = {k: gw.WallModel(base, walls=metrics.walls_on_uniform(part, device="cpu").walls,
                              start=gw.WarmStart()) for k, part in SETTINGS.items()}
    n = mT.n
    rows: dict[str, object] = dict(block=bid, home=np.arange(n), D=mT.cost, pop=mT.pop)
    rows["P0_T"] = mT.P0
    for k, wm in models.items():
        rows[f"P0_{k}"] = wm.base_.lenses.P
        rows[f"own_{k}"] = own_dissipation(wm, wm.base_)
    of0 = frontage(models["A"], models["A"].base_.state)
    dP = {k: np.zeros(n) for k in ("T", *SETTINGS)}
    passes = {k: np.zeros(n, dtype=np.int64) for k in SETTINGS}
    gained = np.zeros(n)
    frees = np.zeros(n, dtype=np.int64)
    freed_pop = np.zeros(n)
    t = {k: 0.0 for k in ("T", *SETTINGS)}
    for j in range(n):
        x = np.zeros(n, dtype=bool)
        x[j] = True
        t0 = time.time()
        dP["T"][j] = mT.P0 - mT.exact(x, 1.0)[1]
        left = mT.walled0 & ~mT.live_homes_of(mT.factors(mT.k_of(x)))
        freed = mT.walled0 & ~left
        frees[j], freed_pop[j] = int(freed.sum()), float(mT.pop[freed].sum())
        t["T"] += time.time() - t0
        for k, wm in models.items():
            t0 = time.time()
            st, s, P = wm.power(x)
            dP[k][j] = wm.base_.lenses.P - P
            passes[k][j] = s.passes
            if k == "A":
                of = frontage(wm, st)
                gained[j] = float(of.sum() - of0.sum() + of0[j])
            t[k] += time.time() - t0
        if j % 50 == 0:
            print(f"{bid}: {j + 1} of {n} ({time.time() - t_all:.0f}s)", flush=True)
    for k, v in dP.items():
        rows[f"dP_{k}"] = v
    for k, v in passes.items():
        rows[f"passes_{k}"] = v
    rows.update(gained=gained, frees=frees, freed_pop=freed_pop, walled=mT.walled0,
                host=socket.gethostname(), peak_rss_gb=gap_calib.peak_gb(),
                wall=time.time() - t_all, **{f"t_{k}": v for k, v in t.items()})
    standing.write(pd.DataFrame(rows), out)
    print(f"{bid}: {n} clearings, {time.time() - t_all:.0f}s ("
          + ", ".join(f"{k} {v:.0f}s" for k, v in t.items()) + ")", flush=True)


def passes_run(bid: str, k: int) -> None:
    """The diode B's passes per solve on `k` random single clearings, cold and warm."""
    [b] = common.build_blocks([bid])
    base = metrics.metric_of(metrics.UNIFORM, device="cpu").clearing(b, demand=POP)
    walls = metrics.walls_on_uniform(SETTINGS["B"], device="cpu").walls
    js = np.random.default_rng(SEED).permutation(base.n)[:k]
    out: dict[str, object] = dict(block=bid, home=js)
    for start in (gw.ColdStart(), gw.WarmStart()):
        wm = gw.WallModel(base, walls=walls, start=start)
        p, P, tt = [], [], []
        for j in js:
            x = np.zeros(base.n, dtype=bool)
            x[j] = True
            t0 = time.time()
            _st, s, Pj = wm.power(x)
            tt.append(time.time() - t0)
            p.append(s.passes)
            P.append(Pj)
        name = type(start).__name__
        out[f"passes_{name}"], out[f"P_{name}"], out[f"t_{name}"] = p, P, tt
        print(f"{bid} {name}: passes per solve median {np.median(p)}, max {max(p)}; "
              f"{np.mean(tt):.2f}s a clearing", flush=True)
    df = pd.DataFrame(out)
    rel = float((df.P_ColdStart / df.P_WarmStart - 1.0).abs().max())
    print(f"{bid}: P cold against warm, {rel:.1e} apart at most", flush=True)
    clearings.Replace().write(df, ROWS / f"passes_{bid}.parquet")


# ------------------------------------------------------------------ the report

def _top(x: NDArray[np.float64], k: int) -> set[int]:
    return set(np.argsort(-x, kind="stable")[:k].tolist())


def _jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def _pair_rows(name: str, g: pd.DataFrame, norm: bool) -> list[dict[str, object]]:
    out = []
    keys = ("T", *SETTINGS)
    gain = {k: (g[f"dP_{k}"] / g[f"P0_{k}"] if norm else g[f"dP_{k}"]).to_numpy() / g.D.to_numpy()
            for k in keys}
    n = len(g)
    k5 = max(1, int(round(TOP_SHARE * n)))
    for x, y in PAIRS:
        out.append(dict(group=name, pair=f"{x}-{y}", n=n,
                        spearman=float(stats.spearmanr(gain[x], gain[y]).statistic),
                        top10=_jaccard(_top(gain[x], TOP_K), _top(gain[y], TOP_K)),
                        top5pct=_jaccard(_top(gain[x], k5), _top(gain[y], k5)), top5pct_k=k5))
    return out


def report() -> None:
    df = pd.concat([pd.read_parquet(ROWS / f"{b}.parquet") for b in BLOCKS
                    if (ROWS / f"{b}.parquet").exists()], ignore_index=True)
    keys = ("T", *SETTINGS)
    pairs, shift, drivers, frees = [], [], [], []
    for name, g in df.groupby("block"):
        pairs += _pair_rows(str(name), g, False)
        dP = {k: g[f"dP_{k}"].to_numpy() for k in keys}
        for x in ("A", "C", "S1"):
            d = dP[x] - dP["B"]
            shift.append(dict(group=str(name), pair=f"{x}-B", ratio=float(d.std() / dP["B"].std()),
                              mean_shift=float(d.mean()), std_shift=float(d.std()),
                              std_B=float(dP["B"].std())))
            own = (g[f"own_{x}"] - g["own_B"]).to_numpy()
            drivers.append(dict(
                group=str(name), pair=f"{x}-B",
                vs_frontage=float(stats.spearmanr(d, g.gained).statistic),
                vs_own=float(stats.spearmanr(d, own).statistic),
                vs_own_pearson=float(np.corrcoef(d, own)[0, 1]),
                vs_pop=float(stats.spearmanr(d, g["pop"]).statistic),
                own_share=float(np.abs(own).sum() / np.abs(d).sum())))
        n = len(g)
        k5 = max(1, int(round(TOP_SHARE * n)))
        freeing = g.frees.to_numpy() > 0
        for k in keys:
            order = np.argsort(-(dP[k] / g.D.to_numpy()), kind="stable")
            rank = np.empty(n, dtype=np.int64)
            rank[order] = np.arange(1, n + 1)
            frees.append(dict(group=str(name), setting=k, n=n, freeing=int(freeing.sum()),
                              in_top10=int(freeing[order[:TOP_K]].sum()),
                              in_top5pct=int(freeing[order[:k5]].sum()), top5pct_k=k5,
                              best_rank=int(rank[freeing].min()) if freeing.any() else -1,
                              median_rank=float(np.median(rank[freeing])) if freeing.any()
                              else np.nan))
    pairs += _pair_rows("pooled", df, True)
    tables = {"pairs": pd.DataFrame(pairs), "shift": pd.DataFrame(shift),
              "drivers": pd.DataFrame(drivers), "frees": pd.DataFrame(frees)}
    # over the blocks: each per-block measure's median (and range)
    for name, key in (("pairs", "pair"), ("shift", "pair"), ("drivers", "pair")):
        t = tables[name]
        per = t[t.group != "pooled"]
        num = per.select_dtypes("number").columns
        med = per.groupby(key)[list(num)].median().reset_index().assign(group="median")
        lo = per.groupby(key)[list(num)].min().reset_index().assign(group="min")
        hi = per.groupby(key)[list(num)].max().reset_index().assign(group="max")
        tables[name] = pd.concat([t, med, lo, hi], ignore_index=True)
    f = tables["frees"]
    tot = f.groupby("setting")[["freeing", "in_top10", "in_top5pct", "top5pct_k"]].sum()
    tables["frees"] = pd.concat([f, tot.reset_index().assign(group="sum")], ignore_index=True)
    pd.set_option("display.width", 200)
    for name, t in tables.items():
        t.to_csv(ROWS / f"summary_{name}.csv", index=False)
        print(t.round(3).to_string(), flush=True)


def main(argv: list[str]) -> None:
    argv, standing = clearings.standing_of(argv)
    if len(argv) == 2 and argv[0] == "block":
        block_run(argv[1], standing=standing)
    elif len(argv) == 3 and argv[0] == "passes":
        passes_run(argv[1], int(argv[2]))
    elif argv == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
