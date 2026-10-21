"""Walls that leak: are symmetric walls worse, or just different? (owner 2026-10-21: "Symmetric
walls might not be equivalent, but are they *worse*? Or just different? ... we would hopefully be
able to default to the symmetric case if it's not worse"; NOTES, "Walls that leak", after the
single-clearing test.) Two analyses on leaky_single.py's blocks and settings.

    S=research/roadless/leaky_split.py      # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S decompose <id> <setting>   # ROWS/decomp/<id>__<setting>.parquet
    uv run python $S regret <id>                # ROWS/regret/<id>.parquet
    uv run python $S report                     # ROWS/summary_{split,split_pairs,regret}.csv

`decompose`: each building j cleared alone at baseline under one setting (leaky_single.SETTINGS'
A, B, C, S1, or T: RUNTIME), its gain dP_j = P0 - P(j cleared) split exactly into self_j, its own
residents' term pop_j (u_j,0 - u_j,j) (the metric keeps a cleared home's demand: its residents'
escape time falls to what the open ground they are left on gives), and others_j, everyone else's;
and pop_j u_j,0, their whole share of P0 (what removing them would take). T's walled-in homes have
no demand, so no self term.

`regret`: per setting X a clearing built greedily from its single-clearing gains (leaky_single's
rows: largest dP_j / D_j first while the area D stays within BUDGET; a crude stand-in for SIMP,
blind to interactions), each scored under every setting's P and T's (RUNTIME's) Lens B count (the
population left walled in, by street reach); and, per setting, the clearing's joint gain against
the sum of its buildings' single gains (the interactions the greedy ignores).

`report`: (1) per setting, the share of the gain that is self over the top TOP_K and the top
TOP_SHARE by dP_j / D_j (median over the blocks), and Spearman's correlation of others_j / D_j
between settings; (2) the regret matrix: for each scoring metric Y and clearing X, Y's shortfall
(G_Y(c_Y) - G_Y(c_X)) / G_Y(c_Y), median over the blocks (min, max), the people each clearing
leaves walled in, each greedy's interaction ratio, and each setting's scoring time per single
clearing (`decompose`'s wall over its clearings).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/leaky_split.py
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_walls as gw  # noqa: E402
import leaky_single as ls  # noqa: E402
import metrics  # noqa: E402

ROWS = ls.ROWS
BUDGET = 0.05            # the greedy's area D (the lenses' D)
TOP_K = ls.TOP_K
TOP_SHARE = ls.TOP_SHARE
# symmetric prices between S1's and A's (the controller 2026-10-21: does one keep S1's regret and
# free like the diodes?), their gains from `decompose` alone (leaky_single never ran them)
EXTRA = {"S5": "Wk1l500", "S10": "Wk1l1000", "S15": "Wk1l1500"}
PARTS = {**ls.SETTINGS, **EXTRA}
KEYS = ("T", *PARTS)
POP = common.PopulationDemand()


class _TScorer:
    """T's (RUNTIME's) P and per-home escape times at a clearing."""

    def __init__(self, m: gcl.Model):
        self.m = m

    def scored(self, x: NDArray[np.bool_]) -> tuple[float, NDArray[np.float64], int]:
        m = self.m
        st = m.state(m.k_of(x))
        u = m.solve(st, m.inj)
        return float(m._rhs(st, m.inj) @ u), m.home_u(st, u), 1


class _WScorer:
    """A W setting's P and per-home escape times at a clearing (its diode loops warm)."""

    def __init__(self, wm: gw.WallModel):
        self.wm = wm

    def scored(self, x: NDArray[np.bool_]) -> tuple[float, NDArray[np.float64], int]:
        _st, s, P = self.wm.power(x)
        return P, self.wm.home_u(s.u), s.passes


def _scorers(bid: str, keys: tuple[str, ...]) -> tuple[dict[str, _TScorer | _WScorer], gcl.Model]:
    """Each setting of `keys`' scorer on `bid`, and T's model (its B (i) and costs)."""
    [b] = common.build_blocks([bid])
    mT = metrics.metric_of(metrics.RUNTIME, device="cpu").clearing(b, demand=POP)
    out: dict[str, _TScorer | _WScorer] = {}
    base = None
    for k in keys:
        if k == "T":
            out[k] = _TScorer(mT)
            continue
        if base is None:
            base = metrics.metric_of(metrics.UNIFORM, device="cpu").clearing(b, demand=POP)
        out[k] = _WScorer(gw.WallModel(
            base, walls=metrics.walls_on_uniform(PARTS[k], device="cpu").walls,
            start=gw.WarmStart()))
    return out, mT


def decompose(bid: str, key: str, *, replace: bool) -> None:
    out = ROWS / "decomp" / f"{bid}__{key}.parquet"
    clearings.check_new(out, replace=replace)
    t0 = time.time()
    sc, mT = _scorers(bid, (key,))
    s = sc[key]
    n = mT.n
    P0, u0, _p = s.scored(np.zeros(n, dtype=bool))
    pop = mT.pop
    dP, self_, passes = np.zeros(n), np.zeros(n), np.zeros(n, dtype=np.int64)
    for j in range(n):
        x = np.zeros(n, dtype=bool)
        x[j] = True
        P, u, passes[j] = s.scored(x)
        dP[j] = P0 - P
        self_[j] = pop[j] * (u0[j] - u[j]) if np.isfinite(u0[j]) else 0.0
    share0 = np.where(np.isfinite(u0), pop * np.nan_to_num(u0), 0.0)
    single = pd.read_parquet(ROWS / f"{bid}.parquet")
    apart = np.nan                  # against leaky_single's gains, where it ran the setting
    if f"dP_{key}" in single.columns:
        stored = single[f"dP_{key}"].to_numpy()
        apart = float(np.abs(dP - stored).max() / max(np.abs(stored).max(), 1e-300))
    clearings.write_once(pd.DataFrame(dict(
        block=bid, setting=key, home=np.arange(n), D=mT.cost, pop=pop, P0=P0, dP=dP, self=self_,
        others=dP - self_, share0=share0, passes=passes, stored_apart=apart,
        host=socket.gethostname(), wall=time.time() - t0)), out, replace=replace)
    print(f"{bid} {key}: {n} clearings decomposed, dP against leaky_single's {apart:.1e} apart, "
          f"{time.time() - t0:.0f}s", flush=True)


def _greedy(gain: NDArray[np.float64], D: NDArray[np.float64]) -> NDArray[np.bool_]:
    """The buildings by gain per cost, largest first, while their D stays within BUDGET."""
    order = np.argsort(-(gain / D), kind="stable")
    x = np.zeros(len(D), dtype=bool)
    used = 0.0
    for j in order:
        if gain[j] <= 0:
            break
        if used + D[j] <= BUDGET:
            x[j] = True
            used += D[j]
    return x


def _gains(bid: str, key: str, single: pd.DataFrame) -> NDArray[np.float64] | None:
    """`key`'s single-clearing gains on `bid`: leaky_single's, or `decompose`'s (None: neither
    ran it)."""
    if f"dP_{key}" in single.columns:
        return np.asarray(single[f"dP_{key}"].to_numpy(), dtype=np.float64)
    path = ROWS / "decomp" / f"{bid}__{key}.parquet"
    if path.exists():
        return np.asarray(pd.read_parquet(path).dP.to_numpy(), dtype=np.float64)
    return None


def _known(g: NDArray[np.float64] | None) -> NDArray[np.float64]:
    if g is None:
        raise ValueError("a setting with no gains")
    return g


def regret(bid: str, *, replace: bool) -> None:
    out = ROWS / "regret" / f"{bid}.parquet"
    clearings.check_new(out, replace=replace)
    t0 = time.time()
    single = pd.read_parquet(ROWS / f"{bid}.parquet")
    D = single.D.to_numpy()
    gains = {k: _gains(bid, k, single) for k in KEYS}
    keys = tuple(k for k in KEYS if gains[k] is not None)
    clear = {k: _greedy(_known(gains[k]), D) for k in keys}
    sc, mT = _scorers(bid, keys)
    rows = []
    n = mT.n
    for y, s in sc.items():
        P0, _u, _p = s.scored(np.zeros(n, dtype=bool))
        for x, c in clear.items():
            P, _u2, passes = s.scored(c)
            rows.append(dict(block=bid, clearing=x, scorer=y, P0=P0, P=P, gain=P0 - P,
                             passes=passes, n_cleared=int(c.sum()), D_used=float(D[c].sum()),
                             single_sum=float(_known(gains[y])[c].sum())))
    for x, c in clear.items():
        left = mT.walled0 & ~mT.live_homes_of(mT.factors(mT.k_of(c)))
        rows.append(dict(block=bid, clearing=x, scorer="B(i)", n_left=int(left.sum()),
                         pop_left=float(mT.cost[left].sum()), n_walled0=int(mT.walled0.sum()),
                         n_cleared=int(c.sum()), D_used=float(D[c].sum())))
    clearings.write_once(pd.DataFrame(rows), out, replace=replace)
    print(f"{bid}: regret matrix, {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ the report

def _read(kind: str) -> pd.DataFrame:
    paths = sorted((ROWS / kind).glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def report() -> None:
    dec = _read("decomp")
    split, pairs = [], []
    for bid, key, g in ((b_, k_, g_) for b_, gb in dec.groupby("block")
                        for k_, g_ in gb.groupby("setting")):
        gain = (g.dP / g.D).to_numpy()
        n = len(g)
        order = np.argsort(-gain, kind="stable")
        row: dict[str, object] = dict(block=bid, setting=key, n=n,
                                      apart=float(g.stored_apart.iloc[0]))
        for label, k in (("top10", TOP_K), ("top5pct", max(1, int(round(TOP_SHARE * n)))),
                         ("all", n)):
            top = g.iloc[order[:k]]
            row[f"self_{label}"] = float(top["self"].sum() / top.dP.sum())
            row[f"share0_{label}"] = float(top.share0.sum() / top.dP.sum())
        row["passes_median"] = float(g.passes.median())
        split.append(row)
    for bid, g in dec.groupby("block"):
        wide = g.pivot(index="home", columns="setting", values="others")
        D = g.groupby("home").D.first()
        have = [k for k in KEYS if k in wide.columns]
        for i, x in enumerate(have):
            for y in have[i + 1:]:
                pairs.append(dict(block=bid, pair=f"{x}-{y}", others=float(stats.spearmanr(
                    wide[x] / D, wide[y] / D).statistic)))
    reg = _read("regret")
    mat = []
    pw = reg[reg.scorer != "B(i)"]
    for rb, ry, rg in ((b_, y_, g_) for b_, gb in pw.groupby("block")
                       for y_, g_ in gb.groupby("scorer")):
        own = float(rg[rg.clearing == ry].gain.iloc[0])
        for _, r in rg.iterrows():
            mat.append(dict(block=rb, scorer=ry, clearing=r.clearing,
                            shortfall=(own - r.gain) / own,
                            interaction=(r.gain / r.single_sum) if r.clearing == ry else np.nan))
    left = reg[reg.scorer == "B(i)"][["block", "clearing", "n_left", "pop_left", "n_walled0"]]
    cost = dec.groupby(["block", "setting"]).agg(n=("home", "size"), wall=("wall", "first"),
                                                 passes=("passes", "median")).reset_index()
    cost["s_per_clearing"] = cost.wall / cost.n
    tables = {"split": pd.DataFrame(split), "split_pairs": pd.DataFrame(pairs),
              "regret": pd.DataFrame(mat), "left": left, "cost": cost}
    pd.set_option("display.width", 220)
    for name, t in tables.items():
        t.to_csv(ROWS / f"summary_{name}.csv", index=False)
    s = tables["split"]
    print(s.groupby("setting")[[c for c in s.columns if c.startswith(("self", "share0"))]]
          .median().round(3).to_string())
    p = tables["split_pairs"]
    print(p.groupby("pair").others.agg(["median", "min", "max"]).round(3).to_string())
    m = tables["regret"]
    print(m.pivot_table(index="clearing", columns="scorer", values="shortfall",
                        aggfunc="median").round(3).to_string())
    print(m.pivot_table(index="clearing", columns="scorer", values="shortfall",
                        aggfunc="max").round(3).to_string())
    print(m.dropna(subset=["interaction"]).pivot_table(index="block", columns="scorer",
                                                       values="interaction").round(3).to_string())
    print(left.pivot_table(index="block", columns="clearing", values="n_left").to_string())
    print(cost.pivot_table(index="block", columns="setting", values="s_per_clearing")
          .round(2).to_string())


def main(argv: list[str]) -> None:
    argv, replace = clearings.replace_flag(argv)
    if len(argv) == 3 and argv[0] == "decompose" and argv[2] in KEYS:
        decompose(argv[1], argv[2], replace=replace)
    elif len(argv) == 2 and argv[0] == "regret":
        regret(argv[1], replace=replace)
    elif argv == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
