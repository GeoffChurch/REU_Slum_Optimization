"""Walled-in homes, round 2: T, U and W, each with its best optimizer (owner 2026-10-24: "try
everything" that might fix the uniform rule's two failures, and whether its optimizer could be
less myopic; NOTES, the section of that name; the brief .superpowers/sdd/2026-10-24-walled-in).

The arms (ARMS), each an optimizer of P at area D BUDGET under one rule's metric (RULES):
T, today's (RUNTIME: free first, held); U, the uniform rule (metrics.UNIFORM, `.f0`); F, U with
every home's demand (metrics.HOMES: F1, gap_clear.HomeDemand); and F's less myopic optimizers,
each a named part of its plan (relax.Plan): Fa free first's crossings as the polish's moves
(`jF8`), Fb path moves (`jV4s32`, gap_joins), Fc walls translucent first (`.e0.05-0.0001`), Fd
Frank-Wolfe's start and bound (`fw40`), Fe SIMP's roundings compared exactly (`k1s1e-06E`: the
eps world at 1e-6 misreads a sealed pocket), Fx their best combination and Tx T with the best of
(b) or (c) (both named once the first wave is in); W, walls that leak (S10, its fit
optimizer GxLr10: leaky_simp's S10x, on a core), on the small blocks. SIMP's default otherwise,
its polish scored exactly on every device (`E`: the eps world at 1e-6 misreads a sealed pocket,
whose leak is of the order of its walls' eps conductance).

    S=research/roadless/walled.py      # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S arm <id> <label> <cpu|gpu> [--replace|--keep-standing]
    uv run python $S block <id> <label,...> <cpu|gpu> [--replace|--keep-standing]
    uv run python $S lenses <id> <cpu|gpu> [--replace|--keep-standing]
    uv run python $S chains <id> <cpu|gpu> [--replace|--keep-standing]
    uv run python $S wlenses <id> [--replace|--keep-standing]
    uv run python $S report

`arm`: the arm's run (relax.one, or leaky_simp's W arm), its row relax's (or leaky_simp's) with the
device, the host, the wall time and the peak memory (relax_rows/<metric>/<plan>/D0.05/).

`block`: the arms in turn, then `lenses` and `chains` (a small block's task).

`lenses`: every arm's clearing that stands (and none) scored under T's, U's and F's metrics:
Lens A of P, P_{1/2} and J_2 on each metric's own baseline and
B (i) (the same under every metric: checked) (ROWS/lenses/<id>.parquet); per home walled in at
baseline, whether each clearing leaves it walled in and its escape time under F's metric after it
against the block's median home at baseline (ROWS/homes/<id>.parquet); per clearing the D its
freeing costs: the cleared buildings whose return alone would wall in again a home it frees, their
D against the population freed (the owner's fairness concern; T's free stage's own D beside it)
(ROWS/fair/<id>.parquet).

`wlenses`: every arm's clearing under W's metric, on a core, the small blocks
(ROWS/wlenses/<id>.parquet).

`chains`: does a path move find free first's crossings unprompted? At baseline under F, free
first's crossings within the budget against the chains (gap_joins.chains, Fb's sources and width,
and CHAIN_WIDE more): per crossing whether a chain is its path, and whether a chain alone frees its
home (B (i)) (ROWS/chains/<id>.parquet).

`report`: per block (summary_blocks.csv), per walled-in home (summary_homes.csv), per pair of
arms under each metric (summary_pairs.csv: each lens's paired difference, mean, median, 95%
Bayesian-bootstrap interval, the blocks ahead and behind), by size (summary_sizes.csv: the people
each arm leaves walled in, D per person freed, the blocks where an arm ends below T's clearing by
its own rule's measure and by how much, each F arm's distance from Fd's bound, compute).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/walled.py
"""
from __future__ import annotations

import gc as pygc
import socket
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_calib  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_joins  # noqa: E402
import leaky_simp  # noqa: E402
import metrics  # noqa: E402
import relax  # noqa: E402
import road_rescore as rr  # noqa: E402

ROWS = HERE / "walled_rows"
BLOCKS = ROWS / "blocks.csv"        # block, size, control (the uniform study's 121 and 5 controls)
BUDGET = relax.D_LENS                # area D 0.05
POWER = 1.0                          # P
POP = common.PopulationDemand()
LENSES = ("L_P", "L_Pq", "L_J2")
BB_DRAWS = 4000                      # the Bayesian bootstrap's draws (consolidated._bb_interval's)
BB_SEED = 20261024
CHAIN_WIDE = 32                      # `chains`: the chains looked at beyond Fb's width

RULES = {"T": metrics.RUNTIME, "U": metrics.UNIFORM, "F": metrics.HOMES}


class Spec(NamedTuple):
    """An arm: its rule (RULES) and its plan (relax.plan_of's spec)."""
    rule: str
    plan: str


_E = "fw0.q3.i10.e0.0001.k1s1e-06.p256w8x2"       # SIMP's default (relax.SIMP_DEFAULT) ...
ARMS = {
    "T": Spec("T", f"{_E}E"),
    "U": Spec("U", f"{_E}E.f0"),
    "F": Spec("F", f"{_E}E.f0"),
    "Fa": Spec("F", f"{_E}jF8E.f0"),
    "Fb": Spec("F", f"{_E}jV4s32E.f0"),
    "Fc": Spec("F", "fw0.q3.i10.e0.05-0.0001.k1s1e-06.p256w8x2E.f0"),
    "Fd": Spec("F", "fw40.q3.i10.e0.0001.k1s1e-06.p256w8x2E.f0"),
    "Fe": Spec("F", "fw0.q3.i10.e0.0001.k1s1e-06E.p256w8x2E.f0"),
}
W_ARM = "S10x"                       # leaky_simp's: S10 (Wk1l1000) under GxLr10, on a core
FD = "Fd"                            # the arm whose row carries Frank-Wolfe's bound


class Arm(NamedTuple):
    label: str
    rule: str
    metric: gcl.ComplexMetric
    plan: relax.Plan


def arm_of(label: str, device: str) -> Arm:
    spec = ARMS[label]
    plan = relax.plan_of(spec.plan)
    return Arm(label, spec.rule, metrics.metric_of(RULES[spec.rule], device=device), plan)


def arm_path(label: str, bid: str) -> Path:
    """The arm's row (relax.rows_of's directory, or leaky_simp's for W); the metric's name is the
    device's alike (its factorization is in no name)."""
    if label == "W":
        return leaky_simp.arm_path(leaky_simp.w_arm(W_ARM), bid)
    a = arm_of(label, "cpu")
    return relax.rows_of(a.plan, a.metric.name, BUDGET) / f"{bid}_p{POWER:g}.parquet"


def run_arm(label: str, bid: str, device: str, *, standing: clearings.Standing) -> None:
    """The arm's run on `bid`, its row with the device, host, wall time and peak memory."""
    if label == "W":
        if device != "cpu":
            raise ValueError("W's arm runs on a core (leaky_simp's)")
        leaky_simp.run_arm(leaky_simp.w_arm(W_ARM), bid, standing=standing)
        return
    path = arm_path(label, bid)
    if not standing.todo(path):
        return
    arm = arm_of(label, device)
    pygc.collect()
    t0 = time.time()
    if device == "gpu":
        with gap_calib.GpuPeak() as peak:
            row = relax.one(bid, POWER, arm.metric, arm.plan, BUDGET, demand=POP)
        gpu_gb = peak.peak
    else:
        row, gpu_gb = relax.one(bid, POWER, arm.metric, arm.plan, BUDGET, demand=POP), np.nan
    row.update(arm=label, device=device, host=socket.gethostname(),
               peak_rss_gb=gap_calib.peak_gb(), gpu_gb=gpu_gb, wall=time.time() - t0)
    standing.write(pd.DataFrame([row]), path)


def _row(label: str, bid: str) -> pd.Series:
    return pd.read_parquet(arm_path(label, bid)).iloc[0]


def cleared(label: str, bid: str, n: int) -> NDArray[np.bool_]:
    r = _row(label, bid)
    if int(r["n"]) != n:
        raise RuntimeError(f"{bid} {label}: a clearing of {r['n']} buildings, the block has {n}")
    x = np.zeros(n, dtype=bool)
    x[np.asarray(r["cleared" if label == "W" else "simp_cleared"], dtype=np.int64)] = True
    return x


def outs(bid: str) -> tuple[Path, Path, Path]:
    return (ROWS / "lenses" / f"{bid}.parquet", ROWS / "homes" / f"{bid}.parquet",
            ROWS / "fair" / f"{bid}.parquet")


def sizes() -> pd.Series:
    return pd.read_csv(BLOCKS).set_index("block")["size"]


def labels_of(bid: str) -> list[str]:
    """The arms whose rows stand on `bid` (W's on the small blocks)."""
    out = [label for label in ARMS if arm_path(label, bid).exists()]
    return out + (["W"] if arm_path("W", bid).exists() else [])


def wlenses(bid: str, *, standing: clearings.Standing) -> None:
    """Every standing arm's clearing of `bid` (W's among them) under W's metric, on a core
    (leaky_simp's view: Lens A on W's own baseline; its B (i), its base's, checked against the
    walled-in homes `lenses` left under T's) (ROWS/wlenses/<id>.parquet)."""
    out = ROWS / "wlenses" / f"{bid}.parquet"
    if not standing.todo(out):
        return
    t0 = time.time()
    labels = labels_of(bid)
    n = int(_row(labels[0], bid)["n"])
    xs = {"none": np.zeros(n, dtype=bool), **{lb: cleared(lb, bid, n) for lb in labels}}
    w = leaky_simp._w_view(bid, "W", leaky_simp.w_arm(W_ARM).metric, xs)
    homes = pd.read_parquet(outs(bid)[1])
    for lb in xs:
        if f"left_{lb}" in homes and not np.array_equal(
                w.left[lb][homes.home.to_numpy()], homes[f"left_{lb}"].to_numpy()):
            raise RuntimeError(f"{bid}: W's B (i) and T's part on {lb!r}")
    standing.write(pd.DataFrame(w.rows), out)
    print(f"{bid}: W's lenses on {len(xs)} clearings; {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ every clearing scored

def _critical(m: gcl.Model, x: NDArray[np.bool_]) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """(the walled-in homes clearing `x` frees, by B (i); the cleared buildings whose return alone
    walls one of them in again). Found by halving: returning a group that walls no freed home in
    again holds none of them (B (i)'s reach only grows as buildings clear), so a group that does
    is split until each part is one building or walls no one in."""
    live = m.live_homes_of(m.factors(m.k_of(x)))
    freed = m.walled0 & live
    crit = np.zeros(m.n, dtype=bool)

    def rewalls(group: NDArray[np.int64]) -> bool:
        y = x.copy()
        y[group] = False
        return bool((freed & ~m.live_homes_of(m.factors(m.k_of(y)))).any())
    todo = [np.flatnonzero(x)] if freed.any() else []
    while todo:
        g = todo.pop()
        if not rewalls(g):
            continue
        if len(g) == 1:
            crit[g[0]] = True
        else:
            todo += [g[:len(g) // 2], g[len(g) // 2:]]
    return freed, crit


class Scored(NamedTuple):
    """One metric's view of a block's clearings (`score`): its lens rows; per clearing the homes'
    escape times and the homes it leaves walled in; the baseline's walled-in and groundless homes,
    the escape times, the costs; under T's metric (B (i) is every metric's) each clearing's freeing
    D (_critical)."""
    rows: list[dict[str, Any]]
    u: dict[str, NDArray[np.float64]]
    left: dict[str, NDArray[np.bool_]]
    walled0: NDArray[np.bool_]
    groundless: NDArray[np.bool_]
    u0: NDArray[np.float64]
    cost: NDArray[np.float64]
    fair: list[dict[str, Any]]


def score(bid: str, rule: str, metric: gcl.ComplexMetric, xs: dict[str, NDArray[np.bool_]], *,
          fair: bool) -> Scored:
    """Every clearing of `bid` under `metric` (its model built here and dropped on return: one
    model in memory at a time), with each clearing's freeing D if `fair`."""
    pygc.collect()
    [b] = common.build_blocks([bid])
    m = metric.clearing(b, demand=POP)
    rows, u, left, fr = [], {}, {}, []
    for label, x in xs.items():
        L = m.base if not x.any() else m.lenses(x)
        rows.append(dict(block=bid, scorer=rule, clearing=label, n_cleared=int(x.sum()),
                         D=float(m.cost[x].sum()), **rr._lens_row(m, L)))
        u[label], left[label] = L.u, m.walled0 & ~L.live
        if fair:
            freed, crit = _critical(m, x)
            fr.append(dict(block=bid, clearing=label, freed_n=int(freed.sum()),
                           freed_share=float(m.cost[freed].sum()), crit_n=int(crit.sum()),
                           crit_D=float(m.cost[crit].sum())))
    return Scored(rows=rows, u=u, left=left, walled0=m.walled0, groundless=m.fx.groundless,
                  u0=m.u_pop0, cost=m.cost, fair=fr)


def lenses(bid: str, device: str, *, standing: clearings.Standing) -> None:
    """Every standing arm's clearing of `bid` under every metric (module docstring)."""
    paths = outs(bid)
    if not any([standing.todo(o) for o in paths]):
        return
    t0 = time.time()
    labels = labels_of(bid)
    n = int(_row(labels[0], bid)["n"])
    xs = {"none": np.zeros(n, dtype=bool), **{lb: cleared(lb, bid, n) for lb in labels}}
    sc = {rule: score(bid, rule, metrics.metric_of(name, device=device), xs, fair=rule == "T")
          for rule, name in RULES.items()}
    rows = [r for s in sc.values() for r in s.rows]
    left = {rule: s.left for rule, s in sc.items()}
    for rule, lf in left.items():
        for label in xs:
            if not np.array_equal(lf[label], left["T"][label]):
                raise RuntimeError(f"{bid}: {rule}'s B (i) and T's part on {label!r}")
    F = sc["F"]
    w0 = sc["T"].walled0
    med = float(np.nanmedian(F.u0))
    homes = pd.DataFrame(dict(block=bid, home=np.flatnonzero(w0), share=F.cost[w0],
                              groundless=F.groundless[w0], u0_F=F.u0[w0] / med,
                              **{f"left_{lb}": left["T"][lb][w0] for lb in xs},
                              **{f"uF_{lb}": F.u[lb][w0] / med for lb in xs}))
    for o, df in zip(paths, (pd.DataFrame(rows), homes, pd.DataFrame(sc["T"].fair)), strict=True):
        standing.write(df, o)
    print(f"{bid}: {int(w0.sum())} walled in; left "
          + ", ".join(f"{lb} {int(left['T'][lb][w0].sum())}" for lb in xs)
          + f"; lenses {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ do path moves find crossings

def chain_check(bid: str, device: str, *, standing: clearings.Standing) -> None:
    out = ROWS / "chains" / f"{bid}.parquet"
    if not standing.todo(out):
        return
    t0 = time.time()
    [b] = common.build_blocks([bid])
    m = metrics.metric_of(metrics.HOMES, device=device).clearing(b, demand=POP)
    fb = relax.plan_of(ARMS["Fb"].plan).joins
    if len(fb) != 1 or not isinstance(fb[0], relax.Chains):
        raise ValueError(f"Fb's plan names one chain source: {ARMS['Fb'].plan}")
    src, width = fb[0].sources, fb[0].width
    none = np.zeros(m.n, dtype=bool)
    crossings = m.freed(budget=BUDGET, held=none).crossings
    ch = gap_joins.chains(m, none, sources=src, width=width + CHAIN_WIDE)
    frees = []
    for c in ch:
        x = none.copy()
        x[list(c.buildings)] = True
        frees.append(m.walled0 & m.live_homes_of(m.factors(m.k_of(x))))
    rows: list[dict[str, Any]] = []
    same_k, frees_k = [], []
    for cr in crossings:
        path = set(cr.path)
        same = [i for i, c in enumerate(ch) if set(c.buildings) == path]
        freeing = [i for i, f in enumerate(frees) if f[cr.home]]
        same_k.append(same[0] if same else -1)
        frees_k.append(freeing[0] if freeing else -1)
        rows.append(dict(block=bid, home=cr.home, n_path=len(cr.path), D=cr.D,
                         same_rank=same_k[-1], frees_rank=frees_k[-1], width=width,
                         n_chains=len(ch), t=time.time() - t0))
    df = pd.DataFrame(rows) if rows else pd.DataFrame(dict(block=[bid], home=[-1]))
    standing.write(df, out)
    print(f"{bid}: {len(crossings)} crossings; a chain is one in the top {width}: "
          f"{sum(0 <= k < width for k in same_k)}, frees its home: "
          f"{sum(0 <= k < width for k in frees_k)}; {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ the report

def _read(kind: str) -> pd.DataFrame:
    paths = sorted((ROWS / kind).glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else \
        pd.DataFrame()


def _bb(d: NDArray[np.float64], rng: np.random.Generator) -> tuple[float, float]:
    """The 95% interval of the mean of `d` under the Bayesian bootstrap."""
    if len(d) == 0:
        return np.nan, np.nan
    wts = rng.dirichlet(np.ones(len(d)), size=BB_DRAWS)
    lo, hi = np.quantile(wts @ d, [0.025, 0.975])
    return float(lo), float(hi)


def report() -> None:
    sz = pd.read_csv(BLOCKS).set_index("block")
    lens = pd.concat([_read("lenses"), _read("wlenses")], ignore_index=True)
    homes = _read("homes")
    fair = _read("fair")
    blocks = sorted(set(lens.block))
    arms = [lb for lb in [*ARMS, "W"] if lb in set(lens.clearing)]
    wide = pd.DataFrame(index=pd.Index(blocks, name="block"))
    wide["size"] = sz["size"].reindex(wide.index)
    wide["control"] = sz["control"].reindex(wide.index)
    for scorer in sorted(set(lens.scorer)):
        for lb in ["none", *arms]:
            sub = lens[(lens.scorer == scorer) & (lens.clearing == lb)].set_index("block")
            for x in (*LENSES, "n_enclosed", "enclosed"):
                wide[f"{x}_{scorer}_{lb}"] = sub[x].reindex(wide.index)
    for lb in arms:
        rows = []
        for bid in blocks:
            p = arm_path(lb, bid)
            rows.append(pd.read_parquet(p).iloc[0] if p.exists() else pd.Series(dtype=object))
        r = pd.DataFrame(rows, index=wide.index)
        for col in ("wall", "t", "peak_rss_gb", "gpu_gb", "device", "simp_grey", "bound_perm",
                    "free_D", "D", "simp_D"):
            if col in r:
                wide[f"{col}_{lb}"] = r[col]
        if lb == "T":
            wide["free_n_T"] = r["free_cleared"].map(lambda v: len(v) if v is not None else 0)
    for lb in arms:
        sub = fair[fair.clearing == lb].set_index("block")
        for col in ("freed_n", "freed_share", "crit_n", "crit_D"):
            wide[f"{col}_{lb}"] = sub[col].reindex(wide.index)
    hw = homes.groupby("block").agg(n_walled=("home", "size"), n_groundless=("groundless", "sum"))
    wide = wide.join(hw)
    wide.reset_index().to_csv(ROWS / "summary_blocks.csv", index=False)
    homes.merge(sz.reset_index(), on="block").to_csv(ROWS / "summary_homes.csv", index=False)
    rng = np.random.default_rng(BB_SEED)
    groups = [(s, wide[(~wide.control) & (wide["size"] == s)])
              for s in ("small", "large", "oversized")]
    groups += [("all walled", wide[~wide.control]), ("control", wide[wide.control])]
    pairs = [(a, "T") for a in arms if a != "T"] + [(a, "U") for a in arms if a[0] == "F"]
    pairs += [(a, "F") for a in arms if a[0] == "F" and a != "F"]
    prow = []
    for name, g in groups:
        if g.empty:
            continue
        for a, b_ in pairs:
            for scorer in sorted(set(lens.scorer)):
                for x in LENSES:
                    ca, cb = f"{x}_{scorer}_{a}", f"{x}_{scorer}_{b_}"
                    if ca not in g or cb not in g:
                        continue
                    d = (g[ca] - g[cb]).dropna().to_numpy(dtype=float)
                    if len(d) == 0:
                        continue
                    lo, hi = _bb(d, rng)
                    prow.append(dict(group=name, a=a, b=b_, scorer=scorer, lens=x, blocks=len(d),
                                     mean=float(d.mean()), median=float(np.median(d)), lo=lo,
                                     hi=hi, ahead=int((d > 1e-9).sum()),
                                     behind=int((d < -1e-9).sum())))
    pd.DataFrame(prow).to_csv(ROWS / "summary_pairs.csv", index=False)
    srow = []
    for name, g in groups:
        if g.empty:
            continue
        hm = homes[homes.block.isin(g.index)]
        row: dict[str, Any] = dict(group=name, blocks=len(g), homes_walled=len(hm),
                                   groundless=int(hm.groundless.sum()))
        for lb in arms:
            lcol = f"left_{lb}"
            if lcol not in hm:
                continue
            on = hm[lcol].notna()
            left = hm[on & hm[lcol].astype(bool)]
            row[f"left_{lb}"] = len(left)
            row[f"left_blocks_{lb}"] = int(left.block.nunique())
            row[f"left_groundless_{lb}"] = int(left.groundless.sum())
            row[f"left_share_{lb}"] = float(left.share.sum())
            row[f"freed_far_{lb}"] = int((~hm[lcol].astype(bool) & on
                                          & (hm[f"uF_{lb}"] > 10.0)).sum())
            fs, cd = g[f"freed_share_{lb}"], g[f"crit_D_{lb}"]
            ratio = (cd / fs)[fs > 0]
            row[f"crit_D_per_freed_median_{lb}"] = float(ratio.median())
            row[f"crit_D_per_freed_pooled_{lb}"] = float(cd.sum() / fs.sum()) if fs.sum() else \
                np.nan
            row[f"crit_D_per_freed_over1_{lb}"] = int((ratio > 1.0).sum())
            rule = "W" if lb == "W" else ARMS[lb].rule
            own = f"L_P_{rule}_{lb}"
            vs = f"L_P_{rule}_T"
            if own in g and vs in g:
                d = (g[own] - g[vs]).dropna()
                row[f"below_T_own_{lb}"] = int((d < -1e-9).sum())
                row[f"below_T_own_mean_{lb}"] = float(d[d < -1e-9].mean()) if (d < -1e-9).any() \
                    else 0.0
            if "F" in str(rule) and f"bound_perm_{FD}" in g:
                gap = (g[f"bound_perm_{FD}"] - g[f"L_P_F_{lb}"]).dropna()
                row[f"bound_gap_median_{lb}"] = float(gap.median())
                row[f"bound_gap_mean_{lb}"] = float(gap.mean())
            if f"wall_{lb}" in g:
                row[f"wall_h_{lb}"] = float(g[f"wall_{lb}"].sum() / 3600.0)
                row[f"wall_median_s_{lb}"] = float(g[f"wall_{lb}"].median())
        if "free_D_T" in g:
            fs = g["freed_share_T"]
            row["free_D_T_mean"] = float(g["free_D_T"].mean())
        srow.append(row)
    t = pd.DataFrame(srow)
    t.to_csv(ROWS / "summary_sizes.csv", index=False)
    # which rule leaves whom: per pair of arms the walled-in homes one leaves and the other frees
    wrow = []
    on = homes[homes.block.isin(wide.index[~wide.control])]
    for a in arms:
        for b_ in arms:
            ca, cb = f"left_{a}", f"left_{b_}"
            if a == b_ or ca not in on or cb not in on:
                continue
            both = on[ca].notna() & on[cb].notna()
            la, lb = on[ca].astype(bool) & both, on[cb].astype(bool) & both
            wrow.append(dict(a=a, b=b_, homes=int(both.sum()),
                             a_leaves_b_frees=int((la & ~lb).sum()),
                             of_them_groundless=int((la & ~lb & on.groundless).sum()),
                             blocks=int(on[la & ~lb].block.nunique())))
    pd.DataFrame(wrow).to_csv(ROWS / "summary_whom.csv", index=False)
    # the uniform study's 32: U's optimizer below T's clearing by U's own measure there
    old = pd.read_csv(HERE / "uniform_rows" / "summary_blocks.csv").set_index("block")
    the32 = old.index[old.dL_P_U < -1e-9]
    g = wide[wide.index.isin(the32)]
    r32 = []
    for lb in arms:
        if lb == "W":
            continue
        for scorer in ("U", "F"):
            own, vs = f"L_P_{scorer}_{lb}", f"L_P_{scorer}_T"
            if own not in g or vs not in g:
                continue
            d = (g[own] - g[vs]).dropna()
            r32.append(dict(arm=lb, scorer=scorer, blocks=len(d), below=int((d < -1e-9).sum()),
                            mean=float(d.mean()), median=float(d.median()), worst=float(d.min()),
                            below_mean=float(d[d < -1e-9].mean()) if (d < -1e-9).any() else 0.0))
    pd.DataFrame(r32).to_csv(ROWS / "summary_32.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(4).T.to_string())


def main(argv: list[str]) -> None:
    argv, standing = clearings.standing_of(argv)
    if len(argv) == 4 and argv[0] == "arm" and argv[3] in ("cpu", "gpu"):
        run_arm(argv[2], argv[1], argv[3], standing=standing)
    elif len(argv) == 4 and argv[0] == "block" and argv[3] in ("cpu", "gpu"):
        for label in argv[2].split(","):
            run_arm(label, argv[1], argv[3], standing=standing)
        lenses(argv[1], argv[3], standing=standing)
        chain_check(argv[1], argv[3], standing=standing)
    elif len(argv) == 3 and argv[0] == "lenses" and argv[2] in ("cpu", "gpu"):
        lenses(argv[1], argv[2], standing=standing)
    elif len(argv) == 3 and argv[0] == "chains" and argv[2] in ("cpu", "gpu"):
        chain_check(argv[1], argv[2], standing=standing)
    elif len(argv) == 2 and argv[0] == "wlenses":
        wlenses(argv[1], standing=standing)
    elif argv == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
