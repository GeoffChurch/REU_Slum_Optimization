"""A uniform rule for walled-in homes: destinations in their pockets, no free first (owner
2026-10-21: "I'd like to measure that trade-off. A uniform rule would be nicer if it works"; NOTES,
the section of that name; the brief .superpowers/sdd/2026-10-21-uniform-leak).

Two arms on a block at area D BUDGET, SIMP's default optimizing P (relax.one):
T, today's: RUNTIME, free first (relax.SIMP_DEFAULT); U, the uniform rule: metrics.UNIFORM (the
leak on every cell open at baseline, the sealed pockets among them; each walled-in home's demand on
its own pocket's ground) and no free first (`.f0`). The polish is scored exactly on a core and in
the eps world on a GPU (as the consolidated pass's presets: the same clearings). Each arm's row is
relax's (relax.rows_of under its metric and plan), with the device, the host, the wall time and the
peak memory.

    S=research/roadless/uniform.py      # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S block <id> <cpu|gpu> [--replace]    # both arms, then the lenses
    uv run python $S report                               # ROWS/summary_*.csv

`block`: the arms (an arm whose row stands is not run again unless `--replace`), then every
clearing (none, T's, U's) scored under both metrics on the same device: Lens A (L_P, L_{P_1/2},
L_J2 on each metric's own baseline: T's over the homes T scores, U's over every home with demand)
and B (i) by street reach (ROWS/lenses/<id>.parquet); the block's facts (ROWS/facts/<id>.parquet:
the walled-in homes, T's free stage and the population it frees, each arm's D, the clearings'
Jaccard); per walled-in home its population share, its escape time under U at baseline and after
each arm's clearing (U's metric: T's has none for it) against the block's median home at baseline
(over the homes T scores: their escape times are T's), and whether each arm's clearing leaves it
walled in (ROWS/homes/<id>.parquet).

`report`: per block (summary_blocks.csv), per walled-in home (summary_homes.csv), and by size
(summary_sizes.csv: the people each arm leaves walled in, those U leaves that T frees, each lens's
paired difference U - T under each metric with its mean and 95% Bayesian-bootstrap interval as the
consolidated reports give them, the D each arm uses, T's free stage's D per unit of the population
it frees, the walled-in homes' escape times under U against the median, the clearings' Jaccard).
The blocks: BLOCKS (the 121 with walled-in homes under RUNTIME and five controls with none).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/uniform.py
"""
from __future__ import annotations

import gc as pygc
import socket
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_calib  # noqa: E402
import gap_clear as gcl  # noqa: E402
import metrics  # noqa: E402
import relax  # noqa: E402
import road_rescore as rr  # noqa: E402
import search  # noqa: E402

ROWS = HERE / "uniform_rows"
BLOCKS = ROWS / "blocks.csv"        # block, size, control (the walled-in blocks' list, controls)
BUDGET = relax.D_LENS                # area D 0.05
POWER = 1.0                          # P
POP = common.PopulationDemand()
LENSES = ("L_P", "L_Pq", "L_J2")
BB_DRAWS = 4000                      # the Bayesian bootstrap's draws (consolidated._bb_interval's)
BB_SEED = 20261021
FAR = 10.0       # a freed home still far: its escape time under U over 10x the block's median


class Arm(NamedTuple):
    """An arm: its label, its metric (on the run's device) and its plan."""
    label: str
    metric: gcl.ComplexMetric
    plan: relax.Plan


class Arms(NamedTuple):
    t: Arm
    u: Arm


def arms_on(device: str) -> Arms:
    """T and U on `device` (module docstring)."""
    scoring = search.polish_scoring_of({"cpu": "E", "gpu": ""}[device])
    plan = relax.plan_of(relax.SIMP_DEFAULT)._replace(polish_scoring=scoring)
    return Arms(t=Arm("T", metrics.metric_of(metrics.RUNTIME, device=device), plan),
                u=Arm("U", metrics.metric_of(metrics.UNIFORM, device=device),
                      plan._replace(free=relax.NoFree())))


def arm_path(arm: Arm, bid: str) -> Path:
    return relax.rows_of(arm.plan, arm.metric.name, BUDGET) / f"{bid}_p{POWER:g}.parquet"


def _outs(bid: str) -> tuple[Path, Path, Path]:
    """The block's lenses, facts and walled-in homes' rows."""
    return (ROWS / "lenses" / f"{bid}.parquet", ROWS / "facts" / f"{bid}.parquet",
            ROWS / "homes" / f"{bid}.parquet")


def run_arm(arm: Arm, bid: str, device: str, *, standing: clearings.Standing) -> None:
    """The arm's SIMP run on `bid`, its row with the device, host, wall time and peak memory; a
    row that stands is read, not run again, unless `standing` replaces it (clearings.Replace)."""
    path = arm_path(arm, bid)
    if standing.reuse(path):
        print(f"{bid} {arm.label}: its row stands ({path.relative_to(HERE)})", flush=True)
        return
    pygc.collect()                      # the last model's plans freed before this one's
    t0 = time.time()
    if device == "gpu":
        with gap_calib.GpuPeak() as peak:
            row = relax.one(bid, POWER, arm.metric, arm.plan, BUDGET, demand=POP)
        gpu_gb = peak.peak
    else:
        row, gpu_gb = relax.one(bid, POWER, arm.metric, arm.plan, BUDGET, demand=POP), np.nan
    row.update(arm=arm.label, device=device, host=socket.gethostname(),
               peak_rss_gb=gap_calib.peak_gb(), gpu_gb=gpu_gb, wall=time.time() - t0)
    standing.write(pd.DataFrame([row]), path)


def _cleared(path: Path, n: int) -> np.ndarray:
    r = pd.read_parquet(path).iloc[0]
    x = np.zeros(n, dtype=bool)
    x[np.asarray(r["simp_cleared"], dtype=np.int64)] = True
    return x


class Scored(NamedTuple):
    """One metric's view of a block (`score`): its lens rows, per clearing (none, T's, U's) the
    homes it leaves walled in, and the baseline's homes."""
    rows: list[dict[str, object]]
    left: tuple[np.ndarray, np.ndarray, np.ndarray]
    u: tuple[np.ndarray, np.ndarray, np.ndarray]
    walled0: np.ndarray
    nodem: np.ndarray
    u0: np.ndarray
    pop: np.ndarray
    cost: np.ndarray
    pocket: np.ndarray
    shadowed: np.ndarray
    after_free: float
    P0: float
    J2_0: float


CLEARINGS = ("none", "T", "U")      # the clearings scored, in Scored.left's order


def score(bid: str, label: str, metric: gcl.ComplexMetric, xs: tuple[np.ndarray, ...],
          held: np.ndarray) -> Scored:
    """The clearings `xs` (CLEARINGS' order) of `bid` under `metric` (its model built here and
    dropped on return: one model in memory at a time), and the population still walled in after
    the free stage's buildings `held` alone."""
    pygc.collect()                      # the last model's plans freed before this one's
    [b] = common.build_blocks([bid])
    m = metric.clearing(b, demand=POP)
    rows, left, u = [], [], []
    for clearing, x in zip(CLEARINGS, xs, strict=True):
        L = m.base if clearing == "none" else m.lenses(x)
        rows.append(dict(block=bid, scorer=label, clearing=clearing, n_cleared=int(x.sum()),
                         D=float(m.cost[x].sum()), **rr._lens_row(m, L)))
        left.append(m.walled0 & ~L.live)
        u.append(L.u)
    return Scored(rows=rows, left=(left[0], left[1], left[2]), u=(u[0], u[1], u[2]),
                  walled0=m.walled0, nodem=m.nodem,
                  u0=m.u_pop0, pop=m.pop, cost=m.cost, pocket=m.dem.pocket,
                  shadowed=m.dem.shadowed, after_free=m.enclosed_left(held), P0=m.base.P,
                  J2_0=m.base.J2)


def lenses(bid: str, device: str, *, standing: clearings.Standing) -> None:
    """Every clearing of `bid` (none, T's, U's) under both metrics; the block's facts; its
    walled-in homes (module docstring); each row written through `standing`
    (clearings.Standing), which answers for all three before any work."""
    outs = _outs(bid)
    if not any([standing.todo(o) for o in outs]):       # every one asked: Refuse raises on any
        return
    t0 = time.time()
    arms = arms_on(device)
    rT = pd.read_parquet(arm_path(arms.t, bid)).iloc[0]
    rU = pd.read_parquet(arm_path(arms.u, bid)).iloc[0]
    n = int(rT["n"])
    xT, xU = _cleared(arm_path(arms.t, bid), n), _cleared(arm_path(arms.u, bid), n)
    xs = (np.zeros(n, dtype=bool), xT, xU)
    held = np.zeros(n, dtype=bool)
    held[np.asarray(rT["free_cleared"], dtype=np.int64)] = True
    sT = score(bid, "T", arms.t.metric, xs, held)
    sU = score(bid, "U", arms.u.metric, xs, held)
    for clearing, a, b in zip(CLEARINGS, sT.left, sU.left, strict=True):
        if not np.array_equal(a, b):
            raise RuntimeError(f"{bid}: T's and U's scorers part on the homes {clearing!r} leaves "
                               "walled in")
    w = sT.walled0
    if not np.array_equal(sU.walled0, w):
        raise RuntimeError(f"{bid}: T and U wall in different homes at baseline")
    scored = ~sT.nodem
    med = float(np.median(sU.u0[scored])) if scored.any() else np.nan
    share = sT.cost
    pU = sU.pop * np.nan_to_num(sU.u0)
    j2U = sU.pop * np.nan_to_num(sU.u0) ** 2
    leftT, leftU = sT.left[1], sT.left[2]
    union = int((xT | xU).sum())
    block = dict(
        block=bid, n=n, n_walled=int(w.sum()), walled_share=float(share[w].sum()),
        nodem_T=int(sT.nodem.sum()), nodem_U=int(sU.nodem.sum()),
        walled_nodem_U=int((w & sU.nodem).sum()), pocket_U=int(sU.pocket.sum()),
        pocket_not_walled=int((sU.pocket & ~w).sum()),
        P0_T=sT.P0, P0_U=sU.P0, J2_0_T=sT.J2_0, J2_0_U=sU.J2_0,
        walled_P0_share_U=float(pU[w].sum() / pU.sum()),
        walled_J2_0_share_U=float(j2U[w].sum() / j2U.sum()),
        free_D=float(rT["free_D"]), free_n=int(held.sum()),
        freed_by_free=float(share[w].sum()) - sT.after_free,
        D_T=float(share[xT].sum()), D_U=float(share[xU].sum()),
        jaccard=float((xT & xU).sum() / union) if union else 1.0,
        left_T=int(leftT.sum()), left_T_share=float(share[leftT].sum()),
        left_U=int(leftU.sum()), left_U_share=float(share[leftU].sum()),
        sealed_U_freed_T=int((leftU & ~leftT).sum()),
        sealed_U_freed_T_share=float(share[leftU & ~leftT].sum()),
        sealed_T_freed_U=int((leftT & ~leftU).sum()),
        t_T=float(rT["t"]), t_U=float(rU["t"]), wall_T=float(rT["wall"]),
        wall_U=float(rU["wall"]), median_u0=med, device=device, host=socket.gethostname(),
        metric_T=arms.t.metric.name, metric_U=arms.u.metric.name, plan_T=arms.t.plan.name,
        plan_U=arms.u.plan.name, wall_lenses=time.time() - t0)
    homes = pd.DataFrame(dict(
        block=bid, home=np.flatnonzero(w), share=share[w], u0_U=sU.u0[w], u0_ratio=sU.u0[w] / med,
        u_T_ratio=sU.u[1][w] / med, u_U_ratio=sU.u[2][w] / med, nodem_U=sU.nodem[w],
        shadowed_U=sU.shadowed[w], left_T=leftT[w], left_U=leftU[w]))
    for o, df in zip(outs, (pd.DataFrame(sT.rows + sU.rows), pd.DataFrame([block]), homes),
                     strict=True):
        standing.write(df, o)
    print(f"{bid}: {int(w.sum())} walled in; T leaves {block['left_T']}, U {block['left_U']}; "
          f"Jaccard {block['jaccard']:.3f}; lenses {time.time() - t0:.0f}s", flush=True)


def block_run(bid: str, device: str, *, standing: clearings.Standing) -> None:
    # a finished block refuses before any work (Refuse), or is done (Keep); every out asked
    if not any([standing.todo(o) for o in _outs(bid)]):
        return
    arms = arms_on(device)
    for arm in arms:
        run_arm(arm, bid, device, standing=standing)
    lenses(bid, device, standing=standing)


# ------------------------------------------------------------------ the report

def _read(kind: str) -> pd.DataFrame:
    paths = sorted((ROWS / kind).glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else \
        pd.DataFrame()


def _bb(d: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """The 95% interval of the mean of `d` under the Bayesian bootstrap."""
    if len(d) == 0:
        return np.nan, np.nan
    wts = rng.dirichlet(np.ones(len(d)), size=BB_DRAWS)
    lo, hi = np.quantile(wts @ d, [0.025, 0.975])
    return float(lo), float(hi)


def report() -> None:
    sizes = pd.read_csv(BLOCKS).set_index("block")
    lens = _read("lenses")
    blk = _read("facts").set_index("block")
    homes = _read("homes")
    blk["size"] = sizes["size"].reindex(blk.index)
    blk["control"] = sizes["control"].reindex(blk.index)
    wide = blk.copy()
    for scorer in ("T", "U"):
        for clearing in ("T", "U"):
            sub = lens[(lens.scorer == scorer) & (lens.clearing == clearing)].set_index("block")
            for x in LENSES:
                wide[f"{x}_{scorer}_{clearing}"] = sub[x].reindex(wide.index)
        for x in LENSES:
            wide[f"d{x}_{scorer}"] = wide[f"{x}_{scorer}_U"] - wide[f"{x}_{scorer}_T"]
    wide["free_D_per_freed"] = np.where(wide.freed_by_free > 0,
                                        wide.free_D / wide.freed_by_free, np.nan)
    wide.reset_index().to_csv(ROWS / "summary_blocks.csv", index=False)
    homes = homes.merge(blk[["size", "median_u0"]].reset_index(), on="block")
    homes.to_csv(ROWS / "summary_homes.csv", index=False)
    rng = np.random.default_rng(BB_SEED)
    out = []
    groups = [("control", wide[wide.control])]
    walled = wide[~wide.control]
    groups += [(s, walled[walled["size"] == s]) for s in ("small", "large", "oversized")]
    groups.append(("all walled", walled))
    for name, g in groups:
        if g.empty:
            continue
        hm = homes[homes.block.isin(g.index)]
        row: dict[str, object] = dict(
            group=name, blocks=len(g), homes_walled=int(g.n_walled.sum()),
            share_walled=float(g.walled_share.sum()), left_T=int(g.left_T.sum()),
            left_U=int(g.left_U.sum()), left_T_share=float(g.left_T_share.sum()),
            left_U_share=float(g.left_U_share.sum()),
            blocks_U_leaves=int((g.left_U > 0).sum()), blocks_T_leaves=int((g.left_T > 0).sum()),
            sealed_U_freed_T=int(g.sealed_U_freed_T.sum()),
            sealed_U_freed_T_share=float(g.sealed_U_freed_T_share.sum()),
            sealed_T_freed_U=int(g.sealed_T_freed_U.sum()),
            D_T_mean=float(g.D_T.mean()), D_U_mean=float(g.D_U.mean()),
            free_D_mean=float(g.free_D.mean()),
            free_D_per_freed_median=float(g.free_D_per_freed.median()),
            free_D_per_freed_p90=float(g.free_D_per_freed.quantile(0.9)),
            free_D_per_freed_max=float(g.free_D_per_freed.max()),
            free_D_per_freed_pooled=(float(g.free_D.sum() / g.freed_by_free.sum())
                                     if g.freed_by_free.sum() > 0 else np.nan),
            jaccard_median=float(g.jaccard.median()), jaccard_min=float(g.jaccard.min()),
            u0_ratio_median=float(hm.u0_ratio.median()) if len(hm) else np.nan,
            u0_ratio_p90=float(hm.u0_ratio.quantile(0.9)) if len(hm) else np.nan,
            u0_ratio_max=float(hm.u0_ratio.max()) if len(hm) else np.nan,
            # the walled-in homes each arm frees (B (i)) and how well U's metric then serves
            # them: their escape time after the clearing against the median at baseline (a home
            # freed on another side keeps its demand in its still-sealed pocket)
            freed_T=int((~hm.left_T).sum()), freed_U=int((~hm.left_U).sum()),
            freed_T_u_median=float(hm.u_T_ratio[~hm.left_T].median()),
            freed_U_u_median=float(hm.u_U_ratio[~hm.left_U].median()),
            freed_T_far=int((hm.u_T_ratio[~hm.left_T] > FAR).sum()),
            freed_U_far=int((hm.u_U_ratio[~hm.left_U] > FAR).sum()),
            walled_P0_share_U_median=float(g.walled_P0_share_U.median()),
            walled_P0_share_U_max=float(g.walled_P0_share_U.max()),
            walled_J2_share_U_median=float(g.walled_J2_0_share_U.median()),
            walled_J2_share_U_max=float(g.walled_J2_0_share_U.max()))
        for scorer in ("T", "U"):
            for x in LENSES:
                d = g[f"d{x}_{scorer}"].to_numpy(dtype=float)
                lo, hi = _bb(d, rng)
                row.update({f"{x}_{scorer}_T_mean": float(g[f"{x}_{scorer}_T"].mean()),
                            f"{x}_{scorer}_U_mean": float(g[f"{x}_{scorer}_U"].mean()),
                            f"d{x}_{scorer}_mean": float(d.mean()),
                            f"d{x}_{scorer}_median": float(np.median(d)),
                            f"d{x}_{scorer}_lo": lo, f"d{x}_{scorer}_hi": hi,
                            f"d{x}_{scorer}_ahead": int((d > 1e-9).sum()),
                            f"d{x}_{scorer}_behind": int((d < -1e-9).sum())})
        out.append(row)
    t = pd.DataFrame(out)
    t.to_csv(ROWS / "summary_sizes.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(4).T.to_string())


def main(argv: list[str]) -> None:
    argv, standing = clearings.standing_of(argv)
    if len(argv) == 3 and argv[0] == "block" and argv[2] in ("cpu", "gpu"):
        block_run(argv[1], argv[2], standing=standing)
    elif argv == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
