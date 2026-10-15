"""Priority-weighted P: can it replace J_2 as the equity view? (BACKLOG 4, the metric decisions;
owner 2026-10-12: "what would happen if you first set the current emitted by deeper buildings
quadratically higher and then did P? Would you get something like J_2? ... Let's see how weighted
P behaves. It would be nice if it could replace J_2.") NOTES, "Priority-weighted P"; decided
2026-10-13: P_{1/2} is the equity view beside P.

P_q: home i injects w_i a_i, a_i = u0_i^q / mean_w(u0^q) (common.PriorityDemand), u0_i its escape
time at baseline under the population's own demand; P_q = f_q^T L^-1 f_q, its lens 1 - P_q / P_q0.
Four objectives (OBJECTIVES: P, J_2, P_{1/2}, P_1), one optimizer for all: SIMP base (PLAN, the
presets' base plan) with free first (relax.one), at area D 0.05 (BUDGET), area population, on the
runtime's metric (METRIC: the gap complex, phase 2b; its model's baseline gives the weights,
gap_clear.Model.inj_q), on the CPU. For the convex ones (power 1 under a fixed demand) the
relaxation's Frank-Wolfe bound too: relax.frank_wolfe from x = 0 with nothing held, in SIMP base's
eps world, so it bounds every clearing within the budget. `score` scores every clearing under
every objective on the complex's model, with each home's escape time under the population's demand
(the tails), B (i), how deep the cleared buildings sit and how far apart the clearings are;
`report` answers the four questions by the criteria fixed in NOTES before any result, and writes
the summaries.

    S=research/roadless/pweight.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S task <id> <item>             # one (block, item): a cluster.py `cpu` task
    uv run python $S score <procs> <ids,|all>     # here, every block whose items are all back
    uv run python $S report

An item is simp-<objective> (SIMP base) or fw-<objective> (the bound; convex objectives only), an
objective P, J<p> (J_p) or Pq<q> (P_q): ITEMS. Rows: ROWS/<simp|fw>/<objective>/<id>.parquet and
ROWS/score/<id>.parquet (ignored); the summaries ROWS/summary_*.csv (committed). ROWS is
pweight_rows/<METRIC's name>. The study itself ran on the raster (uni, then uni+apwfs:
pweight_rows/<simp|fw|score>/ with its summary_*.csv, and pweight_rows/uni+apwfs/), the record of
its answer; its code was the raster scorer's, ported here (phase 2b).
"""
from __future__ import annotations

import multiprocessing
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_factor  # noqa: E402
import metrics  # noqa: E402
import relax  # noqa: E402
import road_rescore as rr  # noqa: E402

# ------------------------------------------------------------------ the run's configuration
# Everything that can change a number is here, and named in the rows (the run's identity).
PLAN = relax.plan_of("fw0.q3.i10.t0.001.e0.0001.k1s1e-06")    # SIMP base: the presets' base plan
BUDGET = relax.D_LENS          # area D 0.05 (one SIMP run is one budget: D 0.10 would be a second)
METRIC = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.SuperLU())  # the CPU's
FW_ITERS = 40                  # the bound's Frank-Wolfe iterations (as relax220_p1's fw40)
def rows_of(metric: str) -> Path:
    """Every row and summary of the study under `metric`, by its name."""
    return HERE / "pweight_rows" / metric


ROWS = rows_of(METRIC.name)
MARGIN = 0.005                 # questions 1 and 2: within this of J_2's clearing
BOOT_DRAWS = 10_000            # the Bayesian bootstrap's draws ...
BOOT_SEED = 20261012           # ... and its seed (one weight matrix for every comparison: the
                               # transparency experiment's, whose report it shared)


@dataclass(frozen=True)
class Objective:
    """J_power under `demand`: P (the population's demand, power 1), J<p> (power p), Pq<q>
    (common.PriorityDemand(q), power 1). `name` names the arm and its rows, q in it."""
    name: str
    demand: common.Demand
    power: float

    @property
    def convex(self) -> bool:
        """P under a fixed demand is convex in the clearing fractions (relax.py), J_p for p > 1
        is not: only a convex objective's Frank-Wolfe gap certifies a bound."""
        return self.power == 1.0


def objective_of(token: str) -> Objective:
    if token == "P":
        return Objective(token, common.PopulationDemand(), 1.0)
    if m := re.fullmatch(r"J([0-9]+(?:\.[0-9]+)?)", token):
        return Objective(token, common.PopulationDemand(), float(m[1]))
    if m := re.fullmatch(r"Pq([0-9]+(?:\.[0-9]+)?)", token):
        return Objective(token, common.PriorityDemand(float(m[1])), 1.0)
    raise ValueError(f"unknown objective {token!r}: P, J<p> or Pq<q>")


OBJECTIVES = tuple(objective_of(t) for t in ("P", "J2", "Pq0.5", "Pq1"))
ITEMS = (*(f"simp-{o.name}" for o in OBJECTIVES),
         *(f"fw-{o.name}" for o in OBJECTIVES if o.convex))


def row_path(stage: str, obj: Objective, bid: str) -> Path:
    return ROWS / stage / obj.name / f"{bid}.parquet"


def score_path(bid: str) -> Path:
    return ROWS / "score" / f"{bid}.parquet"


# ------------------------------------------------------------------ the optimizer and the bound

def bound(bid: str, obj: Objective) -> dict[str, object]:
    """The relaxation's Frank-Wolfe bound on `obj` (convex) at BUDGET: from x = 0, nothing held,
    in SIMP base's eps world (J_eps <= J for every 0/1 clearing, so its lower bound covers every
    real clearing within the budget, free first or not), FW_ITERS iterations."""
    if not obj.convex:
        raise SystemExit(f"{obj.name}: J_{obj.power:g} is not convex, its gap bounds nothing")
    [b] = common.build_blocks([bid])
    c = METRIC.clearing(b, demand=obj.demand)
    rel = c.relaxation(obj.power, q=1.0, eps=PLAN.eps)
    t0 = time.time()
    fw = relax.frank_wolfe(rel, BUDGET, FW_ITERS,
                           log=lambda s: print(f"{bid} {obj.name} {s}", flush=True))
    last = fw["hist"][-1]
    x = fw["x"]
    return dict(block=bid, n=c.n, power=obj.power, demand=obj.demand.name, eps=PLAN.eps,
                fw_iters=FW_ITERS, iters=len(fw["hist"]),
                relaxed_perm=rel.perm(fw["J"]), bound_perm=rel.perm(fw["lb"]),
                last_gap=float(last["gap"] / last["J"]), J0=rel.J0,
                frac_share=float(np.mean((x > 1e-6) & (x < 1 - 1e-6))),
                fw_D=float(c.cost @ x), t=time.time() - t0)


def task(bid: str, item: str) -> None:
    """One (block, item), single-threaded: SIMP base (simp-) or the bound (fw-), its row to
    row_path, written whole (rr.write_parquet)."""
    rr._check_single_threaded()
    stage, _, token = item.partition("-")
    obj = objective_of(token)
    out = row_path(stage, obj, bid)
    t0 = time.time()
    if stage == "simp":
        row = relax.one(bid, obj.power, METRIC, PLAN, BUDGET, demand=obj.demand)
    elif stage == "fw":
        row = bound(bid, obj)
    else:
        raise SystemExit(f"unknown item {item!r}: simp-<objective> or fw-<objective>")
    row.update(objective=obj.name, item=item, budget=BUDGET, host=socket.gethostname(),
               peak_rss_gb=rr._status_gb("VmHWM"), wall=time.time() - t0)
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(pd.DataFrame([row]), out)
    print(f"{time.strftime('%H:%M:%S')} {bid} {item} done in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ every clearing, every lens

POPULATION = common.PopulationDemand()      # the people's own escape: the tails and the depths


def score_block(bid: str) -> None:
    """Every objective's SIMP clearing on `bid` (its simp rows) scored under every objective on
    the complex's model (gap_clear.Model, the population's demand): each objective's demand makes
    its injection from the baseline's escape times under the population's demand (Demand.injection,
    as the optimizer's model did), then one state per clearing and one solve per demand. Per
    (block, clearing): every objective's lens (L_<objective>), each home's escape time under the
    population's demand relative to baseline at the median, p95 and max home (umed, u95, umax:
    tails.py's ratios of quantiles), B (i) (`enclosed`), the area displacement, the depth of the
    cleared buildings relative to the population-weighted mean escape time at baseline
    (depth_own: their own homes' escape time, area-weighted; depth_served: the escape time of the
    homes whose flow passes through them in the clearing, flow-weighted, clear.Catchment's shares
    in the tension's eps world), the Jaccard overlap of the cleared set with every other
    objective's (jac_<objective>), and the clearing's own lens against its row's
    (`own_minus_row`: the model reproduces the optimizer's)."""
    rows = {o.name: pd.read_parquet(row_path("simp", o, bid)).iloc[0] for o in OBJECTIVES}
    t0 = time.time()
    [b] = common.build_blocks([bid])
    m = METRIC.clearing(b, demand=POPULATION)
    live = ~m.nodem
    demands = list(dict.fromkeys(o.demand for o in OBJECTIVES))
    inj = {d: d.injection(m.pop, m.u_pop0, m.nodem) for d in demands}
    st0 = m.state(np.zeros(m.n))
    u_base = {d: m.home_u(st0, m.solve(st0, inj[d])) for d in demands}
    u0 = u_base[POPULATION]
    mean_u0 = float((m.pop[live] * u0[live]).sum() / m.pop[live].sum())
    cleared = {name: np.zeros(m.n, dtype=bool) for name in rows}
    for name, r in rows.items():
        cleared[name][np.asarray(r.simp_cleared, dtype=np.int64)] = True
    catch = clear.Catchment()
    out = []
    for name, r in rows.items():
        x = cleared[name]
        st = m.state(x.astype(np.float64))
        u = {d: m.home_u(st, m.solve(st, inj[d])) for d in demands}
        lens = {}
        for o in OBJECTIVES:
            J = m.J(u[o.demand], inj[o.demand], o.power)
            J0 = m.J(u_base[o.demand], inj[o.demand], o.power)
            lens[o.name] = 1.0 - (J / J0) ** (1.0 / o.power)
        uw = u[POPULATION]
        cand = np.flatnonzero(x)
        own = cand[live[cand]]
        m.removed[:] = x
        V = catch.shares(m.tension(1.0, restore=False).flow(cand.tolist()), len(cand))
        m.removed[:] = False
        flow = m.pop[:, None] * V * live[:, None]
        out.append(dict(
            block=bid, n=m.n, objective=name, n_cleared=int(x.sum()), D=float(m.cost @ x),
            free_D=float(r.free_D), enclosed=m.enclosed_left(x),
            enclosed0=float(m.cost[m.walled0].sum()),
            **{f"L_{k}": v for k, v in lens.items()},
            umed=float(np.nanmedian(uw) / np.nanmedian(u0)),
            u95=float(np.nanpercentile(uw, 95) / np.nanpercentile(u0, 95)),
            umax=float(np.nanmax(uw) / np.nanmax(u0)),
            depth_own=float((m.cost[own] * u0[own]).sum() / m.cost[own].sum() / mean_u0),
            depth_served=float((flow * np.nan_to_num(u0)[:, None]).sum() / flow.sum()
                               / mean_u0),
            **{f"jac_{k}": float((x & v).sum() / (x | v).sum()) for k, v in cleared.items()},
            own_minus_row=float(lens[name] - r.simp_perm), row_perm=float(r.simp_perm),
            t_simp=float(r.t), host=str(r.host)))
    for d, v in inj.items():                       # the demands' spread, for the description
        a = v[live] / m.pop[live]
        for row in out:
            row[f"amax_{d.name}"] = float(a.max())
            row[f"amin_{d.name}"] = float(a.min())
    for row in out:
        row["t_score"] = time.time() - t0
    score_path(bid).parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(pd.DataFrame(out), score_path(bid))
    print(f"{time.strftime('%H:%M:%S')} {bid} n={m.n} scored in {time.time() - t0:.0f}s; own "
          f"lens against the rows within {max(abs(o['own_minus_row']) for o in out):.1e}",
          flush=True)


def _score_one(bid: str) -> str | None:
    try:
        score_block(bid)
    except Exception as e:
        print(f"{bid} FAILED {type(e).__name__}: {e}"[:300], flush=True)
        return bid
    return None


def score(procs: int, ids: list[str]) -> None:
    """score_block on every block of `ids` whose simp rows are all back and that has no score
    row yet, `procs` at a time (fork, one block a child)."""
    rr._check_single_threaded()
    ready = [b for b in ids if all(row_path("simp", o, b).exists() for o in OBJECTIVES)]
    todo = [b for b in ready if not score_path(b).exists()]
    print(f"{len(ids)} blocks, {len(ready)} with every simp row, {len(todo)} to score", flush=True)
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        failed = [f for f in pool.imap_unordered(_score_one, todo) if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


# ------------------------------------------------------------------ the four questions
OWN_TOL = 1e-6        # a clearing's own lens as scored here against its row's (solver noise)
ENCLOSED_TOL = 1e-12  # B (i) shares equal within this (sums of weights)


def _interval(W: np.ndarray, d: np.ndarray) -> tuple[float, float, float]:
    """The mean of the per-block `d` and its Bayesian-bootstrap 95% interval (W's draws)."""
    boot = W @ d
    return float(d.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def _q(s: pd.Series) -> dict[str, float]:
    return dict(median=float(s.median()), p25=float(s.quantile(0.25)),
                p75=float(s.quantile(0.75)), mean=float(s.mean()), min=float(s.min()),
                max=float(s.max()))


def boot_weights(n: int) -> np.ndarray:
    """The Bayesian bootstrap's weights over `n` blocks: BOOT_DRAWS Dirichlet(1, ..., 1) draws,
    seeded BOOT_SEED, one matrix (draws x blocks) for every comparison, so the intervals are
    paired."""
    return np.random.default_rng(BOOT_SEED).dirichlet(np.ones(n), size=BOOT_DRAWS)


def report() -> None:
    """The four questions (NOTES, "Priority-weighted P": the criteria, fixed before any result)
    on the 220, paired by block, Bayesian-bootstrap 95% intervals (boot_weights: BOOT_DRAWS
    Dirichlet(1, ..., 1) draws over the blocks, seeded, one matrix for every comparison).
    Writes ROWS/summary_{lens,questions,bound,describe,cost}.csv."""
    pd.set_option("display.width", 200)
    blocks = tuple(common.recipients())
    absent = [b for b in blocks if not score_path(b).exists()]
    if absent:
        raise SystemExit(f"{len(absent)} blocks unscored, e.g. {absent[:3]}")
    t = pd.concat([pd.read_parquet(score_path(b)) for b in blocks], ignore_index=True)
    names = tuple(o.name for o in OBJECTIVES)
    lens_cols = [f"L_{k}" for k in names]
    if t[lens_cols + ["u95", "umed", "umax"]].isna().any().any() or len(t) != 4 * len(blocks):
        raise SystemExit("not every clearing has every lens on every block")
    worst = float(t.own_minus_row.abs().max())
    print(f"{len(blocks)} blocks. Own lens as scored here against the optimizer's row: within "
          f"{worst:.1e}")
    if worst > OWN_TOL:
        raise SystemExit(f"the scorer does not reproduce the optimizer's lens ({worst:.1e})")

    def wide(col: str) -> pd.DataFrame:
        return t.pivot(index="objective", columns="block", values=col).reindex(
            index=list(names), columns=list(blocks))

    W = boot_weights(len(blocks))
    # every clearing under every lens: medians (and means) over the blocks
    lens = []
    for name in names:
        g = t[t.objective == name]
        lens.append(dict(clearing=name, blocks=len(g),
                         **{f"{c}_median": float(g[c].median()) for c in
                            [*lens_cols, "umed", "u95", "umax", "D", "n_cleared", "depth_own",
                             "depth_served"]},
                         **{f"{c}_mean": float(g[c].mean()) for c in [*lens_cols, "umed", "u95",
                                                                     "umax"]},
                         enclosed_max=float(g.enclosed.max()),
                         enclosed_blocks=int((g.enclosed > ENCLOSED_TOL).sum()),
                         enclosed0_blocks=int((g.enclosed0 > ENCLOSED_TOL).sum())))
    lens_t = pd.DataFrame(lens)
    print("\nEvery clearing (row) under every lens (median over blocks; u: relative to "
          "baseline, lower is better):")
    print(lens_t[["clearing", *[f"{c}_median" for c in [*lens_cols, "umed", "u95", "umax"]],
                  "enclosed_blocks"]].to_string(index=False, float_format="%.4f"))
    # questions 1 and 2: P_q's clearing against J_2's
    enc = wide("enclosed")
    qrows = []
    for arm in ("Pq0.5", "Pq1", "P"):
        behind = int((enc.loc[arm] > enc.loc["J2"] + ENCLOSED_TOL).sum())
        for qty, ref, col in (("J2 lens", "J2", "L_J2"), ("u95", "J2", "u95"),
                              ("umed", "J2", "umed"), ("umax", "J2", "umax"),
                              ("P lens", "P", "L_P"), ("own lens", "J2", f"L_{arm}")):
            if ref == arm:                          # P's P lens against P's own clearing
                continue
            w = wide(col)
            d = (w.loc[arm] - w.loc[ref]).to_numpy(dtype=float)
            mean, lo, hi = _interval(W, d)
            up = d > 0 if "lens" in qty else d < 0  # a lens is better higher, an escape lower
            qrows.append(dict(arm=arm, against=ref, quantity=qty, mean=mean, lo=lo, hi=hi,
                              median=float(np.median(d)), better=int(up.sum()),
                              behind_i=behind))
    qt = pd.DataFrame(qrows)

    def holds(arm: str) -> dict[str, bool]:
        r = qt[qt.arm == arm].set_index("quantity")
        return dict(lens=bool(r.loc["J2 lens", "lo"] >= -MARGIN),
                    u95=bool(r.loc["u95", "hi"] <= MARGIN),
                    enclosed=bool(r.loc["J2 lens", "behind_i"] == 0))

    verdict = {arm: holds(arm) for arm in ("Pq0.5", "Pq1", "P")}
    qt["replaces_J2"] = qt.arm.map(lambda a: all(verdict[a].values()))
    print("\nQuestions 1 and 2: each arm's clearing minus the reference's, mean [95% interval] "
          f"over {len(blocks)} blocks (P beside them, q 0, for reference):")
    for arm in ("Pq0.5", "Pq1", "P"):
        v = verdict[arm]
        print(f"  {arm}: J2 lens within {MARGIN} ({v['lens']}), u95 within {MARGIN} "
              f"({v['u95']}), never behind on B (i) ({v['enclosed']}) -> can replace J_2: "
              f"{all(v.values())}")
        for _, r in qt[qt.arm == arm].iterrows():
            print(f"    {r.quantity:9s} vs {r.against:5s} {r['mean']:+.4f} [{r.lo:+.4f}, "
                  f"{r.hi:+.4f}]  median {r['median']:+.4f}  better on {r.better}")
    # question 3: the certified bound against the rounded clearing
    brows = []
    for o in OBJECTIVES:
        rounded = wide(f"L_{o.name}").loc[o.name]
        if not o.convex:
            brows.append(dict(objective=o.name, convex=False,
                              **{f"rounded_{k}": v for k, v in _q(rounded).items()}))
            continue
        fw = pd.concat([pd.read_parquet(row_path("fw", o, b)) for b in blocks],
                       ignore_index=True).set_index("block").reindex(list(blocks))
        gap = fw.bound_perm - rounded
        brows.append(dict(objective=o.name, convex=True,
                          **{f"rounded_{k}": v for k, v in _q(rounded).items()},
                          **{f"bound_{k}": v for k, v in _q(fw.bound_perm).items()},
                          **{f"relaxed_{k}": v for k, v in _q(fw.relaxed_perm).items()},
                          **{f"gap_{k}": v for k, v in _q(gap).items()},
                          **{f"fwgap_{k}": v for k, v in _q(fw.bound_perm - fw.relaxed_perm
                                                          ).items()},
                          gap_share_median=float((gap / fw.bound_perm).median()),
                          below_bound=int((gap >= -1e-9).sum()),
                          iters_median=float(fw.iters.median()),
                          grey_median=float(fw.frac_share.median())))
    bt = pd.DataFrame(brows)
    print("\nQuestion 3: the certified bound (Frank-Wolfe, every clearing within D) against the "
          "rounded clearing, medians [quartiles] over the blocks:")
    for _, r in bt.iterrows():
        if not r.convex:
            print(f"  {r.objective:6s} rounded {r.rounded_median:.4f}: not convex, no bound")
            continue
        print(f"  {r.objective:6s} rounded {r.rounded_median:.4f}, bound {r.bound_median:.4f}, "
              f"gap {r.gap_median:.4f} [{r.gap_p25:.4f}, {r.gap_p75:.4f}] ({r.gap_share_median:.0%}"
              f" of the bound), relaxed {r.relaxed_median:.4f} (FW gap left {r.fwgap_median:.4f}),"
              f" rounded <= bound on {r.below_bound:.0f}")
    # question 4: how far apart, and where they clear
    drows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            j = t[t.objective == a].set_index("block")[f"jac_{b}"].reindex(list(blocks))
            drows.append(dict(kind="jaccard", a=a, b=b, **_q(j)))
    for col in ("depth_served", "depth_own"):
        w = wide(col)
        for name in names:
            mean, lo, hi = (_interval(W, (w.loc[name] - w.loc["P"]).to_numpy(dtype=float))
                            if name != "P" else (np.nan,) * 3)
            drows.append(dict(kind=col, a=name, b="P", **_q(w.loc[name]), vs_P=mean,
                              vs_P_lo=lo, vs_P_hi=hi))
    dt = pd.DataFrame(drows)
    print("\nQuestion 4: Jaccard of the cleared sets, median [quartiles]:")
    for _, r in dt[dt.kind == "jaccard"].iterrows():
        print(f"  {r.a:6s} {r.b:6s} {r['median']:.3f} [{r.p25:.3f}, {r.p75:.3f}]")
    print("  depth of the cleared buildings, relative to the block's mean escape time at "
          "baseline (median over blocks; minus P's, mean [95% interval]):")
    for _, r in dt[dt.kind != "jaccard"].iterrows():
        print(f"  {r.kind:12s} {r.a:6s} {r['median']:.3f}"
              + (f"  {r.vs_P:+.3f} [{r.vs_P_lo:+.3f}, {r.vs_P_hi:+.3f}]" if r.a != "P" else ""))
    # the cost
    crows = []
    for item in ITEMS:
        stage, _, token = item.partition("-")
        o = objective_of(token)
        r = pd.concat([pd.read_parquet(row_path(stage, o, b)) for b in blocks],
                      ignore_index=True)
        crows.append(dict(item=item, tasks=len(r), task_h=float(r.wall.sum() / 3600),
                          wall_median_s=float(r.wall.median()), wall_max_s=float(r.wall.max()),
                          peak_rss_gb=float(r.peak_rss_gb.max()),
                          hosts=",".join(f"{h}:{n}" for h, n in r.host.value_counts().items())))
    ct = pd.DataFrame(crows)
    print("\nCost:")
    print(ct.to_string(index=False, float_format="%.2f"))
    lens_t.to_csv(ROWS / "summary_lens.csv", index=False)
    qt.to_csv(ROWS / "summary_questions.csv", index=False)
    bt.to_csv(ROWS / "summary_bound.csv", index=False)
    dt.to_csv(ROWS / "summary_describe.csv", index=False)
    ct.to_csv(ROWS / "summary_cost.csv", index=False)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["task"] and len(a) == 3:
        task(a[1], a[2])
    elif a[:1] == ["score"] and len(a) == 3:
        score(int(a[1]), common.recipients() if a[2] == "all" else common.ids_of(a[2]))
    elif a == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)
