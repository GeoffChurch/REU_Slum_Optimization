"""Stage 3's selection (docs/superpowers/plans/2026-10-25-roads-from-the-vote.md; owner 2026-10-25:
"we can dogfood it just for dogfooding's sake"): which candidate pool the lazy ΔP greedy builds
next beside the lineup's, chosen by GeoffChurch/epistemetry, one build at a time.

Arms (ARMS): pools beside the lineup's (dp_greedy.py's `lineup+<pool>`, lazy:5, cost D), each with
its declared factors (stage: simp_oc, adaptive, vote; dyn: oc, g1, g1.5, periodic, frozen; source:
orig, U) and two covariates read from its stand-alone results: `lead`, its own L_P over the
lineup-only lazy build at its own D (a single answer: road_simp's linked roads at their exact D;
an ordered network: vote_roads' device rows at area D 0.05, 0.10, 0.15, the mean), and `km`, the
pool's length (a single answer's whole; an ordered network's at its own D 0.15 cut, the same
scale: frozen U power's whole network runs to 34.5 km, far past D 0.15). An arm of two pools: the
larger lead, the summed length.

The outcome y: a build's mean L_P gain over the lineup-only lazy build (dp_greedy_lineup_lazy5_D)
at area D 0.05, 0.10 and 0.15, each build's L_P interpolated at exact D from its own steps (as the
greedy's curves are read). The three SIMP-pool builds already made are observations.

The belief: epistemetry.models' latent Gaussian model over the factors (Identity kernels, spreads
STAGE_SD, DYN_SD, SOURCE_SD) and the covariates as fixed effects with normal priors (centred 0:
INTERCEPT_SD, LEAD_SD, KM_SD), Gaussian misfit MISFIT_SD per observation; its posterior at the
unbuilt arms (ExactGaussian.posterior), the misfit added, drawn DRAWS times. The hyperparameters
are stated, not fitted: three observations cannot fit five numbers (fit_reml has nothing to
restrict them by).

The choice: epistemetry.design.decide.GittinsIndex (Weitzman's reservation values), the price
PRICE_PER_HOUR (L_P per wall-hour of a build: the rate bracket's bottom, 2e-6 L_P per GPU-second,
at a local core-hour counted as an eighth of a GPU-hour, six cores a build), each arm's cost its
expected hours (HOURS), the best found the largest observed y; it opens the arm whose index beats
it, or stops. Every decision appends to LEDGER: the posterior's means and sds, the indices, the
choice, and the rule's record.

    PYTHONPATH=. uv run python research/roadless/vote_select.py next           # choose, record
    PYTHONPATH=. uv run python research/roadless/vote_select.py outcomes       # y of every build
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from epistemetry.design.decide import GittinsIndex
from epistemetry.models.declarations import (
    AbsentLevel,
    Categorical,
    Covariate,
    Declarations,
    Factor,
    FixedEffect,
    Intercept,
    Normal,
    Observation,
    Part,
    Spread,
)
from epistemetry.models.kernels import Identity
from epistemetry.models.lgm import Design, ExactGaussian, FoldCheapest
from epistemetry.models.likelihoods import KnownVariance

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics  # noqa: E402

BID = "ZAF.9.3.1_1_5810"
FLOW = HERE.parents[1] / ".superpowers/sdd/2026-10-22-flow-roads"     # dp_greedy's builds
SIMP = HERE / "road_simp_rows" / metrics.RUNTIME
VOTE = HERE / "vote_rows" / metrics.RUNTIME
LEDGER = HERE / "vote_rows" / "greedy_selection.json"
BASE = "dp_greedy_lineup_lazy5_D"
TARGETS = (0.05, 0.10, 0.15)
STAGE_SD, DYN_SD, SOURCE_SD = 0.04, 0.02, 0.01      # the factors' spreads (L_P)
INTERCEPT_SD, LEAD_SD, KM_SD = 0.05, 0.5, 0.01      # the fixed effects' priors (per unit, per km)
MISFIT_SD = 0.01                                    # an observation's misfit (L_P)
PRICE_PER_HOUR = 6 / 8 * 3600 * 2e-6                # L_P per wall-hour of a build
DRAWS = 4000
SEED = 20261026

R6 = ".p1e0.0001v6"
R7 = ".p1e0.0001v7"
VR = ".q0.98,0.95,0.9,0.85,0.8.r1.w7.m5j2s0.5"


class Arm(NamedTuple):
    key: str
    pools: tuple[str, ...]          # dp_greedy pool specs beside the lineup's
    stage: str
    dyn: str
    source: str
    hours: float                    # expected build time


ARMS = (
    Arm("simp_r3.5", (f"simp:rs.D0.05.k0.02.r3.5.q1,2,3,3,3.b1,2,4,8,16.i60{R6}",),
        "simp_oc", "oc", "orig", 2.0),
    Arm("simp_r7", (f"simp:rs.D0.05.k0.01.r7.q1,2,3,3,3.b1,2,4,8,16.i60{R6}",),
        "simp_oc", "oc", "orig", 2.0),
    Arm("simp_r7x2", (f"simp:rs.D0.05.k0.01.r7.q1,2,3,3,3.b1,2,4,8,16.i60{R6}",
                      f"simp:rs.D0.15.k0.01.r7.q1,2,3,3,3.b1,2,4,8,16.i60{R6}"),
        "simp_oc", "oc", "orig", 2.4),
    Arm("vote_per_orig", (f"vote:vr.orig.cur.per0.01k.jf300{VR}",), "vote", "periodic", "orig",
        2.0),
    Arm("vote_per_U", (f"vote:vr.U.cur.per0.01k.jf300{VR}",), "vote", "periodic", "U", 2.0),
    Arm("vote_frozen_U_pow", (f"vote:vr.U.pow.frozen.jf300{VR}",), "vote", "frozen", "U", 2.6),
    Arm("adapt_g1", (f"simp:rs.D0.05.k0.01.r7.q1,1,1,1,1.b1,2,4,8,16.i40.uA1.c0{R7}",),
        "adaptive", "g1", "orig", 2.0),
    Arm("adapt_g1.5", (f"simp:rs.D0.05.k0.01.r7.q1,1,1,1,1.b1,2,4,8,16.i40.uA1.5.c0{R7}",),
        "adaptive", "g1.5", "orig", 2.0),
    Arm("adapt_g1_U", (f"simp:rs.D0.05.k0.01.r7.q1,1,1,1,1.b1,2,4,8,16.i40.uA1.cUdevice{R7}",),
        "adaptive", "g1", "U", 2.0),
)


def pool_spec(a: Arm) -> str:
    return "+".join(("lineup", *a.pools))


def run_dir(a: Arm) -> Path:
    """dp_greedy's run directory of the arm's build (its name_of: the pool's name, the pick's, the
    cost's; a pool part `simp:<plan>` is named `simp-<plan>`, `vote:<run>` `vote-<run>`)."""
    parts = ("lineup", *(p.replace(":", "-", 1) for p in a.pools))
    return FLOW / f"dp_greedy_{'+'.join(parts)}_lazy5_D"


def curve(run: Path) -> tuple[np.ndarray, np.ndarray]:
    steps = json.loads((run / "steps.json").read_text())
    return (np.array([0.0] + [s["D"] for s in steps]), np.array([0.0] + [s["L_P"] for s in steps]))


def lazy_at(D: float | np.ndarray) -> np.ndarray:
    d, lp = curve(FLOW / BASE)
    return np.interp(D, d, lp)


def outcome(a: Arm) -> float | None:
    """The arm's y, or None while it is unbuilt (its run lacks built.parquet)."""
    run = run_dir(a)
    if not (run / "built.parquet").exists():
        return None
    d, lp = curve(run)
    t = np.array(TARGETS)
    return float(np.mean(np.interp(t, d, lp) - lazy_at(t)))


def covariates(a: Arm) -> dict[str, float]:
    leads, kms = [], []
    for p in a.pools:
        kind, name = p.split(":", 1)
        if kind == "simp":
            r = pd.read_parquet(SIMP / name / f"{BID}.parquet").iloc[-1]
            leads.append(float(r.roads_L_P - lazy_at(r.roads_D_exact)))
            kms.append(float(r.roads_m) / 1000)
        else:
            rows = pd.read_parquet(VOTE / name / f"{BID}.parquet")
            dev = rows[rows.role == "dev"]
            leads.append(float(np.mean([float(dev.L_P[np.isclose(dev.target, t)].iloc[0])
                                        - float(lazy_at(t)) for t in TARGETS])))
            kms.append(float(dev.km[np.isclose(dev.target, TARGETS[-1])].iloc[0]))
    return dict(lead=max(leads), km=float(sum(kms)))


def declarations() -> Declarations:
    attrs = tuple(Categorical(name=n, absent=AbsentLevel()) for n in ("stage", "dyn", "source"))
    factors = tuple(Factor(name=n, parts=(Part(attribute=n, kernel=Identity()),),
                           spread=Spread(name=f"{n}_sd")) for n in ("stage", "dyn", "source"))
    fixed = (FixedEffect(name="intercept", columns=Intercept(), prior=Normal(sd=INTERCEPT_SD)),
             FixedEffect(name="lead", columns=Covariate(name="lead"), prior=Normal(sd=LEAD_SD)),
             FixedEffect(name="km", columns=Covariate(name="km"), prior=Normal(sd=KM_SD)))
    return Declarations(attributes=attrs, implications=(), nestings=(), factors=factors,
                        fixed=fixed, covariates=("lead", "km"))


THETA = dict(stage_sd=STAGE_SD, dyn_sd=DYN_SD, source_sd=SOURCE_SD)


def belief(cov: dict[str, dict[str, float]], y: dict[str, float]):  # type: ignore[no-untyped-def]
    obs = [Observation(levels=dict(stage=a.stage, dyn=a.dyn, source=a.source),
                       covariates=cov[a.key]) for a in ARMS]
    design = Design.of(declarations(), obs)
    seen = np.array([i for i, a in enumerate(ARMS) if a.key in y], dtype=np.int64)
    model = ExactGaussian.of(design=design, observed=seen,
                             y=np.array([y[ARMS[i].key] for i in seen]),
                             noise=KnownVariance(values=np.full(len(seen), MISFIT_SD ** 2)),
                             folding=FoldCheapest())
    theta = {h: THETA[h] for h in model.hyperparameters}
    return model, theta


def next_choice() -> None:
    cov = {a.key: covariates(a) for a in ARMS}
    y = {a.key: v for a in ARMS if (v := outcome(a)) is not None}
    model, theta = belief(cov, y)
    todo = [i for i, a in enumerate(ARMS) if a.key not in y]
    entry: dict[str, object] = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), observed=y,
                                    covariates=cov, theta=theta, rule=GittinsIndex(
                                        price=PRICE_PER_HOUR).record(),
                                    priors=dict(intercept=INTERCEPT_SD, lead=LEAD_SD, km=KM_SD,
                                                misfit=MISFIT_SD), draws=DRAWS, seed=SEED)
    best = max(y.values())
    if not todo:
        entry.update(choice=None, reason="every arm built")
    else:
        mean, cv = model.posterior(theta, beta=np.zeros(0), targets=np.array(todo))
        cv = cv + MISFIT_SD ** 2 * np.eye(len(todo))
        draws = np.random.default_rng(SEED + len(y)).multivariate_normal(mean, cv, size=DRAWS).T
        rule = GittinsIndex(price=PRICE_PER_HOUR)
        hours = np.array([ARMS[i].hours for i in todo])
        idx = rule.indices(draws, hours)
        pick = rule.choose(draws, hours, best_found=best)
        entry.update(best_found=best, arms={ARMS[i].key: dict(
            mean=float(mean[k]), sd=float(np.sqrt(cv[k, k])), index=float(idx[k]),
            hours=ARMS[i].hours, pool=pool_spec(ARMS[i])) for k, i in enumerate(todo)},
            choice=None if pick is None else ARMS[todo[pick]].key,
            reason="no unbuilt arm's index beats the best found" if pick is None else
            "the largest index, above the best found")
    ledger = json.loads(LEDGER.read_text()) if LEDGER.exists() else []
    ledger.append(entry)
    LEDGER.write_text(json.dumps(ledger, indent=1))
    print(json.dumps({k: entry[k] for k in ("best_found", "choice", "reason") if k in entry}))
    if entry.get("arms"):
        for k, v in entry["arms"].items():  # type: ignore[union-attr]
            print(f"  {k:20s} mean {v['mean']:+.4f} sd {v['sd']:.4f} index {v['index']:+.4f}")
    if entry["choice"] is not None:
        print("POOL", pool_spec(next(a for a in ARMS if a.key == entry["choice"])))


def outcomes() -> None:
    for a in ARMS:
        print(f"{a.key:20s} y {outcome(a)}  {covariates(a)}")


if __name__ == "__main__":
    {"next": next_choice, "outcomes": outcomes}[sys.argv[1]]()
