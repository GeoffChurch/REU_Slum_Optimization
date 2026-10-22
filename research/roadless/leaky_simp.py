"""Walls that leak, stages 2 and 3: SIMP under the walls against today's rule, then an optimizer fit
for the walls (the owner 2026-10-21: the default walls model the symmetric 1,000 m, "They could
both exist as options but we would hopefully be able to default to the symmetric case if it's not
worse"; the test "1,000 m on all, diode B on 20"; then yes to an optimizer suited to the walls
before judging them; NOTES, "Walls that leak", stages 2 and 3; the briefs
.superpowers/sdd/2026-10-21-leaky-walls/).

Arms at area D BUDGET, each an optimizer of P under a metric. T, today's (RUNTIME, free first,
relax.SIMP_DEFAULT: the uniform study's stored arm, uniform.py, at the same RUNTIME; its polish
scored exactly on a core, in the eps world on a GPU, the large controls). Under the walls
(gap_walls on UNIFORM's mesh, leaks and doors: metrics.walls_on_uniform), S10 the symmetric
Wk1l1000 and B diode B (Wk1o10i3000n64, on B_N of the walled-in blocks: `pick`), each with an
`Optimizer`, a Strategy in the rows' identity (relax_rows/<metric>/<its name>/D0.05/):
- `Simp`: relax.one under a plan. Under W the plan names no eps world (W_PLAN, SIMP's default
  without `.e` and `s`, which name T's worlds of a standing building at eps; gap_walls'
  relaxation is W itself at every 0/1 clearing), its polish scored exactly (`E`) on a core as T's
  was, `.f0`; PENALTY_PLAN the same with a steeper penalty (q to 5: stage 3's ruling).
- `Greedy`: walls_greedy's greedy over W's exact gains (`Gx` every candidate re-scored each step,
  `GxL` lazily, `GxLr<radius_m>` lazily with its neighbours re-scored after each step).
The arms by label (ARMS): stage 2's S10 and B (SIMP's default); stage 3's pilot S10p (the
steeper penalty), S10g (GxL), S10l (GxLr10), S10G (Gx, on the pilot's blocks of at most
PILOT_FULL_N buildings: lazy against full), and its fit arms S10x and Bx (FIT, the pilot's
pick). A study (STUDIES: stage2,
pilot, fit) is the arms it scores and the pairs it reports, its rows in its own directory.

    S=research/roadless/leaky_simp.py      # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S pick                         # ROWS/b_blocks.csv (before any result)
    uv run python $S pilot-pick                   # ROWS/pilot_blocks.csv (before any result)
    uv run python $S arm <id> <label> [--replace]  # the arm's row, with host, wall, peak memory
    uv run python $S lenses <study> <id> [--replace]  # <study rows>/{lenses,facts}/<id>.parquet
    uv run python $S homes <study> <id> [--replace]   # <study rows>/homes/<id>.parquet
    uv run python $S report <study>               # <study rows>/summary_{blocks,groups,pairs}.csv

`pick`: diode B's blocks, chosen before any result by a stated rule: the walled-in small blocks
(leaky_rows/blocks.csv, not controls) ordered by their walled-in homes at baseline under RUNTIME
(uniform_rows/summary_blocks.csv's n_walled) and then by id, split into B_STRATA strata of
consecutive blocks (numpy.array_split), B_N / B_STRATA drawn from each without replacement by
numpy.random.default_rng(PICK_SEED). `pilot-pick`: stage 3's pilot by the same rule, PILOT_N blocks
from PILOT_STRATA strata by default_rng(PILOT_SEED), the penalty's tuning blocks (TUNING) left
out.

`arm`: the arm's run (its optimizer under its metric), its row with the host, the wall time and the
peak memory.

`lenses`: every arm's clearing of the study (and none) scored under T's metric, S10's and, on B's
blocks where the study has a B arm, B's: Lens A of P, P_{1/2} and J_2 on each metric's own baseline
(L_P, L_Pq, L_J2: road_rescore's forms) and B (i), the homes and population share left walled in by
T's street reach (W's B (i) is its base's, checked equal clearing by clearing); per block the D
each arm used, the clearings' Jaccard, each arm's wall time, its optimizer's own and its last
iterate's grey share (a greedy's 0).

`homes`: per home walled in at baseline, whether each arm's clearing leaves it walled in (T's
street reach) and its situation by U's placement (leaky.py `placement`: in a pocket, or with no open
ground within ring_m).

`report`: per block (summary_blocks.csv); per group (summary_groups.csv: the walled-in blocks, the
controls, all, and B's blocks) the people each arm leaves walled in (and of them the homes with no
open ground within ring_m), D, Jaccard, time and grey share; per pair of arms under each metric
(summary_pairs.csv) each lens's paired difference with its mean, median, 95% Bayesian-bootstrap
interval (the consolidated reports': BB_DRAWS Dirichlet weights) and the blocks ahead and behind.

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/leaky_simp.py
"""
from __future__ import annotations

import gc as pygc
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_calib  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_walls as gw  # noqa: E402
import metrics  # noqa: E402
import relax  # noqa: E402
import road_rescore as rr  # noqa: E402
import search  # noqa: E402
import walls_greedy as wg  # noqa: E402

ROWS = HERE / "leaky_rows" / "simp"
BLOCKS = HERE / "leaky_rows" / "blocks.csv"     # block, size, control (stage 1's 82)
WALLED = HERE / "uniform_rows" / "summary_blocks.csv"   # block, n_walled (RUNTIME's, at baseline)
B_FILE = ROWS / "b_blocks.csv"
PILOT_FILE = ROWS / "pilot_blocks.csv"
BUDGET = relax.D_LENS                # area D 0.05
POWER = 1.0                          # P
POP = common.PopulationDemand()
B_N = 20                             # diode B's blocks (the owner: "diode B on 20")
B_STRATA = 4
PICK_SEED = 20261021
PILOT_N = 10                         # stage 3's pilot (the controller: "e.g. 10 of the 82")
PILOT_STRATA = 5
PILOT_SEED = 20261023
PILOT_FULL_N = 250                   # the full greedy (Gx) on the pilot's blocks of at most this
                                     # many buildings (its scores ~ 0.05 n^2: lazy against full)
TUNING = ("ZAF.9.1.2_1_3786", "ZAF.9.3.1_1_19570", "ZAF.9.3.1_1_18286")   # the penalty's
LENSES = ("L_P", "L_Pq", "L_J2")
BB_DRAWS = 4000                      # the Bayesian bootstrap's draws (consolidated._bb_interval's)
BB_SEED = 20261022
WALLS = {"S10": "Wk1l1000",          # the owner's default walls (2026-10-21)
         "B": "Wk1o10i3000n64"}      # diode B
# SIMP's default under W: relax.SIMP_DEFAULT without its eps worlds (`.e0.0001`, `s1e-06`: T's), the
# polish scored exactly (W's discrete scorer), no free first
W_PLAN = "fw0.q3.i10.k1.p256w8x2E.f0"
# ... with a steeper penalty, SIMP's continuation to q 5 (stage 3's ruling on TUNING: as default's
# W L_P, half its grey; relax's Heaviside projection .b1-64 near-binary but 0.007 below: NOTES)
PENALTY_PLAN = "fw0.q5.i10.k1.p256w8x2E.f0"
FIT = "GxLr10"                       # stage 3's pick on the pilot (NOTES)
# the walls arms by label: (walls, optimizer's name)
ARMS = {"S10": ("S10", W_PLAN), "B": ("B", W_PLAN), "S10p": ("S10", PENALTY_PLAN),
        "S10g": ("S10", "GxL"), "S10l": ("S10", "GxLr10"), "S10G": ("S10", "Gx"),
        "S10x": ("S10", FIT), "Bx": ("B", FIT)}


# ------------------------------------------------------------------ the optimizers

@dataclass(frozen=True)
class Simp:
    """SIMP under a plan (relax.one): its row relax's, its clearing `simp_cleared`."""
    plan: relax.Plan

    @property
    def name(self) -> str:
        return self.plan.name

    def run(self, bid: str, metric: gw.WallMetric) -> dict[str, object]:
        return relax.one(bid, POWER, metric, self.plan, BUDGET, demand=POP)

    def cleared(self, row: pd.Series) -> NDArray[np.int64]:
        return np.asarray(row["simp_cleared"], dtype=np.int64)

    def grey(self, row: pd.Series) -> float:
        return float(row["simp_grey"])


@dataclass(frozen=True)
class Greedy:
    """A greedy over W's exact gains (walls_greedy): its row the steps', its clearing `cleared`
    in pick order."""
    rule: wg.Full | wg.Lazy | wg.LazyLocal

    @property
    def name(self) -> str:
        return self.rule.name

    def run(self, bid: str, metric: gw.WallMetric) -> dict[str, object]:
        [b] = common.build_blocks([bid])
        t0 = time.time()
        wm = metric.clearing(b, demand=POP)
        res = wg.run(wm, self.rule, budget=BUDGET, power=POWER)
        x = np.zeros(wm.n, dtype=bool)
        order = [s.building for s in res.steps]
        x[order] = True
        J = res.steps[-1].J if res.steps else res.J0
        print(f"{bid} n={wm.n} {metric.name} {self.name}: {len(order)} cleared, L_P "
              f"{1.0 - J / res.J0:.4f} (D {float(wm.cost[x].sum()):.4f}), {res.scores} scores; "
              f"{time.time() - t0:.0f}s", flush=True)
        return dict(block=bid, n=wm.n, power=POWER, demand=POP.name, plan=self.name,
                    along=metric.name, device=metric.device, cleared=order,
                    gains=[s.gain for s in res.steps], ratios=[s.ratio for s in res.steps],
                    J0=res.J0, J=J, perm=float(1.0 - (J / res.J0) ** (1.0 / POWER)),
                    D=float(wm.cost[x].sum()), scores=res.scores,
                    enclosed=wm.enclosed_left(x), t=time.time() - t0)

    def cleared(self, row: pd.Series) -> NDArray[np.int64]:
        return np.asarray(row["cleared"], dtype=np.int64)

    def grey(self, row: pd.Series) -> float:
        return 0.0                  # a 0/1 clearing throughout


def optimizer_of(name: str) -> Simp | Greedy:
    """The optimizer a name spells: a greedy's (walls_greedy's, Gx...) or a relax plan's."""
    if name.startswith("Gx"):
        return Greedy(wg.greedy_of(name))
    plan = relax.plan_of(name)
    if plan.name != name:
        raise ValueError(f"the plan {name!r} spells {plan.name!r}")
    return Simp(plan)


# ------------------------------------------------------------------ the arms

class TArm(NamedTuple):
    """T's arm on a device: its label, its metric and its optimizer (SIMP: stored rows)."""
    label: str
    metric: gcl.ComplexMetric
    opt: Simp


class WArm(NamedTuple):
    """A walls arm on a core: its label, its metric and its optimizer."""
    label: str
    walls: str
    metric: gw.WallMetric
    opt: Simp | Greedy


def t_arm(size: str) -> TArm:
    """T's arm as the uniform study ran a block of `size`: small blocks on a core (its polish
    scored exactly), large on a GPU (uniform.arms_on; the stored rows')."""
    plan_t = relax.plan_of(relax.SIMP_DEFAULT)
    if size == "small":
        return TArm("T", metrics.metric_of(metrics.RUNTIME, device="cpu"),
                    Simp(plan_t._replace(polish_scoring=search.polish_scoring_of("E"))))
    if size == "large":
        return TArm("T", metrics.metric_of(metrics.RUNTIME, device="gpu"), Simp(plan_t))
    raise ValueError(f"no stored T arm for a {size!r} block")


def w_arm(label: str) -> WArm:
    walls, opt = ARMS[label]
    return WArm(label, walls, metrics.walls_on_uniform(WALLS[walls], device="cpu"),
                optimizer_of(opt))


def arm_path(arm: TArm | WArm, bid: str) -> Path:
    """Where relax.rows_of puts a row: relax_rows/<metric>/<optimizer>/D<budget>/<id>_p<power>."""
    return relax.OUT / arm.metric.name / arm.opt.name / f"D{BUDGET:g}" / f"{bid}_p{POWER:g}.parquet"


def sizes() -> pd.Series:
    return pd.read_csv(BLOCKS).set_index("block")["size"]


class Study(NamedTuple):
    """The arms a study scores (labels: T's and ARMS'), the pairs it reports (first less second)
    and its rows' directory."""
    rows: Path
    labels: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]


STUDIES = {
    "stage2": Study(rows=ROWS, labels=("T", "S10", "B"),
                    pairs=(("S10", "T"), ("B", "T"), ("S10", "B"))),
    "pilot": Study(rows=ROWS / "pilot", labels=("T", "S10", "S10p", "S10g", "S10l", "S10G"),
                   pairs=(("S10p", "S10"), ("S10g", "S10"), ("S10g", "S10p"), ("S10g", "S10G"),
                          ("S10l", "S10G"), ("S10l", "S10g"), ("S10l", "S10"), ("S10g", "T"),
                          ("S10l", "T"))),
    "fit": Study(rows=ROWS / "fit", labels=("T", "S10", "B", "S10x", "Bx"),
                 pairs=(("S10x", "Bx"), ("S10x", "T"), ("Bx", "T"), ("S10x", "S10"),
                        ("Bx", "B"))),
}


def arms_on(study: Study, bid: str) -> dict[str, TArm | WArm]:
    """The study's arms on `bid` whose rows stand or must: a B arm on B's blocks only, the full
    greedy on blocks of at most PILOT_FULL_N buildings only."""
    out: dict[str, TArm | WArm] = {}
    for label in study.labels:
        if label == "T":
            out[label] = t_arm(str(sizes()[bid]))
            continue
        arm = w_arm(label)
        if arm.walls == "B" and bid not in b_blocks():
            continue
        if isinstance(arm.opt, Greedy) and isinstance(arm.opt.rule, wg.Full) and \
                int(pd.read_parquet(arm_path(out["T"], bid)).iloc[0]["n"]) > PILOT_FULL_N:
            continue
        out[label] = arm
    return out


# ------------------------------------------------------------------ the blocks

def _stratified(path: Path, n: int, strata: int, seed: int, leave_out: tuple[str, ...]) -> None:
    """`n` of the walled-in small blocks (less `leave_out`) by the module docstring's rule, to
    `path` (refused if it stands)."""
    clearings.check_new(path, replace=False)
    blocks = pd.read_csv(BLOCKS)
    walled = pd.read_csv(WALLED).set_index("block")["n_walled"]
    ids = blocks.block[(blocks["size"] == "small") & ~blocks["control"]
                       & ~blocks.block.isin(leave_out)].to_numpy()
    cand = pd.DataFrame(dict(block=ids, walled=walled.reindex(ids).to_numpy()))
    if cand.walled.isna().any() or (cand.walled < 1).any():
        raise RuntimeError(f"a walled-in block with no walled-in homes in {WALLED.name}")
    cand = cand.sort_values(by=["walled", "block"]).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out = []
    for k, idx in enumerate(np.array_split(np.arange(len(cand)), strata)):
        take = np.sort(rng.choice(idx, n // strata, replace=False))
        out.append(cand.loc[take, ["block", "walled"]].assign(
            stratum=k, stratum_walled_lo=int(cand.walled[idx].min()),
            stratum_walled_hi=int(cand.walled[idx].max()), stratum_blocks=len(idx)))
    df = pd.concat(out, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(df.to_string(), flush=True)


def b_blocks() -> set[str]:
    return set(pd.read_csv(B_FILE).block)


# ------------------------------------------------------------------ an arm's run

def run_arm(arm: WArm, bid: str, *, replace: bool) -> None:
    """The arm's run on `bid` (its optimizer), its row with the host, wall time and peak memory."""
    path = arm_path(arm, bid)
    clearings.check_new(path, replace=replace)
    pygc.collect()
    t0 = time.time()
    row = arm.opt.run(bid, arm.metric)
    row.update(arm=arm.label, host=socket.gethostname(), peak_rss_gb=gap_calib.peak_gb(),
               wall=time.time() - t0)
    clearings.write_once(pd.DataFrame([row]), path, replace=replace)


def _row(arm: TArm | WArm, bid: str) -> pd.Series:
    return pd.read_parquet(arm_path(arm, bid)).iloc[0]


def _cleared(arm: TArm | WArm, bid: str, n: int) -> NDArray[np.bool_]:
    r = _row(arm, bid)
    if int(r["n"]) != n:
        raise RuntimeError(f"{arm_path(arm, bid)}: a clearing of {r['n']} buildings, the block "
                           f"has {n}")
    x = np.zeros(n, dtype=bool)
    x[arm.opt.cleared(r)] = True
    return x


# ------------------------------------------------------------------ every clearing scored

class View(NamedTuple):
    """One metric's view of a block's clearings: its lens rows; per clearing the homes its B (i)
    leaves walled in (the street's reach through open ground: W's its base's, checked against
    T's), the population share they hold and the clearing's D."""
    rows: list[dict[str, object]]
    left: dict[str, NDArray[np.bool_]]
    enclosed: dict[str, float]
    D: dict[str, float]


def _t_view(bid: str, metric: gcl.ComplexMetric, xs: dict[str, NDArray[np.bool_]]) -> View:
    pygc.collect()
    [b] = common.build_blocks([bid])
    m = metric.clearing(b, demand=POP)
    rows, left, enclosed, D = [], {}, {}, {}
    for label, x in xs.items():
        L = m.base if not x.any() else m.lenses(x)
        D[label] = float(m.cost[x].sum())
        rows.append(dict(block=bid, scorer="T", clearing=label, n_cleared=int(x.sum()),
                         D=D[label], **rr._lens_row(m, L)))
        left[label], enclosed[label] = m.walled0 & ~L.live, L.enclosed
    return View(rows=rows, left=left, enclosed=enclosed, D=D)


def _w_view(bid: str, label: str, metric: gw.WallMetric,
            xs: dict[str, NDArray[np.bool_]]) -> View:
    pygc.collect()
    [b] = common.build_blocks([bid])
    wm = metric.clearing(b, demand=POP)
    base = wm.base_.lenses
    rows, left, enclosed, D = [], {}, {}, {}
    for clearing, x in xs.items():
        L = base if not x.any() else wm.lenses(x).lenses
        D[clearing] = float(wm.cost[x].sum())
        rows.append(dict(block=bid, scorer=label, clearing=clearing, n_cleared=int(x.sum()),
                         D=D[clearing], L_P=1.0 - L.P / base.P,
                         L_Pq=1.0 - L.Pq / base.Pq, L_J2=1.0 - (L.J2 / base.J2) ** 0.5,
                         enclosed=L.enclosed, n_enclosed=L.n_enclosed, unknowns=L.unknowns))
        left[clearing], enclosed[clearing] = wm.walled0 & ~L.live, L.enclosed
    return View(rows=rows, left=left, enclosed=enclosed, D=D)


def lenses(study: Study, bid: str, *, replace: bool) -> None:
    """Every arm's clearing of the study on `bid` under every metric (module docstring)."""
    outs = study.rows / "lenses" / f"{bid}.parquet", study.rows / "facts" / f"{bid}.parquet"
    for o in outs:
        clearings.check_new(o, replace=replace)
    t0 = time.time()
    run = arms_on(study, bid)
    at = run["T"]
    rT = _row(at, bid)
    n = int(rT["n"])
    xs = {"none": np.zeros(n, dtype=bool), **{k: _cleared(arm, bid, n) for k, arm in run.items()}}
    w_arms = [a for a in run.values() if isinstance(a, WArm)]
    views = [_t_view(bid, t_arm("small").metric, xs)]
    for walls in WALLS:
        on = [a for a in w_arms if a.walls == walls]
        if on:
            views.append(_w_view(bid, walls, on[0].metric, xs))
    for v in views[1:]:
        for k in xs:
            if not np.array_equal(v.left[k], views[0].left[k]):
                raise RuntimeError(f"{bid}: W's B (i) and T's part on the homes {k!r} leaves "
                                   "walled in")
    vT = views[0]
    left = vT.left
    facts: dict[str, object] = dict(block=bid, n=n, with_b=bid in b_blocks(),
                                    n_walled=int(left["none"].sum()), device_T=str(rT["device"]))
    for k, arm in run.items():
        r = _row(arm, bid)
        facts.update({f"metric_{k}": arm.metric.name, f"opt_{k}": arm.opt.name,
                      f"D_{k}": vT.D[k], f"left_{k}": int(left[k].sum()),
                      f"left_share_{k}": vT.enclosed[k], f"wall_{k}": float(r["wall"]),
                      f"t_{k}": float(r["t"]), f"peak_rss_gb_{k}": float(r["peak_rss_gb"]),
                      f"grey_{k}": arm.opt.grey(r)})
    keys = list(run)
    for i, p in enumerate(keys):
        for q in keys[i + 1:]:
            union = int((xs[p] | xs[q]).sum())
            facts[f"jaccard_{p}_{q}"] = float((xs[p] & xs[q]).sum() / union) if union else 1.0
    facts.update(host=socket.gethostname(), wall_lenses=time.time() - t0)
    rows = [r for v in views for r in v.rows]
    for o, df in zip(outs, (pd.DataFrame(rows), pd.DataFrame([facts])), strict=True):
        clearings.write_once(df, o, replace=replace)
    print(f"{bid}: {facts['n_walled']} walled in; left " + ", ".join(
        f"{k} {facts[f'left_{k}']}" for k in keys) + f"; lenses {time.time() - t0:.0f}s",
        flush=True)


def homes(study: Study, bid: str, *, replace: bool) -> None:
    """Per home walled in at baseline whether each arm's clearing leaves it walled in, by T's
    street reach, and its situation by U's placement (module docstring)."""
    out = study.rows / "homes" / f"{bid}.parquet"
    clearings.check_new(out, replace=replace)
    run = arms_on(study, bid)
    n = int(_row(run["T"], bid)["n"])
    [b] = common.build_blocks([bid])
    m = t_arm("small").metric.clearing(b, demand=POP)
    place = pd.read_parquet(HERE / "leaky_rows" / "placement" / f"{bid}.parquet")
    w = m.walled0
    df = pd.DataFrame(dict(block=bid, home=np.flatnonzero(w), share=m.cost[w],
                           pocket_U=place.pocket_U.to_numpy()[w],
                           noground_U=place.noground_U.to_numpy()[w]))
    for k, arm in run.items():
        x = _cleared(arm, bid, n)
        df[f"left_{k}"] = (w & ~m.live_homes_of(m.factors(m.k_of(x))))[w]
    clearings.write_once(df, out, replace=replace)
    print(f"{bid}: {int(w.sum())} walled in; left " + ", ".join(
        f"{k} {int(df[f'left_{k}'].sum())}" for k in run), flush=True)


# ------------------------------------------------------------------ the report

def _read(study: Study, kind: str) -> pd.DataFrame:
    paths = sorted((study.rows / kind).glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else \
        pd.DataFrame()


def _bb(d: NDArray[np.float64], rng: np.random.Generator) -> tuple[float, float]:
    """The 95% interval of the mean of `d` under the Bayesian bootstrap."""
    if len(d) == 0:
        return np.nan, np.nan
    wts = rng.dirichlet(np.ones(len(d)), size=BB_DRAWS)
    lo, hi = np.quantile(wts @ d, [0.025, 0.975])
    return float(lo), float(hi)


SCORERS = ("T", *WALLS)


def report(study: Study) -> None:
    blocks = pd.read_csv(BLOCKS).set_index("block")
    lens = _read(study, "lenses")
    blk = _read(study, "facts").set_index("block")
    blk["size"] = blocks["size"].reindex(blk.index)
    blk["control"] = blocks["control"].reindex(blk.index)
    wide = blk.copy()
    for scorer in SCORERS:
        for clearing in study.labels:
            sub = lens[(lens.scorer == scorer) & (lens.clearing == clearing)].set_index("block")
            for x in LENSES:
                wide[f"{x}_{scorer}_{clearing}"] = sub[x].reindex(wide.index)
    wide.reset_index().to_csv(study.rows / "summary_blocks.csv", index=False)
    groups = [("walled-in", wide[~wide.control]), ("controls", wide[wide.control]),
              ("all", wide), ("B's blocks", wide[wide.with_b.astype(bool)])]
    grows = []
    for name, g in groups:
        row: dict[str, object] = dict(group=name, blocks=len(g),
                                      homes_walled=int(g.n_walled.sum()))
        for k in study.labels:
            if f"left_{k}" not in g or g[f"left_{k}"].isna().all():
                continue
            h = g[g[f"left_{k}"].notna()]
            row.update({f"blocks_{k}": len(h), f"left_{k}": int(h[f"left_{k}"].sum()),
                        f"left_share_{k}": float(h[f"left_share_{k}"].sum()),
                        f"blocks_left_{k}": int((h[f"left_{k}"] > 0).sum()),
                        f"D_{k}_mean": float(h[f"D_{k}"].mean()),
                        f"wall_{k}_median": float(h[f"wall_{k}"].median()),
                        f"wall_{k}_sum_h": float(h[f"wall_{k}"].sum() / 3600.0),
                        f"t_{k}_median": float(h[f"t_{k}"].median()),
                        f"grey_{k}_median": float(h[f"grey_{k}"].median()),
                        f"peak_rss_gb_{k}_max": float(h[f"peak_rss_gb_{k}"].max())})
        for col in [c for c in g.columns if c.startswith("jaccard_")]:
            v = g[col].dropna()
            row.update({f"{col}_median": float(v.median()) if len(v) else np.nan,
                        f"{col}_min": float(v.min()) if len(v) else np.nan})
        grows.append(row)
    hm = _read(study, "homes")
    for row in grows if len(hm) else []:
        g = dict(groups)[str(row["group"])]
        h = hm[hm.block.isin(g.index)]
        row.update(walled_noground=int(h.noground_U.sum()), walled_pocket=int(h.pocket_U.sum()))
        for k in study.labels:
            if f"left_{k}" not in h:
                continue
            hk = h[h[f"left_{k}"].notna()]
            if len(hk):
                left = hk[f"left_{k}"].astype(bool)
                row.update({f"left_{k}_noground": int((left & hk.noground_U).sum()),
                            f"left_{k}_pocket": int((left & hk.pocket_U).sum())})
    pd.DataFrame(grows).to_csv(study.rows / "summary_groups.csv", index=False)
    rng = np.random.default_rng(BB_SEED)
    prows = []
    for name, g in groups:
        for first, second in study.pairs:
            for scorer in SCORERS:
                cols = [f"{x}_{scorer}_{c}" for x in LENSES for c in (first, second)]
                if any(c not in g for c in cols):
                    continue
                h = g.dropna(subset=cols)
                if h.empty:
                    continue
                row = dict(group=name, pair=f"{first} - {second}", scorer=scorer, blocks=len(h))
                for x in LENSES:
                    d = (h[f"{x}_{scorer}_{first}"] - h[f"{x}_{scorer}_{second}"]).to_numpy(
                        dtype=float)
                    lo, hi = _bb(d, rng)
                    row.update({f"{x}_first_mean": float(h[f"{x}_{scorer}_{first}"].mean()),
                                f"{x}_second_mean": float(h[f"{x}_{scorer}_{second}"].mean()),
                                f"d{x}_mean": float(d.mean()), f"d{x}_median": float(np.median(d)),
                                f"d{x}_lo": lo, f"d{x}_hi": hi,
                                f"d{x}_ahead": int((d > 1e-9).sum()),
                                f"d{x}_behind": int((d < -1e-9).sum())})
                prows.append(row)
    t = pd.DataFrame(prows)
    t.to_csv(study.rows / "summary_pairs.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(grows).round(4).T.to_string())
    print(t.round(4).to_string())


def main(argv: list[str]) -> None:
    argv, replace = clearings.replace_flag(argv)
    if argv == ["pick"]:
        _stratified(B_FILE, B_N, B_STRATA, PICK_SEED, ())
    elif argv == ["pilot-pick"]:
        _stratified(PILOT_FILE, PILOT_N, PILOT_STRATA, PILOT_SEED, TUNING)
    elif len(argv) == 3 and argv[0] == "arm" and argv[2] in ARMS:
        arm = w_arm(argv[2])
        if arm.walls == "B" and argv[1] not in b_blocks():
            raise SystemExit(f"{argv[1]} is not one of diode B's blocks ({B_FILE})")
        run_arm(arm, argv[1], replace=replace)
    elif len(argv) == 3 and argv[0] == "lenses" and argv[1] in STUDIES:
        lenses(STUDIES[argv[1]], argv[2], replace=replace)
    elif len(argv) == 3 and argv[0] == "homes" and argv[1] in STUDIES:
        homes(STUDIES[argv[1]], argv[2], replace=replace)
    elif len(argv) == 2 and argv[0] == "report" and argv[1] in STUDIES:
        report(STUDIES[argv[1]])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
