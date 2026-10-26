"""Block 5810's building clearings at the build-out page's fifteen budgets (area D 0.01 to 0.15 by
0.01), four methods, each clearing scored and painted under metrics.RUNTIME as the page's road
frames are (NOTES, "Clearings on 5810 at fifteen budgets, for the build-out page (2026-10-26)"):

* simp: SIMP's default (relax.SIMP_DEFAULT) under RUNTIME, one run per budget (relax.py one, GPU);
* greedy: the cheap preset's greedy alone (polish_greedy.CHEAP's picker, under CHEAP.on(RUNTIME)
  as the cheap preset runs it), one nested run to D 0.15 (`greedy`), cut at each budget as
  polish_greedy.run cuts its greedy_perm (cut_of: the order to the end of the step a run to that
  budget stops at, then relax.cut_to_budget after free first's held crossings);
* cheap: the cheap preset, greedy and polish (polish_greedy.py under CHEAP.on(RUNTIME) on a core,
  its polish scored exactly: road_rescore's cx rule), one run per budget;
* ff: walled-in round 2's winner (walled.ARMS["Ff"]: SIMP under metrics.HOMES, free first as a
  track; NOTES, "Walled-in homes, round 2"), one run per budget (relax.py one, GPU), scored here
  under RUNTIME and under its own metric.

Each run at the presets' power (road_rescore.CX_POWER: P), but j2: SIMP's default at power 2
(J_2, the far homes weighted by their escape time: owner 2026-10-26, "let's try J_2 as the
objective to see if it clears further in"). Each clearing a state of the plain mesh
(no carving), the road frames' baseline model (compute.py's): its lenses (road_rescore._lens_row),
its current and power painted (compute.paint_state), and checked loudly (the baseline's P against
the page's, D from the cleared set against the optimizer's, the state's clearing against the
plates', the energy identity, the optimizer's own score against the rescoring; the greedy's cut at
each budget against the cheap preset's own greedy_perm clearing there, check_greedy). Then the
five plates per frame (render.py's: the clearing drawn as network_plate draws a destroyed
building, the current, the power, and their gains over the plain baseline on the diverging scale)
into render.PLATES, and clearings.json beside the page.

    S=research/roadless/atlas/roads5810/clear5810.py      # from the repo root, PYTHONPATH=.
    uv run python $S greedy              # the greedy alone to D 0.15 on a core: order and cuts
    uv run python $S frames <method>     # simp | greedy | cheap | ff: states, rasters, checks
    uv run python $S plates <method>     # the five views per frame
    uv run python $S page                # clearings.json

The optimizers' rows stand where relax.py and polish_greedy.py put them; the greedy's order, the
frames' rasters and the runs' logs (each under /usr/bin/time -v: wall clock and peak RSS) in
SCRATCH, whose progress.md has the commands.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple, Protocol

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ROADLESS = ROOT / "research" / "roadless"
sys.path.insert(0, str(ROADLESS))
import clear  # noqa: E402
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_factor  # noqa: E402
import metrics  # noqa: E402
import polish_greedy  # noqa: E402
import relax  # noqa: E402
import road_rescore as rr  # noqa: E402
import search  # noqa: E402
import walled  # noqa: E402

BID = "ZAF.9.3.1_1_5810"
TARGETS = tuple(round(0.01 * i, 2) for i in range(1, 16))     # the page's slider
CHECK_AT = 0.05            # the budget a run's rows path is shown at (`run`)
POWER = rr.CX_POWER        # the presets' objective: P (relax_rows' and polish_rows' `_p1`)
OWN_LENS = {1.0: "L_P", 2.0: "L_J2"}      # an objective's power -> its lens (Lens A's form)
POP = common.PopulationDemand()
SCRATCH = ROOT / ".superpowers" / "sdd" / "2026-10-26-clearings5810"
LOGS = SCRATCH / "logs"    # the queue's: <job>.log, <job>.time (/usr/bin/time -v)
OUT = HERE / "clearings.json"
VIEWS = ("clearing", "current", "power", "current_gain", "power_gain", "time")
P_TOL = 1e-9               # the baseline's P against the page's, relative
D_TOL = 1e-12              # D from the cleared set against the optimizer's
ENERGY_TOL = 1e-8          # summed power against P, relative (the road frames' within 5e-10)
OWN_TOL = 1e-7             # the optimizer's own L_P against the rescoring (the device's FP)


def load(name: str, path: Path) -> ModuleType:
    """The module at `path` under `name` (the site's scripts, read where they live, unedited)."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def tag(target: float) -> str:
    """A frame's name: the target in thousandths, three digits (render.tag's)."""
    return f"{int(round(target * 1000)):03d}"


# ------------------------------------------------------------------ the methods

class Cleared(NamedTuple):
    """A method's clearing at one budget as its optimizer left it."""
    ids: NDArray[np.int64]
    D: float          # the optimizer's D (its model's population share)
    own: float        # the optimizer's own score: L_P under its own metric
    t: float          # the optimizer's own time, s (its model's build not counted)
    job: str          # its run's name in the queue's logs (LOGS)


class Method(Protocol):
    """A clearing method at the page's budgets: its name on the page, how it works in a sentence,
    the metric it optimized under (`own`), its run's identity, its clearing at each target."""

    @property
    def key(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def how(self) -> str: ...

    @property
    def own(self) -> str: ...

    @property
    def power(self) -> float: ...

    def run(self) -> dict[str, str]: ...

    def at(self, target: float) -> Cleared: ...


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


@dataclasses.dataclass(frozen=True, kw_only=True)
class RelaxRun:
    """SIMP (relax.py one) at each budget at `power`, its rows relax.rows_of's."""
    key: str
    name: str
    how: str
    plan: str
    own: str
    power: float

    def rows(self, target: float) -> Path:
        return (relax.rows_of(relax.plan_of(self.plan), self.own, target)
                / f"{BID}_p{self.power:g}.parquet")

    def run(self) -> dict[str, str]:
        return dict(optimizer="relax.py one", plan=self.plan, metric=self.own,
                    rows=_rel(self.rows(CHECK_AT)).replace(f"D{CHECK_AT:g}", "D<target>"))

    def at(self, target: float) -> Cleared:
        r = pd.read_parquet(self.rows(target)).iloc[0]
        if (r["plan"], r["along"], float(r["power"])) != (self.plan, self.own, self.power):
            raise RuntimeError(f"{self.rows(target)}: plan {r['plan']}, metric {r['along']}, "
                               f"power {r['power']}")
        return Cleared(ids=np.asarray(r["simp_cleared"], dtype=np.int64), D=float(r["simp_D"]),
                       own=float(r["simp_perm"]), t=float(r["t"]), job=f"{self.key}_{tag(target)}")


@dataclasses.dataclass(frozen=True, kw_only=True)
class PolishRun:
    """The greedy and its polish (polish_greedy.py) at each budget under the preset's search of
    `own` (`search`: Preset.on's name), its rows polish_greedy.rows_of's."""
    key: str
    name: str
    how: str
    preset: polish_greedy.Preset
    scoring: search.PolishScoring
    own: str
    power: float
    search: str

    def rows(self, target: float) -> Path:
        p = self.preset
        return (polish_greedy.rows_of(p.picker, p.tries, p.width, p.pairs, self.scoring,
                                      self.search, target) / f"{BID}_p{self.power:g}.parquet")

    def run(self) -> dict[str, str]:
        return dict(optimizer="polish_greedy.py", preset=self.rows(CHECK_AT).parent.parent.name,
                    metric=self.search,
                    rows=_rel(self.rows(CHECK_AT)).replace(f"D{CHECK_AT:g}", "D<target>"))

    def row(self, target: float) -> pd.Series:
        return pd.read_parquet(self.rows(target)).iloc[0]

    def at(self, target: float) -> Cleared:
        r = self.row(target)
        return Cleared(ids=np.asarray(r["cleared"], dtype=np.int64), D=float(r["D"]),
                       own=float(r["perm"]), t=float(r["t"]), job=f"{self.key}_{tag(target)}")


GREEDY = SCRATCH / "greedy_order.json"     # not <key>.json: the frames write those


@dataclasses.dataclass(frozen=True, kw_only=True)
class GreedyCut:
    """The preset's greedy alone, one nested run to the last target (`greedy`), cut at each."""
    key: str
    name: str
    how: str
    preset: polish_greedy.Preset
    own: str
    power: float
    search: str

    def run(self) -> dict[str, str]:
        return dict(optimizer="clear5810.py greedy (polish_greedy.run's greedy, nested)",
                    picker=self.preset.picker, metric=self.search, rows=_rel(GREEDY))

    def at(self, target: float) -> Cleared:
        g = json.loads(GREEDY.read_text())
        [r] = [r for r in g["rows"] if r["target"] == target]
        return Cleared(ids=np.asarray(r["cleared"], dtype=np.int64), D=float(r["D"]),
                       own=float(r["greedy_perm"]), t=float(r["t"]), job="greedy")


def methods() -> dict[str, Method]:
    """The five, resolved here (the command line picks one by key)."""
    cheap = polish_greedy.CHEAP
    searched = cheap.on(metrics.metric_of(metrics.RUNTIME, device="cpu")).name
    ff = walled.ARMS["Ff"]
    out: list[Method] = [
        RelaxRun(key="simp", name="SIMP",
                 how="Relaxes every building to a fraction cleared, moves the fractions down P's "
                     "gradient within the budget as the penalty stiffens to whole buildings, "
                     "rounds, then polishes by exact swaps.",
                 plan=relax.SIMP_DEFAULT, own=metrics.RUNTIME, power=POWER),
        RelaxRun(key="j2", name="SIMP, far homes weighted (J₂)",
                 how="SIMP minimizing J₂, each home's escape time squared times its people, so a "
                     "home twice as far counts four times, not twice; the default's settings "
                     "otherwise. Scored here by P like the rest.",
                 plan=relax.SIMP_DEFAULT, own=metrics.RUNTIME, power=2.0),
        GreedyCut(key="greedy", name="Greedy",
                  how="Clears a batch at a time, each to the next 1% of the people, the "
                      "buildings whose clearing lowers P most per person to first order, spread "
                      "over different catchments; one run cut at each budget.",
                  preset=cheap, own=metrics.RUNTIME, power=POWER, search=searched),
        PolishRun(key="cheap", name="Greedy + polish",
                  how="The greedy to the budget, then up to 64 exact swaps of a cleared "
                      "building for a standing one, each the best of eight ranked by P's "
                      "gradient.",
                  preset=cheap, scoring=search.Exactly(), own=metrics.RUNTIME, power=POWER,
                  search=searched),
        RelaxRun(key="ff", name="SIMP, every home counted",
                 how="SIMP optimized under the rule that counts every walled-in home's trip "
                     "from its own ground, the better of starting from nothing and from the "
                     "cheapest crossings that free them; scored here under the page's metric.",
                 plan=ff.plan, own=walled.RULES[ff.rule], power=POWER),
    ]
    return {m.key: m for m in out}


# ------------------------------------------------------------------ the greedy, nested

class Steps:
    """search.Record of the greedy's steps: where each ends in the pick order, its D and its time
    since the greedy began (no state scored: polish_greedy's greedy records nothing)."""

    def __init__(self, *, t0: float) -> None:
        self.t0 = t0
        self.ends: list[int] = []
        self.D: list[float] = []
        self.t: list[float] = []

    def wants(self, s: search.SearchState) -> bool:
        return False

    def on_step(self, s: search.SearchState, o: search.Outcome) -> None:
        self.ends.append(len(s.order))
        self.D.append(s.D)
        self.t.append(time.time() - self.t0)


def crossing(step_D: list[float], budget: float) -> int:
    """The step a greedy run to `budget` stops after (search.Reached: the first whose D reaches
    it)."""
    for k, d in enumerate(step_D):
        if d >= budget - 1e-12:
            return k
    raise ValueError(f"the greedy never reached D {budget} (its last step {step_D[-1]})")


def cut_of(order: list[int], ends: list[int], step_D: list[float], held: NDArray[np.bool_],
           cost: NDArray[np.float64], budget: float) -> NDArray[np.bool_]:
    """polish_greedy.run's greedy_perm clearing at `budget` from a nested run: the pick order to
    the end of the step a run to `budget` stops after (a cut of the whole order would take later
    steps' buildings into what the crossing step leaves), cut by relax.cut_to_budget to what the
    held buildings leave, the held with it."""
    prefix = order[:ends[crossing(step_D, budget)]]
    left = budget - float(cost[held].sum())
    return (relax.cut_to_budget(prefix, cost, left) > 0) | held


def greedy() -> None:
    """The cheap preset's greedy from free first's held crossings to the last target, as
    polish_greedy.run runs it (one run: the trajectory to a budget is the nested run's to the
    step that reaches it, the batches aimed at the next 0.01 whatever the budget); per target its
    cut (cut_of) scored exactly as polish_greedy scores greedy_perm. Free first runs once, at the
    last target (each call rebuilds the complex: minutes on 5810); check_greedy holds its set
    against the cheap preset's at every target. Writes GREEDY."""
    t0 = time.time()
    preset = polish_greedy.CHEAP
    metric = preset.on(metrics.metric_of(metrics.RUNTIME, device="cpu"))
    [b] = common.build_blocks([BID])
    c = metric.clearing(b, demand=POP)
    t_model = time.time() - t0
    rnd = search.greedy_round(preset.picker)
    none = np.zeros(c.n, dtype=bool)
    ts = time.time()
    held = c.free_first(budget=TARGETS[-1], held=none)
    t_free = time.time() - ts
    c.removed[:] = held
    start = (search.Score(*c.exact(held, POWER)) if held.any()
             else search.Score(*c.baseline(POWER)))
    s = search.SearchState.start(c, power=POWER, score=start, world=search.EXACT)
    steps = Steps(t0=time.time())
    search.greedy(rnd, TARGETS[-1])(s, steps)
    t_greedy = steps.t[-1]
    c.removed[:] = False
    rel = c.relaxation(POWER, q=1.0, eps=clear.EPS)
    rows = []
    for t in TARGETS:
        k = crossing(steps.D, t)
        x = cut_of(s.order, steps.ends, steps.D, held, c.cost, t)
        rows.append(dict(target=t, step=k + 1, cleared=np.flatnonzero(x).tolist(),
                         D=float(c.cost[x].sum()),
                         greedy_perm=rel.perm(rel.exact(x.astype(np.float64))),
                         t=t_free + steps.t[k]))
        print(f"{BID} greedy D {t:.2f}: step {k + 1}, {int(x.sum())} buildings, D "
              f"{rows[-1]['D']:.5f}, L_P {rows[-1]['greedy_perm']:.4f}", flush=True)
    info = dict(block=BID, metric=metric.name, picker=preset.picker, power=POWER,
                held=np.flatnonzero(held).tolist(), order=[int(j) for j in s.order],
                ends=steps.ends, step_D=steps.D, step_t=steps.t, cost=c.cost.tolist(), rows=rows,
                t_model=t_model, t_free=t_free, t_greedy=t_greedy, t_total=time.time() - t0,
                peak_rss_gb=rr._status_gb("VmHWM"))
    GREEDY.write_text(json.dumps(info, indent=1))
    print(f"{BID} greedy: {len(steps.ends)} steps to D {steps.D[-1]:.4f} in {t_greedy:.0f}s, "
          f"peak {info['peak_rss_gb']:.1f} GB", flush=True)


def check_greedy(g: dict[str, Any], cheap: dict[float, pd.Series]) -> None:
    """The nested greedy's cut at every target is the clearing polish_greedy.run scored as its
    greedy_perm at that budget (`cheap`: the cheap preset's row per target, each its own run):
    the same held crossings, the same D, the same buildings as the stored cut, the same score.
    Raises otherwise."""
    cost = np.asarray(g["cost"])
    held = np.zeros(len(cost), dtype=bool)
    held[g["held"]] = True
    for t in TARGETS:
        r, row = [r for r in g["rows"] if r["target"] == t], cheap[t]
        if sorted(g["held"]) != sorted(int(j) for j in row["free_cleared"]):
            raise RuntimeError(f"D {t}: the greedy holds {g['held']}, the cheap preset "
                               f"{list(row['free_cleared'])}")
        x = cut_of(g["order"], g["ends"], g["step_D"], held, cost, t)
        D = float(cost[x].sum())
        if abs(D - float(row["greedy_D"])) > D_TOL:
            raise RuntimeError(f"the greedy's cut at D {t}: D {D!r}, the cheap preset's greedy "
                               f"{row['greedy_D']!r}")
        if len(r) != 1 or np.flatnonzero(x).tolist() != r[0]["cleared"]:
            raise RuntimeError(f"the greedy's stored cut at D {t} is not cut_of's")
        if abs(r[0]["greedy_perm"] - float(row["greedy_perm"])) > OWN_TOL:
            raise RuntimeError(f"the greedy's cut at D {t}: L_P {r[0]['greedy_perm']!r}, the "
                               f"cheap preset's greedy {row['greedy_perm']!r}")


def greedy_checked(ms: dict[str, Method]) -> None:
    cheap = ms["cheap"]
    if not isinstance(cheap, PolishRun):
        raise TypeError("the greedy is checked against the cheap preset's rows")
    check_greedy(json.loads(GREEDY.read_text()), {t: cheap.row(t) for t in TARGETS})


# ------------------------------------------------------------------ the frames

def frames(method: Method) -> None:
    """Per target the method's clearing as a state of the plain mesh under RUNTIME (the road
    frames' baseline model), its lenses, its current and power painted (compute.paint_state on
    compute.frame_of's frame), every check loud; for a method optimized under another metric, its
    lenses under that one too. Writes SCRATCH/<key>.npz (per frame the cleared set `x_<fff>`, the
    rasters `j_<fff>`, `p_<fff>` and each home's escape time `u_<fff>`, the Time plates', with the
    baseline's `u_base`) and SCRATCH/<key>.json."""
    cp = load("compute", HERE / "compute.py")
    t0 = time.time()
    base = json.loads((cp.DATA / "baseline.json").read_text())
    bnpz = np.load(cp.DATA / "baseline.npz", allow_pickle=True)       # the site's: WKB as objects
    metric = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.SuperLU())
    [b] = common.build_blocks([BID])
    m = metric.clearing(b, demand=POP)
    if abs(m.base.P - base["P"]) > P_TOL * base["P"]:
        raise RuntimeError(f"the baseline's P {m.base.P!r} is not the page's {base['P']!r}")
    own = (None if method.own == metrics.RUNTIME
           else metrics.metric_of(method.own, device="cpu").clearing(b, demand=POP))
    if own is not None and not np.array_equal(own.cost, m.cost):
        raise RuntimeError(f"{method.own}'s buildings are not RUNTIME's")
    t_model = time.time() - t0
    fr = cp.frame_of(m)
    if (not np.allclose(np.asarray(fr.frame, dtype=float), bnpz["frame"])
            or not np.allclose(np.asarray(fr.lim, dtype=float), bnpz["lim"])):
        raise RuntimeError("the frame is not the baseline plates'")
    if m.carving.n != m.n or len(shapely.from_wkb(bnpz["polys"])) != m.n:
        raise RuntimeError("the plain mesh's atoms are not its buildings, one each")
    W = common.POPULATIONS["area"].weights(np.asarray(b.buildings.outlines))
    share = W / W.sum()
    area = shapely.area(m.geo.polys)
    out: dict[str, NDArray[Any]] = dict(frame=bnpz["frame"], lim=bnpz["lim"], u_base=m.base.u)
    rows = []
    for target in TARGETS:
        ts = time.time()
        c = method.at(target)
        x = np.zeros(m.n, dtype=bool)
        x[c.ids] = True
        if int(x.sum()) != len(c.ids):
            raise RuntimeError(f"{method.key} {target}: a building cleared twice")
        D, D_block = float(m.cost[x].sum()), float(share[x].sum())
        if abs(D - c.D) > D_TOL or abs(D_block - c.D) > D_TOL:
            raise RuntimeError(f"{method.key} {target}: D {D!r} (the block's {D_block!r}) "
                               f"against the optimizer's {c.D!r}")
        st = m.state(m.k_of(x))
        if not np.array_equal(st.k, x.astype(np.float64)):
            raise RuntimeError(f"{method.key} {target}: the state is not the clearing drawn")
        L = m.lenses_of(st)
        s = cp.paint_state(m, st, fr)
        del st
        if max(abs(s["rel_err"]), abs(s["rel_err_tri"])) > ENERGY_TOL:
            raise RuntimeError(f"{method.key} {target}: summed power off P by {s['rel_err']!r} "
                               f"(cells), {s['rel_err_tri']!r} (triangles)")
        if abs(L.P - s["P_energy"]) > 1e-10 * L.P:
            raise RuntimeError(f"{method.key} {target}: the lens's P {L.P!r}, the flow's "
                               f"{s['P_energy']!r}")
        lens = rr._lens_row(m, L)
        own_lens = lens if own is None else rr._lens_row(own, own.lenses(x))
        own_score = own_lens[OWN_LENS[method.power]]
        if abs(own_score - c.own) > OWN_TOL:
            raise RuntimeError(f"{method.key} {target}: the optimizer's own score {c.own!r}, "
                               f"rescored {own_score!r} ({OWN_LENS[method.power]} under "
                               f"{method.own})")
        f = tag(target)
        rows.append(dict(
            f=f, target=target, D=D, D_opt=c.D, n=int(x.sum()), m2=float(area[x].sum()),
            P=L.P, LP=lens["L_P"], Pq=lens["L_Pq"], J2=lens["L_J2"], enclosed=lens["enclosed"],
            n_enclosed=lens["n_enclosed"], own_metric=method.own, own_LP=own_lens["L_P"],
            own_Pq=own_lens["L_Pq"], own_J2=own_lens["L_J2"], own_opt=c.own,
            own_diff=own_score - c.own, own_power=method.power, t_opt=c.t, job=c.job,
            **{k: s[k] for k in cp.SCALARS}, t_state=time.time() - ts))
        out[f"x_{f}"], out[f"j_{f}"], out[f"p_{f}"], out[f"u_{f}"] = x, s["j"], s["p"], L.u
        print(f"{method.key} {f}: {json.dumps(rows[-1])}", flush=True)
        del s
    np.savez(SCRATCH / f"{method.key}.npz", **out)
    info = dict(method=method.key, metric=metric.name, block=BID, run=method.run(),
                base=dict(P=m.base.P, Pq=m.base.Pq, J2=m.base.J2, page_P=base["P"]), rows=rows,
                t_model=t_model, t_total=time.time() - t0, peak_rss_gb=rr._status_gb("VmHWM"))
    (SCRATCH / f"{method.key}.json").write_text(json.dumps(info, indent=1))
    print(f"{method.key}: done in {time.time() - t0:.0f}s, peak {info['peak_rss_gb']:.1f} GB",
          flush=True)


# ------------------------------------------------------------------ the plates

def plates(method: Method) -> None:
    """Per frame the six views (VIEWS) into render.PLATES as `<key>_<fff>_<view>.webp`: the
    clearing (render.network_plate, each cleared building at c 1, no corridor), the current and
    the power (render.flow_plate on the plain baseline's log scales, the cleared outlined), their
    gains over the plain baseline (render.divgain on the diverging scale), and each home's escape
    time over the plain baseline's (render.time_ratio, the cleared in grey). The clearing drawn
    and the flow's are the same saved set (`x_<fff>`)."""
    rd = load("roads5810_render", HERE / "render.py")
    plates_dir: Path = rd.PLATES
    cp = rd.cp
    base = json.loads((cp.DATA / "baseline.json").read_text())
    bnpz = np.load(cp.DATA / "baseline.npz", allow_pickle=True)       # the site's: WKB as objects
    d = np.load(SCRATCH / f"{method.key}.npz")
    if not np.array_equal(d["frame"], bnpz["frame"]) or not np.array_equal(d["lim"], bnpz["lim"]):
        raise RuntimeError("the frames' extent is not the baseline plates'")
    fr = rd.frame_from(bnpz)
    polys = shapely.from_wkb(bnpz["polys"])
    [boundary] = shapely.from_wkb(bnpz["boundary"])
    [streets] = shapely.from_wkb(bnpz["streets"])
    jn, pn = rd.norms(base)
    for target in TARGETS:
        f = tag(target)
        x = d[f"x_{f}"].astype(np.float64)
        stem = f"{method.key}_{f}"
        rd.network_plate(fr, polys, x, None, boundary=boundary, streets=streets,
                         out=plates_dir / f"{stem}_clearing.{rd.EXT}")
        for view, key, norm in (("current", "j", jn), ("power", "p", pn)):
            img = d[f"{key}_{f}"]
            rd.flow_plate(img, norm, fr, polys, x, ramp=rd.rnd.RAMP, outline=True,
                          boundary=boundary, streets=streets,
                          out=plates_dir / f"{stem}_{view}.{rd.EXT}")
            rd.flow_plate(rd.divgain(img, bnpz[key]), rd.GAIN, fr, polys, x, ramp=rd.DIV,
                          outline=True, boundary=boundary, streets=streets,
                          out=plates_dir / f"{stem}_{view}_gain.{rd.EXT}")
        rd.time_plate(fr, polys, rd.time_ratio(d[f"u_{f}"], d["u_base"]), norm=rd.GAIN,
                      ramp=rd.DIV, gone=x > 0, corridor=None, boundary=boundary,
                      streets=streets, out=plates_dir / f"{stem}_time.{rd.EXT}")


# ------------------------------------------------------------------ the page's data

class Usage(NamedTuple):
    """A run's wall clock and peak resident memory (/usr/bin/time -v in LOGS)."""
    wall_s: float
    peak_rss_gb: float


def usage_of(job: str) -> Usage:
    text = (LOGS / f"{job}.time").read_text()
    if "Exit status: 0" not in text:
        raise RuntimeError(f"{job}: its run did not exit 0")
    wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    if wall is None or rss is None:
        raise RuntimeError(f"{job}: no wall clock or peak RSS in its time log")
    secs = sum(float(v) * 60 ** i for i, v in enumerate(reversed(wall[1].split(":"))))
    return Usage(wall_s=secs, peak_rss_gb=int(rss[1]) / 2 ** 20)


def page() -> None:
    """clearings.json: per method its key, display name, how it works, the run's identity and
    per target its row (target order), each frame's plates named."""
    ms = methods()
    greedy_checked(ms)
    base = json.loads((ROOT / ".superpowers/sdd/2026-10-22-site/roads5810/baseline.json")
                      .read_text())
    out: dict[str, Any] = dict(
        block=BID, metric=metrics.RUNTIME, power=POWER, targets=list(TARGETS),
        baseline=dict(P=base["P"], median_j=base["median_j"], median_p=base["median_p"]),
        views=list(VIEWS),
        note=("each clearing a state of the plain mesh (no carving), scored and drawn as the road "
              "frames are; D the cleared buildings' population share (area weights); LP, Pq, J2 "
              "the lenses on the plain baseline; own_LP the method's L_P under the metric it "
              "optimized; gains over the plain baseline"),
        methods={})
    for key, m in ms.items():
        info = json.loads((SCRATCH / f"{key}.json").read_text())
        rows = []
        for r in info["rows"]:
            u = usage_of(r["job"])
            rows.append(dict(
                f=r["f"], target=r["target"], D=r["D"], n=r["n"], m2=r["m2"], LP=r["LP"],
                Pq=r["Pq"], J2=r["J2"], P=r["P"], D_opt=r["D_opt"], own_metric=r["own_metric"],
                own_LP=r["own_LP"], own_opt=r["own_opt"], enclosed=r["enclosed"],
                n_enclosed=r["n_enclosed"], top5_power=r["top5_power"],
                top5_current=r["top5_current"], energy_rel_err=r["rel_err"],
                kirchhoff_max=r["kirchhoff_max"], t_opt=r["t_opt"], run=r["job"],
                run_wall_s=u.wall_s, run_peak_rss_gb=u.peak_rss_gb,
                files={v: f"{key}_{r['f']}_{v}.webp" for v in VIEWS}))
        if [r["target"] for r in rows] != list(TARGETS):
            raise RuntimeError(f"{key}: frames {[r['target'] for r in rows]}")
        out["methods"][key] = dict(key=key, name=m.name, how=m.how, power=m.power, run=m.run(),
                                   rows=rows)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(OUT, flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv == ["greedy"]:
        greedy()
    elif len(argv) == 2 and argv[0] in ("frames", "plates"):
        ms = methods()
        if argv[1] not in ms:
            raise SystemExit(f"{argv[1]!r}: one of {', '.join(ms)}")
        if argv[0] == "frames":
            if argv[1] == "greedy":
                greedy_checked(ms)
            frames(ms[argv[1]])
        else:
            plates(ms[argv[1]])
    elif argv == ["page"]:
        page()
    else:
        raise SystemExit(__doc__)
