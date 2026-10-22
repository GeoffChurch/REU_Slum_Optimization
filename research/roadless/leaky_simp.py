"""Walls that leak, stage 2: SIMP under the walls against today's rule (the owner 2026-10-21: the
default walls model the symmetric 1,000 m, "They could both exist as options but we would
hopefully be able to default to the symmetric case if it's not worse"; the test "1,000 m on all,
diode B on 20"; NOTES, "Walls that leak", stage 2; the brief
.superpowers/sdd/2026-10-21-leaky-walls/brief-stage2.md).

Three arms at area D BUDGET, SIMP's default optimizing P (relax.one): T, today's (RUNTIME, free
first, relax.SIMP_DEFAULT: the uniform study's stored arm, uniform.py, at the same RUNTIME; its
polish scored exactly on a core, in the eps world on a GPU, the large controls); S10, SIMP's
default under the symmetric walls Wk1l1000 (gap_walls on UNIFORM's mesh, leaks and doors:
metrics.walls_on_uniform), no free first; B, the same under diode B (Wk1o10i3000n64) on B_N of the
walled-in blocks (`pick`). Under W the plan names no eps world (W_PLAN: SIMP's default without `.e`
and `s`, which name T's worlds of a standing building at eps; gap_walls.WallRelaxation is W
itself at every 0/1 clearing), its polish scored exactly (`E`) on a core as T's was, `.f0`.

    S=research/roadless/leaky_simp.py      # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S pick                        # ROWS/b_blocks.csv (before any result)
    uv run python $S arm <id> <S10|B> [--replace] # relax.one's row, with host, wall, peak memory
    uv run python $S lenses <id> [--replace]     # ROWS/{lenses,facts}/<id>.parquet
    uv run python $S homes <id> [--replace]      # ROWS/homes/<id>.parquet
    uv run python $S report                      # ROWS/summary_{blocks,groups,pairs}.csv

`pick`: diode B's blocks, chosen before any result by a stated rule: the walled-in small blocks
(leaky_rows/blocks.csv, not controls) ordered by their walled-in homes at baseline under RUNTIME
(uniform_rows/summary_blocks.csv's n_walled) and then by id, split into B_STRATA strata of
consecutive blocks (numpy.array_split), B_N / B_STRATA drawn from each without replacement by
numpy.random.default_rng(PICK_SEED).

`arm`: the arm's SIMP run (relax.one under its metric and plan), its row where relax.rows_of puts
it (relax_rows/<metric>/<plan>/D0.05/<id>_p1.parquet), with the host, the wall time and the peak
memory.

`lenses`: every arm's clearing (and none) scored under T's metric, S10's and, on B's blocks, B's:
Lens A of P, P_{1/2} and J_2 on each metric's own baseline (L_P, L_Pq, L_J2: road_rescore's
forms) and B (i), the homes and population share left walled in by T's street reach (W's B (i)
is its base's, checked equal clearing by clearing); per block the D each arm used, the clearings'
Jaccard, each arm's wall time and SIMP's own.

`homes`: per home walled in at baseline, whether each arm's clearing leaves it walled in (T's
street reach) and its situation by U's placement (leaky.py `placement`: in a pocket, or with no open
ground within ring_m).

`report`: per block (summary_blocks.csv); per group (summary_groups.csv: the walled-in blocks, the
controls, all, and B's blocks) the people each arm leaves walled in (and of them the homes with no
open ground within ring_m), D, Jaccard, time and SIMP's last iterate's grey share; per pair
of arms under each metric (summary_pairs.csv) each lens's paired difference with its mean, median,
95% Bayesian-bootstrap interval (the consolidated reports': BB_DRAWS Dirichlet weights) and the
blocks ahead and behind.

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/leaky_simp.py
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

ROWS = HERE / "leaky_rows" / "simp"
BLOCKS = HERE / "leaky_rows" / "blocks.csv"     # block, size, control (stage 1's 82)
WALLED = HERE / "uniform_rows" / "summary_blocks.csv"   # block, n_walled (RUNTIME's, at baseline)
B_FILE = ROWS / "b_blocks.csv"
BUDGET = relax.D_LENS                # area D 0.05
POWER = 1.0                          # P
POP = common.PopulationDemand()
B_N = 20                             # diode B's blocks (the owner: "diode B on 20")
B_STRATA = 4
PICK_SEED = 20261021
LENSES = ("L_P", "L_Pq", "L_J2")
BB_DRAWS = 4000                      # the Bayesian bootstrap's draws (consolidated._bb_interval's)
BB_SEED = 20261022
S10 = "Wk1l1000"                     # the owner's default walls (2026-10-21)
DIODE_B = "Wk1o10i3000n64"
# SIMP's default under W: relax.SIMP_DEFAULT without its eps worlds (`.e0.0001`, `s1e-06`: T's), the
# polish scored exactly (W's discrete scorer), no free first
W_PLAN = "fw0.q3.i10.k1.p256w8x2E.f0"


class TArm(NamedTuple):
    """T's arm on a device: its label, its metric and its plan."""
    label: str
    metric: gcl.ComplexMetric
    plan: relax.Plan


class WArm(NamedTuple):
    """A walls arm on a core: its label, its metric and its plan."""
    label: str
    metric: gw.WallMetric
    plan: relax.Plan


class Arms(NamedTuple):
    t_cpu: TArm
    t_gpu: TArm
    s10: WArm
    b: WArm


def arms() -> Arms:
    """T (as the uniform study ran it on each device: uniform.arms_on), S10 and B, on a core."""
    plan_t = relax.plan_of(relax.SIMP_DEFAULT)
    w_plan = relax.plan_of(W_PLAN)
    if w_plan.name != W_PLAN:
        raise ValueError(f"W's plan {W_PLAN!r} spells {w_plan.name!r}")
    return Arms(
        t_cpu=TArm("T", metrics.metric_of(metrics.RUNTIME, device="cpu"),
                   plan_t._replace(polish_scoring=search.polish_scoring_of("E"))),
        t_gpu=TArm("T", metrics.metric_of(metrics.RUNTIME, device="gpu"), plan_t),
        s10=WArm("S10", metrics.walls_on_uniform(S10, device="cpu"), w_plan),
        b=WArm("B", metrics.walls_on_uniform(DIODE_B, device="cpu"), w_plan))


def arm_path(arm: TArm | WArm, bid: str) -> Path:
    return relax.rows_of(arm.plan, arm.metric.name, BUDGET) / f"{bid}_p{POWER:g}.parquet"


def t_arm(a: Arms, size: str) -> TArm:
    """T's arm as the uniform study ran a block of `size`: small blocks on a core, large on a GPU
    (uniform.py's blocks; the stored rows')."""
    if size == "small":
        return a.t_cpu
    if size == "large":
        return a.t_gpu
    raise ValueError(f"no stored T arm for a {size!r} block")


def sizes() -> pd.Series:
    return pd.read_csv(BLOCKS).set_index("block")["size"]


# ------------------------------------------------------------------ diode B's blocks

def pick() -> None:
    """Diode B's blocks by the module docstring's rule (ROWS/b_blocks.csv), refused if it stands."""
    clearings.check_new(B_FILE, replace=False)
    blocks = pd.read_csv(BLOCKS)
    walled = pd.read_csv(WALLED).set_index("block")["n_walled"]
    ids = blocks.block[(blocks["size"] == "small") & ~blocks["control"]].to_numpy()
    cand = pd.DataFrame(dict(block=ids, walled=walled.reindex(ids).to_numpy()))
    if cand.walled.isna().any() or (cand.walled < 1).any():
        raise RuntimeError(f"a walled-in block with no walled-in homes in {WALLED.name}")
    cand = cand.sort_values(by=["walled", "block"]).reset_index(drop=True)
    rng = np.random.default_rng(PICK_SEED)
    out = []
    for k, idx in enumerate(np.array_split(np.arange(len(cand)), B_STRATA)):
        take = np.sort(rng.choice(idx, B_N // B_STRATA, replace=False))
        out.append(cand.loc[take, ["block", "walled"]].assign(
            stratum=k, stratum_walled_lo=int(cand.walled[idx].min()),
            stratum_walled_hi=int(cand.walled[idx].max()), stratum_blocks=len(idx)))
    df = pd.concat(out, ignore_index=True)
    ROWS.mkdir(parents=True, exist_ok=True)
    df.to_csv(B_FILE, index=False)
    print(df.to_string(), flush=True)


def b_blocks() -> set[str]:
    return set(pd.read_csv(B_FILE).block)


# ------------------------------------------------------------------ an arm's run

def run_arm(arm: WArm, bid: str, *, replace: bool) -> None:
    """The arm's SIMP run on `bid` (relax.one), its row with the host, wall time and peak memory."""
    path = arm_path(arm, bid)
    clearings.check_new(path, replace=replace)
    pygc.collect()
    t0 = time.time()
    row = relax.one(bid, POWER, arm.metric, arm.plan, BUDGET, demand=POP)
    row.update(arm=arm.label, host=socket.gethostname(), peak_rss_gb=gap_calib.peak_gb(),
               wall=time.time() - t0)
    clearings.write_once(pd.DataFrame([row]), path, replace=replace)


def _cleared(path: Path, n: int) -> NDArray[np.bool_]:
    r = pd.read_parquet(path).iloc[0]
    if int(r["n"]) != n:
        raise RuntimeError(f"{path}: a clearing of {r['n']} buildings, the block has {n}")
    x = np.zeros(n, dtype=bool)
    x[np.asarray(r["simp_cleared"], dtype=np.int64)] = True
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


def _outs(bid: str) -> tuple[Path, Path]:
    return ROWS / "lenses" / f"{bid}.parquet", ROWS / "facts" / f"{bid}.parquet"


def lenses(bid: str, *, replace: bool) -> None:
    """Every arm's clearing of `bid` under every arm's metric (module docstring)."""
    outs = _outs(bid)
    for o in outs:
        clearings.check_new(o, replace=replace)
    t0 = time.time()
    a = arms()
    at = t_arm(a, str(sizes()[bid]))
    with_b = bid in b_blocks()
    run: dict[str, TArm | WArm] = {"T": at, "S10": a.s10, **({"B": a.b} if with_b else {})}
    rT = pd.read_parquet(arm_path(at, bid)).iloc[0]
    n = int(rT["n"])
    xs = {"none": np.zeros(n, dtype=bool),
          **{k: _cleared(arm_path(arm, bid), n) for k, arm in run.items()}}
    views = [_t_view(bid, a.t_cpu.metric, xs), _w_view(bid, "S10", a.s10.metric, xs)]
    if with_b:
        views.append(_w_view(bid, "B", a.b.metric, xs))
    for v in views[1:]:
        for k in xs:
            if not np.array_equal(v.left[k], views[0].left[k]):
                raise RuntimeError(f"{bid}: W's B (i) and T's part on the homes {k!r} leaves "
                                   "walled in")
    vT = views[0]
    left = vT.left
    facts: dict[str, object] = dict(block=bid, n=n, with_b=with_b,
                                    n_walled=int(left["none"].sum()),
                                    metric_T=at.metric.name, plan_T=at.plan.name,
                                    metric_S10=a.s10.metric.name, metric_B=a.b.metric.name,
                                    plan_W=a.s10.plan.name, device_T=str(rT["device"]))
    for k, arm in run.items():
        r = pd.read_parquet(arm_path(arm, bid)).iloc[0]
        facts.update({f"D_{k}": vT.D[k], f"left_{k}": int(left[k].sum()),
                      f"left_share_{k}": vT.enclosed[k],
                      f"wall_{k}": float(r["wall"]), f"t_{k}": float(r["t"]),
                      f"peak_rss_gb_{k}": float(r["peak_rss_gb"])})
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


def homes(bid: str, *, replace: bool) -> None:
    """Per home walled in at baseline whether each arm's clearing leaves it walled in, by T's
    street reach, and its situation by U's placement (module docstring)."""
    out = ROWS / "homes" / f"{bid}.parquet"
    clearings.check_new(out, replace=replace)
    a = arms()
    at = t_arm(a, str(sizes()[bid]))
    run: dict[str, TArm | WArm] = {"T": at, "S10": a.s10,
                                   **({"B": a.b} if bid in b_blocks() else {})}
    n = int(pd.read_parquet(arm_path(at, bid)).iloc[0]["n"])
    [b] = common.build_blocks([bid])
    m = a.t_cpu.metric.clearing(b, demand=POP)
    place = pd.read_parquet(HERE / "leaky_rows" / "placement" / f"{bid}.parquet")
    w = m.walled0
    df = pd.DataFrame(dict(block=bid, home=np.flatnonzero(w), share=m.cost[w],
                           pocket_U=place.pocket_U.to_numpy()[w],
                           noground_U=place.noground_U.to_numpy()[w]))
    for k, arm in run.items():
        x = _cleared(arm_path(arm, bid), n)
        df[f"left_{k}"] = (w & ~m.live_homes_of(m.factors(m.k_of(x))))[w]
    clearings.write_once(df, out, replace=replace)
    print(f"{bid}: {int(w.sum())} walled in; left " + ", ".join(
        f"{k} {int(df[f'left_{k}'].sum())}" for k in run), flush=True)


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


# the pairs compared (first less second), under each scorer that scores both
PAIRS = (("S10", "T"), ("B", "T"), ("S10", "B"))
SCORERS = ("T", "S10", "B")


def report() -> None:
    blocks = pd.read_csv(BLOCKS).set_index("block")
    lens = _read("lenses")
    blk = _read("facts").set_index("block")
    blk["size"] = blocks["size"].reindex(blk.index)
    blk["control"] = blocks["control"].reindex(blk.index)
    # each arm's last iterate's grey share (relax.one's simp_grey: x in (0.05, 0.95))
    a = arms()
    for bid in blk.index:
        run: dict[str, TArm | WArm] = {"T": t_arm(a, str(blk.at[bid, "size"])), "S10": a.s10,
                                       **({"B": a.b} if bool(blk.at[bid, "with_b"]) else {})}
        for k, arm in run.items():
            blk.loc[bid, f"grey_{k}"] = float(pd.read_parquet(arm_path(arm, bid)).iloc[0][
                "simp_grey"])
    wide = blk.copy()
    for scorer in SCORERS:
        for clearing in ("T", "S10", "B"):
            sub = lens[(lens.scorer == scorer) & (lens.clearing == clearing)].set_index("block")
            for x in LENSES:
                wide[f"{x}_{scorer}_{clearing}"] = sub[x].reindex(wide.index)
    wide.reset_index().to_csv(ROWS / "summary_blocks.csv", index=False)
    groups = [("walled-in", wide[~wide.control]), ("controls", wide[wide.control]),
              ("all", wide), ("B's blocks", wide[wide.with_b.astype(bool)])]
    grows = []
    for name, g in groups:
        row: dict[str, object] = dict(group=name, blocks=len(g),
                                      homes_walled=int(g.n_walled.sum()))
        for k in ("T", "S10", "B"):
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
    hm = _read("homes")
    for row in grows:
        g = dict(groups)[str(row["group"])]
        h = hm[hm.block.isin(g.index)]
        row.update(walled_noground=int(h.noground_U.sum()), walled_pocket=int(h.pocket_U.sum()))
        for k in ("T", "S10", "B"):
            hk = h[h[f"left_{k}"].notna()]
            if len(hk):
                left = hk[f"left_{k}"].astype(bool)
                row.update({f"left_{k}_noground": int((left & hk.noground_U).sum()),
                            f"left_{k}_pocket": int((left & hk.pocket_U).sum())})
    pd.DataFrame(grows).to_csv(ROWS / "summary_groups.csv", index=False)
    rng = np.random.default_rng(BB_SEED)
    prows = []
    for name, g in groups:
        for first, second in PAIRS:
            for scorer in SCORERS:
                cols = [f"{x}_{scorer}_{c}" for x in LENSES for c in (first, second)]
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
    t.to_csv(ROWS / "summary_pairs.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(grows).round(4).T.to_string())
    print(t.round(4).to_string())


def main(argv: list[str]) -> None:
    argv, replace = clearings.replace_flag(argv)
    if argv == ["pick"]:
        pick()
    elif len(argv) == 3 and argv[0] == "arm" and argv[2] in ("S10", "B"):
        a = arms()
        bid = argv[1]
        if argv[2] == "B" and bid not in b_blocks():
            raise SystemExit(f"{bid} is not one of diode B's blocks ({B_FILE})")
        run_arm(a.s10 if argv[2] == "S10" else a.b, bid, replace=replace)
    elif len(argv) == 2 and argv[0] == "lenses":
        lenses(argv[1], replace=replace)
    elif len(argv) == 2 and argv[0] == "homes":
        homes(argv[1], replace=replace)
    elif argv == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
