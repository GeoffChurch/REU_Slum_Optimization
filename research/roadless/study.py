"""The lineup on the 220 recipients, scored roadless and by today's permeability, on the SAME
prefixes: the canonical street-first prefix at each displacement budget in BUDGETS (and the whole
network). Lens A is budget 0.10; Lens B is read afterwards from the curve (both metrics are
monotone along the prefix order).

    PYTHONPATH=. uv run python research/roadless/study.py run <workers> <h> <ell> [K] [pop]
    PYTHONPATH=. uv run python research/roadless/study.py report <h> <ell> [K] [pop]

pop: count (each building one unit) or area (proportional to footprint area): both the escape
demand and the displacement weights.
"""
from __future__ import annotations

import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import lifted  # noqa: E402

BUDGETS = (0.02, 0.05, 0.10, 0.15, 0.20)
_BLOCKS: list = []
_ARMS: dict = {}
_CFG: dict = {}


def rows_dir(h: float, ell: float, K: int, pop: str = "count") -> Path:
    return HERE / (f"rows_h{h:g}_ell{ell:g}_K{K}" + ("" if pop == "count" else f"_{pop}"))


def one(i: int) -> None:
    try:
        _one(i)
    except Exception as e:                       # log and move on: one bad block must not kill the pool
        print(f"{time.strftime('%H:%M:%S')} {_BLOCKS[i].block_id} FAILED "
              f"{type(e).__name__}: {e}"[:300], flush=True)


def _one(i: int) -> None:
    from reblock.compare import load_permeability_config
    from reblock.derivations import propose
    from reblock.permeability import EgressContext, permeability
    h, ell, K, pop = _CFG["h"], _CFG["ell"], _CFG["K"], _CFG["pop"]
    out = rows_dir(h, ell, K, pop) / f"{_BLOCKS[i].block_id}.parquet"
    if out.exists():
        return
    b = _BLOCKS[i]
    t0 = time.time()
    pcfg = load_permeability_config(common.REPO / "conf")
    ctx = EgressContext.of(b, pcfg.params)
    p = lifted.Params(ell_m=ell, K=K)
    population = common.POPULATIONS[pop]
    mesh = lifted.UniformMesh(h, offset=lifted.OFFSET)
    carve = common.Scorer(b, mesh, p, rule=common.Carve(), population=population)
    w = carve.w
    n = len(b.buildings)
    rows = []
    for name in common.LINEUP:
        roads = propose(_ARMS[name], b).roads
        if roads is None or len(roads) == 0:
            continue
        seen: dict[int, dict] = {}
        for bud in (*BUDGETS, None):
            pre = roads if bud is None else common.prefix_to(b, roads, bud, w)
            key = len(pre)
            if key in seen:                       # budget unreachable: same (whole) prefix
                rows.append({**seen[key], "budget": bud if bud is not None else np.nan,
                             "full": bud is None})
                continue
            r = dict(block=b.block_id, n=n, arm=name, n_roads=len(pre),
                     road_m=float(pre.geometry.length.sum()),
                     D=common.displacement(b, pre, w),
                     P_old=permeability(ctx, pre), stranded=carve.stranded)
            free = carve.free_of(pre)
            sol = lifted.solve(carve.grid, free, carve.f, p)
            u = carve.home_u_of(sol, free)
            u0 = carve.u0
            r.update(P_carve=1.0 - sol.P / carve.P0, P2_carve=carve.perm_p(u, 2.0),
                     u95=float(np.nanpercentile(u, 95) / np.nanpercentile(u0, 95)),
                     umax=float(np.nanmax(u) / np.nanmax(u0)),
                     umed=float(np.nanmedian(u) / np.nanmedian(u0)))
            seen[key] = r
            rows.append({**r, "budget": bud if bud is not None else np.nan, "full": bud is None})
    df = pd.DataFrame(rows)
    df["P0_carve"] = carve.P0
    tmp = out.with_suffix(f".{os.getpid()}.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, out)
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} n={n} {time.time() - t0:.0f}s", flush=True)


def run(workers: int, h: float, ell: float, K: int, pop: str) -> None:
    global _BLOCKS, _ARMS, _CFG
    _CFG = dict(h=h, ell=ell, K=K, pop=pop)
    rows_dir(h, ell, K, pop).mkdir(exist_ok=True)
    _BLOCKS = common.build_blocks(common.recipients())
    _ARMS = common.arms()
    order = list(range(len(_BLOCKS)))[::-1]          # largest first: the long poles start early
    with multiprocessing.get_context("fork").Pool(workers, maxtasksperchild=4) as pool:
        for _ in pool.imap_unordered(one, order):
            pass


# ------------------------------------------------------------------ report

SHORT = {"clearance": "clear", "clearance_looped": "clear_loop", "cycle_native": "cycle",
         "cycle_native_betweenness_contrast": "cycle_desire", "resistance_lp": "resist_lp",
         "euclidean_grid": "grid", "greedy_arterial_access_displacement": "arterial"}


def _d_at(g: pd.DataFrame, col: str, target: float) -> float:
    """Smallest displacement reaching `target` on the (D, metric) curve through (0, 0), linear
    between recorded prefixes; nan if the whole network does not reach it."""
    pts = g[["D", col]].drop_duplicates().sort_values("D").to_numpy()
    pts = np.vstack([[0.0, 0.0], pts])
    pts[:, 1] = np.maximum.accumulate(pts[:, 1])
    ok = np.nonzero(pts[:, 1] >= target)[0]
    if len(ok) == 0:
        return np.nan
    j = ok[0]
    (d0, p0), (d1, p1) = pts[j - 1], pts[j]
    return float(d0 + (d1 - d0) * (target - p0) / (p1 - p0)) if p1 > p0 else float(d1)


def _ci(x: pd.Series, n: int = 2000) -> str:
    v = x.to_numpy()
    rng = np.random.default_rng(0)
    meds = np.median(v[rng.integers(0, len(v), (n, len(v)))], axis=1)
    return f"[{np.percentile(meds, 2.5):+.3f},{np.percentile(meds, 97.5):+.3f}]"


def _tau(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import kendalltau
    return float(kendalltau(a, b).statistic) if len(a) >= 3 else np.nan


def report(h: float, ell: float, K: int, pop: str = "count") -> None:
    d = pd.concat([pd.read_parquet(p) for p in rows_dir(h, ell, K, pop).glob("*.parquet")],
                  ignore_index=True)
    d["arm"] = d["arm"].map(SHORT)
    nb = d.block.nunique()
    print(f"{nb} blocks, h {h} ell {ell} K {K}\n")
    A = d[(d.budget == 0.10) & (d.D >= 0.10 - 1e-9)]
    if "P_obl" not in A:                          # later runs score carve only
        A = A.assign(P_obl=np.nan)
    print("LENS A (10% displaced): blocks reaching, median perm old / roadless-carve / "
          "roadless-obliterate, median road m")
    for arm, g in A.groupby("arm"):
        print(f"  {arm:13s} {len(g):4d}  {g.P_old.median():.3f}  {g.P_carve.median():.3f}  "
              f"{g.P_obl.median():.3f}  {g.road_m.median():7.0f}")
    print(f"  stranded share: median {d.stranded.median():.3f}, max {d.stranded.max():.3f}")
    taus = [_tau(g.P_old.to_numpy(), g.P_carve.to_numpy()) for _, g in A.groupby("block")]
    print(f"\n  per-block Kendall tau, old vs roadless ranking of the arms reaching 10%: median "
          f"{np.nanmedian(taus):+.3f}, IQR {np.nanpercentile(taus, 25):+.3f}.."
          f"{np.nanpercentile(taus, 75):+.3f}, share < 0: {np.mean(np.array(taus) < 0):.2f}")
    for col in ("P_old", "P_carve"):
        best = A.loc[A.groupby("block")[col].idxmax()].arm.value_counts()
        print(f"  best arm by {col}: " + ", ".join(f"{a} {n}" for a, n in best.items()))
    tc = [_tau(g.P_carve.to_numpy(), g.P_obl.to_numpy()) for _, g in A.groupby("block")]
    print(f"  carve vs obliterate ranking: median tau {np.nanmedian(tc):+.3f}")
    # paired: each arm vs cycle (plain cycle_native) at Lens A
    piv = A.pivot_table(index="block", columns="arm", values=["P_old", "P_carve"])
    print("\n  paired vs cycle at Lens A: median diff old | roadless  [n]")
    for arm in sorted(A.arm.unique()):
        if arm == "cycle":
            continue
        s = piv.xs(arm, axis=1, level=1) - piv.xs("cycle", axis=1, level=1)
        s = s.dropna()
        print(f"    {arm:13s} {s.P_old.median():+.3f} {_ci(s.P_old)} | "
              f"{s.P_carve.median():+.3f} {_ci(s.P_carve)}  [{len(s)}]"
              f"  roadless wins {np.mean(s.P_carve > 0):.2f}")
    # Lens B roadless at fixed targets (the old P* 0.60 is not on the roadless scale)
    curves = d
    PSTARS = (0.25, 0.35)
    print(f"\nLENS B: old P* 0.60; roadless P*' {PSTARS}")
    for pstar in PSTARS:
        rows = []
        for (blk, arm), g in curves.groupby(["block", "arm"]):
            rows.append(dict(block=blk, arm=arm, B_old=_d_at(g, "P_old", 0.60),
                             B_carve=_d_at(g, "P_carve", pstar)))
        B = pd.DataFrame(rows)
        _lens_b(A, B, pstar)


def _lens_b(A: pd.DataFrame, B: pd.DataFrame, pstar: float) -> None:
    print(f"  -- roadless P*' {pstar}")
    print("  arm: reaching old | roadless, median displacement old | roadless")
    for arm, g in B.groupby("arm"):
        print(f"  {arm:13s} {g.B_old.notna().sum():4d} | {g.B_carve.notna().sum():4d}   "
              f"{g.B_old.median():.4f} | {g.B_carve.median():.4f}")
    for col in ("B_old", "B_carve"):
        best = B.dropna(subset=[col]).loc[lambda x: x.groupby("block")[col].idxmin()].arm
        print(f"  best (least displacement) by {col}: "
              + ", ".join(f"{a} {n}" for a, n in best.value_counts().items()))
    # frontier (Lens A perm, Lens B displacement) under each metric, arms reaching both
    AB = A[["block", "arm", "P_old", "P_carve"]].merge(B, on=["block", "arm"])
    for pa, pb, lab in (("P_old", "B_old", "old"), ("P_carve", "B_carve", "roadless")):
        tally: dict[str, int] = {}
        for _, g in AB.dropna(subset=[pa, pb]).groupby("block"):
            for _, r in g.iterrows():
                dom = ((g[pa] >= r[pa]) & (g[pb] <= r[pb]) &
                       ((g[pa] > r[pa]) | (g[pb] < r[pb]))).any()
                if not dom:
                    tally[r.arm] = tally.get(r.arm, 0) + 1
        print(f"  frontier (A, B) {lab}: " + ", ".join(
            f"{a} {n}" for a, n in sorted(tally.items(), key=lambda t: -t[1])))



if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        K = int(sys.argv[5]) if len(sys.argv) > 5 else 8
        pop = sys.argv[6] if len(sys.argv) > 6 else "count"
        run(int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), K, pop)
    else:
        K = int(sys.argv[4]) if len(sys.argv) > 4 else 8
        pop = sys.argv[5] if len(sys.argv) > 5 else "count"
        report(float(sys.argv[2]), float(sys.argv[3]), K, pop)
