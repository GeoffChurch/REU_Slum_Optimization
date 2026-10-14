"""The gap complex's summaries (NOTES, "The gap complex: passability, enclosure and flow without
the raster"): gap_rows/summary_<name>.csv, committed; the rows they read are not.

    S=research/roadless/gap_report.py     # from the repo root, PYTHONPATH=.
    uv run python $S agree|apertures|sweep|compare|simp|flowsweep|all

agree      the agreement with the raster (gap_study agree, crossings): agree_blocks (per block and
           spacing: the raster's enclosed homes, the complex's walled in at 0, RESOLUTION and the
           diagonal resolution, the differences), agree (pooled by spacing), agree_homes (every
           home on which the two disagree at 0 or RESOLUTION, with its exit width), crossings.
apertures  the measured apertures against the true gaps (gap_study apertures), per spacing.
sweep      the passability sweep (gap_study sweep): sweep (per width, spacing and size: the
           walled-in population, free first's D and buildings), exits (the population-weighted
           distribution of the homes' exit widths), bars.
compare    the complex against the raster (gap_compare complex, iso; aperture_rows' lifted):
           compare_levels, compare_tau, compare_homes (Spearman of the escape times), converge.
simp       SIMP under the forms (gap_simp).
flowsweep  the sweep's flow part (gap_compare flowsweep).
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gap_study  # noqa: E402

ROWS = HERE / "gap_rows"
RES = gap_study.RESOLUTION
RES_DIAG = RES * np.sqrt(2.0)          # a 4-connected chain along a diagonal needs this
EXIT_BINS = (0.0, 0.05, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.375, 0.5, 0.625, 0.75, 1.0, 1.5, 2.0,
             3.0, np.inf)
LENSES = ("L_P", "L_Pq0.5", "L_J2")
REACH_TOL = 1e-9
TOP = 3                                # compare_top: the share of a lens its top TOP homes hold


def _summary(name: str, df: pd.DataFrame) -> None:
    path = ROWS / f"summary_{name}.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"{path.name}: {len(df)} rows", flush=True)


def _read(pattern: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(ROWS / pattern)))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else \
        pd.DataFrame()


def _size(bids: pd.Series) -> pd.Series:
    small = set(gap_study.common.recipients())
    big = gap_study._large_sizes()
    return bids.map(lambda b: "small" if b in small else
                    "oversized" if big.get(b, 0.0) > gap_study.UNIFORM_MAX_M else "large")


# ------------------------------------------------------------------ stage 1

def agree() -> None:
    H, B = _read("agree/*.parquet"), _read("agree_blocks/*.parquet")
    H["size"] = _size(H.block)
    rows, pooled, homes = [], [], []
    for s in gap_study.S_SENS:
        e = H[f"exit_s{s:g}"].to_numpy()
        enc = H.raster_enclosed.to_numpy()
        for w in (0.0, RES, RES_DIAG):
            c = e < w if w > 0 else e < 0
            pooled.append(dict(s=s, w=w, raster=int(enc.sum()), complex=int(c.sum()),
                               raster_only=int((enc & ~c).sum()),
                               complex_only=int((c & ~enc).sum()),
                               blocks_differ=int(H[enc != c].block.nunique())))
        for w in (0.0, RES):
            c = e < w if w > 0 else e < 0
            d = H[(enc != c)]
            for _, r in d.iterrows():
                homes.append(dict(s=s, w=w, block=r.block, home=int(r.home), w_pop=r.w,
                                  raster_enclosed=bool(r.raster_enclosed),
                                  raster_shadowed=bool(r.raster_shadowed),
                                  exit=float(r[f"exit_s{s:g}"]),
                                  own_ring_exit=float(r[f"own_s{s:g}"])))
        for bid, h in H.groupby("block"):
            ee, en = h[f"exit_s{s:g}"].to_numpy(), h.raster_enclosed.to_numpy()
            rows.append(dict(block=bid, size=h["size"].iloc[0], s=s, raster=int(en.sum()),
                             complex_0=int((ee < 0).sum()), complex_res=int((ee < RES).sum()),
                             raster_only_0=int((en & (ee >= 0)).sum()),
                             complex_only_0=int((~en & (ee < 0)).sum()),
                             raster_only_res=int((en & (ee >= RES)).sum()),
                             complex_only_res=int((~en & (ee < RES)).sum())))
    _summary("agree", pd.DataFrame(pooled))
    _summary("agree_blocks", pd.DataFrame(rows))
    _summary("agree_homes", pd.DataFrame(homes))
    b = B[(B.s == 0.125)].copy()
    b["same_held"] = b.held == b.raster_held
    _summary("agree_free", b[["block", "w", "walled", "raster_enclosed", "held", "D", "raster_held",
                              "raster_D", "same_held", "passes", "misses", "left", "crossings",
                              "raster_crossings", "t_build", "t_free", "t_raster"]])
    X = _read("crossings/*.parquet")
    if len(X):
        _summary("crossings", X)


APERTURE_BINS = (0.0, 0.125, 0.5, 1.0, 2.0)


def apertures() -> None:
    """The measured apertures against the true gaps (gap_study apertures): per spacing and true
    gap, over the direct pairs, how many a passage joins and the error (measured less true)."""
    A = _read("apertures/*.parquet")
    if not len(A):
        return
    A = A[A.direct]
    rows = []
    for s in gap_study.S_SENS:
        m = A[f"meas_s{s:g}"].to_numpy()
        for lo, hi in zip(APERTURE_BINS[:-1], APERTURE_BINS[1:], strict=True):
            sel = (A.true >= lo).to_numpy() & (A.true < hi).to_numpy()
            e = m[sel] - A.true.to_numpy()[sel]
            ok = np.isfinite(e)
            q = np.quantile(e[ok], (0.5, 0.9, 0.99)) if ok.any() else (np.nan,) * 3
            rows.append(dict(s=s, true_lo=lo, true_hi=hi, pairs=int(sel.sum()),
                             joined=int(ok.sum()), err_min=float(e[ok].min()) if ok.any()
                             else np.nan, err_median=q[0], err_p90=q[1], err_p99=q[2],
                             err_max=float(e[ok].max()) if ok.any() else np.nan))
    _summary("apertures", pd.DataFrame(rows))


def sweep() -> None:
    S = _read("sweep/*/*.parquet")
    if not len(S):
        return
    S["size"] = _size(S.block)
    S["over"] = S.free_D > gap_study.BUDGET
    g = S.groupby(["s", "size", "w"])
    tab = g.agg(blocks=("block", "nunique"), walled=("walled", "sum"),
                walled_pop_median=("walled_pop", "median"),
                walled_pop_p90=("walled_pop", lambda x: float(np.quantile(x, 0.9))),
                free_D_median=("free_D", "median"),
                free_D_p90=("free_D", lambda x: float(np.quantile(x, 0.9))),
                free_D_max=("free_D", "max"), free_n_median=("free_n", "median"),
                blocks_over_budget=("over", "sum"), misses=("misses", "sum"),
                left=("left", "sum"), t_median=("t", "median")).reset_index()
    _summary("sweep", tab)
    Hm = _read("homes/*/*.parquet")
    Hm["size"] = _size(Hm.block)
    Hm["s"] = Hm.spec.str.extract(r"gc\.s([0-9.]+)\.").astype(float)
    ex = []
    for (s, size), h in Hm.groupby(["s", "size"]):
        tot = h.w.sum()
        e = h.exit.to_numpy()
        for lo, hi in zip(EXIT_BINS[:-1], EXIT_BINS[1:], strict=True):
            sel = (e >= lo) & (e < hi)
            ex.append(dict(s=s, size=size, lo=lo, hi=hi, homes=int(sel.sum()),
                           pop_share=float(h.w[sel].sum() / tot)))
        sel = ~np.isfinite(e) & (e < 0)
        ex.append(dict(s=s, size=size, lo=-np.inf, hi=0.0, homes=int(sel.sum()),
                       pop_share=float(h.w[sel].sum() / tot)))
        sel = np.isposinf(e)
        ex.append(dict(s=s, size=size, lo=np.inf, hi=np.inf, homes=int(sel.sum()),
                       pop_share=float(h.w[sel].sum() / tot)))
    _summary("exits", pd.DataFrame(ex))
    # the bars: a walled-in home's pocket, how much roomier than its exit
    bh = Hm[np.isfinite(Hm.exit) & np.isfinite(Hm.fill)].copy()
    bh["length"] = bh.fill - bh.exit
    bars = []
    for (s, size), h in bh.groupby(["s", "size"]):
        for lo, hi in zip(EXIT_BINS[:-1], EXIT_BINS[1:], strict=True):
            sel = (h.exit >= lo) & (h.exit < hi)
            if not sel.any():
                continue
            ww = h.w[sel].to_numpy()
            L = h.length[sel].to_numpy()
            o = np.argsort(L)
            c = np.cumsum(ww[o]) / ww.sum()
            bars.append(dict(s=s, size=size, exit_lo=lo, exit_hi=hi, homes=int(sel.sum()),
                             pop=float(ww.sum()), length_median=float(np.interp(0.5, c, L[o])),
                             length_p10=float(np.interp(0.1, c, L[o])),
                             length_p90=float(np.interp(0.9, c, L[o]))))
    _summary("bars", pd.DataFrame(bars))


# ------------------------------------------------------------------ stage 2

def _b_rank(g: pd.DataFrame, val: str) -> np.ndarray:
    """B's order: the least population left enclosed first, then the larger lens."""
    e, v = g["enclosed"].to_numpy(), g[val].to_numpy()
    ahead = (e[None, :] < e[:, None]) | ((e[None, :] == e[:, None]) & (v[None, :] > v[:, None]))
    return 1.0 + ahead.sum(axis=1)


def _methods(g: pd.DataFrame) -> pd.DataFrame:
    """The methods compared: the stored clearings (each freed first where its model leaves a home
    enclosed) and the road prefixes that reach the budget."""
    g = g.set_index("field")
    for n in ("simp_default", "cheap"):
        if f"{n}+free" in g.index:
            g.loc[n] = g.loc[f"{n}+free"]
    keep = [f for f in g.index if f in ("simp_default", "cheap") or (
        g.loc[f, "kind"] == "road" and g.loc[f, "D"] >= gap_study.BUDGET - REACH_TOL)]
    return g.loc[keep].reset_index()


def compare() -> None:
    from scipy.stats import kendalltau, spearmanr
    models = {"lifted": pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
        str(HERE / "aperture_rows/score/o0/Fws/*.parquet")))], ignore_index=True),
        "iso": _read("compare/iso_0.5/*.parquet")}
    for d in sorted(glob.glob(str(ROWS / "compare/complex_*"))):
        name = Path(d).name.replace("complex_", "complex:")
        models[name] = _read(f"compare/{Path(d).name}/*.parquet")
    models = {k: v for k, v in models.items() if len(v)}
    blocks = sorted(set.intersection(*[set(v.block) for v in models.values()]))
    lev, taus = [], []
    for bid in blocks:
        per = {k: _methods(v[v.block == bid]) for k, v in models.items()}
        names = sorted(set.intersection(*[set(p.field) for p in per.values()]))
        for k, p in per.items():
            p = p.set_index("field").loc[names]
            for f in names:
                lev.append(dict(block=bid, model=k, field=f, enclosed=p.loc[f, "enclosed"],
                                **{L: p.loc[f, L] for L in LENSES}))
        keys = list(per)
        for i, a in enumerate(keys):
            for b2 in keys[i + 1:]:
                pa = per[a].set_index("field").loc[names].reset_index()
                pb = per[b2].set_index("field").loc[names].reset_index()
                for L in LENSES:
                    taus.append(dict(block=bid, a=a, b=b2, lens=L, methods=len(names),
                                     tau=float(kendalltau(_b_rank(pa, L),
                                                          _b_rank(pb, L)).statistic)))
    _summary("compare_levels", pd.DataFrame(lev))
    T = pd.DataFrame(taus)
    _summary("compare_tau_blocks", T)
    _summary("compare_tau", T.groupby(["a", "b", "lens"]).tau.agg(
        median="median", q25=lambda x: float(np.quantile(x, 0.25)), min="min",
        blocks="count").reset_index())
    # the escape times per home at baseline and under SIMP's default
    homes = {"lifted": pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(
        str(HERE / "aperture_rows/homes_score/o0/Fws/*.parquet")))], ignore_index=True),
        "iso": _read("compare/iso_0.5/homes/*.parquet")}
    for d in sorted(glob.glob(str(ROWS / "compare/complex_*"))):
        homes[Path(d).name.replace("complex_", "complex:")] = _read(
            f"compare/{Path(d).name}/homes/*.parquet")
    sp = []
    for bid in blocks:
        hh = {k: v[v.block == bid].set_index("home") for k, v in homes.items() if len(v)}
        keys = list(hh)
        for i, a in enumerate(keys):
            for b2 in keys[i + 1:]:
                for col in ("u0", "u_default"):
                    x, y = hh[a][col], hh[b2][col]
                    ok = x.notna() & y.notna()
                    if ok.sum() > 2:
                        sp.append(dict(block=bid, a=a, b=b2, u=col, homes=int(ok.sum()),
                                       spearman=float(spearmanr(x[ok], y[ok]).statistic),
                                       ratio_median=float(np.median(y[ok] / x[ok]))))
    _summary("compare_homes", pd.DataFrame(sp))
    # the share of each lens its top TOP homes hold, at baseline and under SIMP's default
    top = []
    for k, v in homes.items():
        for bid, h in v[v.block.isin(blocks)].groupby("block"):
            for col in ("u0", "u_default"):
                u = h[col].to_numpy()
                ok = np.isfinite(u)
                terms = {"P": h.w.to_numpy()[ok] * u[ok],
                         "J2": h.w.to_numpy()[ok] * u[ok] ** 2}
                top.append(dict(block=bid, model=k, u=col, **{
                    f"top{TOP}_{L}": float(np.sort(t)[-TOP:].sum() / t.sum())
                    for L, t in terms.items()}))
    _summary("compare_top", pd.DataFrame(top))
    C = _read("compare/converge/*.parquet")
    if len(C):
        _summary("converge", C)


def _set(text: str) -> set[int]:
    return {int(i) for i in text.split()}


def simp() -> None:
    """SIMP under the forms (gap_simp run): the rows less the cleared lists, each pair of forms'
    clearings' Jaccard per block, and every clearing's corridors under one flow (gap_simp cross)."""
    D = _read("simp/*/*/*.parquet")
    if not len(D):
        return
    _summary("simp", D.drop(columns=[c for c in ("cleared", "free_cleared") if c in D]))
    jac = []
    for bid, g in D.groupby("block"):
        cl = dict(zip(g.form, g.cleared.map(_set), strict=True))
        forms = sorted(cl)
        for i, a in enumerate(forms):
            for b2 in forms[i + 1:]:
                u = cl[a] | cl[b2]
                jac.append(dict(block=bid, a=a, b=b2, n_a=len(cl[a]), n_b=len(cl[b2]),
                                jaccard=len(cl[a] & cl[b2]) / len(u) if u else 1.0))
    _summary("simp_jaccard", pd.DataFrame(jac))
    X = _read("simp_cross/*/*.parquet")
    if len(X):
        _summary("simp_cross", X)


def flowsweep() -> None:
    F = _read("compare/flowsweep_*/*/*.parquet")
    if not len(F):
        return
    F["size"] = _size(F.block)
    _summary("flowsweep_blocks", F)
    base = F[F.state == "free"].copy()
    # each block's levels relative to its closing at 0.125 (the raster's resolution)
    ref = base[base.form == "close0.125"].set_index("block")
    for c in ("P", "Pq", "J2"):
        base[f"{c}_rel"] = base[c] / base.block.map(ref[c])
    tab = base.groupby(["size", "form", "w"]).agg(
        blocks=("block", "nunique"), walled_pop=("walled_pop", "median"),
        free_D=("free_D", "median"), P_rel=("P_rel", "median"), Pq_rel=("Pq_rel", "median"),
        J2_rel=("J2_rel", "median"), top_P=("top_P", "median"), top_Pq=("top_Pq", "median"),
        top_J2=("top_J2", "median"), top_J2_p90=("top_J2", lambda x: float(np.quantile(x, 0.9))),
        t=("t_block", "median")).reset_index()
    _summary("flowsweep", tab)


def main(argv: list[str]) -> None:
    jobs = {"agree": agree, "apertures": apertures, "sweep": sweep, "compare": compare,
            "simp": simp, "flowsweep": flowsweep}
    which = list(jobs) if argv[0] == "all" else argv
    for j in which:
        jobs[j]()


if __name__ == "__main__":
    main(sys.argv[1:])
