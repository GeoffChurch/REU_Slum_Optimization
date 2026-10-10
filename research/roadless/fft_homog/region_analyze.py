"""Summaries of the 5810@major experiment's tables, from what the other region_*.py wrote to OUT.

    ... uv run python research/roadless/fft_homog/region_analyze.py tiles|homes|restrict|report

tiles: item 3 (tiles_analyze.py's statistics, with bootstrap intervals within the sample's
strata) -> tiles_resolve_summary.csv; homes: item 1 by distance to the exit ->
region_twoscale_bands.csv, region_decomp_bands.csv; restrict: item 4's scored clearings as
shares of the unrestricted Lens A -> restrict_summary.csv; report: the report's tables.
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import region_common as rc  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

BOOT = 2000      # bootstrap resamples within the sample's deciles (its strata), seed 0


def tiles() -> None:
    t = pd.concat([pd.read_csv(p) for p in sorted(rc.OUT.glob("tiles_resolve_*.csv"))
                   if p.name != "tiles_resolve_summary.csv"], ignore_index=True)
    t = t.drop_duplicates(["r0", "c0", "rule"])
    t = t.merge(pd.read_csv(rc.OUT / "tiles_sample.csv")[["r0", "c0", "rule", "decile"]],
                on=["r0", "c0", "rule"])
    fab = rc.load_fabric()
    J0 = float(np.load(rc.OUT / "fine_base.npz")["J0"])
    # naive proxy: cleared area x the tile's mean baseline fine u
    u = np.zeros(fab.o.shape)
    d = np.load(rc.OUT / "fabric_region.npz")
    unk = d["reach"] & ~fab.ground
    u[unk] = np.load(rc.OUT / "fine_u0.npy")
    t["proxy"] = [-m2 * float(u[r:r + 100, c:c + 100].mean())
                  for r, c, m2 in zip(t.r0, t.c0, t.cleared_m2, strict=True)]
    out = []
    for rule, g in t.groupby("rule"):
        g = g[g.n_bldg > 0]
        gain = -g.dJ_fine / J0
        row = dict(rule=rule, tiles=len(g), median_bldg=g.n_bldg.median(),
                   median_cleared_m2=g.cleared_m2.median(),
                   fine_gain_median_pct=100 * gain.median(), fine_gain_max_pct=100 * gain.max(),
                   fine_gain_min_pct=100 * gain.min())
        k = max(1, len(g) // 10)
        top_f = set(g.nsmallest(k, "dJ_fine").index)
        topm_f = set((g.dJ_fine / g.cleared_m2).nsmallest(k).index)
        rng = np.random.default_rng(0)
        dec = g.decile.to_numpy()
        boot = [np.concatenate([rng.choice(np.flatnonzero(dec == d), size=int((dec == d).sum()))
                                for d in range(10)]) for _ in range(BOOT)]
        for col in ("dJ_ts50", "dJ_lin50", "dJ_ts100", "dJ_lin100", "proxy"):
            row[f"spearman_{col}"] = spearmanr(g.dJ_fine, g[col]).statistic
            for per, key in ((False, ""), (True, "perm2_")):
                f = (g.dJ_fine / g.cleared_m2 if per else g.dJ_fine).to_numpy()
                q = (g[col] / g.cleared_m2 if per else g[col]).to_numpy()
                bs = [spearmanr(f[i], q[i]).statistic for i in boot]
                row[f"spearman_{key}{col}_ci"] = (f"[{np.percentile(bs, 2.5):.3f}, "
                                                  f"{np.percentile(bs, 97.5):.3f}]")
            row[f"spearman_perm2_{col}"] = spearmanr(g.dJ_fine / g.cleared_m2,
                                                     g[col] / g.cleared_m2).statistic
            row[f"top10_overlap_{col}"] = len(top_f & set(g.nsmallest(k, col).index)) / k
            row[f"top10_overlap_perm2_{col}"] = len(
                topm_f & set((g[col] / g.cleared_m2).nsmallest(k).index)) / k
            if col != "proxy":
                r = g[col] / g.dJ_fine
                row[f"ratio_{col}_median"] = float(np.median(r))
                row[f"ratio_{col}_p10"], row[f"ratio_{col}_p90"] = np.percentile(r, [10, 90])
        for col in ("fine_s", "windows50_s", "macro50_s", "windows100_s", "macro100_s",
                    "seconds"):
            row[f"median_{col}"] = float(g[col].median())
        row["median_fine_iters"] = float(g.fine_iters.median())
        row["median_windows50"] = float(g.windows50.median())
        row["median_windows100"] = float(g.windows100.median())
        out.append(row)
    r = pd.DataFrame(out)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print(r.T.to_string())
    r.to_csv(rc.OUT / "tiles_resolve_summary.csv", index=False)


BANDS = ((0, 10), (10, 30), (30, 100), (100, 200), (200, 300), (300, 450), (450, np.inf))


def homes() -> None:
    """Item 1 by distance to the exit, from region_twoscale.py's homes_L<L>_<fill>.npz (readout
    U, macro H 2 m): per band, the homes' median relative error, the band's J_2 error and share,
    and the median two-scale minus fine u."""
    ts_rows = pd.read_csv(rc.OUT / "region_twoscale.csv")
    rows, dec = [], []
    for L in (100, 200, 50):
        for fill in ("open-outside", "extrapolated", "inside-avg"):
            z = np.load(rc.OUT / f"homes_L{L}_{fill}.npz")
            u_f, u_t, hd, w, live = z["u_f"], z["u_t"], z["hd"], z["w"], z["live"]
            ok = live & (u_f > 0)
            J_f = (w[ok] * u_f[ok] ** 2).sum()
            rel = (u_t - u_f) / u_f
            src = ts_rows[(ts_rows.L_m == L) & (ts_rows.fill == fill) & (ts_rows.H_m == 2)
                          & (ts_rows["corr"] == "U")].iloc[0]
            row = dict(L=L, fill=fill, J2_err=src.J2_rel_err, P_err=src.P_rel_err,
                       spearman=src.spearman, med_abs_rel=src.med_abs_rel)
            drow = dict(L=L, fill=fill)
            for lo, hi in BANDS:
                m = ok & (hd >= lo) & (hd < hi)
                key = f"{lo}-{hi if np.isfinite(hi) else ''}"
                row[f"medrel {key}"] = float(np.median(rel[m]))
                row[f"J2err {key}"] = float((w[m] * u_t[m] ** 2).sum() / (w[m] * u_f[m] ** 2).sum()
                                            - 1)
                drow[key] = float(np.median(u_t[m] - u_f[m]))
                if L == 100 and fill == "open-outside":
                    dec.append(dict(band=key, homes=int(m.sum()),
                                    J2_share=float((w[m] * u_f[m] ** 2).sum() / J_f),
                                    fine_median_u=float(np.median(u_f[m]))))
            rows.append(row)
            dec.append(drow)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    r = pd.DataFrame(rows)
    print(r.round(4).to_string())
    print(pd.DataFrame(dec).round(3).to_string())
    r.to_csv(rc.OUT / "region_twoscale_bands.csv", index=False)
    pd.DataFrame(dec).to_csv(rc.OUT / "region_decomp_bands.csv", index=False)


def restrict() -> None:
    """Item 4: each scored clearing's Lens A (lifted uni, a5x8) as a share of its unrestricted
    stored value, joined with its footprint share and budget (gpu_*.csv x restrict_*.csv)."""
    sc = pd.concat([pd.read_csv(p) for p in sorted(rc.OUT.glob("gpu_*.csv"))],
                   ignore_index=True).drop_duplicates("name")
    full = {k: float(sc.loc[sc.name == f"stored_{k}", "lensA"].iloc[0]) for k in ("cheap", "simp")}
    rs = pd.concat([pd.read_csv(p) for p in sorted(rc.OUT.glob("restrict_*.csv"))],
                   ignore_index=True)
    rs = rs.drop_duplicates(["ranking", "rule", "k", "clearing"])
    rows = []
    for _, r in sc.iterrows():
        if r["name"].startswith("stored_"):
            continue
        parts = r["name"].split("_")
        clearing = parts[0]
        topup = parts[-1] == "topup"
        if parts[1] == "alltiles":
            ranking, rule, k = "all tiles", "-", 0
        else:
            ranking, rule, k = parts[1], parts[2], int(parts[3][1:])
        m = rs[(rs.ranking == ranking) & (rs.rule == rule) & (rs.k == k)
               & (rs.clearing == clearing)]
        rows.append(dict(clearing=clearing, ranking=ranking, rule=rule, k=k, topup=topup,
                         tiles=int(m.tiles.iloc[0]) if len(m) else np.nan,
                         share_cleared_in_tiles=float(m.share_cleared_in_tiles.iloc[0])
                         if len(m) else np.nan,
                         n_buildings=int(r.n_buildings), D=float(r.D), lensA=float(r.lensA),
                         frac_of_unrestricted=float(r.lensA) / full[clearing]))
    out = pd.DataFrame(rows).sort_values(["topup", "ranking", "rule", "k", "clearing"])
    pd.set_option("display.width", 220)
    print({k: round(v, 4) for k, v in full.items()})
    print(out.round(4).to_string(index=False))
    out.to_csv(rc.OUT / "restrict_summary.csv", index=False)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:+.{nd}f}%"


def report() -> None:
    """The markdown tables of the final report, from the saved results."""
    b = pd.read_csv(rc.OUT / "region_twoscale_bands.csv")
    keys = [c[len("medrel "):] for c in b.columns if c.startswith("medrel ")]
    rows = []
    for _, r in b.iterrows():
        rows.append({"L (stride)": f"{r.L} ({ {100: 25, 50: 10, 200: 50}[r.L]})",
                     "near-exit": r.fill, "J2 err": pct(r.J2_err), "P err": pct(r.P_err),
                     "Spearman": f"{r.spearman:.4f}", "med abs": f"{100 * r.med_abs_rel:.1f}%",
                     **{k + " m": pct(r[f"medrel {k}"], 0 if abs(r[f"medrel {k}"]) > 0.095
                                      else 1) for k in keys}})
    print(md(pd.DataFrame(rows)))
    print()
    rows = []
    for _, r in b.iterrows():
        rows.append({"L": r.L, "near-exit": r.fill,
                     **{k + " m": pct(r[f"J2err {k}"]) for k in keys}})
    print(md(pd.DataFrame(rows)))
    print()
    d = pd.read_csv(rc.OUT / "region_decomp_bands.csv")
    info = d[d.band.notna()]
    print(md(info.assign(J2_share=(100 * info.J2_share).round(2), fine_median_u=info.fine_median_u
                         .round(1), homes=info.homes.astype(int))[["band", "homes", "J2_share",
                                                                    "fine_median_u"]]))
    dd = d[d.band.isna()].drop(columns=["band", "homes", "J2_share", "fine_median_u"])
    print(md(dd.assign(L=dd.L.astype(int)).round(1)))
    print()
    c = pd.read_csv(rc.OUT / "region_clearings.csv")
    f = pd.read_csv(rc.OUT / "region_clearings_fine.csv").set_index("clearing")
    rows = []
    for model, col in (("lifted uni, a5x8 (stored)", "lensA_lifted_stored"),
                       ("fine scalar, h 0.5", "lensA_fine_scalar")):
        ch, si = float(f.loc["cheap", col]), float(f.loc["simp", col])
        rows.append(dict(model=model, cheap=f"{ch:.4f}", simp=f"{si:.4f}", gap=f"{si - ch:.4f}"))
    for (field, fill), g in c.groupby(["field", "fill"], sort=False):
        ch = float(g.loc[g.clearing == "cheap", "lensA_twoscale"].iloc[0])
        si = float(g.loc[g.clearing == "simp", "lensA_twoscale"].iloc[0])
        rows.append(dict(model=f"two-scale {field}, {fill}", cheap=f"{ch:.4f}", simp=f"{si:.4f}",
                         gap=f"{si - ch:.4f}"))
    print(md(pd.DataFrame(rows)))
    print()
    t = pd.read_csv(rc.OUT / "tiles_resolve_summary.csv").set_index("rule")
    rows = []
    for col, name in (("dJ_ts50", "two-scale L 50"), ("dJ_lin50", "first-order L 50"),
                      ("dJ_ts100", "two-scale L 100"), ("dJ_lin100", "first-order L 100"),
                      ("proxy", "naive proxy")):
        rows.append({"predictor": name,
                     **{f"{q} {r}": f"{t.loc[r, f'{q0}{col}']:.3f}" if q != "top 10%" else
                        f"{100 * t.loc[r, f'{q0}{col}']:.0f}%"
                        for r in ("R20", "STRIP")
                        for q, q0 in (("rho", "spearman_"), ("rho/m2", "spearman_perm2_"),
                                      ("top 10%", "top10_overlap_"))}})
    print(md(pd.DataFrame(rows)))


if __name__ == "__main__":
    {"tiles": tiles, "homes": homes, "restrict": restrict, "report": report}[sys.argv[1]]()
