"""Items 1 and 2 of the 5810@major experiment: two-scale J_2 against the fine scalar solve
(by distance to the exit), and the two stored region clearings' Lens A under both.

    ... uv run python research/roadless/fft_homog/region_twoscale.py base
    ... uv run python research/roadless/fft_homog/region_twoscale.py clearings <workers>

base: every field in OUT (fields_L<L>_s<s>.npz) x the three near-exit treatments (open-outside,
extrapolated: windows under 90% inside take the nearest such window's tensor, inside-avg:
<q>_in <e>_in^-1) x macro H (2 m; 5 m too for L 100), readouts U and U + chi.grad U + w ->
OUT/region_twoscale.csv, OUT/region_decomp.csv, OUT/homes_L<L>_<fill>.npz.
clearings: for L 50 (s 10) and L 100 (s 25), each clearing's touched windows recomputed
(ts.update_field), the macro re-solved under each treatment -> OUT/region_clearings.csv.
"""
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import region_common as rc  # noqa: E402
import ts  # noqa: E402
from region_macro import MacroBase  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

FIELDS = ((100, 25), (50, 10), (200, 50))
FILLS = (("open-outside", None, False), ("extrapolated", 0.9, False), ("inside-avg", None, True))
BANDS = ((0, 10), (10, 30), (30, 100), (100, 300), (300, 600), (600, np.inf))


def home_depth(fab: rc.Fabric) -> np.ndarray:
    """Each home's injection-weighted mean straight-line distance (m) from its ring cells to
    the nearest ground cell (twoscale_run.py's)."""
    dist = ndimage.distance_transform_edt(~fab.ground) * rc.H
    on = fab.owner >= 0
    return (np.bincount(fab.owner[on], weights=(fab.f * dist)[on], minlength=len(fab.w))
            / np.where(fab.w > 0, fab.w, 1))


def base() -> None:
    fab = rc.load_fabric()
    fine = np.load(rc.OUT / "fine_base.npz")
    u_f, J_f, P_f = fine["uh0"], float(fine["J0"]), float(fine["P0"])
    hd = home_depth(fab)
    ok = fab.live & (u_f > 0)
    print(f"fine J2 {J_f:.8g}, P {P_f:.8g}; home depth quantiles (m) 10/50/90/99/max "
          f"{np.round(np.percentile(hd[ok], [10, 50, 90, 99, 100]), 1)}", flush=True)
    share = {b: float((fab.w[ok & (hd >= b[0]) & (hd < b[1])]
                       * u_f[ok & (hd >= b[0]) & (hd < b[1])] ** 2).sum() / J_f) for b in BANDS}
    print("J2 share by band:", {f"{a}-{b}": round(v, 4) for (a, b), v in share.items()},
          flush=True)
    rows, decomp = [], []
    for L, s in FIELDS:
        path = rc.OUT / f"fields_L{L}_s{s}.npz"
        if not path.exists():
            print("missing", path, flush=True)
            continue
        tf = ts.load_field(str(path))
        for fill_name, mi, ia in FILLS:
            sb = ts.fill(tf, mi, ia)
            for H_m in ((2, 5) if L == 100 else (2,)):
                k = int(round(H_m / rc.H))
                with rc.Monitor(f"macro: L {L} {fill_name} H {H_m} m (assemble, SA, CG)") as mon:
                    mac = MacroBase(sb, tf.s, k, fab)
                U, Ux, Uy = mac.evaluate(mac.U0)
                for corr in ("U", "U+chi+w"):
                    uc = U if corr == "U" else U + tf.chi[0] * Ux + tf.chi[1] * Uy + tf.wf
                    uc = np.where(fab.ground, 0.0, uc)
                    u_t = ts.home_u(uc, fab.f, fab.owner, fab.w, fab.live)
                    if corr == "U":
                        assert np.allclose(u_t[ok], mac.uh0[ok], rtol=1e-9, atol=1e-9)
                    J_t, P_t = ts.J(u_t, fab.w, fab.live), float((fab.f * uc).sum())
                    rel = (u_t - u_f) / u_f
                    row = dict(L_m=L, stride_m=s, fill=fill_name, H_m=H_m, corr=corr,
                               macro_unknowns=mac.N, macro_iters=mac.iters,
                               macro_s=mon.cost.wall_s, macro_peak_gb=mon.cost.peak_gb,
                               J2_rel_err=J_t / J_f - 1, P_rel_err=P_t / P_f - 1,
                               spearman=spearmanr(u_f[ok], u_t[ok]).statistic,
                               med_abs_rel=float(np.median(np.abs(rel[ok]))),
                               p90_abs_rel=float(np.percentile(np.abs(rel[ok]), 90)))
                    for lo, hi in BANDS:
                        m = ok & (hd >= lo) & (hd < hi)
                        row[f"medrel_d{lo}"] = float(np.median(rel[m])) if m.any() else np.nan
                        row[f"J2err_d{lo}"] = float((fab.w[m] * u_t[m] ** 2).sum()
                                                    / (fab.w[m] * u_f[m] ** 2).sum() - 1)
                        row[f"J2share_d{lo}"] = share[(lo, hi)]
                        row[f"n_d{lo}"] = int(m.sum())
                        if corr == "U" and H_m == 2:
                            decomp.append(dict(L_m=L, fill=fill_name, band=f"{lo}-{hi}",
                                               fine_median_u=float(np.median(u_f[m])),
                                               median_diff=float(np.median(u_t[m] - u_f[m])),
                                               homes=int(m.sum())))
                    rows.append(row)
                    print({a: (round(b, 4) if isinstance(b, float) else b)
                           for a, b in row.items()}, flush=True)
                    if corr == "U" and H_m == 2:
                        np.savez_compressed(rc.OUT / f"homes_L{L}_{fill_name}.npz", u_f=u_f,
                                            u_t=u_t, hd=hd, w=fab.w, live=fab.live)
                del mac, U, Ux, Uy, uc
        pd.DataFrame(rows).to_csv(rc.OUT / "region_twoscale.csv", index=False)
        pd.DataFrame(decomp).to_csv(rc.OUT / "region_decomp.csv", index=False)


def clearings(workers: int) -> None:
    fab = rc.load_fabric()
    fine = pd.read_csv(rc.OUT / "region_clearings_fine.csv").set_index("clearing")
    rows = []
    for L, s in ((50, 10), (100, 25)):
        tf = ts.load_field(str(rc.OUT / f"fields_L{L}_s{s}.npz"))
        J0 = {}
        for fill_name, mi, ia in FILLS:
            J0[fill_name] = MacroBase(ts.fill(tf, mi, ia), tf.s, 4, fab).J0
        for name, ids in rc.stored_clearings().items():
            oc = rc.open_buildings(fab, ids)
            changed = oc != fab.o
            nwin = int(ts.windows_hit(tf, changed).sum())
            with rc.Monitor(f"two-scale clearing {name} L {L}: {nwin} windows on {workers} "
                            f"workers") as mon:
                tfc = ts.update_field(tf, oc, fab.inside, fab.f, changed, workers)
            for fill_name, mi, ia in FILLS:
                t = time.perf_counter()
                mac = MacroBase(ts.fill(tfc, mi, ia), tf.s, 4, fab)
                lensA = 1 - np.sqrt(mac.J0 / J0[fill_name])
                row = dict(field=f"L{L}_s{s}", fill=fill_name, clearing=name,
                           n_buildings=len(ids), lensA_lifted_stored=rc.STORED_LENS_A[name],
                           lensA_fine_scalar=float(fine.loc[name, "lensA_fine_scalar"]),
                           lensA_twoscale=lensA, J_twoscale=mac.J0, J0_twoscale=J0[fill_name],
                           windows_recomputed=nwin, windows_total=int(tf.have.sum()),
                           update_s=mon.cost.wall_s, update_cpu_s=mon.cost.cpu_s,
                           macro_s=time.perf_counter() - t)
                rows.append(row)
                print({a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()},
                      flush=True)
                del mac
            del tfc
        pd.DataFrame(rows).to_csv(rc.OUT / "region_clearings.csv", index=False)


if __name__ == "__main__":
    if sys.argv[1] == "base":
        base()
    elif sys.argv[1] == "clearings":
        clearings(int(sys.argv[2]))
    else:
        raise SystemExit("base | clearings <workers>")
