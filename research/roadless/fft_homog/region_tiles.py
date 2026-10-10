"""Items 3 and 4 of the 5810@major experiment: where clearing pays, on the region's 50 m tiles
(every tile of 100 x 100 px aligned to the raster origin with every cell inside, tiles.py's),
under tiles_clear.py's two rules, unchanged:
  R20    buildings whose (ring-cell) centre lies in the tile, in random order (seed r0 1e4 + c0),
         until 20% of the tile's footprint area;
  STRIP  every building with footprint in a 4 m strip through the tile centre along the
         baseline macro gradient, clipped to the tile (the building opened whole).

    ... region_tiles.py firstorder <workers>   every tile x rule: the first-order map's dJ_2
            (L 50 m, s 10 m, inside-avg; its windows recomputed) -> OUT/tiles_firstorder.csv
    ... region_tiles.py sample <per rule>      a sample stratified by that predicted gain (deciles,
            equal counts, seed 0) -> OUT/tiles_sample.csv
    ... region_tiles.py resolve <workers> <tag>  the sample: fine dJ_2 (region_fine.FineBase.delta),
            two-scale dJ_2 and the first-order map at L 50 (s 10) and L 100 (s 25), inside-avg
            -> OUT/tiles_resolve_<tag>.csv (runs with different tags share the jobs)
    ... region_tiles.py alltiles <workers> <tag>  every non-empty tile x rule: the two-scale
            re-solve at L 50 and the first-order map at L 100 -> OUT/tiles_alltiles_<tag>.csv
    ... region_tiles.py restrict <src,...>     item 4: tiles ranked by a predicted gain per
            displaced m^2 (lin50, lin100, ts50); the stored clearings' share inside the top
            tiles covering k x the budget's footprint area, and the truncated clearings ->
            OUT/restrict_<srcs>.csv, OUT/restrict_clearings_<srcs>.json (region_score_gpu.py),
            OUT/restrict_tiles_<srcs>.json (the tile sets, for region_topup.py)
"""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]
import fftk  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import region_common as rc  # noqa: E402
import ts  # noqa: E402
from region_macro import MacroBase  # noqa: E402
from scipy import ndimage  # noqa: E402

TILE = 100                 # px: 50 m
K = 4                      # macro element side, px (H 2 m)
BUDGET = 0.05              # D: the stored clearings' budget (population = footprint area)
KS = (2, 3, 5)
G: dict = {}               # the parent's state, inherited by the forked workers


# ------------------------------------------------------------------------- clearings
def clearing(r0: int, c0: int, rule: str) -> np.ndarray:
    fab, n = G["fab"], TILE
    if rule == "R20":
        inn = ((fab.cnt > 0) & (fab.cy >= r0) & (fab.cy < r0 + n) & (fab.cx >= c0)
               & (fab.cx < c0 + n))
        cand = np.flatnonzero(inn)
        rng = np.random.default_rng(r0 * 10000 + c0)
        cand = rng.permutation(cand)
        csum = np.cumsum(fab.area[cand])
        return (cand[: int(np.searchsorted(csum, 0.2 * fab.area[cand].sum())) + 1]
                if len(cand) else cand)
    rcen, ccen = r0 + n // 2, c0 + n // 2
    gx, gy = G["grad"][(r0, c0)]
    nrm = np.hypot(gx, gy)
    ux, uy = (gx / nrm, gy / nrm) if nrm > 0 else (0.0, 1.0)
    ii, jj = np.mgrid[r0:r0 + n, c0:c0 + n]
    dperp = np.abs((jj - ccen) * uy - (ii - rcen) * ux)
    cells = (ii * fab.o.shape[1] + jj)[dperp <= 2.0 / rc.H]
    return np.unique(fab.sb[np.isin(fab.sc, cells)])


def macro_grad(mac: MacroBase, r: int, c: int) -> tuple[float, float]:
    """(dU/dx, dU/dy) per px at cell (r, c): ts.evaluate's formula."""
    Un, k = mac.nodal(mac.U0), mac.k
    Ir, Jn, b, a = r // k, c // k, r / k - r // k, c / k - c // k
    u00, u01, u10, u11 = Un[Ir, Jn], Un[Ir, Jn + 1], Un[Ir + 1, Jn], Un[Ir + 1, Jn + 1]
    return (((u01 - u00) * (1 - b) + (u11 - u10) * b) / k,
            ((u10 - u00) * (1 - a) + (u11 - u01) * a) / k)


# ------------------------------------------------------------------------- windows
def window_sigma(oc: np.ndarray, inside: np.ndarray, tf, bi: int, bj: int):
    """ts._window's sigma and sigma_in for block (bi, bj) of field tf, from the open fraction
    oc (outside the region and the raster: open), without the source corrector."""
    n, s = tf.n, tf.s
    r0, c0 = bi * s + s // 2 - n // 2, bj * s + s // 2 - n // 2
    ny, nx = oc.shape
    ow = np.ones((n, n))
    iw = np.zeros((n, n), dtype=bool)
    a0, a1, b0, b1 = max(r0, 0), min(r0 + n, ny), max(c0, 0), min(c0 + n, nx)
    if a1 > a0 and b1 > b0:
        sl = (slice(a0, a1), slice(b0, b1))
        dst = (slice(a0 - r0, a1 - r0), slice(b0 - c0, b1 - c0))
        ow[dst] = np.where(inside[sl], oc[sl], 1.0)
        iw[dst] = inside[sl]
    P = fftk.Problem.of(ow)
    u, _it, _, _res = fftk.solve(P, np.eye(2), "cg", 1e-8, 5000, 1.0)
    S, _ = P.sigma_of(u[:2])
    if iw.all() or not iw.any():
        return S, S
    gx, gy = fftk.grad(u[:2])
    Q = np.array([[np.mean((P.cx * (gx[j] + (j == 0)))[iw]) for j in range(2)],
                  [np.mean((P.cy * (gy[j] + (j == 1)))[iw]) for j in range(2)]])
    Eg = np.array([[np.mean((gx[j] + (j == 0))[iw]) for j in range(2)],
                   [np.mean((gy[j] + (j == 1))[iw]) for j in range(2)]])
    S_in = Q @ np.linalg.inv(Eg)
    return S, 0.5 * (S_in + S_in.T)


def windows_hit(tf, changed_box: np.ndarray, r_lo: int, c_lo: int) -> list[tuple[int, int]]:
    """Blocks of tf (computed ones) whose window holds a changed cell; changed_box is the
    changed mask on rows r_lo.., columns c_lo.. (ts.windows_hit, locally)."""
    n, s = tf.n, tf.s
    nbi, nbj = tf.have.shape
    rows, cols = np.nonzero(changed_box)
    rows, cols = rows + r_lo, cols + c_lo
    off = s // 2 - n // 2                      # window rows: bi s + off .. + n
    bi_lo = max(int(np.floor((rows.min() - off - n + 1) / s)), 0)
    bi_hi = min(int(np.floor((rows.max() - off) / s)), nbi - 1)
    bj_lo = max(int(np.floor((cols.min() - off - n + 1) / s)), 0)
    bj_hi = min(int(np.floor((cols.max() - off) / s)), nbj - 1)
    integ = np.pad(changed_box.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H_, W_ = changed_box.shape
    out = []
    for bi in range(bi_lo, bi_hi + 1):
        a0 = min(max(bi * s + off - r_lo, 0), H_)
        a1 = min(max(bi * s + off + n - r_lo, 0), H_)
        for bj in range(bj_lo, bj_hi + 1):
            if not tf.have[bi, bj]:
                continue
            b0 = min(max(bj * s + off - c_lo, 0), W_)
            b1 = min(max(bj * s + off + n - c_lo, 0), W_)
            if integ[a1, b1] - integ[a0, b1] - integ[a1, b0] + integ[a0, b0] > 0:
                out.append((bi, bj))
    return out


def dsigma(key: str, oc: np.ndarray, changed_box, r_lo, c_lo) -> tuple[np.ndarray, int]:
    """The inside-avg block tensors' change (ts.fill(., None, True) after - before) with the
    windows holding a changed cell recomputed; and their count."""
    tf, fill_ij = G[f"tf{key}"], G[f"fill{key}"]
    hit = windows_hit(tf, changed_box, r_lo, c_lo)
    D = np.zeros_like(tf.sigma_in)
    for bi, bj in hit:
        D[bi, bj] = window_sigma(oc, G["fab"].inside, tf, bi, bj)[1] - tf.sigma_in[bi, bj]
    return D[fill_ij], len(hit)


def changed_of(oc: np.ndarray):
    fab = G["fab"]
    changed = oc != fab.o
    rows, cols = np.nonzero(changed)
    r_lo, r_hi, c_lo, c_hi = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return changed[r_lo:r_hi, c_lo:c_hi], r_lo, c_lo, float(((oc - fab.o)[changed]).sum()
                                                            * rc.H * rc.H)


# ------------------------------------------------------------------------- setup
def setup(fields: tuple[str, ...], fine: bool) -> None:
    fab = rc.load_fabric()
    G["fab"] = fab
    for key, (L, s) in (("50", (50, 10)), ("100", (100, 25))):
        if key not in fields:
            continue
        tf = ts.load_field(str(rc.OUT / f"fields_L{L}_s{s}.npz"))
        _, (ii, jj) = ndimage.distance_transform_edt(~tf.have, return_indices=True)
        G[f"tf{key}"], G[f"fill{key}"] = tf, (ii, jj)
        t = time.perf_counter()
        G[f"mac{key}"] = MacroBase(ts.fill(tf, None, True), tf.s, K, fab, adjoint=True)
        print(f"macro L {L}: {G[f'mac{key}'].N} unknowns, J0 {G[f'mac{key}'].J0:.8g}, "
              f"{time.perf_counter() - t:.0f} s", flush=True)
    G["tiles"] = rc.full_tiles(fab.inside, TILE)
    m50 = G["mac50"]
    G["grad"] = {(r0, c0): macro_grad(m50, r0 + TILE // 2, c0 + TILE // 2)
                 for r0, c0 in G["tiles"]}
    if fine:
        import region_fine
        t = time.perf_counter()
        G["fine"] = region_fine.FineBase(fab, u0=np.load(rc.OUT / "fine_u0.npy"))
        print(f"fine base: {G['fine'].N0} unknowns, J0 {G['fine'].J0:.8g}, "
              f"{time.perf_counter() - t:.0f} s", flush=True)


# ------------------------------------------------------------------------- jobs
def job_firstorder(args):
    r0, c0, rule = args
    t = time.perf_counter()
    fab = G["fab"]
    ids = clearing(r0, c0, rule)
    row = dict(r0=r0, c0=c0, rule=rule, n_bldg=len(ids),
               footprint_m2=float(fab.area[ids].sum()))
    if len(ids) == 0:
        return {**row, "cleared_m2": 0.0, "dJ_lin50": 0.0, "windows50": 0,
                "seconds": time.perf_counter() - t}
    oc = rc.open_buildings(fab, ids)
    box, r_lo, c_lo, m2 = changed_of(oc)
    D, nwin = dsigma("50", oc, box, r_lo, c_lo)
    lin = G["mac50"].first_order(D)
    return {**row, "cleared_m2": m2, "dJ_lin50": lin, "windows50": nwin,
            "seconds": time.perf_counter() - t}


CLAIMS = rc.OUT / "claims_resolve"


def job_resolve(args):
    """None if another process has claimed this job (several resolve runs share the sample:
    each claims a job by creating its marker file first)."""
    r0, c0, rule = args
    try:
        os.close(os.open(CLAIMS / f"{r0}_{c0}_{rule}", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return None
    t0 = time.perf_counter()
    fab = G["fab"]
    ids = clearing(r0, c0, rule)
    oc = rc.open_buildings(fab, ids)
    box, r_lo, c_lo, m2 = changed_of(oc)
    row = dict(r0=r0, c0=c0, rule=rule, n_bldg=len(ids), cleared_m2=m2)
    fd = G["fine"].delta(oc, rtol=1e-8)
    row.update(dJ_fine=fd["dJ"], dP_fine=fd["dP"], fine_iters=fd["iters"], fine_new=fd["n_new"],
               fine_s=fd["seconds"])
    for key in ("50", "100"):
        t = time.perf_counter()
        D, nwin = dsigma(key, oc, box, r_lo, c_lo)
        tw = time.perf_counter() - t
        md = G[f"mac{key}"].delta(D, rtol=1e-8)
        row.update({f"dJ_ts{key}": md["dJ"], f"dJ_lin{key}": md["dJ_lin"],
                    f"windows{key}": nwin, f"windows{key}_s": tw,
                    f"macro{key}_s": md["seconds"], f"macro{key}_iters": md["iters"]})
    row["seconds"] = time.perf_counter() - t0
    return row


CLAIMS_ALL = rc.OUT / "claims_alltiles"


def job_alltiles(args):
    """Every non-empty tile x rule: the two-scale re-solve at L 50 and the first-order map at
    L 100 (alternative rankings for item 4: the L 50 first-order map overshoots where a 50 m
    window is nearly blocked). None if another run claimed it."""
    r0, c0, rule = args
    try:
        os.close(os.open(CLAIMS_ALL / f"{r0}_{c0}_{rule}", os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return None
    t0 = time.perf_counter()
    fab = G["fab"]
    ids = clearing(r0, c0, rule)
    oc = rc.open_buildings(fab, ids)
    box, r_lo, c_lo, m2 = changed_of(oc)
    row = dict(r0=r0, c0=c0, rule=rule, n_bldg=len(ids), cleared_m2=m2)
    D, nwin = dsigma("50", oc, box, r_lo, c_lo)
    md = G["mac50"].delta(D, rtol=1e-8)
    row.update(dJ_ts50=md["dJ"], dJ_lin50=md["dJ_lin"], windows50=nwin)
    D, nwin = dsigma("100", oc, box, r_lo, c_lo)
    row.update(dJ_lin100=G["mac100"].first_order(D), windows100=nwin,
               seconds=time.perf_counter() - t0)
    return row


def run_pool(fn, jobs, workers: int, out: Path, every: int = 50) -> pd.DataFrame:
    rows = []
    t = time.time()
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
            if row is None:
                continue
            rows.append(row)
            if i % every == 0 or i == len(jobs) - 1:
                shown = {a: (round(b, 4) if isinstance(b, float) else b) for a, b in row.items()}
                print(f"{i + 1}/{len(jobs)} {time.time() - t:.0f} s: {shown}", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return df


# ------------------------------------------------------------------------- item 4
def restrict(sources: tuple[str, ...]) -> None:
    """Rank the tiles by a predicted gain per displaced m^2 (`lin50`: the first-order map at
    L 50, tiles_firstorder.csv, the experiment's ranking; `lin100`, `ts50`: the first-order map
    at L 100 and the two-scale re-solve at L 50, tiles_alltiles_*.csv), take the top tiles
    whose footprint area (buildings by polygon centroid) reaches k x the budget's, and keep
    each stored clearing's buildings inside them."""
    fab = rc.load_fabric()
    fo = pd.read_csv(rc.OUT / "tiles_firstorder.csv")
    # the two-scale L 50 and first-order L 100 predictions: the all-tiles runs, and the sampled
    # tiles' from the resolve runs (the same computation; their claims skip them in alltiles)
    alls = [pd.read_csv(p) for p in sorted(rc.OUT.glob("tiles_alltiles_*.csv"))]
    alls += [pd.read_csv(p)[["r0", "c0", "rule", "n_bldg", "cleared_m2", "dJ_ts50", "dJ_lin50",
                             "dJ_lin100"]]
             for p in sorted(rc.OUT.glob("tiles_resolve_*.csv")) if "summary" not in p.name]
    al = pd.concat(alls, ignore_index=True).drop_duplicates(["r0", "c0", "rule"]) if alls else None
    tot = float(fab.area.sum())
    budget_m2 = BUDGET * tot
    # each building's tile: the one holding its polygon centroid (fully-inside tiles only)
    tiles = rc.full_tiles(fab.inside, TILE)
    tid = {t: i for i, t in enumerate(tiles)}
    br, bc = np.floor(fab.py + 0.5).astype(int), np.floor(fab.px + 0.5).astype(int)
    btile = np.array([tid.get((r - r % TILE, c - c % TILE), -1) for r, c in zip(br, bc,
                                                                               strict=True)])
    tile_fp = np.bincount(btile[btile >= 0], weights=fab.area[btile >= 0], minlength=len(tiles))
    stored = rc.stored_clearings()
    rows, todo, chosen = [], {}, {}
    for name, ids in stored.items():
        todo[f"{name}_alltiles"] = ids[btile[ids] >= 0].tolist()     # the ceiling of any tile set
        a = fab.area[ids]
        rows.append(dict(ranking="all tiles", rule="-", k=0, clearing=name, tiles=len(tiles),
                         tile_footprint_m2=float(tile_fp.sum()), budget_m2=budget_m2,
                         share_cleared_in_tiles=float(a[btile[ids] >= 0].sum() / a.sum()),
                         kept_buildings=int((btile[ids] >= 0).sum()),
                         kept_D=float(a[btile[ids] >= 0].sum() / tot), D=float(a.sum() / tot)))
    for src in sources:
        table, col = (fo, "dJ_lin50") if src == "lin50" else (al, f"dJ_{src}")
        for rank_rule in ("R20", "STRIP"):
            g = table[(table.rule == rank_rule) & (table.n_bldg > 0)].copy()
            g["tile"] = [tid[(r, c)] for r, c in zip(g.r0, g.c0, strict=True)]
            g["gain_m2"] = -g[col] / g.cleared_m2
            # ties (one building in two tiles' clearings) broken by tile index, stably
            g = g.sort_values(["gain_m2", "tile"], ascending=[False, True], kind="mergesort")
            cum = np.cumsum(tile_fp[g.tile.to_numpy()])
            for k in KS:
                m = int(np.searchsorted(cum, k * budget_m2)) + 1
                sel = np.zeros(len(tiles), dtype=bool)
                sel[g.tile.to_numpy()[:m]] = True
                chosen[f"{src}_{rank_rule}_k{k}"] = np.flatnonzero(sel).tolist()
                for name, ids in stored.items():
                    inside = (btile[ids] >= 0) & sel[np.maximum(btile[ids], 0)]
                    a = fab.area[ids]
                    todo[f"{name}_{src}_{rank_rule}_k{k}"] = ids[inside].tolist()
                    rows.append(dict(ranking=src, rule=rank_rule, k=k, clearing=name, tiles=m,
                                     tile_footprint_m2=float(cum[m - 1]), budget_m2=budget_m2,
                                     share_cleared_in_tiles=float(a[inside].sum() / a.sum()),
                                     kept_buildings=int(inside.sum()),
                                     kept_D=float(a[inside].sum() / tot), D=float(a.sum() / tot)))
    df = pd.DataFrame(rows)
    tag = "_".join(sources)
    df.to_csv(rc.OUT / f"restrict_{tag}.csv", index=False)
    (rc.OUT / f"restrict_clearings_{tag}.json").write_text(json.dumps(todo))
    (rc.OUT / f"restrict_tiles_{tag}.json").write_text(json.dumps(chosen))
    pd.set_option("display.width", 200)
    print(df.to_string(), flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "firstorder":
        workers = int(sys.argv[2])
        assert workers <= 12
        with rc.Monitor("tiles: setup (L 50 macro + adjoint)"):
            setup(("50",), fine=False)
        jobs = [(r0, c0, rule) for r0, c0 in G["tiles"] for rule in ("R20", "STRIP")]
        print(len(G["tiles"]), "tiles,", len(jobs), "jobs", flush=True)
        with rc.Monitor(f"tiles: first-order map L 50, every tile x rule, {workers} workers"):
            run_pool(job_firstorder, jobs, workers, rc.OUT / "tiles_firstorder.csv", every=200)
    elif mode == "sample":
        per = int(sys.argv[2])
        fo = pd.read_csv(rc.OUT / "tiles_firstorder.csv")
        rng = np.random.default_rng(0)
        out = []
        for _rule, g in fo.groupby("rule"):
            g = g[g.n_bldg > 0].sort_values("dJ_lin50").reset_index(drop=True)
            dec = np.minimum((np.arange(len(g)) * 10) // len(g), 9)
            for d in range(10):
                idx = np.flatnonzero(dec == d)
                pick = rng.choice(idx, size=min(len(idx), -(-per // 10)), replace=False)
                out.append(g.iloc[np.sort(pick)].assign(decile=d))
        s = pd.concat(out, ignore_index=True)
        s.to_csv(rc.OUT / "tiles_sample.csv", index=False)
        print(s.groupby("rule").size(), flush=True)
    elif mode == "resolve":
        workers = int(sys.argv[2])
        assert workers <= 12
        with rc.Monitor("tiles: setup (fine base, L 50 and L 100 macro + adjoints)"):
            setup(("50", "100"), fine=True)
        s = pd.read_csv(rc.OUT / "tiles_sample.csv")
        jobs = [(int(r), int(c), rule) for r, c, rule in zip(s.r0, s.c0, s.rule, strict=True)]
        CLAIMS.mkdir(exist_ok=True)
        with rc.Monitor(f"tiles: resolve sampled tile-rules ({sys.argv[3]}), {workers} workers"):
            run_pool(job_resolve, jobs, workers, rc.OUT / f"tiles_resolve_{sys.argv[3]}.csv",
                     every=10)
    elif mode == "alltiles":
        workers = int(sys.argv[2])
        assert workers <= 12
        with rc.Monitor("tiles: setup (L 50 and L 100 macro + adjoints)"):
            setup(("50", "100"), fine=False)
        fo = pd.read_csv(rc.OUT / "tiles_firstorder.csv")
        fo = fo[fo.n_bldg > 0]
        jobs = [(int(r), int(c), rule) for r, c, rule in zip(fo.r0, fo.c0, fo.rule, strict=True)]
        CLAIMS_ALL.mkdir(exist_ok=True)
        smp = pd.read_csv(rc.OUT / "tiles_sample.csv")      # resolved there: skip them here
        for r, c, rule in zip(smp.r0, smp.c0, smp.rule, strict=True):
            (CLAIMS_ALL / f"{r}_{c}_{rule}").touch()
        with rc.Monitor(f"tiles: two-scale L 50 + first-order L 100, every non-empty tile x rule "
                        f"({sys.argv[3]}), {workers} workers"):
            run_pool(job_alltiles, jobs, workers, rc.OUT / f"tiles_alltiles_{sys.argv[3]}.csv",
                     every=100)
    elif mode == "restrict":
        restrict(tuple(sys.argv[2].split(",")))
    else:
        raise SystemExit(__doc__)
