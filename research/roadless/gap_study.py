"""The gap complex, stage 1: the complex against the raster, and the passability sweep (NOTES, "The
gap complex: passability, enclosure and flow without the raster"; BACKLOG 4, decision (3)).

    S=research/roadless/gap_study.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S agree <ids,|@file> <procs>       # the agreement check (the gate)
    uv run python $S crossings <ids,|@file> <procs>   # where free first's crossings differ
    uv run python $S apertures <ids,|@file> <procs>   # measured apertures against true gaps
    uv run python $S sweep <ids,|@file> <procs> <s>   # the passability sweep at spacing s
    uv run python $S report                           # the summaries

`agree`: per block, the raster under Fws (lifted's one step rule, uni+apwfs: B's placement,
lifted.Placement, on the metric's mesh, h 0.5 or OVERSIZED_MESH; its enclosed and shadowed homes;
clear.free_first at area D BUDGET from nothing held, its crossings home by home) against the
complex at every S_SENS (each home's exit width; its own-nearest ring's, `_own_ring`; free first at
w 0 and at RESOLUTION, budget BUDGET). No solve: the raster's placement and free first need none.
Rows gap_rows/agree/<id>.parquet (per home) and gap_rows/agree_blocks/<id>.parquet.

`apertures`: per block and S_SENS, for every pair of footprints a passage joins (ends on two
different buildings), the shortest such passage against the pair's true distance (shapely), where
that distance is under APERTURE_MAX; `direct` where the closest approach is a gap in the open
ground beyond the band (not touching, no footprint across it). Rows
gap_rows/apertures/<id>.parquet.

`sweep`: per block at spacing s, every width of WIDTHS: the walled-in homes and their population,
free first (budget SWEEP_BUDGET: every walled-in home, its D and buildings, and the population it
leaves); per home its exit width and its bar (exit width, fill) and population; the triangles'
bars. Rows gap_rows/sweep/<spec>/<id>.parquet (per width), gap_rows/homes/<spec>/<id>.parquet.

`report`: gap_rows/summary_<name>.csv (committed; the rows are not).
"""
from __future__ import annotations

import contextlib
import io
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import gap_complex as gc  # noqa: E402
import lifted  # noqa: E402
import relax  # noqa: E402

# ------------------------------------------------------------------ the run's configuration
# Everything that can change a number is here, and named in the rows (the run's identity).
S_SENS = (0.125, 0.25, 0.5)          # the outlines' spacing (m), the sensitivity
WIDTHS = (0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0)     # the sweep's passable widths (m)
RESOLUTION = 0.125                   # the raster's sub-sample (h / S): the agreement's second w
BUDGET = relax.D_LENS                # area D 0.05: the stored free first's budget
SWEEP_BUDGET = 1.0                   # the sweep's free first: every walled-in home
APERTURE_MAX = 2.0                   # `apertures`: pairs of footprints closer than this (m)
OWN_RING_D = (0.25, 0.5, 0.75)       # `_own_ring`: offsets of the sample curves (m)
OWN_RING_STEP = 0.1                  # ... and the samples' spacing along them (m)
AREA = common.POPULATIONS["area"]
H = 0.5
ELL_M = 3.0                          # the raster's (no solve here: only its placement and reach)
K = 8
OVERSIZED_MESH = "0.5a5x8"
UNIFORM_MAX_M = 49.0                 # aperture.py's: over this many million unknowns, the composite
ROWS = HERE / "gap_rows"


def spec_of(s: float) -> gc.Spec:
    return gc.Spec(s=s, ring_m=lifted.RING_M, band_m=lifted.BAND_M)


def _params() -> lifted.Params:
    return lifted.Params(ELL_M, K, solver=lifted.CpuAMG(), cells=lifted.OpenFraction())


def _large_sizes() -> dict[str, float]:
    d = pd.read_parquet(HERE / "large82_area.parquet")
    return dict(zip(d.block, d.M_unknowns, strict=True))


def mesh_token_of(bid: str) -> str:
    return OVERSIZED_MESH if _large_sizes().get(bid, 0.0) > UNIFORM_MAX_M else "0.5"


def size_of(bid: str) -> str:
    if bid in set(common.recipients()):
        return "small"
    return "oversized" if mesh_token_of(bid) == OVERSIZED_MESH else "large"


def ids_of(arg: str) -> list[str]:
    if arg.startswith("@"):
        return Path(arg[1:]).read_text().split()
    if arg == "all":
        return sorted(set(common.recipients()) | set(_large_sizes()))
    return arg.split(",")


# ------------------------------------------------------------------ the raster's view

def raster_view(block, mesh_token: str) -> clear.Clearing:
    """clear.Clearing on the metric's mesh with B's placement and no baseline solve (only the
    placement and free first are read): common.Scorer's fields up to its solve."""
    p = _params()
    grid = common.mesh_of(mesh_token).build(block, p, AREA)
    sc = common.Scorer.__new__(common.Scorer)
    sc.block, sc.grid, sc.p, sc.rule = block, grid, p, common.Carve()
    sc.polys = np.asarray(block.buildings.outlines)
    sc.tree = shapely.STRtree(sc.polys)
    sc.free0 = grid.ff0
    sc.w = AREA.weights(sc.polys)
    sc.pl = common.placement(grid, sc.polys, p)
    sc.enclosed = sc.pl.enclosed
    return clear.Clearing.on_scorer(sc, rtol=clear.RTOL_TENSION, search=None)


def raster_shadowed(c: clear.Clearing) -> np.ndarray:
    """Placement's shadowed homes: no allowed cell within ring_m whose nearest building is the
    home (lifted.Placement.of's ring), and not enclosed."""
    g, sc = c.sc.grid, c.sc
    allowed = lifted.placeable(g, g.ff0, c.p).ravel()
    at = np.flatnonzero(allowed & (g.dist_b.ravel() <= lifted.RING_M + 1e-9))
    xy = g.xy.reshape(-1, 2)[at]
    idx = shapely.STRtree(sc.polys).query_nearest(shapely.points(xy[:, 0], xy[:, 1]),
                                                   all_matches=False)
    n = np.bincount(idx[1], minlength=len(sc.polys))
    return (n == 0) & ~sc.enclosed


FREE_LINE = re.compile(r"free: home (\d+) .*?D ([0-9.]+), clearing \[([0-9, ]*)\]")


def raster_free(c: clear.Clearing, budget: float) -> tuple[np.ndarray, list]:
    """clear.free_first from nothing held, and its crossings in order (home, buildings), read
    off its log."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        held = clear.free_first(c, budget=budget, held=np.zeros(c.n, dtype=bool))
    cr = [(int(m[1]), [int(x) for x in m[3].replace(",", " ").split()], float(m[2]))
          for m in FREE_LINE.finditer(buf.getvalue())]
    return held, cr


# ------------------------------------------------------------------ the complex's view

def _own_ring(geo: gc.Geometry, cx: gc.Complex, ex: gc.Exits) -> np.ndarray:
    """Per home: the widest exit of its own ring, the open ground within ring_m of its outline
    and nearest to it (Placement's ring), sampled on curves OWN_RING_D off the outline every
    OWN_RING_STEP: each sample's triangle's exit width (inf in the open band); -inf with no
    sample."""
    polys = geo.polys
    pts, home = [], []
    for d in OWN_RING_D:
        rings = shapely.segmentize(shapely.boundary(shapely.buffer(polys, d)), OWN_RING_STEP)
        xy, ix = shapely.get_coordinates(rings, return_index=True)
        pts.append(xy)
        home.append(ix)
    xy, home = np.concatenate(pts), np.concatenate(home)
    P = shapely.points(xy[:, 0], xy[:, 1])
    near = shapely.STRtree(polys).query_nearest(P, all_matches=False)
    mine = np.zeros(len(P), dtype=bool)
    mine[near[0]] = near[1] == home[near[0]]
    out = np.full(len(polys), -np.inf)
    ti, tt = shapely.STRtree(cx.tri_geom).query(P[mine], predicate="intersects")
    sel = np.flatnonzero(mine)
    np.maximum.at(out, home[sel[ti]], ex.width[tt])
    parts = shapely.get_parts(cx.band_open)
    parts = parts[shapely.area(parts) > 0]
    if len(parts):
        bi, _ = shapely.STRtree(parts).query(P[mine], predicate="intersects")
        out[home[sel[bi]]] = np.inf
    return out


def agree_one(bid: str) -> str | None:
    out_h = ROWS / "agree" / f"{bid}.parquet"
    out_b = ROWS / "agree_blocks" / f"{bid}.parquet"
    if out_h.exists() and out_b.exists():
        return None
    try:
        t0 = time.time()
        [b] = common.build_blocks([bid])
        mt = mesh_token_of(bid)
        c = raster_view(b, mt)
        enc = c.sc.enclosed
        shad = raster_shadowed(c)
        rheld, rcross = raster_free(c, BUDGET)
        t_r = time.time() - t0
        n = c.n
        homes = pd.DataFrame(dict(block=bid, home=np.arange(n), w=c.sc.w, cost=c.cost,
                                  raster_enclosed=enc, raster_shadowed=shad,
                                  raster_held=rheld))
        rows = []
        geo = gc.geometry(b)
        for s in S_SENS:
            t1 = time.time()
            spec = spec_of(s)
            cx = gc.build(geo, spec, np.ones(n, dtype=bool), carve=None, grow=0.0)
            ex = gc.exit_widths(cx)
            hm = gc.homes_of(geo, cx)
            he = gc.home_exits(hm, ex)
            homes[f"exit_s{s:g}"] = he
            homes[f"own_s{s:g}"] = _own_ring(geo, cx, ex)
            t_b = time.time() - t1
            for w in (0.0, RESOLUTION):
                t2 = time.time()
                fr = gc.free_first(geo, spec, w=w, grow=0.0, budget=BUDGET, cost=c.cost,
                                   pop=c.sc.w, held0=np.zeros(n, dtype=bool), cx0=cx)
                rows.append(dict(block=bid, mesh=mt, spec=spec.name, s=s, w=w,
                                 walled=int(gc.walled(he, w).sum()),
                                 held=" ".join(map(str, np.flatnonzero(fr.held))),
                                 D=float(c.cost[fr.held].sum()), left=int(fr.left.sum()),
                                 passes=fr.passes, misses=fr.misses,
                                 crossings=";".join(f"{h}:{' '.join(map(str, j))}"
                                                    for h, j, _d in fr.crossings),
                                 nt=cx.nt, t_build=t_b, t_free=time.time() - t2))
        blk = pd.DataFrame(rows)
        blk["raster_enclosed"] = int(enc.sum())
        blk["raster_shadowed"] = int(shad.sum())
        blk["raster_held"] = " ".join(map(str, np.flatnonzero(rheld)))
        blk["raster_D"] = float(c.cost[rheld].sum())
        blk["raster_crossings"] = ";".join(f"{h}:{' '.join(map(str, j))}" for h, j, _d in rcross)
        blk["t_raster"] = t_r
        out_h.parent.mkdir(parents=True, exist_ok=True)
        out_b.parent.mkdir(parents=True, exist_ok=True)
        homes.to_parquet(out_h)
        blk.to_parquet(out_b)
        print(f"{bid}: raster {int(enc.sum())} enclosed, {int(shad.sum())} shadowed; complex "
              + ", ".join(f"s {r['s']:g} w {r['w']:g} {r['walled']}" for r in rows)
              + f" ({time.time() - t0:.0f}s)", flush=True)
        return None
    except Exception as e:     # noqa: BLE001 -- a pool's task reports its block and goes on
        import traceback
        traceback.print_exc()
        return f"{bid}: {e!r}"


# ------------------------------------------------------------------ crossings

def crossings_one(bid: str) -> str | None:
    """Where free first's crossings differ (agree's rows at RESOLUTION, s 0.125): per crossing of
    either model (home, buildings), whether it frees its home in the complex at RESOLUTION (its
    exit width with the buildings cleared), in the raster (common.reached), and in the raster with
    each footprint sub-sample opened only by its own building (no cleared share spread over a mixed
    cell). Rows gap_rows/crossings/<id>.parquet."""
    out = ROWS / "crossings" / f"{bid}.parquet"
    if out.exists():
        return None
    try:
        blk = pd.read_parquet(ROWS / "agree_blocks" / f"{bid}.parquet")
        r = blk[(blk.s == 0.125) & (blk.w == RESOLUTION)].iloc[0]
        [b] = common.build_blocks([bid])
        geo = gc.geometry(b)
        n = len(geo.polys)
        c = raster_view(b, mesh_token_of(bid))
        g = c.sc.grid
        rows = []
        for model, text in (("raster", r.raster_crossings), ("complex", r.crossings)):
            for item in [x for x in text.split(";") if x]:
                h, js = item.split(":")
                h, js = int(h), [int(x) for x in js.split()]
                st = np.ones(n, dtype=bool)
                st[js] = False
                cx = gc.build(geo, spec_of(0.125), st, carve=None, grow=0.0)
                he = gc.home_exits(gc.homes_of(geo, cx), gc.exit_widths(cx))[h]
                held = ~st
                freed = bool(common.reached(c.sc.pl, g, c.open_(held), c.p)[h])
                gone = (c.lab >= 0) & held[np.maximum(c.lab, 0)]
                g2 = lifted.Grid(x0=g.x0, y0=g.y0, h=g.h, inside=g.inside, building=g.building,
                                 ground=g.ground, dist_b=g.dist_b, xy=g.xy, isub=g.isub,
                                 bsub=g.bsub & ~gone, S=g.S)
                own = bool(common.reached(c.sc.pl, g2, g2.ff0, c.p)[h])
                rows.append(dict(block=bid, model=model, home=h,
                                 buildings=" ".join(map(str, js)),
                                 D=float(c.cost[js].sum()), complex_exit=float(he),
                                 raster_frees=freed, raster_frees_own=own,
                                 raster_enclosed=bool(c.sc.enclosed[h])))
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out)
        return None
    except Exception as e:     # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f"{bid}: {e!r}"


# ------------------------------------------------------------------ apertures

def apertures_one(bid: str) -> str | None:
    out = ROWS / "apertures" / f"{bid}.parquet"
    if out.exists():
        return None
    try:
        [b] = common.build_blocks([bid])
        geo = gc.geometry(b)
        n = len(geo.polys)
        tree = shapely.STRtree(geo.polys)
        i, j = tree.query(geo.polys, predicate="dwithin", distance=APERTURE_MAX)
        keep = i < j
        i, j = i[keep], j[keep]
        true = shapely.distance(geo.polys[i], geo.polys[j])
        # a direct gap: the closest approach's segment runs through the open ground beyond the
        # band (no third footprint across it), where the complex has passages
        inner, _band_open = gc._open_ground(geo, np.ones(n, dtype=bool), lifted.BAND_M,
                                            carve=None, grow=0.0)
        shapely.prepare(inner)
        line = shapely.shortest_line(geo.polys[i], geo.polys[j])
        direct = (true > 1e-6) & shapely.covered_by(line, inner.buffer(1e-7))
        rows = pd.DataFrame(dict(block=bid, a=i, b=j, true=true, direct=direct))
        for s in S_SENS:
            cx = gc.build(geo, spec_of(s), np.ones(n, dtype=bool), carve=None,
                          grow=0.0)
            vb = cx.vb.tocsr()
            ok = np.isfinite(cx.ap)
            u, v, L = cx.pu[ok], cx.pv[ok], cx.ap[ok]
            # every (building of u, building of v) pair of every passage, its shortest
            cu, cv = vb[u].tocoo(), vb[v].tocoo()
            # expand: per passage, the buildings at each end (usually one)
            bu = pd.DataFrame(dict(e=cu.row, a=cu.col))
            bv = pd.DataFrame(dict(e=cv.row, b=cv.col))
            m = bu.merge(bv, on="e")
            m = m[m.a != m.b]
            m["L"] = L[m.e.to_numpy()]
            lo, hi = np.minimum(m.a, m.b), np.maximum(m.a, m.b)
            best = m.assign(a=lo, b=hi).groupby(["a", "b"]).L.min()
            rows[f"meas_s{s:g}"] = [best.get((x, y), np.nan) for x, y in zip(i, j, strict=True)]
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_parquet(out)
        return None
    except Exception as e:     # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f"{bid}: {e!r}"


# ------------------------------------------------------------------ the sweep

def sweep_one(job: tuple[str, float]) -> str | None:
    bid, s = job
    spec = spec_of(s)
    out = ROWS / "sweep" / spec.name / f"{bid}.parquet"
    out_h = ROWS / "homes" / spec.name / f"{bid}.parquet"
    out_b = ROWS / "bars" / spec.name / f"{bid}.parquet"
    if out.exists() and out_h.exists() and out_b.exists():
        return None
    try:
        t0 = time.time()
        [b] = common.build_blocks([bid])
        geo = gc.geometry(b)
        n = len(geo.polys)
        wpop = AREA.weights(geo.polys)
        cost = wpop / wpop.sum()
        cx = gc.build(geo, spec, np.ones(n, dtype=bool), carve=None, grow=0.0)
        t_build = time.time() - t0
        ex = gc.exit_widths(cx)
        hm = gc.homes_of(geo, cx)
        he = gc.home_exits(hm, ex)
        br = gc.bars(cx, ex)
        # each home's bar: the bar of its widest ring triangle
        R = hm.ring.tocsr()
        hbar = np.full(n, -1, dtype=np.int64)
        for h in range(n):
            t = R.indices[R.indptr[h]:R.indptr[h + 1]]
            if len(t) and np.isfinite(he[h]):
                hbar[h] = ex.bar[t[np.argmax(ex.width[t])]]
        fill_of = dict(zip(br.id, br.fill, strict=True))
        homes = pd.DataFrame(dict(block=bid, spec=spec.name, home=np.arange(n), w=wpop,
                                  cost=cost, exit=he, bar=hbar,
                                  fill=[fill_of.get(int(x), np.nan) if x >= 0 else np.nan
                                        for x in hbar]))
        bars = pd.DataFrame(dict(block=bid, spec=spec.name, bar=br.id, exit=br.exit,
                                 fill=br.fill, area=br.area))
        rows = []
        for w in WIDTHS:
            t1 = time.time()
            wall = gc.walled(he, w)
            fr = gc.free_first(geo, spec, w=w, grow=0.0, budget=SWEEP_BUDGET, cost=cost,
                               pop=wpop, held0=np.zeros(n, dtype=bool), cx0=cx)
            rows.append(dict(block=bid, spec=spec.name, s=s, w=w, walled=int(wall.sum()),
                             walled_pop=float(cost[wall].sum()), free_n=int(fr.held.sum()),
                             free_D=float(cost[fr.held].sum()),
                             left_pop=float(cost[fr.left].sum()), left=int(fr.left.sum()),
                             passes=fr.passes, misses=fr.misses,
                             held=" ".join(map(str, np.flatnonzero(fr.held))),
                             t=time.time() - t1))
        sw = pd.DataFrame(rows)
        sw["nt"], sw["t_build"], sw["t_block"] = cx.nt, t_build, time.time() - t0
        for p_, d_ in ((out, sw), (out_h, homes), (out_b, bars)):
            p_.parent.mkdir(parents=True, exist_ok=True)
            d_.to_parquet(p_)
        print(f"{bid} {spec.name}: " + ", ".join(
            f"w {r['w']:g} {r['walled']} ({r['walled_pop']:.4f}) D {r['free_D']:.4f}"
            for r in rows) + f" ({time.time() - t0:.0f}s)", flush=True)
        return None
    except Exception as e:     # noqa: BLE001
        import traceback
        traceback.print_exc()
        return f"{bid} s {s:g}: {e!r}"


def _pool(fn, jobs: list, procs: int) -> list[str]:
    if procs == 1:
        return [f for f in map(fn, jobs) if f]
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(procs, maxtasksperchild=4) as pool:
        return [f for f in pool.imap_unordered(fn, jobs) if f]


def main(argv: list[str]) -> None:
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMBA_NUM_THREADS"):
        if os.environ.get(v) != "1":
            raise SystemExit(f"{v}=1 is required (single-threaded workers)")
    cmd = argv[0]
    if cmd == "agree":
        failed = _pool(agree_one, ids_of(argv[1]), int(argv[2]))
    elif cmd == "crossings":
        failed = _pool(crossings_one, ids_of(argv[1]), int(argv[2]))
    elif cmd == "apertures":
        failed = _pool(apertures_one, ids_of(argv[1]), int(argv[2]))
    elif cmd == "sweep":
        ids = ids_of(argv[1])
        if len(ids) > 1:
            # the largest first, so the pool's tail is short (order only: a cluster task, one
            # block, has no large82_area.parquet)
            sizes = _large_sizes()
            ids = sorted(ids, key=lambda i: -sizes.get(i, 0.0))
        failed = _pool(sweep_one, [(i, float(argv[3])) for i in ids], int(argv[2]))
    else:
        raise SystemExit(__doc__)
    if failed:
        print("FAILED:\n" + "\n".join(failed), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
