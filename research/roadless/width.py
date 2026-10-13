"""Uniform conductance with an effective width: does it make corridors form? (BACKLOG 4, the
metric decisions, (3); owner 2026-10-12: "resistance not just within buildings but radiate out
from them, or just have conductance be some monotone function of the distance to the nearest
footprint"; NOTES, "Uniform conductance with an effective width").

The variants (VARIANTS: a name and its lifted.conductance_of spec, the spec in every row): U
`uni+apwfs` (a cell by its open fraction); B_r `uni+b<r>+apwfs` (lifted.Blurred: min(o, the open
share of the disk of radius r m), r 0.5, 1, 1.5); D_s `uni+d<s>+apwfs` (lifted.WallDistance:
min(1, d / s) per open sub-sample, d to the nearest footprint, s 0.5, 1); S `ss100k2n2r30+apwfs`
(the sightline metric); S_t `ss100k2n2r30t+apwfs` (lifted.TurnBoosted: its boost on the turning
edges too). h 0.5, ell 3 m, K 8, area population, B's placement. Three lenses (LENSES): P (the
population's demand, power 1), J_2 (lens (c)) and P_{1/2} (common.PriorityDemand(0.5), power 1).
Rows and summaries under ROWS, width_rows/<lifted.APERTURE>/; the study's own (NOTES, "Uniform
conductance with an effective width"), under the step rule before the face aperture, are
width_rows' channel.parquet, score/, homes/, flows/, simp/, cross/, maps/ and summary_*.csv.

    S=research/roadless/width.py     # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S channel <procs>                  # Q2, the width response, the alignment
    uv run python $S score <id> <variant> <offset> <cpu|gpu>     # Q1, Q3, Q4: one block, variant
    uv run python $S score-all <id> <0|shift> <cpu|gpu>          # ... every variant in turn
    uv run python $S simp <id> <variant>              # Q4: SIMP base, objective P, free first
    uv run python $S cross <id> <variant>             # Q4: the SIMP clearings under a variant
    uv run python $S tars <dir> <ids,> <score|cross>  # a tar per block of a task's inputs
    uv run python $S maps <ids,>                      # Q1: the maps (PNGs), from the flows
    uv run python $S report                           # every question's tables, the summaries

`channel` (here, `procs` processes): the W x 40 m corridor of "The sightline metric's angle
error" with its walls footprints (walled_channel: a dead end behind the demand, the street at the
far end, the lattice checks.channel's), at the angles of sightline_angle.ANGLES and the widths
WIDTHS_ANGLE (Q2); at 0 and 30 deg over WIDTHS (how conduction grows with width); and at 0 deg
under ALIGN_SHIFTS lattice offsets (how much a sub-cell shift moves P); and at WIDTHS_SUBCELL
under SUBCELL_SHIFTS offsets, widths where a shift adds or drops a column of sub-samples. Rows
ROWS/channel.parquet.

`score`: one block under one variant on the lattice at OFFSETS[offset]: the baseline, the stored
clearings (STORED, through clearings.Store) and item 4's road prefixes at area D 0.05 (ROADS;
offset 0 only), each under the three lenses with B (i) and the escape-time tails; on the sentinel
at offset 0 also the no-buildings prior (every inside sub-sample open, the homes' demand as
placed), and the flows of the baseline, SIMP's default and the prior (the flux per cell) and their
corridor indices
(`_indices`); per home its escape time at baseline and under SIMP's default, and its bottleneck
width (`_bottleneck`). Rows ROWS/score/o<offset>/<variant>/<id>.parquet, homes
ROWS/homes/o<offset>/<variant>/<id>.parquet, flows ROWS/flows/<variant>/<id>.parquet.

`simp`: relax.one at SIMP_PLAN (SIMP base), objective P (the population's demand, power 1), area
D 0.05, free first (clear.free_first), under the variant, on the GPU; rows
ROWS/simp/<variant>/<id>.parquet. `cross`: every variant's SIMP clearing of a block scored under
one of CROSS, on the GPU; rows ROWS/cross/<variant>/<id>.parquet.
"""
from __future__ import annotations

import multiprocessing
import socket
import sys
import tarfile
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import ndimage
from shapely.geometry import LineString, Polygon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import clearings  # noqa: E402
import common  # noqa: E402
import lifted  # noqa: E402
import relax  # noqa: E402
import road_rescore as rr  # noqa: E402
from simp_compare import TUNING  # noqa: E402

# ------------------------------------------------------------------ the run's configuration
# Everything that can change a number is here, and named in the rows (the run's identity).
VARIANTS = {"U": "uni+apwfs", "B0.5": "uni+b0.5+apwfs", "B1": "uni+b1+apwfs",
            "B1.5": "uni+b1.5+apwfs", "D0.5": "uni+d0.5+apwfs", "D1": "uni+d1+apwfs",
            "S": "ss100k2n2r30+apwfs", "St": "ss100k2n2r30t+apwfs"}
H = 0.5
ELL_M = 3.0
K = 8
# offset_check.OFFSETS: the metric's lattice offset, then shifts by a half, a quarter and three
# quarters of a cell on both axes
OFFSETS = (lifted.OFFSET, (0.8713, 0.6931), (0.1213, 0.9431), (0.6213, 0.4431))
SENTINEL = ("ZAF.9.3.1_1_22422", "ZAF.9.3.1_1_30848", "ZAF.9.3.1_1_5810", "ZAF.9.5.4_1_9712")
# Q3's blocks and, per block, the shift that moved SIMP's default's Lens A most from the metric's
# offset under U (res_rows/uni/<id>_p2_h0.5_offsets.parquet, 2026-10-10, preB): 20543 0.503 ->
# 0.356, 38366 0.464 -> 0.371, 17608 0.760 -> 0.698
ALIGN = {"ZAF.9.3.1_1_20543": 2, "ZAF.9.3.1_1_38366": 2, "ZAF.9.3.1_1_17608": 3}
STORED = {"simp_default": "relax_rows/uni/fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2/D0.05.p2",
          "cheap": "polish_rows/uni/S0.01cat.P64w8/D0.05.p2"}
ROADS = ("clearance", "clearance_looped", "cycle_native", "cycle_native_betweenness_contrast",
         "resistance_lp", "euclidean_grid")
ROAD_ROWS = HERE / "road_rows"               # item 4's rows; its networks under scratch/
BUDGET = relax.D_LENS                        # area D 0.05
SCORE_RTOL = clear.RTOL_SCORE                # every score: the optimizers' exact one
POP = common.PopulationDemand()
PQ = common.PriorityDemand(0.5)
LENSES = ("P", "J2", "Pq0.5")
SIMP_PLAN = relax.plan_of("fw0.q3.i10.t0.001.e0.0001.k1s1e-06")    # SIMP base
SIMP_DEVICE = "gpu"
CROSS = ("U", "B0.5", "B1", "B1.5")         # the conductances the SIMP clearings are scored under
CROSS_DEVICE = "gpu"                        # cross: minutes on a card, the CPU nodes busy scoring
# corridor indices (`_indices`)
OPEN_M = (1.0, 2.0)        # the open-lane share: flux in cells at least this far from a footprint
COHERENCE_M = 2.0          # the coherence's neighbourhood: a square of this half-side
TOP = 0.10                 # the concentration: the flux share of the top 10% of the open area
WIDTH_STEPS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)   # bottleneck widths tried (m)
ROWS = HERE / "width_rows" / lifted.APERTURE   # every row and summary, by the step rule

# the channel checks
L_C = 40.0
WALL_M = 3.0               # the walls' thickness: beyond every disk (1.5 m + a cell) here
ANGLES = (0, 2, 4, 6, 8, 10, 15, 20, 26.57, 30, 35, 40, 45)     # sightline_angle.ANGLES
WIDTHS_ANGLE = (1.0, 1.5, 2.0, 4.0)
WIDTHS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
ALIGN_SHIFTS = 8           # lattice offsets k / 8 of a cell (k = 0 .. 7) on both axes
WIDTHS_ALIGN = (0.75, 1.0, 1.5, 2.0)
SUBCELL_SHIFTS = 16        # the sub-cell check: k / 16 of a cell, so a shift can add or drop a
                           # column of sub-samples ...
WIDTHS_SUBCELL = (0.19, 0.3, 0.44, 0.69, 1.06)   # ... at widths between two counts of them


# ------------------------------------------------------------------ conductances

def params(spec: str, device: str) -> lifted.Params:
    cond = lifted.conductance_of(spec, lifted.scans_of(device))
    return lifted.Params(ELL_M, K, along=cond.along, cells=cond.cells,
                         solver=lifted.solver_of(device))


# ------------------------------------------------------------------ the channel (Q2)

def walled_channel(alpha_deg: float, W: float, h: float,
                   offset: tuple[float, float]) -> tuple[lifted.Grid, np.ndarray]:
    """checks.channel's W x L_C corridor rotated alpha_deg, its walls footprints WALL_M thick on
    both sides and behind its start (so the factors that look past a cell see walls, not the
    open street side of a block edge), the street the block's far edge. The lattice is
    checks.channel's at `offset` (the corridor's own bounds), shifted a whole number of cells to
    start below the walled block. Demand: the open non-ground cells within 1 m of the dead end,
    summing to 1."""
    a = np.radians(alpha_deg)
    e, n = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])

    def pt(s: float, t: float) -> tuple[float, float]:
        x = s * e + t * n
        return float(x[0]), float(x[1])

    def rect(s0: float, s1: float, t0: float, t1: float) -> Polygon:
        return Polygon([pt(s0, t0), pt(s1, t0), pt(s1, t1), pt(s0, t1)])

    w2, T = W / 2, WALL_M
    block = rect(-T, L_C, -w2 - T, w2 + T)
    walls = [rect(-T, L_C, w2, w2 + T), rect(-T, L_C, -w2 - T, -w2), rect(-T, 0.0, -w2, w2)]
    street = LineString([pt(L_C, -w2 - T), pt(L_C, w2 + T)])
    cx, cy = rect(0.0, L_C, -w2, w2).bounds[:2]
    bx, by = block.bounds[:2]
    off = ((offset[0] + (cx - bx) / h) % 1.0, (offset[1] + (cy - by) / h) % 1.0)
    grid = lifted.Grid.of(block, walls, [street], h, band_m=1.0, offset=off)
    src = LineString([pt(0.0, -w2), pt(0.0, w2)]).buffer(1.0)
    f = (grid.mask_of(src) & (grid.ff0 > 0) & ~grid.ground).astype(float)
    return grid, f / f.sum()


class ChannelJob(NamedTuple):
    check: str          # angle | width | align | subcell
    variant: str        # a VARIANTS key
    W: float
    angle: float
    shift: int          # align, subcell: k of ALIGN_SHIFTS, SUBCELL_SHIFTS (0: the metric's)


def _channel_one(job: ChannelJob) -> dict:
    k = job.shift / (SUBCELL_SHIFTS if job.check == "subcell" else ALIGN_SHIFTS)
    offset = ((lifted.OFFSET[0] + k) % 1.0, (lifted.OFFSET[1] + k) % 1.0)
    grid, f = walled_channel(job.angle, job.W, H, offset)
    t0 = time.time()
    P = lifted.solve(grid, grid.ff0, f, params(VARIANTS[job.variant], "cpu")).P
    return dict(job._asdict(), P=P, t=time.time() - t0)


def channel(procs: int) -> None:
    """Every channel job, `procs` processes; rows ROWS/channel.parquet."""
    rr._check_single_threaded()
    names = list(VARIANTS)
    jobs = ([ChannelJob("angle", v, W, a, 0) for v in names for W in WIDTHS_ANGLE for a in ANGLES]
            + [ChannelJob("width", v, W, a, 0) for v in names for W in WIDTHS for a in (0, 30)]
            + [ChannelJob("align", v, W, 0.0, s)
               for v in names for W in WIDTHS_ALIGN
               for s in range(ALIGN_SHIFTS)]
            + [ChannelJob("subcell", v, W, 0.0, s) for v in names for W in WIDTHS_SUBCELL
               for s in range(SUBCELL_SHIFTS)])
    t0 = time.time()
    with multiprocessing.get_context("fork").Pool(procs) as pool:
        rows = pool.map(_channel_one, jobs, chunksize=1)
    out = ROWS / "channel.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(pd.DataFrame(rows), out)
    print(f"{len(rows)} channel solves in {time.time() - t0:.0f}s -> {out}", flush=True)


# ------------------------------------------------------------------ one block, one variant

def _open_of(g: lifted.Grid, lab: np.ndarray, cleared: np.ndarray, n: int) -> np.ndarray:
    """The open field with the buildings `cleared` (positions) gone (clear.Clearing.open_)."""
    removed = np.zeros(n, dtype=bool)
    removed[cleared] = True
    gone = (lab >= 0) & removed[np.maximum(lab, 0)]
    return (g.isub & (~g.bsub | gone)).mean(axis=-1)


def _wall_distance(g: lifted.Grid, o: np.ndarray) -> np.ndarray:
    """(ny, nx) metres from each cell to the nearest footprint still standing in the open field
    `o` (WallDistance's: the fine lattice's distance transform less half a sub-sample, averaged
    over the cell's sub-samples; inf with no footprint standing)."""
    S = g.S
    stand = lifted._closed(g, o, np) >= 1.0
    ny, nx = g.inside.shape
    if not stand.any():
        return np.full((ny, nx), np.inf)
    d = ndimage.distance_transform_edt(~stand) * (g.h / S) - g.h / (2 * S)
    return d.reshape(ny, S, nx, S).mean(axis=(1, 3))


class Flux(NamedTuple):
    """Per grid cell (ny, nx): S the flux, sum over headings of |signed flux| (current_map.flow's
    layers: half of every along edge's to each end), and J the net flux vector (Jx, Jy)."""
    S: np.ndarray
    Jx: np.ndarray
    Jy: np.ndarray


def _flux(g: lifted.Grid, o: np.ndarray, p: lifted.Params, sol: lifted.Solution) -> Flux:
    v, _th, _m, _gap = lifted.axes(p.K)
    e = v / np.linalg.norm(v, axis=1)[:, None]
    free = np.flatnonzero(o.ravel() > 0)
    N = o.size
    uk = np.zeros((len(free), p.K))
    uk[sol.unk_cell >= 0] = sol.u.reshape(-1, p.K)
    S, Jx, Jy = np.zeros(N), np.zeros(N), np.zeros(N)
    th = p.solver.to_host
    for i, (ca, cb, _ka, _kb, w) in enumerate(lifted.edges(g, o, p)):
        if i >= p.K:                            # the turning families carry no flux
            break
        ca, cb = th(ca), th(cb)
        q = th(w) * (uk[ca, i] - uk[cb, i])
        fa, fb = free[ca], free[cb]
        half = 0.5 * (np.bincount(fa, q, minlength=N) + np.bincount(fb, q, minlength=N))
        S += np.abs(half)
        Jx += half * e[i, 0]
        Jy += half * e[i, 1]
    shape = g.inside.shape
    return Flux(S=S.reshape(shape), Jx=Jx.reshape(shape), Jy=Jy.reshape(shape))


def _indices(g: lifted.Grid, o: np.ndarray, fx: Flux, dist: np.ndarray) -> dict[str, float]:
    """The corridor indices of a flow (NOTES): `open<d>` the share of the flux in cells at least
    d m from a standing footprint, `area<d>` those cells' share of the open area (open<d> /
    area<d> above 1: the walkers favour the open lanes); `coh` the flux-weighted coherence of the
    net flux over a square of half-side COHERENCE_M, sum_c |sum_N J| / sum_c sum_N |J| (1: every
    neighbour's flow parallel; lower where streams split, merge or cross); `top` the share of the
    flux in the TOP of the open area carrying the most."""
    on = o > 0
    S = np.where(on, fx.S, 0.0)
    out: dict[str, float] = {}
    tot, area = float(S.sum()), float(o[on].sum())
    for dm in OPEN_M:
        far = on & (dist >= dm)
        out[f"open{dm:g}"] = float(S[far].sum() / tot)
        out[f"area{dm:g}"] = float(o[far].sum() / area)
    m = 2 * int(round(COHERENCE_M / g.h)) + 1
    bx = ndimage.uniform_filter(np.where(on, fx.Jx, 0.0), m, mode="constant")
    by = ndimage.uniform_filter(np.where(on, fx.Jy, 0.0), m, mode="constant")
    bn = ndimage.uniform_filter(np.where(on, np.hypot(fx.Jx, fx.Jy), 0.0), m, mode="constant")
    out["coh"] = float(np.hypot(bx, by)[on].sum() / bn[on].sum())
    s, a = S[on], o[on]
    order = np.argsort(-s / a)                  # flux per unit of open area, densest first
    cum = np.cumsum(a[order])
    out["top"] = float(s[order][cum <= TOP * cum[-1]].sum() / tot)
    return out


def _bottleneck(sc: common.Scorer, dist: np.ndarray) -> np.ndarray:
    """Per home, its bottleneck width at baseline: the largest of WIDTH_STEPS w such that a path
    of open cells (4-connected, the operator's cell graph) whose clear width 2 d is at least w
    joins one of its demand cells to the street (0 below the first; nan if enclosed)."""
    g = sc.grid
    on = sc.free0 > 0
    ST = sc.pl.S.T.tocsr()                      # homes x cells: each home's demand cells
    out = np.where(sc.enclosed, np.nan, 0.0)
    for w in WIDTH_STEPS:
        wide = on & (2.0 * dist >= w)
        lab, _n = ndimage.label(wide)
        reach = np.zeros(int(lab.max()) + 1, dtype=bool)
        reach[lab[wide & g.ground]] = True
        reach[0] = False
        hit = (ST @ reach[lab].ravel().astype(float)) > 0
        out[hit & ~sc.enclosed] = w
    return out


def _road_fields(b, g: lifted.Grid, bid: str) -> dict[str, tuple[np.ndarray, float]]:
    """Item 4's road prefixes at area D BUDGET (road_rescore's prefix rows and ordered networks),
    carved on `g`: per method, (the open field, the prefix's area displacement)."""
    import geopandas as gpd

    from reblock.budget import road_corridor
    out = {}
    for method in ROADS:
        pre = pd.read_parquet(rr.prefix_dir(ROAD_ROWS) / f"{bid}__{method}.parquet")
        row = pre[(pre["pop"] == "area") & np.isclose(pre.budget, BUDGET)]
        [m] = row.m.astype(int)
        [D] = row.D_area.astype(float)
        ordered = gpd.read_parquet(rr.ordered_path(ROAD_ROWS / "scratch", bid, method))
        out[method] = (common.carved(g, g.sub_of(road_corridor(ordered.iloc[:m]))), D)
    return out


def score_path(off: int, variant: str, bid: str) -> Path:
    return ROWS / "score" / f"o{off}" / variant / f"{bid}.parquet"


def homes_path(off: int, variant: str, bid: str) -> Path:
    return ROWS / "homes" / f"o{off}" / variant / f"{bid}.parquet"


def flows_path(variant: str, bid: str) -> Path:
    return ROWS / "flows" / variant / f"{bid}.parquet"


def score(b, variant: str, off: int, device: str) -> None:
    """One block under one variant at OFFSETS[off], on `device` (the module's docstring)."""
    rr._check_single_threaded()
    t0 = time.time()
    spec, bid = VARIANTS[variant], b.block_id
    p = params(spec, device)
    sc = common.Scorer(b, lifted.UniformMesh(H, offset=OFFSETS[off]), p,
                       population=common.POPULATIONS["area"], demand=POP)
    g, n = sc.grid, len(sc.polys)
    shape = g.inside.shape
    # P_{1/2}'s injection from the scorer's baseline escape times (PriorityDemand.baseline's,
    # without solving that baseline again), and its own baseline
    inj_q = PQ.injection(sc.w, sc.u0, sc.enclosed)
    f_q = sc.pl.field(inj_q, shape)
    sol_q = lifted.solve(g, sc.free0, f_q, p)
    J2_0 = sc.J(sc.u0, 2.0)
    lab = g.label_sub(sc.polys)
    cost = sc.w / sc.w.sum()
    store = clearings.Store.committed()
    fields: dict[str, tuple[str, np.ndarray, float]] = {"baseline": ("baseline", sc.free0, 0.0)}
    for name, run in STORED.items():
        x = store.cleared(run, b, step=1)
        fields[name] = ("stored", _open_of(g, lab, x, n), float(cost[x].sum()))
    if off == 0:
        for name, (o, D) in _road_fields(b, g, bid).items():
            fields[name] = ("road", o, D)
    flows = bid in SENTINEL and off == 0
    if flows:                                   # Q1's no-buildings prior
        fields["prior"] = ("prior", g.isub.mean(axis=-1), np.nan)
    t_setup = time.time() - t0
    rows, kept = [], {}
    u_default = None
    for name, (kind, o, D) in fields.items():
        ts = time.time()
        sy = lifted.System(g, o, p)
        s1 = lifted.solve(g, o, sc.f, p, system=sy, rtol=SCORE_RTOL)
        u = sc.home_u_of(s1, o)
        sq = lifted.solve(g, o, f_q, p, system=sy, rtol=SCORE_RTOL)
        row = dict(block=bid, variant=variant, spec=spec, offset=off, device=device, field=name,
                   kind=kind, D=D, enclosed=sc.enclosed_left(o), P=s1.P, Pq=sq.P,
                   L_P=1.0 - s1.P / sc.P0, L_J2=sc.perm_p(u, 2.0),
                   **{"L_Pq0.5": 1.0 - sq.P / sol_q.P},
                   umed=float(np.nanmedian(u) / np.nanmedian(sc.u0)),
                   u95=float(np.nanpercentile(u, 95) / np.nanpercentile(sc.u0, 95)),
                   umax=float(np.nanmax(u) / np.nanmax(sc.u0)), unknowns=s1.n_unknowns)
        if name == "simp_default":
            u_default = u
        if flows and name in ("baseline", "simp_default", "prior"):
            fx = _flux(g, o, p, s1)
            row.update(_indices(g, o, fx, _wall_distance(g, o)))
            kept[name] = fx.S
        row["t"] = time.time() - ts
        rows.append(row)
        print(f"{bid} {variant} o{off} {name}: L_P {row['L_P']:.4f} L_J2 {row['L_J2']:.4f} "
              f"L_Pq {row['L_Pq0.5']:.4f} ({row['t']:.0f}s)", flush=True)
        del sy
        p.solver.release()
    setup = dict(cells=int(g.inside.sum()), P0=sc.P0, J2_0=J2_0, Pq0=sol_q.P,
                 enclosed0=sc.enclosed0, t_setup=t_setup, t_block=time.time() - t0,
                 host=socket.gethostname(), peak_rss_gb=rr._status_gb("VmHWM"))
    out = pd.DataFrame(rows).assign(**setup)
    score_path(off, variant, bid).parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(out, score_path(off, variant, bid))
    homes = pd.DataFrame(dict(block=bid, variant=variant, offset=off, home=np.arange(n),
                              w=sc.w, enclosed=sc.enclosed, u0=sc.u0, u_default=u_default,
                              bottleneck=_bottleneck(sc, _wall_distance(g, sc.free0))))
    homes_path(off, variant, bid).parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(homes, homes_path(off, variant, bid))
    if flows:
        on = np.flatnonzero(g.inside.ravel())
        fl = pd.DataFrame({"cell": on.astype(np.int32),
                           **{f"S_{k}": v.ravel()[on].astype(np.float32) for k, v in kept.items()}})
        flows_path(variant, bid).parent.mkdir(parents=True, exist_ok=True)
        rr.write_parquet(fl, flows_path(variant, bid))
    print(f"{time.strftime('%H:%M:%S')} {bid} {variant} o{off} done in {time.time() - t0:.0f}s, "
          f"peak RSS {setup['peak_rss_gb']:.1f} GB", flush=True)


def score_all(bid: str, which: str, device: str) -> None:
    """`score` under every variant whose rows are not here yet, one block built once: at the
    metric's offset (`which` 0) or at the block's shift of ALIGN (`shift`)."""
    off = 0 if which == "0" else ALIGN[bid] if which == "shift" else -1
    if off < 0:
        raise SystemExit(f"score_all at 0 or shift, not {which!r}")
    [b] = common.build_blocks([bid])
    for variant in VARIANTS:
        if score_path(off, variant, bid).exists():
            print(f"{bid} {variant} o{off}: rows here already", flush=True)
            continue
        score(b, variant, off, device)


# ------------------------------------------------------------------ SIMP under a variant (Q4)

def simp_path(variant: str, bid: str) -> Path:
    return ROWS / "simp" / variant / f"{bid}.parquet"


def simp(bid: str, variant: str) -> None:
    """SIMP base, objective P, area D BUDGET, free first, under the variant (relax.one)."""
    rr._check_single_threaded()
    t0 = time.time()
    row = relax.one(bid, 1.0, SIMP_DEVICE, SIMP_PLAN, VARIANTS[variant], BUDGET, demand=POP)
    row.update(variant=variant, spec=VARIANTS[variant], budget=BUDGET, host=socket.gethostname(),
               peak_rss_gb=rr._status_gb("VmHWM"), wall=time.time() - t0)
    simp_path(variant, bid).parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(pd.DataFrame([row]), simp_path(variant, bid))
    print(f"{time.strftime('%H:%M:%S')} {bid} simp {variant} done in {time.time() - t0:.0f}s",
          flush=True)


# ------------------------------------------------------------------ the cluster's inputs

def tars(out: Path, ids: list[str], what: str) -> None:
    """Per block, what a task reads that git does not carry, at its repo paths: for `score`, item
    4's prefix rows and ordered networks of ROADS (the stored clearings are committed); for
    `cross`, every variant's SIMP rows."""
    out.mkdir(parents=True, exist_ok=True)
    for bid in ids:
        if what == "score":
            members = [rr.prefix_dir(ROAD_ROWS) / f"{bid}__{m}.parquet" for m in ROADS]
            members += [rr.ordered_path(ROAD_ROWS / "scratch", bid, m) for m in ROADS]
        elif what == "cross":
            members = [simp_path(v, bid) for v in VARIANTS if simp_path(v, bid).exists()]
        else:
            raise SystemExit(f"tars for score or cross, not {what!r}")
        absent = [str(m) for m in members if not m.exists()]
        if absent:
            raise SystemExit(f"{bid}: {len(absent)} inputs absent, e.g. {absent[0]}")
        with tarfile.open(out / f"{bid}.tar", "w") as tf:
            for m in members:
                tf.add(m, arcname=str(m.resolve().relative_to(common.REPO)))
        print(f"{bid}: {len(members)} files, {(out / f'{bid}.tar').stat().st_size / 2**20:.1f} MB",
              flush=True)


# ------------------------------------------------------------------ the SIMP clearings, cross-scored

def cross_path(variant: str, bid: str) -> Path:
    return ROWS / "cross" / variant / f"{bid}.parquet"


def cross(bid: str, variant: str) -> None:
    """Every variant's SIMP clearing of `bid` (simp_path, those present) scored under `variant`:
    the three lenses and B (i), as `score` scores a field. Rows cross_path."""
    rr._check_single_threaded()
    t0 = time.time()
    runs = {v: pd.read_parquet(simp_path(v, bid)).iloc[0] for v in VARIANTS
            if simp_path(v, bid).exists()}
    if not runs:
        raise SystemExit(f"{bid}: no SIMP clearing under any variant")
    [b] = common.build_blocks([bid])
    p = params(VARIANTS[variant], CROSS_DEVICE)
    sc = common.Scorer(b, lifted.UniformMesh(H, offset=OFFSETS[0]), p,
                       population=common.POPULATIONS["area"], demand=POP)
    g, n = sc.grid, len(sc.polys)
    inj_q = PQ.injection(sc.w, sc.u0, sc.enclosed)
    f_q = sc.pl.field(inj_q, g.inside.shape)
    sol_q = lifted.solve(g, sc.free0, f_q, p)
    lab = g.label_sub(sc.polys)
    rows = []
    for v, r in runs.items():
        x = np.asarray(r.simp_cleared, dtype=np.int64)
        o = _open_of(g, lab, x, n)
        sy = lifted.System(g, o, p)
        s1 = lifted.solve(g, o, sc.f, p, system=sy, rtol=SCORE_RTOL)
        sq = lifted.solve(g, o, f_q, p, system=sy, rtol=SCORE_RTOL)
        rows.append(dict(block=bid, variant=variant, spec=VARIANTS[variant], clearing=v,
                         n_cleared=len(x), D=float((sc.w[x] / sc.w.sum()).sum()),
                         enclosed=sc.enclosed_left(o), L_P=1.0 - s1.P / sc.P0,
                         L_J2=sc.perm_p(sc.home_u_of(s1, o), 2.0),
                         **{"L_Pq0.5": 1.0 - sq.P / sol_q.P}, own_perm=float(r.simp_perm)))
        del sy
        p.solver.release()
    cross_path(variant, bid).parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(pd.DataFrame(rows).assign(t=time.time() - t0), cross_path(variant, bid))
    print(f"{time.strftime('%H:%M:%S')} {bid} cross {variant}: {len(rows)} clearings in "
          f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ the maps (Q1)

def _delta(y: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """Monroe, Colaresi & Quinn's delta (their Eq 16, the counts large against any prior), as the
    atlas pages show it: the log ratio of a cell's share of all walking to its share in the
    no-buildings prior; nan where either is 0."""
    ok = (y > 0) & (prior > 0)
    out = np.full(y.shape, np.nan)
    out[ok] = np.log(y[ok] / y[ok].sum()) - np.log(prior[ok] / prior[ok].sum())
    return out


def maps(ids: list[str]) -> None:
    """Per sentinel block: one PNG a variant (the baseline's flow against the no-buildings prior,
    SIMP's default's against it, and the flux SIMP's default adds or takes away), and one of
    every variant's baseline against the prior side by side; ROWS/maps/ (ignored)."""
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    out = ROWS / "maps"
    out.mkdir(parents=True, exist_ok=True)
    for bid in ids:
        [b] = common.build_blocks([bid])
        g = lifted.Grid.of(*lifted._geometry(b), H, offset=OFFSETS[0])
        polys = np.asarray(b.buildings.outlines)
        x = clearings.Store.committed().cleared(STORED["simp_default"], b, step=1)
        kept = np.delete(polys, x)
        ny, nx = g.inside.shape
        ext = (g.x0 - g.h / 2, g.x0 + (nx - 0.5) * g.h, g.y0 - g.h / 2, g.y0 + (ny - 0.5) * g.h)
        aspect = ny / nx
        base: dict[str, np.ndarray] = {}

        def draw(ax, v, title, cleared_outline: bool, **kw):
            im = ax.imshow(v.reshape(ny, nx), origin="lower", extent=ext,
                           interpolation="nearest", **kw)
            gpd.GeoSeries(list(kept if cleared_outline else polys)).plot(ax=ax, color="0.6",
                                                                          lw=0)
            if cleared_outline:
                gpd.GeoSeries(list(polys[x])).boundary.plot(ax=ax, color="#2ca02c", lw=0.6)
            gpd.GeoSeries([b.boundary]).boundary.plot(ax=ax, color="k", lw=0.8)
            ax.set_axis_off()
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=11)
            return im

        for variant in VARIANTS:
            f = pd.read_parquet(flows_path(variant, bid))
            S = {}
            for k in ("baseline", "simp_default", "prior"):
                S[k] = np.full(ny * nx, np.nan)
                S[k][f.cell.to_numpy()] = f[f"S_{k}"].to_numpy(dtype=float)
            d0, d1 = _delta(S["baseline"], S["prior"]), _delta(S["simp_default"], S["prior"])
            base[variant] = d0
            lim = float(np.nanquantile(np.abs(d0), 0.99))
            dS = S["simp_default"] - S["baseline"]
            dlim = float(np.nanquantile(np.abs(dS), 0.995))
            fig, axes = plt.subplots(1, 3, figsize=(3 * 11, 11 * aspect + 2))
            kw = dict(cmap="RdBu_r", vmin=-lim, vmax=lim)
            ims = [draw(axes[0], d0, f"{variant} ({VARIANTS[variant]}): baseline flow against "
                        "the no-buildings prior,\nlog of its share over the prior's (red: more "
                        "walking than in the open field)", False, **kw),
                   draw(axes[1], d1, "SIMP's default (green outlines) against the same prior",
                        True, **kw),
                   draw(axes[2], dS, "flux SIMP's default adds (red) or takes away (blue)", True,
                        cmap="RdBu_r", norm=mcolors.SymLogNorm(dlim * 1e-3, vmin=-dlim,
                                                               vmax=dlim))]
            for ax, im in zip(axes, ims, strict=True):
                fig.colorbar(im, ax=ax, shrink=0.6)
            fig.suptitle(f"{bid} under {variant}", fontsize=14)
            fig.tight_layout()
            fig.savefig(out / f"{bid}_{variant}.png", dpi=110)
            plt.close(fig)
            print(out / f"{bid}_{variant}.png", flush=True)
        fig, axes = plt.subplots(2, 4, figsize=(4 * 9, 2 * 9 * aspect + 2))
        for ax, (variant, d0) in zip(axes.ravel(), base.items(), strict=True):
            lim = float(np.nanquantile(np.abs(d0), 0.99))
            im = draw(ax, d0, f"{variant}: baseline against the prior", False, cmap="RdBu_r",
                      vmin=-lim, vmax=lim)
            fig.colorbar(im, ax=ax, shrink=0.6)
        fig.suptitle(f"{bid}: the baseline's flow against the no-buildings prior, every variant",
                     fontsize=14)
        fig.tight_layout()
        fig.savefig(out / f"{bid}_variants.png", dpi=90)
        plt.close(fig)
        print(out / f"{bid}_variants.png", flush=True)
        # the same, zoomed on current_map's window: a quarter of the block's extent about the
        # median building SIMP's default clears
        cx = float(np.median([q.centroid.x for q in polys[x]]))
        cy = float(np.median([q.centroid.y for q in polys[x]]))
        minx, miny, maxx, maxy = b.boundary.bounds
        half = 0.125 * max(maxx - minx, maxy - miny)
        fig, axes = plt.subplots(2, 4, figsize=(4 * 9, 2 * 9 + 2))
        for ax, (variant, d0) in zip(axes.ravel(), base.items(), strict=True):
            lim = float(np.nanquantile(np.abs(d0), 0.99))
            im = draw(ax, d0, f"{variant}: baseline against the prior (zoom)", False,
                      cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            fig.colorbar(im, ax=ax, shrink=0.6)
        fig.suptitle(f"{bid}: the baseline's flow against the prior, every variant, zoomed",
                     fontsize=14)
        fig.tight_layout()
        fig.savefig(out / f"{bid}_variants_zoom.png", dpi=90)
        plt.close(fig)
        print(out / f"{bid}_variants_zoom.png", flush=True)


# ------------------------------------------------------------------ the report

def _read(kind: str) -> pd.DataFrame:
    files = sorted((ROWS / kind).rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else (
        pd.DataFrame())


def _summary(name: str, df: pd.DataFrame) -> None:
    path = ROWS / f"summary_{name}.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"\n{name} -> {path}\n{df.to_string(index=False, float_format=lambda v: f'{v:.4g}')}",
          flush=True)


LENS_COLS = ("L_P", "L_Pq0.5", "L_J2")          # P, the equity view P_{1/2}, then J_2
REACH_TOL = 1e-9                                # a road prefix reaches D 0.05 at BUDGET - this


def report() -> None:
    """Every question's tables, written to ROWS/summary_*.csv (committed)."""
    from scipy.stats import kendalltau
    pd.set_option("display.width", 220)
    names = list(VARIANTS)
    # Q2: the channel
    ch = pd.read_parquet(ROWS / "channel.parquet")
    a = ch[ch.check == "angle"].groupby(["variant", "W"]).P.agg(lambda s: s.max() / s.min())
    _summary("q2_angle", a.unstack().reindex(names).reset_index())
    w = ch[ch.check == "width"]
    rel = []
    for (ang, W), g in w.groupby(["angle", "W"]):
        u = float(g[g.variant == "U"].P.iloc[0])
        rel += [dict(angle=ang, W=W, variant=r.variant, C_over_U=u / r.P) for r in g.itertuples()]
    t = pd.DataFrame(rel).pivot(index=["angle", "variant"], columns="W", values="C_over_U")
    _summary("q2_width", t.reindex([(ang, v) for ang in (0, 30) for v in names]).reset_index())
    al = ch[ch.check == "align"].groupby(["variant", "W"]).P.agg(lambda s: s.max() / s.min())
    _summary("q2_align", al.unstack().reindex(names).reset_index())
    sub = ch[ch.check == "subcell"].groupby(["variant", "W"]).P.agg(lambda s: s.max() / s.min())
    _summary("q2_subcell", sub.unstack().reindex(names).reset_index())
    sc = _read("score")
    if sc.empty:
        return
    # Q1: the corridor indices on the sentinel
    s0 = sc[sc.offset == 0]
    q1 = s0[s0.block.isin(SENTINEL) & s0.field.isin(["baseline", "simp_default", "prior"])]
    if "coh" in q1:
        q1 = q1[["block", "variant", "field", "open1", "area1", "open2", "area2", "coh", "top"]]
        q1 = q1.assign(lift1=q1.open1 / q1.area1, lift2=q1.open2 / q1.area2)
        _summary("q1_indices", q1.sort_values(["block", "field", "variant"],
                                              key=_variant_key))
    # Q3: a block's level at two offsets
    lv = []
    for bid, k in ALIGN.items():
        for v in names:
            at = {o: sc[(sc.block == bid) & (sc.variant == v) & (sc.offset == o)
                        & (sc.field == "simp_default")] for o in (0, k)}
            if any(len(x) != 1 for x in at.values()):
                continue
            row = dict(block=bid, variant=v, shift=k)
            for col in LENS_COLS:
                row[f"{col}_o0"] = float(at[0][col].iloc[0])
                row[f"{col}_o{k}"] = float(at[k][col].iloc[0])
                row[f"{col}_change"] = row[f"{col}_o{k}"] - row[f"{col}_o0"]
            lv.append(row)
    if lv:
        _summary("q3_levels", pd.DataFrame(lv))
    homes = _read("homes")
    if not homes.empty:
        blocks = {b.block_id: b for b in common.build_blocks(list(ALIGN))}
        fb = {(bid, o): fine_bottleneck(blocks[bid], o) for bid, k in ALIGN.items()
              for o in (0, k)}
        _summary("q3_homes", _home_drift(homes, {key: v[1] for key, v in fb.items()},
                                         {key: v[2] for key, v in fb.items()}))
    # Q4: the 13 tuning blocks, every method under every variant
    t13 = s0[s0.block.isin(TUNING) & s0.kind.isin(["stored", "road"])].copy()
    t13["reaches"] = (t13.kind == "stored") | (t13.D >= BUDGET - REACH_TOL)
    t13 = t13[t13.reaches]
    taus, lens, price = [], [], []
    for bid, gb in t13.groupby("block"):
        ref = gb[gb.variant == "U"].set_index("field")
        for v, g in gb.groupby("variant"):
            g = g.set_index("field").reindex(ref.index)
            if g[list(LENS_COLS)].isna().any().any():
                continue
            for col in LENS_COLS:
                rv = rr._b_rank(g.reset_index(), "enclosed", col)
                ru = rr._b_rank(ref.reset_index(), "enclosed", col)
                taus.append(dict(block=bid, variant=v, lens=col, methods=len(g),
                                 tau=float(kendalltau(ru, rv).statistic)))
                roads = g[g.kind == "road"]
                best = roads[col].idxmax()
                price.append(dict(block=bid, variant=v, lens=col, best_road=best,
                                  simp_minus_best=float(g.loc["simp_default", col]
                                                        - roads[col].max()),
                                  cheap_minus_best=float(g.loc["cheap", col] - roads[col].max()),
                                  best_share=float(roads[col].max() / g.loc["simp_default", col])))
            lens.append(g.reset_index()[["block", "variant", "field", *LENS_COLS]])
    if taus:
        tt = pd.DataFrame(taus)
        _summary("q4_tau", tt.groupby(["variant", "lens"]).tau.agg(
            median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
            min="min", blocks="count").reset_index().sort_values(["lens", "variant"],
                                                                 key=_variant_key))
        pr = pd.DataFrame(price)
        _summary("q4_price", pr.groupby(["variant", "lens"]).agg(
            simp_minus_best=("simp_minus_best", "median"),
            cheap_minus_best=("cheap_minus_best", "median"),
            best_share=("best_share", "median"),
            leader=("best_road", lambda s: s.value_counts().idxmax()),
            leader_blocks=("best_road", lambda s: int(s.value_counts().max()))).reset_index()
            .sort_values(["lens", "variant"], key=_variant_key))
        ln = pd.concat(lens, ignore_index=True)
        _summary("q4_lenses", ln.groupby(["variant", "field"])[list(LENS_COLS)].median()
                 .reset_index().sort_values(["field", "variant"], key=_variant_key))
    simp_rows = _read("simp")
    if not simp_rows.empty:
        _summary("q4_simp", _simp_table(simp_rows, _read("cross")))
        _summary("q4_where", _where(simp_rows, homes))


def _where(simp_rows: pd.DataFrame, homes: pd.DataFrame) -> pd.DataFrame:
    """Where each B_r's SIMP clearing differs from U's, per sentinel block: the buildings both
    clear, U's alone and B_r's alone, by their median footprint area, distance to the block's
    outline (its streets), gap to the nearest other footprint, and depth (the home's escape time
    at baseline under U over the population-weighted mean)."""
    import shapely
    out = []
    for bid, g in simp_rows.groupby("block"):
        sets = {r.variant: set(np.asarray(r.simp_cleared).tolist()) for r in g.itertuples()}
        if "U" not in sets:
            continue
        [b] = common.build_blocks([bid])
        polys = np.asarray(b.buildings.outlines)
        area = shapely.area(polys)
        street = shapely.distance(polys, b.boundary.boundary)
        tree = shapely.STRtree(polys)
        gap = np.full(len(polys), np.inf)
        idx = tree.query(polys, predicate="dwithin", distance=20.0)
        own = idx[0] != idx[1]
        d = shapely.distance(polys[idx[0][own]], polys[idx[1][own]])
        np.minimum.at(gap, idx[0][own], d)
        h0 = homes[(homes.block == bid) & (homes.variant == "U") & (homes.offset == 0)]
        h0 = h0.set_index("home")
        live = ~h0.enclosed
        mean_u0 = float((h0.w[live] * h0.u0[live]).sum() / h0.w[live].sum())
        depth = (h0.u0 / mean_u0).reindex(range(len(polys))).to_numpy()
        for v, s in sets.items():
            if v == "U":
                continue
            for part, ids in (("both", sets["U"] & s), ("U_only", sets["U"] - s),
                              (f"{v}_only", s - sets["U"])):
                j = np.array(sorted(ids), dtype=np.int64)
                out.append(dict(block=bid, variant=v, part=part, buildings=len(j),
                                area_m2=float(np.median(area[j])) if len(j) else np.nan,
                                street_m=float(np.median(street[j])) if len(j) else np.nan,
                                gap_m=float(np.median(gap[j])) if len(j) else np.nan,
                                depth=float(np.nanmedian(depth[j])) if len(j) else np.nan))
    return pd.DataFrame(out)


def _variant_key(s: pd.Series) -> pd.Series:
    order = {v: i for i, v in enumerate(VARIANTS)}
    return s.map(order) if s.name == "variant" else s


FINE_STEPS = (0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0)   # fine bottleneck widths tried (m)
SUBCELL_M = 0.5            # a gap narrower than a cell: behind one, a cell's bottleneck is at most this


def fine_bottleneck(b, off: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """At baseline on the fine lattice (h / 4) at OFFSETS[off]: per cell, its bottleneck width to
    the street, the largest of FINE_STEPS w such that a 4-connected path of open sub-samples whose
    clear width 2 d is at least w (d to the nearest footprint sub-sample, WallDistance's) joins one
    of its sub-samples to a ground cell's (0 if none); and per home, the share of its injection
    (lifted.Placement.S) in cells behind a gap narrower than a cell: a bottleneck of at most
    SUBCELL_M, and narrower than the cell's own clear width (a courtyard behind a crack, not a
    cell sitting in a narrow strip, whose way out is as narrow as it is); and per home, the share
    in cells with no such path at all (bottleneck 0): open sub-samples joined to the street only
    through the operator's cell graph, two 4-adjacent cells each with an open sub-sample whose open
    parts do not touch (a phantom passage, which a finer raster would not have). nan if enclosed.
    `_bottleneck`'s cell-level
    width averages a cell's sub-samples, footprint ones among them, so it cannot see such a gap.
    Geometry only: no conductance."""
    p = params("uni", "cpu")
    g = lifted.Grid.of(*lifted._geometry(b), H, offset=OFFSETS[off])
    pl = common.placement(g, np.asarray(b.buildings.outlines), p)
    S = g.S
    stand = lifted._closed(g, g.ff0, np) >= 1.0
    opened = lifted._fine(g.isub, S) & ~stand
    d = ndimage.distance_transform_edt(~stand) * (g.h / S) - g.h / (2 * S)
    ground = lifted._fine(np.repeat(g.ground[..., None], S * S, axis=-1), S)
    cell_b = np.zeros(g.inside.size)
    for w in FINE_STEPS:
        wide = opened & (2.0 * d >= w - 1e-9)
        lab, _n = ndimage.label(wide)
        reach = np.zeros(int(lab.max()) + 1, dtype=bool)
        reach[lab[wide & ground]] = True
        reach[0] = False
        cell_b[lifted._cell_sum(reach[lab].astype(float), S).ravel() > 0] = w
    ny, nx = g.inside.shape
    own = np.where(opened, 2.0 * d, 0.0).reshape(ny, S, nx, S).max(axis=(1, 3)).ravel()
    room = np.zeros(g.inside.size)                     # its own clear width, as a step
    for w in FINE_STEPS:
        room[own >= w - 1e-9] = w
    pocket = (cell_b <= SUBCELL_M + 1e-9) & (cell_b < room)
    ST = pl.S.T.tocsr()
    behind = ST @ pocket.astype(float)
    phantom = ST @ ((cell_b == 0.0) & (room > 0)).astype(float)
    return (cell_b, np.where(pl.enclosed, np.nan, behind),
            np.where(pl.enclosed, np.nan, phantom))


def _home_drift(homes: pd.DataFrame, behind: dict[tuple[str, int], np.ndarray],
                phantom: dict[tuple[str, int], np.ndarray]) -> pd.DataFrame:
    """Q3, per block of ALIGN, variant and lens (P, J_2): how each home's term in the ratio
    J / J0 under SIMP's default moves between the metric's offset and the block's shift, by the
    share of the home's injection behind a gap narrower than a cell (`fine_bottleneck`, the larger
    of its two alignments): per group its homes, population, share of J0 at the metric's offset,
    and share of the ratio's change, signed and absolute; with the share of J0 the three heaviest
    homes hold at each offset; per group too the population and the share of J0 of its homes
    with over 5% of their injection behind a phantom passage only (`fine_bottleneck`) at either
    alignment."""
    out = []
    groups = (("none", 0.0, 0.0), ("under 5%", 0.0, 0.05), ("over 5%", 0.05, 1.0))
    for bid, k in ALIGN.items():
        if (bid, 0) not in behind or (bid, k) not in behind:
            continue
        share = np.fmax(behind[(bid, 0)], behind[(bid, k)])
        ghost = np.fmax(phantom[(bid, 0)], phantom[(bid, k)]) > 0.05
        for v, g in homes[homes.block == bid].groupby("variant"):
            at = {o: g[g.offset == o].set_index("home").sort_index() for o in (0, k)}
            if any(x.empty for x in at.values()):
                continue
            live = (~(at[0].enclosed | at[k].enclosed)).to_numpy() & ~np.isnan(share)
            w = at[0].w.to_numpy()
            for power, lens in ((1.0, "P"), (2.0, "J2")):
                c, base, top3 = {}, {}, {}
                for o, h in at.items():
                    u0, ud = np.nan_to_num(h.u0.to_numpy()), np.nan_to_num(h.u_default.to_numpy())
                    J0 = float((w[live] * u0[live] ** power).sum())
                    c[o] = np.where(live, w * ud ** power / J0, 0.0)
                    base[o] = np.where(live, w * u0 ** power / J0, 0.0)
                    top3[o] = float(np.sort(base[o])[-3:].sum())
                d = c[k] - c[0]
                total, absum = float(d.sum()), float(np.abs(d).sum())
                for name, lo, hi in groups:
                    m = live & ((share == 0.0) if hi == 0.0 else (share > lo) & (share <= hi))
                    out.append(dict(
                        block=bid, variant=v, shift=k, lens=lens, behind=name,
                        homes=int(m.sum()), population=float(w[m].sum() / w[live].sum()),
                        phantom_population=float(w[m & ghost].sum() / w[live].sum()),
                        J0_share=float(base[0][m].sum()), J0_share_shift=float(base[k][m].sum()),
                        J0_share_phantom=float(base[0][m & ghost].sum()),
                        change=float(d[m].sum()), change_share=float(d[m].sum() / total),
                        abs_share=float(np.abs(d[m]).sum() / absum),
                        ratio_o0=float(c[0].sum()), ratio_shift=float(c[k].sum()),
                        top3_o0=top3[0], top3_shift=top3[k]))
    return pd.DataFrame(out)


def _simp_table(simp_rows: pd.DataFrame, crossed: pd.DataFrame) -> pd.DataFrame:
    """Q4's SIMP runs: per block and variant, the clearing's own P lens, its size, the Jaccard
    overlap with U's, and its lenses under U's conductance and its own (cross)."""
    out = []
    for bid, g in simp_rows.groupby("block"):
        sets = {r.variant: set(np.asarray(r.simp_cleared).tolist()) for r in g.itertuples()}
        for r in g.itertuples():
            row = dict(block=bid, variant=r.variant, own_P=float(r.simp_perm),
                       n_cleared=len(sets[r.variant]), D=float(r.simp_D),
                       free_D=float(r.free_D), enclosed=float(r.enclosed), t=float(r.t))
            if "U" in sets:
                u = sets["U"]
                row["jaccard_U"] = len(u & sets[r.variant]) / len(u | sets[r.variant])
            if not crossed.empty:
                for c in crossed[(crossed.block == bid) & (crossed.clearing == r.variant)
                                 ].to_dict("records"):
                    for col in LENS_COLS:
                        row[f"{col}_under_{c['variant']}"] = float(c[col])
            out.append(row)
    return pd.DataFrame(out)


def main(argv: list[str]) -> None:
    cmd, *a = argv
    if cmd == "channel":
        channel(int(a[0]))
    elif cmd == "score":
        [b] = common.build_blocks([a[0]])
        score(b, a[1], int(a[2]), a[3])
    elif cmd == "score-all":
        score_all(a[0], a[1], a[2])
    elif cmd == "simp":
        simp(a[0], a[1])
    elif cmd == "cross":
        cross(a[0], a[1])
    elif cmd == "tars":
        tars(Path(a[0]), a[1].split(","), a[2])
    elif cmd == "maps":
        maps(a[0].split(","))
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
