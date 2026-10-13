"""The grid fix: steps that conduct by their open width (BACKLOG 4, decision (3), "Next: the grid
fix"; owner 2026-10-13, "LGTM"; NOTES, "The grid fix: steps that conduct by their open width").

Today a step conducts by the smallest open fraction of the cells it touches (lifted.SmallestCell),
how much of each cell is open, not whether the open parts meet, so two 4-adjacent cells whose open
sub-samples do not touch are joined (a phantom passage). The fix (lifted.Aperture) makes a step
conduct by the share of its width open all the way from one cell's centre to the next, read on the
sub-samples. RULES names the rules compared (a name and its lifted.conductance_of spec, the spec in
every row): T `uni` (today's), A `uni+ap` (the aperture, each step's band its own share of the
cross-section, h / |v_k|), Aw `uni+apw` (the band the cell's width h), Aws `uni+apws` (Aw, a
split cell conducting through one part only: lifted.AnchorPart) and Fws `uni+apwfs` (Aws, an
axis-aligned step's tracks across the face: lifted.Band's face). h 0.5, ell 3 m, K 8, area
population, B's placement; lenses P (power 1), P_{1/2} (common.PriorityDemand(0.5), power 1) and
J_2 (lens (c)).

    S=research/roadless/aperture.py      # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S channel <procs>                     # Q3: the corridor (angle, alignment)
    uv run python $S base <id> <cpu|gpu> <mesh> <rules,> [tag]   # Q1, Q2: one block
    uv run python $S base-all <ids,|@file> <procs>       # here: small blocks whole, pockets
    uv run python $S base-local <ids,|@file> <procs> <rules,> [tag]   # here: blocks' base
    uv run python $S moves <ids,|@file> <procs>          # here: Q2's demand moves by cause
    uv run python $S pockets-composite <ids,|@file> <procs>   # here: oversized blocks' pockets
    uv run python $S score <id> <rule> <offset> <cpu|gpu> 0.5   # Q3, Q4: one block's fields
    uv run python $S score-plan <id> <cpu|gpu> <drift rules,> <tuning rules,>   # score_plan's
    uv run python $S tars <dir> <ids,>                   # a tar per block of score's inputs
    uv run python $S simp <id> <rule> <cpu|gpu> <power>  # SIMP base under a rule (the gradient)
    uv run python $S report                              # every question's tables, the summaries

<mesh>: mesh_token_of(id), h 0.5 or OVERSIZED_MESH (named on the command line so cluster.py sizes
the task by it); <T,...> the rules, today's first.

`channel` (here, `procs` processes): width.walled_channel's W x 40 m corridor under every rule, at
the angles of sightline_angle.ANGLES and WIDTHS_ANGLE, at 0 and 30 deg over WIDTHS (how conduction
grows with width), at 0 deg over ALIGN_SHIFTS lattice offsets (WIDTHS_ALIGN) and at WIDTHS_SUBCELL
over SUBCELL_SHIFTS (widths where a shift adds or drops a column of sub-samples). Rows
aperture_rows/channel.parquet.

`base`: one block at baseline under every rule (lifted.UniformMesh(H) at the metric's offset;
AdaptiveMesh a5x8 for a block too large for it, OVERSIZED): per rule the scorer (P0, P_{1/2}'s and
J_2's baselines, B's enclosed homes), each home's escape time under P and P_{1/2}, its placement
(the cells it injects into), the split cells (cells whose open sub-samples form two parts or more,
joined by the cell's one node) and the population whose demand sits in them, and free first's
crossings (clear.free_first at area D BUDGET, nothing held). Rows aperture_rows/base/<id>.parquet
(the block) and aperture_rows/homes/<id>.parquet (per home and rule). `pockets` (in base-all,
here: geometry only), on the uniform grid under today's rule, width.fine_bottleneck's share of
each home's injection behind a sub-cell gap or a phantom passage, and behind a phantom passage;
rows aperture_rows/pockets/<id>.parquet.

`score`: one block under one rule on the lattice at OFFSETS[offset]: the baseline, the stored
clearings (STORED, through clearings.Store) and item 4's road prefixes at area D BUDGET (offset 0
only), each under the three lenses with B (i); and each stored clearing with the homes it leaves
enclosed freed first (clear.free_first under the rule, the clearing held, no budget: its D grows by
what that costs), where it leaves any. Rows aperture_rows/score/o<offset>/<rule>/<id>.parquet;
per home (P, P_{1/2}, J_2 terms at baseline and under SIMP's default) aperture_rows/homes_score/...
"""
from __future__ import annotations

import multiprocessing
import socket
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import clearings  # noqa: E402
import common  # noqa: E402
import lifted  # noqa: E402
import relax  # noqa: E402
import road_rescore as rr  # noqa: E402
import width  # noqa: E402
from simp_compare import TUNING  # noqa: E402

# ------------------------------------------------------------------ the run's configuration
# Everything that can change a number is here, and named in the rows (the run's identity).
RULES = {"T": "uni", "A": "uni+ap", "Aw": "uni+apw", "Aws": "uni+apws", "Fws": "uni+apwfs"}
H = 0.5
ELL_M = 3.0
K = 8
OFFSETS = width.OFFSETS            # offset_check.OFFSETS: the metric's, then three shifts
ALIGN = width.ALIGN                # the drift blocks and the shift that moved each most
SENTINEL = width.SENTINEL
STORED = width.STORED              # SIMP's default and the cheap preset at area D 0.05
ROADS = width.ROADS                # item 4's six road methods
BUDGET = relax.D_LENS              # area D 0.05
SCORE_RTOL = clear.RTOL_SCORE      # every score: the optimizers' exact one
POP = common.PopulationDemand()
PQ = common.PriorityDemand(0.5)
OVERSIZED_MESH = "0.5a5x8"         # the oversized blocks' mesh (the metric's composite)
BEHIND = 0.05                      # a home is behind a pocket when over this share of its
                                   # injection is (width.py's Q3 grouping)
SIMP_PLAN = width.SIMP_PLAN        # SIMP base
ROWS = HERE / "aperture_rows"

# the channel checks (width.py's corridor)
ANGLES = width.ANGLES
WIDTHS_ANGLE = width.WIDTHS_ANGLE
WIDTHS = width.WIDTHS
ALIGN_SHIFTS = width.ALIGN_SHIFTS
WIDTHS_ALIGN = width.WIDTHS_ALIGN
SUBCELL_SHIFTS = width.SUBCELL_SHIFTS
WIDTHS_SUBCELL = width.WIDTHS_SUBCELL


def params(spec: str, device: str) -> lifted.Params:
    cond = lifted.conductance_of(spec, lifted.scans_of(device))
    return lifted.Params(ELL_M, K, along=cond.along, cells=cond.cells, steps=cond.steps,
                         solver=lifted.solver_of(device))


# ------------------------------------------------------------------ the channel (Q3)

class ChannelJob(NamedTuple):
    check: str          # angle | width | align | subcell
    rule: str           # a RULES key
    W: float
    angle: float
    shift: int          # align, subcell: k of ALIGN_SHIFTS, SUBCELL_SHIFTS (0: the metric's)


def _channel_one(job: ChannelJob) -> dict:
    """P of one corridor; the demand (width.walled_channel's: the open cells within 1 m of the
    dead end) only on the cells the rule reaches from the street, renormalized, as Placement
    allows only those on a block (under today's rule every one of them: width.py's numbers)."""
    k = job.shift / (SUBCELL_SHIFTS if job.check == "subcell" else ALIGN_SHIFTS)
    offset = ((lifted.OFFSET[0] + k) % 1.0, (lifted.OFFSET[1] + k) % 1.0)
    grid, f = width.walled_channel(job.angle, job.W, H, offset)
    p = params(RULES[job.rule], "cpu")
    t0 = time.time()
    on = lifted.grounded(grid, grid.ff0, p)
    lost = float(f[~on].sum())
    f = np.where(on, f, 0.0)
    P = lifted.solve(grid, grid.ff0, f / f.sum(), p).P if f.sum() > 0 else np.inf
    return dict(job._asdict(), spec=RULES[job.rule], P=P, demand_lost=lost, t=time.time() - t0)


def channel(procs: int) -> None:
    """Every channel job, `procs` processes; rows aperture_rows/channel.parquet."""
    rr._check_single_threaded()
    names = list(RULES)
    jobs = ([ChannelJob("angle", v, W, a, 0) for v in names for W in WIDTHS_ANGLE for a in ANGLES]
            + [ChannelJob("width", v, W, a, 0) for v in names for W in WIDTHS for a in (0, 30)]
            + [ChannelJob("align", v, W, 0.0, s) for v in names for W in WIDTHS_ALIGN
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


# ------------------------------------------------------------------ one block at baseline (Q1, Q2)

AREA = common.POPULATIONS["area"]
UNIFORM_MAX_M = 49.0     # a block over this many million unknowns at h 0.5 (8 per cell) is
                         # scored on OVERSIZED_MESH: 30796's 48.7M is the largest the cards hold


def mesh_token_of(bid: str) -> str:
    """The mesh `base` scores a block on: uniform h 0.5, or the composite for an oversized one."""
    sizes = pd.read_parquet(HERE / "large82_area.parquet").set_index("block").M_unknowns
    return OVERSIZED_MESH if bid in sizes.index and sizes[bid] > UNIFORM_MAX_M else "0.5"


def base_path(bid: str, tag: str) -> Path:
    """`base`'s rows of a block; `tag` names a later pass over other rules ("" the first)."""
    return ROWS / "base" / f"{bid}{'__' + tag if tag else ''}.parquet"


def homes_path(bid: str, tag: str) -> Path:
    return ROWS / "homes" / f"{bid}{'__' + tag if tag else ''}.parquet"


def _parts(grid: lifted.Grid | lifted.CompositeGrid) -> np.ndarray:
    """(cells flat): how many 4-connected parts each cell's open sub-samples form (0 if none)."""
    S = grid.S
    op = (grid.isub & ~grid.bsub).reshape(-1, S, S)
    n = np.zeros(len(op), dtype=np.int64)
    for i in np.flatnonzero(op.any(axis=(1, 2))):
        n[i] = ndimage.label(op[i])[1]
    return n


def _flood(grid: lifted.Grid) -> np.ndarray:
    """(cells flat) on the uniform grid: the cells the 4-connected flood fill of open sub-samples
    reaches from the ground cells' (aperture_check.flood)."""
    import aperture_check
    return aperture_check.flood(grid)[0].ravel()


def base(b, device: str, token: str, rules: tuple[str, ...], tag: str) -> None:
    """One block at baseline under `rules` (the module's docstring)."""
    rr._check_single_threaded()
    t0 = time.time()
    bid = b.block_id
    mesh = common.mesh_of(token)
    polys = np.asarray(b.buildings.outlines)
    p0 = params(RULES[rules[0]], device)
    grid = mesh.build(b, p0, AREA)
    shape, ncell = grid.inside.shape, grid.inside.size
    uniform = isinstance(mesh, lifted.UniformMesh)
    t_grid = time.time() - t0
    parts = _parts(grid)
    split = parts >= 2
    fl = _flood(grid) if uniform else None
    rows, homes, S_of = [], [], {}
    for r in rules:
        ts = time.time()
        p = params(RULES[r], device)
        sc = common.Scorer.on_grid(b, grid, p, AREA, demand=POP)
        inj_q = PQ.injection(sc.w, sc.u0, sc.enclosed)
        sol_q = lifted.solve(grid, sc.free0, sc.pl.field(inj_q, shape), p, rtol=SCORE_RTOL)
        u_q = sc.home_u_of(sol_q, sc.free0)
        t_solve = time.time() - ts
        S_of[r] = sc.pl.S.tocsc()
        reach = lifted.grounded(grid, sc.free0, p).ravel()
        # free first: the crossings that free the homes enclosed at baseline, at area D BUDGET
        tf = time.time()
        c = clear.Clearing.on_scorer(sc, rtol=clear.RTOL_TENSION, search=None)
        try:
            held = clear.free_first(c, budget=BUDGET, held=np.zeros(c.n, dtype=bool))
            left = sc.enclosed & ~common.reached(sc.pl, grid, c.open_(held), p)
            free_D, free_n = float(c.cost[held].sum()), int(held.sum())
            free_left = float(sc.w[left].sum() / sc.w.sum())
            free_ids = " ".join(map(str, np.flatnonzero(held)))
        except NotImplementedError as e:            # today's rule on a composite mesh
            print(f"{bid} {r}: free first not built here: {e}", flush=True)
            free_D = free_left = np.nan
            free_n, free_ids = -1, ""
        t_free = time.time() - tf
        in_split = np.asarray(sc.pl.S[split].sum(axis=0)).ravel()     # each home's share there
        row = dict(block=bid, rule=r, spec=RULES[r], mesh=mesh.name, device=device,
                   cells=int(grid.inside.sum()), open_cells=int((sc.free0 > 0).sum()),
                   reached=int(reach.sum()), P0=sc.P0, Pq0=sol_q.P, J2_0=sc.J(sc.u0, 2.0),
                   enclosed=int(sc.enclosed.sum()), enclosed0=sc.enclosed0,
                   split=int(split.sum()), split_reached=int((split & reach).sum()),
                   split_pop=float((sc.w * in_split)[~sc.enclosed].sum() / sc.w.sum()),
                   free_D=free_D, free_n=free_n, free_left=free_left, free_ids=free_ids,
                   unknowns=int((sc.free0 > 0).sum() * K), t_solve=t_solve, t_free=t_free)
        if fl is not None:
            row.update(flood=int(fl.sum()), op_only=int((reach & ~fl).sum()),
                       flood_only=int((fl & ~reach).sum()))
        rows.append(row)
        # per home: escape times under P and P_{1/2}, the injections, whether any of its near
        # cells is reached by the flood fill (a home enclosed although open sub-samples near it
        # reach the street: enclosed by the rule, not the geometry), and the share of its
        # injection in cells the rule reaches but the flood fill does not (a pocket joined to the
        # street only through a split cell's node, for the aperture; a phantom passage too,
        # for today's rule)
        near_flood = np.full(len(polys), np.nan)
        beyond = np.full(len(polys), np.nan)
        if fl is not None:
            for j in np.flatnonzero(sc.enclosed):
                cn = grid.cells_near(polys[j], lifted.RING_M + 1e-9)
                near_flood[j] = float(fl[cn].any())
            beyond = np.asarray(sc.pl.S[reach & ~fl].sum(axis=0)).ravel()
        homes.append(pd.DataFrame(dict(
            block=bid, rule=r, home=np.arange(len(polys)), w=sc.w, enclosed=sc.enclosed,
            u0=sc.u0, inj_q=inj_q, u_q=u_q, in_split=in_split, near_flood=near_flood,
            beyond_flood=beyond)))
        print(f"{time.strftime('%H:%M:%S')} {bid} {r}: P0 {sc.P0:.6g} Pq0 {sol_q.P:.6g} enclosed "
              f"{sc.enclosed.sum()} ({sc.enclosed0:.4f}); free D {free_D:.4f} on {free_n}, "
              f"{t_solve:.0f}s + free {t_free:.0f}s", flush=True)
        del sc, c, sol_q
        p.solver.release()
    h = pd.concat(homes, ignore_index=True)
    # the moves against today's placement (built here if today's rule is not among `rules`: the
    # placement alone, no solve): the share of a home's injection that sits in other cells
    ref = S_of["T"] if "T" in S_of else common.placement(
        grid, polys, params(RULES["T"], device)).S.tocsc()
    for r in rules:
        d = 0.5 * abs(S_of[r] - ref).sum(axis=0).A.ravel()
        h.loc[h.rule == r, "moved"] = d
    out = pd.DataFrame(rows).assign(t_grid=t_grid, t_block=time.time() - t0,
                                    host=socket.gethostname(),
                                    peak_rss_gb=rr._status_gb("VmHWM"))
    for path, df in ((base_path(bid, tag), out), (homes_path(bid, tag), h)):
        path.parent.mkdir(parents=True, exist_ok=True)
        rr.write_parquet(df, path)
    print(f"{time.strftime('%H:%M:%S')} {bid} base done in {time.time() - t0:.0f}s, "
          f"{ncell} cells, peak RSS {out.peak_rss_gb.iloc[0]:.1f} GB", flush=True)


BASE_RULES_SMALL = ("T", "A", "Aw", "Aws")     # the small blocks here: every rule


def pockets_path(bid: str) -> Path:
    return ROWS / "pockets" / f"{bid}.parquet"


def pockets(b) -> None:
    """Q1's pockets under today's rule on the uniform grid at the metric's offset: per home the
    share of its injection behind a sub-cell gap or a phantom passage (`behind`) and behind a
    phantom passage (`phantom`), width.fine_bottleneck's (geometry only: no solve). Rows
    aperture_rows/pockets/<id>.parquet."""
    t0 = time.time()
    _cb, behind, phantom = width.fine_bottleneck(b, 0)
    out = pd.DataFrame(dict(block=b.block_id, home=np.arange(len(behind)), behind=behind,
                            phantom=phantom)).assign(t=time.time() - t0)
    pockets_path(b.block_id).parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(out, pockets_path(b.block_id))
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} pockets in {time.time() - t0:.0f}s",
          flush=True)


def composite_flood(g: lifted.CompositeGrid) -> np.ndarray:
    """(cells,) on a composite mesh: the cells the 4-connected flood fill of open sub-samples
    reaches from the ground cells' (the uniform grid's flood, aperture_check.flood): a node per
    sub-sample, a coarse cell's sixteen all open (it is wholly open), sub-samples joined within a
    cell and across a cell's sides to its neighbours' facing ones (a coarse neighbour by any of
    its own), found at every fine-lattice position along each side."""
    from scipy.sparse.csgraph import connected_components
    S, n = g.S, len(g.level)
    S2 = S * S
    op = (g.isub & ~g.bsub).reshape(n, S2)
    r_ = np.arange(S2) // S
    c_ = np.arange(S2) % S
    a, b = [], []
    for s in range(S2):                                   # within a cell
        for t in ((s + 1) if c_[s] < S - 1 else -1, (s + S) if r_[s] < S - 1 else -1):
            if t >= 0:
                ok = op[:, s] & op[:, t]
                cells = np.flatnonzero(ok)
                a.append(cells * S2 + s)
                b.append(cells * S2 + t)
    # sides: (row offset, col offset) of the probe line just outside, and the facing sub-samples
    # of the cell (own) and of a neighbour at the same scale (other), by position along the side
    sides = (("right", lambda k, rs: (g.i0 + k, g.j0 + rs), [(i, S - 1) for i in range(S)],
              [(i, 0) for i in range(S)]),
             ("up", lambda k, rs: (g.i0 + rs, g.j0 + k), [(S - 1, j) for j in range(S)],
              [(0, j) for j in range(S)]))
    for _side, probe, own, other in sides:
        for lev in np.unique(g.level):
            cells = np.flatnonzero(g.level == lev)
            r = 1 << int(lev)
            for k in range(r):                            # every fine position along the side
                py, px = probe(k, r)
                q = g._locate(py[cells].astype(float), px[cells].astype(float))
                hit = q >= 0
                cc, qq = cells[hit], q[hit]
                if lev == 0:
                    fine_q = g.level[qq] == 0
                    for (oi, oj), (ti, tj) in zip(own, other, strict=True):
                        so, st = oi * S + oj, ti * S + tj
                        m = fine_q & op[cc, so] & op[qq, st]
                        a.append(cc[m] * S2 + so)
                        b.append(qq[m] * S2 + st)
                        m2 = ~fine_q & op[cc, so]           # a coarse neighbour: any of its own
                        a.append(cc[m2] * S2 + so)
                        b.append(qq[m2] * S2)
                else:                                     # a coarse cell: by its first sub-sample
                    fine_q = g.level[qq] == 0
                    for ti, tj in other:
                        st = ti * S + tj
                        m = fine_q & op[qq, st]
                        a.append(cc[m] * S2)
                        b.append(qq[m] * S2 + st)
                    a.append(cc[~fine_q] * S2)
                    b.append(qq[~fine_q] * S2)
    a_, b_ = np.concatenate(a), np.concatenate(b)
    N = n * S2
    _nc, lab = connected_components(sp.coo_matrix((np.ones(len(a_), dtype=np.int8), (a_, b_)),
                                                  shape=(N, N)), directed=False)
    lab = lab.reshape(n, S2)
    seed = np.zeros(int(lab.max()) + 1, dtype=bool)
    seed[lab[g.ground][op[g.ground]]] = True
    return (seed[lab] & op).any(axis=1)


def pockets_composite(b) -> None:
    """Q1's pockets on an oversized block, on its composite mesh (OVERSIZED_MESH) under today's
    rule: per home the share of its injection in cells the flood fill does not reach (behind a
    phantom passage, `phantom`); the sub-cell gaps' widths (fine_bottleneck's distance transform)
    are not computed here (`behind` nan). Rows aperture_rows/pockets/<id>.parquet."""
    t0 = time.time()
    p = params(RULES["T"], "cpu")
    g = common.mesh_of(OVERSIZED_MESH).build(b, p, AREA)
    polys = np.asarray(b.buildings.outlines)
    pl = common.placement(g, polys, p)
    fl = composite_flood(g)
    ff = g.ff0.ravel()
    phantom = np.asarray(pl.S[(ff > 0) & ~fl].sum(axis=0)).ravel()
    out = pd.DataFrame(dict(block=b.block_id, home=np.arange(len(polys)), behind=np.nan,
                            phantom=np.where(pl.enclosed, np.nan, phantom))).assign(
        t=time.time() - t0)
    pockets_path(b.block_id).parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(out, pockets_path(b.block_id))
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} composite pockets in "
          f"{time.time() - t0:.0f}s", flush=True)


def _pockets_ov_one(bid: str) -> str | None:
    if pockets_path(bid).exists():
        return None
    try:
        [b] = common.build_blocks([bid])
        pockets_composite(b)
    except Exception as e:                  # noqa: BLE001 -- reported, the run goes on
        print(f"{bid} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        return bid
    return None


def pockets_ov(ids: list[str], procs: int) -> None:
    rr._check_single_threaded()
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        failed = [f for f in pool.imap_unordered(_pockets_ov_one, ids) if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


def moves_path(bid: str) -> Path:
    return ROWS / "moves" / f"{bid}.parquet"


MOVE_RULES = ("Aw", "Aws", "Fws")    # `moves`: the rules whose demand moves are split by cause


def moves(b) -> None:
    """Q2's demand moves by cause (geometry only, the uniform grid at the metric's offset): per
    home, the share of its injection under today's placement that each rule of MOVE_RULES takes
    off cells the 4-connected flood fill of open sub-samples does not reach (`pocket`: behind a
    phantom passage, today reaching the street only through it) and off cells it does reach
    (`sliver`: an edge sliver, the cell's centre footprint, or behind a bent passage, which no
    straight track from a reached centre crosses; or a split cell); a home enclosed under either
    rule moves nothing here. Rows aperture_rows/moves/<id>.parquet."""
    t0 = time.time()
    polys = np.asarray(b.buildings.outlines)
    g = lifted.Grid.of(*lifted._geometry(b), H, offset=OFFSETS[0])
    fl = _flood(g)
    plT = common.placement(g, polys, params(RULES["T"], "cpu"))
    ST = plT.S.tocsc()
    out = dict(block=b.block_id, home=np.arange(len(polys)), enclosed_T=plT.enclosed)
    for r in MOVE_RULES:
        p = params(RULES[r], "cpu")
        ok = lifted.placeable(g, g.ff0, p).ravel()
        pl = common.placement(g, polys, p)
        live = ~plT.enclosed & ~pl.enclosed
        off = ~ok
        out[f"pocket_{r}"] = np.where(live, np.asarray(ST[off & ~fl].sum(axis=0)).ravel(), np.nan)
        out[f"sliver_{r}"] = np.where(live, np.asarray(ST[off & fl].sum(axis=0)).ravel(), np.nan)
        out[f"enclosed_{r}"] = pl.enclosed
    df = pd.DataFrame(out).assign(t=time.time() - t0)
    moves_path(b.block_id).parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(df, moves_path(b.block_id))
    print(f"{time.strftime('%H:%M:%S')} {b.block_id} moves in {time.time() - t0:.0f}s", flush=True)


def _moves_one(bid: str) -> str | None:
    if moves_path(bid).exists():
        return None
    try:
        [b] = common.build_blocks([bid])
        moves(b)
    except Exception as e:                  # noqa: BLE001 -- reported, the run goes on
        print(f"{bid} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        return bid
    return None


def moves_all(ids: list[str], procs: int) -> None:
    rr._check_single_threaded()
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=4) as pool:
        failed = [f for f in pool.imap_unordered(_moves_one, ids) if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


def _local_one(job: tuple[str, bool]) -> str | None:
    """A pool's task (the CPU): a small block's `base` under every rule and its pockets, or (a
    large block, `base` on the cluster) its pockets alone; the block's id if it failed."""
    bid, small = job
    try:
        [b] = common.build_blocks([bid])
        if small and not base_path(bid, "").exists():
            base(b, "cpu", "0.5", BASE_RULES_SMALL, "")
        if not pockets_path(bid).exists():
            pockets(b)
    except Exception as e:                  # noqa: BLE001 -- reported, the run goes on
        print(f"{bid} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        return bid
    return None


def _base_cpu(job: tuple[str, tuple[str, ...], str]) -> str | None:
    """A pool's task: `base` on the CPU under `rules` (a block that fits this machine's memory),
    rows tagged `tag`; the block's id if it failed."""
    bid, rules, tag = job
    if base_path(bid, tag).exists():
        return None
    try:
        [b] = common.build_blocks([bid])
        base(b, "cpu", mesh_token_of(bid), rules, tag)
    except Exception as e:                  # noqa: BLE001 -- reported, the run goes on
        print(f"{bid} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        return bid
    return None


def base_local(ids: list[str], procs: int, rules: tuple[str, ...], tag: str) -> None:
    rr._check_single_threaded()
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        failed = [f for f in pool.imap_unordered(_base_cpu, [(i, rules, tag) for i in ids])
                  if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


def base_all(ids: list[str], procs: int) -> None:
    """`_local_one` for every block of `ids`, `procs` processes: the small ones (the 220) whole,
    the large ones' pockets."""
    rr._check_single_threaded()
    small = set(common.recipients())
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=4) as pool:
        failed = [f for f in pool.imap_unordered(_local_one, [(i, i in small) for i in ids]) if f]
    if failed:
        raise SystemExit(f"{len(failed)} blocks failed: {', '.join(failed)}")


# ------------------------------------------------------------- the fields of one block (Q3, Q4)

def score_path(off: int, rule: str, bid: str) -> Path:
    return ROWS / "score" / f"o{off}" / rule / f"{bid}.parquet"


def homes_score_path(off: int, rule: str, bid: str) -> Path:
    return ROWS / "homes_score" / f"o{off}" / rule / f"{bid}.parquet"


def score(b, rule: str, off: int, device: str) -> None:
    """One block under one rule at OFFSETS[off] (the module's docstring)."""
    rr._check_single_threaded()
    t0 = time.time()
    spec, bid = RULES[rule], b.block_id
    p = params(spec, device)
    sc = common.Scorer(b, lifted.UniformMesh(H, offset=OFFSETS[off]), p, population=AREA,
                       demand=POP)
    g, n = sc.grid, len(sc.polys)
    shape = g.inside.shape
    inj_q = PQ.injection(sc.w, sc.u0, sc.enclosed)
    f_q = sc.pl.field(inj_q, shape)
    sol_q = lifted.solve(g, sc.free0, f_q, p, rtol=SCORE_RTOL)
    uq0 = sc.home_u_of(sol_q, sc.free0)
    J2_0 = sc.J(sc.u0, 2.0)
    lab = g.label_sub(sc.polys)
    cost = sc.w / sc.w.sum()
    c = clear.Clearing.on_scorer(sc, rtol=clear.RTOL_TENSION, search=None)
    store = clearings.Store.committed()
    fields: dict[str, tuple[str, np.ndarray, float, str]] = {
        "baseline": ("baseline", sc.free0, 0.0, "")}
    for name, run in STORED.items():
        x = store.cleared(run, b, step=1)
        o = width._open_of(g, lab, x, n)
        fields[name] = ("stored", o, float(cost[x].sum()), "")
        if sc.enclosed_left(o) > 0:
            # the fix leaves homes enclosed that the clearing does not free: free them first
            # (free_first under this rule, the clearing held, no budget: D grows by the cost)
            held0 = np.zeros(n, dtype=bool)
            held0[x] = True
            held = clear.free_first(c, budget=1.0, held=held0)
            add = np.flatnonzero(held & ~held0)
            fields[name + "+free"] = ("stored+free", c.open_(held), float(cost[held].sum()),
                                      " ".join(map(str, add)))
    if off == 0:
        for name, (o, D) in width._road_fields(b, g, bid).items():
            fields[name] = ("road", o, D, "")
    t_setup = time.time() - t0
    rows, per_home = [], {"u0": sc.u0, "uq0": uq0}
    for name, (kind, o, D, added) in fields.items():
        ts = time.time()
        sy = lifted.System(g, o, p)
        s1 = lifted.solve(g, o, sc.f, p, system=sy, rtol=SCORE_RTOL)
        u = sc.home_u_of(s1, o)
        sq = lifted.solve(g, o, f_q, p, system=sy, rtol=SCORE_RTOL)
        if name == "simp_default":
            per_home.update(u_default=u, uq_default=sc.home_u_of(sq, o))
        row = dict(block=bid, rule=rule, spec=spec, offset=off, device=device, field=name,
                   kind=kind, D=D, added=added, enclosed=sc.enclosed_left(o), P=s1.P, Pq=sq.P,
                   L_P=1.0 - s1.P / sc.P0, L_J2=sc.perm_p(u, 2.0),
                   **{"L_Pq0.5": 1.0 - sq.P / sol_q.P},
                   umed=float(np.nanmedian(u) / np.nanmedian(sc.u0)),
                   u95=float(np.nanpercentile(u, 95) / np.nanpercentile(sc.u0, 95)),
                   umax=float(np.nanmax(u) / np.nanmax(sc.u0)), unknowns=s1.n_unknowns,
                   t=time.time() - ts)
        rows.append(row)
        print(f"{bid} {rule} o{off} {name}: L_P {row['L_P']:.4f} L_J2 {row['L_J2']:.4f} L_Pq "
              f"{row['L_Pq0.5']:.4f} enclosed {row['enclosed']:.4f} D {D:.4f} "
              f"({row['t']:.0f}s)", flush=True)
        del sy
        p.solver.release()
    setup = dict(cells=int(g.inside.sum()), P0=sc.P0, J2_0=J2_0, Pq0=sol_q.P,
                 enclosed0=sc.enclosed0, n_enclosed0=int(sc.enclosed.sum()), t_setup=t_setup,
                 t_block=time.time() - t0, host=socket.gethostname(),
                 peak_rss_gb=rr._status_gb("VmHWM"))
    out = pd.DataFrame(rows).assign(**setup)
    homes = pd.DataFrame(dict(block=bid, rule=rule, offset=off, home=np.arange(n), w=sc.w,
                              inj_q=inj_q, enclosed=sc.enclosed, **per_home))
    for path, df in ((score_path(off, rule, bid), out), (homes_score_path(off, rule, bid), homes)):
        path.parent.mkdir(parents=True, exist_ok=True)
        rr.write_parquet(df, path)
    print(f"{time.strftime('%H:%M:%S')} {bid} {rule} o{off} done in {time.time() - t0:.0f}s, "
          f"peak RSS {setup['peak_rss_gb']:.1f} GB", flush=True)


def score_plan(bid: str, drift: tuple[str, ...], tuning: tuple[str, ...]
               ) -> list[tuple[str, int]]:
    """The (rule, offset) pairs `score` runs on a block: Q3's drift blocks under the rules
    `drift` at the metric's offset and their shift (ALIGN), the 13 tuning blocks (Q4) under the
    rules `tuning` at the metric's offset (today's fields there are width.py's U rows). The runs:
    drift T, Aw, Aws and tuning Aws (ap-score*), then drift and tuning Fws (ap-score-f)."""
    if bid in ALIGN:
        return [(r, o) for o in (0, ALIGN[bid]) for r in drift]
    if bid in TUNING:
        return [(r, 0) for r in tuning]
    raise SystemExit(f"{bid}: neither a drift block nor a tuning block")


def score_all(bid: str, device: str, drift: tuple[str, ...], tuning: tuple[str, ...]) -> None:
    """`score` for every pair of score_plan whose rows are not here yet, the block built once."""
    [b] = common.build_blocks([bid])
    for rule, off in score_plan(bid, drift, tuning):
        if score_path(off, rule, bid).exists():
            print(f"{bid} {rule} o{off}: rows here already", flush=True)
            continue
        score(b, rule, off, device)


def tars(out: Path, ids: list[str]) -> None:
    """Per block, what `score` reads that git does not carry: item 4's prefix rows and ordered
    networks of ROADS (the stored clearings are committed), at their repo paths."""
    width.tars(out, ids, "score")


TODAYS_SIMP = {1.0: "pweight_rows/simp/P", 2.0: "pweight_rows/simp/J2"}   # SIMP base under
                                        # today's rule, free first, area D 0.05: the small blocks


def simp(bid: str, rule: str, device: str, power: float) -> None:
    """SIMP base (SIMP_PLAN), J_power, area D BUDGET, free first, under the rule (relax.one):
    whether the relaxation's gradient works under it. Beside it, today's SIMP base clearing of the
    same objective (TODAYS_SIMP, a small block's) scored under the rule, with the homes it leaves
    enclosed freed first (D grows): SIMP under the rule should not be behind it on its own lens.
    Rows aperture_rows/simp/<rule>/."""
    rr._check_single_threaded()
    t0 = time.time()
    row = relax.one(bid, power, device, SIMP_PLAN, RULES[rule], BUDGET, demand=POP)
    row.update(rule=rule, spec=RULES[rule], budget=BUDGET, host=socket.gethostname(),
               peak_rss_gb=rr._status_gb("VmHWM"), wall=time.time() - t0)
    c = relax._clearing(bid, device, RULES[rule], SIMP_PLAN.mesh, demand=POP)
    rel = relax.Relaxation(c, power)
    x = clearings.Store.committed().cleared(TODAYS_SIMP[power], c.sc.block, step=1)
    held0 = np.zeros(c.n, dtype=bool)
    held0[x] = True
    held = clear.free_first(c, budget=1.0, held=held0)
    for name, r in (("todays", held0), ("todays_free", held)):
        row[f"{name}_perm"] = rel.perm(rel.exact(r.astype(float)))
        row[f"{name}_D"] = float(c.cost[r].sum())
        row[f"{name}_enclosed"] = c.sc.enclosed_left(c.open_(r))
    row["jaccard_todays"] = len(set(x.tolist()) & set(row["simp_cleared"])) / len(
        set(x.tolist()) | set(row["simp_cleared"]))
    out = ROWS / "simp" / rule / f"{bid}_p{power:g}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.write_parquet(pd.DataFrame([row]), out)
    print(f"{time.strftime('%H:%M:%S')} {bid} simp {rule} p{power:g} done in "
          f"{time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ the report

LENS_COLS = ("L_P", "L_Pq0.5", "L_J2")          # P, the equity view P_{1/2}, then J_2
REACH_TOL = width.REACH_TOL                     # a road prefix reaches D 0.05 at BUDGET - this
U_ROWS = width.ROWS / "score"                   # today's rule's fields (width.py's U: the same
                                                # code path, bit for bit; GPU, at 4139177d)
CRACK_M = 0.25           # Q3: a crack one or two sub-samples wide (fine_bottleneck's steps)


def _read(kind: str) -> pd.DataFrame:
    files = sorted((ROWS / kind).rglob("*.parquet"))
    return (pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files
            else pd.DataFrame())


def _summary(name: str, df: pd.DataFrame) -> None:
    path = ROWS / f"summary_{name}.csv"
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"\n{name} -> {path}\n{df.to_string(index=False, float_format=lambda v: f'{v:.4g}')}",
          flush=True)


def _spread(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    """Per group: the blocks, and each column's median, 90th percentile, max and mean."""
    g = df.groupby(by)
    out = g.size().rename("blocks").to_frame()
    for c in cols:
        out[f"{c}_med"] = g[c].median()
        out[f"{c}_p90"] = g[c].quantile(0.9)
        out[f"{c}_max"] = g[c].max()
        out[f"{c}_mean"] = g[c].mean()
    return out.reset_index()


def _size_class(bids: pd.Series) -> pd.Series:
    small = set(common.recipients())
    return bids.map(lambda b: "small" if b in small else "large")


def _q1(homes: pd.DataFrame, pk: pd.DataFrame) -> pd.DataFrame:
    """Q1 per block, today's rule at baseline: the population (homes with over BEHIND of their
    injection behind) and the demand (each home's injection share) behind a sub-cell gap or a
    phantom passage, and behind a phantom passage, and those homes' share of P0, P_{1/2}'s and
    J_2's baselines."""
    t = homes[homes.rule == "T"].merge(pk[["block", "home", "behind", "phantom"]],
                                       on=["block", "home"], how="inner")
    out = []
    for bid, g in t.groupby("block"):
        live = ~g.enclosed.to_numpy()
        w, u0 = g.w.to_numpy(), np.nan_to_num(g.u0.to_numpy())
        tq = (g.inj_q * np.nan_to_num(g.u_q)).to_numpy()
        terms = dict(P=w * u0, Pq=tq, J2=w * u0 ** 2)
        row = dict(block=bid, homes=int(live.sum()))
        for name, share in (("pocket", np.nan_to_num(g.behind.to_numpy())),
                            ("phantom", np.nan_to_num(g.phantom.to_numpy()))):
            hit = live & (share > BEHIND)
            row[f"{name}_homes"] = int(hit.sum())
            row[f"{name}_pop"] = float(w[hit].sum() / w[live].sum())
            row[f"{name}_demand"] = float((w * share)[live].sum() / w[live].sum())
            for lens, x in terms.items():
                row[f"{name}_{lens}"] = float(x[hit].sum() / x[live].sum())
        out.append(row)
    return pd.DataFrame(out)


def _q2(base: pd.DataFrame, homes: pd.DataFrame, mv: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Q2 per block, `rule` against today's at baseline: the homes it encloses (and of them those
    the flood fill reaches, enclosed by the rule rather than the geometry), the homes it frees,
    the demand it moves (and, from `moves`, how much off cells behind a phantom passage and how
    much off edge slivers and bent passages), the split cells, free first's cost, and the
    baselines' change."""
    out = []
    by = {bid: m.set_index("home") for bid, m in mv.groupby("block")} if not mv.empty else {}
    for bid, g in homes.groupby("block"):
        hT, hR = g[g.rule == "T"].set_index("home"), g[g.rule == rule].set_index("home")
        if hR.empty:
            continue
        w = hT.w.to_numpy()
        eT, eR = hT.enclosed.to_numpy(), hR.enclosed.to_numpy()
        new, freed, both = eR & ~eT, eT & ~eR, ~eT & ~eR
        moved = np.nan_to_num(hR.moved.to_numpy())
        nf = hR.near_flood.to_numpy()
        bT = base[(base.block == bid) & (base.rule == "T")].iloc[0]
        bR = base[(base.block == bid) & (base.rule == rule)].iloc[0]
        out.append(dict(
            block=bid, rule=rule, mesh=bR["mesh"], homes=len(w),
            enclosed_T=int(eT.sum()), enclosed=int(eR.sum()),
            new=int(new.sum()), new_pop=float(w[new].sum() / w.sum()),
            new_by_rule=int((new & (nf == 1.0)).sum()),
            new_by_rule_pop=float(w[new & (nf == 1.0)].sum() / w.sum()),
            freed=int(freed.sum()),
            moved=int((both & (moved > 1e-12)).sum()),
            moved_pop=float(w[both & (moved > 1e-12)].sum() / w.sum()),
            moved_demand=float((w * moved)[both].sum() / w.sum()),
            split=int(bR["split"]), split_reached=int(bR["split_reached"]),
            split_pop=float(bR["split_pop"]),
            free_D_T=float(bT["free_D"]), free_n_T=int(bT["free_n"]),
            free_D=float(bR["free_D"]), free_n=int(bR["free_n"]),
            free_left=float(bR["free_left"]),
            P0=float(bR["P0"] / bT["P0"]), Pq0=float(bR["Pq0"] / bT["Pq0"]),
            J2_0=float(bR["J2_0"] / bT["J2_0"]),
            op_only=bR.get("op_only", np.nan), flood_only=bR.get("flood_only", np.nan)))
        m = by.get(bid)
        if m is not None and f"pocket_{rule}" in m:
            pk, sv = (np.nan_to_num(m[f"{c}_{rule}"].to_numpy()) for c in ("pocket", "sliver"))
            out[-1].update(moved_pocket=int((pk > 1e-12).sum()),
                           moved_pocket_pop=float(w[pk > 1e-12].sum() / w.sum()),
                           moved_pocket_demand=float((w * pk).sum() / w.sum()),
                           moved_sliver=int((sv > 1e-12).sum()),
                           moved_sliver_demand=float((w * sv).sum() / w.sum()))
    return pd.DataFrame(out)


def _drift_homes(sh: pd.DataFrame, cells: dict) -> pd.DataFrame:
    """Q3, per drift block, rule and lens (P, P_{1/2}, J_2): how each home's term in the ratio
    J / J0 under SIMP's default moves from the metric's offset to the block's shift, by what the
    home sits behind at either alignment (`cells`: fine_bottleneck per (block, offset): a
    phantom passage; else a crack of at most CRACK_M (one or two sub-samples); else another
    sub-cell gap; else nothing), over BEHIND of its injection: per group its homes, population,
    share of J0 and of the ratio's change (signed and absolute)."""
    out = []
    for (bid, rule), g in sh.groupby(["block", "rule"]):
        k = ALIGN[bid]
        at = {o: g[g.offset == o].set_index("home").sort_index() for o in (0, k)}
        if any(x.empty for x in at.values()):
            continue
        live = (~(at[0].enclosed | at[k].enclosed)).to_numpy()
        w = at[0].w.to_numpy()
        grp = np.full(len(w), "none", dtype=object)
        for name, key in (("sub-cell gap", "behind"), ("crack <= 0.25 m", "crack"),
                          ("phantom", "phantom")):
            share = np.fmax(cells[(bid, 0)][key], cells[(bid, k)][key])
            grp[np.nan_to_num(share) > BEHIND] = name
        for lens in ("P", "Pq", "J2"):
            c, base0 = {}, {}
            for o, h in at.items():
                if lens == "Pq":
                    a = h.inj_q.to_numpy()
                    u0, ud = np.nan_to_num(h.uq0.to_numpy()), np.nan_to_num(h.uq_default.to_numpy())
                    p = 1.0
                else:
                    a = w
                    u0, ud = np.nan_to_num(h.u0.to_numpy()), np.nan_to_num(h.u_default.to_numpy())
                    p = 2.0 if lens == "J2" else 1.0
                J0 = float((a[live] * u0[live] ** p).sum())
                c[o] = np.where(live, a * ud ** p / J0, 0.0)
                base0[o] = np.where(live, a * u0 ** p / J0, 0.0)
            d = c[k] - c[0]
            total, absum = float(d.sum()), float(np.abs(d).sum())
            for name in ("none", "sub-cell gap", "crack <= 0.25 m", "phantom"):
                m = live & (grp == name)
                out.append(dict(block=bid, rule=rule, shift=k, lens=lens, behind=name,
                                homes=int(m.sum()), population=float(w[m].sum() / w[live].sum()),
                                J0_share=float(base0[0][m].sum()),
                                J0_share_shift=float(base0[k][m].sum()),
                                change=float(d[m].sum()), change_share=float(d[m].sum() / total),
                                abs_share=float(np.abs(d[m]).sum() / absum),
                                ratio_o0=float(c[0].sum()), ratio_shift=float(c[k].sum())))
    return pd.DataFrame(out)


def _cells_behind(bid: str, off: int) -> dict[str, np.ndarray]:
    """fine_bottleneck at OFFSETS[off] per home: the injection share behind a sub-cell gap
    (`behind`), a phantom passage (`phantom`), and a crack of at most CRACK_M not a phantom
    passage (`crack`)."""
    [b] = common.build_blocks([bid])
    cb, behind, phantom = width.fine_bottleneck(b, off)
    g = lifted.Grid.of(*lifted._geometry(b), H, offset=OFFSETS[off])
    pl = common.placement(g, np.asarray(b.buildings.outlines), params("uni", "cpu"))
    crack = pl.S.T.tocsr() @ ((cb > 0) & (cb <= CRACK_M + 1e-9)).astype(float)
    return dict(behind=behind, phantom=phantom, crack=np.where(pl.enclosed, np.nan, crack))


def _tau_table(t13: pd.DataFrame, ref: pd.DataFrame, rule: str) -> tuple[pd.DataFrame, ...]:
    """Q4 per tuning block and lens: Kendall tau between today's method order (`ref`, B's order:
    the least population left enclosed, then the lens) and `rule`'s, over the stored clearings
    (each with the homes the rule encloses freed first, where it encloses any: `+free`) and the
    road prefixes that reach D BUDGET; which method leads and which road leads."""
    from scipy.stats import kendalltau
    taus, leads = [], []
    for bid, g in t13.groupby("block"):
        r = ref[ref.block == bid].set_index("field")
        g = g.set_index("field")
        names = [n for n in r.index if n in g.index]
        for n in ("simp_default", "cheap"):        # the rule's version: freed first if needed
            if f"{n}+free" in g.index:
                g.loc[n] = g.loc[f"{n}+free"]
        a, b = r.loc[names].reset_index(), g.loc[names].reset_index()
        for col in LENS_COLS:
            ra, rb = rr._b_rank(a, "enclosed", col), rr._b_rank(b, "enclosed", col)
            taus.append(dict(block=bid, rule=rule, lens=col, methods=len(names),
                             tau=float(kendalltau(ra, rb).statistic)))
            roads = b[b.kind == "road"]
            leads.append(dict(block=bid, rule=rule, lens=col,
                              leader=b.field[np.argmin(rb)], leader_T=a.field[np.argmin(ra)],
                              best_road=roads.field.iloc[np.argmin(rr._b_rank(roads, "enclosed",
                                                                              col))],
                              simp_minus_best=float(b.set_index("field").loc["simp_default", col]
                                                    - roads[col].max())))
    return pd.DataFrame(taus), pd.DataFrame(leads)


def report() -> None:
    """Every question's tables, written to aperture_rows/summary_*.csv (committed)."""
    pd.set_option("display.width", 240)
    names = list(RULES)
    # Q3: the corridor
    ch = pd.read_parquet(ROWS / "channel.parquet")
    for check, label in (("angle", "q3_angle"), ("align", "q3_align"),
                         ("subcell", "q3_subcell")):
        c = ch[ch.check == check].groupby(["rule", "W"])
        t = c.P.agg(lambda s: s.max() / s.min()).unstack().reindex(names).reset_index()
        lost = c.demand_lost.max().unstack().reindex(names)
        t = t.merge(lost.add_prefix("lost_").reset_index(), on="rule")
        _summary(label, t)
    w = ch[ch.check == "width"]
    rel = []
    for (ang, W), g in w.groupby(["angle", "W"]):
        u = float(g[g.rule == "T"].P.iloc[0])
        rel += [dict(angle=ang, W=W, rule=r.rule, C_over_T=u / r.P) for r in g.itertuples()]
    t = pd.DataFrame(rel).pivot(index=["angle", "rule"], columns="W", values="C_over_T")
    _summary("q3_width", t.reset_index())
    # Q1, Q2: every block at baseline
    base_, homes = _read("base"), _read("homes")
    pk = _read("pockets")
    if not base_.empty:
        if not pk.empty:
            q1 = _q1(homes, pk)
            q1.insert(1, "size", _size_class(q1.block))
            _summary("q1_blocks", q1)
            cols = [c for c in q1.columns if c not in ("block", "size", "homes")]
            _summary("q1", _spread(q1, ["size"], cols))
        mv = _read("moves")
        q2 = pd.concat([_q2(base_, homes, mv, r) for r in ("A", "Aw", "Aws", "Fws")],
                       ignore_index=True)
        q2.insert(1, "size", _size_class(q2.block))
        _summary("q2_blocks", q2)
        cols = [c for c in ("new", "new_pop", "new_by_rule", "new_by_rule_pop", "freed", "moved",
                            "moved_pop", "moved_demand", "moved_pocket", "moved_pocket_pop",
                            "moved_pocket_demand", "moved_sliver", "moved_sliver_demand", "split",
                            "split_pop", "free_D_T", "free_D", "free_n_T", "free_n", "P0", "Pq0",
                            "J2_0") if c in q2]
        sp_ = _spread(q2, ["rule", "size"], cols)
        tot = q2.groupby(["rule", "size"]).agg(
            blocks_new=("new", lambda s: int((s > 0).sum())), homes_new=("new", "sum"),
            homes_new_by_rule=("new_by_rule", "sum"), homes_freed=("freed", "sum"),
            blocks_enclosed_T=("enclosed_T", lambda s: int((s > 0).sum())),
            blocks_enclosed=("enclosed", lambda s: int((s > 0).sum())),
            split_cells=("split", "sum"), free_left_max=("free_left", "max")).reset_index()
        _summary("q2", sp_.merge(tot, on=["rule", "size"]))
    # Q3: the drift blocks' levels and who carries the change
    sc = _read("score")
    if not sc.empty:
        lv = []
        for (bid, rule), g in sc[sc.block.isin(list(ALIGN))].groupby(["block", "rule"]):
            k = ALIGN[bid]
            at = {o: g[(g.offset == o) & (g.field == "simp_default")] for o in (0, k)}
            if any(len(x) != 1 for x in at.values()):
                continue
            row = dict(block=bid, rule=rule, shift=k)
            for col in LENS_COLS:
                row[f"{col}_o0"] = float(at[0][col].iloc[0])
                row[f"{col}_o{k}"] = float(at[k][col].iloc[0])
                row[f"{col}_change"] = row[f"{col}_o{k}"] - row[f"{col}_o0"]
            row["enclosed_o0"] = float(at[0].enclosed.iloc[0])
            row["enclosed_shift"] = float(at[k].enclosed.iloc[0])
            lv.append(row)
        if lv:
            _summary("q3_levels", pd.DataFrame(lv))
        sh = _read("homes_score")
        if not sh.empty:
            cells = {(bid, o): _cells_behind(bid, o) for bid, k in ALIGN.items() for o in (0, k)}
            _summary("q3_homes", _drift_homes(sh[sh.block.isin(list(ALIGN))], cells))
        # Q4: the 13 tuning blocks, today's method order against the rule's
        ref = pd.concat([pd.read_parquet(f) for f in sorted((U_ROWS / "o0" / "U").glob(
            "*.parquet"))], ignore_index=True)
        s0 = sc[(sc.offset == 0) & sc.block.isin(TUNING)]
        reach = (s0.kind != "road") | (s0.D >= BUDGET - REACH_TOL)
        ref_reach = ref[ref.block.isin(TUNING)]
        ref_reach = ref_reach[(ref_reach.kind != "road") | (ref_reach.D >= BUDGET - REACH_TOL)]
        taus, leads = [], []
        for rule, g in s0[reach].groupby("rule"):
            t, ld = _tau_table(g, ref_reach, rule)
            taus.append(t)
            leads.append(ld)
        if taus:
            tt = pd.concat(taus, ignore_index=True)
            _summary("q4_tau", tt.groupby(["rule", "lens"]).tau.agg(
                median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75),
                min="min", blocks="count").reset_index())
            ld = pd.concat(leads, ignore_index=True)
            _summary("q4_leads", ld.groupby(["rule", "lens"]).agg(
                leader=("leader", lambda s: s.value_counts().idxmax()),
                leader_blocks=("leader", lambda s: int(s.value_counts().max())),
                leader_T=("leader_T", lambda s: s.value_counts().idxmax()),
                best_road=("best_road", lambda s: s.value_counts().idxmax()),
                best_road_blocks=("best_road", lambda s: int(s.value_counts().max())),
                simp_minus_best=("simp_minus_best", "median")).reset_index())
            st = s0[s0.kind.isin(["stored", "stored+free"])]
            _summary("q4_stored", st[["block", "rule", "field", "D", "added", "enclosed",
                                      *LENS_COLS]])
    simp_rows = _read("simp")
    if not simp_rows.empty:
        _summary("simp", simp_rows[[c for c in (
            "block", "rule", "power", "simp_perm", "simp_D", "free_D", "free_cleared", "enclosed",
            "todays_perm", "todays_D", "todays_enclosed", "todays_free_perm", "todays_free_D",
            "todays_free_enclosed", "jaccard_todays", "t") if c in simp_rows]])


def _ids(arg: str) -> list[str]:
    return Path(arg[1:]).read_text().split() if arg.startswith("@") else arg.split(",")


def main(argv: list[str]) -> None:
    cmd, *a = argv
    if cmd == "channel":
        channel(int(a[0]))
    elif cmd == "base":
        token, rules = a[2], tuple(a[3].split(","))     # the mesh as cluster.py sized the task
        if not set(rules) <= set(RULES):
            raise SystemExit(f"rules {rules}: any of {list(RULES)}")
        [b] = common.build_blocks([a[0]])
        base(b, a[1], token, rules, a[4] if len(a) > 4 else "")
    elif cmd == "base-all":
        base_all(_ids(a[0]), int(a[1]))
    elif cmd == "pockets-composite":
        pockets_ov(_ids(a[0]), int(a[1]))
    elif cmd == "moves":
        moves_all(_ids(a[0]), int(a[1]))
    elif cmd == "base-local":
        rules = tuple(a[2].split(","))
        if not set(rules) <= set(RULES):
            raise SystemExit(f"rules {rules}: any of {list(RULES)}")
        base_local(_ids(a[0]), int(a[1]), rules, a[3] if len(a) > 3 else "")
    elif cmd == "score":
        if a[4] != "0.5":
            raise SystemExit("score runs on uniform h 0.5 (its mesh token)")
        [b] = common.build_blocks([a[0]])
        score(b, a[1], int(a[2]), a[3])
    elif cmd == "score-plan":
        score_all(a[0], a[1], tuple(a[2].split(",")), tuple(a[3].split(",")))
    elif cmd == "tars":
        tars(Path(a[0]), _ids(a[1]))
    elif cmd == "simp":
        simp(a[0], a[1], a[2], float(a[3]))
    elif cmd == "report":
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
