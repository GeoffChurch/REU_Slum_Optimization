"""Walls that leak, measured one setting at a time (owner 2026-10-21: "we should launch these one at
a time so we can be adaptive"; NOTES, "Walls that leak: every home's source inside it, each wall
priced as a length"; the brief .superpowers/sdd/2026-10-21-leaky-walls). Each setting is a walls
part (gap_walls: `Wk<k_in>l<ell>` symmetric, `Wk<k_in>o<ell_out>i<ell_in>n<passes>` a diode) on
UNIFORM's mesh, leaks and doors (metrics.walls_on_uniform), scored at baseline against T (RUNTIME).

    S=research/roadless/leaky.py        # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S block <id> <part> [--replace]    # ROWS/<part>/{facts,homes}/<id>.parquet
    uv run python $S report <part>                     # ROWS/<part>/summary_*.csv
    uv run python $S placement <id> [--replace]        # ROWS/placement/<id>.parquet

`block`, at baseline (nothing cleared), T's model and W's on the same core: per home (homes/) its
population share, whether T walls it in (B (i)) or gives it no demand, its escape time under T and
under W and their ratio, and its own wall: the share of its escape time spent crossing its own
outline (its escape time less the potential just outside its walls, the mean over the passages
out of it weighted by their outward current, its street faces' outside at 0, over its escape
time); per block (facts/) the homes' counts, P0, P_{1/2},0 and J_2,0 under each, the walled-in
homes' shares of W's P0 and J_2,0, the through-traffic (the current entering buildings from open
ground over all the current injected; and from another building, through a party wall or a zone two
buildings share), the shares of W's P0 dissipated in walls (wall passages
and the inside street faces) and in interiors (passages inside one building set), and the cost:
each model's unknowns and its baseline scoring's time (the state, its factorizations and the two
solves, P's and P_{1/2}'s), and W's passes per solve.

`report`: per block (summary_blocks.csv) and pooled over the walled-in blocks and over the controls
(summary_groups.csv): (1) of the open homes (connected under T at baseline, with T's demand) the
share whose escape time moves by more than MOVED from T's, and the ratio W / T's median, p10 and
p90 (a home with no footprint in the mesh, under a barrier's cut, has no escape time under W and is
left out); (2) the walled-in homes' escape time against their block's median home under W (median,
p90, max; and apart by their situation, `placement`: in a pocket, or with no open ground within
ring_m); (3) their share of W's P0 and J_2,0; (4) the own wall's share, median and p90, for open
and walled-in homes apart; (5) the through-traffic and the dissipation's shares; (6) the unknowns
and times W / T, and the passes; (7) the open homes' rank agreement with T (Spearman's correlation
of their escape times under W and T, by block: median, p10, min); (8) P0 under W over T's, by block
(median, p10, p90); (9) the walls' dissipation split by the way its current crosses (out of a
building onto open ground, the inside street faces among them; into one from open ground; party
walls, building to building, a shared zone's two steps among them), median and pooled;
(10) the diode's passes per solve (median, max; a block whose loop hits its cap raises and has no
row).

The blocks: BLOCKS (the small blocks with walled-in homes under RUNTIME, and the uniform rule's five
controls).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/leaky.py
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
import scipy.sparse as sp
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_calib  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_walls as gw  # noqa: E402
import metrics  # noqa: E402

ROWS = HERE / "leaky_rows"
BLOCKS = ROWS / "blocks.csv"        # block, size, control
POP = common.PopulationDemand()
MOVED = 0.05     # an open home's escape time moved: W / T off 1 by more than this


class Own(NamedTuple):
    """Per home its own wall (`own_wall`): the outward current through its walls and street
    faces, the potential just outside them (that current's mean), and the share of its escape
    time spent crossing them."""
    current: np.ndarray
    outside: np.ndarray
    share: np.ndarray


class Flow(NamedTuple):
    """W's flow at a state under the population's demand (`flow`): the injection, the current
    entering buildings from open ground and from another building (a party wall, or a zone two
    buildings share entered from one of them), the dissipation in walls (split by the way its
    current crosses them: out of a building onto open ground, the inside street faces among
    them; into one from open ground; party walls, from one building's grown footprint to
    another's, a shared zone's two steps among them), in interiors and in all."""
    injected: float
    entering: float
    entering_bld: float
    walls: float
    walls_out: float
    walls_in: float
    walls_party: float
    interiors: float
    total: float


def own_wall(wm: gw.WallModel, wl: gw.WallLenses) -> Own:
    """Each home's own wall under the population's demand (module docstring)."""
    st, s = wl.state, wl.pop
    c = wm.sys.cells
    w = st.wall
    stand = sp.diags(st.standing.astype(np.float64))
    za = (wm.zone[wm.pa[w]] @ stand).tocsr()
    zb = (wm.zone[wm.pb[w]] @ stand).tocsr()
    za.eliminate_zeros()
    zb.eliminate_zeros()
    za.data[:] = 1.0
    zb.data[:] = 1.0
    both = za.multiply(zb)
    out_a = (za - both).tocoo()           # (wall passage, home): it leaves the home first to second
    out_b = (zb - both).tocoo()           # ... second to first
    cur_w = s.g[w] * (s.u[c.ea[w]] - s.u[c.eb[w]])
    n = wm.n
    cur = np.zeros(n)
    mom = np.zeros(n)
    for coo, sign, cell in ((out_a, 1.0, c.eb[w]), (out_b, -1.0, c.ea[w])):
        keep = coo.data > 0
        q, h = coo.row[keep], coo.col[keep]
        i_out = np.maximum(sign * cur_w[q], 0.0)
        cur += np.bincount(h, weights=i_out, minlength=n)
        mom += np.bincount(h, weights=i_out * s.u[cell[q]], minlength=n)
    # the inside street faces: out of the home to the street (potential 0)
    zf = (wm.zone[wm.f_tri] @ stand).tocsr()
    zf.eliminate_zeros()
    zf = zf.tocoo()
    cur += np.bincount(zf.col, weights=st.Gf[zf.row] * s.u[c.fc[zf.row]], minlength=n)
    zg = (wm.zone[wm.g_tri] @ stand).tocsr()
    zg.eliminate_zeros()
    zg = zg.tocoo()
    cur += np.bincount(zg.col, weights=st.Gg[zg.row] * s.u[c.gcell[zg.row]], minlength=n)
    outside = np.where(cur > 0, mom / np.maximum(cur, 1e-300), np.nan)
    uh = wl.lenses.u
    return Own(current=cur, outside=outside, share=(uh - outside) / uh)


def flow(wm: gw.WallModel, wl: gw.WallLenses) -> Flow:
    """W's through-traffic and dissipation under the population's demand (module docstring)."""
    st, s = wl.state, wl.pop
    c = wm.sys.cells
    du = s.u[c.ea] - s.u[c.eb]
    cur = s.g * du
    diss = s.g * du ** 2
    wall = np.zeros(len(c.pe), dtype=bool)
    wall[st.wall] = True
    ia, ib = st.inside[wm.pa], st.inside[wm.pb]
    # per wall passage its current, the side it comes from inside a building, and whether it
    # enters one going that way (first to second: `enter`; back: `leave`)
    w = st.wall
    fwd = cur[w] > 0
    i_w = np.abs(cur[w])
    enters = np.where(fwd, st.enter, st.leave)
    from_in = np.where(fwd, ia[w], ib[w])
    party = ia[w] & ib[w]                   # inside on both sides: building to building
    d_w = diss[w]
    cell_in = np.zeros(c.n, dtype=bool)
    cell_in[c.cell[st.inside]] = True
    rhs = wm.rhs(st, wm.pop)
    graph = wm.sys.graph
    total = float(diss.sum() + ((st.G + graph.leak) * s.u ** 2)[st.unknown].sum())
    street_walls = float((st.G * s.u ** 2)[cell_in & st.unknown].sum())
    return Flow(injected=float(rhs.sum()), entering=float(i_w[enters & ~from_in].sum()),
                entering_bld=float(i_w[enters & from_in].sum()),
                walls=float(d_w.sum()) + street_walls,
                walls_out=float(d_w[~party & ~enters].sum()) + street_walls,
                walls_in=float(d_w[~party & enters].sum()),
                walls_party=float(d_w[party].sum()),
                interiors=float(diss[ia & ib & ~wall].sum()), total=total)


def placement(bid: str, *, standing: clearings.Standing) -> None:
    """Per home of `bid` where U puts it (its cached mesh's demand, every setting's base): placed in
    a pocket (walled in at baseline, its points on open ground no street reaches) or with no open
    ground within ring_m at all (U's homes without demand); ROWS/placement/<id>.parquet."""
    out = ROWS / "placement" / f"{bid}.parquet"
    if not standing.todo(out):
        return
    metric = metrics.metric_of(metrics.UNIFORM, device="cpu")
    [b] = common.build_blocks([bid])
    fx = metric.fixed(b, cache=gcl.CACHE, carving=gcl.Carving.none(gc.geometry(b)))
    df = pd.DataFrame(dict(block=bid, home=np.arange(len(fx.dem.pocket)), pocket_U=fx.dem.pocket,
                           noground_U=fx.dem.walled))
    standing.write(df, out)


def _outs(part: str, bid: str) -> tuple[Path, Path]:
    return ROWS / part / "facts" / f"{bid}.parquet", ROWS / part / "homes" / f"{bid}.parquet"


def block_run(bid: str, part: str, *, standing: clearings.Standing) -> None:
    """The block at baseline under T and under the walls `part` (module docstring)."""
    outs = _outs(part, bid)
    if not any([standing.todo(o) for o in outs]):   # every one asked
        return
    t_all = time.time()
    metric_w = metrics.walls_on_uniform(part, device="cpu")
    metric_t = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    pygc.collect()
    mt = metric_t.clearing(b, demand=POP)
    zero = np.zeros(mt.n, dtype=bool)
    t = time.time()
    lt = mt.lenses(zero)
    t_T = time.time() - t
    u_T, nodem_T, walled = lt.u, mt.nodem.copy(), mt.walled0.copy()
    P0_T, Pq0_T, J2_0_T, unk_T, pop = lt.P, lt.Pq, lt.J2, lt.unknowns, mt.pop.copy()
    share = mt.cost.copy()
    del mt, lt
    pygc.collect()
    wm = metric_w.model(b, demand=POP, start=gw.WarmStart())
    if not np.array_equal(wm.walled0, walled):
        raise RuntimeError(f"{bid}: W's base walls in other homes than T at baseline")
    t = time.time()
    n_before = dict(wm.timing.n)
    wl = wm.lenses(zero)
    t_W = time.time() - t
    if not np.array_equal(wl.lenses.live, ~walled):
        raise RuntimeError(f"{bid}: W's B (i) is not T's at baseline")
    base = wm.base_.lenses
    if abs(wl.lenses.P / base.P - 1.0) > 1e-9:
        raise RuntimeError(f"{bid}: W's baseline scored twice apart")
    own = own_wall(wm, wl)
    fl = flow(wm, wl)
    u_W = wl.lenses.u
    ok_w = ~wm.nodem
    med = float(np.median(u_W[ok_w]))
    open_ = ~walled & ~nodem_T & ~wm.nodem    # connected under T, an escape time under each
    ratio = u_W / u_T
    pU = pop * np.nan_to_num(u_W)
    j2 = pop * np.nan_to_num(u_W) ** 2
    st = wl.state
    homes = pd.DataFrame(dict(
        block=bid, home=np.arange(len(pop)), share=share, walled=walled, nodem_T=nodem_T,
        nodem_W=wm.nodem, u_T=u_T, u_W=u_W, ratio=ratio, u_W_med=u_W / med,
        own_current=own.current, own_outside=own.outside, own_share=own.share))
    facts = dict(
        block=bid, part=part, metric_W=metric_w.name, metric_T=metric_t.name, n=len(pop),
        n_walled=int(walled.sum()), n_open=int(open_.sum()), nodem_T=int(nodem_T.sum()),
        nodem_W=int(wm.nodem.sum()), P0_T=P0_T, Pq0_T=Pq0_T, J2_0_T=J2_0_T, P0_W=base.P,
        Pq0_W=base.Pq, J2_0_W=base.J2, median_u_W=med,
        moved=float(np.mean(np.abs(ratio[open_] - 1.0) > MOVED)) if open_.any() else np.nan,
        ratio_median=float(np.median(ratio[open_])) if open_.any() else np.nan,
        walled_P0_share=float(pU[walled].sum() / pU.sum()),
        walled_J2_share=float(j2[walled].sum() / j2.sum()),
        injected=fl.injected, entering=fl.entering, through=fl.entering / fl.injected,
        entering_bld=fl.entering_bld, through_bld=fl.entering_bld / fl.injected,
        diss_total=fl.total, diss_walls=fl.walls / base.P, diss_interiors=fl.interiors / base.P,
        diss_walls_out=fl.walls_out / base.P, diss_walls_in=fl.walls_in / base.P,
        diss_walls_party=fl.walls_party / base.P,
        energy_rel=fl.total / base.P - 1.0,
        walls=len(st.wall), inside_street_faces=int(st.inside[wm.f_tri].sum()),
        inside_obtuse_exits=int(st.inside[wm.g_tri].sum()),
        unknowns_T=unk_T, unknowns_W=wl.lenses.unknowns, t_T=t_T, t_W=t_W,
        factorizations_W=wm.timing.n["factorization"] - n_before["factorization"],
        passes_pop=wl.pop.passes, passes_q=wl.q.passes,
        passes_base=wm.base_.pop.passes + wm.base_.q.passes,
        host=socket.gethostname(), peak_rss_gb=gap_calib.peak_gb(), wall=time.time() - t_all)
    for o, df in zip(outs, (pd.DataFrame([facts]), homes), strict=True):
        standing.write(df, o)
    print(f"{bid} {part}: {int(walled.sum())} walled in; open homes moved {facts['moved']:.3f} "
          f"(ratio median {facts['ratio_median']:.3f}); through {facts['through']:.3f}; walls "
          f"{facts['diss_walls']:.3f} of P0; unknowns {unk_T} -> {wl.lenses.unknowns}; "
          f"{t_T:.1f}s -> {t_W:.1f}s; {facts['wall']:.0f}s", flush=True)


# ------------------------------------------------------------------ the report

def _read(part: str, kind: str) -> pd.DataFrame:
    paths = sorted((ROWS / part / kind).glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else \
        pd.DataFrame()


def _read_placement(blocks: list[str]) -> pd.DataFrame:
    """U's placement rows (`placement`) of `blocks`, each required."""
    paths = [ROWS / "placement" / f"{b}.parquet" for b in blocks]
    missing = [p.stem for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f"no placement rows for {missing[:5]} (leaky.py placement <id>)")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def _q(x: pd.Series, q: float) -> float:
    return float(x.quantile(q)) if len(x) else np.nan


def report(part: str) -> None:
    sizes = pd.read_csv(BLOCKS).set_index("block")
    blk = _read(part, "facts").set_index("block")
    homes = _read(part, "homes")
    blk["size"] = sizes["size"].reindex(blk.index)
    blk["control"] = sizes["control"].reindex(blk.index)
    blk["unknowns_x"] = blk.unknowns_W / blk.unknowns_T
    blk["time_x"] = blk.t_W / blk.t_T
    blk["P0_x"] = blk.P0_W / blk.P0_T
    # (7) the open homes' rank agreement with T, by block
    op_all = homes[~homes.walled & ~homes.nodem_T & ~homes.nodem_W]
    blk["spearman"] = op_all.groupby("block").apply(
        lambda h: float(stats.spearmanr(h.u_T, h.u_W).statistic)).reindex(blk.index)
    blk.reset_index().to_csv(ROWS / part / "summary_blocks.csv", index=False)
    homes = homes.merge(blk[["control"]].reset_index(), on="block")
    # (2) by situation: U's placement (`placement`), every block's
    place = _read_placement(list(blk.index))
    homes = homes.merge(place, on=["block", "home"], how="left", validate="one_to_one")
    w = homes[homes.walled]
    if w.pocket_U.isna().any() or (w.pocket_U.astype(bool) == w.noground_U.astype(bool)).any():
        raise RuntimeError("a walled-in home neither in a pocket nor without open ground, or both "
                           "(or a block without its placement rows)")
    homes["pocket_U"] = homes.pocket_U.astype(bool)
    homes["noground_U"] = homes.noground_U.astype(bool)
    out = []
    for name, sel in (("walled-in blocks", ~blk.control), ("controls", blk.control),
                      ("all", blk.control | ~blk.control)):
        g = blk[sel]
        hm = homes[homes.block.isin(g.index)]
        op = hm[~hm.walled & ~hm.nodem_T & ~hm.nodem_W]
        wh = hm[hm.walled]
        r = op.ratio
        row: dict[str, object] = dict(
            group=name, blocks=len(g), homes=len(hm), open_homes=len(op), walled_homes=len(wh),
            nodem_W=int(g.nodem_W.sum()),
            moved_pooled=float(np.mean(np.abs(r - 1.0) > MOVED)) if len(r) else np.nan,
            moved_block_median=float(g.moved.median()), moved_block_min=float(g.moved.min()),
            moved_block_max=float(g.moved.max()),
            ratio_median=_q(r, 0.5), ratio_p10=_q(r, 0.1), ratio_p90=_q(r, 0.9),
            ratio_up=float(np.mean(r > 1.0 + MOVED)) if len(r) else np.nan,
            ratio_down=float(np.mean(r < 1.0 - MOVED)) if len(r) else np.nan,
            walled_med_median=_q(wh.u_W_med, 0.5), walled_med_p90=_q(wh.u_W_med, 0.9),
            walled_med_max=float(wh.u_W_med.max()) if len(wh) else np.nan,
            walled_pocket=int(wh.pocket_U.sum()), walled_noground=int(wh.noground_U.sum()),
            walled_pocket_med_median=_q(wh.u_W_med[wh.pocket_U], 0.5),
            walled_pocket_med_p90=_q(wh.u_W_med[wh.pocket_U], 0.9),
            walled_noground_med_median=_q(wh.u_W_med[wh.noground_U], 0.5),
            walled_noground_med_p90=_q(wh.u_W_med[wh.noground_U], 0.9),
            walled_P0_share_median=float(g.walled_P0_share.median()),
            walled_P0_share_max=float(g.walled_P0_share.max()),
            walled_J2_share_median=float(g.walled_J2_share.median()),
            walled_J2_share_max=float(g.walled_J2_share.max()),
            own_open_median=_q(op.own_share, 0.5), own_open_p90=_q(op.own_share, 0.9),
            own_walled_median=_q(wh.own_share, 0.5), own_walled_p90=_q(wh.own_share, 0.9),
            through_median=float(g.through.median()),
            through_pooled=float(g.entering.sum() / g.injected.sum()),
            through_bld_median=float(g.through_bld.median()),
            through_bld_pooled=float(g.entering_bld.sum() / g.injected.sum()),
            diss_walls_median=float(g.diss_walls.median()),
            diss_interiors_median=float(g.diss_interiors.median()),
            diss_walls_out_median=float(g.diss_walls_out.median()),
            diss_walls_in_median=float(g.diss_walls_in.median()),
            diss_walls_party_median=float(g.diss_walls_party.median()),
            diss_walls_out_pooled=float((g.diss_walls_out * g.P0_W).sum() / g.P0_W.sum()),
            diss_walls_in_pooled=float((g.diss_walls_in * g.P0_W).sum() / g.P0_W.sum()),
            diss_walls_party_pooled=float((g.diss_walls_party * g.P0_W).sum() / g.P0_W.sum()),
            spearman_median=float(g.spearman.median()), spearman_p10=_q(g.spearman, 0.1),
            spearman_min=float(g.spearman.min()),
            P0_x_median=float(g.P0_x.median()), P0_x_p10=_q(g.P0_x, 0.1),
            P0_x_p90=_q(g.P0_x, 0.9),
            passes_median=float(np.median(np.concatenate([g.passes_pop, g.passes_q]))),
            energy_rel_max=float(g.energy_rel.abs().max()),
            unknowns_x_median=float(g.unknowns_x.median()),
            unknowns_x_max=float(g.unknowns_x.max()),
            time_x_median=float(g.time_x.median()), time_x_max=float(g.time_x.max()),
            t_T_sum=float(g.t_T.sum()), t_W_sum=float(g.t_W.sum()),
            passes_pop_max=int(g.passes_pop.max()), passes_q_max=int(g.passes_q.max()),
            wall_sum_h=float(g.wall.sum() / 3600.0),
            peak_rss_gb_max=float(g.peak_rss_gb.max()))
        out.append(row)
    t = pd.DataFrame(out)
    t.to_csv(ROWS / part / "summary_groups.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(4).T.to_string())


def main(argv: list[str]) -> None:
    argv, standing = clearings.standing_of(argv)
    if len(argv) == 3 and argv[0] == "block":
        block_run(argv[1], argv[2], standing=standing)
    elif len(argv) == 2 and argv[0] == "report":
        report(argv[1])
    elif len(argv) == 2 and argv[0] == "placement":
        placement(argv[1], standing=standing)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
