"""A home is freed only when its own ground opens: free first by the previous test against the test
(owner 2026-10-21, "No, its own pocket must open"; NOTES, "A home is freed only when its own ground
opens"; the test in gap_complex's HOMES).

The previous test freed a walled-in home when any ground within ring_m of its outline reached the
street, a neighbour's cleared interior among it; the test frees it only when its own ground (the
open ground within ring_m at baseline) does, a groundless home (none) as before. The previous test
is the test with every home read as groundless, so it needs no code of its own: free first on the
complex with every home groundless (gap_complex.free_first), and the flow model's reach of each
home's whole ring (gap_flow.ring_of, as Fixed.of finds it), checked against each other home for
home as Model.free_first checks the test.

`block <id>`: on the block's RUNTIME model (on a core), free first alone at area D BUDGET under the
previous test and under the test; each clearing's walled-in homes under both tests; and the
uniform study's stored clearings (T's and U's SIMP answers at D 0.05, STORED) recounted under both.
Rows: ROWS/blocks/<id>.parquet (the block's) and ROWS/homes/<id>.parquet (its walled-in homes').

`pockets <id>`: how much of each freed home's own ground opens (a home is freed when ANY of it
reaches the street, a sliver of open ground within ring_m among it): on the complex free first runs
on (the block's RUNTIME fixed mesh's baseline complex and groundless homes; no flow model), per
walled-in home with own ground its own ground's area, its pockets (its own ground's components at
the filtration width), and at the test's free first's clearing and the stored T and U clearings
the share of its own ground's area reaching the street and whether its largest pocket does.
Rows: ROWS/pockets/<id>.parquet.

`stored <relax rows>`: the uniform study's T and U rows at D 0.05 under <relax rows> (its worktree's
relax_rows, git-ignored there) written to STORED: per block and arm its metric, plan, free stage
and final clearing (committed: what was recounted).

`report`: per block (summary_blocks.csv), per walled-in home (summary_homes.csv) and by size
(summary_sizes.csv): the homes and people each test's free first leaves walled in, its D and D per
unit of the population it frees (median, pooled, the blocks over 1, the worst), the homes the
previous test freed through a neighbour that the test leaves walled in at that clearing and how
many the test's free first frees by opening their own ground (and of those, by clearing the home
itself), and the stored clearings' walled-in homes under both tests; and of the homes each
clearing frees, how much of their own ground opens (summary_pockets.csv, from `pockets`). The
blocks: BLOCKS (the 121 with walled-in homes under RUNTIME: the uniform study's).

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/own_ground.py
"""
from __future__ import annotations

import socket
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_calib  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_flow as gf  # noqa: E402
import metrics  # noqa: E402
import relax  # noqa: E402

ROWS = HERE / "own_rows"
BLOCKS = ROWS / "blocks.csv"            # block, size, n, walled, walled_share (the 121)
STORED = ROWS / "stored_clearings.csv"  # block, arm, metric, plan, free_cleared, simp_cleared
BUDGET = relax.D_LENS                   # area D 0.05 (the uniform study's)
POP = common.PopulationDemand()
ARMS = ("T", "U")


def _ids(s: object) -> np.ndarray:
    """A stored clearing's buildings (space-separated; empty: none)."""
    return np.asarray([int(v) for v in str(s).split()], dtype=np.int64) if isinstance(s, str) \
        else np.zeros(0, dtype=np.int64)


def stored(root: Path) -> None:
    rows = []
    for path in sorted(root.glob(f"*/*/D{BUDGET:g}/*_p1.parquet")):
        r = pd.read_parquet(path).iloc[0]
        rows.append(dict(block=str(r["block"]), arm=str(r["arm"]), metric=str(r["along"]),
                         plan=str(r["plan"]),
                         free_cleared=" ".join(map(str, np.asarray(r["free_cleared"]).tolist())),
                         simp_cleared=" ".join(map(str, np.asarray(r["simp_cleared"]).tolist()))))
    d = pd.DataFrame(rows).sort_values(["block", "arm"])
    if d.duplicated(["block", "arm"]).any():
        raise ValueError(f"{root}: a block's arm stored twice")
    ROWS.mkdir(parents=True, exist_ok=True)
    d.to_csv(STORED, index=False)
    print(f"{len(d)} rows ({d.block.nunique()} blocks, arms {sorted(d.arm.unique())}) -> {STORED}",
          flush=True)


def pockets_run(bid: str, *, replace: bool) -> None:
    out = ROWS / "pockets" / f"{bid}.parquet"
    clearings.check_new(out, replace=replace)
    t0 = time.time()
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    fx = metric.fixed(b, cache=gcl.CACHE, carving=gcl.Carving.none(gc.geometry(b)))
    cut = metric.exits.barriers.block(b)          # the model's block (ComplexMetric.model)
    geo = gc.geometry(cut)
    pop = common.AreaPopulation().weights(np.asarray(cut.buildings.outlines))   # the model's
    cost = pop / pop.sum()
    n = len(pop)
    w, grow = metric.form.filtration_w, metric.form.grow
    base = fx.cx0.complex()
    own = gc.own_of(geo, base, fx.groundless)
    walled0 = gc.walled(gc.home_exits(gc.homes_own(geo, own, base), gc.exit_widths(base)), w)
    zero = np.zeros(n, dtype=bool)
    fr = gc.free_first(geo, metric.spec, w=w, grow=grow, budget=BUDGET, cost=cost, pop=pop,
                       held0=zero, cx0=base, groundless=fx.groundless)
    st = pd.read_csv(STORED, dtype={"free_cleared": str, "simp_cleared": str})
    st = st[st.block == bid].set_index("arm")
    xs = {"own": fr.held}
    for arm in ARMS:
        x = zero.copy()
        x[_ids(st.loc[arm, "simp_cleared"])] = True
        xs[arm] = x
    comp, _street = gc.components(base, w)
    R = own.tri.tocsr()
    homes = np.flatnonzero(walled0 & ~fx.groundless)
    reached = {}
    for name, x in xs.items():
        cx = gc.build(geo, metric.spec, ~x, carve=None, grow=grow)
        live = (gc.located(cx.tri_geom, own.xy) @ (gc.exit_widths(cx).width >= w)) > 0
        reached[name] = np.asarray(live).ravel()
    rows = []
    for h in homes:
        tris = R.indices[R.indptr[h]:R.indptr[h + 1]]
        a = base.area[tris]
        pk, inv = np.unique(comp[tris], return_inverse=True)
        pa = np.bincount(inv, weights=a)
        big = int(np.argmax(pa))
        row: dict[str, object] = dict(block=bid, home=int(h), share=float(cost[h]),
                                      own_area=float(a.sum()), pockets=len(pk),
                                      largest_share=float(pa[big] / a.sum()))
        for name, r in reached.items():
            on = r[tris]
            row[f"freed_{name}"] = bool(on.any())
            row[f"reached_{name}"] = float(a[on].sum() / a.sum())
            row[f"largest_{name}"] = bool(on[inv == big].any())
        rows.append(row)
    clearings.write_once(pd.DataFrame(rows), out, replace=replace)
    print(f"{bid}: {len(homes)} walled-in homes with own ground; {time.time() - t0:.0f}s",
          flush=True)


def _outs(bid: str) -> tuple[Path, Path]:
    return ROWS / "blocks" / f"{bid}.parquet", ROWS / "homes" / f"{bid}.parquet"


class Tests(NamedTuple):
    """A clearing's reach of the walled-in homes under both tests: the previous (`was`: each
    home's whole ring) and the test (`own`: Model.live_homes_of)."""
    was: np.ndarray
    own: np.ndarray


def block_run(bid: str, *, replace: bool) -> None:
    outs = _outs(bid)
    for o in outs:
        clearings.check_new(o, replace=replace)
    t0 = time.time()
    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    [b] = common.build_blocks([bid])
    m = metric.clearing(b, demand=POP)
    t_model = time.time() - t0
    n = m.n
    zero = np.zeros(n, dtype=bool)
    w = m.walled0
    if not w.any():
        raise ValueError(f"{bid}: no home walled in under RUNTIME")
    _ib, band_open = gc._open_ground(m.geo, np.ones(n, dtype=bool), m.spec.band_m, carve=None,
                                     grow=0.0)
    ring = gf.ring_of(m.geo, m.spec.ring_m, m.fx.fm.xy, m.fx.fm.tri, band_open)

    def tests(x: np.ndarray) -> Tests:
        st = m.factors(m.k_of(x))
        street = st.reach.street[st.sys.cells.cell]
        return Tests(was=np.asarray(ring.live(street)), own=m.live_homes_of(st))

    t1 = time.time()
    fr = gc.free_first(m.geo, m.spec, w=m.form.filtration_w, grow=m.form.grow, budget=BUDGET,
                       cost=m.cost, pop=m.pop, held0=zero, cx0=m.fx.cx0.complex(),
                       groundless=np.ones(n, dtype=bool))
    t_was = time.time() - t1
    at_was = tests(fr.held)
    if not np.array_equal(fr.left, w & ~at_was.was):
        raise RuntimeError(f"{bid}: the previous test's free first and the flow model part on "
                           f"homes {np.flatnonzero(fr.left ^ (w & ~at_was.was)).tolist()}")
    t1 = time.time()
    held = m.free_first(budget=BUDGET, held=zero)
    t_own = time.time() - t1
    at_own = tests(held)
    st = pd.read_csv(STORED, dtype={"free_cleared": str, "simp_cleared": str})
    st = st[st.block == bid].set_index("arm")
    xs: dict[str, np.ndarray] = {}
    for arm in ARMS:
        x = zero.copy()
        x[_ids(st.loc[arm, "simp_cleared"])] = True
        xs[arm] = x
    free_T = zero.copy()
    free_T[_ids(st.loc["T", "free_cleared"])] = True
    at = {arm: tests(x) for arm, x in xs.items()}
    cost, bare = m.cost, m.fx.groundless
    # freed by the previous test through a neighbour, the test leaving them walled in there
    via = w & at_was.was & ~at_was.own

    def share(mask: np.ndarray) -> float:
        return float(cost[mask].sum())

    row: dict[str, object] = dict(
        block=bid, n=n, n_walled=int(w.sum()), walled_share=share(w),
        n_groundless=int((w & bare).sum()), groundless_share=share(w & bare),
        nodem=int(m.nodem.sum()),
        D_was=share(fr.held), D_own=share(held), held_was=int(fr.held.sum()),
        held_own=int(held.sum()), crossings_was=len(fr.crossings), passes_was=fr.passes,
        misses_was=fr.misses,
        freed_was=int((w & at_was.was).sum()), freed_was_share=share(w & at_was.was),
        freed_own=int((w & at_own.own).sum()), freed_own_share=share(w & at_own.own),
        left_was=int((w & ~at_was.was).sum()), left_was_share=share(w & ~at_was.was),
        left_own=int((w & ~at_own.own).sum()), left_own_share=share(w & ~at_own.own),
        left_own_at_was=int((w & ~at_was.own).sum()), left_own_at_was_share=share(w & ~at_was.own),
        via=int(via.sum()), via_share=share(via),
        via_freed_own=int((via & at_own.own).sum()),
        via_self_own=int((via & at_own.own & held).sum()),
        self_was=int((w & at_was.was & fr.held).sum()),
        self_own=int((w & at_own.own & held).sum()),
        groundless_freed_was=int((w & bare & at_was.was).sum()),
        groundless_freed_own=int((w & bare & at_own.own).sum()),
        stored_free_alike=bool(np.array_equal(fr.held, free_T)),
        **{f"left_{arm}_{t}": int((w & ~getattr(at[arm], t)).sum())
           for arm in ARMS for t in Tests._fields},
        **{f"left_{arm}_{t}_share": share(w & ~getattr(at[arm], t))
           for arm in ARMS for t in Tests._fields},
        D_T=share(xs["T"]), D_U=share(xs["U"]),
        t_model=t_model, t_was=t_was, t_own=t_own, host=socket.gethostname(),
        peak_rss_gb=gap_calib.peak_gb(), metric=metric.name, wall=time.time() - t0)
    h = np.flatnonzero(w)
    homes = pd.DataFrame(dict(
        block=bid, home=h, share=cost[h], groundless=bare[h], nodem=m.nodem[h],
        freed_was=at_was.was[h], own_at_was=at_was.own[h], freed_own=at_own.own[h],
        was_at_own=at_own.was[h], held_was=fr.held[h], held_own=held[h],
        **{f"{arm}_{t}": getattr(at[arm], t)[h] for arm in ARMS for t in Tests._fields}))
    for o, df in zip(outs, (pd.DataFrame([row]), homes), strict=True):
        clearings.write_once(df, o, replace=replace)
    print(f"{bid}: {int(w.sum())} walled in ({row['n_groundless']} groundless); the previous "
          f"test's free first D {row['D_was']:.4f} leaves {row['left_was']}, frees {row['via']} "
          f"through a neighbour alone; the test's D {row['D_own']:.4f} leaves {row['left_own']}, "
          f"frees {row['via_freed_own']} of those ({row['via_self_own']} by clearing the home); "
          f"stored T leaves {row['left_T_was']} -> {row['left_T_own']}, U {row['left_U_was']} -> "
          f"{row['left_U_own']}; {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ the report

def _read(kind: str) -> pd.DataFrame:
    paths = sorted((ROWS / kind).glob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True) if paths else \
        pd.DataFrame()


def _per_freed(D: pd.Series, freed: pd.Series) -> pd.Series:
    """D per unit of the population freed, per block (nan where none is freed)."""
    return pd.Series(np.where(freed > 0, D / freed.where(freed > 0, 1.0), np.nan), index=D.index)


def report() -> None:
    sizes = pd.read_csv(BLOCKS).set_index("block")
    blk = _read("blocks").set_index("block")
    homes = _read("homes")
    missing = sorted(set(sizes.index) - set(blk.index))
    if missing:
        print(f"{len(missing)} blocks not run: {missing[:8]}", flush=True)
    blk["size"] = sizes["size"].reindex(blk.index)
    for t in ("was", "own"):
        blk[f"D_per_freed_{t}"] = _per_freed(blk[f"D_{t}"], blk[f"freed_{t}_share"])
    blk.reset_index().to_csv(ROWS / "summary_blocks.csv", index=False)
    homes = homes.merge(blk[["size"]].reset_index(), on="block")
    homes.to_csv(ROWS / "summary_homes.csv", index=False)
    out = []
    groups = [("all", blk)]
    groups += [(s, blk.loc[(blk["size"] == s).to_numpy()])
               for s in ("small", "large", "oversized")]
    for name, g in groups:
        if g.empty:
            continue
        r: dict[str, object] = dict(group=name, blocks=len(g), homes_walled=int(g.n_walled.sum()),
                                    share_walled=float(g.walled_share.sum()),
                                    groundless=int(g.n_groundless.sum()))
        for t in ("was", "own"):
            per = g[f"D_per_freed_{t}"]
            worst = per.idxmax() if per.notna().any() else ""
            r.update({
                f"left_{t}": int(g[f"left_{t}"].sum()),
                f"left_{t}_share": float(g[f"left_{t}_share"].sum()),
                f"left_{t}_blocks": int((g[f"left_{t}"] > 0).sum()),
                f"D_{t}_mean": float(g[f"D_{t}"].mean()),
                f"D_{t}_sum": float(g[f"D_{t}"].sum()),
                f"freed_{t}_sum": float(g[f"freed_{t}_share"].sum()),
                f"D_per_freed_{t}_median": float(per.median()),
                f"D_per_freed_{t}_pooled": (float(g[f"D_{t}"].sum() / g[f"freed_{t}_share"].sum())
                                            if g[f"freed_{t}_share"].sum() > 0 else np.nan),
                f"D_per_freed_{t}_over1": int((per > 1).sum()),
                f"D_per_freed_{t}_over2": int((per > 2).sum()),
                f"D_per_freed_{t}_blocks": int(per.notna().sum()),
                f"D_per_freed_{t}_max": float(per.max()),
                f"D_per_freed_{t}_worst": worst,
                f"self_{t}": int(g[f"self_{t}"].sum()),
                f"groundless_freed_{t}": int(g[f"groundless_freed_{t}"].sum())})
        r.update(left_own_at_was=int(g.left_own_at_was.sum()),
                 left_own_at_was_share=float(g.left_own_at_was_share.sum()),
                 via=int(g.via.sum()), via_share=float(g.via_share.sum()),
                 via_blocks=int((g.via > 0).sum()), via_freed_own=int(g.via_freed_own.sum()),
                 via_self_own=int(g.via_self_own.sum()),
                 changed_blocks=int((g.D_was != g.D_own).sum()),
                 stored_free_alike=int(g.stored_free_alike.sum()))
        for arm in ARMS:
            for t in ("was", "own"):
                r[f"left_{arm}_{t}"] = int(g[f"left_{arm}_{t}"].sum())
                r[f"left_{arm}_{t}_blocks"] = int((g[f"left_{arm}_{t}"] > 0).sum())
                r[f"left_{arm}_{t}_share"] = float(g[f"left_{arm}_{t}_share"].sum())
        out.append(r)
    table = pd.DataFrame(out)
    table.to_csv(ROWS / "summary_sizes.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.round(5).T.to_string())
    pk = _read("pockets")
    if pk.empty:
        return
    pk = pk.merge(blk[["size"]].reset_index(), on="block")
    rows = []
    for clearing in ("own", "T", "U"):
        for size in ("small", "large", "oversized", "all"):
            g = pk if size == "all" else pk.loc[(pk["size"] == size).to_numpy()]
            f = g[g[f"freed_{clearing}"]]
            rows.append(dict(
                clearing=clearing, size=size, homes=len(g), freed=len(f),
                reached_median=float(f[f"reached_{clearing}"].median()),
                reached_under_half=int((f[f"reached_{clearing}"] < 0.5).sum()),
                reached_under_tenth=int((f[f"reached_{clearing}"] < 0.1).sum()),
                largest_sealed=int((~f[f"largest_{clearing}"]).sum()),
                largest_sealed_blocks=int(f.loc[~f[f"largest_{clearing}"], "block"].nunique())))
    pt = pd.DataFrame(rows)
    pt.to_csv(ROWS / "summary_pockets.csv", index=False)
    print(pt.round(4).to_string())


def main(argv: list[str]) -> None:
    argv, replace = clearings.replace_flag(argv)
    if len(argv) == 2 and argv[0] == "block":
        block_run(argv[1], replace=replace)
    elif len(argv) == 2 and argv[0] == "pockets":
        pockets_run(argv[1], replace=replace)
    elif len(argv) == 2 and argv[0] == "stored":
        stored(Path(argv[1]))
    elif argv == ["report"]:
        report()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
