"""Crossings in Shock Reblock's networks, against an end-sliding null (crossings.py's runs).

Each built ear's path is the step's recorded candidate whose ends are its pick row's. Each end of
ear i joins the road its pixel's nearest bounding triangle belongs to at state i (crossings.py's
`road`: an earlier ear k, or the street). An end on an ear k is placed on k's centreline (its
path) at arc length s, on side +1 or -1 of it.

A crossing is two ends of different ears on one road k, on opposite sides, whose arc lengths
differ by at most TOL_M (half a road: the two arms' 7 m corridors overlap at least half their
width); TOL_WIDE_M (a whole road: the corridors touch) is reported beside it, and the offsets |ds|
of every opposite-side pair, nearest first.

The null slides every end along the road it joins within what its face allows: the end of ear i
on road k is redrawn uniformly among the sink pixels of ear i's face at state i whose road is k
(on the same side), each end independently, NULL_DRAWS times; the crossings counted the same way.
p = (1 + draws with at least the observed count) / (1 + NULL_DRAWS); a pooled test sums the
observed counts and the draws across runs.

H1/H2 per end (step 3 of the brief): its lag (i - k), its arm's share of path pixels in buildings,
the straight lane through its junction in the original fabric (LANE: the open run on each side of
k's corridor along k's normal at the junction), its place relative to its face's deepest pixel at
state i (projected onto k), and k's ridge (the pre-road potential along k's path at state k: the
junction's u percentile along it).

    A=research/roadless/atlas/crossings/analyze.py      # from the repo root, PYTHONPATH=.
    uv run python $A [--budget D] <run dir>...  # writes ends[_D<D>].parquet, pairs, halves per
                                                # run, prints the table
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import ears  # noqa: E402
import growth as gr  # noqa: E402

ROAD_W = gr.ROAD_W
TOL_M = ROAD_W / 2       # m: a crossing's arms at most this far apart along the road
TOL_WIDE_M = ROAD_W      # m: the wide tolerance, reported beside it
NULL_DRAWS = 20_000
NULL_SEED = 0
LANE_MAX_M = 20.0        # m: a lane's open run counted to this far past the road's edge
LANE_STEP_M = 0.5        # m: the lane's samples
TAN_M = 2.0              # m: the road's tangent from its centreline this far either way
PIX = ears.PIX


class Rasters(NamedTuple):
    shape: tuple[int, int]
    lo: NDArray[np.float64]
    xy: NDArray[np.float64]
    inside: NDArray[np.bool_]
    bld: NDArray[np.float64]
    pf: NDArray[np.int32]        # states x pixels
    sink: NDArray[np.bool_]
    road: NDArray[np.int16]
    u: NDArray[np.float32]

    def pix(self, xy: NDArray[np.float64]) -> NDArray[np.int64]:
        ij = np.floor((xy - self.lo) / PIX).astype(np.int64)
        ij[:, 0] = np.clip(ij[:, 0], 0, self.shape[0] - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, self.shape[1] - 1)
        return np.asarray(ij[:, 0] * self.shape[1] + ij[:, 1])


class Run(NamedTuple):
    dir: Path
    world: str
    plan: str
    ra: Rasters
    picks: pd.DataFrame
    paths: list[NDArray[np.float64]]       # per step the built ear's path (m, the raster's frame)
    bld_share: NDArray[np.float64]         # per step its path pixels' share in buildings


def load(d: Path, *, budget: float | None) -> Run:
    """The run in `d`; with `budget`, its prefix a run to that D would have built (the ears before
    the first state at D >= budget: greedy, so the same ears)."""
    s = np.load(d / "states.npz")
    shape = (int(s["shape"][0]), int(s["shape"][1]))
    lo = s["lo"]
    i, j = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    xy = np.column_stack([lo[0] + (i.ravel() + 0.5) * PIX, lo[1] + (j.ravel() + 0.5) * PIX])
    ra = Rasters(shape=shape, lo=lo, xy=xy, inside=s["tri"] >= 0, bld=s["bld"], pf=s["pf"],
                 sink=s["sink"], road=s["road"], u=s["u"])
    picks = pd.read_parquet(d / "picks.parquet")
    if budget is not None:
        D = pd.read_parquet(d / "steps.parquet").D.to_numpy()
        reach = np.flatnonzero(D >= budget)
        if not len(reach):
            raise ValueError(f"{d}: never reached D {budget}")
        picks = picks.iloc[:int(reach[0])]
    cands = pd.read_parquet(d / "cands.parquet")
    paths, share = [], []
    for r in picks.itertuples():
        c = cands[(cands.step == r.step)]
        hit = [q for q in c.itertuples()
               if (q.path_x[0], q.path_y[0], q.path_x[-1], q.path_y[-1]) == (r.ax, r.ay, r.bx, r.by)]
        if len(hit) != 1:
            raise ValueError(f"{d}: step {r.step}: {len(hit)} candidates match its pick")
        q = hit[0]
        paths.append(np.column_stack([q.path_x, q.path_y]))
        share.append(float(np.mean(np.asarray(q.bld) > 0)))
    w = json.loads((d / "world.json").read_text())
    return Run(dir=d, world=w["world"], plan=w["plan"], ra=ra, picks=picks, paths=paths,
               bld_share=np.asarray(share))


class Placed(NamedTuple):
    s: NDArray[np.float64]       # arc length on the road's centreline
    side: NDArray[np.int64]      # +1 / -1
    d: NDArray[np.float64]       # distance from it


def place(line: LineString, xy: NDArray[np.float64]) -> Placed:
    pts = shapely.points(xy[:, 0], xy[:, 1])
    s = np.asarray(shapely.line_locate_point(line, pts))
    foot = shapely.line_interpolate_point(line, s)
    a = shapely.line_interpolate_point(line, np.maximum(s - TAN_M, 0))
    b = shapely.line_interpolate_point(line, np.minimum(s + TAN_M, line.length))
    t = np.column_stack([shapely.get_x(b) - shapely.get_x(a), shapely.get_y(b) - shapely.get_y(a)])
    f = np.column_stack([shapely.get_x(foot), shapely.get_y(foot)])
    r = xy - f
    cross = t[:, 0] * r[:, 1] - t[:, 1] * r[:, 0]
    return Placed(s=s, side=np.where(cross >= 0, 1, -1), d=np.hypot(*r.T))


def normal_at(line: LineString, s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """(the centreline's point at s, its unit normal on side +1)."""
    a = line.interpolate(max(s - TAN_M, 0))
    b = line.interpolate(min(s + TAN_M, line.length))
    t = np.array([b.x - a.x, b.y - a.y])
    t /= np.hypot(*t)
    p = line.interpolate(s)
    return np.array([p.x, p.y]), np.array([-t[1], t[0]])


def lane(ra: Rasters, p: NDArray[np.float64], n: NDArray[np.float64]) -> float:
    """The open run (m) past the road's edge along n from p in the original fabric: to the first
    building or the block's edge, at most LANE_MAX_M."""
    r = np.arange(ROAD_W / 2, ROAD_W / 2 + LANE_MAX_M + 1e-9, LANE_STEP_M)
    k = ra.pix(p + r[:, None] * n)
    blocked = (ra.bld[k] > 0) | ~ra.inside[k]
    hit = np.flatnonzero(blocked)
    return float(r[hit[0]] - ROAD_W / 2) if len(hit) else LANE_MAX_M


def ends_of(run: Run) -> tuple[pd.DataFrame, list[NDArray[np.float64]], list[NDArray[np.int64]]]:
    """Per end of every ear: its ear, road, place on the road and the H1/H2 measures; and per end
    on an ear its null pool (arc lengths, sides) on that road."""
    ra = run.ra
    lines = [LineString(p) for p in run.paths]
    rows, pool_s, pool_side = [], [], []
    for i, path in enumerate(run.paths):
        face = int(run.picks.face.iloc[i])
        for which, xy in (("a", path[0]), ("b", path[-1])):
            k = int(ra.road[i][ra.pix(xy[None])[0]])
            row = dict(ear=i, end=which, road=k, x=xy[0], y=xy[1], lag=i - k if k >= 0 else -1,
                       bld_share=run.bld_share[i])
            if k >= 0:
                line = lines[k]
                pl = place(line, xy[None])
                s, side = float(pl.s[0]), int(pl.side[0])
                adm = np.flatnonzero(ra.sink[i] & (ra.pf[i] == face) & (ra.road[i] == k))
                pa = place(line, ra.xy[adm])
                keep = pa.side == side
                pool_s.append(pa.s[keep])
                pool_side.append(pa.side[keep])
                # the face's deepest pixel at state i, onto k
                infc = np.flatnonzero(ra.pf[i] == face)
                deep = infc[np.argmax(ra.u[i][infc])]
                sd = float(place(line, ra.xy[deep][None]).s[0])
                ps = pa.s[keep]
                deep_pct = (float((np.abs(ps - sd) <= abs(s - sd)).mean()) if len(ps)
                            else np.nan)
                # k's ridge: u at state k along k's path
                uk = ra.u[k][ra.pix(run.paths[k])]
                sk = place(line, run.paths[k]).s
                u_here = float(np.interp(s, sk, uk))
                p0, nrm = normal_at(line, s)
                row.update(s=s, side=side, d=float(pl.d[0]), road_len=line.length,
                           s_frac=s / line.length, pool_n=int(keep.sum()),
                           pool_lo=float(pa.s[keep].min()) if keep.any() else np.nan,
                           pool_hi=float(pa.s[keep].max()) if keep.any() else np.nan,
                           s_deep=sd, deep_pct=deep_pct, ridge_pct=float((uk <= u_here).mean()),
                           ridge_argmax_s=float(sk[np.argmax(uk)]),
                           lane_arm=lane(ra, p0, side * nrm), lane_far=lane(ra, p0, -side * nrm))
            rows.append(row)
    return pd.DataFrame(rows), pool_s, pool_side


class Pairs(NamedTuple):
    a: NDArray[np.int64]          # rows of the on-ear ends
    b: NDArray[np.int64]
    ds: NDArray[np.float64]       # observed |ds|


def opposite_pairs(on: pd.DataFrame) -> Pairs:
    a, b = [], []
    for k, grp in on.groupby("road"):
        idx = grp.index.to_numpy()
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                p, q = on.loc[idx[x]], on.loc[idx[y]]
                if p.ear != q.ear and p.side != q.side:
                    a.append(idx[x])
                    b.append(idx[y])
    a_, b_ = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    pos = {v: n for n, v in enumerate(on.index)}
    ia = np.array([pos[v] for v in a_], dtype=np.int64)
    ib = np.array([pos[v] for v in b_], dtype=np.int64)
    s = on.s.to_numpy()
    return Pairs(a=ia, b=ib, ds=np.abs(s[ia] - s[ib]) if len(ia) else np.zeros(0))


class Null(NamedTuple):
    counts: NDArray[np.int64]          # draws: crossings at TOL_M
    counts_wide: NDArray[np.int64]     # at TOL_WIDE_M


def null_of(pairs: Pairs, pools: list[NDArray[np.float64]], *, seed: int) -> Null:
    rng = np.random.default_rng(seed)
    S = np.stack([p[rng.integers(len(p), size=NULL_DRAWS)] if len(p) else
                  np.full(NULL_DRAWS, np.nan) for p in pools], axis=1)
    if not len(pairs.a):
        z = np.zeros(NULL_DRAWS, dtype=np.int64)
        return Null(counts=z, counts_wide=z)
    ds = np.abs(S[:, pairs.a] - S[:, pairs.b])
    return Null(counts=(ds <= TOL_M).sum(axis=1), counts_wide=(ds <= TOL_WIDE_M).sum(axis=1))


def halves_of(run: Run) -> pd.DataFrame:
    """Per road k, at the state after it: the faces on its two sides (the two halves of the face
    it split; a side's largest face when it already borders several), each's deepest pixel
    placed on k (s_plus, s_minus), and the share of independent uniform draws from the two
    sides' sink pixels on k at least as close as the two deepest spots (pct: small where the
    halves' deepest spots sit across from one stretch of k)."""
    ra = run.ra
    rows = []
    rng = np.random.default_rng(NULL_SEED)
    for k, path in enumerate(run.paths):
        i = k + 1
        if i >= len(ra.pf):
            break
        line = LineString(path)
        on_k = np.flatnonzero(ra.sink[i] & (ra.road[i] == k) & (ra.pf[i] >= 0))
        if not len(on_k):
            continue
        pl = place(line, ra.xy[on_k])
        row: dict[str, float] = dict(road=k, road_len=line.length)
        pools = {}
        for side, tag in ((1, "plus"), (-1, "minus")):
            m = pl.side == side
            if not m.any():
                break
            labs, cnt = np.unique(ra.pf[i][on_k[m]], return_counts=True)
            f = int(labs[np.argmax(cnt)])
            infc = np.flatnonzero(ra.pf[i] == f)
            deep = infc[np.argmax(ra.u[i][infc])]
            row[f"s_{tag}"] = float(place(line, ra.xy[deep][None]).s[0])
            row[f"u_{tag}"] = float(ra.u[i][deep])
            pools[tag] = pl.s[m & (ra.pf[i][on_k] == f)]
        else:
            ds = abs(row["s_plus"] - row["s_minus"])
            a = pools["plus"][rng.integers(len(pools["plus"]), size=4000)]
            b = pools["minus"][rng.integers(len(pools["minus"]), size=4000)]
            row.update(ds_deep=ds, pct=float((np.abs(a - b) <= ds).mean()))
            rows.append(row)
    return pd.DataFrame(rows)


def analyze(d: Path, *, budget: float | None) -> tuple[dict[str, object], Null]:
    run = load(d, budget=budget)
    tag = "" if budget is None else f"_D{budget:g}"
    halves_of(run).to_parquet(d / f"halves{tag}.parquet")
    ends, pool_s, _ = ends_of(run)
    on = ends[ends.road >= 0].copy()
    pairs = opposite_pairs(on)
    null = null_of(pairs, pool_s, seed=NULL_SEED)
    obs = int((pairs.ds <= TOL_M).sum())
    obs_w = int((pairs.ds <= TOL_WIDE_M).sum())
    cross = np.zeros(len(on), dtype=bool)
    cross[pairs.a[pairs.ds <= TOL_M]] = True
    cross[pairs.b[pairs.ds <= TOL_M]] = True
    on["crossing"] = cross
    # each crossing pair's partner and the halves' deepest spots
    on["partner"] = -1
    for x, y in zip(pairs.a[pairs.ds <= TOL_M], pairs.b[pairs.ds <= TOL_M], strict=True):
        on.iloc[x, on.columns.get_loc("partner")] = int(on.iloc[y].ear)
        on.iloc[y, on.columns.get_loc("partner")] = int(on.iloc[x].ear)
    ends = ends.merge(on[["ear", "end", "crossing", "partner"]], on=["ear", "end"], how="left")
    ends["crossing"] = ends.crossing.fillna(False).astype(bool)
    ends.to_parquet(d / f"ends{tag}.parquet")
    pd.DataFrame(dict(a_ear=on.ear.to_numpy()[pairs.a], b_ear=on.ear.to_numpy()[pairs.b],
                      road=on.road.to_numpy()[pairs.a], ds=pairs.ds)).to_parquet(d / f"pairs{tag}.parquet")
    row = dict(world=run.world, plan=run.plan, budget=budget, ears=len(run.paths), ends_on_ears=len(on),
               opposite_pairs=len(pairs.a), obs=obs, null_mean=float(null.counts.mean()),
               p=float((1 + (null.counts >= obs).sum()) / (1 + NULL_DRAWS)),
               obs_wide=obs_w, null_mean_wide=float(null.counts_wide.mean()),
               p_wide=float((1 + (null.counts_wide >= obs_w).sum()) / (1 + NULL_DRAWS)),
               min_ds=float(pairs.ds.min()) if len(pairs.ds) else np.nan)
    return row, null


def pooled(nulls: list[Null], rows: list[dict[str, object]], *, wide: bool) -> dict[str, float]:
    key = "obs_wide" if wide else "obs"
    obs = int(sum(int(r[key]) for r in rows))  # type: ignore[call-overload]
    tot = np.sum([n.counts_wide if wide else n.counts for n in nulls], axis=0)
    return dict(obs=obs, null_mean=float(tot.mean()),
                p=float((1 + (tot >= obs).sum()) / (1 + NULL_DRAWS)))


def main(argv: list[str]) -> None:
    budget = None
    if argv[:1] == ["--budget"]:
        budget, argv = float(argv[1]), argv[2:]
    if not argv:
        raise SystemExit(__doc__)
    rows, nulls = [], []
    for a in argv:
        r, n = analyze(Path(a), budget=budget)
        rows.append(r)
        nulls.append(n)
    t = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.drop(columns="plan").to_string(index=False))
    print("pooled", pooled(nulls, rows, wide=False), "wide", pooled(nulls, rows, wide=True))


if __name__ == "__main__":
    main(sys.argv[1:])
