"""Crossings in Shock Reblock's networks, against an end-sliding null (crossings.py's runs).

Each built ear's path is the step's recorded candidate whose ends are its pick row's. Each end of
ear i joins the road its pixel's nearest bounding triangle belongs to at state i (crossings.py's
`road`: an earlier ear k, or the street). An end on an ear k is placed on k's centreline (its
path) at arc length s, on side +1 or -1 of it.

A crossing is two ends of different ears on one road k, on opposite sides, whose feet on k's
centreline (the junctions) lie at most TOL_M apart (half a road: the two arms' 7 m corridors
overlap at least half their width); wider tolerances (TOLS: a whole road, the corridors touching;
then jogged T's) are reported beside it, and every opposite-side pair's offset (pairs.parquet).

Two nulls (Pools). The uniform null slides every end along the road it joins within what its face allows: the end of ear i
on road k is redrawn uniformly among the sink pixels of ear i's face at state i whose road is k
(on the same side), each end independently, NULL_DRAWS times; the crossings counted the same way.
p = (1 + draws with at least the observed count) / (1 + NULL_DRAWS); a pooled test sums the
observed counts and the draws across runs. The lattice null ("l" columns) slides a source end (an
`a` end: where Paths.ears' search started) only among that step's sources (sources_of): the starts
sit one per SRC_M square of a grid every face shares, so starts on the two sides of a road line up.

H1/H2 per end (step 3 of the brief): its lag (i - k), its arm's share of path pixels in buildings,
the straight lane through its junction in the original fabric (LANE: the open run on each side of
k's corridor along k's normal at the junction), its place relative to its face's deepest pixel at
state i (projected onto k), and k's ridge (the pre-road potential along k's path at state k: the
junction's u percentile along it).

    A=research/roadless/atlas/crossings/analyze.py      # from the repo root, PYTHONPATH=.
    uv run python $A [--budget D] [--out <csv>] <run dir>...  # writes ends[_D<D>].parquet, pairs, halves per
                                                # run, prints the table
"""
from __future__ import annotations

import json
import re
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
TOLS = (TOL_M, ROAD_W, 10.0, 15.0, 20.0)   # m: the tolerances reported (TOL_M the definition;
                                           # ROAD_W the corridors touching; wider: jogged T's)
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
        if len(reach):          # else it stopped short of the budget: every ear
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
    foot: NDArray[np.float64]    # the centreline's point at s (n x 2)


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
    return Placed(s=s, side=np.where(cross >= 0, 1, -1), d=np.hypot(*r.T), foot=f)


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


def sources_of(ra: Rasters, i: int, *, face_m2: float) -> NDArray[np.bool_]:
    """Per pixel whether it is one of Paths.ears' sources at state i: the first sink pixel (by
    index) of each SRC_M square among the sink pixels of faces of face_m2 and more (a face's area
    its pixels', a close stand-in for its triangles')."""
    pf = ra.pf[i]
    area = np.bincount(np.maximum(pf, 0), weights=(pf >= 0).astype(float)) * PIX * PIX
    live = (pf >= 0) & (area[np.maximum(pf, 0)] >= face_m2)
    bnd = np.flatnonzero(live & ra.sink[i])
    key = np.floor(ra.xy[bnd] / ears.SRC_M).astype(np.int64)
    _, first = np.unique(key[:, 0] * 1_000_000 + key[:, 1], return_index=True)
    out = np.zeros(len(pf), dtype=bool)
    out[bnd[first]] = True
    return out


class Pools(NamedTuple):
    """Per on-ear end its admissible junctions' feet on its road (n x 2): `uniform` every sink
    pixel of its face on the road; `lattice` the same, but a source end (an `a` end) among the
    step's sources only (sources_of: one per SRC_M square on a grid shared by every face)."""
    uniform: list[NDArray[np.float64]]
    lattice: list[NDArray[np.float64]]


def ends_of(run: Run) -> tuple[pd.DataFrame, Pools]:
    """Per end of every ear: its ear, road, place on the road and the H1/H2 measures; and per end
    on an ear its null pools on that road."""
    ra = run.ra
    lines = [LineString(p) for p in run.paths]
    face_m2 = float(re.search(r"\.f([0-9.]+)\.", run.plan)[1])  # type: ignore[index]
    rows: list[dict[str, object]] = []
    pools = Pools(uniform=[], lattice=[])
    for i, path in enumerate(run.paths):
        face = int(run.picks.face.iloc[i])
        src = sources_of(ra, i, face_m2=face_m2)
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
                pools.uniform.append(pa.foot[keep])
                lat = keep & src[adm] if which == "a" else keep
                pools.lattice.append(pa.foot[lat])
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
                row.update(s=s, fx=float(pl.foot[0, 0]), fy=float(pl.foot[0, 1]), side=side, d=float(pl.d[0]), road_len=line.length,
                           s_frac=s / line.length, pool_n=int(keep.sum()),
                           pool_lattice_n=int(lat.sum()),
                           on_lattice=bool(src[ra.pix(xy[None])[0]]),
                           pool_lo=float(pa.s[keep].min()) if keep.any() else np.nan,
                           pool_hi=float(pa.s[keep].max()) if keep.any() else np.nan,
                           s_deep=sd, deep_pct=deep_pct, ridge_pct=float((uk <= u_here).mean()),
                           ridge_argmax_s=float(sk[np.argmax(uk)]),
                           lane_arm=lane(ra, p0, side * nrm), lane_far=lane(ra, p0, -side * nrm))
            rows.append(row)
    return pd.DataFrame(rows), pools


class Pairs(NamedTuple):
    a: NDArray[np.int64]          # rows of the on-ear ends
    b: NDArray[np.int64]
    ds: NDArray[np.float64]       # observed: the distance between their junctions (feet on k)


def opposite_pairs(on: pd.DataFrame) -> Pairs:
    if on.empty:
        z = np.zeros(0, dtype=np.int64)
        return Pairs(a=z, b=z, ds=np.zeros(0))
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
    f = on[["fx", "fy"]].to_numpy()
    return Pairs(a=ia, b=ib, ds=np.hypot(*(f[ia] - f[ib]).T) if len(ia) else np.zeros(0))


class Null(NamedTuple):
    counts: NDArray[np.int64]          # draws x TOLS: crossings at each tolerance


def null_of(pairs: Pairs, pools: list[NDArray[np.float64]], *, seed: int) -> Null:
    """`pools`: per on-ear end its admissible junctions' feet on its road (n x 2)."""
    rng = np.random.default_rng(seed)
    if not len(pairs.a):
        return Null(counts=np.zeros((NULL_DRAWS, len(TOLS)), dtype=np.int64))
    S = np.stack([p[rng.integers(len(p), size=NULL_DRAWS)] for p in pools], axis=1)
    ds = np.hypot(*np.moveaxis(S[:, pairs.a] - S[:, pairs.b], -1, 0))
    return Null(counts=np.stack([(ds <= t).sum(axis=1) for t in TOLS], axis=1))


def test(obs: int, draws: NDArray[np.int64]) -> tuple[float, float]:
    """(the null mean, the one-sided p of at least `obs`)."""
    return float(draws.mean()), float((1 + (draws >= obs).sum()) / (1 + len(draws)))


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


NULLS = ("uniform", "lattice")      # Pools' fields: the null's column prefix "" and "l"


def analyze(d: Path, *, budget: float | None) -> tuple[dict[str, object], dict[str, Null]]:
    run = load(d, budget=budget)
    tag = "" if budget is None else f"_D{budget:g}"
    halves_of(run).to_parquet(d / f"halves{tag}.parquet")
    ends, pools = ends_of(run)
    on = ends[ends.road >= 0].copy()
    pairs = opposite_pairs(on)
    nulls = {"uniform": null_of(pairs, pools.uniform, seed=NULL_SEED),
             "lattice": null_of(pairs, pools.lattice, seed=NULL_SEED)}
    cross = np.zeros(len(on), dtype=bool)
    cross[pairs.a[pairs.ds <= TOL_M]] = True
    cross[pairs.b[pairs.ds <= TOL_M]] = True
    on["crossing"] = cross
    on["partner"] = -1
    for x, y in zip(pairs.a[pairs.ds <= TOL_M], pairs.b[pairs.ds <= TOL_M], strict=True):
        on.iloc[x, on.columns.get_loc("partner")] = int(on.iloc[y].ear)
        on.iloc[y, on.columns.get_loc("partner")] = int(on.iloc[x].ear)
    ends = ends.merge(on[["ear", "end", "crossing", "partner"]], on=["ear", "end"], how="left")
    ends["crossing"] = ends.crossing.fillna(False).astype(bool)
    ends.to_parquet(d / f"ends{tag}.parquet")
    pd.DataFrame(dict(a_ear=on.ear.to_numpy()[pairs.a], b_ear=on.ear.to_numpy()[pairs.b],
                      a_end=on.end.to_numpy()[pairs.a], b_end=on.end.to_numpy()[pairs.b],
                      road=on.road.to_numpy()[pairs.a], ds=pairs.ds)
                 ).to_parquet(d / f"pairs{tag}.parquet")
    row: dict[str, object] = dict(
        world=run.world, plan=run.plan, budget=budget, ears=len(run.paths), ends_on_ears=len(on),
        opposite_pairs=len(pairs.a), min_ds=float(pairs.ds.min()) if len(pairs.ds) else np.nan)
    for n, t in enumerate(TOLS):
        obs = int((pairs.ds <= t).sum())
        row[f"obs{t:g}"] = obs
        for kind, pre in zip(NULLS, ("", "l"), strict=True):
            mean, p = test(obs, nulls[kind].counts[:, n])
            row.update({f"{pre}null{t:g}": mean, f"{pre}p{t:g}": p})
    return row, nulls


def pooled(nulls: list[dict[str, Null]], rows: list[dict[str, object]]) -> dict[str, object]:
    """The runs' crossings summed against their nulls' draws summed (independent runs)."""
    out: dict[str, object] = dict(world="pooled", ears=sum(int(r["ears"]) for r in rows),  # type: ignore[call-overload]
                                  opposite_pairs=sum(int(r["opposite_pairs"]) for r in rows))  # type: ignore[call-overload]
    for n, t in enumerate(TOLS):
        obs = int(sum(int(r[f"obs{t:g}"]) for r in rows))  # type: ignore[call-overload]
        out[f"obs{t:g}"] = obs
        for kind, pre in zip(NULLS, ("", "l"), strict=True):
            tot = np.sum([x[kind].counts for x in nulls], axis=0)
            mean, p = test(obs, tot[:, n])
            out.update({f"{pre}null{t:g}": mean, f"{pre}p{t:g}": p})
    return out


def main(argv: list[str]) -> None:
    budget, out = None, None
    while argv[:1] in (["--budget"], ["--out"]):
        if argv[0] == "--budget":
            budget = float(argv[1])
        else:
            out = Path(argv[1])
        argv = argv[2:]
    if not argv:
        raise SystemExit(__doc__)
    rows, nulls = [], []
    for a in argv:
        r, n = analyze(Path(a), budget=budget)
        rows.append(r)
        nulls.append(n)
    t = pd.DataFrame(rows + [pooled(nulls, rows)])
    if out is not None:
        t.to_csv(out, index=False, float_format="%.4g")
    cols = ["world", "ears", "opposite_pairs"] + [
        f"{k}{x:g}" for x in TOLS[:2] for k in ("obs", "null", "p", "lnull", "lp")]
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print(t[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
