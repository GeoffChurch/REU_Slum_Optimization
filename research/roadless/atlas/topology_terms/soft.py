"""Soft versions of topology's all-or-nothing frontage objective along each network's whole build
trajectory (owner 2026-11-05: "A soft objective lets us evaluate the trajectory rather than just
the end state"), on the face-off block (terms.py's networks, block and conventions).

States: topology's kept run in its build order (terms.topology_kept) and its six bisections; the
best sample (alpha 100 seed 0) and its bisections; the group Steiner floor and its bisections;
Shock Reblock by step (terms.shock: the paved corridor, its skeleton the centreline); the Loop
Network by canonical prefix (terms.cycle). The bisections' roads are read back from
rows/<part>.bisections.json (terms.bisected wrote each new edge).

Per state, over HOMES (buildings; each sits in exactly one parcel, the parcel that contains it, so
a home's access is its parcel's), every home counted. A home is removed in part: c_i =
common.MEASURE.c (reblock.destroyed.RampedDestroyed, tau 0.10: c_i = min(1, X_i / tau), X_i the
share of its footprint under the paved corridor; D = sum c_i w_i / sum w_i, area weights). Home i
counts as served (at its network depth) with weight 1 - c_i and as unserved with weight c_i, at a
fixed unserved value that cannot move with the network: depth D0 = one_past_deepest of the block's
NO-ROAD peel, metres M0 = the block's largest no-road metres (the deepest home's distance with no
road at all). M0 and not each home's own no-road metres: that is 0 for a street-fronting home, so
destroying one would cost nothing. Removal rule (Removal, a strategy per part): our methods
`corridor` (c from their paved corridor); topology's family both `none` (c = 0, its own model:
lines that displace no one) and `corridor` (c from its 7 m corridor, as D charges). Measures:
  P@1        sum (1 - c_i) [home i's parcel touches a street-connected road] / N, under `line`
             (within STREET_TOL of a centreline: their convention) and `corridor` (within
             STREET_TOL of the street-connected paved area: ours);
  depth      reblock.derive.access.parcel_access_layers per home: P@d = sum (1 - c_i)
             [depth_i <= d] / N (d = 1..4); mean = sum ((1 - c_i) depth_i + c_i D0) / N;
             access_burden = sum ((1 - c_i) depth_i^2 + c_i D0^2) (the mixture's expectation);
             max over the homes with c_i < 1 (a soft max is D0 as soon as any home is touched, so
             it is the surviving homes' max: their k analogue, 1 is universal access);
  metres     reblock.derive.geometric_access.geometric_access_distances (tol STREET_TOL):
             (1 - c_i) m_i + c_i M0, its mean and 90th percentile;
  L_P        reblock.permeability.permeability, PermeabilityParams as shipped (conf/
             permeability.yaml) but g_walk in G_WALKS. Over PARCELS (the electrical mesh is the
             parcels'; a removed home's parcel stays a node), so independent of the removal rule.
             Every network as 7 m road centrelines (Shock Reblock's: its corridor's skeleton).
Depth and metres are given under both conventions: the peel and the metres are reblock's
(parcel_access_layers, geometric_access_distances) seeded from the street-connected centrelines
under `line`, and the same BFS and Dijkstra (peel_from, metres_from, checked equal to reblock's on
lines every CHECK_EVERY states) seeded from the street-connected paved area under `corridor`.

    S=research/roadless/atlas/topology_terms/soft.py    # from the repo root, PYTHONPATH=.
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
        PYTHONDONTWRITEBYTECODE=1 uv run python $S rows <block> <part>   # soft/<part>.parquet
    ... $S report <block>                                              # soft_report.txt, soft.png
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
import yaml
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))           # research/roadless: common, road_rescore
sys.path.insert(0, str(HERE.parent))               # atlas: block_site
import common  # noqa: E402
import road_rescore as rr  # noqa: E402
import terms as T  # noqa: E402
from block_site import Site  # noqa: E402

from reblock.derive.access import (  # noqa: E402
    STREET_TOL,
    ParcelAdjacency,
    one_past_deepest,
    parcel_access_layers,
    street_connectivity,
)
from reblock.derive.geometric_access import geometric_access_distances  # noqa: E402
from reblock.permeability import (  # noqa: E402
    EgressContext,
    PermeabilityParams,
    permeability,
    with_width,
)

OUT = HERE / "soft"
G_WALKS = (0.01, 0.1, 1.0, 2.0)   # 0.1 shipped; 2.0 = a 7 m road's lane (20) over their 10x
DEPTHS = (1, 2, 3, 4)
PARTS = ("topology", "samples", "steiner", "shock", "cycle_native")


class Removal(NamedTuple):
    """How much of each home a state removes (c_i in [0, 1])."""
    name: str
    c_of: Callable[[Any, Any], np.ndarray]          # (block, paved corridor) -> c


NONE = Removal("none", lambda b, paved: np.zeros(len(b.buildings.outlines)))
CORRIDOR = Removal("corridor", lambda b, paved: common.MEASURE.c(
    b.buildings, None if shapely.is_empty(paved) else paved))
# Topology's family under its own model (none) and under the 7 m corridor D charges it for.
REMOVALS = {"topology": (NONE, CORRIDOR), "samples": (NONE, CORRIDOR),
            "steiner": (NONE, CORRIDOR), "shock": (CORRIDOR,), "cycle_native": (CORRIDOR,)}
# Shock Reblock's paved corridor (networks.Grown's smoothed triangles) stops 0.91 -- 0.98 m short
# of the street on 41785, every part of every step: the corridor's edge at the street, one raster
# cell from it (terms.joined_to_street's allowance), joins it.
CORRIDOR_JOIN_M = 1.5
CHECK_EVERY = 5          # every this many states, the generic peel and metres checked on lines


def shipped_params() -> dict[str, float]:
    """conf/permeability.yaml's PermeabilityParams fields (the class has no defaults)."""
    raw = yaml.safe_load((common.REPO / "conf" / "permeability.yaml").read_text())["params"]
    return {k: float(v) for k, v in raw.items() if k != "_target_"}


class Block(NamedTuple):
    """The block's fixed parts for these measures."""
    b: Any
    home_parcel: np.ndarray       # per building, the parcel containing it
    w: np.ndarray                 # area weights per building (D's)
    adj: ParcelAdjacency
    egress: dict[float, EgressContext]
    d0: int                       # the unserved depth: one_past_deepest of the no-road peel
    m0: float                     # the unserved metres: the largest no-road metres


def block_of(b: Any) -> Block:
    parcels = np.array(b.parcels.geometry, dtype=object)
    bi, pi = shapely.STRtree(parcels).query(shapely.points(b.buildings.xy), predicate="within")
    if len(bi) != len(b.buildings.outlines) or len(set(pi.tolist())) != len(pi):
        raise AssertionError("buildings and parcels are not one to one")
    home_parcel = np.empty(len(bi), dtype=np.int64)
    home_parcel[bi] = pi
    sys.path.insert(0, str(T.HERE.parent / "roads5810"))
    import networks as nw  # noqa: PLC0415
    params = shipped_params()
    adj = ParcelAdjacency.of(b, STREET_TOL)
    egress = {g: EgressContext(adj, PermeabilityParams(**(params | {"g_walk": g})))
              for g in G_WALKS}
    B = Block(b=b, home_parcel=home_parcel, w=nw.area_weights(b), adj=adj, egress=egress, d0=0,
              m0=0.0)
    street = seed_line(b, None)
    return B._replace(d0=one_past_deepest(int(peel_from(B, street).max()), len(parcels)),
                      m0=float(metres_from(B, street).max()))


def seed_line(b: Any, roads: gpd.GeoDataFrame | None) -> Any:
    """The peel's own seed (street_connectivity): the street and the street-connected roads."""
    return street_connectivity(b.streets, roads, STREET_TOL).seed_geom


def seed_corridor(b: Any, paved: Any) -> Any:
    """The street and the parts of the paved area within CORRIDOR_JOIN_M of it (a part is one
    paved piece, its loops' holes included: its outline's inner rings are the same road)."""
    street = shapely.union_all(np.array(b.streets.geometry, dtype=object))
    if paved is None or shapely.is_empty(paved):
        return street
    parts = shapely.get_parts(paved)
    live = parts[shapely.dwithin(parts, street, CORRIDOR_JOIN_M)]
    return shapely.union_all(np.r_[np.array([street], dtype=object), live])


def peel_from(B: Block, seed: Any) -> np.ndarray:
    """parcel_access_layers' BFS from the parcels within STREET_TOL of `seed`: the same peel,
    seeded by any geometry (every parcel reached, as on 41785: its adjacency is connected)."""
    parcels = np.array(B.b.parcels.geometry, dtype=object)
    layer = np.zeros(len(parcels), dtype=np.int64)
    frontier = list(np.flatnonzero(shapely.dwithin(parcels, seed, STREET_TOL)))
    layer[frontier] = 1
    k = 1
    while frontier:
        k += 1
        nxt = sorted({j for i in frontier for j in B.adj.neighbours[i] if layer[j] == 0})
        layer[nxt] = k
        frontier = nxt
    if (layer == 0).any():               # the soft measures' D0 assumes every parcel is reached
        raise AssertionError(f"{int((layer == 0).sum())} parcels unreached by the peel")
    return layer


def metres_from(B: Block, seed: Any) -> np.ndarray:
    """geometric_access_distances' Dijkstra (representative points, adjacency weighted by their
    distance) from the parcels within STREET_TOL of `seed`."""
    parcels = np.array(B.b.parcels.geometry, dtype=object)
    pts = shapely.get_coordinates(shapely.point_on_surface(parcels))
    n = len(parcels)
    u = np.array([i for i, nb in enumerate(B.adj.neighbours) for j in nb if i < j], dtype=np.int64)
    v = np.array([j for i, nb in enumerate(B.adj.neighbours) for j in nb if i < j], dtype=np.int64)
    w = np.maximum(np.linalg.norm(pts[u] - pts[v], axis=1), 1e-12)
    g = sp.coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
    src = np.flatnonzero(shapely.dwithin(parcels, seed, STREET_TOL))
    if not len(src):
        return np.full(n, np.inf)
    return dijkstra(g, indices=src, min_only=True)


def measures(B: Block, net: T.Net, removals: tuple[Removal, ...], *, D_known: float | None,
             check: bool) -> list[dict[str, Any]]:
    """One row per removal rule."""
    b = B.b
    D = float((CORRIDOR.c_of(b, net.paved) * B.w).sum() / B.w.sum())
    lines = (with_width(gpd.GeoDataFrame(geometry=list(net.lines)), T.ROAD_W)
             if len(net.lines) else None)
    seeds = {"line": seed_line(b, lines), "corridor": seed_corridor(b, net.paved)}
    depth, metres = {}, {}
    for conv, seed in seeds.items():
        depth[conv] = peel_from(B, seed)[B.home_parcel].astype(float)
        metres[conv] = metres_from(B, seed)[B.home_parcel]
        if conv == "line" and check:                # the generic peel is reblock's on lines
            ref = parcel_access_layers(B.adj, lines, unreached=one_past_deepest).to_numpy()
            mref = geometric_access_distances(b, lines, tol=STREET_TOL).to_numpy()
            if not (np.array_equal(ref[B.home_parcel], depth[conv])
                    and np.allclose(mref[B.home_parcel], metres[conv], rtol=1e-9, atol=1e-9)):
                raise AssertionError(f"{net.method} {net.stop}: the generic peel or metres "
                                     "differ from reblock's")
    lp = {f"L_P@{g:g}": float(permeability(ctx, lines)) for g, ctx in B.egress.items()}
    rows_ = []
    for rem in removals:
        c = rem.c_of(b, net.paved)
        s_ = 1.0 - c
        n = len(c)
        out: dict[str, Any] = dict(method=net.method, removal=rem.name, stage=net.stage,
                                   stop=net.stop, new_m=float(shapely.length(net.lines).sum())
                                   if len(net.lines) else 0.0, D=D, D_known=D_known,
                                   homes_lost=float(c.sum()), whole_lost=int((c >= 1.0).sum()))
        for conv in seeds:
            d, m = depth[conv], metres[conv]
            for k in DEPTHS:
                out[f"{conv}_P@{k}"] = float((s_ * (d <= k)).sum() / n)
            out[f"{conv}_depth_mean"] = float((s_ * d + c * B.d0).sum() / n)
            out[f"{conv}_burden"] = float((s_ * d ** 2 + c * B.d0 ** 2).sum())
            out[f"{conv}_depth_max"] = float(d[c < 1.0].max())
            ms = s_ * m + c * B.m0
            out[f"{conv}_m_mean"] = float(ms.mean())
            out[f"{conv}_m_p90"] = float(np.percentile(ms, 90))
        rows_.append(out | lp)
    return rows_


def with_bisections(net: T.Net, part: str) -> list[T.Net]:
    """`net` then each of its bisections, read back from rows/<part>.bisections.json."""
    rows_ = [r for r in json.loads((T.ROWS / f"{part}.bisections.json").read_text())
             if r["method"] == net.method]
    out, cur = [net], net
    for r in sorted(rows_, key=lambda r: r["k"]):
        cur = T.with_lines(cur, np.asarray(shapely.from_wkt(r["wkt"]), dtype=object), stop=r["k"])
        out.append(cur)
    if len(out) != T.BISECTIONS + 1:
        raise AssertionError(f"{net.method}: {len(out) - 1} bisections stored")
    return out


def grid_D(site: Site, key: str) -> dict[int, float]:
    g = json.loads((site.kept / f"{key}_grid.json").read_text())
    rows_ = [r for r in g["rows"] if r["status"] != "stopped"] + [g["whole"]]
    return {int(r["stop"]): float(r["D"]) for r in rows_}


def nets_of(part: str, b: Any, site: Site) -> tuple[list[T.Net], dict[int, float]]:
    """The part's states and the D the build-out rows give them (by stop, build stage only)."""
    match part:
        case "topology":
            kept = T.topology_kept(site)
            return kept + with_bisections(kept[-1], part)[1:], grid_D(site, T.TOPOLOGY_GRID)
        case "samples":
            best = T.whole("topology.a100.s0", T.RUNS / "topology.a100.s0.parquet")
            return with_bisections(best, part), {}
        case "steiner":
            return with_bisections(T.whole("steiner", T.RUNS / "steiner.ordered.parquet"),
                                   part), {}
        case "shock":
            return T.shock(b, site), grid_D(site, T.SHOCK)
        case "cycle_native":
            return T.cycle(site), {}
    raise SystemExit(f"{part}: not one of {PARTS}")


def rows(bid: str, part: str) -> None:
    site = Site.of(bid)
    [b] = common.build_blocks([bid])
    B = block_of(b)
    t0 = time.time()
    nets, Dk = nets_of(part, b, site)
    out = [r for k, n in enumerate(nets)
           for r in measures(B, n, REMOVALS[part],
                             D_known=Dk.get(n.stop) if n.stage == "build" else None,
                             check=k % CHECK_EVERY == 0 or k == len(nets) - 1)]
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(out).to_parquet(OUT / f"{part}.parquet")
    print(f"{part}: {len(nets)} states, D0 {B.d0}, M0 {B.m0:.1f} m ({time.time() - t0:.0f}s)",
          flush=True)


# ------------------------------------------------------------------ the report

NAMES = {"topology": "topology (kept, a10 s0)", "topology.a100.s0": "topology (best, a100 s0)",
         "steiner": "group Steiner floor tree", "shock": "Shock Reblock",
         "cycle_native": "Loop Network"}
COLOURS = {"topology": "#2a78d6", "topology.a100.s0": "#eb6834", "steiner": "#1baf7a",
           "shock": "#eda100", "cycle_native": "#e87ba4"}
# the four compared at matched budgets: (method, removal) -> letter
CONTEST = {("topology", "none"): "T", ("topology", "corridor"): "t", ("shock", "corridor"): "S",
           ("cycle_native", "corridor"): "L"}
# (column, higher is better)
MEASURES = (("line_P@1", True), ("corridor_P@1", True), ("line_P@2", True), ("line_P@3", True),
            ("line_P@4", True), ("line_depth_mean", False), ("corridor_depth_mean", False),
            ("line_burden", False), ("line_depth_max", False), ("corridor_depth_max", False),
            ("line_m_mean", False), ("line_m_p90", False), ("corridor_m_mean", False),
            ("corridor_m_p90", False), ("L_P@0.01", True), ("L_P@0.1", True), ("L_P@1", True),
            ("L_P@2", True))
TABLE = ("new_m", "D", "homes_lost", "whole_lost", "line_P@1", "corridor_P@1", "line_P@2",
         "line_P@3", "line_P@4", "line_depth_mean", "corridor_depth_mean", "line_burden",
         "line_depth_max", "corridor_depth_max", "line_m_mean", "line_m_p90", "corridor_m_mean",
         "corridor_m_p90", "L_P@0.01", "L_P@0.1", "L_P@1", "L_P@2")
QUOTA = HERE / "runs" / "quota"


def name(key: tuple[str, str]) -> str:
    m, r = key
    return NAMES[m] + ("" if r == "none" or m in ("shock", "cycle_native") else ", c 7 m")


def trajectories() -> dict[tuple[str, str], pd.DataFrame]:
    """Per (method, removal) its states in build order (build stage by stop, then bisections)."""
    df = pd.concat([pd.read_parquet(OUT / f"{p}.parquet") for p in PARTS], ignore_index=True)
    df["order"] = np.where(df.stage == "build", 0, 1)
    return {k: d.sort_values(["order", "stop"]).reset_index(drop=True)
            for k, d in df.groupby(["method", "removal"], sort=False)}


def at(d: pd.DataFrame, col: str, x: float) -> pd.Series | None:
    """The method's state at budget x on axis `col`: its last state with col <= x (None before
    its first)."""
    ok = d[d[col] <= x + 1e-9]
    return None if ok.empty else ok.iloc[-1]


def quota_floor(n_parcels: int) -> pd.DataFrame | None:
    """runs/quota/q*.json with monotone envelopes: f(q) is nondecreasing in q, so a lower bound
    at q holds at every larger q (lb_env = running max upward) and a tree for q serves every
    smaller q (ub_env = running min downward)."""
    recs = [json.loads(p.read_text()) for p in sorted(QUOTA.glob("q*.json"))]
    if not recs:
        return None
    d = pd.DataFrame(recs).sort_values("q").reset_index(drop=True)
    d["P"] = d.q / n_parcels
    d["lb_env"] = np.maximum.accumulate(d.lb.to_numpy())
    d["ub_env"] = np.minimum.accumulate(d.ub.to_numpy()[::-1])[::-1]
    d["gap"] = (d.ub_env - d.lb_env) / d.ub_env.where(d.ub_env > 0)
    return d


def report(bid: str) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    tr = trajectories()
    lines_out: list[str] = []

    def say(x: str = "") -> None:
        lines_out.append(x)
        print(x, flush=True)

    [b] = common.build_blocks([bid])
    B = block_of(b)
    n_homes = len(B.home_parcel)
    say(f"{bid}: soft frontage measures along each trajectory (soft.py docstring)")
    say(f"{n_homes} homes, one per parcel, every one counted; home i served with weight 1 - c_i "
        f"(RampedDestroyed, tau 0.10), unserved with weight c_i at depth D0 {B.d0} / metres M0 "
        f"{B.m0:.1f} (the no-road block's); removal `none` = c 0, `corridor` = c of the paved "
        "corridor (topology's: 7 m)")
    for m in ("topology", "shock"):
        d = tr[(m, "corridor")].dropna(subset=["D_known"])
        say(f"D here vs the build-out rows ({m}): max |diff| {np.abs(d.D - d.D_known).max():.4f} "
            f"over {len(d)} stops")
    summary: dict[str, Any] = dict(D0=B.d0, M0=B.m0)
    say("\n== at the Demo budgets (shortest prefix with D >= target) ==")
    rows_: list[dict[str, Any]] = []
    for t in T.DEMO_D:
        for k in CONTEST:
            d = tr[k]
            hit = d[(d.stage == "build") & (d.D >= t)]
            r = hit.iloc[0] if len(hit) else d[d.stage == "build"].iloc[-1]
            rows_.append(dict(target=t, method=name(k), stop=int(r.stop),
                              **{c: r[c] for c in TABLE}))
    say(pd.DataFrame(rows_).round(3).T.to_string(header=False))
    summary["demo"] = rows_
    say("\n== end states ==")
    ends = []
    for k, d in tr.items():
        b_ = d[d.stage == "build"].iloc[-1]
        ends.append(dict(method=name(k), state="build end", stop=int(b_.stop),
                         **{c: b_[c] for c in TABLE}))
        if (d.stage == "bisect").any():
            e = d.iloc[-1]
            ends.append(dict(method=name(k), state=f"+{int(e.stop)} bisections",
                             stop=int(e.stop), **{c: e[c] for c in TABLE}))
    say(pd.DataFrame(ends).round(3).T.to_string(header=False))
    summary["ends"] = ends
    say("\n== who is best at matched budgets: T topology (kept) c 0, t the same c 7 m, S Shock "
        "Reblock, L the Loop Network ==")
    for axis, grid in (("new_m", np.arange(100.0, 2201.0, 100.0)),
                       ("D", np.round(np.arange(0.02, 0.72, 0.02), 2))):
        say(f"-- axis {axis}: a method's last state at or below the budget, absent past its end; "
            f"* where that state is more than {'0.03' if axis == 'D' else '60 m'} below it (a gap "
            "in its stored states) --")
        table: dict[str, list[str]] = {}
        cross: dict[str, dict[str, list[str]]] = {}
        for col, hi in MEASURES:
            best: list[str] = []
            sign: dict[str, list[int]] = {"T": [], "t": []}
            for x in grid:
                vals = {}
                for k, letter in CONTEST.items():
                    if axis == "D" and letter == "T":
                        continue            # c 0 topology has no D of its own model
                    built = tr[k][tr[k].stage == "build"]
                    if x > float(built[axis].iloc[-1]) + (0.01 if axis == "D" else 20):
                        continue            # past its end: soft measures fall with D, so a
                        #                     carried end state would favour stopping early
                    r = at(built, axis, float(x))
                    if r is not None:
                        vals[letter] = (float(r[col]), r[axis] < x - (0.03 if axis == "D" else 60))
                for tl in sign:
                    if tl in vals and "S" in vals:
                        dv = vals[tl][0] - vals["S"][0]
                        sign[tl].append(int(np.sign(dv if hi else -dv)))
                    else:
                        sign[tl].append(0)
                if len(vals) < 2:
                    best.append("-")
                    continue
                key = max if hi else min
                win = key(vals, key=lambda k: vals[k][0])
                ties = [k for k, v in vals.items() if abs(v[0] - vals[win][0]) < 1e-9]
                best.append("".join(sorted(ties)) + ("*" if vals[win][1] else ""))
            table[col] = best
            cross[col] = {tl: [f"{grid[i]:g}" for i in range(1, len(grid))
                               if sg[i] and sg[i - 1] and sg[i] != sg[i - 1]]
                          for tl, sg in sign.items() if any(sg)}
        say(pd.DataFrame(table, index=[f"{x:g}" for x in grid]).T.to_string())
        for tl in ("T", "t"):
            say(f"{tl} vs S, the better one changes at: " + "; ".join(
                f"{c}: {', '.join(v[tl]) if v[tl] else 'never'}" for c, v in cross.items()
                if tl in v))
        summary[f"rank_{axis}"] = dict(table=table, crossings=cross)
    say("\n== each build's best state per measure (soft measures with c > 0 peak inside the "
        "build) ==")
    opt = []
    for k, d in tr.items():
        built = d[d.stage == "build"]
        if len(built) < 3:
            continue
        for col, hi in MEASURES:
            v = built[col].to_numpy(dtype=float)
            i = int(np.argmax(v) if hi else np.argmin(v))
            opt.append(dict(method=name(k), measure=col, start=v[0], best=v[i], end=v[-1],
                            at_m=float(built.new_m.iloc[i]), at_D=float(built.D.iloc[i]),
                            at_stop=int(built.stop.iloc[i])))
    say(pd.DataFrame(opt).round(3).to_string(index=False))
    summary["optima"] = opt
    say("\n== non-monotone steps (a measure getting worse as a road is added) ==")
    mono = {}
    for k, d in tr.items():
        for col, hi in MEASURES:
            v = d[col].to_numpy(dtype=float)
            dv = np.diff(v) * (1 if hi else -1)
            bad = np.flatnonzero(dv < -1e-9)
            if len(bad):
                i = int(bad[np.argmin(dv[bad])])
                mono[f"{k[0]}|{k[1]}|{col}"] = dict(
                    n=len(bad), steps=len(dv), worst=float(-dv[i]),
                    at=f"{d.stage[i + 1]} {int(d.stop[i + 1])}",
                    every=[f"{d.stage[j + 1]} {int(d.stop[j + 1])}" for j in bad])
                say(f"{name(k):<34}{col:<22}{len(bad):3d} of {len(dv):3d} steps; worst "
                    f"{-dv[i]:.4g} at {d.stage[i + 1]} {int(d.stop[i + 1])}")
    if not mono:
        say("none")
    summary["nonmonotone"] = mono
    qf = quota_floor(len(b.parcels))
    if qf is not None:
        say("\n== the quota floor: least parcel-boundary road giving >= q parcels a corner on a "
            "road (quota.py; c = 0, topology's graph only: NOT a bound on Shock Reblock or the "
            "Loop Network) ==")
        cols = ["q", "P", "lb", "ub", "lb_env", "ub_env", "gap", "lp_lb", "lp_converged",
                "lp_rounds", "heur_ub", "milp_ub", "milp_bound", "proven", "lp_cap_s",
                "milp_cap_s"]
        served = []
        for q in qf.q:
            pq = QUOTA / f"q{int(q)}.parquet"
            if not pq.exists():
                served.append(np.nan)
                continue
            roads = with_width(gpd.read_parquet(pq), T.ROAD_W)
            served.append(int((peel_from(B, seed_line(b, roads if len(roads) else None))
                               == 1).sum()))
        qf["line_served"] = served          # the tree's parcels served under soft.py's `line`
        cols.append("line_served")
        say(qf[cols].round(4).to_string(index=False))
        tk = tr[("topology", "none")]
        tk = tk[tk.stage == "build"]
        above = []
        for _, r in qf.iterrows():
            hit = tk[tk["line_P@1"] >= r.P - 1e-9]
            if len(hit):
                above.append(dict(q=int(r.q), topology_m=float(hit.new_m.iloc[0]),
                                  floor_lb=float(r.lb_env), floor_ub=float(r.ub_env),
                                  over_lb=float(hit.new_m.iloc[0] / r.lb_env - 1)
                                  if r.lb_env > 0 else None))
        say("topology (kept) against the floor, first state reaching each q:")
        say(pd.DataFrame(above).round(3).to_string(index=False))
        summary["quota"] = qf[cols + ["P"]].to_dict("records")
        summary["topology_vs_quota"] = above
    (HERE / "soft_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    (HERE / "soft_report.txt").write_text("\n".join(lines_out) + "\n")
    panels = (("line_P@1", "P@1, line (homes on a centreline)"),
              ("corridor_P@1", "P@1, corridor (homes on the paved road)"),
              ("line_depth_mean", "mean peel depth, line"),
              ("line_depth_max", "max peel depth of surviving homes, line"),
              ("line_m_mean", "mean metres to a road, line"),
              ("corridor_m_mean", "mean metres to a road, corridor"),
              ("L_P@0.1", "L_P, g_walk 0.1 (shipped)"),
              ("L_P@2", "L_P, g_walk 2.0 (their 10x walking)"))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, (col, title) in zip(axes.ravel(), panels, strict=True):
        if col == "line_P@1" and qf is not None:
            ax.fill_betweenx(qf.P, qf.lb_env, qf.ub_env, color="#c8c7c1", alpha=0.6, lw=0,
                             label="quota floor, lower to upper bound (c 0, boundary graph)")
            ax.plot(qf.lb_env, qf.P, color="#52514e", lw=1)
        for (m, r), d in tr.items():
            b_, s_ = d[d.stage == "build"], d[d.stage == "bisect"]
            c_ = COLOURS[m]
            dotted = r == "corridor" and m not in ("shock", "cycle_native")
            kw = dict(color=c_, lw=1.2 if dotted else 2, ls=":" if dotted else "-",
                      label=name((m, r)))
            if len(b_) > 1:
                ax.plot(b_.new_m, b_[col], **kw)
            else:
                ax.plot(b_.new_m, b_[col], "o" if not dotted else "x", color=c_, ms=6,
                        label=name((m, r)))
            if len(s_) and not dotted:
                ax.plot(np.r_[b_.new_m.iloc[-1:], s_.new_m], np.r_[b_[col].iloc[-1:], s_[col]],
                        "--", color=c_, lw=1.5)
        ax.set_title(title, fontsize=9.5, color="#0b0b0b")
        ax.set_xlabel("new road (m)", fontsize=8.5, color="#52514e")
        ax.grid(color="#e6e5e0", lw=0.6)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=6.8, loc="upper left")
    fig.suptitle(f"{bid}: soft frontage measures along each build (every home counted, weight "
                 "1 - c_i; dashed: their bisections; dotted: topology's family with c from its "
                 "7 m corridor)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(HERE / "soft.png", dpi=120)
    say(str(HERE / "soft.png"))


if __name__ == "__main__":
    rr._check_single_threaded()
    match sys.argv[1:]:
        case ["rows", bid, part]:
            rows(bid, part)
        case ["report", bid]:
            report(bid)
        case _:
            raise SystemExit(__doc__)
