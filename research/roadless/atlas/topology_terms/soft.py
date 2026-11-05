"""Soft versions of topology's all-or-nothing frontage objective along each network's whole build
trajectory (owner 2026-11-05: "A soft objective lets us evaluate the trajectory rather than just
the end state"), on the face-off block (terms.py's networks, block and conventions).

States: topology's kept run in its build order (terms.topology_kept) and its six bisections; the
best sample (alpha 100 seed 0) and its bisections; the group Steiner floor and its bisections;
Shock Reblock by step (terms.shock: the paved corridor, its skeleton the centreline); the Loop
Network by canonical prefix (terms.cycle). The bisections' roads are read back from
rows/<part>.bisections.json (terms.bisected wrote each new edge).

Per state, over HOMES (buildings; each sits in exactly one parcel, the parcel that contains it, so
a home's access is its parcel's), the homes the state removes left out: a home is removed when the
paved corridor takes at least tau of its footprint (common.MEASURE's ramp at 1, so D counts it a
whole home), and D (area weights, common.MEASURE) charges for it already -- for our methods only
(REMOVES): topology's paths are lines that displace no one in its model. Measures:
  P@1        share of the kept homes whose parcel touches a street-connected road, under `line`
             (within STREET_TOL of a centreline: their convention) and `corridor` (within
             STREET_TOL of the paved corridor's edge: ours);
  depth      reblock.derive.access.parcel_access_layers per home, unreached one_past_deepest (as
             reblock does): mean, P@d = share at depth <= d (d = 1..4), access_burden (sum of
             depth^2 over kept homes), max (their k analogue: 1 is universal access);
  metres     reblock.derive.geometric_access.geometric_access_distances (tol STREET_TOL): mean and
             90th percentile;
  L_P        reblock.permeability.permeability, PermeabilityParams as shipped (conf/
             permeability.yaml) but g_walk in G_WALKS. Over PARCELS (the electrical mesh is the
             parcels'; a removed home's parcel stays a node). Every network as 7 m road
             centrelines (Shock Reblock's: its corridor's skeleton at 7 m).
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

from reblock.budget import access_burden  # noqa: E402
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
# The methods whose homes are dropped from the access measures: ours, which clear their corridor.
# Topology's family keeps every home: its paths are lines along parcel boundaries that displace
# no one in its own model (at 7 m the same rule would remove 559 of its 638 homes and leave the
# access measures to the 79 left; `removed_at_7m` records that count for every state).
REMOVES = ("shock", "cycle_native")
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
    return Block(b=b, home_parcel=home_parcel, w=nw.area_weights(b), adj=adj, egress=egress)


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
    """parcel_access_layers' BFS (unreached one_past_deepest) from the parcels within STREET_TOL of
    `seed`: the same peel, seeded by any geometry."""
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
    if (layer == 0).any():
        layer[layer == 0] = one_past_deepest(int(layer.max()), len(layer))
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


def measures(B: Block, net: T.Net, *, D_known: float | None, removes: bool,
             check: bool) -> dict[str, Any]:
    b = B.b
    c = common.MEASURE.c(b.buildings, net.paved if not shapely.is_empty(net.paved) else None)
    D = float((c * B.w).sum() / B.w.sum())
    kept = c < 1.0 if removes else np.ones(len(c), bool)
    lines = (with_width(gpd.GeoDataFrame(geometry=list(net.lines)), T.ROAD_W)
             if len(net.lines) else None)
    out: dict[str, Any] = dict(method=net.method, stage=net.stage, stop=net.stop,
                               new_m=float(shapely.length(net.lines).sum()) if len(net.lines)
                               else 0.0, D=D, D_known=D_known, removed=int((~kept).sum()),
                               removed_at_7m=int((c >= 1.0).sum()), homes=int(kept.sum()))
    seeds = {"line": seed_line(b, lines), "corridor": seed_corridor(b, net.paved)}
    for conv, seed in seeds.items():
        depth = peel_from(B, seed)
        if conv == "line" and check:                # the generic peel is reblock's on lines
            ref = parcel_access_layers(B.adj, lines, unreached=one_past_deepest).to_numpy()
            mref = geometric_access_distances(b, lines, tol=STREET_TOL).to_numpy()
            if not (np.array_equal(ref, depth)
                    and np.allclose(mref, metres_from(B, seed), rtol=1e-9, atol=1e-9)):
                raise AssertionError(f"{net.method} {net.stop}: the generic peel or metres "
                                     "differ from reblock's")
        dh = depth[B.home_parcel][kept].astype(float)
        dall = depth[B.home_parcel].astype(float)
        out[f"{conv}_P@1_all"] = float((dall <= 1).mean())      # removed homes kept in
        out[f"{conv}_depth_mean_all"] = float(dall.mean())
        out[f"{conv}_P@1"] = float((dh <= 1).mean())
        for d in DEPTHS[1:]:
            out[f"{conv}_P@{d}"] = float((dh <= d).mean())
        out[f"{conv}_depth_mean"] = float(dh.mean())
        out[f"{conv}_burden"] = access_burden(pd.Series(dh))
        out[f"{conv}_depth_max"] = float(dh.max())
        m = metres_from(B, seed)[B.home_parcel][kept]
        out[f"{conv}_m_mean"] = float(np.mean(m))
        out[f"{conv}_m_p90"] = float(np.percentile(m, 90))
        out[f"{conv}_m_unreached"] = int((~np.isfinite(m)).sum())
    for g, ctx in B.egress.items():
        out[f"L_P@{g:g}"] = float(permeability(ctx, lines))
    return out


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
    out = [measures(B, n, D_known=Dk.get(n.stop) if n.stage == "build" else None,
                    removes=part in REMOVES, check=k % CHECK_EVERY == 0 or k == len(nets) - 1)
           for k, n in enumerate(nets)]
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(out).to_parquet(OUT / f"{part}.parquet")
    print(f"{part}: {len(nets)} states ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ the report

NAMES = {"topology": "topology (kept, a10 s0)", "topology.a100.s0": "topology (best, a100 s0)",
         "steiner": "group Steiner floor", "shock": "Shock Reblock",
         "cycle_native": "Loop Network"}
COLOURS = {"topology": "#2a78d6", "topology.a100.s0": "#eb6834", "steiner": "#1baf7a",
           "shock": "#eda100", "cycle_native": "#e87ba4"}
# (column, higher is better)
MEASURES = (("line_P@1", True), ("corridor_P@1", True), ("line_P@2", True), ("line_P@3", True),
            ("line_P@4", True), ("line_depth_mean", False), ("corridor_depth_mean", False),
            ("line_burden", False), ("line_depth_max", False), ("corridor_depth_max", False),
            ("line_m_mean", False), ("line_m_p90", False), ("corridor_m_mean", False),
            ("corridor_m_p90", False), ("L_P@0.01", True), ("L_P@0.1", True), ("L_P@1", True),
            ("L_P@2", True))
TABLE = ("new_m", "D", "homes", "line_P@1", "corridor_P@1", "line_P@1_all", "corridor_P@1_all",
         "line_depth_mean", "corridor_depth_mean", "line_P@2", "line_P@3", "line_P@4",
         "line_burden", "line_depth_max", "corridor_depth_max", "line_m_mean", "line_m_p90",
         "corridor_m_mean", "corridor_m_p90", "L_P@0.01", "L_P@0.1", "L_P@1", "L_P@2")


def trajectories() -> dict[str, pd.DataFrame]:
    """Per method its states in build order (build stage by stop, then the bisections)."""
    df = pd.concat([pd.read_parquet(OUT / f"{p}.parquet") for p in PARTS], ignore_index=True)
    df["order"] = np.where(df.stage == "build", 0, 1)
    return {m: d.sort_values(["order", "stop"]).reset_index(drop=True)
            for m, d in df.groupby("method", sort=False)}


def at(d: pd.DataFrame, col: str, x: float) -> pd.Series | None:
    """The method's state at budget x on axis `col`: its last state with col <= x (None before
    its first)."""
    ok = d[d[col] <= x + 1e-9]
    return None if ok.empty else ok.iloc[-1]


def report(bid: str) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    tr = trajectories()
    lines_out: list[str] = []

    def say(x: str = "") -> None:
        lines_out.append(x)
        print(x, flush=True)

    say(f"{bid}: soft frontage measures along each trajectory (soft.py docstring)")
    say("homes = buildings, each in its own parcel; L_P over parcels; removed homes dropped from "
        "the access measures for Shock Reblock and the Loop Network only (REMOVES); *_all keep "
        "them")
    # D agreement with the build-out rows
    for m in ("topology", "shock"):
        d = tr[m].dropna(subset=["D_known"])
        say(f"D here vs the build-out rows ({m}): max |diff| {np.abs(d.D - d.D_known).max():.4f} "
            f"over {len(d)} stops")
    summary: dict[str, Any] = {}
    # 1. the Demo budgets
    say("\n== at the Demo budgets (shortest prefix with D >= target) ==")
    rows_: list[dict[str, Any]] = []
    for t in T.DEMO_D:
        for m in ("topology", "shock", "cycle_native"):
            d = tr[m]
            hit = d[(d.stage == "build") & (d.D >= t)]
            r = hit.iloc[0] if len(hit) else d[d.stage == "build"].iloc[-1]
            rows_.append(dict(target=t, method=NAMES[m], stop=int(r.stop),
                              **{c: r[c] for c in TABLE}))
    demo = pd.DataFrame(rows_)
    say(demo.round(3).T.to_string(header=False))
    summary["demo"] = rows_
    # 2. end states
    say("\n== end states ==")
    ends = []
    for m, d in tr.items():
        b_ = d[d.stage == "build"].iloc[-1]
        ends.append(dict(method=NAMES[m], state="build end", stop=int(b_.stop),
                         **{c: b_[c] for c in TABLE}))
        if (d.stage == "bisect").any():
            e = d.iloc[-1]
            ends.append(dict(method=NAMES[m], state=f"+{int(e.stop)} bisections",
                             stop=int(e.stop), **{c: e[c] for c in TABLE}))
    say(pd.DataFrame(ends).round(3).T.to_string(header=False))
    summary["ends"] = ends
    # 3. rankings at matched road length and at matched D
    say("\n== who is best, topology (kept) vs Shock Reblock vs the Loop Network, at matched "
        "budgets ==")
    cmp_ = ("topology", "shock", "cycle_native")
    for axis, grid in (("new_m", np.arange(100.0, 2201.0, 100.0)),
                       ("D", np.round(np.arange(0.02, 0.72, 0.02), 2))):
        say(f"-- axis {axis}: a method's last state at or below the budget; * where that state is "
            f"more than {'0.03' if axis == 'D' else '60 m'} below it (past its end, or a gap) --")
        table: dict[str, list[str]] = {}
        cross: dict[str, list[str]] = {}
        for col, hi in MEASURES:
            best, sign = [], []
            for x in grid:
                vals = {}
                for m in cmp_:
                    r = at(tr[m][tr[m].stage == "build"], axis, float(x))
                    if r is not None:
                        vals[m] = (float(r[col]), r[axis] < x - (0.03 if axis == "D" else 60))
                if len(vals) < 2:
                    best.append("-")
                    sign.append(0)
                    continue
                key = max if hi else min
                win = key(vals, key=lambda k: vals[k][0])
                tie = sum(abs(v[0] - vals[win][0]) < 1e-9 for v in vals.values()) > 1
                best.append("tie" if tie else ("T" if win == "topology" else
                                               "S" if win == "shock" else "L")
                            + ("*" if vals[win][1] else ""))
                if "topology" in vals and "shock" in vals:
                    dv = vals["topology"][0] - vals["shock"][0]
                    sign.append(int(np.sign(dv if hi else -dv)))
                else:
                    sign.append(0)
            table[col] = best
            cross[col] = [f"{grid[k]:g}" for k in range(1, len(grid))
                          if sign[k] and sign[k - 1] and sign[k] != sign[k - 1]]
        frame = pd.DataFrame(table, index=[f"{x:g}" for x in grid]).T
        say(frame.to_string())
        say("topology vs Shock Reblock: the better one changes at " + "; ".join(
            f"{c}: {', '.join(v) if v else 'never'}" for c, v in cross.items()))
        summary[f"rank_{axis}"] = dict(table=table, crossings=cross)
    # 4. monotonicity
    say("\n== non-monotone steps (a measure getting worse as a road is added) ==")
    mono = {}
    for m, d in tr.items():
        for col, hi in MEASURES:
            v = d[col].to_numpy(dtype=float)
            dv = np.diff(v) * (1 if hi else -1)
            bad = np.flatnonzero(dv < -1e-9)
            if len(bad):
                k = int(bad[np.argmin(dv[bad])])
                mono[f"{m}|{col}"] = dict(n=len(bad), steps=len(dv), worst=float(-dv[k]),
                                          at=f"{d.stage[k + 1]} {int(d.stop[k + 1])}")
                say(f"{NAMES[m]:<26}{col:<22}{len(bad):3d} of {len(dv):3d} steps; worst "
                    f"{-dv[k]:.4g} at {d.stage[k + 1]} {int(d.stop[k + 1])}")
    summary["nonmonotone"] = mono
    (HERE / "soft_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    (HERE / "soft_report.txt").write_text("\n".join(lines_out) + "\n")
    # 5. the plot
    panels = (("line_P@1", "P@1, line (homes on a centreline)"),
              ("corridor_P@1", "P@1, corridor (homes on the paved road)"),
              ("line_depth_mean", "mean peel depth, line"),
              ("line_depth_max", "max peel depth (their k), line"),
              ("line_m_mean", "mean metres to a road, line"),
              ("corridor_m_mean", "mean metres to a road, corridor"),
              ("L_P@0.1", "L_P, g_walk 0.1 (shipped)"),
              ("L_P@2", "L_P, g_walk 2.0 (their 10x walking)"))
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, (col, title) in zip(axes.ravel(), panels, strict=True):
        for m, d in tr.items():
            b_, s_ = d[d.stage == "build"], d[d.stage == "bisect"]
            col_ = COLOURS[m]
            if len(b_) > 1:
                ax.plot(b_.new_m, b_[col], color=col_, lw=2, label=NAMES[m])
            else:
                ax.plot(b_.new_m, b_[col], "o", color=col_, ms=7, label=NAMES[m])
            if len(s_):
                ax.plot(np.r_[b_.new_m.iloc[-1:], s_.new_m], np.r_[b_[col].iloc[-1:], s_[col]],
                        "--", color=col_, lw=1.5)
        ax.set_title(title, fontsize=9.5, color="#0b0b0b")
        ax.set_xlabel("new road (m)", fontsize=8.5, color="#52514e")
        ax.grid(color="#e6e5e0", lw=0.6)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle(f"{bid}: soft frontage measures along each build (dashed: their bisections). "
                 "Shock Reblock and the Loop Network: removed homes dropped from the access "
                 "panels", fontsize=10.5)
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
