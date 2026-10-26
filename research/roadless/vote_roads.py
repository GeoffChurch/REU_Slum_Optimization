"""Roads from the vote (docs/superpowers/plans/2026-10-25-roads-from-the-vote.md, stage 1; owner
2026-10-25): clear some buildings, solve the walking flow under RUNTIME, and lay roads along the
ridges of its current, a vote for where people move. Current is conserved, so its ridges run as
tributaries into the street.

A run is a vote's source, its signal and its dynamics, every one a Strategy resolved at the CLI and
named in the run:
  source: `orig`, the block as it stands; or a clearing (CLEARINGS/<name>.csv, its buildings'
    indices), cleared for the vote: `U` (metrics.UNIFORM's `.f0` SIMP at D 0.05 under P, the
    walled-in study's), `Uq0.5` (the same under P_{1/2}: `clearing`).
  signal: `cur`, the current density per triangle on open ground (gap_maps.CellFlow.jt); `pow`,
    the power dissipated per m^2 there (the frozen pilot's, power_density).
  dynamics: `frozen`, one vote, its roads in vote order; `per<step>k` / `per<step>d`, the vote
    solved again each `step` of D with the roads built carved, the clearing kept (k) or dropped
    (d) in it.
The vote painted on the block's BlockRaster at RES; per level (QUANTILES, the highest first) the
skeleton of the cells at or above that quantile (betweenness' ridges), its pixel steps merged into
lines. A line's stretch inside the corridors already built is dropped; what is left (MIN_M long and
more) is joined at each end to the street (when its inscribed disk, the distance off the level
set, reaches it within JOIN_SLACK_M: the street band and the skeleton's pull-back) or to a built
road (an end within ROAD_W / 2 + JOIN_SLACK_M of one), by the shortest line, and kept if linked to
the street (road_simp.linked's rule, the built roads with it). Within a level the linked lines go
in by their mean vote, the highest first; when none left touches the network, the join decides the
highest-voted floating one: `jr` drops it, `jf<max>` traces it downstream along the current
(Traced) to the street or a road. Roads ROAD_W wide.

Scored (`score`): the network's prefixes cut at the area D targets (TARGETS) on one carved mesh
(own_mesh's method), per role: `dev`, the roads alone (the clearing a
device: k 0, D the corridors'); `real`, the clearing in the state too (Model.cleared_road_state)
and in D (its buildings and the corridors under common.MEASURE, area weights), with the clearing
alone as its first row. Lenses L_P, L_{P_1/2}, L_J2 (road_rescore._lens_row).

    S=research/roadless/vote_roads.py          # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S copy-clearing <name> <relax row parquet>   # a relax row's clearing, kept here
    uv run python $S clearing <block> <metric> <plan> <q> <cpu|gpu> <name>   # P_q SIMP, kept here
    uv run python $S build <block> <source> <signal> <dynamics> <join>
    uv run python $S score <block> <source> <signal> <dynamics> <join>
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Protocol, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_maps as gm  # noqa: E402
import metrics  # noqa: E402
import road_rescore as rr  # noqa: E402
import road_simp  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402
from reblock.derive.access import STREET_TOL  # noqa: E402
from reblock.destroyed import IncrementalDestroyed  # noqa: E402
from reblock.methods.betweenness.raster import BlockRaster  # noqa: E402
from reblock.methods.betweenness.ridges import _segments  # noqa: E402
from reblock.methods.betweenness.thinning import skeletonize  # noqa: E402

RES = 1.0                                     # m: the vote's raster (the betweenness sources')
QUANTILES = (0.98, 0.95, 0.90, 0.85, 0.80)    # the ridges' levels, the highest first (the pilot's)
ROAD_W = road_simp.ROAD_W                     # m: a road's width
MIN_M = 5.0                                   # m: a line's piece shorter than this is dropped
JOIN_SLACK_M = 2.0                            # m: the joins' reach past an end's inscribed disk
                                              # (the 1 m street band, a pixel's diagonal)
TRACE_STEP_M = 0.5                            # m: a downstream trace's step (Traced)
TARGETS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, 0.15)    # the area D cuts
LAST = 0.15                                   # the periodic build stops once D reaches it
ROWS = HERE / "vote_rows"
CLEARINGS = ROWS / "clearings"                # committed: each clearing's buildings


# ------------------------------------------------------------------ the Strategies

class Source(Protocol):
    """Where the vote is solved: the buildings it clears (per building, 1 cleared)."""

    @property
    def name(self) -> str: ...

    def k(self, n: int) -> NDArray[np.float64]: ...


class Original:
    name = "orig"

    def k(self, n: int) -> NDArray[np.float64]:
        return np.zeros(n)


@dataclass(frozen=True)
class Cleared:
    """A kept clearing (CLEARINGS/<name>.csv)."""
    clearing: str

    @property
    def name(self) -> str:
        return self.clearing

    def k(self, n: int) -> NDArray[np.float64]:
        df = pd.read_csv(CLEARINGS / f"{self.clearing}.csv")
        if int(df.n.iloc[0]) != n:
            raise ValueError(f"{self.clearing}: a clearing of {int(df.n.iloc[0])} buildings, the "
                             f"block has {n}")
        k = np.zeros(n)
        k[df.cleared.to_numpy()] = 1.0
        return k


def source_of(spec: str) -> Source:
    return Original() if spec == "orig" else Cleared(spec)


class Signal(Protocol):
    """The vote per triangle on the state's open ground (NaN elsewhere)."""

    @property
    def name(self) -> str: ...

    def of(self, st: gcl.State, fl: gm.CellFlow) -> NDArray[np.float64]: ...


class Current:
    name = "cur"

    def of(self, st: gcl.State, fl: gm.CellFlow) -> NDArray[np.float64]:
        return np.asarray(fl.jt, dtype=np.float64)


class Power:
    name = "pow"

    def of(self, st: gcl.State, fl: gm.CellFlow) -> NDArray[np.float64]:
        return power_density(st, fl)


def signal_of(spec: str) -> Signal:
    signals: dict[str, Signal] = {"cur": Current(), "pow": Power()}
    if spec not in signals:
        raise SystemExit(f"{spec}: a signal is cur or pow")
    return signals[spec]


@dataclass(frozen=True)
class Frozen:
    name = "frozen"


@dataclass(frozen=True)
class Periodic:
    """The vote solved again each `step` of D, the clearing kept in it or dropped."""
    step: float
    keep: bool

    @property
    def name(self) -> str:
        return f"per{self.step:g}{'k' if self.keep else 'd'}"


Dynamics = Frozen | Periodic


def dynamics_of(spec: str) -> Dynamics:
    if spec == "frozen":
        return Frozen()
    m = re.fullmatch(r"per([0-9.]+)([kd])", spec)
    if m is None:
        raise SystemExit(f"{spec}: dynamics frozen or per<step>k|d")
    return Periodic(step=float(m[1]), keep=m[2] == "k")


def run_name(source: Source, signal: Signal, dyn: Dynamics, join: Join) -> str:
    q = ",".join(f"{v:g}" for v in QUANTILES)
    return (f"vr.{source.name}.{signal.name}.{dyn.name}.{join.name}.q{q}.r{RES:g}.w{ROAD_W:g}"
            f".m{MIN_M:g}j{JOIN_SLACK_M:g}s{TRACE_STEP_M:g}")


def join_of(spec: str) -> Join:
    if spec == "jr":
        return Reach()
    m = re.fullmatch(r"jf([0-9.]+)", spec)
    if m is None:
        raise SystemExit(f"{spec}: a join is jr or jf<max m>")
    return Traced(max_m=float(m[1]))


# ------------------------------------------------------------------ the vote

def power_density(st: gcl.State, fl: gm.CellFlow) -> NDArray[np.float64]:
    """Per triangle the power dissipated over its area (the frozen pilot's): a passage's power
    Tp du^2 split between its two triangles as their resistances, a street face's to its
    triangle, a leak's over its cell's triangles by area."""
    c, r, fm = st.sys.cells, st.reach, st.fm
    u = fl.u
    pp = r.Tp * (u[c.ea] - u[c.eb]) ** 2
    pf = r.Gf * u[c.fc] ** 2
    pl = fl.leak * u
    ta = 0.5 * np.abs(gm._cross(fm.xy, fm.tri))
    A_, B_ = fm.pa[c.pe], fm.pb[c.pe]
    ka, kb = st.kt[A_], st.kt[B_]
    s = ka + kb
    sa = np.where(st.binds[c.pe] | (s <= 0), 0.5, kb / np.where(s > 0, s, 1.0))
    trip = (np.bincount(A_, weights=pp * sa, minlength=fm.nt)
            + np.bincount(B_, weights=pp * (1.0 - sa), minlength=fm.nt)
            + np.bincount(fm.fa[c.pf], weights=pf, minlength=fm.nt))
    carea = np.bincount(c.cell, weights=ta, minlength=c.n)
    trip += pl[c.cell] * ta / np.maximum(carea[c.cell], 1e-300)
    return np.asarray(trip / np.maximum(ta, 1e-300))


class Vote(NamedTuple):
    """The vote on the raster (NaN off the state's open ground and the paved triangles) and the
    current's direction there (`jx`, `jy`: its cell's mean current density)."""
    img: NDArray[np.float64]
    jx: NDArray[np.float64]
    jy: NDArray[np.float64]


def vote_image(m: gcl.Model, st: gcl.State, signal: Signal, raster: BlockRaster,
               paved: NDArray[np.bool_]) -> Vote:
    """The vote on the raster: the signal on the state's open ground off the paved triangles."""
    fl = gm.flow_of(m, st, m.pop)
    cell = st.sys.cells.cell
    on = st.reach.live[cell] & ~paved
    ny, nx = raster.shape
    ox, oy = m.geo.origin
    frame = gm.Frame(x0=raster.x0 - ox, y0=raster.y0 - oy, h=RES, ny=ny, nx=nx)

    def img(v: NDArray[np.float64]) -> NDArray[np.float64]:
        return gm.paint(m, np.where(on, v, np.nan), frame)
    return Vote(img=img(signal.of(st, fl)), jx=img(fl.j[cell, 0]), jy=img(fl.j[cell, 1]))


def _sample(img: NDArray[np.float64], raster: BlockRaster, g: Any) -> float:
    xy = shapely.get_coordinates(shapely.segmentize(g, RES))
    j = np.clip(np.rint((xy[:, 0] - raster.x0) / RES).astype(int), 0, img.shape[1] - 1)
    i = np.clip(np.rint((xy[:, 1] - raster.y0) / RES).astype(int), 0, img.shape[0] - 1)
    v = img[i, j]
    return float(np.nanmean(v)) if np.isfinite(v).any() else 0.0


class Line:
    """A candidate road: its geometry (block CRS) and its mean vote."""

    def __init__(self, geom: Any, vote: float) -> None:
        self.geom, self.vote = geom, vote


def level_lines(img: NDArray[np.float64], raster: BlockRaster, q: float,
                street: Any, built: list[Any]) -> list[Line]:
    """One level's ridge lines off the built corridors, joined to the street or a built road
    (module docstring), in no order."""
    free = raster.free & np.isfinite(img)
    vals = img[free]
    if vals.size == 0 or float(vals.max()) == float(vals.min()):
        return []
    mask = free & (img >= np.quantile(vals, q))
    skel = skeletonize(mask)
    segs = _segments(skel, raster)
    if not segs:
        return []
    lines = shapely.get_parts(shapely.line_merge(shapely.MultiLineString(segs)))
    corr = shapely.union_all(shapely.buffer(np.asarray(built, dtype=object), ROAD_W / 2)) \
        if built else None
    inside = ndimage.distance_transform_edt(mask) * RES
    built_lines = shapely.union_all(np.asarray(built, dtype=object)) if built else None
    out = []
    for g in lines:
        pieces = shapely.get_parts(shapely.difference(g, corr)) if corr is not None else [g]
        for p in pieces:
            if shapely.get_type_id(p) != 1 or shapely.length(p) < MIN_M:
                continue
            joins = []
            for end in (shapely.get_point(p, 0), shapely.get_point(p, -1)):
                x, y = shapely.get_x(end), shapely.get_y(end)
                i = int(np.clip(round((y - raster.y0) / RES), 0, mask.shape[0] - 1))
                j = int(np.clip(round((x - raster.x0) / RES), 0, mask.shape[1] - 1))
                if shapely.distance(end, street) <= inside[i, j] + JOIN_SLACK_M:
                    joins.append(shapely.shortest_line(end, street))
                elif built_lines is not None and shapely.distance(end, built_lines) <= \
                        ROAD_W / 2 + JOIN_SLACK_M:
                    joins.append(shapely.shortest_line(end, built_lines))
            geom = shapely.line_merge(shapely.union_all([p, *joins])) if joins else p
            out.append(Line(geom, _sample(img, raster, p)))
    return out


class Join(Protocol):
    """What becomes of a candidate no other line links to the street: joined some other way, or
    dropped (None)."""

    @property
    def name(self) -> str: ...

    def link(self, c: Line, target: Any, v: Vote, raster: BlockRaster) -> Line | None: ...


class Reach:
    """Only the joins of level_lines: a candidate left floating is dropped."""
    name = "jr"

    def link(self, c: Line, target: Any, v: Vote, raster: BlockRaster) -> Line | None:
        return None


@dataclass(frozen=True)
class Traced:
    """A floating candidate traced downstream from an end along the current (TRACE_STEP_M
    steps) until it comes within ROAD_W / 2 of the street or a road, at most max_m: the shorter
    of its two ends' traces, joined to it by the shortest line (none: dropped)."""
    max_m: float

    @property
    def name(self) -> str:
        return f"jf{self.max_m:g}"

    def link(self, c: Line, target: Any, v: Vote, raster: BlockRaster) -> Line | None:
        best = None
        for end in (shapely.get_point(c.geom, 0), shapely.get_point(c.geom, -1)):
            path = _trace(np.array([shapely.get_x(end), shapely.get_y(end)]), target, v, raster,
                          self.max_m)
            if path is not None and (best is None or shapely.length(path) < shapely.length(best)):
                best = path
        if best is None:
            return None
        return Line(shapely.line_merge(shapely.union_all([c.geom, best])), c.vote)


def _trace(p: NDArray[np.float64], target: Any, v: Vote, raster: BlockRaster,
           max_m: float) -> Any:
    """From p along the current to within ROAD_W / 2 of `target` (prepared), its path and the
    shortest line on; None if the current fails (off open ground, or nothing flowing)."""
    path = [p.copy()]
    for _ in range(int(max_m / TRACE_STEP_M)):
        pt = shapely.Point(*p)
        if shapely.dwithin(target, pt, ROAD_W / 2):
            snap = shapely.get_coordinates(shapely.shortest_line(pt, target))
            return shapely.LineString(np.vstack([np.asarray(path), snap[1:]])) \
                if len(path) > 1 or not np.allclose(snap[0], snap[1]) else None
        i = int(round((p[1] - raster.y0) / RES))
        j = int(round((p[0] - raster.x0) / RES))
        if not (0 <= i < v.jx.shape[0] and 0 <= j < v.jx.shape[1]):
            return None
        d = np.array([v.jx[i, j], v.jy[i, j]])
        n = float(np.hypot(*d))
        if not np.isfinite(n) or n <= 0.0:
            return None
        p = p + TRACE_STEP_M * d / n
        path.append(p.copy())
    return None


def grow(cands: list[Line], street: Any, built: list[Any], join: Join, v: Vote,
         raster: BlockRaster) -> list[Line]:
    """The candidates linked to the street, each in turn the highest-voted one that touches the
    street (within STREET_TOL) or what is in so far (road_simp.linked's rule, grown one line at
    a time so that every prefix is linked); when none does, the highest-voted one left is joined
    by `join` or dropped."""
    order: list[Line] = []
    left = sorted(cands, key=lambda c: -c.vote)
    near = [shapely.distance(c.geom, street) <= STREET_TOL for c in left]
    net = (shapely.union_all(np.asarray(built, dtype=object)) if built
           else shapely.GeometryCollection())
    while left:
        on = [i for i, c in enumerate(left)
              if near[i] or (not net.is_empty and shapely.distance(c.geom, net) <= 1e-6)]
        if on:
            c = left.pop(on[0])
            near.pop(on[0])
        else:
            target = shapely.union(street, net)
            shapely.prepare(target)
            joined = join.link(left.pop(0), target, v, raster)
            near.pop(0)
            if joined is None:
                continue
            c = joined
        order.append(c)
        net = shapely.union(net, c.geom)
    return order


# ------------------------------------------------------------------ the build

@dataclass
class World:
    """The block, its metric and the plain model, built once."""
    b: Any
    metric: gcl.ComplexMetric
    geo: gc.Geometry
    raster: BlockRaster
    street: Any
    W: NDArray[np.float64]

    @staticmethod
    def of(bid: str) -> World:
        [b] = common.build_blocks([bid])
        return World(b=b, metric=metrics.metric_of(metrics.RUNTIME, device="cpu"),
                     geo=gc.geometry(b), raster=BlockRaster.of(b, RES),
                     street=shapely.union_all(b.streets.geometry.to_numpy()),
                     W=common.POPULATIONS["area"].weights(np.asarray(b.buildings.outlines)))


def vote(world: World, k: NDArray[np.float64], signal: Signal, built: list[Any]) -> Vote:
    """The vote with the clearing k and the roads `built` carved (none: the plain mesh)."""
    b, metric = world.b, world.metric
    if not built:
        m = metric.clearing(b, demand=common.PopulationDemand())
        return vote_image(m, m.state(k), signal, world.raster, np.zeros(m.fx.fm.nt, dtype=bool))
    net = gpd.GeoDataFrame(geometry=built, crs=b.crs).assign(width_m=ROAD_W)
    carving = gcl.Carving.of(world.geo, [[world.geo.local(road_corridor(net))]])
    with tempfile.TemporaryDirectory() as tmp:
        fx = metric.fixed(b, cache=Path(tmp), carving=carving)
    m = metric.model(b, fx, demand=common.PopulationDemand())
    prefix = gcl.Prefix(net=0, stop=1)
    paved = carving.paved(prefix, m.cen_)
    return vote_image(m, m.cleared_road_state(k, prefix), signal, world.raster, paved)


def ordered_lines(world: World, v: Vote, built: list[Any], join: Join) -> list[Line]:
    """Every level's linked lines, the highest level first (module docstring)."""
    out: list[Line] = []
    net = list(built)
    for q in QUANTILES:
        lines = grow(level_lines(v.img, world.raster, q, world.street, net), world.street, net,
                     join, v, world.raster)
        out += lines
        net += [c.geom for c in lines]
    return out


def build(bid: str, source: Source, signal: Signal, dyn: Dynamics, join: Join) -> None:
    t0 = time.time()
    world = World.of(bid)
    n = len(world.geo.polys)
    k = source.k(n)
    out = ROWS / metrics.RUNTIME / run_name(source, signal, dyn, join)
    out.mkdir(parents=True, exist_ok=True)
    built: list[Any] = []
    votes: list[float] = []
    steps: list[dict[str, Any]] = []
    inc = IncrementalDestroyed(common.MEASURE, world.b.buildings)
    if isinstance(dyn, Frozen):
        lines = ordered_lines(world, vote(world, k, signal, []), [], join)
        built, votes = [c.geom for c in lines], [c.vote for c in lines]
        steps.append(dict(step=0, roads=len(built), t=time.time() - t0))
    else:
        target = dyn.step
        while True:
            kv = k if (dyn.keep or not built) else np.zeros(n)
            lines = ordered_lines(world, vote(world, kv, signal, built), built, join)
            added = 0
            for c in lines:
                inc.add(road_corridor(gpd.GeoDataFrame(geometry=[c.geom], crs=world.b.crs).assign(
                    width_m=ROAD_W)))
                built.append(c.geom)
                votes.append(c.vote)
                added += 1
                if float((inc.c() * world.W).sum() / world.W.sum()) >= target:
                    break
            D = float((inc.c() * world.W).sum() / world.W.sum())
            steps.append(dict(step=len(steps), roads=len(built), added=added, D=D,
                              t=time.time() - t0))
            print(json.dumps(steps[-1]), flush=True)
            if added == 0 or D >= LAST:
                break
            while target <= D:
                target = round(target + dyn.step, 10)
    net = gpd.GeoDataFrame(dict(vote=votes), geometry=built, crs=world.b.crs).assign(
        width_m=ROAD_W)
    if len(net) and not road_simp.linked(net.geometry.to_numpy(), world.street).all():
        raise RuntimeError(f"{bid} {out.name}: a road not linked to the street")
    net.to_parquet(out / f"{bid}_roads.parquet")
    (out / f"{bid}_build.json").write_text(json.dumps(steps, indent=1))
    print(f"{bid} {out.name}: {len(net)} roads, {net.length.sum() / 1000:.2f} km "
          f"({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ the score

def cuts(world: World, net: gpd.GeoDataFrame, k0: NDArray[np.float64]) -> list[tuple[float, int]]:
    """(D, prefix length) at each target D the roads reach, D counting the clearing k0's buildings
    too (a real clearing; zeros: the roads alone)."""
    inc = IncrementalDestroyed(common.MEASURE, world.b.buildings)
    if k0.any():
        inc.add(shapely.union_all(np.asarray(world.b.buildings.outlines)[k0 > 0]))
    Ds = []
    for i in range(len(net)):
        inc.add(road_corridor(cast("gpd.GeoDataFrame", net.iloc[[i]])))
        Ds.append(float((inc.c() * world.W).sum() / world.W.sum()))
    D = np.asarray(Ds)
    out = []
    for t in TARGETS:
        j = int(np.searchsorted(D, t))
        if j < len(D):
            out.append((t, j + 1))
    return out


def score(bid: str, source: Source, signal: Signal, dyn: Dynamics, join: Join) -> None:
    t0 = time.time()
    world = World.of(bid)
    n = len(world.geo.polys)
    out = ROWS / metrics.RUNTIME / run_name(source, signal, dyn, join)
    net = gpd.read_parquet(out / f"{bid}_roads.parquet")
    k = source.k(n)
    roles = {"dev": np.zeros(n)} | ({"real": k} if k.any() and not (
        isinstance(dyn, Periodic) and not dyn.keep) else {})
    stops = {role: cuts(world, net, kk) for role, kk in roles.items()}
    ms = sorted({mk for cs in stops.values() for _t, mk in cs})
    carving = gcl.Carving.of(world.geo, [[world.geo.local(rr.corridor_of(net, mk)) for mk in ms]])
    with tempfile.TemporaryDirectory() as tmp:
        fx = world.metric.fixed(world.b, cache=Path(tmp), carving=carving)
    m = world.metric.model(world.b, fx, demand=common.PopulationDemand())
    rows = []
    for role, kk in roles.items():
        if role == "real":
            L = m.lenses_of(m.state(kk))
            d0 = float(world.W[kk > 0].sum() / world.W.sum())
            rows.append(dict(role=role, target=0.0, roads=0, km=0.0, D=d0,
                             **rr._lens_row(m, L)))
        for t, mk in stops[role]:
            prefix = gcl.Prefix(net=0, stop=ms.index(mk) + 1)
            L = m.cleared_road_lenses(kk, prefix)
            inc_k = IncrementalDestroyed(common.MEASURE, world.b.buildings)
            inc_k.add(shapely.union_all(np.asarray(
                [*np.asarray(world.b.buildings.outlines)[kk > 0], rr.corridor_of(net, mk)],
                dtype=object)))
            rows.append(dict(role=role, target=t, roads=mk,
                             km=float(net.iloc[:mk].length.sum() / 1000),
                             D=float((inc_k.c() * world.W).sum() / world.W.sum()),
                             **rr._lens_row(m, L)))
            print(json.dumps({key: rows[-1][key] for key in ("role", "target", "roads", "km", "D",
                                                            "L_P")}), flush=True)
    df = pd.DataFrame(rows).assign(block=bid, run=out.name, metric=world.metric.name,
                                   nt=fx.fm.nt, t_total=time.time() - t0)
    df.to_parquet(out / f"{bid}.parquet")
    print(f"{bid} {out.name}: scored {len(df)} rows ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------ the clearings

def copy_clearing(name: str, row: Path) -> None:
    """A relax row's clearing (`simp_cleared`, `free_cleared`), kept as CLEARINGS/<name>.csv."""
    r = pd.read_parquet(row).iloc[0]
    cleared = np.union1d(np.asarray(r.simp_cleared, dtype=np.int64),
                         np.asarray(r.free_cleared, dtype=np.int64))
    CLEARINGS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(cleared=cleared)).assign(
        n=int(r.n), block=r.block, D=float(r.simp_D), plan=r.plan, along=r.along,
        demand=r.demand, power=float(r.power), source=str(row)).to_csv(
        CLEARINGS / f"{name}.csv", index=False)
    print(f"{name}: {len(cleared)} buildings of {int(r.n)}, D {float(r.simp_D):.4f}")


def clearing(bid: str, metric_name: str, plan_spec: str, q: float, device: str,
             name: str) -> None:
    """relax.one under P_q (common.PriorityDemand(q)) at D 0.05, its clearing kept."""
    import relax
    metric = metrics.metric_of(metric_name, device=device)
    plan = relax.plan_of(plan_spec)
    r = relax.one(bid, 1.0, metric, plan, relax.D_LENS, demand=common.PriorityDemand(q))
    row = ROWS / "clearings_rows" / f"{name}.parquet"
    row.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r]).to_parquet(row)
    copy_clearing(name, row)


def main(argv: list[str]) -> None:
    if len(argv) == 3 and argv[0] == "copy-clearing":
        copy_clearing(argv[1], Path(argv[2]))
    elif len(argv) == 7 and argv[0] == "clearing":
        clearing(argv[1], argv[2], argv[3], float(argv[4]), argv[5], argv[6])
    elif len(argv) == 6 and argv[0] in ("build", "score"):
        {"build": build, "score": score}[argv[0]](
            argv[1], source_of(argv[2]), signal_of(argv[3]), dynamics_of(argv[4]),
            join_of(argv[5]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
