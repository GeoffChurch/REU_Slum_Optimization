"""Grid templates: the top-down reblocking's baseline (owner 2026-10-27;
.superpowers/sdd/2026-10-27-partition/brief.md, part 3).

A layout is a family of road lines across the block: parallel lines (`par`) at spacing a, angle
theta and offset o (a fraction of a), or two-way (`grid`: the same and the lines square to them
at the same spacing); theta is read from the block's principal axis (the buildings' centroids
weighted by floor area). Each line, clipped to the block, is snapped to the gaps: the cheapest
path in D between its ends within SNAP_W of it (skimage's MCP on a PIXEL raster; a pixel costs
LEN_COST per metre plus, within ROAD_W / 2 of a building, that building's D per metre of a road
through it, w_b ROAD_W / (tau area_b): common.MEASURE's ramp below its knee), then built as a
road ROAD_W wide (road_corridor).

A layout is scored in road_simp's eps world (one solve: the triangles its corridors cover, by
centroid, paved as a road, kappa 10; D the ramp over the covered shares, C the sheet's added), J
= P + lambda C (road_simp.SheetPrice). Over the budget, its lines are dropped in the order of their
first-order gain per D (a line's gain: -sum dP/ds over the triangles only it covers, at the whole
layout's state; its D: what dropping it alone saves), D read exactly after each drop until within
the budget, then solved once more. The search (`search`): every layout of COARSE, then the best's
REFINE neighbours (theta +- REFINE_DEG, a x REFINE_A, o at REFINE_O); the best layout within the
budget scored exactly on its own carved mesh with partition.measures (sub-blocks, dead ends, the
depth measures, the damage term's cut). No bend (the brief's optional bend along the principal
axis: not built).

    T=research/roadless/templates.py                   # from the repo root, PYTHONPATH=.
    uv run python $T search <block> <D> <price> cpu|gpu [--replace]

Rows in templates_rows/<metric>/tp.D<D>.s<price>.v<REV>/<block>.parquet: a row per layout tried
(its parameters, lines, J, P, D, C in the eps world, before and after the drops) and the best's
exact scores (`best_*`), beside the best's ground and lines (<block>_ground.parquet,
_lines.parquet).
"""
from __future__ import annotations

import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
from scipy import ndimage
from skimage.graph import MCP_Geometric

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_calib  # noqa: E402
import metrics  # noqa: E402
import partition as pt  # noqa: E402
import road_simp as rs  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402

F64: TypeAlias = NDArray[np.float64]
B: TypeAlias = NDArray[np.bool_]

ROAD_W = rs.ROAD_W
PIXEL = rs.PIXEL
SNAP_W = 15.0            # m: a line's snapped path stays within this of it
LEN_COST = 2e-6          # D per metre of road on open ground (a path's length, priced lightly)
SIMPLIFY_M = 1.0         # m: a snapped path simplified to this
MIN_PIECE_M = 20.0       # m: a clipped line's piece shorter than this is dropped
# the search: (family, theta from the principal axis (deg), spacing a (m), offset o)
COARSE = tuple([("par", th, a, 0.0) for th in (0, 45, 90, 135) for a in (50, 80, 120, 180)]
               + [("grid", th, a, 0.0) for th in (0, 45) for a in (80, 120, 180)])
REFINE_DEG = 10.0
REFINE_A = (0.8, 1.25)
REFINE_O = (0.25, 0.5, 0.75)
REV = 1
OUT = HERE / "templates_rows"


@dataclass(frozen=True, kw_only=True)
class Layout:
    family: str          # par | grid
    theta: float         # deg from the principal axis
    a: float             # m: the spacing
    o: float             # the offset, a fraction of a

    @property
    def name(self) -> str:
        return f"{self.family}.th{self.theta:g}.a{self.a:g}.o{self.o:g}"


class Block:
    """The block's template problem: its eps world (road_simp.Roads, no clearing), the snapping
    raster and the principal axis (module docstring)."""

    def __init__(self, b: Any, metric: Any) -> None:
        self.b = b
        m = metric.clearing(b, demand=common.PopulationDemand())
        none = np.zeros(m.n, dtype=bool)
        self.rd = rs.Roads(m, radius=ROAD_W, cleared=none, real=False, free=none)
        self.geo = m.geo
        polys = np.asarray(self.geo.polys)
        area = shapely.area(polys)
        cen = shapely.get_coordinates(shapely.centroid(polys))
        c = cen - np.average(cen, axis=0, weights=area)
        ev, vec = np.linalg.eigh((c * area[:, None]).T @ c)
        self.theta0 = float(np.degrees(np.arctan2(vec[1, -1], vec[0, -1])))
        self.boundary = self.geo.local(b.boundary)
        x0, y0, x1, y1 = shapely.bounds(self.boundary)
        self.lo = np.array([x0, y0]) - PIXEL
        self.shape = (int(np.ceil((x1 - x0) / PIXEL)) + 2, int(np.ceil((y1 - y0) / PIXEL)) + 2)
        i, j = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]), indexing="ij")
        self.cx, self.cy = self.lo[0] + PIXEL * (i + 0.5), self.lo[1] + PIXEL * (j + 0.5)
        shapely.prepare(self.boundary)
        self.inside = shapely.contains_xy(self.boundary, self.cx, self.cy)
        # each pixel within ROAD_W / 2 of a building: that building's D per metre of road
        owner = np.full(self.shape, -1, dtype=np.int64)
        tree = shapely.STRtree(polys)
        p, k = tree.query(shapely.points(self.cx.ravel(), self.cy.ravel()), predicate="within")
        owner.ravel()[p] = k
        dist, idx = ndimage.distance_transform_edt(owner < 0, return_indices=True)
        near = owner[idx[0], idx[1]]
        rate = self.rd.w * ROAD_W / (common.MEASURE.tau * area)
        self.cost = LEN_COST + np.where(dist * PIXEL <= ROAD_W / 2, rate[near], 0.0)
        self.P0 = self.rd.P0
        self.y = np.zeros(m.n)

    def lines(self, lay: Layout) -> list[Any]:
        """The layout's lines, clipped to the block and snapped (local frame)."""
        out = []
        for th in ((lay.theta,) if lay.family == "par" else (lay.theta, lay.theta + 90.0)):
            ang = np.radians(self.theta0 + th)
            t, n = np.array([np.cos(ang), np.sin(ang)]), np.array([-np.sin(ang), np.cos(ang)])
            ctr = np.asarray(shapely.get_coordinates(shapely.centroid(self.boundary))[0])
            half = float(np.hypot(*self.shape)) * PIXEL
            for k in np.arange(-half // lay.a - 1, half // lay.a + 2):
                c = ctr + (k + lay.o) * lay.a * n
                seg = shapely.linestrings([c - half * t, c + half * t])
                for piece in shapely.get_parts(shapely.intersection(seg, self.boundary)):
                    if shapely.length(piece) >= MIN_PIECE_M:
                        out.append(self.snap(piece))
        return [g for g in out if g is not None]

    def snap(self, piece: Any) -> Any:
        """The cheapest path in D between the piece's ends within SNAP_W of it (MCP on the
        raster's window about the piece)."""
        band = shapely.buffer(piece, SNAP_W)
        x0, y0, x1, y1 = shapely.bounds(band)
        i0, j0 = (np.floor((np.array([x0, y0]) - self.lo) / PIXEL).astype(int) - 1).clip(0)
        i1, j1 = np.minimum(np.ceil((np.array([x1, y1]) - self.lo) / PIXEL).astype(int) + 1,
                            self.shape)
        cx, cy = self.cx[i0:i1, j0:j1], self.cy[i0:i1, j0:j1]
        shapely.prepare(band)
        ok = self.inside[i0:i1, j0:j1] & shapely.contains_xy(band, cx, cy)
        if ok.sum() < 2:
            return None
        cost = np.where(ok, self.cost[i0:i1, j0:j1], np.inf)
        ends = shapely.get_coordinates(piece)[[0, -1]]
        i, j = np.nonzero(ok)
        ij = [(int(i[k]), int(j[k])) for k in
              (int(np.argmin((cx[i, j] - e[0]) ** 2 + (cy[i, j] - e[1]) ** 2)) for e in ends)]
        mcp = MCP_Geometric(cost)
        mcp.find_costs([ij[0]], [ij[1]])
        path = np.asarray(mcp.traceback(ij[1]), dtype=np.float64)
        if len(path) < 2:
            return None
        xy = self.lo + PIXEL * (path + np.array([i0, j0]) + 0.5)
        return shapely.simplify(shapely.linestrings(xy), SIMPLIFY_M)

    def tri_sets(self, lines: list[Any]) -> list[B]:
        """Per line the triangles its corridor covers (by centroid, as Carving.paved)."""
        cen = self.rd.m.cen_
        out = []
        for g in lines:
            cor = shapely.buffer(g, ROAD_W / 2, cap_style="flat")
            shapely.prepare(cor)
            out.append(shapely.contains_xy(cor, cen[:, 0], cen[:, 1]))
        return out

    def world(self, on: B) -> rs.World:
        r = on.astype(np.float64)
        return self.rd.world_of(r, r, self.y, 1.0, 1.0)

    def evaluate(self, sets: list[B], keep: B, price: rs.SheetPrice
                 ) -> tuple[float, float, float, float, F64]:
        """(J, P, D, C, dP/ds per triangle) of the kept lines, one solve."""
        on = np.zeros(self.rd.m.fx.fm.nt, dtype=bool)
        for k in np.flatnonzero(keep):
            on |= sets[k]
        w = self.world(on)
        P, gs, _ = self.rd.P_grad(w)
        cost = price.charge(self.rd, w, rs.OC()).cost
        return (P + cost) / self.P0, P, self.rd.D(w.r, w.c), self.rd.C(w), gs

    def D_of(self, sets: list[B], keep: B) -> float:
        on = np.zeros(self.rd.m.fx.fm.nt, dtype=bool)
        for k in np.flatnonzero(keep):
            on |= sets[k]
        w = self.world(on)
        return self.rd.D(w.r, w.c)


def fit(blk: Block, lay: Layout, *, budget: float, price: rs.SheetPrice) -> dict[str, Any]:
    """The layout within the budget (module docstring): its row and its kept lines."""
    t0 = time.time()
    lines = blk.lines(lay)
    sets = blk.tri_sets(lines)
    keep = np.ones(len(lines), dtype=bool)
    J, P, D, C, gs = blk.evaluate(sets, keep, price)
    row = dict(layout=lay.name, family=lay.family, theta=lay.theta, a=lay.a, o=lay.o,
               lines=len(lines), J_all=J, L_P_all=1.0 - P / blk.P0, D_all=D, C_all=C)
    if D > budget:
        cover = np.sum(sets, axis=0)
        gain = np.array([-(gs[s & (cover == 1)]).sum() for s in sets])
        dD = np.array([D - blk.D_of(sets, keep & (np.arange(len(lines)) != k))
                       for k in range(len(lines))])
        for k in np.argsort(gain / np.maximum(dD, 1e-12)):
            keep[k] = False
            if blk.D_of(sets, keep) <= budget:
                break
        J, P, D, C, _ = blk.evaluate(sets, keep, price)
    row |= dict(kept=int(keep.sum()), J=J, L_P=1.0 - P / blk.P0, D=D, C=C, t=time.time() - t0)
    row["_lines"] = [g for g, k in zip(lines, keep, strict=True) if k]
    return row


def search(bid: str, budget: float, price: rs.SheetPrice, device: str, *,
           standing: clearings.Standing) -> None:
    metric = metrics.metric_of(metrics.RUNTIME, device=device)
    out = OUT / metrics.RUNTIME / f"tp.D{budget:g}.{price.name}.v{REV}" / f"{bid}.parquet"
    if not standing.todo(out):
        return
    t0 = time.time()
    [b] = common.build_blocks([bid])
    blk = Block(b, metric)
    print(f"{bid}: principal axis {blk.theta0:.1f} deg, {len(COARSE)} coarse layouts "
          f"({time.time() - t0:.0f}s)", flush=True)
    rows = []

    def tryit(lay: Layout) -> None:
        r = fit(blk, lay, budget=budget, price=price)
        rows.append(r)
        print(f"  {lay.name}: {r['lines']} lines (D {r['D_all']:.4f}, L_P {r['L_P_all']:.4f}) "
              f"-> {r['kept']} kept: J {r['J']:.4f}, L_P {r['L_P']:.4f}, D {r['D']:.4f}, C "
              f"{r['C']:.0f} m^2 ({r['t']:.0f}s)", flush=True)

    for fam, th, a, o in COARSE:
        tryit(Layout(family=fam, theta=th, a=a, o=o))
    best = min(rows, key=lambda r: r["J"])
    fam, th, a, o = best["family"], best["theta"], best["a"], best["o"]
    near = ([Layout(family=fam, theta=th + d, a=a, o=o) for d in (-REFINE_DEG, REFINE_DEG)]
            + [Layout(family=fam, theta=th, a=a * f, o=o) for f in REFINE_A]
            + [Layout(family=fam, theta=th, a=a, o=oo) for oo in REFINE_O if oo != o])
    for lay in near:
        tryit(lay)
    best = min(rows, key=lambda r: r["J"])
    lines = [blk.geo.utm(g) for g in best["_lines"]]
    net = gpd.GeoDataFrame(geometry=lines, crs=b.crs).assign(width_m=ROAD_W)
    ground = shapely.intersection(road_corridor(net), b.boundary)
    out.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(geometry=[ground], crs=b.crs).to_parquet(
        out.with_name(f"{bid}_ground.parquet"))
    net.to_parquet(out.with_name(f"{bid}_lines.parquet"))
    sc = pt.score_answer(b, metric, ground, shapely.multilinestrings(lines))
    print(f"{bid} best {best['layout']} ({best['kept']} lines): L_P {sc['L_P']:.4f}, L_J2 "
          f"{sc['L_J2']:.4f}, D {sc['D_exact']:.4f}, C {sc['C']:.0f} m^2, {sc['faces']} "
          f"sub-blocks (largest {sc['face_largest_m2']:.0f} m^2), road {sc['road_m']:.0f} m, "
          f"dead ends {sc['dead_ends']}, deep within 50 m {sc['deep_within50']:.2f}, L_P deep "
          f"{sc['L_P_deep']:.4f}, cut {sc['damage_cut']:.4f} ({time.time() - t0:.0f}s)",
          flush=True)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "_lines"} for r in rows]).assign(
        block=bid, budget=budget, price=price.price, best=best["layout"],
        **{f"best_{k}": v for k, v in sc.items()}, device=device, host=socket.gethostname(),
        peak_rss_gb=gap_calib.peak_gb(), t_total=time.time() - t0)
    standing.write(df, out)


def main(argv: list[str]) -> None:
    argv, standing = clearings.standing_of(argv)
    if len(argv) == 5 and argv[0] == "search" and argv[4] in ("cpu", "gpu"):
        search(argv[1], float(argv[2]), rs.SheetPrice(price=float(argv[3])), argv[4],
               standing=standing)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
