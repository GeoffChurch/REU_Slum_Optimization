"""Why Shock Reblock makes 4-way crossings (owner 2026-11-19: "It would be quite interesting if
they are promoted as a side-effect of an otherwise agnostic mechanism").

Runs ears.py's `build` unchanged on a world (a World Strategy: the real block, or synthetic fabric
on it) and records what the crossings analysis (analyze.py) needs and the picks rows lack: every
step's costed candidates with their centrelines (ears.Ear.line; the built ear is the one whose
ends the step's pick row names; `bld` per pixel its line crosses, ears.line_pixels), and per
state the raster's face labels, sink pixels, the road each sink pixel's nearest bounding triangle
belongs to (its tri_step; -1 the street) and the potential u. The recorder is a Candidates
wrapper (Recording), so ears.py's loop and plan are untouched; the plan's name is the wrapped
one's.

Worlds:
  block:<bid>                      the block as it is
  fabric:<bid>:<shape>:<seed>      no fabric: <bid>'s footprints resampled with replacement, each
                                   turned by a uniform random angle and dropped at a uniform random
                                   point (random sequential adsorption: inside the block at
                                   SETBACK_M from its edge, CLEAR_M from every footprint already
                                   placed), to the block's own count or until MAX_TRIES misses;
                                   <shape> `same` (the block's boundary and streets, its OSM ways
                                   outside its interior only), `rect` (a RECT_ASPECT rectangle of
                                   the block's area, all edges street, no OSM features) or `tilt`
                                   (that rectangle turned TILT_DEG off the raster's axes, so its
                                   roads do not run along the ear sources' SRC_M grid)
  lattice:<bid>:<seed>             H2's positive control: <bid>'s area as a RECT_ASPECT rectangle
                                   of LAT_W x LAT_W m buildings on a LAT_W + LAT_GAP m grid (straight
                                   alleys LAT_GAP wide both ways), the grid turned by a random angle

    C=research/roadless/atlas/crossings/crossings.py    # from the repo root, PYTHONPATH=.
    uv run python $C run <world> <plan>
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree
from shapely import affinity
from shapely.geometry import LineString, Polygon, box

HERE = Path(__file__).resolve().parent
ROADLESS = HERE.parents[1]
sys.path.insert(0, str(ROADLESS))
import common  # noqa: E402
import ears  # noqa: E402
import gap_exits as gx  # noqa: E402
import growth as gr  # noqa: E402

ROWS = HERE / "rows"
SETBACK_M = 1.0        # fabric: a footprint this far inside the block's edge
CLEAR_M = 1.0          # fabric: footprints at least this far apart (no two homes overlap)
MAX_TRIES = 200_000    # fabric: drops before it stops short of the block's count
RECT_ASPECT = 1.5      # fabric/lattice `rect`: width over height
TILT_DEG = 30.0        # fabric `tilt`: the rectangle turned this far off the raster's (and the
                       # ear sources' SRC_M grid's) axes
LAT_W = 8.0            # m: lattice: a building's side
LAT_GAP = 3.0          # m: lattice: an alley's width
INTERIOR_M = 5.0       # m: fabric `same`: an OSM way meeting the block this far in is fabric


# ------------------------------------------------------------------ worlds

class World(Protocol):
    @property
    def name(self) -> str: ...

    def block(self) -> Any: ...


def _real(bid: str) -> Any:
    [b] = common.build_blocks([bid])
    return b


def _empty_osm(like: gx.Osm) -> gx.Osm:
    return gx.Osm(ways=like.ways.iloc[:0], nodes=like.nodes.iloc[:0], areas=like.areas.iloc[:0])


def _rect_of(b: Any) -> Polygon:
    x0, y0, x1, y1 = b.boundary.bounds
    h = float(np.sqrt(b.boundary.area / RECT_ASPECT))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return box(cx - RECT_ASPECT * h / 2, cy - h / 2, cx + RECT_ASPECT * h / 2, cy + h / 2)


def _synthetic(b: Any, *, name: str, boundary: Polygon, polys: list[Polygon], streets: Any,
               osm: gx.Osm) -> Any:
    """`b` with its fabric replaced: footprints `polys` (anchors their representative points),
    parcels their Voronoi cells in `boundary`, streets `streets`, OSM `osm`; uncacheable."""
    crs = b.crs
    anchors = [p.representative_point() for p in polys]
    fp = gpd.GeoDataFrame({"area_in_meters": [p.area for p in polys],
                           "anchor": gpd.GeoSeries(anchors, crs=crs)},
                          geometry=gpd.GeoSeries(polys, crs=crs), crs=crs)
    cells = shapely.voronoi_polygons(shapely.multipoints(anchors), extend_to=boundary)
    parts = [c.intersection(boundary) for c in shapely.get_parts(cells)]
    parts = [p if p.geom_type == "Polygon" else max(p.geoms, key=lambda q: q.area)
             for p in parts if not p.is_empty]
    parcels = gpd.GeoDataFrame({"parcel_id": range(len(parts))}, geometry=parts, crs=crs)
    st = gpd.GeoDataFrame(geometry=[streets], crs=crs)
    return replace(b, block_id=name, boundary=boundary, parcels=parcels, streets=st,
                   building_geometries=fp, source_content_hash=None,
                   attrs={**b.attrs, "osm": osm})


def _drop(src: list[Polygon], region: Polygon, *, n: int, rng: np.random.Generator) -> list[Polygon]:
    """Random sequential adsorption of `n` footprints drawn from `src` into `region`."""
    inner = region.buffer(-SETBACK_M)
    shapely.prepare(inner)
    x0, y0, x1, y1 = inner.bounds
    placed: list[Polygon] = []
    grid: dict[tuple[int, int], list[int]] = {}
    cell = 20.0
    tries = 0
    while len(placed) < n and tries < MAX_TRIES:
        tries += 1
        p = src[int(rng.integers(len(src)))]
        c = p.centroid
        q = affinity.rotate(affinity.translate(p, -c.x, -c.y), float(rng.uniform(0, 360)),
                            origin=(0, 0))
        q = affinity.translate(q, float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1)))
        if not inner.contains(q):
            continue
        qb = q.buffer(CLEAR_M)
        bx0, by0, bx1, by1 = qb.bounds
        near = {k for i in range(int(bx0 // cell), int(bx1 // cell) + 1)
                for j in range(int(by0 // cell), int(by1 // cell) + 1) for k in grid.get((i, j), [])}
        if any(qb.intersects(placed[k]) for k in near):
            continue
        k = len(placed)
        placed.append(q)
        qx0, qy0, qx1, qy1 = q.bounds
        for i in range(int(qx0 // cell), int(qx1 // cell) + 1):
            for j in range(int(qy0 // cell), int(qy1 // cell) + 1):
                grid.setdefault((i, j), []).append(k)
    return placed


@dataclass(frozen=True, kw_only=True)
class RealBlock:
    bid: str

    @property
    def name(self) -> str:
        return f"block:{self.bid}"

    def block(self) -> Any:
        return _real(self.bid)


@dataclass(frozen=True, kw_only=True)
class Fabric:
    """No fabric (module docstring)."""
    bid: str
    shape: str          # "same" | "rect"
    seed: int

    def __post_init__(self) -> None:
        if self.shape not in ("same", "rect", "tilt"):
            raise ValueError(f"{self}: shape same|rect|tilt")

    @property
    def name(self) -> str:
        return f"fabric:{self.bid}:{self.shape}:{self.seed}"

    def block(self) -> Any:
        b = _real(self.bid)
        rng = np.random.default_rng(self.seed)
        src = list(b.building_geometries.geometry)
        osm: gx.Osm = b.attrs["osm"]
        if self.shape == "same":
            region = b.boundary
            streets = shapely.union_all(list(b.streets.geometry))
            core = b.boundary.buffer(-INTERIOR_M)
            ways = osm.ways[~osm.ways.geometry.intersects(core)]
            osm = gx.Osm(ways=ways, nodes=osm.nodes.iloc[:0], areas=osm.areas.iloc[:0])
        else:
            region = _rect_of(b)
            if self.shape == "tilt":
                region = affinity.rotate(region, TILT_DEG, origin="centroid")
            streets = region.exterior
            osm = _empty_osm(osm)
        polys = _drop(src, region, n=len(src), rng=rng)
        return _synthetic(b, name=f"SYN.{self.name.replace(':', '_')}", boundary=region,
                          polys=polys, streets=LineString(streets.coords)
                          if streets.geom_type == "LinearRing" else streets, osm=osm)


@dataclass(frozen=True, kw_only=True)
class Lattice:
    """H2's positive control (module docstring)."""
    bid: str
    seed: int

    @property
    def name(self) -> str:
        return f"lattice:{self.bid}:{self.seed}"

    def block(self) -> Any:
        b = _real(self.bid)
        rng = np.random.default_rng(self.seed)
        region = _rect_of(b)
        inner = region.buffer(-SETBACK_M)
        c = region.centroid
        ang = float(rng.uniform(0, 90))
        R = float(np.hypot(*np.subtract(region.bounds[2:], region.bounds[:2])))
        pitch = LAT_W + LAT_GAP
        off = rng.uniform(0, pitch, 2)
        polys = []
        for i in np.arange(-R, R, pitch):
            for j in np.arange(-R, R, pitch):
                sq = box(i + off[0], j + off[1], i + off[0] + LAT_W, j + off[1] + LAT_W)
                q = affinity.translate(affinity.rotate(sq, ang, origin=(0, 0)), c.x, c.y)
                if inner.contains(q):
                    polys.append(q)
        return _synthetic(b, name=f"SYN.{self.name.replace(':', '_')}", boundary=region,
                          polys=polys, streets=LineString(region.exterior.coords),
                          osm=_empty_osm(b.attrs["osm"]))


def world_of(spec: str) -> World:
    kind, *rest = spec.split(":")
    if kind == "block" and len(rest) == 1:
        return RealBlock(bid=rest[0])
    if kind == "fabric" and len(rest) == 3:
        return Fabric(bid=rest[0], shape=rest[1], seed=int(rest[2]))
    if kind == "lattice" and len(rest) == 2:
        return Lattice(bid=rest[0], seed=int(rest[1]))
    raise SystemExit(f"{spec}: a world is block:<bid> | fabric:<bid>:<same|rect>:<seed> | "
                     "lattice:<bid>:<seed>")


# ------------------------------------------------------------------ the recorder

class Recording:
    """A Candidates that delegates to `inner` and records, per call (one per state: build asks
    once a step), the state's rasters and the costed candidates' paths."""

    def __init__(self, inner: ears.Candidates) -> None:
        self.inner = inner
        self.states: list[dict[str, np.ndarray]] = []
        self.cands: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def pairs(self) -> int:
        return self.inner.pairs

    def ears(self, e: ears.EarWorld, faces: ears.Face, gain: Any, *, cost: gr.Cost,
             scale: float) -> list[ears.Ear]:
        g, ra = e.g, e.ra
        step = len(self.states)
        pf = np.where(ra.tri >= 0, faces.lab[np.maximum(ra.tri, 0)], -1).astype(np.int32)
        sink = e.sinks()
        att = np.flatnonzero(e.bounding())
        _, k = cKDTree(g.cen[att]).query(ra.xy[sink], k=1)
        road = np.full(len(pf), -2, dtype=np.int16)
        road[sink] = g.step[att[k]]
        self.states.append(dict(pf=pf, sink=sink, road=road, u=e.u_pix.astype(np.float32)))
        out = self.inner.ears(e, faces, gain, cost=cost, scale=scale)
        for r, q in enumerate(out):
            xy = q.line
            self.cands.append(dict(step=step, rank=r, face=q.face, cost=q.cost, first=q.first,
                                   path_x=xy[:, 0].tolist(), path_y=xy[:, 1].tolist(),
                                   bld=ra.bld[ears.line_pixels(ra.lo, ra.shape, xy)].tolist()))
        return out


def run(spec: str, plan_spec: str) -> None:
    t0 = time.time()
    w = world_of(spec)
    plan = ears.Plan.of(plan_spec)
    rec = Recording(plan.candidates)
    plan = replace(plan, candidates=rec)
    b = w.block()
    metric = ears.metrics.metric_of(ears.metrics.RUNTIME, device="cpu")
    e = ears.EarWorld(gr.growth_of(b, metric, contrast=plan.contrast))
    out = ROWS / w.name.replace(":", "_") / plan.name
    print(f"{w.name} {plan.name}: {e.g.nt} triangles, {len(b.building_geometries)} footprints, "
          f"raster {e.ra.shape} ({time.time() - t0:.0f}s)", flush=True)
    ears.build(e, plan, out=out, log=lambda s: print(s, flush=True))
    ra = e.ra
    pd.DataFrame(rec.cands).to_parquet(out / "cands.parquet")
    st = rec.states
    np.savez_compressed(out / "states.npz",
                        pf=np.stack([s["pf"] for s in st]), sink=np.stack([s["sink"] for s in st]),
                        road=np.stack([s["road"] for s in st]), u=np.stack([s["u"] for s in st]),
                        shape=np.asarray(ra.shape), lo=ra.lo, tri=ra.tri, bld=ra.bld)
    (out / "world.json").write_text(json.dumps(dict(
        world=w.name, plan=plan.name, block_id=b.block_id,
        boundary=shapely.to_wkt(b.boundary), n_buildings=len(b.building_geometries),
        coverage=float(b.building_geometries.area.sum() / b.boundary.area),
        t=time.time() - t0)))
    print(f"done ({time.time() - t0:.0f}s)", flush=True)


def main(argv: list[str]) -> None:
    if len(argv) == 3 and argv[0] == "run":
        run(argv[1], argv[2])
    elif len(argv) == 2 and argv[0] == "fabric":     # the synthetic block's footprints, a look
        b = world_of(argv[1]).block()
        print(b.block_id, len(b.building_geometries),
              f"coverage {b.building_geometries.area.sum() / b.boundary.area:.3f}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
