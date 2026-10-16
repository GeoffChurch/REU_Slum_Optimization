"""The region model's exits on the gap complex (BACKLOG 4, decision (4); NOTES, "Exits as leaks: the
region model on the complex (5810@major)"; the brief .superpowers/sdd/2026-10-13-region-leaks).

Everything compiles into conductances on the block's fixed mesh (gap_clear). Three Strategies, each
resolved once where the metric is built (gap_clear.complex_of from the metric's name) and held by
it (gap_clear.ComplexMetric.exits):

BARRIERS (`Barriers`). `NoBarriers`: the block as it is. `OsmBarriers(line_m, cross_m)`: the
railways and water inside a region (its OSM features, regions.Osm, read with the region from the
country's Geofabrik extract) cut out of the region's polygon, so they are obstacles that are never
cleared and hold no demand, and the region's edge is no exit where one lies on it. A railway track
or a waterway (not in a tunnel or culvert) is a strip `line_m` wide about its line (flat ends);
water areas (natural=water, landuse reservoir or basin, waterway riverbank) are themselves.
They are open only at real crossings: a way of the walking network (any highway) on a bridge or in
a tunnel over a barrier, or through a railway crossing node (railway=crossing or level_crossing:
the way's piece within `line_m` of the node; a node on a track that no way passes through, the
segment across the track there, `line_m` either side of it), each a gap `cross_m` wide about the
way's line. A
barrier changes the mesh: the fixed mesh is built on the cut block, so its cache key (the block's
boundary) and the metric's name (the suffix) both carry it.

LEAKS (`Leak`), conductances to ground on the diagonal of the system, on the cells that are
unknowns at baseline (live and not ground with every footprint standing, the edge open) and
nowhere else, the same for every state: B's enclosed homes stay enclosed, a cleared building's
interior and shy zone conduct but do not leak, the sparsity pattern is unchanged and P stays
convex in the clearing. `NoLeak`: the boundary only. `InternalLeak(open_m, foot_m)`: destinations
inside, two constants of the model, the same on every block and region (owner 2026-10-15: "set
the hyperparameters once and they would apply to all blocks"; NOTES, "The region's leaks:
destinations at buildings and on open ground (5810@major)"). The open ground leaks sigma /
open_m^2 per m^2 of it (sigma the sheet conductance), and each building's destinations sigma /
foot_m^2 per m^2 of its footprint (its population, by area), on the cells where its homes inject
(gap_flow.demand_on's points), as they inject; a length inf is none of that leak. The open ground
alone is the screened Laplacian -div(sigma grad u) + (sigma / open_m^2) u = f discretized cell by
cell, whose point source decays as K0(r / open_m) (the `leak-scale` guard): for a walker it ends a
trip at a constant rate per unit time anywhere on the open ground, a trip's straight-line length
r then having the density (r / open_m^2) K0(r / open_m), median 1.257 open_m. The constants were
chosen once, on 5810@major with its edge closed (region_leaks `calibrate`).

EDGE (`Edge`): `OpenEdge`, the street's faces ground (as the per-block metric has them);
`ClosedEdge`, no street face grounds, so every trip ends at a leak: the reference the constants
are chosen on (its leaks on the cells the open edge's street reaches).

BOOST (`Boost`): `NoBoost`; `StreetBoost(factor, classes)`: the sheet conductance times `factor`
on the triangles whose centroid lies within BOOST_M of the line of an OSM way whose highway tag
(`_link` folded) is one of `classes` (a passage by the harmonic mean of its two triangles'
factors, a street face by its triangle's). The streets' lever (owner 2026-10-16: "it rewards the
infrastructure investment"; "a 10x boost is I think reasonable"), on the drivable streets.

THE NAME. The per-block metric (no barriers, no leak, no boost, the edge open: the block's whole
street ring the exit) has no suffix, so its name and digest are as before. Any other names its
parts in a suffix `+x[B<line_m>c<cross_m>][E0][L[o<open_m>][b<foot_m>]][S<factor>h<classes>]
v<EXITS_CODE>` (the classes joined by `-`) before the search's `@e<eps>` (a number as `:g` spells
it, or its repr where that would round it);
EXITS_CODE digests this module's constants (digests.constants_of), so editing them renames only
the metrics that use them. `exits_of` parses a suffix back.

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/gap_exits.py
"""
from __future__ import annotations

import dataclasses
import math
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import geopandas as gpd
import numpy as np
import shapely
from numpy.typing import NDArray

import digests

if TYPE_CHECKING:
    import gap_clear
    import gap_complex as gc

    from reblock.contracts import Block

RAIL = ("rail", "light_rail", "narrow_gauge")         # railway=* tracks that are barriers (a tram
                                                      # runs in the street; disused ones are not)
WATERWAYS = ("river", "canal", "stream", "drain", "ditch")   # waterway lines that are barriers
WATER_NATURAL = ("water",)                            # water areas: natural=*
WATER_LANDUSE = ("reservoir", "basin")                # ... landuse=*
WATER_WATERWAY = ("riverbank",)                       # ... waterway=* (an area's)
CROSSING_NODES = ("crossing", "level_crossing")       # railway=* nodes where a way crosses
NODE_TOL_M = 0.5        # a way passes through a crossing node within this (m; OSM shares the node)
BOOST_M = 3.0           # StreetBoost: a triangle within this of a way's line is on it (m: half a
                        # residential street's 6 m)


# ------------------------------------------------------------------ a region's OSM features

class Osm(NamedTuple):
    """A region's OSM features meeting its face (regions.osm_of, read from the country's Geofabrik
    extract where the region is built, Block.attrs["osm"]), in the block's CRS, their tags as
    columns ("" where absent): `ways` (lines: highway, railway, waterway; bridge and tunnel, a tag
    other than "no"), `nodes` (points: railway) and `areas` (multipolygons: natural, landuse,
    waterway)."""
    ways: gpd.GeoDataFrame
    nodes: gpd.GeoDataFrame
    areas: gpd.GeoDataFrame


def osm_in(block: Block) -> Osm:
    """`block`'s OSM features: a region's (a block without them raises)."""
    osm = block.attrs["osm"]
    if not isinstance(osm, Osm):
        raise TypeError(f"{block.block_id}: attrs['osm'] is a {type(osm).__name__}, not an Osm")
    return osm


# ------------------------------------------------------------------ leaks

class LeakSite(NamedTuple):
    """What a leak reads of a block's model (gap_clear.Model), per cell of its fixed mesh: `area`
    (m^2), `live` (an unknown at baseline with the edge open: live and not ground, every footprint
    standing), `foot` (the footprint area its homes inject there, m^2: gap_clear.System.Pc @ each
    building's footprint area), and the sheet conductance `sigma`."""
    area: NDArray[np.float64]
    live: NDArray[np.bool_]
    foot: NDArray[np.float64]
    sigma: float


class Leak(Protocol):
    """A leak family: its name's part and its conductance to ground per cell."""

    @property
    def name(self) -> str: ...

    def per_cell(self, site: LeakSite) -> NDArray[np.float64]: ...


@dataclass(frozen=True)
class NoLeak:
    """The boundary only: no conductance to ground inside."""

    @property
    def name(self) -> str:
        return ""

    def per_cell(self, site: LeakSite) -> NDArray[np.float64]:
        return np.zeros(len(site.area))


def num(x: float) -> str:
    """A number as a name spells it: `:g`, or its repr where `:g` would round it (the name
    parses back to the same float)."""
    g = f"{x:g}"
    return g if float(g) == x else repr(x)


@dataclass(frozen=True, kw_only=True)
class InternalLeak:
    """Destinations inside (module docstring): sigma / open_m^2 per m^2 of the open ground live
    at baseline, and sigma / foot_m^2 per m^2 of a building's footprint where its homes inject;
    a length inf: none of that leak."""
    open_m: float
    foot_m: float

    def __post_init__(self) -> None:
        if not (self.open_m > 0.0 and self.foot_m > 0.0
                and min(self.open_m, self.foot_m) < math.inf):
            raise ValueError(f"an internal leak's lengths are positive, one finite: open_m "
                             f"{self.open_m}, foot_m {self.foot_m}")

    @property
    def name(self) -> str:
        return ("L" + (f"o{num(self.open_m)}" if self.open_m < math.inf else "")
                + (f"b{num(self.foot_m)}" if self.foot_m < math.inf else ""))

    def per_foot(self, site: LeakSite) -> float:
        """The buildings' leak per m^2 of footprint: sigma / foot_m^2 (a home's own building's
        destination drains it times the home's footprint and its potential where it injects:
        region_leaks' self-trips)."""
        return site.sigma / self.foot_m ** 2

    def per_cell(self, site: LeakSite) -> NDArray[np.float64]:
        open_ground = np.where(site.live, site.sigma * site.area / self.open_m ** 2, 0.0)
        buildings = np.where(site.live, site.sigma * site.foot / self.foot_m ** 2, 0.0)
        return open_ground + buildings


# ------------------------------------------------------------------ barriers

class BarrierParts(NamedTuple):
    """A region's barriers (OsmBarriers.parts), in the block's CRS: `barrier` what is cut from the
    region (the strips and areas less the crossings' gaps), `strips` and `areas` before the gaps,
    `gaps` the crossings' gaps, `crossings` each crossing's way piece (bridges and tunnels whole,
    a crossing node's piece within line_m of it) and its kind (`bridge`, `tunnel`, `node`)."""
    barrier: Any
    strips: Any
    areas: Any
    gaps: Any
    crossings: list[tuple[str, Any]]


class Barriers(Protocol):
    """Obstacles never cleared (module docstring): its name's part and the block its mesh is
    built on."""

    @property
    def name(self) -> str: ...

    def block(self, block: Block) -> Block: ...


@dataclass(frozen=True)
class NoBarriers:
    @property
    def name(self) -> str:
        return ""

    def block(self, block: Block) -> Block:
        return block


@dataclass(frozen=True, kw_only=True)
class OsmBarriers:
    """The region's railways and water as obstacles, open at its crossings (module docstring):
    `line_m` a track's or a waterway's width, `cross_m` a crossing's."""
    line_m: float
    cross_m: float

    @property
    def name(self) -> str:
        return f"B{self.line_m:g}c{self.cross_m:g}"

    def parts(self, block: Block) -> BarrierParts:
        """The barriers of `block`, a region (its attrs' `osm`: regions.Osm; a block without one
        raises)."""
        osm = osm_in(block)
        ways, nodes, areas = osm.ways, osm.nodes, osm.areas
        lines = ways[ways.railway.isin(RAIL) | (ways.waterway.isin(WATERWAYS) & ~ways.tunnel)]
        strips = shapely.union_all(shapely.buffer(lines.geometry.to_numpy(), self.line_m / 2.0,
                                                  cap_style="flat"))
        water = areas[areas.natural.isin(WATER_NATURAL) | areas.landuse.isin(WATER_LANDUSE)
                      | areas.waterway.isin(WATER_WATERWAY)]
        wet = shapely.union_all(shapely.make_valid(water.geometry.to_numpy()))
        whole = shapely.union(strips, wet)
        walk = ways[ways.highway != ""]
        crossings: list[tuple[str, Any]] = []
        over = walk[(walk.bridge | walk.tunnel)
                    & shapely.intersects(walk.geometry.to_numpy(), whole)]
        for g, bridge in zip(over.geometry, over.bridge, strict=True):
            crossings.append(("bridge" if bridge else "tunnel", g))
        walk_g = walk.geometry.to_numpy()
        rail_g = ways[ways.railway.isin(RAIL)].geometry.to_numpy()
        for node in nodes[nodes.railway.isin(CROSSING_NODES)].geometry.to_numpy():
            through = walk_g[shapely.dwithin(walk_g, node, NODE_TOL_M)]
            if len(through):
                crossings.append(("node", shapely.intersection(
                    shapely.union_all(through), shapely.buffer(node, self.line_m))))
                continue
            # a crossing tagged on the track whose way the extract lacks (38366's two): the
            # crossing straight across the track at the node, within line_m of it
            track = rail_g[shapely.dwithin(rail_g, node, NODE_TOL_M)]
            if not len(track):
                raise ValueError(f"{block.block_id}: a railway crossing node at {node} on no way "
                                 "and no track")
            crossings.append(("node-across", _across(track[0], node, self.line_m)))
        gaps = shapely.union_all(shapely.buffer(np.asarray([g for _k, g in crossings],
                                                           dtype=object), self.cross_m / 2.0))
        return BarrierParts(barrier=shapely.difference(whole, gaps), strips=strips, areas=wet,
                            gaps=gaps, crossings=crossings)

    def block(self, block: Block) -> Block:
        """`block` with its barriers cut out of its polygon (its streets as they are: no face of
        the mesh lies on the street line within a barrier, so the edge there is no exit); a block
        no barrier meets is itself, polygon and all (a difference with nothing would reorder its
        rings: another mesh key, and a mesh the per-block metric's only to floating point)."""
        barrier = self.parts(block).barrier
        if not shapely.intersects(block.boundary, barrier):
            return block
        cut = shapely.make_valid(shapely.difference(block.boundary, barrier))
        polys = [p for p in shapely.get_parts(cut)
                 if shapely.get_type_id(p) == 3 and shapely.area(p) > 0]
        return dataclasses.replace(block, boundary=shapely.multipolygons(polys)
                                   if len(polys) > 1 else polys[0])


def _across(track: Any, node: Any, half: float) -> Any:
    """The segment 2 `half` long across `track` (a line) at `node`, perpendicular to it there:
    a crossing tagged on a track with no way through it."""
    s = shapely.line_locate_point(track, node)
    a = np.asarray(shapely.get_coordinates(shapely.line_interpolate_point(
        track, max(s - 0.5, 0.0))))
    b = np.asarray(shapely.get_coordinates(shapely.line_interpolate_point(
        track, min(s + 0.5, shapely.length(track)))))
    t = (b - a)[0]
    n = np.array([-t[1], t[0]]) / float(np.hypot(t[0], t[1]))
    p = np.asarray(shapely.get_coordinates(node))[0]
    return shapely.LineString([p - half * n, p + half * n])


# ------------------------------------------------------------------ the street boost

class Boost(Protocol):
    """The sheet conductance's boost on the OSM ways (module docstring)."""

    @property
    def name(self) -> str: ...

    def mesh(self, fm: gap_clear.FullMesh, geo: gc.Geometry,
             block: Block) -> gap_clear.FullMesh: ...


@dataclass(frozen=True)
class NoBoost:
    @property
    def name(self) -> str:
        return ""

    def mesh(self, fm: gap_clear.FullMesh, geo: gc.Geometry,
             block: Block) -> gap_clear.FullMesh:
        return fm


@dataclass(frozen=True, kw_only=True)
class StreetBoost:
    """sigma x `factor` on the triangles on an OSM way of one of `classes` (module docstring:
    highway tags, sorted and distinct, `_link` folded)."""
    factor: float
    classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not (self.factor > 0.0 and self.classes
                and self.classes == tuple(sorted(set(self.classes)))
                and all(re.fullmatch(r"[a-z_]+", c) for c in self.classes)):
            raise ValueError(f"a boost's factor positive and its highway classes sorted, "
                             f"distinct, [a-z_]+: {self.factor}, {self.classes}")

    @property
    def name(self) -> str:
        return f"S{num(self.factor)}h{'-'.join(self.classes)}"

    def on_way(self, fm: gap_clear.FullMesh, geo: gc.Geometry, block: Block) -> NDArray[np.bool_]:
        """Per triangle of `fm`: its centroid within BOOST_M of a way of `block` of one of the
        classes (its attrs' `osm`; a block without one raises)."""
        ways = osm_in(block).ways
        of = ways.highway.str.removesuffix("_link").isin(self.classes).to_numpy()
        lines = geo.local(ways.geometry.to_numpy()[of])
        cen = fm.xy[fm.tri].mean(axis=1)
        qi, _qw = shapely.STRtree(lines).query(shapely.points(cen[:, 0], cen[:, 1]),
                                               predicate="dwithin", distance=BOOST_M)
        on = np.zeros(fm.nt, dtype=bool)
        on[qi] = True
        return on

    def mesh(self, fm: gap_clear.FullMesh, geo: gc.Geometry,
             block: Block) -> gap_clear.FullMesh:
        b = np.where(self.on_way(fm, geo, block), self.factor, 1.0)
        hm = 2.0 * b[fm.pa] * b[fm.pb] / (b[fm.pa] + b[fm.pb])
        return dataclasses.replace(fm, T=fm.T * hm, Tf=fm.Tf * b[fm.fa])


# ------------------------------------------------------------------ the edge

class Edge(Protocol):
    """The region's edge (module docstring): its name's part and the mesh it leaves."""

    @property
    def name(self) -> str: ...

    def mesh(self, fm: gap_clear.FullMesh) -> gap_clear.FullMesh: ...


@dataclass(frozen=True)
class OpenEdge:
    @property
    def name(self) -> str:
        return ""

    def mesh(self, fm: gap_clear.FullMesh) -> gap_clear.FullMesh:
        return fm


@dataclass(frozen=True)
class ClosedEdge:
    """No street face grounds: each one's conductance to ground 0 (the reference the leak's
    constants are chosen on)."""

    @property
    def name(self) -> str:
        return "E0"

    def mesh(self, fm: gap_clear.FullMesh) -> gap_clear.FullMesh:
        return dataclasses.replace(fm, Tf=np.zeros_like(fm.Tf))


# ------------------------------------------------------------------ the exits, named

@dataclass(frozen=True, kw_only=True)
class Exits:
    """The four Strategies a metric holds (module docstring), named by `suffix`."""
    barriers: OsmBarriers | NoBarriers
    edge: OpenEdge | ClosedEdge
    leak: NoLeak | InternalLeak
    boost: NoBoost | StreetBoost

    @property
    def suffix(self) -> str:
        """'' for the per-block exits; else `+x<parts>v<EXITS_CODE>`."""
        parts = self.barriers.name + self.edge.name + self.leak.name + self.boost.name
        return f"+x{parts}v{EXITS_CODE}" if parts else ""


def per_block() -> Exits:
    """The block's whole street ring the exit: no barrier, the edge open, no leak, no boost (no
    suffix)."""
    return Exits(barriers=NoBarriers(), edge=OpenEdge(), leak=NoLeak(), boost=NoBoost())


_N = r"[0-9]+(?:\.[0-9]+)?(?:e[-+]?[0-9]+)?"
# the suffix as gap_clear's name grammar embeds it: barriers, edge, leak, boost, digest (8 groups)
SUFFIX_RE = (rf"(?:\+x(?:B({_N})c({_N}))?(E0)?(?:L(?:o({_N}))?(?:b({_N}))?)?"
             rf"(?:S({_N})h([a-z_]+(?:-[a-z_]+)*))?v([0-9a-f]+))?")


def exits_of(groups: tuple[str | None, ...]) -> Exits:
    """The Exits SUFFIX_RE's groups name (none matched: the per-block exits); a digest not this
    code's raises (the rows are another code's)."""
    line, cross, closed, open_m, foot_m, factor, classes, code = groups
    if code is None:
        return per_block()
    if code != EXITS_CODE:
        raise ValueError(f"exits code {code}, but this code's constants digest to {EXITS_CODE} "
                         "(its rows are another code's)")
    barriers: OsmBarriers | NoBarriers = (NoBarriers() if line is None else OsmBarriers(
        line_m=float(line), cross_m=float(str(cross))))
    edge: OpenEdge | ClosedEdge = OpenEdge() if closed is None else ClosedEdge()
    leak: NoLeak | InternalLeak = (NoLeak() if open_m is None and foot_m is None else InternalLeak(
        open_m=math.inf if open_m is None else float(open_m),
        foot_m=math.inf if foot_m is None else float(foot_m)))
    boost: NoBoost | StreetBoost = (NoBoost() if factor is None else StreetBoost(
        factor=float(factor), classes=tuple(str(classes).split("-"))))
    return Exits(barriers=barriers, edge=edge, leak=leak, boost=boost)


# Exempt from EXITS_CODE, each because it cannot change a result
_EXEMPT = {
    "gap_exits.EXITS_CODE": "the digest itself",
    "gap_exits.SUFFIX_RE": "the name's grammar, not a result's constant",
    "gap_exits.TYPE_CHECKING": "typing's flag, imported: False when run",
}


def exits_constants() -> dict[str, str]:
    """This module's constants (digests.constants_of): what EXITS_CODE digests."""
    return digests.constants_of((sys.modules[__name__],), exempt=_EXEMPT, external={})


EXITS_CODE = digests.digest_of(exits_constants())
