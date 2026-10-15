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
the way's piece within `line_m` of the node), each a gap `cross_m` wide about the way's line. A
barrier changes the mesh: the fixed mesh is built on the cut block, so its cache key (the block's
boundary) and the metric's name (the suffix) both carry it.

LEAKS (`Leak`), conductances to ground on the diagonal of the system, on the cells that are
unknowns at baseline (live and not ground with every footprint standing) and nowhere else, the
same for every state: B's enclosed homes stay enclosed, a cleared building's interior and shy zone
conduct but do not leak, the sparsity pattern is unchanged and P stays convex in the clearing.
`NoLeak`: the boundary only. `InternalLeak(s, lam)`: destinations inside the region, one family
with two parameters (NOTES, "The region's leaks: destinations at buildings and on open ground
(5810@major)"). Its total is sigma |A| / lam^2, |A| the area of the cells live at baseline and
sigma the sheet conductance; a share `s` of it lies at buildings, on the cells where the homes
inject (gap_flow.demand_on's points) in proportion to each cell's population injection at
baseline, and 1 - s on open ground, sigma |c| / lam^2 on every such cell (|c| its area). s = 0 is
the screened Laplacian -div(sigma grad u) + (sigma / lam^2) u = f discretized cell by cell, whose
point source decays as K0(r / lam) (the `leak-scale` guard): for a walker it ends a trip at a
constant rate per unit time anywhere on the open ground, a trip's straight-line length r then
having the density (r / lam^2) K0(r / lam), median 1.257 lam, mean (pi / 2) lam and root mean
square 2 lam. s = 1 puts every destination at the buildings, by area population. At a given lam
the trips are shorter the larger s is: the region's lam is calibrated to a median trip instead
(region_leaks `calibrate`), and recorded in the name.

BOOST (`Boost`): `NoBoost`; `StreetBoost(factor)`: the sheet conductance times `factor` on the
triangles whose centroid lies within BOOST_M of an OSM way's line (a passage by the harmonic mean
of its two triangles' factors, a street face by its triangle's): a check only.

THE NAME. The per-block metric (no barriers, no leak, no boost: the block's whole street ring the
exit) has no suffix, so its name and digest are as before. Any other names its parts in a suffix
`+x[B<line_m>c<cross_m>][L<lam>s<s>][S<factor>]v<EXITS_CODE>` before the search's `@e<eps>`;
EXITS_CODE digests this module's constants (digests.constants_of), so editing them renames only
the metrics that use them. `exits_of` parses a suffix back.

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/gap_exits.py
"""
from __future__ import annotations

import dataclasses
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
    (m^2), `live` (an unknown at baseline: live and not ground, every footprint standing), `inj`
    (the population's injection at baseline: gap_clear.System.Pc @ pop), and the sheet conductance
    `sigma`."""
    area: NDArray[np.float64]
    live: NDArray[np.bool_]
    inj: NDArray[np.float64]
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


@dataclass(frozen=True, kw_only=True)
class InternalLeak:
    """Destinations inside the region (module docstring): the total sigma |A| / lam^2 over the
    cells live at baseline, the share `s` of it at buildings by each cell's population injection
    at baseline and 1 - s on open ground by area. At s = 0 and s = 1 each cell's conductance is
    the other share's term times 0 added to its own: the open ground's alone and the buildings'
    alone, bit for bit (the `leak-fold` guard)."""
    s: float
    lam: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.s <= 1.0 and self.lam > 0.0):
            raise ValueError(f"a leak's building share s in [0, 1] and lam > 0: s {self.s}, "
                             f"lam {self.lam}")

    @property
    def name(self) -> str:
        return f"L{self.lam:g}s{self.s:g}"

    def total(self, site: LeakSite) -> float:
        """sigma |A| / lam^2, |A| the cells' area live at baseline."""
        return float(site.sigma * site.area[site.live].sum() / self.lam ** 2)

    def kappa(self, site: LeakSite) -> float:
        """The buildings' share's conductance per unit of population injection: s times the total
        over the injection on the cells live at baseline (a home's own building's destination
        drains kappa pop_h times its potential where it injects: region_leaks' self-trips)."""
        return self.s * self.total(site) / float(np.where(site.live, site.inj, 0.0).sum())

    def per_cell(self, site: LeakSite) -> NDArray[np.float64]:
        open_ground = np.where(site.live, site.sigma * site.area / self.lam ** 2, 0.0)
        inj = np.where(site.live, site.inj, 0.0)
        buildings = self.total(site) * inj / float(inj.sum())
        return (1.0 - self.s) * open_ground + self.s * buildings


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
        for node in nodes[nodes.railway.isin(CROSSING_NODES)].geometry.to_numpy():
            through = walk_g[shapely.dwithin(walk_g, node, NODE_TOL_M)]
            if not len(through):
                raise ValueError(f"{block.block_id}: a railway crossing node at {node} with no "
                                 "way through it")
            crossings.append(("node", shapely.intersection(shapely.union_all(through),
                                                           shapely.buffer(node, self.line_m))))
        gaps = shapely.union_all(shapely.buffer(np.asarray([g for _k, g in crossings],
                                                           dtype=object), self.cross_m / 2.0))
        return BarrierParts(barrier=shapely.difference(whole, gaps), strips=strips, areas=wet,
                            gaps=gaps, crossings=crossings)

    def block(self, block: Block) -> Block:
        """`block` with its barriers cut out of its polygon (its streets as they are: no face of
        the mesh lies on the street line within a barrier, so the edge there is no exit)."""
        cut = shapely.make_valid(shapely.difference(block.boundary, self.parts(block).barrier))
        polys = [p for p in shapely.get_parts(cut)
                 if shapely.get_type_id(p) == 3 and shapely.area(p) > 0]
        return dataclasses.replace(block, boundary=shapely.multipolygons(polys)
                                   if len(polys) > 1 else polys[0])


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
    """sigma x `factor` on the triangles on an OSM way (module docstring)."""
    factor: float

    @property
    def name(self) -> str:
        return f"S{self.factor:g}"

    def on_way(self, fm: gap_clear.FullMesh, geo: gc.Geometry, block: Block) -> NDArray[np.bool_]:
        """Per triangle of `fm`: its centroid within BOOST_M of a way of `block` (its attrs'
        `osm`; a block without one raises)."""
        ways = osm_in(block).ways
        lines = geo.local(ways.geometry.to_numpy()[(ways.highway != "").to_numpy()])
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


# ------------------------------------------------------------------ the exits, named

@dataclass(frozen=True, kw_only=True)
class Exits:
    """The three Strategies a metric holds (module docstring), named by `suffix`."""
    barriers: OsmBarriers | NoBarriers
    leak: NoLeak | InternalLeak
    boost: NoBoost | StreetBoost

    @property
    def suffix(self) -> str:
        """'' for the per-block exits; else `+x<parts>v<EXITS_CODE>`."""
        parts = self.barriers.name + self.leak.name + self.boost.name
        return f"+x{parts}v{EXITS_CODE}" if parts else ""


def per_block() -> Exits:
    """The block's whole street ring the exit: no barrier, no leak, no boost (no suffix)."""
    return Exits(barriers=NoBarriers(), leak=NoLeak(), boost=NoBoost())


_N = r"[0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?"
# the suffix as gap_clear's name grammar embeds it: barriers, leak, boost, digest (6 groups)
SUFFIX_RE = rf"(?:\+x(?:B({_N})c({_N}))?(?:L({_N})s({_N}))?(?:S({_N}))?v([0-9a-f]+))?"


def exits_of(groups: tuple[str | None, ...]) -> Exits:
    """The Exits SUFFIX_RE's groups name (none matched: the per-block exits); a digest not this
    code's raises (the rows are another code's)."""
    line, cross, lam, share, factor, code = groups
    if code is None:
        return per_block()
    if code != EXITS_CODE:
        raise ValueError(f"exits code {code}, but this code's constants digest to {EXITS_CODE} "
                         "(its rows are another code's)")
    barriers: OsmBarriers | NoBarriers = (NoBarriers() if line is None else OsmBarriers(
        line_m=float(line), cross_m=float(str(cross))))
    leak: NoLeak | InternalLeak = (NoLeak() if lam is None else InternalLeak(
        s=float(str(share)), lam=float(lam)))
    boost: NoBoost | StreetBoost = (NoBoost() if factor is None
                                    else StreetBoost(factor=float(factor)))
    return Exits(barriers=barriers, leak=leak, boost=boost)


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
