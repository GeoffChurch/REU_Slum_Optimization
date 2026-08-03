"""Permeability: collective egress modeled as an electrical flow on the parcel graph.

Every parcel injects 1 unit of "escape current"; the existing street is ground (potential
0). The metric is total dissipated power P = b^T L^-1 b (b = ones over parcels), normalized
as permeability = 1 - P(roads) / P(no_roads). Lower P => easier collective egress. Roads only
ADD conductance (never remove any), so P is monotone non-increasing in the road set by
construction => permeability is monotone non-decreasing.

Graph model (nodes = parcel centroids; ground = eliminated node at potential 0):
  - footpath mesh (always present): adjacent parcel pairs (parcel_adjacency), conductance
    g_walk * max(eps, (dist - r_i - r_j)/dist) -- the CLEARANCE fraction
    (`_footpath_conductance`):
    the share of the centroid-to-centroid line lying in neither building, i.e. the gap at the point
    where the two footprints come closest. r_i is parcel i's own footprint radius (`parcel_radii`,
    half the building's nearest-neighbour distance -- the same disk `displacement` charges for), so
    the estimate is local. It reads ~1 for footprints far apart relative to their size and collapses
    toward the floor `eps` when the two disks nearly touch.

    This replaced a single block-median corridor half-width (`r0 = r0_frac * median NN distance`,
    shape `1 - 2*r0/dist`), which is the same quantity with both radii averaged away -- so a
    mixed-density block assumed one gap in its packed core and at its sparse edge alike.

    The raw shape is rescaled per block so its
    MEDIAN over the mesh equals what the plain g_walk/dist model's median would be at the same
    g_walk -- this keeps g_walk playing its original role as the footpath/road BALANCE knob
    (permeability is invariant to scaling every conductance together, but a road of a given width
    is pinned at its own g/dist, so only the footpath level relative to that road level moves the
    metric) rather than an incidental side effect of the corridor shape's own median.
  - road upgrade: an adjacency edge is "road-covered" if a road's own buffer(width_m/2) intersects
    the centroid-to-centroid segment; a covered edge takes `max(footpath, road)` per direction,
    where the road term is `road_conductance` at the width that road gives that direction
    (`edge_conductances`). Roads carry their own `width_m` -- there is no global corridor width and
    no default; a road set without one is an error.

    The `max` is what makes monotonicity structural: an upgrade can never lower an edge, for every
    edge and every region, so no clamp is needed. An earlier model had the road REPLACE the
    footpath outright and therefore had to cap every footpath edge at its own would-be upgrade;
    that cap is gone. Measured on 19,023 mesh edges across 60 real blocks, it never once fired
    (`notes/2026-07-31-width-is-per-road.md`), so removing it left every published number unchanged.
  - ground edges: a parcel within STREET_TOL of the (unioned) street geometry gets a ground
    edge of conductance g_street, folded straight into that parcel's Laplacian diagonal
    (ground is eliminated, never a graph node).

Ported from the validated research prototype (contention_power / p_benefit) -- the graph
assembly and sparse solve are transcribed verbatim; do not re-derive. The corridor footpath model
+ its fair-normalization is ported from the scratchpad `conductance_variants.py` /
`combination_experiment.py` / `r0_sweep.py` experiments, which found it massively improves
method discrimination on multiblock_density_compactness (D=10% method-spread ~0.9pts ->
~11.8pts) at g_walk=0.1; those swept the block-median r0 that the per-pair clearance replaced.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import shapely
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from shapely import STRtree
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency

if TYPE_CHECKING:
    from reblock.budget import Curve

FOOTPATH_EPS = 0.02   # floor on the clearance fraction (an edge never hits 0
                       # conductance, so the mesh graph's topological connectivity to ground
                       # never breaks purely from the footpath model, however tight the gap)


@dataclass(frozen=True)
class PermeabilityParams:
    g_walk: float = 0.1
    # Conductance per METRE of usable width, per unit distance. Replaces the old `g_road` ("the
    # conductance of a standard road"), which only meant something relative to a standard width --
    # and so needed a global `corridor_m` that every road silently inherited. With width carried per
    # road there is no standard, so the parameter is per-metre and no reference width exists.
    #
    # The CALIBRATED quantity is the conductance of one lane, 20.0 -- that is what was tuned against
    # g_walk for method discrimination. So when the lane width was re-based from 2.5 m to 3.0 m
    # (2026-07-31, see min_two_way_width_m), this rate had to fall with it: a wider belief about how
    # much SPACE a lane takes is not a claim that a lane carries more traffic.
    # 20.0 / 3.0 m keeps one lane at exactly 20.0, so a default road's conductance is
    # unchanged by the re-base and only its footprint moved.
    g_road_per_m: float = 6.666666666666667
    g_street: float = 20.0
    # Irreducible margin of a road corridor -- verge, drainage, wall clearance -- paid ONCE
    # regardless of how many lanes it carries. It does two jobs with one number:
    #   * a one-way street is NOT half a two-way one, because both pay the same margin:
    #     W_two = 2*lane + margin, W_one = lane + margin, so W_one/W_two > 1/2;
    #   * usable capacity is (W - margin), so conductance is AFFINE in width, and zero for a road
    #     too narrow to use at all.
    # The two meet: a one-way road matches a two-way's PER-DIRECTION conductance at
    # W_one = (W_two + margin)/2 -- 3.5 m at the defaults, not 3.0. That ">somewhat wider than half"
    # is derived, not asserted.
    # Consequence worth knowing: widening is SUPERLINEAR in capacity (the margin is paid once), so
    # this rewards fewer, wider roads over many narrow ones -- which is what real street hierarchies
    # look like, but it is a behavioural change, not just realism.
    road_margin_m: float = 1.0
    # Narrowest BUILDABLE road, by direction. Conductance is affine in width, which is right ABOVE
    # these floors -- extra width really does buy throughput, because in a dense settlement one
    # parked vehicle or vendor otherwise blocks the way outright -- and fiction BELOW them: a road
    # with less than one lane per direction of travel cannot carry that traffic at all. A two-way
    # road needs room for two directions at once, so its floor is the higher one.
    #
    # This is the guard that was missing when a `two_way_3.5` arm -- 2.5 m usable, one car, scored
    # as two 1.25 m lanes running simultaneously -- decided the one-way comparison. See
    # `notes/2026-07-31-one-way-is-dominated.md`.
    #
    # Stated as two widths rather than `margin + k * lane`, because a clear-width minimum is what
    # access standards actually specify, and it keeps an invented lane constant out of the model.
    #
    # RE-BASED 2026-07-31 (owner sign-off) from 3.5/6.0 to 4.0/7.0. The old pair implied a 2.5 m
    # lane, which is barely a light vehicle (~1.8-2.0 m) and leaves nothing for a service vehicle
    # (~2.5 m) plus clearance. Access for fire, ambulance and refuse is the binding constraint in
    # these settlements and is commonly cited at 3.0-4.0 m clear per lane; 3.0 m is the low end of
    # that range and is what these encode. ENGINEERING JUDGEMENT, not a specific jurisdiction's
    # clause -- if a real local standard says otherwise these are one edit away, and both are
    # exposed in conf/permeability.yaml.
    #
    # This is a re-baseline: a buildable two-way street is 7 m, not 6 m, so every method's roads
    # displace more than they used to. Permeability per covered edge is unchanged (see
    # g_road_per_m), so what moved is the COST side only -- the same function, honestly priced.
    min_one_way_width_m: float = 4.0
    min_two_way_width_m: float = 7.0
    # Scales the per-parcel footprint radii the footpath clearance is measured against. 1.0 uses
    # `budget.building_radii` as-is (half the nearest-neighbour distance), which is a geometric
    # fact rather than a tuned constant -- unlike the r0_frac=0.55 this replaces, which existed to
    # size a single block-median corridor. Kept as a knob because a metric change of this kind has
    # to be recalibratable, not because a value other than 1.0 is known to be better.
    radius_frac: float = 1.0


def _road_corridor(roads: GeoDataFrame | None, half_width_m: float) -> BaseGeometry | None:
    """Buffered road-union geometry at an EXPLICIT half-width, or None if no roads.

    Takes the half-width as an argument rather than reading a global: road width is a
    property of each road now, so there is no repo-wide corridor to inherit.
    """
    if roads is None or len(roads) == 0:
        return None
    union = unary_union(list(roads.geometry))
    if union.is_empty:
        return None
    return union.buffer(half_width_m)


def parcel_radii(block: Block, params: PermeabilityParams) -> NDArray[np.float64]:
    """Per-PARCEL footprint radius, in parcel order -- the disk `displacement` already charges for.

    Replaces a block-median corridor half-width (`r0 = r0_frac * median NN distance`). That was one
    number for the whole block, so a mixed-density block got the same assumed gap in its packed core
    and at its sparse edge; this uses each building's own radius, which `budget.building_radii`
    already computes as half its nearest-neighbour distance.

    Parcels are Voronoi cells of the building points, so the correspondence is exactly one point per
    parcel -- but NOT in index order (verified: parcel i does not contain point i), so it is
    resolved
    by containment. A parcel with no contained point (degenerate geometry) gets radius 0, which
    makes
    its edges read as fully open rather than fully blocked -- the same direction the old code failed
    in when a block had too few points to define a neighbour.
    """
    from reblock.budget import building_radii  # deferred: avoids the budget<->permeability cycle

    n = len(block.parcels)
    out = np.zeros(n, dtype=np.float64)
    pts = block.building_points
    if n == 0 or len(pts) < 2:
        return out
    radii = building_radii(pts)
    xy = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])
    hit = STRtree(shapely.points(xy)).query(
        np.asarray(list(block.parcels.geometry), dtype=object), predicate="contains")
    out[hit[0]] = radii[hit[1]]
    return params.radius_frac * out


def _footpath_conductance(dist: NDArray[np.float64], r_sum: NDArray[np.float64], g_walk: float,
                          eps: float = FOOTPATH_EPS) -> NDArray[np.float64]:
    """Footpath conductance from the CLEARANCE between the two footprints an edge runs between:
    `g_walk * max(eps, (dist - r_i - r_j) / dist)`, fair-normalized (see module docstring) so its
    median over `dist` equals g_walk * median(1/dist) -- the median the plain g_walk/dist baseline
    would give at the same g_walk.

    `(dist - r_i - r_j)/dist` is the fraction of the centroid-to-centroid line that is not inside
    either building: the gap at the point where the two disks come closest. The previous form used
    `1 - 2*r0/dist` with a single block-median r0, which is the same quantity with both radii
    replaced by the block median -- so it is this estimator with the local information averaged
    away.

    `dist` must cover the WHOLE adjacency mesh (every footpath edge, not just currently
    road-uncovered ones): the mesh -- and therefore this normalization -- is a property of the
    block's parcel geometry alone, independent of any one road prefix.
    """
    if dist.size == 0:
        return np.zeros(0, dtype=np.float64)
    shape = np.maximum(eps, (dist - r_sum) / dist)
    shape_median = float(np.median(shape))
    if shape_median <= 0.0:
        return np.zeros_like(dist)
    target_median = g_walk * float(np.median(1.0 / dist))
    return (target_median / shape_median) * shape


ONEWAY_COL = "oneway"


WIDTH_COL = "width_m"


DEFAULT_ROAD_WIDTH_M = 7.0
"""Total width of a road a method emits when nothing else specifies one.

Equals `PermeabilityParams.min_two_way_width_m`: the default road is the narrowest street that can
actually carry two directions. Re-based from 6.0 with the floors on 2026-07-31.

Every method carries this as a `road_width_m` field, so it is a default a caller can override --
never a global the metric falls back to. The metric itself has no default at all: roads without a
`width_m` are an error.
"""


def with_width(roads: GeoDataFrame, width_m: float, *, oneway: bool = False) -> GeoDataFrame:
    """Stamp the mandatory `width_m` (and `oneway`) columns on the roads a method emits."""
    out = roads.copy()
    out[WIDTH_COL] = float(width_m)
    out[ONEWAY_COL] = bool(oneway)
    return out


def road_conductance(params: PermeabilityParams, width_for_direction: NDArray[np.float64],
                     dist: NDArray[np.float64]) -> NDArray[np.float64]:
    """Conductance a corridor of total width `width_for_direction` gives over `dist`.

    Affine in width and floored at zero: usable capacity is `(W - road_margin_m)`, because the
    margin (verge, drainage, wall clearance) is consumed before any lane can fit. There is no
    reference width -- `g_road_per_m` is per metre of usable width -- which is what lets the global
    `corridor_m` go away entirely.
    """
    usable = np.maximum(0.0, width_for_direction - params.road_margin_m)
    return params.g_road_per_m * usable / np.maximum(dist, 1e-12)


def buildable_widths(roads: GeoDataFrame, params: PermeabilityParams,
                     ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """`(width_m, oneway)` for every road, after checking each one could actually be built.

    Two refusals, both about scoring something that cannot exist:

    * no `width_m` column at all -- width is mandatory, there is no global corridor to inherit;
    * a width below its direction's floor -- `min_two_way_width_m` for a two-way road (it must fit
      two directions at once), `min_one_way_width_m` for a one-way one.

    The floors matter because conductance is affine in width with no lower knee: without them the
    model happily reports a 3.5 m two-way road as two 1.25 m lanes running side by side, which is
    how an unbuildable road came to decide the one-way comparison. Above the floors the continuum is
    real and stays -- a wider road genuinely carries more, because a single parked vehicle no longer
    blocks it -- so this is a floor, not a quantization.
    """
    if WIDTH_COL not in roads.columns:
        raise ValueError(
            f"Proposal.roads must carry a '{WIDTH_COL}' column: road width is mandatory since the "
            f"global corridor width was removed. Methods set it on the roads they emit.")
    widths = roads[WIDTH_COL].to_numpy(dtype=float)
    oneway = (roads[ONEWAY_COL].to_numpy(dtype=bool) if ONEWAY_COL in roads.columns
              else np.zeros(len(roads), dtype=bool))
    floors = np.where(oneway, params.min_one_way_width_m, params.min_two_way_width_m)
    bad = widths < floors
    if bad.any():
        k = int(np.argmax(bad))
        kind = "one-way" if oneway[k] else "two-way"
        raise ValueError(
            f"road {k} is {widths[k]:g} m, below the {floors[k]:g} m floor for a {kind} road "
            f"({int(bad.sum())} of {len(roads)} roads are too narrow). A {kind} road narrower than "
            f"that cannot carry the traffic the conductance model would credit it with"
            + (" -- a two-way road has to fit both directions at once." if not oneway[k] else "."))
    return widths, oneway


def lane_width(params: PermeabilityParams, width_m: float, *, oneway: bool = False) -> float:
    """Corridor width available to ONE direction of travel on a road of total width `width_m`.

    A two-way road pays the margin once and its lanes split what is left; a one-way road gives the
    whole width to its permitted direction. This is the only place the two-way split is defined.
    """
    return width_m if oneway else params.road_margin_m + (width_m - params.road_margin_m) / 2.0


def edge_conductances(
    cx: NDArray[np.float64], cy: NDArray[np.float64], rows: NDArray[np.int64],
    cols: NDArray[np.int64], dist: NDArray[np.float64], footpath_g: NDArray[np.float64],
    roads: GeoDataFrame | None, params: PermeabilityParams,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-edge (forward, backward) conductance. Roads are DIRECTED options over the footpath.

    Every road carries its own `width_m`; there is no global corridor width to inherit. What a road
    offers a given direction is decided by width, not by a boost factor:

        two-way,  total width W  ->  each direction gets W/2 + margin/2 of corridor, i.e. half the
                                     USABLE width once the once-paid margin is set aside
        one-way,  total width W  ->  the permitted direction gets all of W, the other nothing

    Direction availability is a HARD test (`cos t >= 0`): a road is a 1D curve, so it either carries
    you toward `j` or it does not. A CROSSING edge therefore stays symmetric -- two parcels facing
    each other across a one-way street do not care which way its traffic runs.

    ## Switch, not sum

    A covered edge takes `max(footpath, road)`, NOT `footpath + road`: the corridor IS the road once
    built, so this models replacement. Treating pavement and carriageway as genuinely parallel --
    arguably truer, since you really can walk against a one-way street -- would ADD them, raising
    every covered edge by about `g_walk` and shifting every published number by ~10%. That
    re-baseline is the only reason it is not the default.

    ## Cost

    Queries are per ROAD (tens to hundreds), not per EDGE (tens of thousands): one STRtree over the
    mesh edges, then one query per road. A per-edge loop here would undo the vectorization that made
    the metric 4.4x faster at region scale.

    Each road SEGMENT is buffered separately, where the pre-width model buffered the road union once
    (per-road widths make a single union buffer impossible). Both approximate the same region, but
    `buffer` polygonalizes a circle, and the union's vertices land differently from each segment's,
    so an edge grazing the boundary can flip. MEASURED over 60 real blocks: 1 flip in 19,023 mesh
    edges, on an edge 2.9984 m from the centreline of a 3 m half-width corridor -- 1.6 mm inside,
    which the union buffer cut out and this one (correctly) keeps. It moved that block's
    permeability by 9.4e-10 relative; the other 59 were bit-identical
    (`notes/2026-07-31-width-is-per-road.md`). Everything else about the reduction is exact.

    ## Properties

    * Monotone: roads enter only through a `max` with the footpath, and conductance rises with
      width, so no conductance can ever fall. Monotonicity is load-bearing (module docstring).
    * Symmetric exactly when no road is one-way, so the directed solve is only used when needed.
    """
    gf, gb = footpath_g.copy(), footpath_g.copy()
    if roads is None or len(roads) == 0:
        return gf, gb
    widths, oneway = buildable_widths(roads, params)
    if rows.size == 0:
        return gf, gb

    edge_lines = shapely.linestrings(
        np.column_stack([np.stack([cx[rows], cx[cols]], axis=1).ravel(),
                         np.stack([cy[rows], cy[cols]], axis=1).ravel()]),
        indices=np.repeat(np.arange(rows.size, dtype=np.int64), 2))
    tree = STRtree(edge_lines)
    ev = np.column_stack([cx[cols] - cx[rows], cy[cols] - cy[rows]])
    ev = ev / np.maximum(np.hypot(ev[:, 0], ev[:, 1]), 1e-12)[:, None]

    for k, geom in enumerate(roads.geometry):
        w = float(widths[k])
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            cs = list(part.coords)
            for a, b in zip(cs, cs[1:], strict=False):
                if a == b:
                    continue
                seg = LineString([a, b])
                hit = tree.query(seg.buffer(w / 2.0), predicate="intersects")
                if not len(hit):
                    continue
                if not oneway[k]:
                    g = road_conductance(
                        params, np.full(hit.size, lane_width(params, w)), dist[hit])
                    gf[hit] = np.maximum(gf[hit], g)
                    gb[hit] = np.maximum(gb[hit], g)
                else:
                    d = np.array([b[0] - a[0], b[1] - a[1]], dtype=np.float64)
                    d = d / max(float(np.hypot(d[0], d[1])), 1e-12)
                    c = ev[hit] @ d
                    g = road_conductance(
                        params, np.full(hit.size, lane_width(params, w, oneway=True)), dist[hit])
                    fwd, bwd = hit[c >= 0.0], hit[c <= 0.0]
                    gf[fwd] = np.maximum(gf[fwd], g[c >= 0.0])
                    gb[bwd] = np.maximum(gb[bwd], g[c <= 0.0])
    return gf, gb


def has_oneway(roads: GeoDataFrame | None) -> bool:
    """True iff any road is flagged one-way -- switches the symmetric solve to the directed one."""
    return (roads is not None and len(roads) > 0 and ONEWAY_COL in roads.columns
            and bool(roads[ONEWAY_COL].to_numpy(dtype=bool).any()))


def egress_power(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    adj: list[set[int]] | None = None,
    radii: NDArray[np.float64] | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """P = b^T L^-1 b for the grounded parcel-centroid Laplacian described in the module
    docstring; b = ones(n) (every parcel injects 1 unit of escape current). Also returns the
    per-parcel potentials v (for the heatmap). (+inf, zeros(n)) if no parcel is
    street-fronting (no path to ground at all -- an ungrounded network has no well-defined
    dissipated power for a nonzero current injection). `radii` lets a caller freeze the per-parcel
    footprint radii (`parcel_radii`) across repeated calls on the same block (mirrors `adj`);
    computed internally when omitted."""
    parcels = block.parcels
    n = len(parcels)
    if n == 0:
        return float("inf"), np.zeros(0, dtype=np.float64)
    geoms = list(parcels.geometry)
    centroids = [g.centroid for g in geoms]
    cx = np.array([c.x for c in centroids], dtype=np.float64)
    cy = np.array([c.y for c in centroids], dtype=np.float64)

    adj = adj if adj is not None else parcel_adjacency(geoms, STREET_TOL)
    radii = radii if radii is not None else parcel_radii(block, params)


    # --- ground membership: parcel polygon within STREET_TOL of the (unioned) street geometry
    street_union = unary_union(list(block.streets.geometry)) if len(block.streets) else None
    if street_union is not None and not street_union.is_empty:
        ground = np.array([g.distance(street_union) <= STREET_TOL for g in geoms], dtype=bool)
    else:
        ground = np.zeros(n, dtype=bool)
    if not ground.any():
        return float("inf"), np.zeros(n, dtype=np.float64)

    # --- footpath mesh: (i, j, dist) for every adjacency edge (i < j, dist > 0), over the WHOLE
    # mesh regardless of road coverage -- `_footpath_conductance`'s fair-normalization needs the
    # full dist distribution, not just currently-uncovered edges (see its docstring).
    rows: list[int] = []
    cols: list[int] = []
    dists: list[float] = []
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            dist = float(np.hypot(cx[i] - cx[j], cy[i] - cy[j]))
            if dist <= 0.0:
                continue
            rows.append(i)
            cols.append(j)
            dists.append(dist)

    rows_arr = np.asarray(rows, dtype=np.int64)
    cols_arr = np.asarray(cols, dtype=np.int64)
    dist_arr = np.asarray(dists, dtype=np.float64)

    diag = np.zeros(n, dtype=np.float64)
    gf_arr = gb_arr = np.zeros(0, dtype=np.float64)
    if dist_arr.size:
        # ONE path: per-road widths give (forward, backward) directly, symmetric iff nothing is
        # one-way. The old split -- a vectorized boolean `covered` for the symmetric case and a
        # separate directional branch -- disappeared with the global corridor width, since every
        # edge's road conductance now depends on WHICH road covers it and how wide that road is.
        r_sum = radii[rows_arr] + radii[cols_arr]
        footpath_g = _footpath_conductance(dist_arr, r_sum, params.g_walk)
        gf_arr, gb_arr = edge_conductances(
            cx, cy, rows_arr, cols_arr, dist_arr, footpath_g, roads, params)
        conds_arr = np.maximum(gf_arr, gb_arr)
        np.add.at(diag, rows_arr, conds_arr)
        np.add.at(diag, cols_arr, conds_arr)
    diag[ground] += params.g_street

    # off-diagonal entries: -g_ij at (i,j) and (j,i)
    if dist_arr.size:
        off_rows = np.concatenate([rows_arr, cols_arr])
        off_cols = np.concatenate([cols_arr, rows_arr])
        off_data = np.concatenate([-conds_arr, -conds_arr])
    else:
        off_rows = np.zeros(0, dtype=np.int64)
        off_cols = np.zeros(0, dtype=np.int64)
        off_data = np.zeros(0, dtype=np.float64)

    all_rows = np.concatenate([off_rows, np.arange(n, dtype=np.int64)])
    all_cols = np.concatenate([off_cols, np.arange(n, dtype=np.int64)])
    all_data = np.concatenate([off_data, diag])

    lap = coo_matrix((all_data, (all_rows, all_cols)), shape=(n, n)).tocsr()
    b = np.ones(n, dtype=np.float64)
    if not np.array_equal(gf_arr, gb_arr) if dist_arr.size else False:
        # DIRECTED. Per-edge cost is a convex asymmetric quadratic in the NET flow, so this is an
        # ordinary Laplacian solve whose conductance follows the sign of its own solution, iterated
        # to a fixed point (convex => the fixed point is the optimum). Scored as EGRESS + INGRESS
        # halved: undirected those are identical by reciprocity, so this reduces to the symmetric
        # branch's value exactly, and directed they diverge -- which is the entire point, since an
        # out-tree serves egress perfectly and fails ingress.
        shunt = ground.astype(np.float64) * params.g_street
        pe, v = _directed_power(n, rows_arr, cols_arr, gf_arr, gb_arr, shunt, b)
        pi, _v2 = _directed_power(n, rows_arr, cols_arr, gb_arr, gf_arr, shunt, b)
        if not np.isfinite(pe) or not np.isfinite(pi):
            return float("inf"), np.zeros(n, dtype=np.float64)
        return 0.5 * (pe + pi), v
    v = spsolve(cast(csr_matrix, lap), b)
    if not np.all(np.isfinite(v)):
        return float("inf"), np.zeros(n, dtype=np.float64)
    p = float(b @ v)
    return p, cast(NDArray[np.float64], v)


def _directed_power(
    n: int, rows: NDArray[np.int64], cols: NDArray[np.int64], gf: NDArray[np.float64],
    gb: NDArray[np.float64], ground_term: NDArray[np.float64], b: NDArray[np.float64],
    iters: int = 40,
) -> tuple[float, NDArray[np.float64]]:
    """min-energy flow where each edge's conductance depends on the DIRECTION of flow through it.

    Damped IRLS: solve, read the flow signs, re-weight, repeat. `ground_term` is the per-node ground
    shunt (`g_street` on grounded parcels, 0 elsewhere), passed EXPLICITLY. A first version tried to
    back it out of the assembled diagonal, which was built with `max(gf, gb)` while the iteration
    re-derives degrees from the current `g` -- so the difference between the two aggregations leaked
    into the shunt, inflated conductance to ground, and made one-way roads score BETTER than
    two-way. Restricting a road cannot improve flow; that impossibility is what exposed the bug.
    """
    g = np.sqrt(gf * gb)
    prev: float | None = None
    v = np.zeros(n, dtype=np.float64)
    for _ in range(iters):
        deg = np.zeros(n, dtype=np.float64)
        np.add.at(deg, rows, g)
        np.add.at(deg, cols, g)
        rr = np.concatenate([rows, cols, np.arange(n, dtype=np.int64)])
        cc = np.concatenate([cols, rows, np.arange(n, dtype=np.int64)])
        dd = np.concatenate([-g, -g, deg + ground_term])
        lap = coo_matrix((dd, (rr, cc)), shape=(n, n)).tocsr()
        v = spsolve(cast(csr_matrix, lap), b)
        if not np.all(np.isfinite(v)):
            return float("inf"), np.zeros(n, dtype=np.float64)
        x = g * (v[rows] - v[cols])
        g_new = np.where(x >= 0, gf, gb)
        power = float(np.sum(x * x / g_new) + np.sum(ground_term * v * v))
        if prev is not None and abs(power - prev) <= 1e-12 * max(1.0, abs(prev)):
            return power, cast(NDArray[np.float64], v)
        prev = power
        g = 0.5 * g + 0.5 * g_new
    return (float(prev) if prev is not None else float("inf")), cast(NDArray[np.float64], v)


def permeability(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    p0: float | None = None,
    adj: list[set[int]] | None = None,
    radii: NDArray[np.float64] | None = None,
) -> float:
    """1 - P(roads)/P(no_roads); p0 lets a caller freeze the no-roads baseline (avoids
    recomputing it inside a sweep). `radii` likewise lets a caller freeze the per-parcel
    half-width (mirrors `adj`); computed internally when omitted. Guards: no-roads baseline that
    is non-finite or <= 0 (ungrounded block) -> nan; a roaded network that comes out
    ungrounded/non-finite is not reachable in practice (roads only add ground/conductance) but is
    guarded defensively via the same non-finite check on p1."""
    if radii is None:
        radii = parcel_radii(block, params)
    if p0 is None:
        p0, _ = egress_power(block, None, params, adj=adj, radii=radii)
    if not np.isfinite(p0) or p0 <= 0.0:
        return float("nan")
    p1, _ = egress_power(block, roads, params, adj=adj, radii=radii)
    if not np.isfinite(p1):
        return float("-inf")
    return 1.0 - p1 / p0


def permeability_curve(
    block: Block,
    roads: GeoDataFrame,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    n_points: int = 20,
    tol: float = STREET_TOL,
    progress: Callable[[int, int], None] | None = None,
) -> Curve:
    """A Curve whose x is cumulative added road length (m) and whose y is `permeability` per
    drainage-ordered prefix (monotone non-decreasing; see the module docstring -- roads only add
    conductance, so P is monotone non-increasing under Rayleigh monotonicity and permeability
    monotone non-decreasing). Mirrors `budget.displacement_curve`'s structure: reuses the
    drainage-ordered `_sweep`. The no-roads baseline p0 is computed ONCE via
    `egress_power(block, None, params)` and frozen across every sample (rather than recomputed
    per prefix inside `permeability`). `adj` (parcel_adjacency, an STRtree spatial join -- costly
    at region scale) is likewise built ONCE here and threaded through every `egress_power`/
    `permeability` call: adjacency is a function of `block.parcels` geometry alone, invariant
    across road prefixes, exactly the precomputed-adj pattern `prefix_to_depth` already uses. The
    adaptive corridor half-width r0 (`_adaptive_r0`) is likewise a function of `block` alone
    (independent of road prefix), so it is also computed ONCE here and threaded through.
    Deferred import of `reblock.budget` avoids a module-level import cycle (budget.py
    imports `permeability`/`PermeabilityParams` from this module).

    `progress`, if given, is called `progress(call_index, total_calls)` (1-indexed) after each
    per-prefix `permeability` solve -- `total_calls = n_points + 1` (the no-roads baseline plus
    every sample point `_sweep` requests; a budget step that adds no new road is deduped by
    `_sweep` itself, so the last observed `call_index` can be < `total_calls`). Purely a
    progress-reporting hook for slow, many-region batch callers (e.g.
    `scripts/calibrate_permeability.py`) -- it has no effect on the returned Curve, and `None`
    (the default) adds no overhead."""
    from reblock.budget import Curve, _sweep  # deferred: breaks the budget<->permeability cycle

    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    radii = parcel_radii(block, params)
    p0, _ = egress_power(block, None, params, adj=adj, radii=radii)
    calls = 0

    def f(prefix: GeoDataFrame | None) -> float:
        nonlocal calls
        calls += 1
        value = permeability(block, prefix, params, p0=p0, adj=adj, radii=radii)
        if progress is not None:
            progress(calls, n_points + 1)
        return value

    costs, vals = _sweep(block, roads, f, n_points, tol)
    return Curve(costs, vals)


def parcel_potentials(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
) -> pd.Series:
    """Per-parcel potentials v (grounded egress-flow solve under `roads`), indexed by
    `parcel_id` -- feeds the `_perm` heatmap coloring."""
    _, v = egress_power(block, roads, params)
    return pd.Series(v, index=pd.Index(block.parcels["parcel_id"], name="parcel_id"))
