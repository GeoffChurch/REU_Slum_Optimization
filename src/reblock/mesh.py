"""The road-INDEPENDENT half of the permeability mesh.

Nodes are parcel centroids and edges are parcel adjacency -- both functions of parcel geometry
ALONE. Roads never add a node or an edge; they only raise the conductance of edges that already
exist. That is what keeps Rayleigh's nested-edge-set requirement satisfied, and it is the property
three earlier mesh redesigns broke by letting an access edge MOVE when roads were added
(`3a8dd25`, permeability falling ~9%).

Splitting this out means a whole prefix sweep builds it ONCE: it cannot change as roads are added.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely import STRtree
from shapely.geometry import LineString
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency

if TYPE_CHECKING:
    from reblock.permeability import PermeabilityParams

FOOTPATH_EPS = 0.02   # floor on the clearance fraction (an edge never hits 0
                       # conductance, so the mesh graph's topological connectivity to ground
                       # never breaks purely from the footpath model, however tight the gap)


@dataclass(frozen=True)
class Mesh:
    """The road-independent parcel graph: nodes, undirected edges (`rows[k] < cols[k]`, each
    stored once), the footpath conductance and centroid-to-centroid segment of every edge, and
    which parcels are grounded (street-fronting)."""
    cx: NDArray[np.float64]
    cy: NDArray[np.float64]
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    dist: NDArray[np.float64]
    footpath_g: NDArray[np.float64]
    ground: NDArray[np.bool_]
    segments: NDArray[np.object_]   # centroid-to-centroid LineStrings, one per edge
    n: int


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


def footpath_mesh(
    block: Block,
    params: PermeabilityParams,
    *,
    adj: list[set[int]] | None = None,
    radii: NDArray[np.float64] | None = None,
) -> Mesh:
    """Build the road-independent mesh: nodes are parcel centroids, edges are parcel adjacency
    (`parcel_adjacency`, one entry per undirected pair, `i < j`, `dist > 0`), each edge's footpath
    conductance is `_footpath_conductance` over the per-parcel radii (`parcel_radii`), and `ground`
    flags parcels within `STREET_TOL` of the (unioned) street geometry. `dist`/`footpath_g` cover
    the WHOLE mesh regardless of any later road coverage -- `_footpath_conductance`'s
    fair-normalization needs the full distribution (see its docstring).

    `adj` and `radii` let a caller freeze parcel adjacency and per-parcel footprint radii across
    repeated calls on the same block (both are functions of `block` alone, invariant across road
    prefixes); computed internally when omitted.
    """
    parcels = block.parcels
    n = len(parcels)
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

    if dist_arr.size:
        r_sum = radii[rows_arr] + radii[cols_arr]
        footpath_g = _footpath_conductance(dist_arr, r_sum, params.g_walk)
    else:
        footpath_g = np.zeros(0, dtype=np.float64)

    segments = np.array(
        [LineString([(cx[i], cy[i]), (cx[j], cy[j])])
         for i, j in zip(rows_arr.tolist(), cols_arr.tolist(), strict=True)],
        dtype=object)

    return Mesh(cx, cy, rows_arr, cols_arr, dist_arr, footpath_g, ground, segments, n)
