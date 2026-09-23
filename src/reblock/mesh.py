"""The road-INDEPENDENT half of the permeability mesh.

Nodes are parcel centroids and edges are parcel adjacency -- both functions of parcel geometry
ALONE. Roads never add a node or an edge; they only raise the conductance of edges that already
exist. That is what keeps Rayleigh's nested-edge-set requirement satisfied, and it is the property
three earlier mesh redesigns broke by letting an access edge MOVE when roads were added
(`3a8dd25`, permeability falling ~9%).

Splitting this out means a whole prefix sweep builds it ONCE: it cannot change as roads are added.
`permeability.EgressContext.mesh` is where that once lives.
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

if TYPE_CHECKING:
    from reblock.permeability import EgressContext, PermeabilityParams

FOOTPATH_EPS = 0.02   # floor on the clearance fraction (an edge never hits 0
                       # conductance, so the mesh graph's topological connectivity to ground
                       # never breaks purely from the footpath model, however tight the gap)


@dataclass(frozen=True)
class Mesh:
    """The road-independent parcel graph: nodes, undirected edges (`rows[k] < cols[k]`, each
    stored once), the footpath conductance and centroid-to-centroid segment of every edge, and
    which parcels are grounded (street-fronting).

    Every array is READ-ONLY. One mesh serves every solve and every figure on its block
    (`EgressContext.mesh`), and `GraphFigure` hands these same arrays on to renderers and bakers --
    so an in-place edit anywhere downstream would silently move every later solve on the block.
    Read-only turns that into an error at the edit."""
    cx: NDArray[np.float64]
    cy: NDArray[np.float64]
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    dist: NDArray[np.float64]
    footpath_g: NDArray[np.float64]
    ground: NDArray[np.bool_]
    segments: NDArray[np.object_]   # centroid-to-centroid LineStrings, one per edge
    n: int

    def __post_init__(self) -> None:
        for a in (self.cx, self.cy, self.rows, self.cols, self.dist, self.footpath_g,
                  self.ground, self.segments):
            a.flags.writeable = False


def parcel_radii(block: Block, params: PermeabilityParams) -> NDArray[np.float64]:
    """Per-PARCEL footprint radius, in parcel order -- the disk `displacement` already charges for.

    Replaces a block-median corridor half-width (`r0 = r0_frac * median NN distance`). That was one
    number for the whole block, so a mixed-density block got the same assumed gap in its packed core
    and at its sparse edge; this uses each building's own radius, which `buildings.SpacingDiscs`
    already computes as half its nearest-neighbour distance.

    Parcels are Voronoi cells of the building points, so the correspondence is exactly one point per
    parcel -- but NOT in index order (verified: parcel i does not contain point i), so it is
    resolved
    by containment. A parcel with no contained point (degenerate geometry) gets radius 0, which
    makes
    its edges read as fully open rather than fully blocked -- the same direction the old code failed
    in when a block had too few points to define a neighbour.
    """

    n = len(block.parcels)
    out = np.zeros(n, dtype=np.float64)
    buildings = block.buildings
    if n == 0 or len(buildings) < 2:
        return out
    # The tier's POINTS, never the geometry column: at the footprint tier that holds polygons, and
    # `xy` is the anchor the parcels were tessellated on -- so each lands in its own parcel.
    hit = STRtree(shapely.points(buildings.xy)).query(
        np.asarray(list(block.parcels.geometry), dtype=object), predicate="contains")
    out[hit[0]] = buildings.radii[hit[1]]
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


def footpath_mesh(ctx: EgressContext) -> Mesh:
    """Build the road-independent mesh for `ctx.block` under `ctx.params`: nodes are parcel
    centroids, edges are the context's parcel adjacency (one entry per undirected pair, `i < j`,
    `dist > 0`), each edge's footpath conductance is `_footpath_conductance` over the context's
    per-parcel radii (`parcel_radii`), and `ground` flags parcels within `STREET_TOL` of the
    (unioned) street geometry. `dist`/`footpath_g` cover the WHOLE mesh regardless of any later road
    coverage -- `_footpath_conductance`'s fair-normalization needs the full distribution (see its
    docstring).

    What `EgressContext.mesh` caches; read that rather than calling this, which builds a new one.
    """
    block, params = ctx.block, ctx.params
    parcels = block.parcels
    n = len(parcels)
    geoms = list(parcels.geometry)
    centroids = [g.centroid for g in geoms]
    cx = np.array([c.x for c in centroids], dtype=np.float64)
    cy = np.array([c.y for c in centroids], dtype=np.float64)

    adj = ctx.adjacency.neighbours
    radii = ctx.radii

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
