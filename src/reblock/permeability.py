"""Permeability: collective egress modeled as an electrical flow on the parcel graph.

Every parcel injects 1 unit of "escape current"; the existing street is ground (potential
0). The metric is total dissipated power P = b^T L^-1 b (b = ones over parcels), normalized
as permeability = 1 - P(roads) / P(no_roads). Lower P => easier collective egress. Roads only
ADD conductance (never remove any), so P is monotone non-increasing in the road set by
construction => permeability is monotone non-decreasing.

Graph model (nodes = parcel centroids; ground = eliminated node at potential 0):
  - footpath mesh (always present): adjacent parcel pairs (parcel_adjacency), conductance
    g_walk * max(eps, 1 - 2*r0/dist(centroid_i, centroid_j)) -- an "open-corridor fraction"
    (`_footpath_conductance`) that reads ~1 (fully open) for centroids far apart relative to r0
    and collapses toward the floor `eps` for centroids closer than 2*r0 (a cramped, obstructed
    line between them). r0 (`_adaptive_r0`) is `r0_frac` * median nearest-neighbour distance
    among the block's building points, so the corridor half-width scales with local building
    density rather than a fixed absolute metre count. The raw shape is rescaled per block so its
    MEDIAN over the mesh equals what the plain g_walk/dist model's median would be at the same
    g_walk -- this keeps g_walk playing its original role as the footpath/road BALANCE knob
    (permeability is invariant to scaling every conductance together, but roads are pinned below
    at a fixed g_road/dist, so only the footpath level relative to that fixed road level moves the
    metric) rather than an incidental side effect of the corridor shape's own median. A footpath
    edge's conductance is additionally capped at g_road/dist (its own would-be road upgrade) so
    upgrading an edge to a road never LOWERS its conductance, for every edge, regardless of
    region -- the monotonicity property below needs this to hold unconditionally, not just on the
    calibration region where it happens to hold already without the cap.
  - road upgrade: an adjacency edge is "road-covered" if roads.buffer(corridor_m) intersects
    the centroid-to-centroid segment; a covered edge's conductance is g_road / dist instead
    of the footpath conductance above.
  - ground edges: a parcel within STREET_TOL of the (unioned) street geometry gets a ground
    edge of conductance g_street, folded straight into that parcel's Laplacian diagonal
    (ground is eliminated, never a graph node).

Ported from the validated research prototype (contention_power / p_benefit) -- the graph
assembly and sparse solve are transcribed verbatim; do not re-derive. The r0-corridor footpath
model + its fair-normalization is ported from the scratchpad `conductance_variants.py` /
`combination_experiment.py` / `r0_sweep.py` experiments, which found it massively improves
method discrimination on multiblock_density_compactness (D=10% method-spread ~0.9pts ->
~11.8pts) at g_walk=0.1, r0~3m.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import Block
from reblock.derive.access import STREET_TOL
from reblock.derive.adjacency import parcel_adjacency

if TYPE_CHECKING:
    from reblock.budget import Curve

FOOTPATH_EPS = 0.02   # floor on the r0-corridor open-corridor fraction (an edge never hits 0
                       # conductance, so the mesh graph's topological connectivity to ground
                       # never breaks purely from the footpath model, independent of r0)


@dataclass(frozen=True)
class PermeabilityParams:
    g_walk: float = 0.1
    g_road: float = 20.0
    g_street: float = 20.0
    corridor_m: float = 3.0
    r0_frac: float = 0.55   # r0 = r0_frac * median(building nearest-neighbour distance); see
                             # `_adaptive_r0`. Calibrated so r0 lands ~3m on
                             # multiblock_density_compactness (median NN ~5.2m) -- the flat
                             # plateau (1.5-3.5m all give ~11-11.8pts D=10% spread) of the
                             # r0-corridor discrimination sweep.


def _road_corridor(roads: GeoDataFrame | None, corridor_m: float) -> BaseGeometry | None:
    """Precomputed buffered-road-union geometry, or None if no roads."""
    if roads is None or len(roads) == 0:
        return None
    union = unary_union(list(roads.geometry))
    if union.is_empty:
        return None
    return union.buffer(corridor_m)


def _adaptive_r0(block: Block, params: PermeabilityParams) -> float:
    """r0 (corridor half-width, m) = `r0_frac` * median nearest-neighbour distance among the
    block's building points -- adaptive so the corridor width scales with local building density
    rather than a fixed absolute metre count (portable across regions of very different
    densities; see the module docstring). Fewer than 2 building points (no defined neighbour)
    -> r0 = 0.0, which degenerates the corridor shape to a constant 1 everywhere (no local-
    geometry modulation -- `_footpath_conductance` then reduces to the plain g_walk/dist model),
    a safe fallback for blocks without a building point cloud (e.g. synthetic test fixtures)."""
    from reblock.budget import building_radii  # deferred: avoids the budget<->permeability cycle

    if len(block.building_points) < 2:
        return 0.0
    radii = building_radii(block.building_points, params.corridor_m)
    return params.r0_frac * float(np.median(2.0 * radii))   # building_radii returns NN/2


def _footpath_conductance(dist: NDArray[np.float64], r0: float, g_walk: float,
                          eps: float = FOOTPATH_EPS) -> NDArray[np.float64]:
    """r0-corridor footpath conductance: g_walk * max(eps, 1 - 2*r0/dist), fair-normalized (see
    module docstring) so its median over `dist` equals g_walk * median(1/dist) -- the median the
    plain g_walk/dist baseline would give at the same g_walk. `dist` must cover the WHOLE
    adjacency mesh (every footpath edge, not just currently road-uncovered ones): the mesh -- and
    therefore this normalization -- is a property of the block's parcel geometry alone,
    independent of any one road prefix."""
    if dist.size == 0:
        return np.zeros(0, dtype=np.float64)
    shape = np.maximum(eps, 1.0 - 2.0 * r0 / dist)
    shape_median = float(np.median(shape))
    if shape_median <= 0.0:
        return np.zeros_like(dist)
    target_median = g_walk * float(np.median(1.0 / dist))
    return (target_median / shape_median) * shape


def egress_power(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    adj: list[set[int]] | None = None,
    r0: float | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """P = b^T L^-1 b for the grounded parcel-centroid Laplacian described in the module
    docstring; b = ones(n) (every parcel injects 1 unit of escape current). Also returns the
    per-parcel potentials v (for the heatmap). (+inf, zeros(n)) if no parcel is
    street-fronting (no path to ground at all -- an ungrounded network has no well-defined
    dissipated power for a nonzero current injection). `r0` lets a caller freeze the adaptive
    corridor half-width (`_adaptive_r0`) across repeated calls on the same block (mirrors `adj`);
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
    r0 = r0 if r0 is not None else _adaptive_r0(block, params)

    corridor = _road_corridor(roads, params.corridor_m)

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
    if dist_arr.size:
        road_g = params.g_road / dist_arr
        # monotonicity guard: a footpath edge can never out-conduct the road it would upgrade to,
        # for every edge regardless of region (see module docstring) -- so upgrading uncovered ->
        # covered never LOWERS that edge's conductance.
        footpath_g = np.minimum(_footpath_conductance(dist_arr, r0, params.g_walk), road_g)
        covered = np.zeros(dist_arr.size, dtype=bool)
        if corridor is not None:
            covered = np.array(
                [corridor.intersects(LineString([(cx[i], cy[i]), (cx[j], cy[j])]))
                 for i, j in zip(rows_arr.tolist(), cols_arr.tolist(), strict=True)],
                dtype=bool)
        conds_arr = np.where(covered, road_g, footpath_g)
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
    v = spsolve(cast(csr_matrix, lap), b)
    if not np.all(np.isfinite(v)):
        return float("inf"), np.zeros(n, dtype=np.float64)
    p = float(b @ v)
    return p, cast(NDArray[np.float64], v)


def permeability(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    p0: float | None = None,
    adj: list[set[int]] | None = None,
    r0: float | None = None,
) -> float:
    """1 - P(roads)/P(no_roads); p0 lets a caller freeze the no-roads baseline (avoids
    recomputing it inside a sweep). `r0` likewise lets a caller freeze the adaptive corridor
    half-width (mirrors `adj`); computed internally when omitted. Guards: no-roads baseline that
    is non-finite or <= 0 (ungrounded block) -> nan; a roaded network that comes out
    ungrounded/non-finite is not reachable in practice (roads only add ground/conductance) but is
    guarded defensively via the same non-finite check on p1."""
    if r0 is None:
        r0 = _adaptive_r0(block, params)
    if p0 is None:
        p0, _ = egress_power(block, None, params, adj=adj, r0=r0)
    if not np.isfinite(p0) or p0 <= 0.0:
        return float("nan")
    p1, _ = egress_power(block, roads, params, adj=adj, r0=r0)
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
    r0 = _adaptive_r0(block, params)
    p0, _ = egress_power(block, None, params, adj=adj, r0=r0)
    calls = 0

    def f(prefix: GeoDataFrame | None) -> float:
        nonlocal calls
        calls += 1
        value = permeability(block, prefix, params, p0=p0, adj=adj, r0=r0)
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
