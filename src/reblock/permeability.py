"""Permeability: collective egress modeled as an electrical flow on the parcel graph.

Every parcel injects 1 unit of "escape current"; the existing street is ground (potential
0). The metric is total dissipated power P = b^T L^-1 b (b = ones over parcels), normalized
as permeability = 1 - P(roads) / P(no_roads). Lower P => easier collective egress. Roads only
ADD conductance (never remove any), so P is monotone non-increasing in the road set by
construction => permeability is monotone non-decreasing.

Graph model (nodes = parcel centroids; ground = eliminated node at potential 0):
  - footpath mesh (always present): adjacent parcel pairs (parcel_adjacency), conductance
    g_walk / dist(centroid_i, centroid_j).
  - road upgrade: an adjacency edge is "road-covered" if roads.buffer(corridor_m) intersects
    the centroid-to-centroid segment; a covered edge's conductance is g_road / dist instead
    of g_walk / dist.
  - ground edges: a parcel within STREET_TOL of the (unioned) street geometry gets a ground
    edge of conductance g_street, folded straight into that parcel's Laplacian diagonal
    (ground is eliminated, never a graph node).

Ported from the validated research prototype (contention_power / p_benefit) -- the graph
assembly and sparse solve are transcribed verbatim; do not re-derive.
"""
from __future__ import annotations

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


@dataclass(frozen=True)
class PermeabilityParams:
    g_walk: float = 1.0
    g_road: float = 20.0
    g_street: float = 20.0
    corridor_m: float = 3.0


def _road_corridor(roads: GeoDataFrame | None, corridor_m: float) -> BaseGeometry | None:
    """Precomputed buffered-road-union geometry, or None if no roads."""
    if roads is None or len(roads) == 0:
        return None
    union = unary_union(list(roads.geometry))
    if union.is_empty:
        return None
    return union.buffer(corridor_m)


def egress_power(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    adj: list[set[int]] | None = None,
) -> tuple[float, NDArray[np.float64]]:
    """P = b^T L^-1 b for the grounded parcel-centroid Laplacian described in the module
    docstring; b = ones(n) (every parcel injects 1 unit of escape current). Also returns the
    per-parcel potentials v (for the heatmap). (+inf, zeros(n)) if no parcel is
    street-fronting (no path to ground at all -- an ungrounded network has no well-defined
    dissipated power for a nonzero current injection)."""
    parcels = block.parcels
    n = len(parcels)
    if n == 0:
        return float("inf"), np.zeros(0, dtype=np.float64)
    geoms = list(parcels.geometry)
    centroids = [g.centroid for g in geoms]
    cx = np.array([c.x for c in centroids], dtype=np.float64)
    cy = np.array([c.y for c in centroids], dtype=np.float64)

    adj = adj if adj is not None else parcel_adjacency(geoms, STREET_TOL)

    corridor = _road_corridor(roads, params.corridor_m)

    # --- ground membership: parcel polygon within STREET_TOL of the (unioned) street geometry
    street_union = unary_union(list(block.streets.geometry)) if len(block.streets) else None
    if street_union is not None and not street_union.is_empty:
        ground = np.array([g.distance(street_union) <= STREET_TOL for g in geoms], dtype=bool)
    else:
        ground = np.zeros(n, dtype=bool)
    if not ground.any():
        return float("inf"), np.zeros(n, dtype=np.float64)

    # --- footpath mesh + road upgrade edges (i < j, deduped via adjacency's set structure)
    rows: list[int] = []
    cols: list[int] = []
    conds: list[float] = []
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            dist = float(np.hypot(cx[i] - cx[j], cy[i] - cy[j]))
            if dist <= 0.0:
                continue
            covered = False
            if corridor is not None:
                seg = LineString([(cx[i], cy[i]), (cx[j], cy[j])])
                covered = corridor.intersects(seg)
            g_edge = params.g_road if covered else params.g_walk
            g_ij = g_edge / dist
            rows.append(i)
            cols.append(j)
            conds.append(g_ij)

    diag = np.zeros(n, dtype=np.float64)
    if conds:
        conds_arr = np.asarray(conds, dtype=np.float64)
        rows_arr = np.asarray(rows, dtype=np.int64)
        cols_arr = np.asarray(cols, dtype=np.int64)
        np.add.at(diag, rows_arr, conds_arr)
        np.add.at(diag, cols_arr, conds_arr)
    diag[ground] += params.g_street

    # off-diagonal entries: -g_ij at (i,j) and (j,i)
    if conds:
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
) -> float:
    """1 - P(roads)/P(no_roads); p0 lets a caller freeze the no-roads baseline (avoids
    recomputing it inside a sweep). Guards: no-roads baseline that is non-finite or <= 0
    (ungrounded block) -> nan; a roaded network that comes out ungrounded/non-finite is not
    reachable in practice (roads only add ground/conductance) but is guarded defensively via
    the same non-finite check on p1."""
    if p0 is None:
        p0, _ = egress_power(block, None, params, adj=adj)
    if not np.isfinite(p0) or p0 <= 0.0:
        return float("nan")
    p1, _ = egress_power(block, roads, params, adj=adj)
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
    across road prefixes, exactly the precomputed-adj pattern `access_benefit`/`prefix_to_depth`
    already use. Deferred import of `reblock.budget` avoids a module-level import cycle (budget.py
    imports `permeability`/`PermeabilityParams` from this module)."""
    from reblock.budget import Curve, _sweep  # deferred: breaks the budget<->permeability cycle

    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    p0, _ = egress_power(block, None, params, adj=adj)

    def f(prefix: GeoDataFrame | None) -> float:
        return permeability(block, prefix, params, p0=p0, adj=adj)

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
