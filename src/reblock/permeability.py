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
    the centroid-to-centroid segment; a covered edge takes `max(footpath, road)`, where the road
    term is `road_conductance` at that road's width (`edge_conductances`). Roads carry their own
    `width_m` -- there is no global corridor width and no default; a road set without one is an
    error.

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
from reblock.mesh import Mesh, footpath_mesh
from reblock.mesh import _footpath_conductance as _footpath_conductance
from reblock.mesh import parcel_radii as parcel_radii

if TYPE_CHECKING:
    from reblock.budget import Curve


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
    # (2026-07-31, see min_road_width_m), this rate had to fall with it: a wider belief about how
    # much SPACE a lane takes is not a claim that a lane carries more traffic.
    # 20.0 / 3.0 m keeps one lane at exactly 20.0, so a default road's conductance is
    # unchanged by the re-base and only its footprint moved.
    g_road_per_m: float = 6.666666666666667
    g_street: float = 20.0
    # Irreducible margin of a road corridor -- verge, drainage, wall clearance -- paid ONCE
    # regardless of how many lanes it carries. Usable capacity is therefore (W - margin), which
    # makes conductance AFFINE in width and zero for a road too narrow to use at all.
    # Consequence worth knowing: widening is SUPERLINEAR in capacity (the margin is paid once), so
    # this rewards fewer, wider roads over many narrow ones -- which is what real street hierarchies
    # look like, but it is a behavioural change, not just realism.
    road_margin_m: float = 1.0
    # Narrowest BUILDABLE road. Conductance is affine in width, which is right ABOVE this floor --
    # extra width really does buy throughput, because in a dense settlement one parked vehicle or
    # vendor otherwise blocks the way outright -- and fiction BELOW it: a road with less than one
    # lane per direction of travel cannot carry that traffic at all, and a road has to fit both
    # directions at once.
    #
    # This is the guard that was missing when a 3.5 m road -- 2.5 m usable, one car, scored as two
    # 1.25 m lanes running simultaneously -- decided a comparison it had no business deciding.
    #
    # Stated as a clear width rather than `margin + k * lane`, because a clear-width minimum is what
    # access standards actually specify, and it keeps an invented lane constant out of the model.
    #
    # RE-BASED 2026-07-31 (owner sign-off) from 6.0 to 7.0. The old value implied a 2.5 m lane,
    # which is barely a light vehicle (~1.8-2.0 m) and leaves nothing for a service vehicle
    # (~2.5 m) plus clearance. Access for fire, ambulance and refuse is the binding constraint in
    # these settlements and is commonly cited at 3.0-4.0 m clear per lane; 3.0 m is the low end of
    # that range and is what this encodes. ENGINEERING JUDGEMENT, not a specific jurisdiction's
    # clause -- if a real local standard says otherwise it is one edit away, and it is exposed in
    # conf/permeability.yaml.
    #
    # This is a re-baseline: a buildable street is 7 m, not 6 m, so every method's roads displace
    # more than they used to. Permeability per covered edge is unchanged (see g_road_per_m), so
    # what moved is the COST side only -- the same function, honestly priced.
    min_road_width_m: float = 7.0
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


WIDTH_COL = "width_m"


DEFAULT_ROAD_WIDTH_M = 7.0
"""Total width of a road a method emits when nothing else specifies one.

Equals `PermeabilityParams.min_road_width_m`: the default road is the narrowest street that can
actually carry two directions. Re-based from 6.0 with the floor on 2026-07-31.

Every method carries this as a `road_width_m` field, so it is a default a caller can override --
never a global the metric falls back to. The metric itself has no default at all: roads without a
`width_m` are an error.
"""


def with_width(roads: GeoDataFrame, width_m: float) -> GeoDataFrame:
    """Stamp the mandatory `width_m` column on the roads a method emits."""
    out = roads.copy()
    out[WIDTH_COL] = float(width_m)
    return out


def road_conductance(params: PermeabilityParams, width_m: NDArray[np.float64],
                     dist: NDArray[np.float64]) -> NDArray[np.float64]:
    """Conductance a road of TOTAL width `width_m` gives ONE direction of travel over `dist`.

    Affine in width and floored at zero: the margin (verge, drainage, wall clearance) is consumed
    before any lane can fit, and the `(W - road_margin_m)` left over splits between the two
    directions. There is no reference width -- `g_road_per_m` is per metre of usable width -- which
    is what lets the global `corridor_m` go away entirely.

    At the defaults a 7 m street leaves 6 m usable, so each direction gets 3 m and therefore exactly
    20.0: the calibrated conductance of one lane. That equality is what pins `g_road_per_m`, and
    `tests/test_permeability_width.py` guards it.
    """
    usable = np.maximum(0.0, width_m - params.road_margin_m) / 2.0
    return params.g_road_per_m * usable / np.maximum(dist, 1e-12)


def buildable_widths(roads: GeoDataFrame, params: PermeabilityParams) -> NDArray[np.float64]:
    """`width_m` for every road, after checking each one could actually be built.

    Two refusals, both about scoring something that cannot exist:

    * no `width_m` column at all -- width is mandatory, there is no global corridor to inherit;
    * a width below `min_road_width_m`, the narrowest road that fits two directions at once.

    The floor matters because conductance is affine in width with no lower knee: without it the
    model happily reports a 3.5 m road as two 1.25 m lanes running side by side. Above the floor the
    continuum is real and stays -- a wider road genuinely carries more, because a single parked
    vehicle no longer blocks it -- so this is a floor, not a quantization.
    """
    if WIDTH_COL not in roads.columns:
        raise ValueError(
            f"Proposal.roads must carry a '{WIDTH_COL}' column: road width is mandatory since the "
            f"global corridor width was removed. Methods set it on the roads they emit.")
    widths = roads[WIDTH_COL].to_numpy(dtype=float)
    bad = widths < params.min_road_width_m
    if bad.any():
        k = int(np.argmax(bad))
        raise ValueError(
            f"road {k} is {widths[k]:g} m, below the {params.min_road_width_m:g} m floor "
            f"({int(bad.sum())} of {len(roads)} roads are too narrow). A road narrower than that "
            f"cannot carry the traffic the conductance model would credit it with -- it has to fit "
            f"both directions at once.")
    return cast(NDArray[np.float64], widths)


def edge_conductances(
    segments: NDArray[np.object_], dist: NDArray[np.float64], footpath_g: NDArray[np.float64],
    roads: GeoDataFrame | None, params: PermeabilityParams,
) -> NDArray[np.float64]:
    """Per-edge conductance: the footpath, raised wherever a road covers the edge.

    An edge is road-covered when a road segment's own `buffer(width_m/2)` intersects that edge's
    centroid-to-centroid segment (`Mesh.segments`). Every road carries its own `width_m`; there is
    no global corridor width to inherit, and what the road offers is decided by width alone
    (`road_conductance`).

    ## Switch, not sum

    A covered edge takes `max(footpath, road)`, NOT `footpath + road`: the corridor IS the road once
    built, so this models replacement. Treating pavement and carriageway as genuinely parallel would
    ADD them, raising every covered edge by about `g_walk` and shifting every published number by
    ~10%. That re-baseline is the only reason it is not the default.

    ## Cost

    Queries are per ROAD (tens to hundreds), not per EDGE (tens of thousands): one STRtree over the
    mesh edges, then one query per road segment. A per-edge loop here would undo the vectorization
    that made the metric 4.4x faster at region scale.

    Each road SEGMENT is buffered separately, where the pre-width model buffered the road union once
    (per-road widths make a single union buffer impossible). Both approximate the same region, but
    `buffer` polygonalizes a circle, and the union's vertices land differently from each segment's,
    so an edge grazing the boundary can flip. MEASURED over 60 real blocks: 1 flip in 19,023 mesh
    edges, on an edge 2.9984 m from the centreline of a 3 m half-width corridor -- 1.6 mm inside,
    which the union buffer cut out and this one (correctly) keeps. It moved that block's
    permeability by 9.4e-10 relative; the other 59 were bit-identical
    (`notes/2026-07-31-width-is-per-road.md`). Everything else about the reduction is exact.

    ## Monotonicity

    Roads enter only through a `max` with the footpath, and conductance rises with width, so no
    conductance can ever fall. That is load-bearing -- see the module docstring.
    """
    g = footpath_g.copy()
    if roads is None or len(roads) == 0 or segments.size == 0:
        return g
    widths = buildable_widths(roads, params)

    tree = STRtree(list(segments))
    for k, geom in enumerate(roads.geometry):
        w = float(widths[k])
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            cs = list(part.coords)
            for a, b in zip(cs, cs[1:], strict=False):
                if a == b:
                    continue
                hit = tree.query(LineString([a, b]).buffer(w / 2.0), predicate="intersects")
                if not len(hit):
                    continue
                g[hit] = np.maximum(g[hit], road_conductance(params, w, dist[hit]))
    return g


@dataclass(frozen=True)
class EgressSolution:
    """One grounded egress solve, with the assembly it was built from.

    `egress_power` returns only (P, v) because that is all the metric needs. The graph FIGURE
    (`reblock.perm_graph`) also needs the per-edge conductances the Laplacian was assembled from --
    so the assembly is exposed here once, rather than re-derived (which this module's docstring
    forbids) or recomputed alongside a second mesh build that would be free to disagree with the
    first.

    `conductance` is one value per `mesh` edge, after road upgrades: `edge_conductances`' output.
    """
    p: float
    potential: NDArray[np.float64]
    mesh: Mesh
    conductance: NDArray[np.float64]


def solve_egress(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    adj: list[set[int]] | None = None,
    radii: NDArray[np.float64] | None = None,
) -> EgressSolution:
    """The one grounded-Laplacian assembly and sparse solve. See `egress_power` for the model, and
    `EgressSolution` for why the intermediate quantities are returned rather than discarded.

    Degenerate cases keep `egress_power`'s contract exactly: an ungrounded block (no parcel within
    STREET_TOL of a street) or a non-finite solve yields `p = inf` and zero potentials, still
    paired with the mesh and conductances that were built -- a caller that wants to REFUSE those
    (the figure generator does) checks `p` rather than being handed a silently-zero field."""
    parcels = block.parcels
    n = len(parcels)

    mesh = footpath_mesh(block, params, adj=adj, radii=radii)
    rows_arr, cols_arr, dist_arr = mesh.rows, mesh.cols, mesh.dist
    conds_arr = edge_conductances(mesh.segments, dist_arr, mesh.footpath_g, roads, params)
    zeros = np.zeros(n, dtype=np.float64)

    if not mesh.ground.any():
        return EgressSolution(float("inf"), zeros, mesh, conds_arr)

    diag = np.zeros(n, dtype=np.float64)
    np.add.at(diag, rows_arr, conds_arr)
    np.add.at(diag, cols_arr, conds_arr)
    diag[mesh.ground] += params.g_street

    # off-diagonal entries: -g_ij at (i,j) and (j,i)
    off_rows = np.concatenate([rows_arr, cols_arr])
    off_cols = np.concatenate([cols_arr, rows_arr])
    off_data = np.concatenate([-conds_arr, -conds_arr])

    all_rows = np.concatenate([off_rows, np.arange(n, dtype=np.int64)])
    all_cols = np.concatenate([off_cols, np.arange(n, dtype=np.int64)])
    all_data = np.concatenate([off_data, diag])

    lap = coo_matrix((all_data, (all_rows, all_cols)), shape=(n, n)).tocsr()
    b = np.ones(n, dtype=np.float64)
    v = spsolve(cast(csr_matrix, lap), b)
    if not np.all(np.isfinite(v)):
        return EgressSolution(float("inf"), zeros, mesh, conds_arr)
    return EgressSolution(float(b @ v), cast(NDArray[np.float64], v), mesh, conds_arr)


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
    sol = solve_egress(block, roads, params, adj=adj, radii=radii)
    return sol.p, sol.potential


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
