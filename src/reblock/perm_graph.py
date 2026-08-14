"""The permeability graph in DRAWABLE form: one derivation, two renderings.

`permeability_graph` turns a block plus a road set into a flat, serialisable description of the
egress graph -- node positions and potentials, edge endpoints, conductances, which edges a road
raised, and the current flowing along each. `reblock.render.render_graph` draws it to PNG; the site
generator serialises the same structure to JSON for the browser widget. One definition of what the
graph IS, so the picture and the widget cannot disagree about it.

DELIBERATELY FREE OF MATPLOTLIB. The JSON path must not drag a plotting stack behind it, and the
browser explorer boots this module's import closure under Pyodide -- which is why the drawing lives
in `render.py` and only the derivation lives here.

Nothing here re-derives the metric. `solve_egress` performs the one Laplacian assembly and solve;
this module adds only what a picture needs and the metric does not: the per-edge current, and the
mask of edges a road actually raised.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from geopandas import GeoDataFrame
from numpy.typing import NDArray

from reblock.contracts import Block
from reblock.permeability import PermeabilityParams, solve_egress


@dataclass(frozen=True)
class GraphFigure:
    """Everything needed to DRAW the egress graph, and nothing that will not serialise to JSON.

    Node arrays are in `block.parcels` order (length `n`); edge arrays are in `Mesh` order
    (length `m`, each undirected pair stored once with `rows[k] < cols[k]`).

    `ground_g` carries the per-node conductance to ground rather than a bool, for two reasons: the
    energy identity in `tests/test_perm_graph.py` needs the value, and a halo can be drawn weighted
    by the conductance actually present. The bool is `ground_g > 0`, so nothing is lost.

    `upgraded` is stored rather than left to each renderer to recompute from
    `conductance > footpath_g`: computing it once, here, is what stops the PNG and the browser
    widget forming two opinions about which edges the road raised. Note what it claims -- an edge a
    road COVERS whose road term comes in below the footpath keeps the footpath under `max()` and
    reads as not upgraded. That is the honest caption: the drawing shows the edges the road actually
    raised. Possible in principle; never observed in 19,023 mesh edges over 60 real blocks
    (notes/2026-07-31-width-is-per-road.md).

    `current` is signed `rows -> cols`: positive means flow from `rows[k]` toward `cols[k]`.
    """
    cx: NDArray[np.float64]
    cy: NDArray[np.float64]
    potential: NDArray[np.float64]
    ground_g: NDArray[np.float64]
    rows: NDArray[np.int64]
    cols: NDArray[np.int64]
    conductance: NDArray[np.float64]
    footpath_g: NDArray[np.float64]
    upgraded: NDArray[np.bool_]
    current: NDArray[np.float64]
    n: int
    p: float


def permeability_graph(
    block: Block,
    roads: GeoDataFrame | None,
    params: PermeabilityParams = PermeabilityParams(),  # noqa: B008 (frozen, immutable)
    *,
    adj: list[set[int]] | None = None,
    radii: NDArray[np.float64] | None = None,
) -> GraphFigure:
    """The drawable form of `block`'s egress graph under `roads`.

    One solve (`solve_egress`), then two derived quantities: `upgraded` and `current`. `adj` and
    `radii` are threaded exactly as `egress_power`/`permeability` accept them, so a region-scale
    caller does not rebuild `parcel_adjacency` per figure.

    Raises `ValueError` for an ungrounded block. `solve_egress` reports those as `p = inf` with zero
    potentials, because a network with no path to ground has no well-defined dissipated power; a
    figure built from those zeros would show no flow anywhere, which is wrong in a way a reader
    cannot see. This is a figure generator, not a batch metric -- there is no aggregate for it to
    keep marching through.
    """
    sol = solve_egress(block, roads, params, adj=adj, radii=radii)
    if not np.isfinite(sol.p):
        raise ValueError(
            f"block {block.block_id!r} is ungrounded (no parcel within STREET_TOL of a street), so "
            f"its egress power is infinite and every potential is zero -- there is no flow to draw")
    mesh = sol.mesh
    ground_g = np.where(mesh.ground, params.g_street, 0.0)
    current = sol.conductance * (sol.potential[mesh.rows] - sol.potential[mesh.cols])
    return GraphFigure(
        cx=mesh.cx, cy=mesh.cy, potential=sol.potential, ground_g=ground_g,
        rows=mesh.rows, cols=mesh.cols,
        conductance=sol.conductance, footpath_g=mesh.footpath_g,
        upgraded=sol.conductance > mesh.footpath_g,
        current=current, n=mesh.n, p=sol.p)


GraphLayer = Literal["conductance", "current"]

GRAPH_LAYERS: dict[GraphLayer, Callable[[GraphFigure], NDArray[np.float64]]] = {
    "conductance": lambda f: f.conductance,
    "current": lambda f: f.current,
}
"""The two quantities an edge's WIDTH can encode, each spelled as a field access.

A table of accessors rather than `getattr(figure, layer)`: the set is closed while this line is
being written, so a renamed `GraphFigure` field must be a type error at one site, not a silently
blank drawing. Lives here, beside the dataclass whose fields it names, so the PNG renderer, the
figure generator and (in piece C) the JSON emitter all read one definition of what a layer means.
"""
