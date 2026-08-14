"""render: per-parcel heatmaps, before and after an intervention.

`render_before` draws a block's status-quo heatmap (method-independent); `render_after` draws the
post-intervention heatmap plus the proposed new roads. Both take the caller's already-computed
`layers` (a `pd.Series` indexed by `parcel_id`) rather than recomputing them, and both take an
explicit `vmax` so a before/after pair for the same block can be put on one shared colour scale.

Two colorings (`field=`): `"depth"` (default) is the access-depth layers from
`reblock.derive.access.parcel_access_layers` -- an integer count of parcels-from-a-street, `_CMAP`,
`vmin=1`. `"perm"` is the continuous per-parcel egress potential from
`reblock.permeability.parcel_potentials` -- a HIGHER potential means a HARDER escape, `_PERM_CMAP`,
`vmin=0`. Both are drawn dark = hard/deep, light = easy/shallow, so the two colorings read the same
way despite one being an integer depth and the other a continuous potential.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from pyproj import CRS
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from reblock.contracts import BBox, Block, Metrics, Proposal
from reblock.perm_graph import GRAPH_LAYERS, GraphFigure, GraphLayer

_CMAP = "YlOrRd"          # depth coloring: pale -> dark red as access depth grows
_PERM_CMAP = _CMAP        # perm now shares the depth red (YlOrRd) scale (was "PuBu" blue)
_BOUNDARY_COLOR = "#222222"
_ROAD_COLOR = "#1E90FF"
_CONTEXT_OUTLINE = "#dddddd"
_CONTEXT_PT = "#c9c9c9"
_OWN_PT = "#333333"
_DISPLACED_PT = "#c0392b"
_POINT_RADIUS_M = 2.0   # geographic radius (m) of a building/parcel point marker (x sqrt(weight))


def short_label(label: str, limit: int = 80) -> str:
    """A filesystem-safe shortening of a (possibly huge) block/region label: kept verbatim when
    short, else truncated to `limit-ish` chars with a stable hash suffix for uniqueness. A region's
    block_id is `region:` + `+`.join(member ids), which for a big region is thousands of chars and
    blows past the 255-char filename limit -- this keeps `run`'s output filenames (and compare's
    curve filenames) bounded while staying unique + deterministic."""
    if len(label) <= limit:
        return label
    return f"{label[:limit - 20]}...{hashlib.sha256(label.encode()).hexdigest()[:8]}"


def google_maps_url(geom: BaseGeometry, crs: CRS) -> str:
    """A Google Maps link centred on `geom`, zoomed to fit its bounding box -- handy to eyeball
    WHERE a reblocked block or region actually sits on the ground. Reprojects `geom` from `crs` to
    lon/lat (EPSG:4326); the zoom is derived from the wider bbox span."""
    b = gpd.GeoSeries([geom], crs=crs).to_crs(4326).total_bounds
    lat, lon = float((b[1] + b[3]) / 2), float((b[0] + b[2]) / 2)
    span = max(float(b[2] - b[0]), float(b[3] - b[1]))
    zoom = 16 if span <= 0 else int(min(18, max(11, round(math.log2(720 / span)))))
    return f"https://www.google.com/maps/@{lat:.5f},{lon:.5f},{zoom}z"


def frame_bbox(geoms: gpd.GeoDataFrame | gpd.GeoSeries, pad_frac: float = 0.3) -> BBox:
    """A padded square bbox centred on `geoms`' total_bounds -- the render view, and the bbox
    the context query is windowed to. Square + padded so the selection dominates with a context
    margin. Public (not `_`-prefixed): the caller (emit.py) computes this ONCE per render and
    uses it both to window the context query (`source.block_geometries`/`building_points`) and
    to set the axes view (`frame=` below), so the two never drift apart.
    """
    minx, miny, maxx, maxy = geoms.total_bounds
    half = max(maxx - minx, maxy - miny) / 2 + 1.0
    half += half * pad_frac
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    return (cx - half, cy - half, cx + half, cy + half)


def _parcels_with_layer(block: Block, layers: pd.Series) -> gpd.GeoDataFrame:
    """`block.parcels` with a `layer` column looked up from `layers` by
    `parcel_id` (not position), so a shuffled/relabelled `parcels` frame still
    gets the right value per row. Parcels absent from `layers` (shouldn't
    happen) come back NaN and geopandas greys them out.
    """
    parcels = block.parcels.copy()
    parcels["layer"] = parcels["parcel_id"].map(layers)
    return parcels


def _point_disks(points: gpd.GeoDataFrame, radius_m: float | None = None) -> gpd.GeoDataFrame:
    """Points as geographic-size disks, so markers scale with the map extent -- a dense region no
    longer collapses into a screen-size (matplotlib `markersize`) thicket the way fixed-point
    markers do. If a `radius` column is present, each disk uses it verbatim (the per-building
    footprint disks, radius = NN/2 -- see budget.building_radii); elif a `weight` column is
    present, each disk's radius is `radius_m` scaled by sqrt(weight) so its AREA is proportional to
    the weight; else all disks share `radius_m`."""
    if "radius" in points.columns:
        radii = points["radius"].to_numpy()
    elif "weight" in points.columns:
        radii = (radius_m or 0.0) * (points["weight"].to_numpy() ** 0.5)
    else:
        radii = radius_m or 0.0
    return gpd.GeoDataFrame(geometry=points.geometry.buffer(radii), crs=points.crs)


_FIELD_CMAP: dict[str, str] = {"depth": _CMAP, "perm": _PERM_CMAP}
_FIELD_VMIN: dict[str, float] = {"depth": 1, "perm": 0}


def _draw_boundary_and_streets(ax: Axes, block: Block) -> None:
    """The block outline and the EXISTING street network, in `_BOUNDARY_COLOR`.

    Shared by the heatmap and the graph renderers: both need it, and the MultiPolygon reasoning
    below is not worth stating twice.

    A single block (or a gap-free region) is a Polygon -- draw its ring. A gappy multi-block
    region's boundary is a MultiPolygon whose `.boundary` is one ring PER member: redundant with
    the inter-block streets drawn below, so skip it there (drawing it added a misleading
    convex-hull-like outline across the empty gaps between members).

    The street network is never skipped. For a single block it is the outer ring; for a region it
    also carries the inter-block streets between members -- existing egress the 'before' access
    depth is measured against, so they must be visible (a parcel next to one is shallow, not deep),
    and for a gappy region this IS the region outline.
    """
    if isinstance(block.boundary, Polygon):
        gpd.GeoSeries([block.boundary], crs=block.crs).boundary.plot(
            ax=ax, color=_BOUNDARY_COLOR, linewidth=1.3)
    if block.streets is not None and not block.streets.empty:
        block.streets.plot(ax=ax, color=_BOUNDARY_COLOR, linewidth=1.3)


def _draw_heatmap(
    block: Block, layers: pd.Series, vmax: float, *,
    field: Literal["depth", "perm"] = "depth",
    context_outlines: gpd.GeoDataFrame | None = None,
    context_points: gpd.GeoDataFrame | None = None,
    own_points: gpd.GeoDataFrame | None = None,
    displaced_points: gpd.GeoDataFrame | None = None,
    frame: BBox | None = None,
) -> Figure:
    parcels = _parcels_with_layer(block, layers)
    cmap = _FIELD_CMAP[field]
    vmin = _FIELD_VMIN[field]

    # (16, 16): no colorbar now (removed -- the coloring's meaning lives in the READMEs), so the
    # bare map fills the canvas edge-to-edge. At 300 dpi this tight-crops (bbox_inches="tight",
    # pad_inches=0 -- see save_render) to roughly a 3500-4000 px long edge, comfortably sharp when
    # printed at 3-4 ft poster scale.
    fig, ax = plt.subplots(figsize=(16, 16))
    parcels.plot(ax=ax, column="layer", cmap=cmap, vmin=vmin, vmax=vmax,
                 edgecolor="#999999", linewidth=0.4)

    view = frame if frame is not None else frame_bbox(block.parcels)
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])

    # Dimmed context (neighbouring blocks' outlines + building points), drawn under the
    # selection's own boundary/streets/points so the selection reads unambiguously on top.
    if context_outlines is not None and not context_outlines.empty:
        context_outlines.plot(ax=ax, facecolor="none", edgecolor=_CONTEXT_OUTLINE, linewidth=0.4)
    if context_points is not None and not context_points.empty:
        _point_disks(context_points, _POINT_RADIUS_M).plot(
            ax=ax, color=_CONTEXT_PT, alpha=0.6, linewidth=0)

    _draw_boundary_and_streets(ax, block)

    if own_points is not None and not own_points.empty:
        _point_disks(own_points, _POINT_RADIUS_M).plot(ax=ax, color=_OWN_PT, linewidth=0)
    # Displaced sites (own_points' building-footprint disks, radius = NN/2): shaded grey->red by
    # their displacement fraction c = max(0, 1 - d/r) -- drawn on top of own_points, the cost of
    # the road made visible next to it, magnitude and all (not just a binary in/out mark).
    if displaced_points is not None and not displaced_points.empty:
        disks = _point_disks(displaced_points)                 # uses the `radius` column
        # Fixed red, opacity = graze probability c: a barely-grazed home is nearly transparent, a
        # certainly-displaced one solid red.
        colors = [(1.0, 0.0, 0.0, float(ci)) for ci in displaced_points["c"].to_numpy()]
        disks.plot(ax=ax, color=colors, zorder=5, linewidth=0)

    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def render_before(
    block: Block, layers: pd.Series, *, vmax: float,
    field: Literal["depth", "perm"] = "depth",
    context_outlines: gpd.GeoDataFrame | None = None,
    context_points: gpd.GeoDataFrame | None = None,
    own_points: gpd.GeoDataFrame | None = None,
    frame: BBox | None = None,
) -> Figure:
    """Status-quo heatmap for `block` (method-independent). `field` selects the coloring:
    `"depth"` (default) colors by the access-depth `layers`; `"perm"` colors by per-parcel egress
    potential (see the module docstring)."""
    fig = _draw_heatmap(
        block, layers, vmax, field=field,
        context_outlines=context_outlines, context_points=context_points, own_points=own_points,
        frame=frame,
    )
    return fig


def render_after(
    block: Block, proposal: Proposal, layers: pd.Series, *, vmax: float,
    field: Literal["depth", "perm"] = "depth",
    metrics: Metrics | None = None,
    context_outlines: gpd.GeoDataFrame | None = None,
    context_points: gpd.GeoDataFrame | None = None,
    own_points: gpd.GeoDataFrame | None = None,
    displaced_points: gpd.GeoDataFrame | None = None,
    frame: BBox | None = None,
) -> Figure:
    """Post-intervention heatmap for `block`, plus `proposal.roads`. `field` selects the coloring
    (see `render_before`). `displaced_points` (own building sites, each carrying a displacement
    fraction `c` and disk `radius`, see emit.py) are shaded grey->red by `c`."""
    fig = _draw_heatmap(
        block, layers, vmax, field=field,
        context_outlines=context_outlines, context_points=context_points, own_points=own_points,
        displaced_points=displaced_points,
        frame=frame,
    )
    ax = fig.axes[0]
    if proposal.roads is not None and not proposal.roads.empty:
        # per-road width; there is no global corridor to fall back on
        roads_buffered = gpd.GeoDataFrame(
            geometry=proposal.roads.geometry.buffer(
                proposal.roads["width_m"].to_numpy(dtype=float) / 2.0), crs=block.crs)
        roads_buffered.plot(ax=ax, color=_ROAD_COLOR, zorder=4)

    return fig


_GRAPH_PARCEL_EDGE = "#cccccc"     # the wireframe parcels recede to; graph is the subject
_EDGE_GREY = "#8c8c8c"
_EDGE_LW_MIN = 0.15                # a hairline, so the mesh stays present rather than vanishing
# Tuned for a MESH-ONLY width_norm (gen_perm_graph.py's `norms`, fix round 1 Finding A/C) -- the
# norm no longer includes upgraded edges, so plain mesh edges now use the map's full range instead
# of its bottom sliver. Re-tune this if the norm's definition changes again.
_EDGE_LW_MAX = 3.0
# Upgraded edges do not use the frac-of-width_norm map at all (fix round 1 Finding B): see
# render_graph's docstring for why a fixed width is the honest choice here. 1.0pt (fix round 3,
# down from 2.5) -- at 2.5, ~65 criss-crossing centroid-to-centroid edges along a narrow corridor
# painted it as one opaque slab (the corridor buffer beneath, and the mesh/nodes inside it, were
# both fully obscured); 1.0 is thin enough that the mesh, nodes and pale corridor tint show through
# the whole corridor's length while the road is still an unmistakably bold, distinct blue feature.
_UPGRADED_LW = 1.0
_GROUND_HALO = "#1a1a1a"
_NODE_RADIUS_FRAC = 0.18           # of the median edge length, so this holds at region scale


def render_graph(
    figure: GraphFigure,
    block: Block,
    *,
    layer: GraphLayer,
    vmax: float,
    width_norm: float,
    frame: BBox | None = None,
    roads: gpd.GeoDataFrame | None = None,
) -> Figure:
    """The egress graph itself: nodes coloured by potential, edges widthed by `layer`.

    `layer` picks what edge width encodes -- `"conductance"` shows the clearance-fraction mesh (and,
    with roads, which edges they raised); `"current"` shows the drainage, `i = g(phi_i - phi_j)`,
    where the tree appears and a road visibly concentrates flow into itself. Both quantities live on
    the same `GraphFigure`; the choice is the caller's, made once where a figure SET is defined
    rather than re-derived per draw.

    `vmax` (node potential) and `width_norm` (the edge quantity) are explicit for the same reason
    `render_before`/`render_after` take an explicit `vmax`: a before/after pair has to share its
    scales or the comparison means nothing. The caller derives `width_norm` from the MESH-ONLY
    (non-upgraded) edges, pooled across before and after, so the mesh's own spread stays legible;
    against that norm, upgraded edges land at or near its top already -- all of them, for
    conductance; most, for current (fix round 1, Finding A).

    Upgraded edges do NOT draw at that computed width regardless. Because most of them already
    saturate the map, a variable width there would look like a measurement while mostly carrying
    none -- worse than not encoding at all. So upgraded edges draw at a fixed `_UPGRADED_LW`
    instead, and rely on `_ROAD_COLOR` alone to say "a road raised this edge" (fix round 1, Finding
    B). This is a deliberate, honest downgrade: colour still distinguishes them; width no longer
    pretends to.

    Parcels are drawn as a pale wireframe, NOT filled by potential. Filling them would state the
    same quantity twice in two shapes and drown the graph -- this is not the `perm` choropleth with
    dots on top. Node colour uses `_PERM_CMAP` with `vmin=0`, the same scale that choropleth uses,
    so colour means the same thing on both images a reader meets on one page.
    """
    fig, ax = plt.subplots(figsize=(16, 16))

    block.parcels.plot(ax=ax, facecolor="none", edgecolor=_GRAPH_PARCEL_EDGE, linewidth=0.4)

    view = frame if frame is not None else frame_bbox(block.parcels)
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])

    # The corridor under the graph, so corridor and upgraded edges read as one fact. Dissolved PER
    # WIDTH GROUP before buffering, mirroring permeability._road_corridor's unary_union-then-buffer
    # -- GeoDataFrame.plot() draws one patch per row, and overlapping translucent patches compound
    # toward opacity wherever segments join (a drainage-ordered prefix is many short segments along
    # one corridor, overlapping heavily at every joint), so a per-row buffer would never actually
    # deliver alpha=0.25 along most of the corridor. Road width is per-road here (no global corridor
    # width -- see permeability.py's module docstring), so union within each width group and buffer
    # each group's union at its own half-width; this collapses to a single polygon in the normal
    # case where every road in the set shares one width.
    if roads is not None and not roads.empty:
        widths = roads["width_m"].to_numpy(dtype=float)
        corridor = gpd.GeoDataFrame(
            geometry=[unary_union(list(roads.geometry[widths == w])).buffer(float(w) / 2.0)
                      for w in np.unique(widths)],
            crs=block.crs)
        corridor.plot(ax=ax, color=_ROAD_COLOR, alpha=0.25, zorder=2, linewidth=0)

    _draw_boundary_and_streets(ax, block)

    # Edges: ONE LineCollection per colour, never a per-row plot -- a region's ~60k edges are
    # cheap as two collections and hopeless as 60k artists.
    quantity = GRAPH_LAYERS[layer](figure)
    segs = np.stack([
        np.column_stack([figure.cx[figure.rows], figure.cy[figure.rows]]),
        np.column_stack([figure.cx[figure.cols], figure.cy[figure.cols]]),
    ], axis=1)
    frac = np.clip(np.abs(quantity) / width_norm, 0.0, 1.0) if width_norm > 0 else np.zeros(
        len(quantity))
    widths = _EDGE_LW_MIN + frac * (_EDGE_LW_MAX - _EDGE_LW_MIN)

    plain = ~figure.upgraded
    if plain.any():
        ax.add_collection(LineCollection(
            segs[plain], linewidths=widths[plain], colors=_EDGE_GREY, zorder=3))
    if figure.upgraded.any():
        # Fixed width, not `widths[figure.upgraded]` -- see the docstring above. Every upgraded
        # edge clips near `frac=1.0` regardless, so a variable width here would be decorative, not
        # informative; `_UPGRADED_LW` plus `_ROAD_COLOR` is the honest encoding.
        ax.add_collection(LineCollection(
            segs[figure.upgraded], linewidths=_UPGRADED_LW, colors=_ROAD_COLOR, zorder=4))

    # Nodes as geographic-radius disks (the `_point_disks` treatment), so this does not collapse
    # into a screen-size thicket at region scale.
    node_r = _NODE_RADIUS_FRAC * (float(np.median(np.hypot(
        figure.cx[figure.rows] - figure.cx[figure.cols],
        figure.cy[figure.rows] - figure.cy[figure.cols]))) if len(figure.rows) else 1.0)
    nodes = gpd.GeoDataFrame(
        {"phi": figure.potential},
        geometry=gpd.points_from_xy(figure.cx, figure.cy).buffer(node_r), crs=block.crs)

    grounded = figure.ground_g > 0.0
    if grounded.any():
        nodes[grounded].geometry.buffer(node_r * 0.6).plot(
            ax=ax, facecolor="none", edgecolor=_GROUND_HALO, linewidth=1.6, zorder=5)
    nodes.plot(ax=ax, column="phi", cmap=_PERM_CMAP, vmin=0.0, vmax=vmax, linewidth=0, zorder=6)

    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def save_render(fig: Figure, path: str | Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
