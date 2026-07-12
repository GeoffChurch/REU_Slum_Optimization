"""render: per-parcel access-depth heatmaps, before and after an intervention.

`render_before` draws a block's status-quo access depth (method-independent);
`render_after` draws the post-intervention depth plus the proposed new roads.
Both take the caller's already-computed `layers` (see
`reblock.derive.access.parcel_access_layers`) rather than recomputing them, and
both take an explicit `vmax` so a before/after pair for the same block can be
put on one shared colour scale.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from reblock.contracts import BBox, Block, Metrics, Proposal

_CMAP = "YlOrRd"
_BOUNDARY_COLOR = "#222222"
_ROAD_COLOR = "#08306b"
_CONTEXT_OUTLINE = "#dddddd"
_CONTEXT_PT = "#c9c9c9"
_OWN_PT = "#333333"
_DISPLACED_PT = "#c0392b"


def frame_bbox(geoms: gpd.GeoDataFrame | gpd.GeoSeries, pad_frac: float = 0.6) -> BBox:
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


def _draw_heatmap(
    block: Block, layers: pd.Series, vmax: int, *,
    context_outlines: gpd.GeoDataFrame | None = None,
    context_points: gpd.GeoDataFrame | None = None,
    own_points: gpd.GeoDataFrame | None = None,
    displaced_points: gpd.GeoDataFrame | None = None,
    frame: BBox | None = None,
) -> Figure:
    parcels = _parcels_with_layer(block, layers)

    fig, ax = plt.subplots()
    parcels.plot(ax=ax, column="layer", cmap=_CMAP, vmin=1, vmax=vmax,
                 edgecolor="#999999", linewidth=0.3)

    view = frame if frame is not None else frame_bbox(block.parcels)
    ax.set_xlim(view[0], view[2])
    ax.set_ylim(view[1], view[3])

    # Dimmed context (neighbouring blocks' outlines + building points), drawn under the
    # selection's own boundary/streets/points so the selection reads unambiguously on top.
    if context_outlines is not None and not context_outlines.empty:
        context_outlines.plot(ax=ax, facecolor="none", edgecolor=_CONTEXT_OUTLINE, linewidth=0.3)
    if context_points is not None and not context_points.empty:
        context_points.plot(ax=ax, color=_CONTEXT_PT, markersize=2, alpha=0.6)

    gpd.GeoSeries([block.boundary], crs=block.crs).boundary.plot(
        ax=ax, color=_BOUNDARY_COLOR, linewidth=1.0)
    # The existing street network, drawn like the boundary. For a single block this is the
    # outer ring (already drawn above); for a region it also carries the inter-block streets
    # between members -- existing egress the 'before' access depth is measured against, so
    # they must be visible (a parcel next to one is shallow, not deep).
    if block.streets is not None and not block.streets.empty:
        block.streets.plot(ax=ax, color=_BOUNDARY_COLOR, linewidth=1.0)

    if own_points is not None and not own_points.empty:
        own_points.plot(ax=ax, color=_OWN_PT, markersize=1.25)  # half-radius (markersize is area)
    # Displaced sites (own_points that fall inside a committed road's corridor): a hollow ring
    # drawn on top of own_points -- the cost of the straight road made visible next to it.
    if displaced_points is not None and not displaced_points.empty:
        displaced_points.plot(ax=ax, facecolor="none", edgecolor=_DISPLACED_PT,
                              markersize=40, linewidths=1.2, zorder=5)

    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=Normalize(vmin=1, vmax=vmax))
    fig.colorbar(sm, ax=ax).set_label("access depth (parcels from a street)")

    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def render_before(
    block: Block, layers: pd.Series, *, vmax: int,
    context_outlines: gpd.GeoDataFrame | None = None,
    context_points: gpd.GeoDataFrame | None = None,
    own_points: gpd.GeoDataFrame | None = None,
    frame: BBox | None = None,
) -> Figure:
    """Status-quo access-depth heatmap for `block` (method-independent)."""
    fig = _draw_heatmap(
        block, layers, vmax,
        context_outlines=context_outlines, context_points=context_points, own_points=own_points,
        frame=frame,
    )
    fig.axes[0].set_title(f"{block.block_id} — before")
    return fig


def render_after(
    block: Block, proposal: Proposal, layers: pd.Series, *, vmax: int,
    metrics: Metrics | None = None,
    context_outlines: gpd.GeoDataFrame | None = None,
    context_points: gpd.GeoDataFrame | None = None,
    own_points: gpd.GeoDataFrame | None = None,
    displaced_points: gpd.GeoDataFrame | None = None,
    frame: BBox | None = None,
) -> Figure:
    """Post-intervention access-depth heatmap for `block`, plus `proposal.roads`. `displaced_points`
    (own building sites inside the proposal's road corridor, see emit.py) are marked distinctly."""
    fig = _draw_heatmap(
        block, layers, vmax,
        context_outlines=context_outlines, context_points=context_points, own_points=own_points,
        displaced_points=displaced_points,
        frame=frame,
    )
    ax = fig.axes[0]
    if proposal.roads is not None and not proposal.roads.empty:
        proposal.roads.plot(ax=ax, color=_ROAD_COLOR, linewidth=2.0)

    title = f"{block.block_id} — after"
    if metrics is not None:
        delta_k = metrics.values.get("delta_k")
        if delta_k is not None:
            title += f" (Δk={delta_k:.0f})"
    ax.set_title(title)
    return fig


def save_render(fig: Figure, path: str | Path) -> None:
    fig.savefig(path, dpi=140, bbox_inches="tight")
