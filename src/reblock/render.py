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

from reblock.contracts import Block, Metrics, Proposal

_CMAP = "YlOrRd"
_BOUNDARY_COLOR = "#222222"
_ROAD_COLOR = "#08306b"


def _parcels_with_layer(block: Block, layers: pd.Series) -> gpd.GeoDataFrame:
    """`block.parcels` with a `layer` column looked up from `layers` by
    `parcel_id` (not position), so a shuffled/relabelled `parcels` frame still
    gets the right value per row. Parcels absent from `layers` (shouldn't
    happen) come back NaN and geopandas greys them out.
    """
    parcels = block.parcels.copy()
    parcels["layer"] = parcels["parcel_id"].map(layers)
    return parcels


def _draw_heatmap(block: Block, layers: pd.Series, vmax: int) -> Figure:
    parcels = _parcels_with_layer(block, layers)

    fig, ax = plt.subplots()
    parcels.plot(ax=ax, column="layer", cmap=_CMAP, vmin=1, vmax=vmax,
                 edgecolor="#999999", linewidth=0.3)
    gpd.GeoSeries([block.boundary], crs=block.crs).boundary.plot(
        ax=ax, color=_BOUNDARY_COLOR, linewidth=1.0)
    # The existing street network, drawn like the boundary. For a single block this is the
    # outer ring (already drawn above); for a region it also carries the inter-block streets
    # between members -- existing egress the 'before' access depth is measured against, so
    # they must be visible (a parcel next to one is shallow, not deep).
    if block.streets is not None and not block.streets.empty:
        block.streets.plot(ax=ax, color=_BOUNDARY_COLOR, linewidth=1.0)

    sm = plt.cm.ScalarMappable(cmap=_CMAP, norm=Normalize(vmin=1, vmax=vmax))
    fig.colorbar(sm, ax=ax).set_label("access depth (parcels from a street)")

    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def render_before(block: Block, layers: pd.Series, *, vmax: int) -> Figure:
    """Status-quo access-depth heatmap for `block` (method-independent)."""
    fig = _draw_heatmap(block, layers, vmax)
    fig.axes[0].set_title(f"{block.block_id} — before")
    return fig


def render_after(
    block: Block, proposal: Proposal, layers: pd.Series, *, vmax: int,
    metrics: Metrics | None = None,
) -> Figure:
    """Post-intervention access-depth heatmap for `block`, plus `proposal.roads`."""
    fig = _draw_heatmap(block, layers, vmax)
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
