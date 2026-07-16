"""render: per-parcel access-depth heatmaps, before and after an intervention.

`render_before` draws a block's status-quo access depth (method-independent);
`render_after` draws the post-intervention depth plus the proposed new roads.
Both take the caller's already-computed `layers` (see
`reblock.derive.access.parcel_access_layers`) rather than recomputing them, and
both take an explicit `vmax` so a before/after pair for the same block can be
put on one shared colour scale.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from pyproj import CRS
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from reblock.contracts import BBox, Block, Metrics, Proposal

_CMAP = "YlOrRd"
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


def title_label(block_id: str) -> str:
    """A short, non-stretching plot title for a block or region. A region's block_id
    (`region:` + `+`.join(member ids)) is up to thousands of chars, which stretches the whole figure
    to fit the title on one line -- so collapse it to `region of N blocks`. A single block keeps its
    own (already short) id."""
    if block_id.startswith("region:"):
        return f"region of {block_id[len('region:'):].count('+') + 1} blocks"
    return block_id


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


def _draw_heatmap(
    block: Block, layers: pd.Series, vmax: int, *,
    context_outlines: gpd.GeoDataFrame | None = None,
    context_points: gpd.GeoDataFrame | None = None,
    own_points: gpd.GeoDataFrame | None = None,
    displaced_points: gpd.GeoDataFrame | None = None,
    frame: BBox | None = None,
) -> Figure:
    parcels = _parcels_with_layer(block, layers)

    fig, ax = plt.subplots(figsize=(10, 10))
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
        _point_disks(context_points, _POINT_RADIUS_M).plot(
            ax=ax, color=_CONTEXT_PT, alpha=0.6, linewidth=0)

    # Outline. A single block (or a gap-free region) is a Polygon -- draw its ring. A gappy
    # multi-block region's boundary is a MultiPolygon whose `.boundary` is one ring PER member:
    # redundant with the inter-block streets drawn below, so skip it there (drawing it added a
    # misleading convex-hull-like outline across the empty gaps between members).
    if isinstance(block.boundary, Polygon):
        gpd.GeoSeries([block.boundary], crs=block.crs).boundary.plot(
            ax=ax, color=_BOUNDARY_COLOR, linewidth=1.0)
    # The existing street network. For a single block this is the outer ring; for a region it also
    # carries the inter-block streets between members -- existing egress the 'before' access depth
    # is measured against, so they must be visible (a parcel next to one is shallow, not deep). For
    # a gappy region this IS the region outline (the boundary above is skipped).
    if block.streets is not None and not block.streets.empty:
        block.streets.plot(ax=ax, color=_BOUNDARY_COLOR, linewidth=1.0)

    if own_points is not None and not own_points.empty:
        _point_disks(own_points, _POINT_RADIUS_M).plot(ax=ax, color=_OWN_PT, linewidth=0)
    # Displaced sites (own_points' building-footprint disks, radius = NN/2): shaded grey->red by
    # their displacement fraction c = max(0, 1 - d/r) -- drawn on top of own_points, the cost of
    # the road made visible next to it, magnitude and all (not just a binary in/out mark).
    if displaced_points is not None and not displaced_points.empty:
        disks = _point_disks(displaced_points)                 # uses the `radius` column
        disks["c"] = displaced_points["c"].to_numpy()
        disks.plot(ax=ax, column="c", cmap="Reds", vmin=0.0, vmax=1.0, zorder=5, linewidth=0)

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
    fig.axes[0].set_title(f"{title_label(block.block_id)} — before")
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
    (own building sites, each carrying a displacement fraction `c` and disk `radius`, see emit.py)
    are shaded grey->red by `c`."""
    fig = _draw_heatmap(
        block, layers, vmax,
        context_outlines=context_outlines, context_points=context_points, own_points=own_points,
        displaced_points=displaced_points,
        frame=frame,
    )
    ax = fig.axes[0]
    if proposal.roads is not None and not proposal.roads.empty:
        corridor_m = float(cast(float, proposal.params.get("corridor_m", 3.0)))
        roads_buffered = gpd.GeoDataFrame(
            geometry=proposal.roads.geometry.buffer(corridor_m), crs=block.crs)
        roads_buffered.plot(ax=ax, color=_ROAD_COLOR, zorder=4)

    title = f"{title_label(block.block_id)} — after"
    if metrics is not None:
        delta_k = metrics.values.get("delta_k")
        if delta_k is not None:
            title += f" (Δk={delta_k:.0f})"
    ax.set_title(title)
    return fig


def save_render(fig: Figure, path: str | Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
