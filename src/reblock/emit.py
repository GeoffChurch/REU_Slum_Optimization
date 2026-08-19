"""Output emitters: consumers of a run's RunOutput. `render_results` draws, per
block, a shared-vmax before + one after per proposal; `flagged_map` draws the
city choropleth of the screen's flagged blocks. `main` (the Hydra edge) gates
each on its config flag. A scorecard/compare emitter is planned future work.
"""
from __future__ import annotations

import colorsys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.ticker import PercentFormatter
from numpy.typing import NDArray

from reblock.contracts import Block, Metrics, Proposal, Result, Source
from reblock.method_labels import friendly_method_name
from reblock.render import (
    _CONTEXT_PT,
    _OWN_PT,
    _POINT_RADIUS_M,
    _point_disks,
    frame_bbox,
    render_after,
    render_before,
    save_render,
    short_label,
)

if TYPE_CHECKING:
    from reblock.compare import MethodCurve
    from reblock.metric import BlockMetric

_KCOMPLEXITY = "kcomplexity"


@dataclass
class RenderConfig:
    enabled: bool = False
    format: str = "png"       # only "png" is implemented
    layout: str = "separate"  # only "separate" is implemented


def _kcomplexity_metrics(metrics: tuple[Metrics, ...]) -> Metrics | None:
    """The kcomplexity `Metrics` in a Result's metrics, if scored -- the eval
    that emits the per-parcel access-depth arrays render consumes
    (`fields["access_before"]` / `fields["access_after"]`)."""
    return next((m for m in metrics if m.eval == _KCOMPLEXITY), None)


def _displaced_points(block: Block, proposal: Proposal) -> gpd.GeoDataFrame:
    """`block.building_points` with a per-point displacement fraction `c` = max(0, 1 - d/r)
    (r = NN/2, see budget) and its disk `radius`, for the render to shade. Empty when there are no
    points or no proposed roads."""
    from reblock.budget import building_radii
    pts = block.building_points
    if pts.empty or proposal.roads is None or proposal.roads.empty:
        return cast(gpd.GeoDataFrame, pts.iloc[:0])
    radii = building_radii(pts)
    corridor = _corridor(proposal.roads)
    d = pts.geometry.distance(corridor).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(radii > 0.0, 1.0 - d / radii, np.where(d <= 0.0, 1.0, 0.0))
    out = pts.copy()
    out["c"] = np.clip(c, 0.0, 1.0)
    out["radius"] = radii
    return cast(gpd.GeoDataFrame, out[out["c"] > 0.0])


def _corridor(roads: gpd.GeoDataFrame) -> shapely.geometry.base.BaseGeometry:
    """Paved footprint: every road buffered by its OWN half-width."""
    return roads.geometry.buffer(roads["width_m"].to_numpy(dtype=float) / 2.0).union_all()


def pct_paved(roads: gpd.GeoDataFrame | None, block_area: float) -> float:
    """Fraction of the block's area under the roads' paved footprint -- the same buffer the
    displacement metric uses. 0 for an empty road set or a non-positive block area."""
    if roads is None or len(roads) == 0 or block_area <= 0:
        return 0.0
    return float(_corridor(roads).area / block_area)


def pct_displaced(roads: gpd.GeoDataFrame | None, building_points: gpd.GeoDataFrame,
                  radii: NDArray[np.float64]) -> float:
    """Fraction of buildings-equivalent displaced: Σcᵢ / n_buildings (see budget.displacement)."""
    from reblock.budget import displacement
    n = len(building_points)
    if roads is None or len(roads) == 0 or n == 0:
        return 0.0
    return displacement(building_points, radii, roads) / n


def _member_ids(block_id: str) -> list[str]:
    """The member block ids of a render's selection: a region block_id is
    ``region:`` + ``+``-joined sorted member ids (see ``region.region_block``); a plain block is
    itself. Used to split own vs surrounding context by id, not by fragile boundary geometry."""
    return block_id[len("region:"):].split("+") if block_id.startswith("region:") else [block_id]


def render_results(results: list[Result], out_dir: Path, cfg: RenderConfig,
                   source: Source) -> None:
    """Per block: a shared-`vmax` `{block_id}_before.png` + one
    `{block_id}_{proposal}_after.png` per Result. Reads the kcomplexity
    access-depth arrays from `Result.metrics` (render never recomputes the
    peel), so a block scored without kcomplexity is skipped. `source` supplies the
    surrounding context (neighbouring block outlines + building points): each block's
    render frame windows the query, so the dimmed context is only ever what's actually
    visible in that frame."""
    if cfg.format != "png" or cfg.layout != "separate":
        raise NotImplementedError(
            f"render supports format=png/layout=separate only; "
            f"got format={cfg.format!r} layout={cfg.layout!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_block: dict[str, list[Result]] = {}
    for r in results:
        by_block.setdefault(r.block.block_id, []).append(r)
    for group in by_block.values():
        _render_block_group(group, out_dir, source)


def flagged_map(blocks_path: str, flagged_ids: list[str], out_dir: Path) -> Path | None:
    """Binary city choropleth: every metro block drawn as light-grey context, the
    flagged ones highlighted red. Re-reads the blocks parquet geometry (kept out of
    the Screen so it stays a pure selector). Returns the written path, or None if
    there are no ids. Gating is the caller's (cfg.flagged_map.enabled)."""
    if not flagged_ids:
        return None
    blocks = gpd.read_parquet(blocks_path, columns=["block_id", "geometry"])
    blocks["block_id"] = blocks["block_id"].astype(str)
    blocks["flagged"] = blocks["block_id"].isin(set(flagged_ids))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 16))
    unflagged = blocks[~blocks["flagged"]]
    flagged = blocks[blocks["flagged"]]
    # Informal blocks are small polygons on a wide metro extent, so a thin edge (not
    # just a fill) is what makes them visible; unflagged get a mid-grey that reads on
    # white, flagged a bolder red + heavier edge to stand out against that context.
    if not unflagged.empty:
        unflagged.plot(ax=ax, color="#cccccc", edgecolor="#9a9a9a", linewidth=0.3)
    if not flagged.empty:
        flagged.plot(ax=ax, color="#c0392b", edgecolor="#7b241c", linewidth=0.5)
    ax.set_title(f"{int(blocks['flagged'].sum())} of {len(blocks)} blocks flagged")
    ax.set_axis_off()
    ax.margins(0)
    out_path = out_dir / "flagged_map.png"
    save_render(fig, out_path)
    plt.close(fig)
    return out_path


def region_map(source: Source, regions: list[list[str]],
               seed_groups: list[list[str]], out_dir: Path, *,
               selection: list[str] | None = None,
               depths: dict[str, float] | None = None,
               metric_name: str = "score",
               metric: BlockMetric | None = None) -> Path | None:
    """Two maps for a region build. `screen.png`: the metro coloured by the configured metric's
    fine score (`depths`, from the screen's fine pass) on the absolute 0..max scale -- a
    continuous ramp, no bucketing -- with screen-DESELECTED blocks blanked, and the whole expanded
    region located (dark member outline + a locator box), clipped to the bulk block extent.
    `region.png`: the region's member blocks coloured by that same score against dimmed context,
    the pre-expansion seed outlined heavily, plus building points. When `depths` is None/empty (no
    scoring screen), both maps fall back to a flat located fill (NO proxy colouring). `selection`
    is the screen's flagged block_ids; `depths` maps block_id -> the metric's fine score.
    Neither map has a colorbar or title (`region.png`'s member-count/seed-count title is gone too,
    matching the bare `screen.png`); `screen.png` also has no per-member outline -- just the
    coloured fills and the thick black bounding-box locator -- so the metric colours read
    unoccluded; `metric_name`
    is kept for signature compatibility with callers (default "score", the generic fallback for a
    screen without a metric) but is no longer rendered onto either map. `metric` (the same
    BlockMetric) scores any member the screen didn't flag
    (region growth reaching beyond the flagged set) by its own `fine`, so the region map is one
    metric end to end; without it those members fall back to true peel depth. Writes both; returns
    the `region.png` path, or None if there are no regions."""
    from matplotlib.patches import Rectangle

    from reblock.region import block_depths
    if not regions:
        return None
    geoms = source.block_geometries()
    geoms["block_id"] = geoms["block_id"].astype(str)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_seed_ids = {b for seeds in seed_groups for b in seeds}
    all_member_ids = {b for region in regions for b in region}

    sel = set(selection) if selection else set()
    score_map: dict[str, float] = dict(depths) if depths else {}
    vmax = float(max(score_map.values())) if score_map else 1.0
    geoms["score"] = geoms["block_id"].map(score_map)      # NaN where deselected / unknown
    flagged = geoms[geoms["block_id"].isin(sel)] if sel else geoms.iloc[:0]
    blanked = geoms[~geoms["block_id"].isin(sel)] if sel else geoms
    members = geoms[geoms["block_id"].isin(all_member_ids)]
    seeds = geoms[geoms["block_id"].isin(all_seed_ids)]
    frame = frame_bbox(members.geometry) if not members.empty else None

    # --- screen.png: flagged blocks by the metric's score (0..max, continuous), deselected blanked.
    # No colorbar, no title, and no per-member outline (`edgecolor="black"` used to trace every
    # member's own boundary on top of its fill, occluding the metric colors it's meant to show) --
    # only the thick black bounding-box `Rectangle` below locates the region.
    fig_s, ax_s = plt.subplots(figsize=(16, 16))
    if not blanked.empty:
        blanked.plot(ax=ax_s, color="white", edgecolor="#dcdcdc", linewidth=0.12)
    if not flagged.empty and score_map:
        flagged.plot(ax=ax_s, column="score", cmap="YlOrRd", vmin=0, vmax=vmax,
                     edgecolor="#33333330", linewidth=0.12)
    if frame is not None:
        ax_s.add_patch(Rectangle((frame[0], frame[1]), frame[2] - frame[0], frame[3] - frame[1],
                                 linewidth=1.6, edgecolor="#111111", facecolor="none", zorder=10))
    bnd = geoms.geometry.bounds
    ax_s.set_xlim(float(bnd["minx"].quantile(0.01)), float(bnd["maxx"].quantile(0.99)))
    ax_s.set_ylim(float(bnd["miny"].quantile(0.01)), float(bnd["maxy"].quantile(0.99)))
    ax_s.set_aspect("equal")
    ax_s.set_axis_off()
    ax_s.margins(0)
    save_render(fig_s, out_dir / "screen.png")
    plt.close(fig_s)

    # --- region.png: members by score against dimmed context + seed outline + points ---
    # member scores: from `depths` where present; any members the screen didn't map (DenseCluster
    # growth reaching beyond the flagged set) are scored by the SAME `metric` -- its `fine` from the
    # member's count/area/perim (geoms is in UTM, so area/length are metres) + peel depth, peeled in
    # ONE batched `block_depths` call only when the metric needs it. So the region map is one metric
    # end to end. Without a metric (IdentityScreen) they fall back to true peel depth.
    missing = [b for b in all_member_ids if b not in score_map]
    fallback: dict[str, float] = {}
    if missing and metric is not None:
        mg = geoms[geoms["block_id"].isin(missing)]
        md = block_depths(source, missing) if metric.needs_peel else {}
        fallback = {str(bid): metric.fine(md.get(str(bid), 0.0), float(cnt),
                                          float(g.area), float(g.length))
                    for bid, cnt, g in zip(mg["block_id"], mg["building_count"], mg.geometry,
                                           strict=True)}
    elif missing:
        fallback = block_depths(source, missing)
    member_score = {b: score_map.get(b, fallback.get(b, 0.0)) for b in all_member_ids}
    m_vmax = float(max([v for v in member_score.values() if v] or [1.0]))
    members = members.copy()
    members["score"] = members["block_id"].map(member_score)
    fig_r, ax_r = plt.subplots(figsize=(16, 16))
    geoms.plot(ax=ax_r, color="#eeeeee", edgecolor="#cccccc", linewidth=0.3)
    if not members.empty and member_score:
        members.plot(ax=ax_r, column="score", cmap="YlOrRd", vmin=0, vmax=m_vmax,
                     edgecolor="#8a8a8a", linewidth=0.4)
    elif not members.empty:
        members.plot(ax=ax_r, color="#c0392b", edgecolor="#8a8a8a", linewidth=0.4)
    if not seeds.empty:
        seeds.plot(ax=ax_r, facecolor="none", edgecolor="black", linewidth=2.2)
    if frame is not None:
        ax_r.set_xlim(frame[0], frame[2])
        ax_r.set_ylim(frame[1], frame[3])
        pts = source.building_points(frame)
        if not pts.empty:
            members_union = members.geometry.union_all()
            own_pts = cast(gpd.GeoDataFrame, pts[pts.within(members_union)])
            context_pts = cast(gpd.GeoDataFrame, pts[~pts.within(members_union)])
            if not context_pts.empty:
                _point_disks(context_pts, _POINT_RADIUS_M).plot(
                    ax=ax_r, color=_CONTEXT_PT, alpha=0.6, linewidth=0)
            if not own_pts.empty:
                _point_disks(own_pts, _POINT_RADIUS_M).plot(ax=ax_r, color=_OWN_PT, linewidth=0)
    ax_r.set_aspect("equal")
    ax_r.set_axis_off()
    ax_r.margins(0)
    out_path = out_dir / "region.png"
    save_render(fig_r, out_path)
    plt.close(fig_r)
    return out_path


# Every method draws in a fixed colour, keyed on its position in the canonical method registry
# (`list(cfg.all_methods)` -- the global list of all methods, threaded in as `method_order`), so a
# method reads the SAME colour in every curve no matter which others share the run. Hues are spaced
# evenly around the HSV wheel at i/N; the N points are taken from [0, 1) -- NOT [0, 1] -- because
# the wheel wraps (hue 0 == hue 1), so an inclusive endpoint would land the last method on the
# first's colour (the "n+1 bound"). The index is into the FULL registry, not the subset a run
# selected, which is what makes the colour run-independent: a method dropped from one pass no
# longer recolours the rest (the matplotlib-default-cycle bug this replaced).
_HSV_S, _HSV_V = 0.65, 0.85

# The frontier plot's own axis labels and stroke styling, named rather than written inline in
# `compare_report` below, because a SECOND renderer draws the same chart: the browser widget on the
# Methods index (web/src/widgets/frontier.ts). scripts/gen_frontier_bundle bakes these values into
# examples/method-comparison/frontier.json, so the widget draws with what this plot drew with by
# construction instead of by two lists being kept in step by hand -- the widget replaces this exact
# PNG on the page, so a divergence would mean JS-off and JS-on readers see different charts.
# `method_colors` (the curve colours) and `friendly_method_name` (the legend names) are shared
# the same way, and both axes are PercentFormatter'd -- see `compare_report`.
FRONTIER_X_LABEL = "displacement"
FRONTIER_Y_LABEL = "permeability"
FRONTIER_LW = 2.5
FRONTIER_GUIDE_LW = 1.0
FRONTIER_GUIDE_COLOR = "gray"


def method_colors(method_order: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    """Map each method name to its RGB colour, hue = i/N around the HSV wheel where i is the
    method's index in `method_order` (the canonical registry) and N = len(method_order). N hues
    from [0, 1) so the wheel's wrap never collides two methods; see the note above `_HSV_S`."""
    n = max(len(method_order), 1)
    return {name: colorsys.hsv_to_rgb(i / n, _HSV_S, _HSV_V)
            for i, name in enumerate(method_order)}


def compare_report(results: list[MethodCurve], out_dir: Path,
                   *, method_order: Sequence[str],
                   matched_displacement: float, matched_permeability: float,
                   frontier_xmax: float) -> None:
    """ONE frontier curve per block/region: permeability (y) vs displacement (x), every method
    overlaid, no title. `results` is the flat (method x block x metric) list from reblock.compare,
    where `metric` is either "permeability" (`curve.cost` = cumulative added road length (m),
    `curve.benefit` = permeability per drainage-ordered prefix) or "displacement" (`curve.benefit`
    = cumulative Σcᵢ/n_buildings, the homes-displaced fraction). The two curves are index-aligned
    (same drainage-ordered `_sweep` over the same roads), so the displacement curve's per-prefix
    benefit re-bases the permeability curve's x-axis from raw road length onto displacement -- see
    `disp_x` below. Writes `frontier_permeability.csv` (the full (displacement, permeability)
    samples per method) and one `frontier_{block_id}.png` per block/region -- x-axis
    "displacement", y-axis "permeability", both `PercentFormatter`'d (both are [0,1) fractions).
    `method_order` is the canonical method registry (`list(cfg.all_methods)`) that fixes each
    method's curve colour run-independently -- it must cover every method in `results`. Each
    frontier also draws the two calibrated lens cutoffs from `conf/permeability.yaml` (the same
    thresholds `scripts.compare_budgets`'s two-lens driver grades methods against) as thin dashed
    guide lines: `matched_displacement` (Lens A, vertical) and `matched_permeability` (Lens B,
    horizontal). A `results` with no "permeability" rows writes nothing (no benefit metric to
    plot)."""
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    by_metric: dict[str, list[MethodCurve]] = {}
    for r in results:
        by_metric.setdefault(r.metric, []).append(r)
    perm_results = by_metric.get("permeability", [])
    if not perm_results:
        return
    colors = method_colors(method_order)   # one stable name->colour map for every plot
    # permeability is plotted against cumulative DISPLACEMENT (fraction of homes displaced), not
    # road length: the displacement curve is index-aligned, so its per-prefix Σcᵢ/n_buildings is
    # the x-axis.
    disp_x: dict[tuple[str, str], list[float]] = {
        (r.block_id, r.method): list(r.curve.benefit) for r in by_metric.get("displacement", [])}
    with (out_dir / "frontier_permeability.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "block", "displacement", "permeability"])
        for r in perm_results:
            xs = disp_x.get((r.block_id, r.method), r.curve.cost)
            for x, y in zip(xs, r.curve.benefit, strict=True):
                w.writerow([r.method, r.block_id, f"{x:.4f}", f"{y:.6g}"])
    by_block: dict[str, list[MethodCurve]] = {}
    for r in perm_results:
        by_block.setdefault(r.block_id, []).append(r)
    for block_id, curves in by_block.items():
        fig, ax = plt.subplots(figsize=(12, 9))
        for mc in curves:
            xs = disp_x.get((block_id, mc.method), mc.curve.cost)
            ax.plot(xs, mc.curve.benefit, marker="o", ms=9, lw=FRONTIER_LW,
                    label=friendly_method_name(mc.method), color=colors[mc.method])
        # The two calibrated lens cutoffs (conf/permeability.yaml) as thin dashed guides, drawn
        # UNDER the curves (low zorder) so they read as reference lines, not data -- Lens A's
        # matched displacement (vertical) and Lens B's matched permeability (horizontal); see
        # scripts/compare_budgets.py's two-lens driver, which grades every method against these
        # exact thresholds.
        ax.axvline(matched_displacement, ls="--", lw=FRONTIER_GUIDE_LW,
                   color=FRONTIER_GUIDE_COLOR, zorder=0.5,
                   label=f"matched displacement = {matched_displacement:.0%}")
        ax.axhline(matched_permeability, ls="--", lw=FRONTIER_GUIDE_LW,
                   color=FRONTIER_GUIDE_COLOR, zorder=0.5,
                   label=f"matched permeability = {matched_permeability:.0%}")
        ax.set_xlabel(FRONTIER_X_LABEL, fontsize=16)
        ax.set_ylabel(FRONTIER_Y_LABEL, fontsize=16)
        # DISPLAY ONLY -- `frontier_permeability.csv` above already holds every sample, including
        # the ones past the limit, so nothing measured is lost by clipping the view. Methods have
        # no common terminal, so without this the axis autoscales to whichever ran longest and
        # squashes the range where the lens guides sit. See conf/permeability.yaml's frontier_xmax.
        ax.set_xlim(0.0, frontier_xmax)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.tick_params(labelsize=13)
        ax.legend(fontsize=13)
        save_render(fig, out_dir / f"frontier_{block_id}.png")
        plt.close(fig)


def _render_block_group(group: list[Result], out_dir: Path, source: Source) -> None:
    block = group[0].block
    # access_before is method-independent: take it from the first Result that
    # carries kcomplexity metrics; a block scored without kcomplexity has no
    # peel layers to draw and is skipped.
    kc_first = next(
        (kc for r in group if (kc := _kcomplexity_metrics(r.metrics)) is not None), None)
    if kc_first is None:
        return
    access_before = kc_first.fields["access_before"]
    # access_after can only shrink depth, so access_before.max() bounds the
    # shared color scale across the before and every after.
    vmax = int(access_before.max())

    # One frame for the whole group: it windows the context query AND sets the axes view
    # (single source of truth, frame_bbox in render.py), so the two never drift apart.
    frame = frame_bbox(block.parcels)
    outlines = source.block_geometries(frame)
    outlines["block_id"] = outlines["block_id"].astype(str)
    pts = source.building_points(frame)
    # Split the selection's own member block(s) from the surrounding context by block_id -- robust
    # where a geometric `within(block.boundary)` is not: `block.boundary` is the union of Voronoi
    # PARCELS (which can poke outside the raw block polygon) or, for a disjoint convex_hull region,
    # the bulging hull -- either would mis-split a member outline or a gap neighbour. Own outlines
    # are dropped (else dimmed over their own heatmap); points split against the TIGHT member union
    # (matching region_map), so a convex_hull region's gap points read as context, not own.
    is_member = outlines["block_id"].isin(_member_ids(block.block_id))
    context_outlines = cast(gpd.GeoDataFrame, outlines[~is_member])
    member_union = (outlines[is_member].geometry.union_all() if is_member.any()
                    else block.boundary)
    own_points = cast(gpd.GeoDataFrame, pts[pts.within(member_union)])
    context_points = cast(gpd.GeoDataFrame, pts[~pts.within(member_union)])

    fig_before = render_before(
        block, access_before, vmax=vmax, frame=frame,
        context_outlines=context_outlines, context_points=context_points, own_points=own_points,
    )
    save_render(fig_before, out_dir / f"{short_label(block.block_id)}_before.png")
    plt.close(fig_before)

    for i, r in enumerate(group):
        kc = _kcomplexity_metrics(r.metrics)
        if kc is None:
            continue
        # proposal_id defaults to "" (a method may leave it unset); fall back to
        # a per-proposal index so multiple afters never collide/overwrite.
        name = r.proposal.proposal_id or f"proposal{i}"
        fig_after = render_after(
            block, r.proposal, kc.fields["access_after"], vmax=vmax, metrics=kc, frame=frame,
            context_outlines=context_outlines, context_points=context_points,
            own_points=own_points, displaced_points=_displaced_points(block, r.proposal),
        )
        save_render(fig_after, out_dir / f"{short_label(block.block_id)}_{name}_after.png")
        plt.close(fig_after)
