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
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter
from numpy.typing import NDArray

from reblock.contracts import Block, Metrics, Proposal, Result, Source
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
    corridor_m = cast(float, proposal.params.get("corridor_m", 3.0))
    radii = building_radii(pts, corridor_m)
    corridor = proposal.roads.geometry.buffer(corridor_m).union_all()
    d = pts.geometry.distance(corridor).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(radii > 0.0, 1.0 - d / radii, np.where(d <= 0.0, 1.0, 0.0))
    out = pts.copy()
    out["c"] = np.clip(c, 0.0, 1.0)
    out["radius"] = radii
    return cast(gpd.GeoDataFrame, out[out["c"] > 0.0])


def pct_paved(roads: gpd.GeoDataFrame | None, corridor_m: float, block_area: float) -> float:
    """Fraction of the block's area under the roads' corridor footprint
    (union(roads).buffer(corridor_m)) -- the same buffer the displacement metric uses. 0 for an
    empty road set or a non-positive block area."""
    if roads is None or len(roads) == 0 or block_area <= 0:
        return 0.0
    return float(roads.geometry.buffer(corridor_m).union_all().area / block_area)


def pct_displaced(roads: gpd.GeoDataFrame | None, corridor_m: float,
                  building_points: gpd.GeoDataFrame,
                  radii: NDArray[np.float64]) -> float:
    """Fraction of buildings-equivalent displaced: Σcᵢ / n_buildings (see budget.displacement)."""
    from reblock.budget import displacement
    n = len(building_points)
    if roads is None or len(roads) == 0 or n == 0:
        return 0.0
    return displacement(building_points, radii, roads, corridor_m) / n


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
    fig, ax = plt.subplots(figsize=(10, 10))
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
    is the screen's flagged block_ids; `depths` maps block_id -> the metric's fine score;
    `metric_name` labels the colorbar/title (default "score", the generic fallback for a screen
    without a metric). `metric` (the same BlockMetric) scores any member the screen didn't flag
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

    # --- screen.png: flagged blocks by the metric's score (0..max, continuous), deselected blanked
    fig_s, ax_s = plt.subplots(figsize=(10, 10))
    if not blanked.empty:
        blanked.plot(ax=ax_s, color="white", edgecolor="#dcdcdc", linewidth=0.12)
    if not flagged.empty and score_map:
        flagged.plot(ax=ax_s, column="score", cmap="YlOrRd", vmin=0, vmax=vmax,
                     edgecolor="#33333330", linewidth=0.12)
        sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        fig_s.colorbar(sm, ax=ax_s, fraction=0.03, pad=0.01, label=metric_name)
    if not members.empty:
        members.plot(ax=ax_s, facecolor="none", edgecolor="black", linewidth=0.2)
    if frame is not None:
        ax_s.add_patch(Rectangle((frame[0], frame[1]), frame[2] - frame[0], frame[3] - frame[1],
                                 linewidth=1.6, edgecolor="#111111", facecolor="none", zorder=10))
    bnd = geoms.geometry.bounds
    ax_s.set_xlim(float(bnd["minx"].quantile(0.01)), float(bnd["maxx"].quantile(0.99)))
    ax_s.set_ylim(float(bnd["miny"].quantile(0.01)), float(bnd["maxy"].quantile(0.99)))
    ax_s.set_aspect("equal")
    ax_s.set_axis_off()
    ax_s.set_title(f"{metric_name} (0..{vmax:.3g}); {len(all_member_ids)} blocks reblocked")
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
    fig_r, ax_r = plt.subplots(figsize=(10, 10))
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
    ax_r.set_title(f"{len(all_member_ids)} member block(s); {len(seeds)} seed(s) outlined")
    out_path = out_dir / "region.png"
    save_render(fig_r, out_path)
    plt.close(fig_r)
    return out_path


_METRIC_YLABELS = {
    "external_connectivity": "external connectivity (fraction of access-burden removed)",
    "internal_connectivity": "internal connectivity (backup-route redundancy, mean 1 − R/R_geo)",
    "displacement": "homes displaced (%)",
}

# Every method draws in a fixed colour, keyed on its position in the canonical method registry
# (`list(cfg.all_methods)` -- the global list of all methods, threaded in as `method_order`), so a
# method reads the SAME colour in every curve no matter which others share the run. Hues are spaced
# evenly around the HSV wheel at i/N; the N points are taken from [0, 1) -- NOT [0, 1] -- because
# the wheel wraps (hue 0 == hue 1), so an inclusive endpoint would land the last method on the
# first's colour (the "n+1 bound"). The index is into the FULL registry, not the subset a run
# selected, which is what makes the colour run-independent: a method dropped from one pass no
# longer recolours the rest (the matplotlib-default-cycle bug this replaced).
_HSV_S, _HSV_V = 0.65, 0.85


def _method_colors(method_order: Sequence[str]) -> dict[str, tuple[float, float, float]]:
    """Map each method name to its RGB colour, hue = i/N around the HSV wheel where i is the
    method's index in `method_order` (the canonical registry) and N = len(method_order). N hues
    from [0, 1) so the wheel's wrap never collides two methods; see the note above `_HSV_S`."""
    n = max(len(method_order), 1)
    return {name: colorsys.hsv_to_rgb(i / n, _HSV_S, _HSV_V)
            for i, name in enumerate(method_order)}


def compare_report(results: list[MethodCurve], out_dir: Path,
                   *, method_order: Sequence[str]) -> None:
    """Per metric (external_connectivity, internal_connectivity, displacement): a per-method
    summary table + overlaid cost-benefit curves per block. `results` is the flat (method x block
    x metric) list from reblock.compare. The PLOTTED x-axis differs by metric: for the two benefit
    metrics (external_connectivity, internal_connectivity) it is cumulative DISPLACEMENT (fraction
    of homes displaced, from the index-aligned displacement curve's Σcᵢ/n_buildings -- see `disp_x`
    below); for "displacement" it stays cumulative added road length (m). The stored `Curve.cost`
    (and every CSV written below) remain cumulative added road length (m) for every metric
    regardless -- only the plotted x-axis for the two benefit metrics is re-based onto displacement.
    For the two benefit metrics, writes `frontier_{metric}.csv` (the full (road length, benefit)
    samples per method -- no scalar rank, because a single AUC to a shared road-length cap
    penalised the road-efficient methods: one reaching high benefit at low road ranked below a
    pave-everything method that reached slightly more at several times the road). For
    "displacement" (a RISING cost, never inverted) writes `displacement_vs_length.csv` (the full
    (road length, Σcᵢ/n_buildings) samples per method) and accumulates `displacement_table.csv`
    (mean terminal displaced_fraction per method). `method_order` is the canonical method registry
    (`list(cfg.all_methods)`) that fixes each method's curve colour run-independently -- it must
    cover every method in `results`."""
    import csv
    from collections import defaultdict
    out_dir.mkdir(parents=True, exist_ok=True)
    by_metric: dict[str, list[MethodCurve]] = {}
    for r in results:
        by_metric.setdefault(r.metric, []).append(r)
    colors = _method_colors(method_order)   # one stable name->colour map for every plot
    # method -> [terminal displaced_fraction, ...]
    disp_terminal: dict[str, list[float]] = defaultdict(list)
    # The two benefit curves are plotted against cumulative DISPLACEMENT (fraction of homes
    # displaced), not road length: the displacement curve is index-aligned (same drainage-ordered
    # _sweep over the same roads), so its per-prefix Σcᵢ/n_buildings is the x-axis. (The
    # displacement metric itself stays vs length.)
    disp_x: dict[tuple[str, str], list[float]] = {
        (r.block_id, r.method): list(r.curve.benefit) for r in by_metric.get("displacement", [])}
    for metric, metric_results in by_metric.items():
        by_block: dict[str, list[MethodCurve]] = {}
        for r in metric_results:
            by_block.setdefault(r.block_id, []).append(r)
        if metric == "displacement":
            with (out_dir / "displacement_vs_length.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "block", "road_length_m", "displacement"])
                for r in metric_results:
                    for c, b in zip(r.curve.cost, r.curve.benefit, strict=True):
                        w.writerow([r.method, r.block_id, f"{c:.4f}", f"{b:.4f}"])
                    disp_terminal[r.method].append(r.curve.benefit[-1])
        else:
            with (out_dir / f"frontier_{metric}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "block", "road_length_m", "benefit"])
                for r in metric_results:
                    for c, b in zip(r.curve.cost, r.curve.benefit, strict=True):
                        w.writerow([r.method, r.block_id, f"{c:.4f}", f"{b:.6g}"])
        ylabel = _METRIC_YLABELS[metric]
        for block_id, curves in by_block.items():
            fig, ax = plt.subplots(figsize=(7, 5))
            for mc in curves:
                if metric == "displacement":
                    xs, lab = mc.curve.cost, f"{mc.method} ({int(mc.curve.cost[-1])} m)"
                else:                                    # benefit vs fraction of homes displaced
                    xs = disp_x.get((block_id, mc.method), mc.curve.cost)
                    lab = f"{mc.method} ({xs[-1] * 100:.0f}% homes)"
                ax.plot(xs, mc.curve.benefit, marker="o", label=lab, color=colors[mc.method])
            if metric == "displacement":
                ax.set_xlabel("added road length (m)")
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))    # stored [0,1] -> "45%"
            else:
                ax.set_xlabel("homes displaced (%)")
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))    # disp_x is [0,1]
            ax.set_ylabel(ylabel)
            ax.set_title(f"cost-benefit ({metric}): {block_id}")
            ax.legend()
            stem = "displacement" if metric == "displacement" else f"curve_{metric}"
            save_render(fig, out_dir / f"{stem}_{block_id}.png")
            plt.close(fig)
    if disp_terminal:
        from statistics import mean
        with (out_dir / "displacement_table.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "displaced_fraction", "n_blocks"])
            for m, rows in sorted(disp_terminal.items(), key=lambda kv: -mean(kv[1])):
                w.writerow([m, f"{mean(rows):.4f}", len(rows)])


def depth_vs_road_report(block: Block, roads_by_method: dict[str, gpd.GeoDataFrame], out_dir: Path,
                         *, method_order: Sequence[str], label: str) -> None:
    """A max-access-depth vs added-road curve per method (roads added in drainage order), with a dot
    each time a method first drives the region's MAX depth to a new integer floor -- so "road to
    reach depth D" reads straight off it, and a fixed network (osm_footpaths) plateaus at its floor
    instead of ever crossing lower. Writes `depth_vs_road_<label>.png`."""
    from reblock.animate import depth_sweep
    colors = _method_colors(method_order)
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, roads in roads_by_method.items():
        cutoffs, depths = depth_sweep(block, roads)
        col = colors[name]
        ax.plot(cutoffs, depths, drawstyle="steps-post", color=col,
                label=f"{name} ({int(cutoffs[-1])} m)")
        prev: int | None = None
        mx: list[float] = []
        my: list[float] = []
        for c, d in zip(cutoffs, depths, strict=True):     # dot at each new integer-depth floor
            if prev is None or int(d) < prev:
                mx.append(float(c))
                my.append(float(d))
                prev = int(d)
        ax.plot(mx, my, "o", color=col, ms=5)
    ax.set_xlabel("added road length (m)")
    ax.set_ylabel("max access depth (parcels from a street)")
    ax.set_title(f"access depth vs added road: {label}")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_render(fig, out_dir / f"depth_vs_road_{label}.png")
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
