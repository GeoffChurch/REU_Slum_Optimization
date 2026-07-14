"""Output emitters: consumers of a run's RunOutput. `render_results` draws, per
block, a shared-vmax before + one after per proposal; `flagged_map` draws the
city choropleth of the screen's flagged blocks. `main` (the Hydra edge) gates
each on its config flag. A scorecard/compare emitter is planned future work.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

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
    """`block.building_points` sites inside `proposal`'s road corridor
    (`proposal.params["corridor_m"]`, default 3.0) -- the render's "cost made visible" mark, ONLY
    for a displacement-cost proposal (a frontage/length method displaces nothing by design, so
    marking sites near its roads as 'displaced' would mislead). Empty (no crash) otherwise, or if
    there are no building points or no proposed roads."""
    pts = block.building_points
    if (proposal.params.get("cost") != "displacement"
            or pts.empty or proposal.roads is None or proposal.roads.empty):
        return cast(gpd.GeoDataFrame, pts.iloc[:0])
    corridor_m = cast(float, proposal.params.get("corridor_m", 3.0))
    corridor = proposal.roads.geometry.buffer(corridor_m).union_all()
    return cast(gpd.GeoDataFrame, pts[pts.within(corridor)])


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
               seed_groups: list[list[str]], out_dir: Path) -> Path | None:
    """Two maps for a region build. `screen.png`: the city depth-proxy choropleth (sqrt(n*A)/P --
    what the screen keys on to find deep fabric), with the WHOLE expanded region located (dark
    member outline + a locator box), the view clipped to the bulk block extent. `region.png`: the
    region's member blocks coloured by that same proxy against dimmed context, the pre-expansion
    **seed** (`seed_groups`, before `RegionBuilder.build` ran) outlined in a heavy edge (needed for
    `convex_hull`, which expands past the seed), plus the building points (member points normal,
    the rest dimmed). Writes both; returns the `region.png` path, or None if there are no regions.
    Gating is the caller's (cfg.region_map.enabled). `source` supplies all candidate outlines
    (`block_geometries()`, read in full -- cheap) and the building points, windowed to the region's
    frame (the expensive layer, so only it is queried narrow)."""
    from matplotlib.patches import Rectangle
    if not regions:
        return None
    geoms = source.block_geometries()
    geoms["block_id"] = geoms["block_id"].astype(str)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_seed_ids = {b for seeds in seed_groups for b in seeds}
    all_member_ids = {b for region in regions for b in region}

    # Per-block depth proxy sqrt(n*A)/P (what the screen keys on) -- the shared colour scale.
    has_proxy = "building_count" in geoms.columns
    vmax = 1.0
    if has_proxy:
        utm = geoms.to_crs(geoms.estimate_utm_crs())
        area = geoms["block_area_m2"] if "block_area_m2" in geoms.columns else utm.geometry.area
        perim = utm.geometry.length
        geoms["proxy"] = np.sqrt(geoms["building_count"] * area) / perim.where(perim > 0)
        # Cap high (p99, not p97): the proxy is heavy-tailed, so a lower cap saturates most of the
        # metro at max colour; p99 keeps the deep fabric standing out against a paler background.
        vmax = float(geoms["proxy"].quantile(0.99)) or 1.0
    members = geoms[geoms["block_id"].isin(all_member_ids)]
    seeds = geoms[geoms["block_id"].isin(all_seed_ids)]
    frame = frame_bbox(members.geometry) if not members.empty else None

    # --- screen.png: the city depth-proxy choropleth (what the screen detects), with the WHOLE
    # expanded region located (dark outline + locator box) -- not just the seed. ---
    fig_s, ax_s = plt.subplots(figsize=(10, 10))
    if has_proxy:
        geoms.plot(ax=ax_s, column="proxy", cmap="YlOrRd", vmin=0, vmax=vmax,
                   linewidth=0, missing_kwds={"color": "#e6e6e6"})
    else:
        geoms.plot(ax=ax_s, color="#e6e6e6", linewidth=0)
    if not members.empty:
        members.plot(ax=ax_s, facecolor="none", edgecolor="#111111", linewidth=0.5)
    if frame is not None:
        ax_s.add_patch(Rectangle((frame[0], frame[1]), frame[2] - frame[0], frame[3] - frame[1],
                                 linewidth=1.6, edgecolor="#111111", facecolor="none", zorder=10))
    # Clip to the bulk block extent so a few far-flung outlier blocks don't pad the view with
    # whitespace; equal aspect keeps the city's true shape.
    bnd = geoms.geometry.bounds
    ax_s.set_xlim(float(bnd["minx"].quantile(0.01)), float(bnd["maxx"].quantile(0.99)))
    ax_s.set_ylim(float(bnd["miny"].quantile(0.01)), float(bnd["maxy"].quantile(0.99)))
    ax_s.set_aspect("equal")
    ax_s.set_axis_off()
    ax_s.set_title(f"depth proxy √(n·A)/P; {len(all_member_ids)} blocks reblocked")
    save_render(fig_s, out_dir / "screen.png")
    plt.close(fig_s)

    # --- region.png: the region's member blocks coloured by that same proxy against dimmed
    # context, the pre-expansion seed outlined heavily, plus the building points. ---
    fig_r, ax_r = plt.subplots(figsize=(10, 10))
    geoms.plot(ax=ax_r, color="#eeeeee", edgecolor="#cccccc", linewidth=0.3)
    if not members.empty and has_proxy:
        members.plot(ax=ax_r, column="proxy", cmap="YlOrRd", vmin=0, vmax=vmax,
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
    "access": "fraction of access-burden removed",
    "efficiency": "network efficiency E",
    "directness": "directness (1/circuity)",
    "resistance": "fraction of egress resistance removed",
}


def compare_report(results: list[MethodCurve], out_dir: Path, cost: str = "length") -> None:
    """Per metric (access, efficiency, directness, resistance): a per-method summary table +
    overlaid cost-benefit curves per block. `results` is the flat (method x block x metric) list
    from reblock.compare. `cost` sets the x-axis (road density m/ha, or buildings displaced) AND
    the table: for "length" a `auc_table_{metric}.csv` (mean AUC, higher = better); for
    "displacement" a `tradeoff_table_{metric}.csv` (mean terminal benefit + mean buildings
    displaced) -- because AUC over the displacement axis inverts (a method that displaces nothing
    scores 0)."""
    import csv
    from statistics import mean
    out_dir.mkdir(parents=True, exist_ok=True)
    by_metric: dict[str, list[MethodCurve]] = {}
    for r in results:
        by_metric.setdefault(r.metric, []).append(r)
    for metric, metric_results in by_metric.items():
        by_block: dict[str, list[MethodCurve]] = {}
        for r in metric_results:
            by_block.setdefault(r.block_id, []).append(r)
        if cost == "displacement":
            # AUC over the displacement axis is meaningless -- a home-sparing method displaces 0,
            # so its curve has no width and AUC->0, ranking the BEST method worst. Report instead
            # the two numbers that matter: terminal navigability and total buildings displaced.
            by_bd: dict[str, list[tuple[float, float]]] = {}
            for r in metric_results:
                by_bd.setdefault(r.method, []).append((r.curve.benefit[-1], r.curve.cost[-1]))
            with (out_dir / f"tradeoff_table_{metric}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "mean_terminal_benefit",
                            "mean_buildings_displaced", "n_blocks"])
                for m, bd in sorted(by_bd.items(), key=lambda kv: -mean(b for b, _ in kv[1])):
                    w.writerow([m, f"{mean(b for b, _ in bd):.4f}",
                                f"{mean(d for _, d in bd):.1f}", len(bd)])
        else:
            by_method: dict[str, list[float]] = {}
            for r in metric_results:
                by_method.setdefault(r.method, []).append(r.auc)
            with (out_dir / f"auc_table_{metric}.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["method", "mean_auc", "n_blocks"])
                for m, aucs in sorted(by_method.items(), key=lambda kv: -mean(kv[1])):
                    w.writerow([m, f"{mean(aucs):.4f}", len(aucs)])
        ylabel = _METRIC_YLABELS[metric]
        xlabel = "buildings displaced" if cost == "displacement" else "road density (m/ha)"
        for block_id, curves in by_block.items():
            fig, ax = plt.subplots(figsize=(7, 5))
            for mc in curves:
                label = (f"{mc.method} ({int(mc.curve.cost[-1])} displaced)"
                         if cost == "displacement" else f"{mc.method} (AUC {mc.auc:.2f})")
                ax.plot(mc.curve.cost, mc.curve.benefit, marker="o", label=label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"cost-benefit ({metric}): {block_id}")
            ax.legend()
            save_render(fig, out_dir / f"curve_{metric}_{block_id}.png")
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
