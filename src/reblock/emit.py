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

from reblock.contracts import Block, Metrics, Proposal, Result, Source
from reblock.render import (
    _CONTEXT_PT,
    _OWN_PT,
    frame_bbox,
    render_after,
    render_before,
    save_render,
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
    (`proposal.params["corridor_m"]`, default 3.0) -- the render's "cost made visible" mark.
    Empty (no crash) if there are no building points or no proposed roads."""
    pts = block.building_points
    if pts.empty or proposal.roads is None or proposal.roads.empty:
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
    """The region-builder's map: every candidate block drawn as light-grey context, then each
    region's member blocks filled in a distinct colour (by region index) and outlined, and
    finally the pre-expansion **seed** blocks (`seed_groups`, before `RegionBuilder.build`
    ran) outlined again in a heavier, high-contrast edge -- so you can see both what the
    builder pulled into each region AND which blocks were the original seed (essential for
    `convex_hull`, which expands past the seed) -- plus the building points: member points
    normal, the rest dimmed. Writes `region_map.png`; returns the path, or None if there are no
    regions. Gating is the caller's (cfg.region_map.enabled). Models `flagged_map`'s
    metro-context style. `source` supplies all candidate outlines (`block_geometries()`, read
    in full -- cheap) and the building points, windowed to the region's frame (the expensive
    layer, so only it is queried narrow)."""
    from matplotlib import colormaps
    if not regions:
        return None
    geoms = source.block_geometries()
    geoms["block_id"] = geoms["block_id"].astype(str)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    # Every candidate block as pale context (thin edge so small informal blocks stay visible),
    # matching flagged_map; the region members are then painted over it.
    geoms.plot(ax=ax, color="#eeeeee", edgecolor="#bdbdbd", linewidth=0.3)
    cmap = colormaps["tab10"]
    n_members = 0
    for i, region in enumerate(regions):
        members = geoms[geoms["block_id"].isin(set(region))]
        if members.empty:
            continue
        n_members += len(members)
        members.plot(ax=ax, color=cmap(i % cmap.N), edgecolor="#333333",
                     linewidth=0.8, alpha=0.85)
    # Outline the pre-expansion seed blocks on top, unfilled (facecolor="none" so the
    # region-colour fill underneath stays visible) with a heavy black edge -- this is what
    # makes a convex_hull region's fill-in legible against its original seed.
    n_seeds = 0
    for seeds in seed_groups:
        seed_blocks = geoms[geoms["block_id"].isin(set(seeds))]
        if seed_blocks.empty:
            continue
        n_seeds += len(seed_blocks)
        seed_blocks.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=2.2)
    # Frame the view on the region members (with context padding) rather than the whole
    # metro -- a small region on a wide candidate extent is otherwise an invisible speck,
    # defeating the point of showing which blocks the builder pulled in. The same frame windows
    # the (expensive) building-points query, so only in-frame points are ever fetched/drawn.
    all_members = geoms[geoms["block_id"].isin({b for region in regions for b in region})]
    if not all_members.empty:
        frame = frame_bbox(all_members.geometry)
        ax.set_xlim(frame[0], frame[2])
        ax.set_ylim(frame[1], frame[3])
        pts = source.building_points(frame)
        if not pts.empty:
            members_union = all_members.geometry.union_all()
            own_pts = pts[pts.within(members_union)]
            context_pts = pts[~pts.within(members_union)]
            if not context_pts.empty:
                context_pts.plot(ax=ax, color=_CONTEXT_PT, markersize=2, alpha=0.6)
            if not own_pts.empty:
                own_pts.plot(ax=ax, color=_OWN_PT, markersize=4)
    ax.set_title(f"{len(regions)} region(s), {n_members} member block(s) of {len(geoms)} "
                 f"candidates ({n_seeds} seed block(s) outlined)")
    ax.set_axis_off()
    out_path = out_dir / "region_map.png"
    save_render(fig, out_path)
    plt.close(fig)
    return out_path


_METRIC_YLABELS = {
    "access": "fraction of access-burden removed",
    "efficiency": "network efficiency E",
    "directness": "directness (1/circuity)",
}


def compare_report(results: list[MethodCurve], out_dir: Path) -> None:
    """Per metric (access, efficiency, directness): an aggregate AUC table (mean AUC per
    method) + overlaid cost-benefit curves per block. `results` is the flat
    (method x block x metric) list from reblock.compare."""
    import csv
    from statistics import mean
    out_dir.mkdir(parents=True, exist_ok=True)
    by_metric: dict[str, list[MethodCurve]] = {}
    for r in results:
        by_metric.setdefault(r.metric, []).append(r)
    for metric, metric_results in by_metric.items():
        by_method: dict[str, list[float]] = {}
        by_block: dict[str, list[MethodCurve]] = {}
        for r in metric_results:
            by_method.setdefault(r.method, []).append(r.auc)
            by_block.setdefault(r.block_id, []).append(r)
        with (out_dir / f"auc_table_{metric}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "mean_auc", "n_blocks"])
            for m, aucs in sorted(by_method.items(), key=lambda kv: -mean(kv[1])):
                w.writerow([m, f"{mean(aucs):.4f}", len(aucs)])
        ylabel = _METRIC_YLABELS[metric]
        for block_id, curves in by_block.items():
            fig, ax = plt.subplots(figsize=(7, 5))
            for mc in curves:
                ax.plot(mc.curve.cost, mc.curve.benefit, marker="o",
                        label=f"{mc.method} (AUC {mc.auc:.2f})")
            ax.set_xlabel("road density (m/ha)")
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
    save_render(fig_before, out_dir / f"{block.block_id}_before.png")
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
        save_render(fig_after, out_dir / f"{block.block_id}_{name}_after.png")
        plt.close(fig_after)
