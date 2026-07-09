"""Output emitters: consumers of a run's RunOutput. `render_results` draws, per
block, a shared-vmax before + one after per proposal; `flagged_map` draws the
city choropleth of the screen's flagged blocks. `main` (the Hydra edge) gates
each on its config flag. A scorecard/compare emitter is planned future work.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from reblock.contracts import Metrics, Result
from reblock.render import render_after, render_before, save_render

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


def render_results(results: list[Result], out_dir: Path, cfg: RenderConfig) -> None:
    """Per block: a shared-`vmax` `{block_id}_before.png` + one
    `{block_id}_{proposal}_after.png` per Result. Reads the kcomplexity
    access-depth arrays from `Result.metrics` (render never recomputes the
    peel), so a block scored without kcomplexity is skipped."""
    if cfg.format != "png" or cfg.layout != "separate":
        raise NotImplementedError(
            f"render supports format=png/layout=separate only; "
            f"got format={cfg.format!r} layout={cfg.layout!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_block: dict[str, list[Result]] = {}
    for r in results:
        by_block.setdefault(r.block.block_id, []).append(r)
    for group in by_block.values():
        _render_block_group(group, out_dir)


def flagged_map(blocks_path: str, flagged_ids: list[str], out_dir: Path) -> Path | None:
    """Binary city choropleth: every metro block drawn as light-grey context, the
    flagged ones highlighted red. Re-reads the blocks parquet geometry (kept out of
    the Screen so it stays a pure selector). Returns the written path, or None if
    there are no ids. Gating is the caller's (cfg.flagged_map.enabled)."""
    import geopandas as gpd
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


def _render_block_group(group: list[Result], out_dir: Path) -> None:
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

    fig_before = render_before(block, access_before, vmax=vmax)
    save_render(fig_before, out_dir / f"{block.block_id}_before.png")
    plt.close(fig_before)

    for i, r in enumerate(group):
        kc = _kcomplexity_metrics(r.metrics)
        if kc is None:
            continue
        # proposal_id defaults to "" (a method may leave it unset); fall back to
        # a per-proposal index so multiple afters never collide/overwrite.
        name = r.proposal.proposal_id or f"proposal{i}"
        fig_after = render_after(block, r.proposal, kc.fields["access_after"],
                                 vmax=vmax, metrics=kc)
        save_render(fig_after, out_dir / f"{block.block_id}_{name}_after.png")
        plt.close(fig_after)
