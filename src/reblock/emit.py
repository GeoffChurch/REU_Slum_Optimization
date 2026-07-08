"""Output emitters: consumers of run()'s Result list. F1 ships the render
emitter (per block: a shared-vmax before + one after per proposal). The
enabled-emitter registry/fan-out arrives in F4 with the scorecard emitter.
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
    format: str = "png"       # F1: "png" only
    layout: str = "separate"  # F1: "separate" only


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
            f"render F1 supports format=png/layout=separate only; "
            f"got format={cfg.format!r} layout={cfg.layout!r}")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_block: dict[str, list[Result]] = {}
    for r in results:
        by_block.setdefault(r.block.block_id, []).append(r)
    for group in by_block.values():
        _render_block_group(group, out_dir)


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
