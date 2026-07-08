"""Hydra entrypoint: composes conf/{data,method,eval} config groups into a
pluggable Source -> [Method] -> [Eval] pipeline, and renders before/after
heatmaps under the Hydra run dir.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, cast

import hydra
import matplotlib.pyplot as plt
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from reblock.contracts import Block, Eval, Method, Metrics, Proposal, Result, Source
from reblock.render import render_after, render_before, save_render

log = logging.getLogger(__name__)

_KCOMPLEXITY = "kcomplexity"


def _kcomplexity_metrics(metrics: tuple[Metrics, ...]) -> Metrics | None:
    """The kcomplexity `Metrics` in a proposal's metrics, if it was scored --
    the eval that emits the per-parcel access-depth arrays render consumes
    (`fields["access_before"]` / `fields["access_after"]`)."""
    return next((m for m in metrics if m.eval == _KCOMPLEXITY), None)


@dataclass
class RunConfig:
    """Flat, ergonomic constructor for direct/programmatic use (tests, small
    scripts): `RunConfig(shapefile=..., alpha=..., ...)` builds a single-source/
    single-method/single-eval pipeline without going through Hydra compose.
    `data`/`method`/`eval` are the same `_target_`-bearing shapes Hydra's
    config-group composition produces (see conf/config.yaml) -- when left
    unset, __post_init__ derives them from the flat fields below, so `run()`
    has exactly one code path (`hydra.utils.instantiate`) regardless of
    whether `cfg` came from here or from `hydra.compose`.
    """
    shapefile: str = "???"
    region_id: str = "phule"
    alpha: float = 2.0
    seed: int = 0
    max_blocks: int = 1
    # ShapefileSource fails loud instead of guessing a CRS when a shapefile
    # has no .prj (e.g. Phule Nagar); this states that assumption explicitly.
    # None preserves "fail loud" as the CLI default too.
    assumed_crs: int | None = None
    render_dir: str | None = None
    data: Any = None
    method: Any = None
    eval: Any = None

    def __post_init__(self) -> None:
        if self.data is None:
            self.data = {
                "_target_": "reblock.data.shapefile.ShapefileSource",
                "path": self.shapefile, "region_id": self.region_id,
                "assumed_crs": self.assumed_crs,
            }
        if self.method is None:
            self.method = {
                "_target_": "reblock.methods.topology.TopologyMethod",
                "alpha": self.alpha, "seed": self.seed,
            }
        if self.eval is None:
            self.eval = [{"_target_": "reblock.eval.kcomplexity.KComplexityEval"}]


def run(cfg: RunConfig | DictConfig, *, render_base: Path | None = None) -> list[Result]:
    # Per-element instantiate for the eval LIST (not instantiate(cfg.eval) as a
    # whole): instantiating a ListConfig of @dataclass _target_s short-circuits
    # to schema-validated DictConfig nodes instead of calling the constructor.
    # cfg.method is a single _target_ dict, so instantiate(cfg.method) calls the
    # constructor directly and is safe.
    source = cast(Source, instantiate(cfg.data))
    method = cast(Method, instantiate(cfg.method))
    evals = cast("list[Eval]", [instantiate(e) for e in cfg.eval])

    render_dir = (render_base / cfg.render_dir
                  if render_base is not None and cfg.render_dir else None)
    if render_dir is not None:
        render_dir.mkdir(parents=True, exist_ok=True)

    region = source.region()
    results: list[Result] = []
    for block in islice(region.blocks, cfg.max_blocks):
        proposal = method.propose(block)
        metrics = tuple(ev.score(block, proposal) for ev in evals)
        results.append(Result(block=block, proposal=proposal, metrics=metrics))

        if render_dir is not None:
            _render_block(block, [(proposal, metrics)], render_dir)

    return results


def _render_block(block: Block, per_proposal: list[tuple[Proposal, tuple[Metrics, ...]]],
                  render_dir: Path) -> None:
    # Render reads the per-parcel access-depth arrays the kcomplexity eval
    # already emits into Metrics.fields (access_before / access_after) rather
    # than recomputing the BFS peel -- so rendering a block REQUIRES the
    # kcomplexity eval to have run on it. `access_before` is method-independent,
    # so take it from the first proposal that carries kcomplexity metrics; a
    # block scored without kcomplexity has no peel layers and is skipped.
    kc_first = next(
        (kc for _, m in per_proposal if (kc := _kcomplexity_metrics(m)) is not None), None)
    if kc_first is None:
        return

    access_before = kc_first.fields["access_before"]
    # access_after can only shrink depth (adding roads never pushes a parcel
    # farther from a street), so access_before.max() bounds the shared scale.
    vmax = int(access_before.max())

    fig_before = render_before(block, access_before, vmax=vmax)
    save_render(fig_before, render_dir / f"{block.block_id}_before.png")
    plt.close(fig_before)

    for i, (proposal, metrics) in enumerate(per_proposal):
        kc = _kcomplexity_metrics(metrics)
        if kc is None:
            continue
        # proposal_id defaults to "" (a method may leave it unset); fall back
        # to a per-proposal index so multiple afters never collide/overwrite.
        name = proposal.proposal_id or f"proposal{i}"
        fig_after = render_after(block, proposal, kc.fields["access_after"], vmax=vmax, metrics=kc)
        save_render(fig_after, render_dir / f"{block.block_id}_{name}_after.png")
        plt.close(fig_after)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    render_base = Path(HydraConfig.get().runtime.output_dir)
    for r in run(cfg, render_base=render_base):
        log.info("%s %s", r.block.block_id, {m.eval: dict(m.values) for m in r.metrics})


if __name__ == "__main__":
    main()
