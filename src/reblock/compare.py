"""Hydra entrypoint: sweep cost_benefit_curve over screened blocks x a list of methods,
emit the aggregate AUC table + per-block curve plots. Config only at the edge (like run.py).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import hydra
from geopandas import GeoDataFrame
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from reblock.budget import (
    Curve,
    access_benefit,
    auc,
    cost_benefit_curve,
    efficiency_directness_curves,
)
from reblock.contracts import Method, Screen, Source
from reblock.derivations import propose
from reblock.emit import compare_report as compare_report
from reblock.pipeline import select_blocks

log = logging.getLogger(__name__)

# The three lenses every method is graded on: access (burden removed), network-efficiency
# (E, mean 1/distance), and directness (mean euclid/distance, i.e. 1/circuity).


@dataclass(frozen=True)
class MethodCurve:
    method: str
    block_id: str
    metric: str
    curve: Curve
    auc: float


def compare(cfg: DictConfig) -> list[MethodCurve]:
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    names = list(cfg.methods)   # config keys -> the AUC-table labels (not method.identity)
    methods = [cast(Method, instantiate(cfg.all_methods[name])) for name in names]
    _, blocks = select_blocks(source, screen, cfg.max_blocks)

    # one curve per (block, method, metric); a per-(block, metric) common cost cap = the max
    # full road density (the cost axis is metric-independent, so this cap is the same across
    # metrics for a given block -- grouping by (block_id, metric) is still the clean structure).
    raw: list[tuple[str, str, str, Curve]] = []
    for block in blocks:
        for name, method in zip(names, methods, strict=True):
            # every Method here always populates roads (never a None-roads Proposal).
            roads = cast(GeoDataFrame, propose(method, block).roads)
            access = cost_benefit_curve(block, roads, benefit_fn=access_benefit)
            eff, direct = efficiency_directness_curves(block, roads)   # one sweep -> both curves
            raw.append((name, block.block_id, "access", access))
            raw.append((name, block.block_id, "efficiency", eff))
            raw.append((name, block.block_id, "directness", direct))
    results: list[MethodCurve] = []
    groups = {(b, metric) for _, b, metric, _ in raw}
    for block_id, metric in groups:
        group = [(m, c) for m, b, met, c in raw if b == block_id and met == metric]
        cap = max((c.cost[-1] for _, c in group if c.cost), default=0.0)
        for m, c in group:
            results.append(MethodCurve(m, block_id, metric, c, auc(c, cap)))
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="compare_config")
def main(cfg: DictConfig) -> None:
    results = compare(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    compare_report(results, out_dir)
    for r in sorted(results, key=lambda r: (r.metric, -r.auc)):
        log.info("%s %s %s AUC=%.3f", r.metric, r.block_id, r.method, r.auc)


if __name__ == "__main__":
    main()
