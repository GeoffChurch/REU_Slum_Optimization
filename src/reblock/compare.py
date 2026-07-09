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

from reblock.budget import Curve, auc, cost_benefit_curve
from reblock.contracts import Method, Screen, Source
from reblock.derivations import propose
from reblock.emit import compare_report as compare_report
from reblock.pipeline import select_blocks

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MethodCurve:
    method: str
    block_id: str
    curve: Curve
    auc: float


def compare(cfg: DictConfig) -> list[MethodCurve]:
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    names = list(cfg.methods)   # config keys -> the AUC-table labels (not method.identity)
    methods = [cast(Method, instantiate(cfg.all_methods[name])) for name in names]
    _, blocks = select_blocks(source, screen, cfg.max_blocks)

    # one curve per (block, method); a per-block common cost cap = the max full road density.
    raw: list[tuple[str, str, Curve]] = []
    for block in blocks:
        for name, method in zip(names, methods, strict=True):
            # every Method here always populates roads (never a None-roads Proposal).
            roads = cast(GeoDataFrame, propose(method, block).roads)
            raw.append((name, block.block_id, cost_benefit_curve(block, roads)))
    results: list[MethodCurve] = []
    for block_id in {b for _, b, _ in raw}:
        block_curves = [(m, c) for m, b, c in raw if b == block_id]
        cap = max((c.cost[-1] for _, c in block_curves if c.cost), default=0.0)
        for m, c in block_curves:
            results.append(MethodCurve(m, block_id, c, auc(c, cap)))
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="compare_config")
def main(cfg: DictConfig) -> None:
    results = compare(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    compare_report(results, out_dir)
    for r in sorted(results, key=lambda r: -r.auc):
        log.info("%s %s AUC=%.3f", r.block_id, r.method, r.auc)


if __name__ == "__main__":
    main()
