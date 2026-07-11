"""Hydra entrypoint: sweep cost_benefit_curve over screened blocks/regions x a list of
methods, emit the aggregate AUC table + per-region curve plots. Config only at the edge
(like run.py).
"""
from __future__ import annotations

import hashlib
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
    resistance_benefit,
)
from reblock.contracts import Block, Method, Screen, Source
from reblock.derivations import propose
from reblock.emit import compare_report as compare_report
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder, region_reblock

log = logging.getLogger(__name__)

# The four lenses every method is graded on: access (burden removed), network-efficiency
# (E, mean 1/distance), directness (mean euclid/distance, i.e. 1/circuity), and resistance
# (grounded egress resistance, redundancy-aware, benefit = fraction of egress resistance removed).


@dataclass(frozen=True)
class MethodCurve:
    method: str
    block_id: str    # a plain block_id for a singleton region, else the region label
    metric: str
    curve: Curve
    auc: float


def _region_label(region: list[Block]) -> str:
    """The MethodCurve row label: the plain block_id for a singleton region (unchanged from
    before regions existed), else its members' ids joined ('+'-separated, sorted for
    determinism) -- truncated with a stable hash suffix if that would make an unreasonably
    long filename in compare_report's curve_{metric}_{label}.png."""
    if len(region) == 1:
        return region[0].block_id
    label = "+".join(sorted(b.block_id for b in region))
    if len(label) <= 80:
        return label
    return f"{label[:60]}...{hashlib.sha256(label.encode()).hexdigest()[:8]}"


def compare(cfg: DictConfig) -> list[MethodCurve]:
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    block_groups = (
        [[str(b) for b in group] for group in cfg.block_ids]
        if cfg.block_ids is not None else None
    )
    names = list(cfg.methods)   # config keys -> the AUC-table labels (not method.identity)
    methods = [cast(Method, instantiate(cfg.all_methods[name])) for name in names]
    regions = build_regions(source, screen, region_builder, block_groups, cfg.max_blocks)
    cost = str(cfg.get("cost", "length"))   # curve x-axis: "length" (m/ha) | "displacement"
    corridor_m = float(cfg.get("corridor_m", 3.0))

    # one curve per (region, method, metric); a per-(region, metric) common cost cap = the max
    # full road density (the cost axis is metric-independent, so this cap is the same across
    # metrics for a given region -- grouping by (region_label, metric) is still the clean
    # structure).
    raw: list[tuple[str, str, str, Curve]] = []
    for region in regions:
        if not region:
            continue
        label = _region_label(region)
        for name, method in zip(names, methods, strict=True):
            if len(region) == 1:
                # Singleton region: the exact pre-region single-block path.
                block = region[0]
                roads = cast(GeoDataFrame, propose(method, block).roads)
            else:
                # Multi-block region: reblock jointly. The region-block's streets ARE the full
                # existing network (perimeter + inter-block); the method's added roads are graded
                # against that existing-network baseline (existing inter-block streets are egress,
                # not part of the intervention).
                result = region_reblock(region, method, [])
                block = result.block
                roads = cast(GeoDataFrame, result.proposal.roads)
            access = cost_benefit_curve(block, roads, benefit_fn=access_benefit,
                                        cost=cost, corridor_m=corridor_m)
            eff, direct = efficiency_directness_curves(block, roads, cost=cost,
                                                       corridor_m=corridor_m)   # one sweep -> both
            resistance = cost_benefit_curve(block, roads, benefit_fn=resistance_benefit,
                                            cost=cost, corridor_m=corridor_m)
            raw.append((name, label, "access", access))
            raw.append((name, label, "efficiency", eff))
            raw.append((name, label, "directness", direct))
            raw.append((name, label, "resistance", resistance))
    results: list[MethodCurve] = []
    groups = {(label, metric) for _, label, metric, _ in raw}
    for label, metric in groups:
        group = [(m, c) for m, lbl, met, c in raw if lbl == label and met == metric]
        cap = max((c.cost[-1] for _, c in group if c.cost), default=0.0)
        for m, c in group:
            results.append(MethodCurve(m, label, metric, c, auc(c, cap)))
    return results


@hydra.main(version_base=None, config_path="../../conf", config_name="compare_config")
def main(cfg: DictConfig) -> None:
    results = compare(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    cost = str(cfg.get("cost", "length"))
    compare_report(results, out_dir, cost=cost)
    if cost == "displacement":   # AUC inverts on the displacement axis -- log benefit + displaced
        for r in sorted(results, key=lambda r: (r.metric, -r.curve.benefit[-1])):
            log.info("%s %s %s: benefit=%.3f, %d displaced", r.metric, r.block_id, r.method,
                     r.curve.benefit[-1], int(r.curve.cost[-1]))
    else:
        for r in sorted(results, key=lambda r: (r.metric, -r.auc)):
            log.info("%s %s %s AUC=%.3f", r.metric, r.block_id, r.method, r.auc)


if __name__ == "__main__":
    main()
