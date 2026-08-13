"""Hydra entrypoint: sweep permeability_curve/displacement_curve over screened blocks/regions x a
list of methods, emit the permeability-vs-displacement frontier + per-region curve plot. Config
only at the edge (like run.py).
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
from omegaconf import DictConfig, OmegaConf, open_dict
from shapely.ops import unary_union

from reblock.budget import Curve, building_radii, displacement_curve
from reblock.contracts import Block, Method, Screen, Source
from reblock.derivations import propose
from reblock.emit import compare_report as compare_report
from reblock.emit import pct_displaced, pct_paved
from reblock.permeability import PermeabilityParams, permeability_curve
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder, region_reblock
from reblock.render import google_maps_url, short_label

log = logging.getLogger(__name__)

# The two-metric basis every method is graded on: permeability (collective-egress benefit, a
# grounded electrical flow -- see reblock.permeability) and displacement (a rising cost, never
# inverted -- see `displacement_curve`). One frontier: permeability (y) vs displacement (x).


@dataclass(frozen=True)
class PermeabilityConfig:
    """Everything `conf/permeability.yaml` configures, as named fields.

    A tuple would do the same job until someone inserts a field, at which point every positional
    unpack silently shifts by one and keeps type-checking. `frontier_xmax` was exactly that
    insertion, which is why this is a dataclass now.
    """
    params: PermeabilityParams
    matched_displacement: float   # Lens A -- the D every method's prefix is truncated to
    matched_permeability: float   # Lens B -- the P* every method's prefix is truncated to
    frontier_xmax: float          # frontier plot x-limit; see `compare_report`


def load_permeability_config(config_dir: Path = Path("conf")) -> PermeabilityConfig:
    """`conf/permeability.yaml` -> `PermeabilityConfig`. The ONE reader of that file.

    There were three copies of this before -- here, `scripts.compare_budgets` and
    `scripts.calibrate_permeability` -- and this one had silently dropped `road_margin_m`, so the
    frontier path built `PermeabilityParams` with the dataclass default while the other two read
    the configured value. Inert only because the configured value happened to equal the default;
    editing the yaml would have moved two of the three call sites. Hence one implementation.
    """
    raw = cast(DictConfig, OmegaConf.load(config_dir / "permeability.yaml"))
    return PermeabilityConfig(
        params=PermeabilityParams(g_walk=float(raw.g_walk),
                                  g_road_per_m=float(raw.g_road_per_m),
                                  g_street=float(raw.g_street),
                                  road_margin_m=float(raw.road_margin_m),
                                  radius_frac=float(raw.radius_frac)),
        matched_displacement=float(raw.matched_displacement),
        matched_permeability=float(raw.matched_permeability),
        frontier_xmax=float(raw.frontier_xmax))


@dataclass(frozen=True)
class MethodCurve:
    method: str
    block_id: str    # a plain block_id for a singleton region, else the region label
    metric: str
    curve: Curve
    pct_paved: float = 0.0
    pct_displaced: float = 0.0


def _region_label(region: list[Block]) -> str:
    """The MethodCurve row label: the plain block_id for a singleton region (unchanged from
    before regions existed), else its members' ids joined ('+'-separated, sorted for
    determinism) -- `short_label`-truncated with a stable hash suffix if that would make an
    unreasonably long filename in compare_report's frontier_{label}.png."""
    if len(region) == 1:
        return region[0].block_id
    return short_label("+".join(sorted(b.block_id for b in region)))


def _expand_method_sweep(cfg: DictConfig, names: list[str], methods: list[Method]) -> None:
    """Optional `cfg.method_sweep` ({base, param, values}): expand ONE base method config over a
    param's values -- one instantiated method per value, labelled `{base}_{param}{value}` -- and
    append to `names`/`methods`. Avoids a hand-written `all_methods` entry per swept value, e.g.
    `method_sweep={base: clearance, param: repulsion, values: [-3, 0, 3]}` replaces three
    `clearance_rep_*` entries. The variant is merged INTO `cfg.all_methods` so a `${...}`
    interpolation in the base (e.g. `substrate: ${substrate}`) still resolves against the root."""
    sweep = cfg.get("method_sweep")
    if not sweep:
        return
    # item access, not attribute: `sweep.values` would hit DictConfig's `.values()` method.
    base_key = str(sweep["base"])
    param = str(sweep["param"])
    with open_dict(cfg.all_methods):   # the composed config is struct-locked; allow new keys
        for v in sweep["values"]:
            vname = f"{base_key}_{param}{float(v):g}"
            cfg.all_methods[vname] = OmegaConf.merge(cfg.all_methods[base_key], {})
            OmegaConf.update(cfg.all_methods[vname], param, v)
            names.append(vname)
            methods.append(cast(Method, instantiate(cfg.all_methods[vname])))


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
    _expand_method_sweep(cfg, names, methods)   # optional: sweep one base method over a param
    regions = build_regions(source, screen, region_builder, block_groups, cfg.max_blocks)
    params = load_permeability_config().params

    # one curve per (region, method, metric); the stored Curve.cost is always cumulative added
    # road length (m) -- metric-independent, so no shared cap needs computing. (emit.compare_report
    # re-plots the permeability curve against cumulative displacement instead, for display only --
    # this stored cost is unaffected.)
    raw: list[tuple[str, str, str, Curve, float, float]] = []
    for region in regions:
        if not region:
            continue
        label = _region_label(region)
        # Log each selection's locator link (like reblock.run does) so a captured run log is a
        # self-documenting record of what was graded -- the READMEs point readers at this link.
        log.info("%s map: %s", label,
                 google_maps_url(unary_union([b.boundary for b in region]), region[0].crs))
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
            block_area = float(block.parcels.geometry.union_all().area)
            radii = building_radii(block.building_points)
            pp = pct_paved(roads, block_area)
            pd_ = pct_displaced(roads, block.building_points, radii)
            perm = permeability_curve(block, roads, params)
            disp = displacement_curve(block, roads, radii)
            raw.append((name, label, "permeability", perm, pp, pd_))
            raw.append((name, label, "displacement", disp, pp, pd_))
    # No cross-method normalization: the frontier is reported as raw (road length, benefit)
    # samples per method (see emit.compare_report), so there's no shared cost cap to compute.
    return [MethodCurve(m, label, metric, c, pct_paved=pp, pct_displaced=pd_)
            for m, label, metric, c, pp, pd_ in raw]


@hydra.main(version_base=None, config_path="../../conf", config_name="compare_config")
def main(cfg: DictConfig) -> None:
    results = compare(cfg)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    pcfg = load_permeability_config()
    # The canonical registry drives per-method curve colours: `all_methods` is the global method
    # list, and `compare()` has already merged any `method_sweep` variants into it, so this covers
    # every method that could appear in `results`. A method's colour is its index here -- the full
    # registry, not the run's selected subset -- so it stays put when a pass drops another method.
    compare_report(results, out_dir, method_order=[str(k) for k in cfg.all_methods],
                   matched_displacement=pcfg.matched_displacement,
                   matched_permeability=pcfg.matched_permeability,
                   frontier_xmax=pcfg.frontier_xmax)
    # Log each method's terminal: permeability (benefit, road length, %paved) -- no scalar rank --
    # and displacement (rising cost, never inverted) separately.
    for r in sorted(results, key=lambda r: (r.metric, -r.curve.benefit[-1])):
        if r.metric == "displacement":
            log.info("%s %s: %.1f%% of homes displaced", r.block_id, r.method,
                     r.pct_displaced * 100)
        else:
            log.info("%s %s %s: benefit=%.3f at %.0f m (%.1f%% paved)", r.metric, r.block_id,
                     r.method, r.curve.benefit[-1], r.curve.cost[-1], r.pct_paved * 100)


if __name__ == "__main__":
    main()
