"""Over-provisioning calibration probe for the dual-target connectivity joint target.

For each internal-capable method (`greedy_arterial_repulsion`, `clearance_looped`,
`euclidean_grid`), over-provisioned per `conf/joint_target.yaml`'s `over_provision` knobs, on each
of the 6 multiblock example regions (the {depth, depth_density, density_compactness} metrics x
{capetown, nairobi} cities, loaded like `scripts/gen_multiblock_example.py`) plus the pinned
method-comparison block (`scripts/gen_method_comparison.py`'s `ZAF.9.3.1_1_40972`), this builds the
three index-aligned curves (`access_benefit` external, `commute_ratio_benefit` internal, the
fractional `displacement_curve`) and reports the per-(method, region) `max_internal_within` at the
config's `e_min`/`d_max`. It then proposes `i_min = min over regions of
greedy_arterial_repulsion's max_internal_within * 0.95` (the reference method's reliable floor) and
reports which methods clear that floor on which regions.

This script writes NO repo constants -- it only prints. A human reviews the table + proposal and
bakes the agreed `(i_min, e_min, d_max)` into `conf/joint_target.yaml` by hand.

Run:  pixi run python -m scripts.calibrate_joint_target
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from reblock.budget import (
    Curve,
    access_benefit,
    building_radii,
    commute_ratio_benefit,
    cost_benefit_curve,
    displacement_curve,
)
from reblock.contracts import Block, Method, Screen, Source
from reblock.derivations import propose
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder, region_block

METHODS = ("greedy_arterial_repulsion", "clearance_looped", "euclidean_grid")
REFERENCE_METHOD = "greedy_arterial_repulsion"
METRICS = ("depth", "depth_density", "density_compactness")
CITIES = ("capetown", "nairobi")
# The deepest block in a topology-tractable size window, pinned in gen_method_comparison.py.
PINNED_BLOCK_ID = "ZAF.9.3.1_1_40972"
CORRIDOR_M = 3.0
CONFIG_DIR = Path("conf").resolve()


def max_internal_within(external: Curve, internal: Curve, displacement: Curve, *,
                        e_min: float, d_max: float) -> float:
    """Max internal benefit at any sample with external >= e_min and displacement <= d_max; -inf if
    none. The largest internal a method can deliver while clearing the external floor within
    budget."""
    best = float("-inf")
    for i in range(len(external.cost)):
        if external.benefit[i] >= e_min and displacement.benefit[i] <= d_max:
            best = max(best, internal.benefit[i])
    return best


@dataclass(frozen=True)
class RegionSpec:
    """One example region to probe: a display `name`, the hydra overrides that select it (metric
    + data + screen + region_builder for a multiblock region, or an explicit pinned block), and the
    explicit seed group to hand `build_regions` (None = let the screen decide)."""
    name: str
    overrides: list[str]
    block_ids: list[list[str]] | None


def _multiblock_specs() -> list[RegionSpec]:
    """The 6 multiblock example regions, one per (metric, city) -- mirrors
    `scripts/gen_multiblock_example.py`'s screen + region-build overrides."""
    return [
        RegionSpec(
            name=f"{metric}/{city}",
            overrides=[f"metric={metric}", f"data={city}_full", "screen=dense_compact",
                      "region_builder=dense_cluster", "region_builder.max_buildings=3000",
                      "max_blocks=1"],
            block_ids=None,
        )
        for metric in METRICS for city in CITIES
    ]


def _method_comparison_spec() -> RegionSpec:
    """The pinned single-block method-comparison flagship -- mirrors
    `scripts/gen_method_comparison.py`'s load of `ZAF.9.3.1_1_40972`."""
    return RegionSpec(
        name="method_comparison",
        overrides=["data=capetown_full", f"block_ids=[[{PINNED_BLOCK_ID}]]", "max_blocks=1"],
        block_ids=[[PINNED_BLOCK_ID]],
    )


def region_specs() -> list[RegionSpec]:
    return [*_multiblock_specs(), _method_comparison_spec()]


def _over_provision_overrides(over_provision: dict[str, dict[str, object]]) -> list[str]:
    """Flatten `conf/joint_target.yaml`'s `over_provision.<method>` dotted-key knobs into hydra
    override strings `all_methods.<method>.<key>=<value>`, for every configured method at once (a
    single compose call configures all methods, mirroring gen_multiblock_example.py's pattern). A
    key starting with `+` (e.g. `"+max_anchors"`) is a key ABSENT from the inline `all_methods.*`
    config -- hydra's add-new-key `+` override form is required to set it (plain `key=value` only
    works for a key the base config already has), exactly why gen_multiblock_example.py writes
    `+all_methods.greedy_arterial_repulsion.max_anchors=64` rather than a plain override."""
    out = []
    for method, knobs in over_provision.items():
        for key, value in knobs.items():
            if key.startswith("+"):
                out.append(f"+all_methods.{method}.{key[1:]}={value}")
            else:
                out.append(f"all_methods.{method}.{key}={value}")
    return out


def _compose_region(spec: RegionSpec, method_overrides: list[str]) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="compare_config",
                       overrides=[*spec.overrides, *method_overrides])


def _load_region_block(cfg: DictConfig, spec: RegionSpec) -> Block:
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    source.block_ids = None                                          # type: ignore[attr-defined]
    region = build_regions(source, screen, region_builder, spec.block_ids, 1)[0]
    return region_block(region)


def _instantiate_methods(cfg: DictConfig) -> dict[str, Method]:
    return {name: cast(Method, instantiate(cfg.all_methods[name])) for name in METHODS}


def _load_joint_target_config() -> tuple[float, float, list[str]]:
    """(e_min, d_max, over_provision hydra overrides) from `conf/joint_target.yaml`. `i_min` is
    NOT read here -- it is the placeholder this probe proposes a value for."""
    raw = cast(DictConfig, OmegaConf.load(CONFIG_DIR / "joint_target.yaml"))
    e_min = float(raw.e_min)
    d_max = float(raw.d_max)
    over_provision = cast("dict[str, dict[str, object]]",
                          OmegaConf.to_container(raw.over_provision, resolve=True))
    return e_min, d_max, _over_provision_overrides(over_provision)


def main() -> None:
    e_min, d_max, method_overrides = _load_joint_target_config()
    print(f"e_min={e_min}, d_max={d_max}, over-provision overrides: {method_overrides}\n")

    results: dict[str, dict[str, float]] = {}
    for spec in region_specs():
        print(f"=== region {spec.name} ===")
        cfg = _compose_region(spec, method_overrides)
        block = _load_region_block(cfg, spec)
        radii = building_radii(block.building_points, CORRIDOR_M)
        row: dict[str, float] = {}
        for name, method in _instantiate_methods(cfg).items():
            prop = propose(method, block)
            roads = prop.roads
            if roads is None or roads.empty:
                print(f"  {name}: no roads proposed -- skip")
                row[name] = float("-inf")
                continue
            ext = cost_benefit_curve(block, roads, benefit_fn=access_benefit)
            internal = cost_benefit_curve(block, roads, benefit_fn=commute_ratio_benefit)
            disp = displacement_curve(block, roads, radii, corridor_m=CORRIDOR_M)
            mi = max_internal_within(ext, internal, disp, e_min=e_min, d_max=d_max)
            row[name] = mi
            print(f"  {name}: max_internal_within={mi:.4f}")
        results[spec.name] = row

    print(f"\n=== per-(method, region) max_internal_within (e_min={e_min}, d_max={d_max}) ===")
    print(f"{'region':<24}" + "".join(f"{m:>26}" for m in METHODS))
    for region_name, row in results.items():
        cells = "".join(f"{row.get(m, float('-inf')):>26.4f}" for m in METHODS)
        print(f"{region_name:<24}{cells}")

    arterial_vals = [row[REFERENCE_METHOD] for row in results.values()
                     if row.get(REFERENCE_METHOD, float("-inf")) > float("-inf")]
    if not arterial_vals:
        print(f"\nno region produced a finite {REFERENCE_METHOD} max_internal_within "
              "-- cannot propose i_min")
        return
    i_min = min(arterial_vals) * 0.95
    print(f"\nproposed i_min = min over regions of {REFERENCE_METHOD}'s max_internal_within * 0.95"
          f" = {i_min:.4f}")
    print(f"proposed joint target: (i_min={i_min:.4f}, e_min={e_min}, d_max={d_max})")

    print(f"\n=== survivors at proposed i_min={i_min:.4f} ===")
    for name in METHODS:
        survived = [r for r, row in results.items() if row.get(name, float("-inf")) >= i_min]
        killed = [r for r, row in results.items() if row.get(name, float("-inf")) < i_min]
        print(f"  {name}: {len(survived)}/{len(results)} regions survive")
        print(f"    survives: {survived}")
        print(f"    killed:   {killed}")


if __name__ == "__main__":
    main()
