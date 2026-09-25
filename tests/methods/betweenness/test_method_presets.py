"""Each preset builds the method it names, with the betweenness source it names -- never the
config-wide `desire_source` default (osm), which would build a different method silently."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from reblock.methods.betweenness import BetweennessDesire, PriorDeviance, RawShare
from reblock.methods.cycle_native import CycleNativeReblocker
from reblock.methods.demand_greedy import DemandGreedyReblocker
from reblock.methods.desire_substrate import DesireWarpedSubstrate
from reblock.methods.loop_closure import LoopClosureRefiner
from reblock.presets import load_method, load_methods

CASES = [("betweenness_tree", False, RawShare),
         ("betweenness_tree_contrast", False, PriorDeviance),
         ("betweenness_looped", True, RawShare),
         ("betweenness_looped_contrast", True, PriorDeviance)]


def _dg(m: object, looped: bool) -> DemandGreedyReblocker:
    if looped:
        assert isinstance(m, LoopClosureRefiner)
        assert (m.budget_frac, m.search_radius_m) == (0.30, 60.0)   # the measured loop settings
        m = m.base
    assert isinstance(m, DemandGreedyReblocker)
    return m


@pytest.mark.parametrize("name,looped,contrast", CASES)
def test_method_preset_builds_what_it_names(name: str, looped: bool, contrast: type) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="config", overrides=[f"method={name}", "shapefile=x"])
    src = _dg(load_method(cfg.method), looped).desire_source
    assert isinstance(src, BetweennessDesire) and isinstance(src.contrast, contrast)
    assert (src.res_m, src.r0_m, src.bend_lambda, src.max_sources, src.seed) == \
        (1.0, 2.0, 50.0, 400, 0)
    assert src.quantiles == (0.80, 0.85, 0.90, 0.95, 0.98)


def test_compare_config_lists_all_four() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=["shapefile=x"])
    registry = load_methods(cfg.all_methods)
    for name, looped, contrast in CASES:
        src = _dg(registry[name], looped).desire_source
        assert isinstance(src, BetweennessDesire) and isinstance(src.contrast, contrast)


CYCLE_CASES = [("cycle_native_betweenness", RawShare),
               ("cycle_native_betweenness_contrast", PriorDeviance)]


def _warped(m: object, plain: CycleNativeReblocker) -> BetweennessDesire:
    """The method is plain cycle_native in every setting but its substrate, which is the configured
    substrate warped by the measured attraction; returns the desire source it warps by."""
    assert isinstance(m, CycleNativeReblocker)
    sub = m.substrate
    assert isinstance(sub, DesireWarpedSubstrate)
    assert (sub.buffer_m, sub.eps, sub.gamma) == (3.0, 0.1, 0.5)
    assert sub.base == plain.substrate
    assert replace(m, substrate=plain.substrate) == plain
    assert isinstance(sub.desire_source, BetweennessDesire)
    return sub.desire_source


@pytest.mark.parametrize("name,contrast", CYCLE_CASES)
def test_cycle_native_preset_warps_only_the_substrate(name: str, contrast: type) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="config", overrides=[f"method={name}", "shapefile=x"])
        plain_cfg = compose(config_name="config", overrides=["method=cycle_native", "shapefile=x"])
    plain = load_method(plain_cfg.method)
    assert isinstance(plain, CycleNativeReblocker)
    src = _warped(load_method(cfg.method), plain)
    assert isinstance(src.contrast, contrast)


def test_compare_config_lists_both_cycle_native_presets() -> None:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=["shapefile=x"])
    registry = load_methods(cfg.all_methods)
    plain = registry["cycle_native"]
    assert isinstance(plain, CycleNativeReblocker)
    for name, contrast in CYCLE_CASES:
        assert isinstance(_warped(registry[name], plain).contrast, contrast)
