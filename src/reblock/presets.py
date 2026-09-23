"""The ONE place configuration becomes objects.

Hydra's `instantiate` returns `Any`. Every configured object enters the typed program through a
function here, which builds it and checks it against the protocol its caller is about to rely on --
so the `Any` stops in this module, and a preset that builds the wrong kind of thing fails at load,
naming its `_target_`, rather than at first use hours into a run.

The classes the presets name declare no field defaults, so a preset that omits, misspells or
renames a field is a `TypeError` from its constructor -- also here, also at load.

Loading is EAGER: an entry point builds everything it is configured with before it does any work,
so a broken `all_methods` entry fails in the first second instead of after the screen has run. That
makes every constructor part of startup, which is why none of them may touch the network (a
desire-line source fetches on `desire_field`, footprint tiles on `for_blocks`;
`data/provision.cached_kblock_source` alone provisions when called, and it is the data source, which
every entry point built first already).

`tests/test_presets.py` loads every preset of every config group, and every example variant,
through these functions and checks each field against its annotation -- which is what puts "load
time" in CI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

from reblock.contracts import Eval, Method, Screen, Source
from reblock.methods.desire_lines import DesireLineSource
from reblock.methods.osm_footpaths import FootpathSource
from reblock.methods.substrates import Substrate
from reblock.metric import BlockMetric, Gate
from reblock.permeability import PermeabilityParams
from reblock.region import RegionBuilder

T = TypeVar("T")


def _mismatch(node: DictConfig, built: object, kind: str) -> TypeError:
    return TypeError(f"{node['_target_']} built a {type(built).__qualname__}, which is not a "
                     f"{kind}")


# One function per kind rather than one generic `load(node, kind)`: the kinds are a closed set
# known where each is called, and a Protocol cannot be passed where `type[T]` is expected.

def load_source(node: DictConfig) -> Source:
    built = instantiate(node)
    if not isinstance(built, Source):
        raise _mismatch(node, built, "Source")
    return built


def load_screen(node: DictConfig) -> Screen:
    built = instantiate(node)
    if not isinstance(built, Screen):
        raise _mismatch(node, built, "Screen")
    return built


def load_region_builder(node: DictConfig) -> RegionBuilder:
    built = instantiate(node)
    if not isinstance(built, RegionBuilder):
        raise _mismatch(node, built, "RegionBuilder")
    return built


def load_method(node: DictConfig) -> Method:
    built = instantiate(node)
    if not isinstance(built, Method):
        raise _mismatch(node, built, "Method")
    return built


def load_methods(nodes: DictConfig) -> dict[str, Method]:
    """Every entry of an `all_methods` registry, by its config key -- ALL of them, not only the
    ones a run selects, so a broken entry cannot wait for the run that happens to pick it."""
    return {str(name): load_method(nodes[name]) for name in nodes}


def load_evals(nodes: ListConfig) -> list[Eval]:
    """Per element: `instantiate` on the whole ListConfig would hand back schema-validated
    DictConfig nodes for `@dataclass` targets instead of constructing them."""
    evals: list[Eval] = []
    for node in nodes:
        built = instantiate(node)
        if not isinstance(built, Eval):
            raise _mismatch(node, built, "Eval")
        evals.append(built)
    return evals


def load_desire_source(node: DictConfig) -> DesireLineSource:
    built = instantiate(node)
    if not isinstance(built, DesireLineSource):
        raise _mismatch(node, built, "DesireLineSource")
    return built


def load_footpath_source(node: DictConfig) -> FootpathSource:
    built = instantiate(node)
    if not isinstance(built, FootpathSource):
        raise _mismatch(node, built, "FootpathSource")
    return built


def load_substrate(node: DictConfig) -> Substrate:
    built = instantiate(node)
    if not isinstance(built, Substrate):
        raise _mismatch(node, built, "Substrate")
    return built


def load_metric(node: DictConfig) -> BlockMetric:
    built = instantiate(node)
    if not isinstance(built, BlockMetric):
        raise _mismatch(node, built, "BlockMetric")
    return built


def load_permeability_params(node: DictConfig) -> PermeabilityParams:
    built = instantiate(node)
    if not isinstance(built, PermeabilityParams):
        raise _mismatch(node, built, "PermeabilityParams")
    return built


def load_gate(node: DictConfig) -> Gate:
    built = instantiate(node)
    if not isinstance(built, Gate):
        raise _mismatch(node, built, "Gate")
    return built


def load_research(node: DictConfig, kind: type[T]) -> T:
    """A research object -- `reblock.transplant`, `reblock.data.pools` -- checked against the class
    its caller names.

    Generic where every loader above is not, because this module must not import the research
    code: it is in shipped modules' import closures, and a GW constant would then sit in their
    code hashes (`tests/transplant/test_isolation.py`). So it cannot name the kind; the caller,
    which may import it, does. A research object reaches a shipped Method only as a configured
    field, and carries its own code in its `identity`.
    """
    built = instantiate(node)
    if not isinstance(built, kind):
        raise _mismatch(node, built, kind.__qualname__)
    return built


@dataclass(frozen=True)
class Stages:
    """What every entry point reads blocks through: where they come from, which the screen flags,
    and how a seed group grows into the region that is reblocked."""

    source: Source
    screen: Screen
    region_builder: RegionBuilder


def load_stages(cfg: DictConfig) -> Stages:
    return Stages(source=load_source(cfg.data), screen=load_screen(cfg.screen),
                  region_builder=load_region_builder(cfg.region_builder))
