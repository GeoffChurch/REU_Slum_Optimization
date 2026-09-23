"""Every preset in conf/ loads through the typed loader, with every field of the declared type.

`reblock.presets` is where configuration becomes objects, and the classes the presets name declare
no field defaults -- so constructing one IS the check that its preset spells every field, and a
missing, misspelled or renamed key is a `TypeError` from the constructor. That check only happens
when something loads the preset, which for most presets meant a real run. This loads all of them.

What construction cannot catch is a value of the wrong TYPE: Hydra hands a YAML string to a `Path`
field, or a quoted number to an `int` one, without complaint, and the object is built. So every
built object's fields are also checked, shallowly, against their annotations.
"""
from __future__ import annotations

import ast
import collections.abc
import dataclasses
import functools
import inspect
import types
import typing
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

import reblock.data.provision
from reblock.contracts import Method
from reblock.presets import (
    load_desire_source,
    load_evals,
    load_footpath_source,
    load_gate,
    load_method,
    load_methods,
    load_metric,
    load_region_builder,
    load_screen,
    load_source,
    load_stages,
    load_substrate,
)
from tests.methods.test_arterial import ARTERIAL
from tests.methods.test_clearance import CLEARANCE
from tests.methods.test_cycle_native import CYCLE
from tests.methods.test_euclidean_grid import GRID
from tests.methods.test_peel import PEEL
from tests.methods.test_resistance_greedy import GREEDY
from tests.methods.test_topology_method import TOPOLOGY
from tests.test_loop_closure import LOOPS

CONF = Path("conf").resolve()
ROOT = Path(__file__).resolve().parents[1]
# Keys a preset leaves `???` for the run to supply, and the value supplied here.
REQUIRED = {"desire_source=pbf": ["desire_source.pbf_path._args_=[/nonexistent/x.osm.pbf]"]}


def _compose(config_name: str, overrides: list[str]) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        return compose(config_name=config_name, overrides=overrides)


@pytest.fixture(autouse=True)
def _full_city_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`cached_kblock_source` provisions when called -- downloads on a cold cache. Seed an empty one
    with the two files it checks for, so the full-city presets load offline and read nothing."""
    for city in ("capetown", "nairobi"):
        (tmp_path / f"blocks_{city}_full.parquet").touch()
        (tmp_path / f"buildings_{city}_full.parquet").touch()
    monkeypatch.setattr(reblock.data.provision, "DEFAULT_CACHE", tmp_path)


# ------------------------------------------------------------------------------------------------
# The shallow type check


def _conforms(value: object, hint: object) -> bool:
    """Does `value` have the type `hint` names -- one level deep, with container elements checked
    and Protocols checked through `isinstance`? An annotation this cannot read raises rather than
    passing, so a new kind of field cannot slip through unchecked."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        return any(_conforms(value, a) for a in args)
    if hint is type(None):
        return value is None
    if origin is typing.Literal:
        return value in args
    if hint is bool:
        return isinstance(value, bool)
    if hint is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if hint is float:                     # the numeric tower, as mypy reads it: an int is a float
        return isinstance(value, int | float) and not isinstance(value, bool)
    if origin is collections.abc.Callable:
        return callable(value)
    if origin in (list, tuple, collections.abc.Sequence):
        if not isinstance(value, origin) or isinstance(value, str):
            return False
        items = list(typing.cast(collections.abc.Sequence[object], value))
        if origin is tuple and not (len(args) == 2 and args[1] is Ellipsis):
            return len(items) == len(args) and all(map(_conforms, items, args))
        return all(_conforms(v, args[0]) for v in items)
    if isinstance(hint, type):
        # a Protocol that is not runtime_checkable raises here, which is the answer wanted
        return isinstance(value, hint)
    raise TypeError(f"no shallow check for annotation {hint!r}")


def _annotated_fields(obj: object) -> Iterator[tuple[str, object, object]]:
    """(name, value, annotation) for every setting `obj` was constructed with: a dataclass's init
    fields, or a plain class's `__init__` parameters -- which each of these stores under its own
    name, so a missing attribute raises rather than skipping the parameter."""
    cls = type(obj)
    if dataclasses.is_dataclass(obj):
        hints = typing.get_type_hints(cls)
        for f in dataclasses.fields(obj):
            if f.init:
                yield f.name, getattr(obj, f.name), hints[f.name]
    else:
        hints = typing.get_type_hints(cls.__init__)
        for name, p in inspect.signature(cls.__init__).parameters.items():
            if name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                yield name, getattr(obj, name), hints[name]


def _configured(value: object) -> bool:
    """A value the presets built, whose own fields are checked in turn."""
    return (type(value).__module__.startswith("reblock.")
            and not isinstance(value, type | functools.partial))


def _type_errors(obj: object, where: str) -> list[str]:
    errors: list[str] = []
    for name, value, hint in _annotated_fields(obj):
        here = f"{where}.{name}"
        if not _conforms(value, hint):
            errors.append(f"{here} = {value!r} is not a {hint}")
        children = value if isinstance(value, list | tuple) else [value]
        for i, child in enumerate(children):
            if _configured(child):
                errors += _type_errors(child, here if child is value else f"{here}[{i}]")
    return errors


def _assert_typed(obj: object, where: str) -> None:
    errors = _type_errors(obj, where)
    assert not errors, "\n".join(errors)


# ------------------------------------------------------------------------------------------------
# Every preset of every config group


def _load_data(cfg: DictConfig) -> list[object]:
    return [load_source(cfg.data)]


def _load_screen(cfg: DictConfig) -> list[object]:
    return [load_screen(cfg.screen)]


def _load_metric(cfg: DictConfig) -> list[object]:
    return [load_metric(cfg.metric), load_gate(cfg.metric_gate)]


# Each config group, and how to load what it configures. `buildings` and `building_count` have no
# object of their own: they are interpolated into the data source and the scoring screen, and are
# loaded through those. A new group missing here fails `test_every_config_group_has_a_loader`.
GROUP_LOADERS: dict[str, tuple[list[str], Callable[[DictConfig], list[object]]]] = {
    "data": ([], _load_data),
    "screen": ([], _load_screen),
    "region_builder": ([], lambda cfg: [load_region_builder(cfg.region_builder)]),
    "method": ([], lambda cfg: [load_method(cfg.method)]),
    "eval": ([], lambda cfg: list(load_evals(cfg.eval))),
    "substrate": ([], lambda cfg: [load_substrate(cfg.substrate)]),
    "desire_source": ([], lambda cfg: [load_desire_source(cfg.desire_source),
                                        load_footpath_source(cfg.desire_source)]),
    "metric": ([], _load_metric),
    "buildings": (["data=capetown"], _load_data),
    "building_count": (["screen=dense_compact"], _load_screen),
}


def _groups() -> list[str]:
    return sorted(p.name for p in CONF.iterdir() if p.is_dir() and p.name != "example")


def _presets() -> list[tuple[str, str]]:
    return [(group, p.stem) for group in _groups()
            for p in sorted((CONF / group).glob("*.yaml")) if not p.stem.startswith("_")]


def test_every_config_group_has_a_loader() -> None:
    assert set(_groups()) == set(GROUP_LOADERS)


@pytest.mark.parametrize(("group", "preset"), _presets(), ids=[f"{g}={p}" for g, p in _presets()])
def test_every_preset_loads_with_typed_fields(group: str, preset: str) -> None:
    extra, load = GROUP_LOADERS[group]
    choice = f"{group}={preset}"
    cfg = _compose("config", ["shapefile=x", *extra, choice, *REQUIRED.get(choice, [])])
    built = load(cfg)
    assert built
    for i, obj in enumerate(built):
        _assert_typed(obj, f"{choice}[{i}]")


# ------------------------------------------------------------------------------------------------
# Every example variant, with its own overrides, on every data config that loads offline


def _variants() -> list[str]:
    return sorted(p.stem for p in (CONF / "example").glob("*.yaml"))


def _data_presets() -> list[str]:
    return sorted(p.stem for p in (CONF / "data").glob("*.yaml"))


@pytest.mark.parametrize("data", _data_presets())
@pytest.mark.parametrize("variant", [None, *_variants()], ids=lambda v: v or "compare_config")
def test_every_example_variant_loads_with_typed_fields(variant: str | None, data: str) -> None:
    overrides = ["shapefile=x", f"data={data}"] + ([f"+example={variant}"] if variant else [])
    cfg = _compose("compare_config", overrides)
    stages = load_stages(cfg)
    methods = load_methods(cfg.all_methods)
    assert set(cfg.methods) <= set(methods)
    loaded: dict[str, object] = {
        "data": stages.source, "screen": stages.screen, "region_builder": stages.region_builder,
        "desire_source": load_desire_source(cfg.desire_source),
        "substrate": load_substrate(cfg.substrate), "metric": load_metric(cfg.metric),
        "metric_gate": load_gate(cfg.metric_gate),
        **{f"eval[{i}]": e for i, e in enumerate(load_evals(cfg.eval))},
        **{f"all_methods.{name}": m for name, m in methods.items()},
    }
    for where, obj in loaded.items():
        _assert_typed(obj, f"{variant or 'compare_config'}:{where}")


# ------------------------------------------------------------------------------------------------
# The configurations the method tests start from are the shipped ones


# Each method's tests spell one configuration and vary a setting at a time from it; each says it is
# the shipped preset, and this is what keeps that true when a preset is tuned.
TEST_BASES: dict[str, Method] = {
    "clearance": CLEARANCE, "cycle_native": CYCLE, "euclidean_grid": GRID,
    "greedy_arterial": ARTERIAL, "loop_closure": LOOPS, "peel": PEEL,
    "resistance_greedy": GREEDY, "topology": TOPOLOGY,
}


@pytest.mark.parametrize("preset", sorted(TEST_BASES))
def test_method_tests_start_from_the_shipped_preset(preset: str) -> None:
    shipped = load_method(_compose("config", ["shapefile=x", f"method={preset}"]).method)
    assert shipped == TEST_BASES[preset]


# ------------------------------------------------------------------------------------------------
# The loader is the only way in


def _calls_instantiate(path: Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "hydra.utils" and any(
                a.name == "instantiate" for a in node.names):
            return True
        if isinstance(node, ast.Attribute) and node.attr == "instantiate" and isinstance(
                node.value, ast.Attribute) and node.value.attr == "utils":
            return True
    return False


def test_only_the_loader_calls_hydra_instantiate() -> None:
    """`instantiate` returns `Any`; a second caller is a second place a preset can build the wrong
    kind of thing and have it cast into a Method, and nothing checks the cast."""
    callers = sorted(str(p.relative_to(ROOT)) for d in ("src", "scripts", "tests", "web/src/py")
                     for p in (ROOT / d).rglob("*.py") if _calls_instantiate(p))
    assert callers == ["src/reblock/presets.py"]
