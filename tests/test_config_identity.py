"""A configured method's cache key covers every setting that can change its roads.

Each method once spelled its identity by hand -- a second copy of its field list -- and the copies
drifted: most omitted `road_width_m`, two their `PermeabilityParams`, so a sweep over any of those
returned another setting's cached roads with no error. Identities are now DERIVED from the fields
(`derive_graph.config_identity`); these tests hold every configured method to that, through the
nested objects it is configured with.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Hashable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from reblock.derive_graph import Identified, config_identity
from reblock.presets import load_methods

# Settings that cannot change the output, by qualified field name -- each the one field its class
# exempts. A new entry here needs a reason in that class.
CANNOT_CHANGE_OUTPUT = {"GreedyArterialReblocker.workers", "ShortlistEngine.threads",
                        "BetweennessDesire.workers"}

# Needs a live fetch to construct a cacheable instance; their keys are covered by the helper tests.
NEEDS_NETWORK = {"osm_footpaths", "demand_greedy"}


@dataclass(frozen=True)
class _Leaf:
    a: int = 1
    b: float = 2.0


@dataclass(frozen=True)
class _Source:
    identity: Hashable | None


@dataclass(frozen=True)
class _Config:
    leaf: _Leaf = field(default_factory=_Leaf)
    source: _Source = field(default_factory=lambda: _Source(("src", 1)))
    flag: bool = False
    workers: int = 4


def test_every_field_is_in_the_key_unless_exempt() -> None:
    base = _Config()
    for changed in (dataclasses.replace(base, flag=True),
                    dataclasses.replace(base, leaf=_Leaf(a=2)),
                    dataclasses.replace(base, source=_Source(("src", 2)))):
        assert config_identity(changed) != config_identity(base), changed
    assert config_identity(dataclasses.replace(base, workers=8), exempt=frozenset({"workers"})) \
        == config_identity(base, exempt=frozenset({"workers"}))


def test_a_nested_uncacheable_input_makes_the_whole_config_uncacheable() -> None:
    assert config_identity(_Config(source=_Source(None))) is None


def test_an_exemption_must_name_a_real_field() -> None:
    with pytest.raises(ValueError, match="does not have"):
        config_identity(_Config(), exempt=frozenset({"wokers"}))


def test_a_value_with_no_identity_raises_rather_than_being_left_out() -> None:
    @dataclass(frozen=True)
    class Holder:
        thing: object

    with pytest.raises(TypeError, match="has no identity"):
        config_identity(Holder(thing=object()))


def _configured_methods() -> dict[str, Identified]:
    with initialize_config_dir(version_base=None, config_dir=str(Path("conf").resolve())):
        cfg = compose(config_name="compare_config", overrides=["shapefile=x"])
    out: dict[str, Identified] = {}
    for name, method in load_methods(cfg.all_methods).items():
        if name in NEEDS_NETWORK:
            continue
        assert isinstance(method, Identified), name
        out[name] = method
    return out


def _perturbed(obj: object) -> Iterator[tuple[str, object]]:
    """Every copy of `obj` with ONE numeric or boolean setting changed, reached through nested
    dataclasses, each labelled `Class.field`."""
    assert dataclasses.is_dataclass(obj) and not isinstance(obj, type)
    for f in dataclasses.fields(obj):
        # By name over the declared schema, no default: a field that is not there raises.
        value = getattr(obj, f.name)
        label = f"{type(obj).__name__}.{f.name}"
        new: object
        if isinstance(value, bool):
            new = not value
        elif isinstance(value, int):
            new = value + 1
        elif isinstance(value, float):
            new = value * 1.5 + 0.25
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for inner_label, inner in _perturbed(value):
                yield inner_label, dataclasses.replace(obj, **{f.name: inner})
            continue
        else:
            continue
        yield label, dataclasses.replace(obj, **{f.name: new})


@pytest.mark.parametrize("name", sorted(_configured_methods()))
def test_every_setting_of_every_configured_method_changes_its_key(name: str) -> None:
    """FAULT INJECTION: dropping `road_width_m` from a method's key (an `exempt` entry, or a hand
    identity that forgets it) fails this for that method."""
    method = _configured_methods()[name]
    key = method.identity
    assert key is not None, f"{name} is configured cacheable but has no key"
    outside, split = [], []
    for label, changed in _perturbed(method):
        assert isinstance(changed, Identified)
        same = changed.identity == key
        if same and label not in CANNOT_CHANGE_OUTPUT:
            outside.append(label)
        if not same and label in CANNOT_CHANGE_OUTPUT:
            split.append(label)
    assert not outside, (
        f"{name}: changing {outside} leaves its cache key unchanged, so a sweep over it returns "
        f"another setting's cached roads")
    assert not split, f"{name}: {split} cannot change the roads but splits the cache key"
