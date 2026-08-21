"""The committed RegionGrow bundle: schema, .d.ts parity, and identity against production.

One @pytest.mark.slow test carries every assertion that needs the city parquet. pytest-xdist scopes
`scope="module"` fixtures PER WORKER, not per session, so a module-scoped city load runs once per
worker and D2 lost 18 minutes to exactly that. Same shape as tests/test_frontier_bundle.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.dts_keys import json_keys, ts_field_names

BUNDLE = Path("examples/region-grow/hood.json")
DTS = Path("web/src/hood.d.ts")

pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="bundle not baked")


@pytest.fixture(scope="session")
def bundle() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(BUNDLE.read_text(encoding="utf-8"))
    return result


def test_dts_declares_exactly_the_keys_the_bundle_carries(bundle: dict[str, Any]) -> None:
    """Bidirectional: a field renamed in Python becomes a TypeScript error, and a field declared
    but never emitted is caught too."""
    declared = ts_field_names(DTS.read_text(encoding="utf-8"))
    carried = json_keys(bundle)
    assert carried - declared == set(), "carried but not declared"
    assert declared - carried == set(), "declared but not carried"


def test_adjacency_is_symmetric_and_excludes_self(bundle: dict[str, Any]) -> None:
    adj = {i: set(b["adj"]) for i, b in enumerate(bundle["blocks"])}
    for i, neighbours in adj.items():
        assert i not in neighbours, f"block {i} is adjacent to itself"
        for j in neighbours:
            assert i in adj[j], f"{i}->{j} is not mirrored"


def test_every_coordinate_is_at_centimetre_precision(bundle: dict[str, Any]) -> None:
    """`cm` rounds to 2 dp. A coordinate carrying more is one that bypassed the quantiser."""
    for b in bundle["blocks"]:
        for ring in b["rings"]:
            for x, y in ring:
                assert round(x, 2) == x and round(y, 2) == y, (b["block_id"], x, y)


def test_the_neighbourhood_carries_its_holed_blocks(bundle: dict[str, Any]) -> None:
    """7 of the 213 blocks have an interior ring, measured. If this drops to 0 the bundle went
    through `polygon_ring` (which would have raised) or a ring list got flattened -- neither of
    which changes any count the other tests check."""
    holed = [b["block_id"] for b in bundle["blocks"] if len(b["rings"]) > 1]
    assert sorted(holed) == [
        "ZAF.9.3.1_1_38616", "ZAF.9.3.1_1_38935", "ZAF.9.3.1_1_40664", "ZAF.9.3.1_1_40963",
        "ZAF.9.3.1_1_41055", "ZAF.9.3.1_1_41838", "ZAF.9.3.1_1_41976",
    ]


def test_the_shipped_budget_floor_is_a_no_op_on_the_seed(bundle: dict[str, Any]) -> None:
    """At `budget.min` the region is the seed ALONE -- the design's §1.3 finding, which the
    widget's caption states. If this ever changes, that caption is wrong."""
    floor = [c for c in bundle["reference"] if c["max_buildings"] == bundle["budget"]["min"]
             and c["seed"] == bundle["seed"]]
    assert len(floor) == 1, "the floor budget must be among the reference cases"
    assert floor[0]["order"] == [bundle["seed"]]


def test_reference_cases_are_prefixes_of_one_another(bundle: dict[str, Any]) -> None:
    """Growth is nested (design §1.4), so for one seed a bigger budget's order must EXTEND a
    smaller one's, not merely contain it. A set-containment assertion would pass against a
    reordering, and order is the whole teaching point."""
    seeds = {c["seed"] for c in bundle["reference"]}
    assert seeds, "no reference cases at all would make every loop below iterate zero times"
    for seed in seeds:
        cases = sorted((c for c in bundle["reference"] if c["seed"] == seed),
                       key=lambda c: c["max_buildings"])
        # Without this, a seed carrying ONE case makes `zip(cases, cases[1:])` empty and the
        # assertion below never runs -- the loop passes by not executing.
        assert len(cases) >= 2, f"{seed} has {len(cases)} reference case(s); nothing to compare"
        for small, big in zip(cases, cases[1:], strict=False):
            assert big["order"][:len(small["order"])] == small["order"], (
                seed, small["max_buildings"], big["max_buildings"])


@pytest.mark.slow
def test_bundle_is_what_production_builds_today() -> None:
    """The identity test, and the reason Task 1 exists: every reference case is recomputed by
    calling `DenseClusterRegionBuilder` ITSELF, not a copy of its rule. Also re-derives the
    neighbourhood, so a stale bundle fails here rather than shipping.

    `hops=HOPS` (the generator's own constant), NOT a re-typed `7`: this is the exact value a
    stale-neighbourhood regression would drift against, and re-typing it here would make the
    re-typed copy, not the generator, the thing this test actually pins.
    """
    from reblock.region import DenseClusterRegionBuilder
    from scripts.gen_region_grow import HOPS, load_blocks, neighbourhood

    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    blocks = load_blocks(bundle["city"])
    assert blocks.crs is not None and not blocks.crs.is_geographic, (
        "the bake must project before it grows -- dwithin(0.5) is 55 km in lon/lat")

    ids = [b["block_id"] for b in bundle["blocks"]]
    assert neighbourhood(blocks, bundle["seed"], hops=HOPS) == ids, "stale neighbourhood"

    for case in bundle["reference"]:
        got = DenseClusterRegionBuilder(max_buildings=case["max_buildings"]).build(
            blocks, [[case["seed"]]])[0]
        assert got == case["order"], (case["seed"], case["max_buildings"])
