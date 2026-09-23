"""The authoring bundle: the block Pyodide rebuilds, at full precision.

Full float64 and not the cm-rounded form every render bundle ships. MEASURED on this bundle's own
reference roads (`scripts/gen_authoring_block.REFERENCE_ROADS`): rounding every coordinate to
centimetres and re-solving, baseline included, moves `crossing` by 5.3129e-05 and `spur` by
4.6081e-05. Design §1.4's 4.71e-05 is the same effect measured on the clearance METHOD's road set;
`spur`'s is the smallest of the three, and therefore the binding anchor
`web/test/pyodide-parity.test.ts` sizes `PARITY_TOL` against.

`web/test/pyodide-parity.test.ts` asserts the browser reproduces CPython's number on the SAME
bundle, and a tolerance widened past 4.6e-05 -- the smallest of the effects a parity guard must
not absorb -- would be one that has stopped measuring the runtime. That is what these tests keep
the bundle fit for.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tests.dts_keys import json_keys, ts_field_names

OUT = Path("examples/authoring/block.json")
README = Path("examples/authoring/README.md")
DTS = Path("web/src/authoring.d.ts")
GEN_AUTHORING_BLOCK = Path("scripts/gen_authoring_block.py")
# The Explore page's spine block. Moved from ZAF.9.3.1_1_40972 (263 parcels) on
# 2026-09-20 -- see `scripts/_example_block.py`. `tests/test_screen_map_bundle.py`'s
# SPINE_SOURCES is what keeps the five bundles agreeing on it; this constant is one
# baker's end of that.
SPINE = "ZAF.9.3.1_1_5810"


@pytest.fixture(scope="session")
def bundle() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(OUT.read_text(encoding="utf-8"))
    return result


def test_dts_declares_exactly_the_keys_the_bundle_carries(bundle: dict[str, Any]) -> None:
    assert json_keys(bundle) - ts_field_names(DTS.read_text(encoding="utf-8")) == set()
    assert ts_field_names(DTS.read_text(encoding="utf-8")) - json_keys(bundle) == set()


def test_the_committed_dts_is_what_the_generator_writes() -> None:
    """The key-set check above compares NAMES, so it is blind to a hand edit that leaves them
    alone -- `number[]` changed to `string[]`, a `?` made optional, a doc comment deleted. Design
    §9 says generated files are never hand-edited; `tests/test_web_bundle.py`,
    `tests/test_frontier_bundle.py` and `tests/test_displacement_field_bundle.py` each enforce it
    the same way."""
    from scripts.gen_authoring_block import DTS_TEMPLATE
    assert DTS.read_text(encoding="utf-8") == DTS_TEMPLATE, (
        "web/src/authoring.d.ts was hand-edited; regenerate it: "
        "pixi run python -m scripts.gen_authoring_block")


def test_the_committed_readme_is_what_the_generator_writes() -> None:
    """Every number in it is read out of the bundle, so a stale README is a bundle and a
    description of it that disagree -- the failure `examples/nairobi/README.md` already had once
    (89 blocks claimed, 43 in every `meta.json` beside it).

    Reads the committed JSON into the generator's own TypedDict rather than taking the untyped
    `bundle` fixture, which is what `tests/test_screen_map_bundle.py` does for the same pin."""
    from scripts.gen_authoring_block import AuthoringBundle, readme_markdown
    typed: AuthoringBundle = json.loads(OUT.read_text(encoding="utf-8"))
    assert README.read_text(encoding="utf-8") == readme_markdown(typed), (
        "examples/authoring/README.md is stale or hand-edited; regenerate it: "
        "pixi run python -m scripts.gen_authoring_block")


def test_the_bundle_carries_the_configured_params(bundle: dict[str, Any]) -> None:
    """The browser builds its `PermeabilityParams` from the bundle's `params` -- `conf/` does not
    travel with the wheel `micropip` installs -- so a yaml edit with no re-bake would leave this
    widget scoring roads by different parameters from every other figure on the site. Every field,
    compared exactly, both ways."""
    import dataclasses

    from reblock.compare import load_permeability_config
    shipped = load_permeability_config().params
    baked = bundle["params"]
    assert set(baked) == {f.name for f in dataclasses.fields(shipped)}
    for name, value in baked.items():
        assert value == getattr(shipped, name), (
            f"field {name}: bundle {value!r} vs conf {getattr(shipped, name)!r}; re-bake: "
            f"pixi run python -m scripts.gen_authoring_block")


def test_it_is_the_spine_block_and_a_projected_crs(bundle: dict[str, Any]) -> None:
    """The same block every other stage of the site follows. 32734 is UTM 34S, projected --
    `Block.__post_init__` rejects a geographic CRS, and that failure in the browser would surface
    as a Pyodide traceback in a figcaption rather than as anything a reader could act on."""
    assert bundle["block_id"] == SPINE
    assert bundle["crs_epsg"] == 32734


def test_the_coordinates_are_not_quantised(bundle: dict[str, Any]) -> None:
    """The defect this bundle exists to avoid. A cm-rounded coordinate is an exact multiple of
    0.01, so a bundle that went through `_bundle_io.cm` would have EVERY coordinate land on that
    grid. Real UTM metres do not."""
    xs = [p[0] for ring in bundle["parcels"] for r in ring for p in r]
    on_grid = sum(1 for x in xs if math.isclose(x, round(x, 2), rel_tol=0, abs_tol=1e-12))
    assert on_grid < len(xs) // 2, (
        f"{on_grid} of {len(xs)} parcel x-coordinates sit exactly on the cm grid; this bundle "
        "must ship full float64 (design §1.4)")


def test_every_per_parcel_column_has_one_entry_per_parcel(bundle: dict[str, Any]) -> None:
    n = len(bundle["parcels"])
    assert len(bundle["parcel_id"]) == n
    assert len(bundle["nodes"]["cx"]) == n
    assert len(bundle["nodes"]["cy"]) == n
    assert len(bundle["nodes"]["ground"]) == n
    assert len(bundle["baseline"]["potential"]) == n


def test_every_per_edge_column_has_one_entry_per_edge(bundle: dict[str, Any]) -> None:
    m = len(bundle["edges"]["rows"])
    assert len(bundle["edges"]["cols"]) == m
    assert len(bundle["edges"]["footpath_g"]) == m


def test_edge_endpoints_are_valid_parcel_indices(bundle: dict[str, Any]) -> None:
    """An out-of-range index would reach `potential[rows[i]]` in the widget's own current
    calculation and read `undefined`, which arithmetic turns into NaN and canvas turns into
    nothing drawn -- silent, and shaped exactly like an empty graph."""
    n = len(bundle["parcels"])
    for key in ("rows", "cols"):
        assert all(0 <= i < n for i in bundle["edges"][key]), key
    # `footpath_mesh` stores each undirected edge ONCE, `continue`-ing on `j <= i`, so this is the
    # mesh's own invariant rather than an incidental property of this block. Asserted because the
    # range check above cannot see a shifted `rows` column: `rows` maxes out at n-2, so adding 1 to
    # every entry leaves it inside [0, n) and reddens nothing.
    assert all(r < c for r, c in zip(bundle["edges"]["rows"], bundle["edges"]["cols"],
                                     strict=True))


def test_building_points_are_one_per_parcel(bundle: dict[str, Any]) -> None:
    """`mesh.parcel_radii` resolves point-to-parcel by CONTAINMENT, not by index -- parcels are
    Voronoi cells of these points, so the count matches while the order does not."""
    assert len(bundle["building_points"]) == len(bundle["parcels"])


def test_every_building_carries_a_positive_radius(bundle: dict[str, Any]) -> None:
    """The browser rebuilds the block as `Discs(building_points, building_radii)`, so the two are
    one list read in lock step: a short `building_radii` raises there, and a zero radius opens that
    parcel's footpath edges completely (`parcel_radii`'s fallback), which the parity tests would
    only catch if a reference road happened to route through it."""
    assert len(bundle["building_radii"]) == len(bundle["building_points"])
    assert all(r > 0.0 for r in bundle["building_radii"])


def test_the_baseline_is_the_no_roads_solve(bundle: dict[str, Any]) -> None:
    """Road-invariant (design §1.6), so it is baked rather than recomputed in the browser on
    every edit. A finite positive p0 is what `permeability` divides by."""
    assert math.isfinite(bundle["baseline"]["p0"])
    assert bundle["baseline"]["p0"] > 0.0


def test_reference_cases_are_present_and_shaped(bundle: dict[str, Any]) -> None:
    """What `web/test/pyodide-parity.test.ts` compares the browser against, so that the parity
    test needs no Python process at test time -- the same device `field.json` and `hood.json`
    already carry a `reference` array of production's own answers for.

    Shape only. The VALUES are checked where they are produced: `gen_authoring_block`'s bake-time
    round trip re-solves each road on a block rebuilt from this JSON and refuses to write on any
    difference at all."""
    ref = bundle["reference"]
    assert len(ref) >= 2
    for c in ref:
        assert c["name"] and len(c["road"]) >= 2
        assert all(len(p) == 2 for p in c["road"])
        assert math.isfinite(c["permeability"])


def _type_checking_solve_imports(tree: ast.Module) -> set[str]:
    """Every name `scripts/gen_authoring_block.py` imports from `web.src.py.solve` inside its
    `if TYPE_CHECKING:` block -- the STATIC half `pixi run typecheck` actually looks at."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"):
            for stmt in node.body:
                if isinstance(stmt, ast.ImportFrom) and stmt.module == "web.src.py.solve":
                    names.update(alias.asname or alias.name for alias in stmt.names)
    return names


def _runtime_solve_bindings(tree: ast.Module) -> set[str]:
    """Every name `scripts/gen_authoring_block.py` binds off the `importlib`-loaded module inside
    its `if not TYPE_CHECKING:` block -- the RUNTIME half that actually executes, and that mypy
    never looks inside (it treats `TYPE_CHECKING` as always true). `_solve_mod` itself is excluded:
    it is the loaded module, not one of the names re-exported from it."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not) and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id == "TYPE_CHECKING"):
            for stmt in node.body:
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)
                        and stmt.targets[0].id != "_solve_mod"):
                    names.add(stmt.targets[0].id)
    return names


def test_the_type_checking_and_runtime_solve_bindings_name_the_same_set() -> None:
    """`scripts/gen_authoring_block.py` gets `AuthoringBundle` and its four sibling `TypedDict`s
    plus `block_from_bundle` from `web/src/py/solve.py` through TWO independent, hand-maintained
    lists: a `TYPE_CHECKING`-guarded static import (what mypy checks) and a parallel
    `if not TYPE_CHECKING:` block of `NAME = _solve_mod.NAME` assignments (what actually runs at
    import time) -- two lists because mypy treats `TYPE_CHECKING` as always true and therefore
    never looks inside the second block at all, so nothing type-checked ever sees it.

    DEMONSTRATED: deleting `ReferenceCase = _solve_mod.ReferenceCase` from the runtime block, with
    `solve.py` and the static import both left untouched, leaves `pixi run typecheck` reporting
    'Success: no issues found' -- mypy only ever sees the (unmodified) `TYPE_CHECKING` branch --
    while `pixi run python -m scripts.gen_authoring_block` raises `NameError: name 'ReferenceCase'
    is not defined`. That asymmetry (checked-but-not-run vs. run-but-not-checked) is exactly what
    this test exists to catch statically, before either command runs.
    """
    tree = ast.parse(GEN_AUTHORING_BLOCK.read_text(encoding="utf-8"))
    type_checking = _type_checking_solve_imports(tree)
    runtime = _runtime_solve_bindings(tree)
    # Guards the guard: an empty/empty comparison would pass vacuously if the AST shapes this
    # walks for ever stopped matching the file (a refactor of the `if`/`import` structure, say).
    assert type_checking, "no `if TYPE_CHECKING:` import from web.src.py.solve found at all"
    assert type_checking == runtime, (
        f"only under TYPE_CHECKING (checked, not run): {sorted(type_checking - runtime)}; only at "
        f"runtime (run, not checked): {sorted(runtime - type_checking)}. A name in exactly one "
        f"list is either type-checked without ever running, or runs without ever being checked.")
