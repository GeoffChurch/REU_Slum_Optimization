"""`web/src/py/solve.py` under CPython.

This module is the payload Pyodide runs, and it is plain Python -- so its logic is tested here,
where a failure is a stack trace rather than a browser console.

DEBT: `web/test/pyodide-parity.test.ts` does not exist at this commit -- Task 5 of this piece adds
it. It will have one job only: prove the RUNTIME agrees with what this file already proves about
the LOGIC. Splitting it that way is what will keep that expensive test cheap to interpret.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from reblock.contracts import Block


# `web/src/py/` is not an importable package -- no `__init__.py`, and `web/` is not on the path
# -- because it is a directory of things that ship to a browser, not a Python package. Load it by
# path, the way `scripts/gen_site_pages.py` already loads `method_labels.py`.
def _load_solve():
    spec = importlib.util.spec_from_file_location("solve", Path("web/src/py/solve.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_solve_mod = _load_solve()
block_from_bundle = _solve_mod.block_from_bundle
solve = _solve_mod.solve

BUNDLE = json.loads(Path("examples/authoring/block.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def block():
    return block_from_bundle(BUNDLE)


def test_reconstruction_matches_the_source_block(block: Block) -> None:
    assert block.block_id == BUNDLE["block_id"]
    assert len(block.parcels) == len(BUNDLE["parcels"])
    assert len(block.building_points) == len(BUNDLE["building_points"])


def test_every_reference_case_reproduces_its_baked_answer(block: Block) -> None:
    """Exact equality, not a tolerance: both sides are CPython on the same float64 input, so any
    difference is a defect rather than noise. The parity test is where a tolerance belongs, and
    only if the two RUNTIMES are shown to disagree."""
    for case in BUNDLE["reference"]:
        got = solve(block, case["road"], BUNDLE["baseline"]["p0"])
        assert got["permeability"] == case["permeability"], case["name"]


def test_the_returned_arrays_are_the_shapes_the_widget_indexes(block: Block) -> None:
    """The widget's own per-edge current calculation is `conductance[i] * (potential[rows[i]] -
    potential[cols[i]])` (design §1.6), so a short array would read `undefined` there, become
    NaN, and draw nothing -- indistinguishable from an empty graph."""
    got = solve(block, BUNDLE["reference"][0]["road"], BUNDLE["baseline"]["p0"])
    assert len(got["potential"]) == len(BUNDLE["parcels"])
    assert len(got["conductance"]) == len(BUNDLE["edges"]["rows"])


def test_road_length_is_the_polyline_length(block: Block) -> None:
    road = [[0.0, 0.0], [3.0, 4.0], [3.0, 14.0]]
    got = solve(block, road, BUNDLE["baseline"]["p0"])
    assert math.isclose(got["roadMetres"], 15.0, rel_tol=1e-12)


def test_a_road_that_is_a_single_point_is_refused(block: Block) -> None:
    """Refused HERE as well as in the widget: the widget's check is for the reader, this one is
    the contract. Without this guard, a one-point 'polyline' would reach shapely's `LineString`
    constructor and raise `GEOSException: IllegalArgumentException: point array must contain 0
    or >1 elements` -- not a `ValueError`, and not a silently-built degenerate line. MEASURED by
    fault injection: deleting the guard above reddens this test with that GEOSException
    uncaught, not with `pytest.raises` reporting a mismatch."""
    with pytest.raises(ValueError, match="at least two"):
        solve(block, [[0.0, 0.0]], BUNDLE["baseline"]["p0"])
