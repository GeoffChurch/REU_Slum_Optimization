"""`web/src/py/solve.py` under CPython.

This module is the payload Pyodide runs, and it is plain Python -- so its logic is tested here,
where a failure is a stack trace rather than a browser console.

`web/test/pyodide-parity.test.ts` has one job only: prove the RUNTIME agrees with what this file
already proves about the LOGIC. Splitting it that way is what keeps that expensive test cheap to
interpret. MEASURED there: on `examples/authoring/block.json`'s two reference roads the two
runtimes agree exactly on `crossing` and differ by four units in the last place (2.220446e-16) on
`spur`, so that test states an absolute tolerance rather than exact equality; the reasoning is in
its own `PARITY_TOL`.
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
    assert len(block.building_geometries) == len(BUNDLE["building_points"])


#: How far a fresh CPython solve may sit from the baked answer. See the test below for why this
#: is a tolerance rather than exact equality, and where the number comes from.
BAKE_TOL = 1e-12


def test_every_reference_case_reproduces_its_baked_answer(block: Block) -> None:
    """A stated tolerance, because this comparison spans two MACHINES, not two runtimes.

    This asserted exact equality until it failed on GitHub's runner: `crossing` came back as
    0.699004769769015 against the baked 0.6990047697690152, a difference of 2.2e-16 -- about two
    units in the last place at that magnitude. The same commit had passed the same assertion on a
    different runner minutes earlier, so it is not a defect and not a stale bake: `permeability`
    ends in a sparse solve, and the BLAS kernel that solve dispatches to depends on the CPU it
    finds. The bundle's numbers were baked on one machine; every machine that reads them back is a
    different one.

    The old docstring here said "both sides are CPython on the same float64 input, so any
    difference is a defect rather than noise". The input is the same; the arithmetic underneath it
    is not.

    1e-12 is stated, not discovered:

    * ~4,500x the 2.2e-16 actually observed between two runners, so genuine BLAS variation has room
      that a 4x margin would not give it -- and unlike the runtime-parity comparison, this one is a
      distribution across machines rather than a constant between two fixed artifacts;
    * ~4.7e7 times SMALLER than 4.71e-05, the smallest effect `web/src/authoring.d.ts` records as
      one a guard on these numbers must not absorb. A wrong reconstruction -- a lost coordinate, a
      changed ring winding, a dtype shift -- moves the answer by orders of magnitude more than
      this, which is what this test exists to catch.

    The runtime-parity guard keeps its own much tighter 1e-15: both of ITS sides are fixed (wasm
    f64 is deterministic by specification, and the baked side is committed), so it compares a
    constant, not a distribution. A re-bake on a different machine would move the baked side by
    roughly the 2 ULP seen here, which that tolerance still absorbs with room to spare.
    """
    for case in BUNDLE["reference"]:
        got = solve(block, case["road"], BUNDLE["baseline"]["p0"])
        difference = abs(got["permeability"] - case["permeability"])
        assert difference <= BAKE_TOL, (
            f"{case['name']}: solved {got['permeability']!r}, baked {case['permeability']!r} "
            f"(|difference| {difference:.6e}, tolerance {BAKE_TOL:.0e})")


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
