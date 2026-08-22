"""The authoring bundle: the block Pyodide reconstructs, at full precision.

Full float64 and not the cm-rounded form every render bundle ships (design §1.4): quantising
moves permeability by 4.71e-05 on this block, and `web/test/pyodide-parity.test.ts` asserts the
browser reproduces CPython's number on the SAME bundle -- a tolerance that would have to be
widened past the quantisation error to accommodate rounding is a tolerance that has stopped
measuring the runtime.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from tests.dts_keys import json_keys, ts_field_names

OUT = Path("examples/authoring/block.json")
DTS = Path("web/src/authoring.d.ts")
SPINE = "ZAF.9.3.1_1_40972"


@pytest.fixture(scope="session")
def bundle() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(OUT.read_text(encoding="utf-8"))
    return result


def test_dts_declares_exactly_the_keys_the_bundle_carries(bundle: dict[str, Any]) -> None:
    assert json_keys(bundle) - ts_field_names(DTS.read_text(encoding="utf-8")) == set()
    assert ts_field_names(DTS.read_text(encoding="utf-8")) - json_keys(bundle) == set()


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


def test_building_points_are_one_per_parcel(bundle: dict[str, Any]) -> None:
    """`mesh.parcel_radii` resolves point-to-parcel by CONTAINMENT, not by index -- parcels are
    Voronoi cells of these points, so the count matches while the order does not."""
    assert len(bundle["building_points"]) == len(bundle["parcels"])


def test_the_baseline_is_the_no_roads_solve(bundle: dict[str, Any]) -> None:
    """Road-invariant (design §1.6), so it is baked rather than recomputed in the browser on
    every edit. A finite positive p0 is what `permeability` divides by."""
    assert math.isfinite(bundle["baseline"]["p0"])
    assert bundle["baseline"]["p0"] > 0.0


def test_reference_cases_are_present_and_shaped(bundle: dict[str, Any]) -> None:
    """What `web/test/pyodide-parity.test.ts` compares the browser against, so that the parity
    test needs no Python process at test time -- the same device `field.json` and `hood.json`
    already use."""
    ref = bundle["reference"]
    assert len(ref) >= 2
    for c in ref:
        assert c["name"] and len(c["road"]) >= 2
        assert all(len(p) == 2 for p in c["road"])
        assert math.isfinite(c["permeability"])
