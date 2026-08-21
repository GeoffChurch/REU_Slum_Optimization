"""The committed city tier: schema, column alignment, and precision/recall against the bake-off CSV.

The heavy test is ONE @pytest.mark.slow (see tests/test_region_grow_bundle.py's docstring for why).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tests.dts_keys import json_keys, ts_field_names

OUT = Path("examples/screen-map")
DTS = Path("web/src/screen_map.d.ts")
CSV_PATH = Path("examples/screen-bakeoff/screen_comparison.csv")

pytestmark = pytest.mark.skipif(not (OUT / "capetown.json").exists(), reason="tier not baked")

# The bundle's `reblock.metric`-style metric names -> gen_screen_bakeoff.py's own display strings
# in screen_comparison.csv's `metric` column (that script's own METRICS list). Explicit and closed:
# the row lookup this replaces matched by `startswith(floor["metric"].split("_")[0])`, which is
# exactly the fragile runtime-string reach into a known-at-authoring-time set this project's own
# methodology forbids -- "depth_density_proxy" and "density_compactness" share the prefix "density"
# once split on "_", so that lookup was one rename away from picking the wrong row silently.
METRIC_CSV_NAME = {
    "depth_density_proxy": "depth_density proxy   √(nA)/P · n/A",
    "density": "density   n/A",
    "density_compactness": "density_compactness   n/P²",
    "depth_proxy": "depth proxy   √(nA)/P",
}


def _metric(name: str, n: float, a: float, p: float) -> float:
    """The four cheap screens (design §3.1), recomputed independently of `reblock.metric` and of
    `scripts/gen_screen_map.py`'s own `_score` -- two paths computing the same formula and agreeing
    is the guard; importing either would make this a test of the import, not the arithmetic."""
    if name == "depth_density_proxy":
        return math.sqrt(n * a) / p * (n / a)
    if name == "density":
        return n / a
    if name == "density_compactness":
        return n / p ** 2
    if name == "depth_proxy":
        return math.sqrt(n * a) / p
    raise ValueError(name)


@pytest.fixture(scope="session")
def capetown() -> dict[str, Any]:
    result: dict[str, Any] = json.loads((OUT / "capetown.json").read_text(encoding="utf-8"))
    return result


@pytest.fixture(scope="session")
def nairobi() -> dict[str, Any]:
    result: dict[str, Any] = json.loads((OUT / "nairobi.json").read_text(encoding="utf-8"))
    return result


def test_dts_declares_exactly_the_keys_both_bundles_carry(capetown: dict[str, Any],
                                                          nairobi: dict[str, Any]) -> None:
    declared = ts_field_names(DTS.read_text(encoding="utf-8"))
    carried = json_keys(capetown) | json_keys(nairobi)
    assert carried - declared == set(), "carried but not declared"
    # `informal` is declared optional and carried only by Cape Town, so it is in `carried`.
    assert declared - carried == set(), "declared but not carried"


@pytest.mark.parametrize("city", ["capetown", "nairobi"])
def test_every_column_has_n_blocks_entries(city: str, request: pytest.FixtureRequest) -> None:
    """A truncated column would shorten the map without changing its shape -- no error, no blank
    canvas, just fewer blocks than the city has."""
    b = request.getfixturevalue(city)
    for column in ("block_id", "n", "area_m2", "perimeter_m", "rings"):
        assert len(b[column]) == b["n_blocks"], (city, column)


def test_capetown_carries_ground_truth_and_nairobi_does_not(capetown: dict[str, Any],
                                                             nairobi: dict[str, Any]) -> None:
    """Nairobi has no published informal layer (reblock.data.informal records the search). The
    field is ABSENT, not null -- a null column is a field that looks answerable and is not."""
    assert len(capetown["informal"]) == capetown["n_blocks"]
    assert set(capetown["informal"]) <= {0, 1}
    assert "informal" not in nairobi


def test_the_interior_rings_survived(capetown: dict[str, Any], nairobi: dict[str, Any]) -> None:
    """Measured: 6,990 Cape Town and 1,139 Nairobi blocks have a hole. Losing them changes no
    count any other test here checks."""
    assert sum(len(r) - 1 for r in capetown["rings"]) == 6990
    assert sum(len(r) - 1 for r in nairobi["rings"]) == 1139


def test_precision_and_recall_at_the_shipped_floor_match_the_bakeoff(
        capetown: dict[str, Any]) -> None:
    """Two independently computed paths agreeing. The CSV comes from gen_screen_bakeoff.py's own
    ranking; this recomputes from the bundle's raw n/A/P and ground-truth column. The numbers are
    READ from the CSV, never restated here -- a literal would make this a test of my typing.
    """
    rows = {r["metric"]: r for r in csv.DictReader(CSV_PATH.open(encoding="utf-8"))
            if r.get("floor")}
    assert rows, "the bake-off CSV must carry at least one shipped floor"

    n = capetown["n"]
    a = capetown["area_m2"]
    p = capetown["perimeter_m"]
    informal = capetown["informal"]
    total_informal = sum(informal)

    for floor in capetown["floors"]:
        if floor["precision"] is None:
            continue
        row = rows[METRIC_CSV_NAME[floor["metric"]]]
        scores = [_metric(floor["metric"], n[i], a[i], p[i]) for i in range(capetown["n_blocks"])]
        selected = [i for i, s in enumerate(scores) if s >= floor["value"]]
        hits = sum(informal[i] for i in selected)
        assert len(selected) == int(float(row["floor_n"])), floor["metric"]
        assert math.isclose(hits / len(selected), float(row["floor_prec"]), rel_tol=1e-6)
        assert math.isclose(hits / total_informal, float(row["floor_recall"]), rel_tol=1e-6)
