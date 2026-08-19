"""Nothing recomputes this bundle between the baker and the browser, so these tests are the only
thing between a bad bake and a chart that reads wrong."""
import json
import math
import re
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

BUNDLE = Path("examples/method-comparison/frontier.json")
DTS = Path("web/src/frontier.d.ts")
LENS = Path("examples/method-comparison/lens_permeability.csv")
RUN_LOG = Path("examples/method-comparison/run.log")


@pytest.fixture(scope="module")
def bundle() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(BUNDLE.read_text(encoding="utf-8"))
    return result


def test_every_curve_is_internally_consistent(bundle: dict[str, Any]) -> None:
    assert len(bundle["methods"]) == 8
    for name, c in bundle["methods"].items():
        n = len(c["road_m"])
        assert len(c["displacement"]) == len(c["permeability"]) == n, name
        assert n > 1, name


def test_both_axes_are_monotone(bundle: dict[str, Any]) -> None:
    """Guards that the widget's target search (a binary search over each curve) is VALID -- not
    that the roads are correctly ordered. road_m/displacement/permeability are monotone in the SIZE
    of a growing prefix regardless of which specific roads have accumulated by step m, so a wrong
    drainage ordering still passes this test (confirmed empirically: task-3-report.md's row-1 fault
    injection swapped in unordered roads and this test stayed green). Ordering is guarded by
    test_terminals_agree_with_the_committed_lens_csv instead."""
    for name, c in bundle["methods"].items():
        for key in ("road_m", "displacement", "permeability"):
            v = c[key]
            assert all(b >= a - 1e-9 for a, b in zip(v, v[1:], strict=False)), \
                f"{name}.{key} not monotone"


def test_starts_at_zero(bundle: dict[str, Any]) -> None:
    for name, c in bundle["methods"].items():
        assert c["road_m"][0] == 0.0, name
        assert c["displacement"][0] == 0.0, name
        assert c["permeability"][0] == 0.0, name


def test_targets_match_the_live_config(bundle: dict[str, Any]) -> None:
    """The widget boots its target lines here, and the fallback PNG draws its dashed guides from the
    same config. If these drift, the widget contradicts the image it replaces."""
    from reblock.compare import load_permeability_config

    pcfg = load_permeability_config()
    assert bundle["matched_displacement"] == pytest.approx(pcfg.matched_displacement)
    assert bundle["matched_permeability"] == pytest.approx(pcfg.matched_permeability)
    assert bundle["frontier_xmax"] == pytest.approx(pcfg.frontier_xmax)


def _json_keys(node: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(node, dict):
        for k, v in node.items():
            out.add(k)
            out |= _json_keys(v)
    elif isinstance(node, list):
        for v in node:
            out |= _json_keys(v)
    return out


def test_dts_declares_the_bundle_keys(bundle: dict[str, Any]) -> None:
    """Catches 'regenerated one file, not the other'. Recursive, because every field the chart reads
    per frame is nested -- a top-level-only check would miss all of them."""
    declared = set(re.findall(r"^\s+(\w+)[?]?:", DTS.read_text(encoding="utf-8"), flags=re.M))
    # Method names are data, not declared fields: `methods` is a Record<string, MethodCurve>.
    keys = _json_keys(bundle) - set(bundle["methods"])
    assert keys <= declared, f"bundle keys missing from frontier.d.ts: {sorted(keys - declared)}"


def test_terminals_agree_with_the_committed_lens_csv(bundle: dict[str, Any]) -> None:
    """Artifact vs artifact -- the check that catches a CHANGED PIN, which the parity test
    structurally cannot, because it shares its loader with the baker."""
    rows = {}
    for line in LENS.read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split(",")
        # `reached` (last column) is False for a method that never clears P* -- osm_footpaths is the
        # documented case. Its CSV row is that method's own terminal, not a Lens-B prefix, so there
        # is no crossing index to compare and including it would raise StopIteration below.
        if f[5].strip().lower() != "true":
            continue
        rows[f[0]] = (float(f[1]), float(f[3]))          # road_m, permeability at the Lens-B prefix
    assert rows, "no method reached P* -- the CSV or the target changed, not the bundle"
    for name, (road_m, perm) in rows.items():
        c = bundle["methods"][name]
        # The Lens-B prefix is the least m clearing matched_permeability; find it the same way.
        m = next(i for i, p in enumerate(c["permeability"]) if p >= bundle["matched_permeability"])
        assert c["road_m"][m] == pytest.approx(road_m, abs=0.05), name
        assert c["permeability"][m] == pytest.approx(perm, rel=1e-4), name


def test_curve_length_matches_the_run_log_segment_count(bundle: dict[str, Any]) -> None:
    """Closes a CI blind spot: an off-by-one loop bound in the baker (`range(len(ordered))`
    instead of `range(len(ordered) + 1)`) is invisible to every other fast test here -- it drops
    only the LAST prefix, so starts-at-zero, internal consistency, monotonicity and the
    lens-terminal comparison (whose crossing sits well before each method's final index) all stay
    green. Only the slow parity test catches it, and that test skips in CI. `run.log` closes the
    gap cheaply: it was written by scripts/compare_budgets (via gen_example.py), a different
    script at a different time, logging each method's segment count `N` independently of both the
    baker and this bundle -- so a bug shared between the baker and the bundle can't fool this check
    too. No `slow` marker, no skip: this must run in CI, which is the entire point."""
    log_text = RUN_LOG.read_text(encoding="utf-8")
    counts = dict(re.findall(r"reblocked (\w+): (\d+) segments", log_text))
    assert counts, "run.log format changed -- update the regex rather than let this pass on 0 rows"
    assert set(counts) == set(bundle["methods"]), (
        f"log/bundle method sets differ: {sorted(set(counts) ^ set(bundle['methods']))}")
    for name, n in counts.items():
        c = bundle["methods"][name]
        # N segments -> N+1 prefixes (index 0 is the no-roads prefix, matching test_starts_at_zero).
        assert len(c["road_m"]) == int(n) + 1, name


@pytest.mark.slow
def test_permeability_matches_the_solver_at_every_prefix(bundle: dict[str, Any]) -> None:
    """THE parity test. Developer-local by design: it needs ~/.cache/reblock's city data, and CI
    must stay hermetic (tests/conftest.py:19-20). Mirrors tests/data/test_osm_extract.py's
    convention."""
    blocks = Path.home() / ".cache" / "reblock" / "blocks_capetown_full.parquet"
    if not blocks.exists():
        pytest.skip("needs the capetown_full cache; run "
                     "`pixi run python -m scripts.gen_frontier_bundle`")

    from geopandas import GeoDataFrame

    from reblock.budget import street_first_ordered
    from reblock.compare import load_permeability_config
    from reblock.derive.access import STREET_TOL
    from reblock.derive.adjacency import parcel_adjacency
    from reblock.permeability import egress_power, permeability
    from scripts._example_block import load_example_block

    block, roads_by_method = load_example_block()
    params = load_permeability_config().params
    adj = parcel_adjacency(list(block.parcels.geometry), STREET_TOL)
    p0, _ = egress_power(block, None, params, adj=adj)
    for name, roads in roads_by_method.items():
        ordered = street_first_ordered(block, roads, STREET_TOL)
        got = [permeability(block, cast(GeoDataFrame, ordered.iloc[:m]), params, p0=p0, adj=adj)
               for m in range(len(ordered) + 1)]
        np.testing.assert_allclose(bundle["methods"][name]["permeability"], got, rtol=1e-5,
                                   atol=1e-9, err_msg=name)


def test_labels_and_colours_match_what_reblock_draws(bundle: dict[str, Any]) -> None:
    """The widget's legend must be the FALLBACK PNG's legend. `emit.compare_report` names each curve
    with `friendly_method_name` and colours it with `method_colors(method_order)`, where
    `run_permeability_lenses` passes `method_order=list(methods)` -- the selected set, in order --
    and the baker bakes both from those same two functions.

    "Parity by construction" is only true while something checks the construction, and this bundle
    is a COMMITTED artifact: it can be regenerated at a different time from the code that draws the
    PNG, or not regenerated at all after a label or a hue changes. That staleness is what this
    catches -- the baker sharing its inputs with emit.py cannot.
    """
    from reblock.emit import method_colors
    from reblock.method_labels import friendly_method_name

    try:
        from matplotlib.colors import to_hex
    except ModuleNotFoundError:                                     # pragma: no cover
        pytest.skip("needs matplotlib for the exact hex conversion emit.py's colours go through")

    names = list(bundle["methods"])
    expected = {n: to_hex(rgb) for n, rgb in method_colors(names).items()}
    for name, curve in bundle["methods"].items():
        assert curve["label"] == friendly_method_name(name), name
        assert curve["colour"] == expected[name], name
    # Distinct colours, or two curves are indistinguishable on the chart while every per-method
    # assertion above still passes.
    colours = {c["colour"] for c in bundle["methods"].values()}
    assert len(colours) == len(names), f"methods share a colour: {sorted(colours)}"


def test_chart_block_matches_the_figures_own_styling(bundle: dict[str, Any]) -> None:
    """Same staleness guard for the chart block: the widget's axis titles, curve width and guide
    styling are `reblock.emit`'s own module constants -- the values `compare_report` draws the PNG
    with -- so a change there that is not re-baked here would have JS-on and JS-off readers looking
    at two different charts (which is exactly what fix round 1 was opened for)."""
    from reblock.emit import (
        FRONTIER_GUIDE_COLOR,
        FRONTIER_GUIDE_LW,
        FRONTIER_LW,
        FRONTIER_X_LABEL,
        FRONTIER_Y_LABEL,
    )

    chart = bundle["chart"]
    assert chart["x_label"] == FRONTIER_X_LABEL
    assert chart["y_label"] == FRONTIER_Y_LABEL
    assert chart["line_width"] == pytest.approx(FRONTIER_LW)
    assert chart["guide_colour"] == FRONTIER_GUIDE_COLOR
    assert chart["guide_width"] == pytest.approx(FRONTIER_GUIDE_LW)

    # The web chart's own affordances, which the PNG has no equivalent for. `slider_step` must be a
    # WHOLE percentage point: every target the widget can be set to is printed as `{:.0%}` (emit's
    # own format for the same two numbers), and a finer step would print a percentage the guide is
    # not actually at. Both calibrated standards must sit on that grid for the same reason.
    step = chart["slider_step"]
    assert step == pytest.approx(0.01), "slider_step is no longer one percentage point"
    for target in ("matched_displacement", "matched_permeability"):
        assert bundle[target] / step == pytest.approx(round(bundle[target] / step)), target
    # niceTicks(0, max, tick_target) must put its LAST tick exactly on each axis end, or svg.ts's
    # plot rect (recovered from the tick extremes) is an inset of the data area rather than equal to
    # it. Reproduced here rather than trusted: this is the one chart value with a geometric
    # consequence, and it is chosen in the baker, not in reblock.
    for axis_max in (bundle["frontier_xmax"], chart["permeability_max"]):
        step_size = axis_max / chart["tick_target"]
        mag = 10.0 ** math.floor(math.log10(step_size))
        nice = next(k * mag for k in (1, 2, 2.5, 5, 10) if k * mag >= step_size)
        assert axis_max / nice == pytest.approx(round(axis_max / nice)), axis_max
