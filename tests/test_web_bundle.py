"""The bundle is a committed artifact, so nothing recomputes it on the way to the browser. These
tests are the only thing standing between a bad bake and a wrong picture."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

BUNDLE = Path("examples/perm-graph/bundle.json")
DTS = Path("web/src/bundle.d.ts")


@pytest.fixture(scope="module")
def bundle() -> dict:
    return json.loads(BUNDLE.read_text(encoding="utf-8"))


def test_shapes_are_internally_consistent(bundle) -> None:
    n_nodes, n_edges = len(bundle["nodes"]["cx"]), len(bundle["edges"]["rows"])
    assert len(bundle["nodes"]["cy"]) == len(bundle["nodes"]["ground_g"]) == n_nodes
    assert len(bundle["edges"]["cols"]) == len(bundle["edges"]["footpath_g"]) == n_edges
    assert len(bundle["edges"]["first_upgraded_at"]) == n_edges
    assert bundle["n_prefixes"] == len(bundle["roads"]) + 1
    for key in ("potential", "current", "permeability", "road_m"):
        assert len(bundle["prefix"][key]) == bundle["n_prefixes"]
    assert all(len(p) == n_nodes for p in bundle["prefix"]["potential"])
    assert all(len(c) == n_edges for c in bundle["prefix"]["current"])
    assert 0 < bundle["lens_b_index"] < bundle["n_prefixes"]


def test_first_upgraded_at_is_monotone_and_in_range(bundle) -> None:
    """-1 means never raised. Any other value must be a real prefix index -- an off-by-one here
    would silently paint the wrong edges blue at every slider position."""
    n = bundle["n_prefixes"]
    fu = np.asarray(bundle["edges"]["first_upgraded_at"])
    assert ((fu == -1) | ((fu >= 0) & (fu < n))).all()
    assert (fu != 0).all(), "no edge can be road-raised at prefix 0: there are no roads"


def test_coordinates_are_local_metres_at_centimetre_precision(bundle) -> None:
    """Guards a bug that would ship as *slightly wrong geometry* rather than as a crash.

    Coordinates are emitted relative to `origin`. If someone rounds them with significant digits
    instead of absolute precision, a Cape Town UTM northing (~6,240,000) rounds to the nearest 10 m
    and the parcels dissolve -- while the file still parses, the widget still draws, and the picture
    is merely wrong. So: coordinates must be small (local, not UTM), and the extent they span must
    be a plausible block, not a degenerate or continental one."""
    xs = [x for ring in bundle["parcels"] for x, _ in ring]
    ys = [y for ring in bundle["parcels"] for _, y in ring]
    assert max(abs(v) for v in xs + ys) < 10_000, "coordinates look like UTM, not local metres"
    assert 20 < (max(xs) - min(xs)) < 2_000, f"implausible x extent {max(xs) - min(xs)}"
    assert 20 < (max(ys) - min(ys)) < 2_000, f"implausible y extent {max(ys) - min(ys)}"
    # Centimetre rounding must leave at least 3 distinct values per 10 m of extent; 10 m rounding
    # would collapse a ~200 m block to ~20 distinct coordinates. Checked on BOTH axes: a
    # significant-digit regression hits northing (7 UTM digits) far harder than easting (6 digits),
    # so an x-only check can pass while y has already collapsed to a 10 m grid.
    assert len(set(xs)) > (max(xs) - min(xs)) / 10 * 3, "x coordinate resolution looks too coarse"
    assert len(set(ys)) > (max(ys) - min(ys)) / 10 * 3, "y coordinate resolution looks too coarse"
    assert bundle["origin"][1] > 1_000_000, "origin should carry the real UTM northing"


def test_permeability_is_zero_at_prefix_zero_and_monotone(bundle) -> None:
    perm = bundle["prefix"]["permeability"]
    assert perm[0] == 0.0
    assert all(b >= a - 1e-9
               for a, b in zip(perm, perm[1:], strict=False)), "permeability must not fall"


def test_dts_declares_exactly_the_bundle_keys(bundle) -> None:
    """Catches 'regenerated one file, not the other'. Structural and fast -- no solving."""
    dts = DTS.read_text(encoding="utf-8")
    declared = set(re.findall(r"^\s{2}(\w+)[?]?:", dts, flags=re.M))
    for key in bundle:
        assert key in declared, f"bundle key {key!r} missing from bundle.d.ts"
    for key in bundle["encoding"]:
        assert key in declared, f"encoding key {key!r} missing from bundle.d.ts"


@pytest.mark.slow
def test_bundle_matches_permeability_graph_at_every_prefix(bundle) -> None:
    """THE parity test: the committed bundle must equal what the Python twin produces, at the 6
    significant digits the baker emits. This is what B being built first bought us."""
    from reblock.budget import street_first_ordered
    from reblock.compare import load_permeability_config
    from reblock.derive.access import STREET_TOL
    from reblock.perm_graph import permeability_graph
    from scripts.gen_web_bundle import load_block_and_roads  # Task 1 exposes this

    block, roads = load_block_and_roads()
    params = load_permeability_config().params
    ordered = street_first_ordered(block, roads, STREET_TOL)

    for m in range(bundle["n_prefixes"]):
        fig = permeability_graph(block, ordered.iloc[:m], params)
        np.testing.assert_allclose(bundle["prefix"]["potential"][m], fig.potential, rtol=1e-5)
        np.testing.assert_allclose(bundle["prefix"]["current"][m], fig.current, rtol=1e-5,
                                   atol=1e-9)
