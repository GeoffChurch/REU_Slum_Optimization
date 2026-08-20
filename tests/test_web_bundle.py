"""The bundle is a committed artifact, so nothing recomputes it on the way to the browser. These
tests are the only thing standing between a bad bake and a wrong picture."""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.dts_keys import json_keys, ts_field_names

BUNDLE = Path("examples/perm-graph/bundle.json")
DTS = Path("web/src/bundle.d.ts")


@pytest.fixture(scope="module")
def bundle() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(BUNDLE.read_text(encoding="utf-8"))
    return result


def test_shapes_are_internally_consistent(bundle: dict[str, Any]) -> None:
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


def test_first_upgraded_at_is_monotone_and_in_range(bundle: dict[str, Any]) -> None:
    """-1 means never raised. Any other value must be a real prefix index -- an off-by-one here
    would silently paint the wrong edges blue at every slider position."""
    n = bundle["n_prefixes"]
    fu = np.asarray(bundle["edges"]["first_upgraded_at"])
    assert ((fu == -1) | ((fu >= 0) & (fu < n))).all()
    assert (fu != 0).all(), "no edge can be road-raised at prefix 0: there are no roads"


def test_coordinates_are_local_metres_at_centimetre_precision(bundle: dict[str, Any]) -> None:
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


def test_permeability_is_zero_at_prefix_zero_and_monotone(bundle: dict[str, Any]) -> None:
    perm = bundle["prefix"]["permeability"]
    assert perm[0] == 0.0
    assert all(b >= a - 1e-9
               for a, b in zip(perm, perm[1:], strict=False)), "permeability must not fall"


def test_dts_declares_exactly_the_bundle_keys(bundle: dict[str, Any]) -> None:
    """Catches 'regenerated one file, not the other'. Structural and fast -- no solving.

    Fix wave, I3: the original regex (`^\\s{2}(\\w+)[?]?:`) matched only 2-space-indented lines, so
    it checked the top-level `Bundle`/`Encoding` keys but NOT `nodes.{cx,cy,ground_g}`,
    `edges.{rows,cols,footpath_g,first_upgraded_at}`, `prefix.{potential,current,permeability,
    road_m}`, `width_norm.*`, `roads[].*` -- every field the canvas actually reads each frame was in
    the unguarded half. Walking the parsed JSON recursively (`tests.dts_keys.json_keys`) closes
    that. Also made BIDIRECTIONAL: `.d.ts` declaring a key the bundle no longer has is a regression
    too (a renamed Python field otherwise leaves a dead, misleading declaration behind), and this
    direction was free -- `declared - bundle_keys` was already empty before this fix."""
    declared = ts_field_names(DTS.read_text(encoding="utf-8"))
    present = json_keys(bundle)
    missing = present - declared
    assert not missing, f"bundle keys missing from bundle.d.ts: {sorted(missing)}"
    extra = declared - present
    assert not extra, f"bundle.d.ts declares keys the bundle does not have: {sorted(extra)}"


def test_encoding_matches_reblock_render_live_constants(bundle: dict[str, Any]) -> None:
    """I7: nothing previously guarded `encoding` against `reblock.render`'s live constants -- change
    `_EDGE_LW_MAX` there (or any of the other width/colour constants, or the ramp's source colormap)
    and every PNG moves while the committed bundle silently keeps whatever number was baked in,
    forever. Fast: no block loading, no solving, no `slow` marker -- imports `reblock.render`
    directly and reuses the baker's own `_ramp` sampling (not a re-implementation of it, which
    would just be a second place for the two to drift) to reproduce the ramp from the live
    colormap name."""
    from reblock.render import (
        _BOUNDARY_COLOR,
        _CONTEXT_OUTLINE,
        _EDGE_GREY,
        _EDGE_LW_MAX,
        _EDGE_LW_MIN,
        _NODE_RADIUS_FRAC,
        _PERM_CMAP,
        _ROAD_COLOR,
        _UPGRADED_LW,
    )
    from scripts.gen_web_bundle import _ramp

    e = bundle["encoding"]
    assert e["edge_lw_min"] == _EDGE_LW_MIN
    assert e["edge_lw_max"] == _EDGE_LW_MAX
    assert e["upgraded_lw"] == _UPGRADED_LW
    assert e["node_radius_frac"] == _NODE_RADIUS_FRAC
    assert e["road_color"] == _ROAD_COLOR
    assert e["boundary_color"] == _BOUNDARY_COLOR
    assert e["parcel_color"] == _CONTEXT_OUTLINE
    assert e["edge_color"] == _EDGE_GREY
    assert e["ramp"] == _ramp(_PERM_CMAP)


def test_bundle_matches_perm_graph_json_at_the_caption_precision(bundle: dict[str, Any]) -> None:
    """I7: `gen_web_bundle.py` and `gen_perm_graph.py` each define their own VARIANT/METHOD pin and
    each load the block independently -- nothing compared the two artifacts, so re-pinning one but
    not the other would make the widget describe a different block than the caption underneath it,
    silently. (The slow parity test above cannot catch this: it shares `load_block_and_roads` with
    the baker, so a bad pin there would just make both sides of that comparison agree.) Compared at
    the caption's own precision (`gen_site_pages.py`'s `_perm_graph_figures`: whole metres, one
    decimal of percent), not exact float equality, which is stricter than what a reader can see."""
    meta = json.loads(
        Path("examples/perm-graph/perm_graph.json").read_text(encoding="utf-8"))
    i = bundle["lens_b_index"]
    assert bundle["block_id"] == meta["block_id"]
    assert round(bundle["prefix"]["road_m"][i]) == round(meta["road_m"])
    assert round(bundle["prefix"]["permeability"][i] * 100, 1) == round(
        meta["permeability_after"] * 100, 1)


@pytest.mark.slow
def test_bundle_matches_permeability_graph_at_every_prefix(bundle: dict[str, Any]) -> None:
    """THE parity test: the committed bundle must equal what the Python twin produces, at the 6
    significant digits the baker emits. This is what B being built first bought us.

    DEVELOPER-LOCAL BY DESIGN. This is `slow` (needs a warm derivation cache -- `ensure_city_data`
    reaches Dataverse for the Cape Town blocks parquet and Open Buildings for the matching tile on
    a cold cache, at a 900 s timeout) and `pixi run test` -- which `.github/workflows/ci.yml` runs
    on every PR -- does not deselect `slow` (see the marker's own registration comment in
    pyproject.toml: deselecting it by default would stop anyone running the guard without
    remembering an override, so it stays opt-in via a cache check instead of opt-out via addopts).
    So instead of running on a cold checkout and downloading ~1 GB, this test SKIPS when the local
    artifact it needs is absent, mirroring the established convention at
    tests/data/test_osm_extract.py:303-307. A contributor with a warm `~/.cache/reblock` (anyone
    who has run the baker or an example generator) gets the real guard for free; CI and a fresh
    clone get neither the guard nor the download."""
    blocks = Path.home() / ".cache" / "reblock" / "blocks_capetown_full.parquet"
    if not blocks.exists():
        pytest.skip("needs the capetown_full cache; run "
                    "`pixi run python -m scripts.gen_web_bundle`")

    from typing import cast

    from geopandas import GeoDataFrame

    from reblock.budget import street_first_ordered
    from reblock.compare import load_permeability_config
    from reblock.derive.access import STREET_TOL
    from reblock.perm_graph import permeability_graph
    from scripts.gen_web_bundle import load_block_and_roads  # Task 1 exposes this

    block, roads = load_block_and_roads()
    params = load_permeability_config().params
    ordered = street_first_ordered(block, roads, STREET_TOL)

    for m in range(bundle["n_prefixes"]):
        fig = permeability_graph(block, cast(GeoDataFrame, ordered.iloc[:m]), params)
        np.testing.assert_allclose(bundle["prefix"]["potential"][m], fig.potential, rtol=1e-5)
        np.testing.assert_allclose(bundle["prefix"]["current"][m], fig.current, rtol=1e-5,
                                   atol=1e-9)
