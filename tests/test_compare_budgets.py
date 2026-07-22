from pathlib import Path
from typing import cast

import geopandas as gpd
import pytest
from pyproj import CRS
from shapely.geometry import LineString, Polygon

from reblock.contracts import Block
from reblock.permeability import PermeabilityParams

UTM = CRS.from_epsg(32643)


def _street_block(x0: int, block_id: str) -> Block:
    # A 3x3 grid of unit parcels fronting a street on its bottom edge, offset to x0 so two of them
    # tile into a small 2-block region.
    polys = [Polygon([(x0 + i, j), (x0 + i + 1, j), (x0 + i + 1, j + 1), (x0 + i, j + 1)])
             for i in range(3) for j in range(3)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(9))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[LineString([(x0, 0), (x0 + 3, 0)])], crs=UTM)
    return Block(block_id=block_id, crs=UTM, boundary=boundary, parcels=parcels, streets=streets)


def test_load_permeability_config_reads_the_committed_yaml() -> None:
    from scripts.compare_budgets import load_permeability_config

    params, matched_displacement, matched_permeability = load_permeability_config()

    assert params.g_walk == 1.0 and params.g_road == 20.0 and params.g_street == 20.0
    assert params.corridor_m == 3.0
    assert 0.0 < matched_displacement < 1.0
    assert 0.0 < matched_permeability < 1.0


def test_run_permeability_lenses_writes_tables_and_renders(tmp_path: Path) -> None:
    # End-to-end glue smoke test on a tiny 2-block region with a real reblocker (DijkstraReblocker
    # paves everything, so it reaches a shallow depth / high permeability). Asserts the frontier +
    # the two lens tables + a before/after render per lens per coloring + the per-method GIF are
    # all written from the SAME reblock (no second propose), and that no retired two-lens/
    # external-internal artifact reappears.
    from reblock.methods.dijkstra import DijkstraReblocker
    from scripts.compare_budgets import run_permeability_lenses

    region = [_street_block(0, "a"), _street_block(4, "b")]
    rows = run_permeability_lenses(
        region, {"dijkstra": DijkstraReblocker()}, tmp_path,
        matched_displacement=0.3, matched_permeability=0.1, params=PermeabilityParams())

    assert len(rows) == 1
    (row,) = rows
    assert row.method == "dijkstra"
    assert row.disp_road_m >= 0.0 and row.perm_road_m >= 0.0
    assert 0.0 <= row.disp_permeability
    assert 0.0 <= row.perm_displacement <= 1.0
    assert isinstance(row.reached, bool)

    # the frontier (permeability + displacement curves), from the SAME reblock as the lenses below
    assert (tmp_path / "frontier_permeability.csv").exists()
    assert list(tmp_path.glob("frontier_*.png"))

    # the two lens outcome tables
    assert (tmp_path / "lens_displacement.csv").exists()
    assert (tmp_path / "lens_permeability.csv").exists()
    disp_text = (tmp_path / "lens_displacement.csv").read_text()
    assert "dijkstra" in disp_text
    perm_text = (tmp_path / "lens_permeability.csv").read_text()
    assert "dijkstra" in perm_text and "reached" in perm_text

    # before, both colorings, once per region
    assert (tmp_path / "before_depth.jpg").exists()
    assert (tmp_path / "before_perm.jpg").exists()

    # after, per method per lens, both colorings
    for tag in ("disp", "perm"):
        assert (tmp_path / f"after_dijkstra_{tag}_depth.jpg").exists()
        assert (tmp_path / f"after_dijkstra_{tag}_perm.jpg").exists()

    # per-method reblock GIF (unchanged)
    assert (tmp_path / "reblock_dijkstra.gif").exists()

    # retired two-lens / external-internal artifacts must not reappear
    assert not (tmp_path / "lens_a_external.csv").exists()
    assert not (tmp_path / "lens_b_matched.csv").exists()
    assert not (tmp_path / "frontier_external_connectivity.csv").exists()
    assert not (tmp_path / "frontier_internal_connectivity.csv").exists()
    assert not (tmp_path / "displacement_vs_length.csv").exists()
    assert not list(tmp_path.glob("curve_*.png"))
    assert not list(tmp_path.glob("displacement_*.png"))
    assert not list(tmp_path.glob("depth_vs_road_*.png"))


def test_run_permeability_lenses_reblocks_once_per_method_not_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # The frontier + both lenses must all reuse the SAME reblock -- region_reblock is called
    # exactly ONCE per method, never a second time (which would silently double wall-clock on a
    # real thousands-of-buildings region).
    import scripts.compare_budgets as cb
    from reblock.methods.dijkstra import DijkstraReblocker
    from reblock.region import region_reblock as real  # same object cb re-imported

    calls = {"n": 0}

    def counting(*a: object, **k: object) -> object:
        calls["n"] += 1
        return real(*a, **k)   # type: ignore[arg-type]

    monkeypatch.setattr(cb, "region_reblock", counting)
    region = [_street_block(0, "a"), _street_block(4, "b")]
    cb.run_permeability_lenses(region, {"dijkstra": DijkstraReblocker()}, tmp_path,
                               matched_displacement=0.3, matched_permeability=0.1,
                               params=PermeabilityParams())
    assert calls["n"] == 1        # one method -> one reblock; frontier + both lenses reuse it


def test_run_permeability_lenses_singleton_region_skips_region_reblock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    # A singleton region takes the exact pre-region single-block `propose()` path (mirrors
    # `reblock.compare.compare`'s own singleton branch), NOT `region_reblock`: `region_reblock`/
    # `region_block` unions a single block's `streets` rows into ONE (Multi)LineString row, which a
    # method that filters `streets.geometry` by `isinstance(..., LineString)` (e.g. TopologyMethod,
    # used single-block-only by gen_method_comparison.py) would then see as empty street geometry.
    import scripts.compare_budgets as cb
    from reblock.methods.dijkstra import DijkstraReblocker

    def _boom(*a: object, **k: object) -> object:
        raise AssertionError("region_reblock must not be called for a singleton region")

    monkeypatch.setattr(cb, "region_reblock", _boom)
    region = [_street_block(0, "a")]
    cb.run_permeability_lenses(region, {"dijkstra": DijkstraReblocker()}, tmp_path,
                               matched_displacement=0.3, matched_permeability=0.1,
                               params=PermeabilityParams())
