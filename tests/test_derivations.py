from collections.abc import Iterator
from pathlib import Path
from typing import cast

import geopandas as gpd
import joblib
import pandas as pd
import pytest
from pyproj import CRS
from shapely.geometry import Polygon

import reblock.derivations as D
import reblock.derive_graph as dg
from reblock.contracts import Block, Proposal
from reblock.derive.access import parcel_access_layers

UTM = CRS.from_epsg(32643)


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    # The session-wide REBLOCK_CACHE_DIR (tests/conftest.py) keeps derive_graph's
    # L2 disk cache alive for the whole pytest session; the autouse `_clear_l1`
    # fixture there only drops L1. Without a per-test L2 repoint too, a block
    # identity reused across tests in this file (all `_grid_block("deadbeef")`,
    # same `block.identity`) would L2-hit on a key an earlier test already
    # populated, undercounting this test's own spy calls. Mirrors
    # tests/test_derive_graph.py's `_isolate`.
    monkeypatch.setattr(dg, "memory", joblib.Memory(location=str(tmp_path), verbose=0))
    monkeypatch.setattr(dg, "_l2", dg.memory.cache(dg._l2_impl, ignore=["fn", "inputs"]))
    dg.clear_l1()
    yield
    dg.clear_l1()


def _grid_block(hash_: str) -> Block:
    polys = [Polygon([(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)])
             for i in range(3) for j in range(3)]
    parcels = gpd.GeoDataFrame({"parcel_id": list(range(9))}, geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels,
                 streets=streets, source_content_hash=hash_)


def test_access_before_matches_direct_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    box = {"n": 0}
    real = parcel_access_layers

    def spy(block: Block, roads: gpd.GeoDataFrame | None = None, **kw: object) -> pd.Series:
        box["n"] += 1
        return real(block, roads)
    monkeypatch.setattr(D, "parcel_access_layers", spy)

    block = _grid_block("deadbeef")
    out1 = D.access_before(block)
    out2 = D.access_before(block)                 # cache hit
    assert box["n"] == 1
    assert out1.equals(out2)
    assert out1.equals(real(block, None))         # value identical to direct call


def test_before_and_after_use_distinct_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    box = {"n": 0}
    real = parcel_access_layers

    def spy(block: Block, roads: gpd.GeoDataFrame | None = None, **kw: object) -> pd.Series:
        box["n"] += 1
        return real(block, roads)
    monkeypatch.setattr(D, "parcel_access_layers", spy)

    block = _grid_block("deadbeef")
    prop = Proposal(block_id="g", crs=UTM, block_identity=block.identity, proposal_id="peel")
    D.access_before(block)
    D.access_after(block, prop)                   # distinct fn.identity -> distinct key
    assert box["n"] == 2


def test_bypass_when_hash_empty() -> None:
    block = _grid_block("")                        # identity None -> uncacheable
    out = D.access_before(block)
    assert isinstance(out, pd.Series)             # computes directly, no cache touch
