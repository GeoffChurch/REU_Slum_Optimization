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


def test_propose_matches_direct_and_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    from reblock.methods.peel import PeelReblocker

    box = {"n": 0}
    method = PeelReblocker()
    real_propose = method.propose

    def spy(block: Block, prior: Proposal | None = None) -> Proposal:
        box["n"] += 1
        return real_propose(block, prior)
    monkeypatch.setattr(method, "propose", spy)

    block = _grid_block("deadbeef")
    direct = real_propose(block)
    out1 = D.propose(method, block)
    out2 = D.propose(method, block)               # cache hit -> no second spy call
    assert box["n"] == 1
    assert out1.roads is not None and out2.roads is not None and direct.roads is not None
    assert (sorted(g.wkt for g in out1.roads.geometry)
            == sorted(g.wkt for g in direct.roads.geometry))


def _member_block(block_id: str, hash_: str, x0: int) -> Block:
    # A cacheable 3x3 grid member at x-offset x0 with a non-empty source hash, so the
    # region_block built from two of these has a non-empty (deterministic) identity.
    polys = [Polygon([(x0 + i, j), (x0 + i + 1, j), (x0 + i + 1, j + 1), (x0 + i, j + 1)])
             for i in range(3) for j in range(3)]
    parcels = gpd.GeoDataFrame({"parcel_id": [f"{block_id}-{k}" for k in range(9)]},
                               geometry=polys, crs=UTM)
    boundary = cast(Polygon, parcels.geometry.union_all())
    streets = gpd.GeoDataFrame(geometry=[boundary.boundary], crs=UTM)
    return Block(block_id=block_id, crs=UTM, boundary=boundary, parcels=parcels,
                 streets=streets, source_content_hash=hash_)


def test_region_reblock_routes_through_the_propose_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    # region_reblock must reblock via the memoized derivations.propose so a region's (expensive)
    # proposal is computed once and shared by both compare and render, and re-runs hit the L2 disk
    # cache. The region Block's source_content_hash is deterministic in its members, so its
    # identity is stable across the two region_block() rebuilds inside the two region_reblock()
    # calls -> the second call is a cache hit and the method's propose runs exactly once.
    from reblock.methods.peel import PeelReblocker
    from reblock.region import region_reblock

    box = {"n": 0}
    method = PeelReblocker()
    real_propose = method.propose

    def spy(block: Block, prior: Proposal | None = None) -> Proposal:
        box["n"] += 1
        return real_propose(block, prior)
    monkeypatch.setattr(method, "propose", spy)

    members = [_member_block("a", "cafe01", x0=0), _member_block("b", "cafe02", x0=3)]
    r1 = region_reblock(members, method, [])
    r2 = region_reblock(members, method, [])      # cache hit -> spy not called a second time
    assert box["n"] == 1
    assert r1.proposal.roads is not None and r2.proposal.roads is not None
    assert (sorted(g.wkt for g in r1.proposal.roads.geometry)
            == sorted(g.wkt for g in r2.proposal.roads.geometry))
