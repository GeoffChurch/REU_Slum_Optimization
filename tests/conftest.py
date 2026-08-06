"""Session-wide test isolation for the derivation caches (see
src/reblock/derive_graph.py).

REBLOCK_CACHE_DIR is set HERE, at module load, BEFORE any `reblock.*` import.
pytest imports conftest.py before collecting test modules, so this is the
first place any `reblock` code can run -- setting the env var here (rather
than via a fixture, which only runs after import) means every module-level
`joblib.Memory(location=...)` in the codebase (derive_graph.py and any future
derivation module) binds to this session tmp dir the moment it's
imported, never touching the user's real ~/.cache/reblock/derivations.

This matters because `joblib.Memory.__init__` eagerly creates its directory
(and a .gitignore inside it) at construction time -- a fixture-based repoint
(monkeypatching `memory` after import) is too late to prevent that eager
directory creation against the real cache dir; only an import-time env
override prevents it.

We set it UNCONDITIONALLY (not `setdefault`): the test suite must always be
hermetic, even if a developer/CI has exported REBLOCK_CACHE_DIR at a persistent
scratch cache for normal use -- tests must never read or write that.

tests/test_run.py also shells out to `python -m reblock.run` via subprocess:
that child process inherits our environment (it's launched without an
explicit `env=` override), so it too binds to this tmp dir when it freshly
imports derive_graph.py.

Per-test monkeypatches (tests/test_derive_graph.py) still work on top of
this: they further repoint `memory` (and rebind the
relevant cached wrapper) onto their own tmp_path, and pytest undoes those
afterward, falling back down to the session tmp dir this module-load
env-set establishes.
"""

from __future__ import annotations

import os
import tempfile

os.environ["REBLOCK_CACHE_DIR"] = tempfile.mkdtemp(prefix="reblock-test-cache-")

from collections.abc import Iterator  # noqa: E402

import geopandas as gpd  # noqa: E402
import pytest  # noqa: E402
from pyproj import CRS  # noqa: E402
from shapely.geometry import LineString, Polygon  # noqa: E402

import reblock.derive_graph as _dg  # noqa: E402
from reblock.contracts import Block  # noqa: E402

UTM = CRS.from_epsg(32734)


@pytest.fixture(autouse=True)
def _clear_l1() -> Iterator[None]:
    _dg.clear_l1()
    yield
    _dg.clear_l1()


@pytest.fixture
def real_block() -> Block:
    """A 10x10 grid of unit parcels tiling a 10x10 square, south edge (y=0) is the street --
    same `_grid_block` pattern tests/test_permeability.py uses (k=10, cell=1.0), pulled into a
    shared fixture so tests/test_mesh.py doesn't have to import across test modules."""
    k, cell = 10, 1.0
    polys, ids = [], []
    for r in range(k):
        for c in range(k):
            x0, x1, y0, y1 = c * cell, (c + 1) * cell, r * cell, (r + 1) * cell
            polys.append(Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
            ids.append(r * k + c)
    parcels = gpd.GeoDataFrame({"parcel_id": ids}, geometry=polys, crs=UTM)
    streets = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (k * cell, 0)])], crs=UTM)
    boundary = Polygon([(0, 0), (k * cell, 0), (k * cell, k * cell), (0, k * cell)])
    return Block(block_id="g", crs=UTM, boundary=boundary, parcels=parcels, streets=streets)
