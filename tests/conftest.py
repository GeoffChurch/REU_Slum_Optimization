"""Session-wide test isolation for the L2 derivation cache (see
src/reblock/cache.py). Without this, every test that builds a real block with
a non-empty source_content_hash (tests/test_run.py, tests/data/test_kblock_source.py,
tests/data/test_shapefile_source.py, ...) would flow through the module-level
cached wrappers bound at import time to the REAL ~/.cache/reblock/derivations,
making the suite read/write the user's real cache and non-hermetic.

tests/test_run.py also shells out to `python -m reblock.run` via subprocess: that
child process never sees our in-process monkeypatch of cache.memory, so we also
set REBLOCK_CACHE_DIR in the environment (inherited by the subprocess, which is
launched without an explicit `env=` override) so cache.py's module-level
_CACHE_DIR/memory bind to the same tmp dir when it's freshly imported there.

tests/test_cache.py's own per-function monkeypatches still work on top of this:
they further repoint cache.memory (and rebind the relevant wrapper) to their own
tmp_path, and pytest restores those afterward, un-doing back down to the state
this fixture establishes.
"""

from __future__ import annotations

from collections.abc import Iterator

import joblib
import pytest

import reblock.cache as cache


@pytest.fixture(scope="session", autouse=True)
def _isolate_derivation_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    mp = pytest.MonkeyPatch()
    tmp_dir = tmp_path_factory.mktemp("derivations")
    mp.setenv("REBLOCK_CACHE_DIR", str(tmp_dir))
    mp.setattr(cache, "memory", joblib.Memory(location=str(tmp_dir), verbose=0))
    mp.setattr(cache, "_access_impl_cached",
               cache.cached(cache._access_impl, ignore=["block", "roads"]))
    mp.setattr(cache, "_geometric_impl_cached",
               cache.cached(cache._geometric_impl, ignore=["block", "roads"]))
    mp.setattr(cache, "_voronoi_impl_cached",
               cache.cached(cache._voronoi_impl, ignore=["poly", "points", "crs"]))
    mp.setattr(cache, "_propose_impl_cached",
               cache.cached(cache._propose_impl, ignore=["method", "block"]))
    yield
    mp.undo()
