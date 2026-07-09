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

import pytest  # noqa: E402

import reblock.derive_graph as _dg  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_l1() -> Iterator[None]:
    _dg.clear_l1()
    yield
    _dg.clear_l1()
