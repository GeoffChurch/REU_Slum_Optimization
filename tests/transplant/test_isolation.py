"""Nothing shipped reaches the research code, so editing it can never invalidate a derivation.

A derivation's cache key hashes its import closure (`derive_graph._closure_paths`). If any
`reblock` module outside `reblock.transplant` and `reblock.data.pools` imported them, a tweak to a
GW constant would silently invalidate cached proposals -- and a full examples regeneration costs
hours. Checked over EVERY other module, not just the derivation ones, because a closure is
transitive: any module a derivation might reach later has to be clean now.
"""
from __future__ import annotations

from pathlib import Path

import reblock
from reblock.derive_graph import _closure_paths

SRC = Path(reblock.__file__).resolve().parent
POOLS = SRC / "data" / "pools.py"
TRANSPLANT = SRC / "transplant"


def _research(path: Path) -> bool:
    return path == POOLS or TRANSPLANT in path.parents


def _module(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(("reblock", *(parts[:-1] if parts[-1] == "__init__" else parts)))


def test_no_shipped_module_imports_the_research_code() -> None:
    """FAULT INJECTION: adding `from reblock.transplant.gw import GWParams` to
    `reblock/methods/demand_greedy.py` fails this, naming that module."""
    shipped = [p for p in SRC.rglob("*.py") if not _research(p)]
    assert len(shipped) > 50, "the walk found almost nothing -- src/reblock moved"
    leaks = {_module(p): sorted(str(q.relative_to(SRC)) for q in _closure_paths(_module(p))
                                if _research(q))
             for p in shipped}
    assert {m: r for m, r in leaks.items() if r} == {}


def test_the_research_modules_are_where_this_test_looks() -> None:
    """Otherwise a move would leave the check above vacuously green."""
    assert POOLS.is_file()
    assert (TRANSPLANT / "gw.py").is_file()
    assert any(_research(q) for q in _closure_paths("reblock.transplant.snap"))
