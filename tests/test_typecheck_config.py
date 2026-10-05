"""What `make typecheck` actually checks.

`[tool.mypy] files` is the one list of what the gate type-checks, because the Makefile's
`typecheck-py` runs mypy with no file arguments. A file argument added there would OVERRIDE `files`
entirely and silently shrink the gate to whatever it names; the first test here is the check on
that.

Both `files` entries and the walk name whole directories, so the other way a file drops out of the
gate is `exclude`, which the directory walk consults. That is checked too: it must skip exactly the
files it names.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# The files the gate skips: none. Spelled out rather than derived from the config: adding a file
# here is a decision to stop type-checking it.
EXCLUDED: frozenset[str] = frozenset()


def _config() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_gate_runs_mypy_on_its_configured_files() -> None:
    """The `typecheck-py` recipe is `uv run mypy` with flags only. Every word after `mypy` that is
    not a flag would be a path, and paths on the command line override `[tool.mypy] files`."""
    lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    recipe = lines[lines.index("typecheck-py:") + 1]
    words = recipe.split()
    assert recipe.startswith("\t") and words[:3] == ["uv", "run", "mypy"], recipe
    assert [w for w in words[3:] if not w.startswith("-")] == [], (
        f"typecheck-py passes paths, which override [tool.mypy] files: {recipe.strip()!r}")
    assert _config()["tool"]["mypy"]["files"], "[tool.mypy] files is empty: the gate checks nothing"


def test_exclude_skips_exactly_the_named_files() -> None:
    """Walks every directory the gate names and applies `exclude` the way mypy's directory walk
    does: `re.search` against the path relative to the repo root, `/`-separated, with a trailing `/`
    on directories -- a matched directory takes its whole subtree with it. A pattern broader than
    intended (`^scripts/`, `matrix`) fails here instead of silently shrinking the gate, and a
    pattern that stops matching its file fails here instead of letting a file back in unannounced.
    """
    cfg = _config()
    # An absent `exclude` is mypy's own "skip nothing" -- the state this pins, not a fallback.
    patterns = [re.compile(p) for p in cfg["tool"]["mypy"].get("exclude", [])]

    def excluded(rel: str) -> bool:
        return any(p.search(rel) for p in patterns)

    skipped: set[str] = set()
    walked = 0
    for entry in cfg["tool"]["mypy"]["files"]:
        root = ROOT / entry
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            rel = path.relative_to(ROOT)
            walked += 1
            dirs = [f"{parent.as_posix()}/" for parent in rel.parents if parent != Path(".")]
            if excluded(rel.as_posix()) or any(excluded(d) for d in dirs):
                skipped.add(rel.as_posix())
    assert walked > len(EXCLUDED), "the walk found nothing -- the gate's directories moved"
    assert skipped == EXCLUDED, (
        f"skipped but not named: {sorted(skipped - EXCLUDED)}; named but not skipped: "
        f"{sorted(EXCLUDED - skipped)}")
