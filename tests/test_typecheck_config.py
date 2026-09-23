"""What `pixi run typecheck` actually checks.

There are two path lists -- `[tool.mypy] files` and the explicit file arguments in the
`typecheck-py` pixi task -- and only one of them is consulted, because explicit command-line
arguments OVERRIDE `files` entirely. The repo has kept them in sync by convention. This is the check
that convention never had.

Both lists name whole directories, so the other way a file drops out of the gate is `exclude`, which
the directory walk consults. That is checked too: it must skip exactly the files it names.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# The only files the gate skips. Both import the gitignored `scratchpad/ot/` spike, which mypy
# cannot resolve on a checkout without it, pending the owner's decision on vendoring that code.
# Spelled out rather than derived from the config: adding a file here is a decision to stop
# type-checking it.
EXCLUDED = {"scripts/pair_matrix.py", "scripts/consensus_matrix.py"}


def _config() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_the_two_mypy_lists_name_the_same_paths() -> None:
    """`typecheck-py` passes explicit path args, which OVERRIDE `[tool.mypy] files` -- so a module
    added to `files` alone is silently not type-checked by the gate, and a module dropped from the
    cmdline is silently not checked even though `files` still lists it.

    Compares EVERY path on both sides, not just the `.py` ones: `src`, `tests` and `scripts` are the
    entries whose accidental removal would hide the most, and a `.py`-only comparison would not
    notice any of them going missing.
    """
    cfg = _config()
    cmd = cfg["tool"]["pixi"]["tasks"]["typecheck-py"]
    args = cmd.split()
    assert args[0] == "mypy", f"typecheck-py no longer starts with mypy: {cmd!r}"
    # Every non-flag argument is a path. True while the only flag is `--strict`; a future flag that
    # takes a SEPARATE value word (`--config-file x.toml`) would need excluding here, and would
    # announce itself by failing this test rather than by quietly widening the set.
    cmdline = {a for a in args[1:] if not a.startswith("-")}
    listed = set(cfg["tool"]["mypy"]["files"])
    assert cmdline == listed, (
        f"only on the cmdline: {sorted(cmdline - listed)}; only in [tool.mypy] files: "
        f"{sorted(listed - cmdline)}. A path in just one list is not covered by the gate.")


def test_exclude_skips_exactly_the_named_files() -> None:
    """Walks every directory the gate names and applies `exclude` the way mypy's directory walk
    does: `re.search` against the path relative to the repo root, `/`-separated, with a trailing `/`
    on directories -- a matched directory takes its whole subtree with it. A pattern broader than
    intended (`^scripts/`, `matrix`) fails here instead of silently shrinking the gate, and a
    pattern that stops matching its file fails here instead of letting a file back in unannounced.
    """
    cfg = _config()
    patterns = [re.compile(p) for p in cfg["tool"]["mypy"]["exclude"]]

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
