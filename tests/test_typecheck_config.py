"""The two lists that decide what `pixi run typecheck` actually checks.

There are two of them -- `[tool.mypy] files` and the explicit file arguments in the `typecheck-py`
pixi task -- and only one of them is consulted, because explicit command-line arguments OVERRIDE
`files` entirely. The repo has kept them in sync by convention. This is the check that convention
never had.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_two_mypy_lists_name_the_same_paths() -> None:
    """`typecheck-py` passes explicit path args, which OVERRIDE `[tool.mypy] files` -- so a module
    added to `files` alone is silently not type-checked by the gate, and a module dropped from the
    cmdline is silently not checked even though `files` still lists it.

    Compares EVERY path on both sides, not just the `.py` ones: `src` and `tests` are the two
    entries whose accidental removal would hide the most, and a `.py`-only comparison would not
    notice either of them going missing.
    """
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
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
