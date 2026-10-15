"""A code's identity as a digest of its constants: every public module-level constant (an
UPPER_CASE name) of the modules named, by `<module>.<NAME>` and its repr, sha256 of their JSON, 8
hex digits. The complex's code (gap_clear.CODE: gap_clear, gap_complex, gap_flow, gap_factor and
clear.EPS) and the exits' (gap_exits.EXITS_CODE: the region model's leaks, barriers and boost)
each have one, so editing a constant renames only the metrics whose code it is (NOTES, "Exits as
leaks: the region model on the complex (5810@major)").

Typed as `search.py`'s docstring says:

    MYPYPATH=research/roadless uv run mypy --strict --follow-imports=silent \\
        research/roadless/digests.py
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping
from types import ModuleType

import numpy as np


def constants_of(modules: tuple[ModuleType, ...], *, exempt: Mapping[str, str],
                 external: Mapping[str, Callable[[], object]]) -> dict[str, str]:
    """Every constant of `modules` by `<module>.<NAME>`, as its repr: a number, a string, a tuple,
    an array or a frozen dataclass; any other kind refused (a new constant's kind decided where
    its code's identity is, not passed over). `exempt`: the names that cannot change a result,
    each with its reason; `external`: constants of other modules the code reads, by key."""
    out = {}
    for mod in modules:
        short = mod.__name__.rsplit(".", 1)[-1]
        for name, value in vars(mod).items():
            key = f"{short}.{name}"
            if name.startswith("_") or not name.isupper() or key in exempt:
                continue
            if not (isinstance(value, (bool, int, float, str, tuple, np.ndarray))
                    or (dataclasses.is_dataclass(value) and not isinstance(value, type))):
                raise TypeError(f"{key}: a constant of kind {type(value).__name__}: in its "
                                "code's digest or exempt, decided there")
            out[key] = repr(value)
    for key, read in external.items():
        out[key] = repr(read())
    return out


def digest_of(constants: Mapping[str, str]) -> str:
    """The constants digested: 8 hex digits."""
    blob = json.dumps(dict(constants), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:8]
