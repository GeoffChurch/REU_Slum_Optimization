"""Comparing a generated `.d.ts` against the bundle it declares -- for every bundle that has one.

Three committed browser bundles now ship with a generated `.d.ts` beside them, and each needs the
same bidirectional check: every key the artifact carries is declared, and every key the declaration
names is carried. The two helpers below are the awkward half of that, and they got their current
shape from a fix (piece C, I3) that a second hand-written copy would not inherit -- see
`ts_field_names`. Shared rather than duplicated per test module, and shared through a non-test
module rather than by importing across test files, which is the precedent tests/scoring_fixtures.py
set.
"""
from __future__ import annotations

import re
from typing import Any

_LINE_COMMENT = re.compile(r"//.*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)


def ts_field_names(dts: str) -> set[str]:
    """Every field name the .d.ts declares, at ANY nesting depth -- not just 2-space-indented
    top-level lines. A `.d.ts` here mixes two nesting styles (some interfaces' fields sit on their
    own 4-space-indented lines; others sit inline on one line, e.g. `nodes: { cx: number[]; cy:
    number[]; ground_g: number[] }`), so a line-anchored, fixed-indent regex only ever sees the
    outer style. Comments are stripped first: a bare `key:` regex would otherwise treat
    "// Regenerate: pixi run ..." as declaring a field named `Regenerate`."""
    stripped = _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", dts))
    return set(re.findall(r"(\w+)\s*\??:\s", stripped))


def json_keys(obj: Any) -> set[str]:
    """Every dict key appearing anywhere inside `obj`, through both dicts and lists, as bare names
    (not paths) -- e.g. `edges.footpath_g` and `roads[].width_m` both surface as `footpath_g` and
    `width_m`. Bare names are what a `.d.ts` interface declares too, so this is the comparable shape
    on both sides."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            keys |= json_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            keys |= json_keys(item)
    return keys
