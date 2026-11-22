"""Rev 4's migration (ears.REV, owner 2026-11-22): b now charges for the building share of the
road's cross-section (ears.road_share), not for the centreline's pixel being a building's; binf
is the limit b -> inf (lexicographic), not buildings impassable; a loop ends on its source's
ring (EarWorld.arcs: a path to a hole's sinks splits nothing, and rev 3 offered such paths);
and a loop counts only if it cuts its face into two pieces of growth.FACE_M2 and more less its
corridor's share (ears.cuts: rev 4 ranks only ears build can build).

A b0 plan's paths cost 1 a metre either way, and a divides plan reads no b, ring or cut: their
rev 3 rows are rev 4's, so `.v3` becomes `.v4` in their names wherever they appear under
research/roadless (the names of directories and files, symlink targets, world.json's `plan`, the
text of tracked files; tracked files renamed with `git mv`), as atlas/routers/migrate.py did for
the router token. Except a b0 paths row that rev 4 does not replay (`replayed`): one that offered,
at some state, a candidate rev 4 does not rank (its ends on two rings, or a piece it cuts too
small), so that rev 4 offers another there. (A source's best under rev 4 is its best under rev
3 when that one is ranked, and is no better otherwise, so the best `pairs` sources' ears are rev
3's exactly when every one rev 3 offered is ranked.)

A stale rev 3 row (a b > 0 paths plan's: b0.05, b20, binf, ...; or that b0 one) under a `rows`
directory (atlas/crossings/rows/<world>/, atlas/lookahead/rows/[check/]<block>/) moves, not
deleted, to the same place under the sibling `rows_old_bcost` (gitignored by `rows*/`), and a
symlink to one is re-pointed there. Elsewhere (growth_rows, roads5810's plates, atlas/blocks' grid
json, and the tracked text naming them: the Demo's and roads5810's pages and builders) a b > 0
rev 3 name stays as it is: those are rows of the published pages, read under their own names,
and regenerating them is the owner's call.

A rows archive (`rows_*`: rows_old_bcost, rows_steiner_deleted) is left as it is.
Idempotent: a migrated name matches the pattern no more. A destination that exists stops it.

    M=research/roadless/atlas/routers/migrate_bcost.py   # from the repo root
    uv run python $M     # dry run: what it would do
    uv run python $M --apply
"""
from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # research/roadless
sys.path.insert(0, str(ROOT))
import ears  # noqa: E402
import growth as gr  # noqa: E402

REPO = ROOT.parents[1]
PLAN = re.compile(r"ears\.(?:(?P<paths>paths)\.[a-z0-9]+\.b(?P<b>[0-9.]+|inf)\.f|divides\.)"
                  r"[A-Za-z0-9.+-]*?\.D[0-9.]+\.v3(?![0-9])")
OLD = "rows"
NEW = "rows_old_bcost"
SKIP = {".venv", "__pycache__", ".git"}


def skipped(d: str) -> bool:
    """A directory not walked: tooling's, and a rows archive (rows_old_bcost, rows_steiner_deleted:
    rows of plans no longer run, kept as they were)."""
    return d in SKIP or d.startswith(OLD + "_")


class Replay(NamedTuple):
    """A rev 3 row's candidates under rev 4's rules (`replay`): how many it offered, how many end
    on two rings, how many cut a piece too small, and the first state with either."""
    offered: int
    cross: int
    small: int
    first: int


@functools.cache
def replay(row: Path) -> Replay:
    """The rev 3 b0 paths row `row` (a crossings row: its states' pixel faces, states.npz, and
    every candidate offered, cands.parquet; symlinks followed) under rev 4's rules: at each state
    the faces' Arcs (ears.arcs_of), each candidate's ends' rings and the pieces it cuts
    (ears.cuts, its line's swept area: as Grid8's, Theta's and Visibility's searches take it;
    Pulled's line is the string-pulled one and Eikonal's the descent, whose swept areas rev 4
    takes from the grid path and the characteristic)."""
    z = np.load(row / "states.npz")
    c = pd.read_parquet(row / "cands.parquet")
    shape = (int(z["shape"][0]), int(z["shape"][1]))
    lo = np.asarray(z["lo"], dtype=np.float64)
    cross = small = 0
    first = -1
    for step, grp in c.groupby("step", sort=True):
        pf = z["pf"][int(step)]
        ar = ears.arcs_of(pf.reshape(shape), z["sink"][int(step)].astype(bool))
        for x, y in zip(grp.path_x, grp.path_y, strict=True):
            line = np.column_stack([x, y]).astype(np.float64)
            a, b = ears._pix_of_lo(lo, shape, line[[0, -1]])
            two = bool(ar.ring[a] != ar.ring[b])
            swept = float(ears._tris(line[:-1], line[1:], lo).sum())
            L = float(np.hypot(*np.diff(line, axis=0).T).sum())
            P, Q = ears.cuts(ar, int(pf[a]), int(a), np.array([b]), np.array([swept]))
            tiny = not bool(min(P[0], Q[0]) - ears.CUT_ALLOW * gr.PATCH_R * L >= gr.FACE_M2)
            cross += two
            small += tiny and not two
            if (two or tiny) and first < 0:
                first = int(step)
    return Replay(offered=len(c), cross=cross, small=small, first=first)


def replayed(row: Path) -> bool:
    """Whether rev 4 offers what the rev 3 b0 paths row `row` offered at every state (`replay`):
    none of its candidates on two rings or cutting a piece too small."""
    return replay(row.resolve()).first < 0


def stale(p: Path) -> bool:
    """Whether the entry p (a row, a file, a symlink's target) names a rev 3 plan that rev 4 runs
    differently: b > 0, or b0 not `replayed` (a row of a `rows` directory; a b0 name elsewhere, a
    log's, is no row)."""
    for m in PLAN.finditer(p.name):
        if m["paths"] and float(m["b"]) > 0.0:
            return True
        if m["paths"] and OLD in p.parts[:-1] and not replayed(p):
            return True
    return False


def renamed(s: str) -> str:
    """s with every rev 3 plan name made rev 4's but the b > 0 ones (a caller asks `stale` for
    the holed b0 ones first)."""
    return PLAN.sub(lambda m: m[0][:-1] + "4" if not m["paths"] or float(m["b"]) == 0.0
                    else m[0], s)


def moved(path: str) -> str:
    """`path` (a symlink's target) re-pointed: a stale entry under a `rows` directory to the same
    place under `rows_old_bcost`, else its rev 3 names made rev 4's."""
    p = Path(path.rstrip("/"))
    parts = p.parts
    if stale(p) and OLD in parts[:-1]:
        k = len(parts) - 1 - parts[::-1].index(OLD)
        out = str(Path(*parts[:k], NEW, *parts[k + 1:]))
    elif stale(p):
        out = str(p)
    else:
        out = renamed(str(p))
    return out + ("/" if path.endswith("/") else "")


def tracked() -> set[Path]:
    out = subprocess.run(["git", "ls-files", "-z", str(ROOT)], cwd=REPO, check=True,
                         capture_output=True).stdout.decode().split("\0")
    return {(REPO / p).resolve() for p in out if p}


def walk() -> list[Path]:
    """Every entry under ROOT, not following symlinks, not descending into a stale entry under a
    rows directory (it moves whole) nor into a skipped one."""
    out = []
    for d, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [x for x in dirs if not skipped(x)]
        out += [Path(d) / x for x in dirs + files]
        dirs[:] = [x for x in dirs if not (OLD in Path(d).relative_to(ROOT).parts
                                           and stale(Path(d) / x))]
    return out


def main(apply: bool) -> None:
    tr = tracked()
    paths = walk()
    # 1: symlinks' targets (before anything moves under them)
    for p in paths:
        if p.is_symlink():
            t = os.readlink(p)
            t2 = moved(t)
            if t2 != t:
                print(f"relink {p.relative_to(ROOT)}: -> {t2}")
                if apply:
                    p.unlink()
                    p.symlink_to(t2)
    # 2: text in tracked files and in the world.json of every row that stays
    for p in sorted(paths):
        if p.is_symlink() or not p.is_file():
            continue
        if p.name == "world.json":
            if stale(p.parent):
                continue
        elif p.resolve() not in tr:
            continue
        try:
            s = p.read_text()
        except UnicodeDecodeError:
            continue
        s2 = renamed(s)
        if s2 != s:
            print(f"rewrite {p.relative_to(ROOT)}")
            if apply:
                p.write_text(s2)
    # 3: names of what stays, deepest first
    for p in sorted(paths, key=lambda q: -len(q.parts)):
        if stale(p):
            continue
        q = p.with_name(renamed(p.name))
        if q == p:
            continue
        if q.exists() or q.is_symlink():
            raise SystemExit(f"{q} exists: not overwriting it with {p}")
        git = p.resolve() in tr and not p.is_symlink()
        print(f"{'git mv' if git else 'rename'} {p.relative_to(ROOT)} -> {q.name}")
        if apply:
            if git:
                subprocess.run(["git", "mv", str(p), str(q)], cwd=REPO, check=True)
            else:
                p.rename(q)
    # 4: stale entries under a rows directory, to rows_old_bcost
    for p in paths:
        rel = p.relative_to(ROOT).parts
        if OLD not in rel[:-1] or not stale(p):
            continue
        k = rel.index(OLD)
        q = ROOT.joinpath(*rel[:k], NEW, *rel[k + 1:])
        if q.exists() or q.is_symlink():
            raise SystemExit(f"{q} exists: not overwriting it with {p}")
        print(f"move {p.relative_to(ROOT)} -> {q.relative_to(ROOT)}")
        if apply:
            q.parent.mkdir(parents=True, exist_ok=True)
            p.rename(q)


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--apply"]):
        raise SystemExit(__doc__)
    main(sys.argv[1:] == ["--apply"])
