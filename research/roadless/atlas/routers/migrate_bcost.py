"""Rev 4's migration (ears.REV, owner 2026-11-22): b now charges for the building share of the
road's cross-section (ears.road_share), not for the centreline's pixel being a building's, and binf
is the limit b -> inf (lexicographic), not buildings impassable.

A b0 plan's paths cost 1 a metre either way, and a divides plan reads no b: their rev 3 rows are
rev 4's, so `.v3` becomes `.v4` in their names wherever they appear under research/roadless (the
names of directories and files, symlink targets, world.json's `plan`, the text of tracked files;
tracked files renamed with `git mv`), as atlas/routers/migrate.py did for the router token.

A b > 0 plan's rev 3 rows (b0.05, b20, binf, ...) are another cost's: each one under a `rows`
directory (atlas/crossings/rows/<world>/, atlas/lookahead/rows/[check/]<block>/) moves, not
deleted, to the same place under the sibling `rows_old_bcost` (gitignored by `rows*/`), and a
symlink to one is re-pointed there. Elsewhere (growth_rows, roads5810's plates, atlas/blocks' grid
json, and the tracked text naming them: the Demo's and roads5810's pages and builders) a b > 0
rev 3 name stays as it is: those are rows of the published pages, read under their own names,
and regenerating them is the owner's call.

A rows archive (`rows_*`: rows_old_bcost, rows_steiner_deleted) is left as it is.
Idempotent: a migrated name matches neither pattern again. A destination that exists stops it.

    M=research/roadless/atlas/routers/migrate_bcost.py   # from the repo root
    python $M            # dry run: what it would do
    python $M --apply
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # research/roadless
REPO = ROOT.parents[1]
PLAN = re.compile(r"ears\.(?:paths\.[a-z0-9]+\.b(?P<b>[0-9.]+|inf)\.f|divides\.)"
                  r"[A-Za-z0-9.+-]*?\.D[0-9.]+\.v3(?![0-9])")
OLD = "rows"
NEW = "rows_old_bcost"
SKIP = {".venv", "__pycache__", ".git"}


def skipped(d: str) -> bool:
    """A directory not walked: tooling's, and a rows archive (rows_old_bcost, rows_steiner_deleted:
    rows of plans no longer run, kept as they were)."""
    return d in SKIP or d.startswith(OLD + "_")


def same(m: re.Match[str]) -> bool:
    """Whether the rev 3 plan named by m is rev 4's too: b0, or divides (no b)."""
    return m["b"] is None or float(m["b"]) == 0.0


def renamed(s: str) -> str:
    """s with every b0 or divides rev 3 plan name made rev 4's."""
    return PLAN.sub(lambda m: m[0][:-1] + "4" if same(m) else m[0], s)


def stale(name: str) -> bool:
    """Whether `name` names a b > 0 rev 3 plan."""
    return any(not same(m) for m in PLAN.finditer(name))


def moved(path: str) -> str:
    """`path` (a symlink's target) re-pointed: a b > 0 rev 3 entry under a `rows` directory to
    the same place under `rows_old_bcost`, b0 names renamed."""
    parts = Path(path).parts
    for k, part in enumerate(parts):
        if part == OLD and any(stale(q) for q in parts[k + 1:]):
            parts = (*parts[:k], NEW, *parts[k + 1:])
            break
    return renamed(str(Path(*parts))) + ("/" if path.endswith("/") else "")


def tracked() -> set[Path]:
    out = subprocess.run(["git", "ls-files", "-z", str(ROOT)], cwd=REPO, check=True,
                         capture_output=True).stdout.decode().split("\0")
    return {(REPO / p).resolve() for p in out if p}


def walk() -> list[Path]:
    """Every entry under ROOT, not following symlinks, not descending into a stale entry (it
    moves whole) nor into SKIP."""
    out = []
    for d, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [x for x in dirs if not skipped(x)]
        out += [Path(d) / x for x in dirs + files]
        dirs[:] = [x for x in dirs if not (stale(x) and OLD in Path(d).relative_to(ROOT).parts)]
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
    # 2: text in tracked files and in every world.json (b0 and divides names only)
    for p in sorted(paths):
        if p.is_symlink() or not p.is_file():
            continue
        if not (p.resolve() in tr or p.name == "world.json"):
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
    # 3: b0 and divides names, deepest first
    for p in sorted(paths, key=lambda q: -len(q.parts)):
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
    # 4: b > 0 rev 3 entries under a rows directory, to rows_old_bcost
    for p in paths:
        rel = p.relative_to(ROOT).parts
        if OLD not in rel[:-1] or not stale(p.name):
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
