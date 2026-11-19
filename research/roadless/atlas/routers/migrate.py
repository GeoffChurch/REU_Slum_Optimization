"""The router token's migration (ears.Router, 2026-11-19): every Shock Reblock paths plan named
before it ran on the 8-neighbour grid, so its name gains `.grid8` after `paths` (no-legacy rule:
the old names are refused by ears.Plan.of, so nothing reads them until they are migrated).

`ears.paths.b<bld>...` becomes `ears.paths.grid8.b<bld>...` wherever it appears under
research/roadless: in the names of directories and files (growth_rows/<block>/<plan>/, the
crossings and lookahead rows and logs, roads5810's plates, atlas/blocks' grid json), in symlink
targets, in world.json's `plan`, and in the text of tracked files (EARS_RUN, EARS_PATHS, SHOCK,
the run lists, docstrings, roads5810/index.html). The pattern needs a digit after `paths.b`, so a
migrated name never matches again (idempotent). Tracked files are renamed with `git mv`.

    M=research/roadless/atlas/routers/migrate.py       # from the repo root
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
OLD = re.compile(r"ears\.paths\.(?=b[0-9])")
NEW = "ears.paths.grid8."
SKIP = {".venv", "__pycache__", ".git"}


def tracked() -> set[Path]:
    out = subprocess.run(["git", "ls-files", "-z", str(ROOT)], cwd=REPO, check=True,
                         capture_output=True).stdout.decode().split("\0")
    return {(REPO / p).resolve() for p in out if p}


def walk() -> list[Path]:
    out = []
    for d, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [x for x in dirs if x not in SKIP]
        out += [Path(d) / x for x in dirs + files]
    return out


def main(apply: bool) -> None:
    tr = tracked()
    paths = walk()
    # 1: symlinks' targets (before anything moves under them)
    links = [p for p in paths if p.is_symlink() and OLD.search(os.readlink(p))]
    for p in links:
        t = os.readlink(p)
        print(f"relink {p.relative_to(ROOT)}: {t} -> {OLD.sub(NEW, t)}")
        if apply:
            p.unlink()
            p.symlink_to(OLD.sub(NEW, t))
    # 2: text in tracked files and in every world.json
    for p in sorted(paths):
        if p.is_symlink() or not p.is_file():
            continue
        if not (p.resolve() in tr or p.name == "world.json"):
            continue
        try:
            s = p.read_text()
        except UnicodeDecodeError:
            continue
        n = len(OLD.findall(s))
        if n:
            print(f"rewrite {p.relative_to(ROOT)}: {n} name(s)")
            if apply:
                p.write_text(OLD.sub(NEW, s))
    # 3: names, deepest first
    for p in sorted(paths, key=lambda q: -len(q.parts)):
        if not OLD.search(p.name):
            continue
        q = p.with_name(OLD.sub(NEW, p.name))
        if q.exists():
            raise SystemExit(f"{q} exists: not overwriting it with {p}")
        git = p.resolve() in tr
        print(f"{'git mv' if git else 'rename'} {p.relative_to(ROOT)} -> {q.name}")
        if apply:
            if git:
                subprocess.run(["git", "mv", str(p), str(q)], cwd=REPO, check=True)
            else:
                p.rename(q)


if __name__ == "__main__":
    if sys.argv[1:] not in ([], ["--apply"]):
        raise SystemExit(__doc__)
    main(sys.argv[1:] == ["--apply"])
