"""Unpack a tar of inputs into a cluster task's checkout, each file renamed into place (cluster.py's
`cpu` kind runs this before a task's command).

The tasks of a run share their checkout, and the `cpu` kind runs a block's items as separate
tasks that start together, each unpacking the block's tar. `tar -x` writes a file where it
lies, so one task could read an input another task's extraction has just truncated. Here a task
extracts into a directory of its own inside the checkout and renames each file onto its path: a
rename replaces a file whole (the same filesystem), so a reader opens the old copy or the new,
both complete and the same. Only regular files, at relative paths below the checkout
(tarfile's `data` filter refuses the rest).

    python -u research/roadless/cluster_unpack.py <tar>      # in the checkout's root
"""
from __future__ import annotations

import os
import sys
import tarfile
import tempfile
from pathlib import Path


def unpack(tar: Path, root: Path) -> int:
    """`tar`'s files into `root` at their paths, each by one rename; returns how many."""
    with tarfile.open(tar) as tf:
        members = tf.getmembers()
        odd = [m.name for m in members if not (m.isfile() or m.isdir())]
        if odd:
            raise SystemExit(f"{tar}: only files and directories, not {odd[:3]}")
        with tempfile.TemporaryDirectory(dir=root, prefix=".unpack-") as tmp:
            tf.extractall(tmp, filter="data")
            files = [m.name for m in members if m.isfile()]
            for name in files:
                dst = root / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.replace(Path(tmp) / name, dst)
    return len(files)


if __name__ == "__main__":
    [arg] = sys.argv[1:]
    n = unpack(Path(arg), Path.cwd())
    print(f"cluster_unpack: {n} files from {arg}", flush=True)
