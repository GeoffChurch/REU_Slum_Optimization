"""Does `Solver.release` (a new System first hands cupy's cached free blocks back) change a
result? The translucent greedy on one block, once as it is and once with the release a no-op,
each in a fresh process; every step's D, perm, perm1 and clearing order must be equal. Prints the
verdict and exits non-zero on any difference.

    python -u research/roadless/releaseprobe.py <id> <along>[@<search>] <d_max>
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pandas as pd  # noqa: E402

COLUMNS = ["step", "D", "perm", "perm1", "cleared"]


def variant(bid: str, along: str, d_max: float, out: Path, release: bool) -> None:
    """One greedy, rows to `out` (run in its own process)."""
    import clear
    import common
    import lifted
    if not release:
        lifted.GpuAMG.release = lambda self: None   # type: ignore[method-assign]
    scans = lifted.scans_of("gpu")
    specs = along.split("@")
    (b,) = common.build_blocks([bid])
    clear.greedy_block(b, clear.picker_of("S0.01cat", clear.sweep_of("gpu")), 0.5, d_max, out,
                       common.POPULATIONS["area"], 2.0, lifted.along_of(specs[0], scans),
                       lifted.solver_of("gpu"),
                       lifted.along_of(specs[1], scans) if len(specs) == 2 else None)


def main(bid: str, along: str, d_max: str) -> int:
    with tempfile.TemporaryDirectory() as d:
        rows = {}
        for release in (True, False):
            out = Path(d) / f"{release}.parquet"
            subprocess.run([sys.executable, "-u", __file__, "variant", bid, along, d_max, str(out),
                            str(release)], check=True)
            rows[release] = pd.read_parquet(out)[COLUMNS]
    a, b = rows[True], rows[False]
    same = len(a) == len(b) and all(
        a[c].tolist() == b[c].tolist() if c != "cleared"
        else [list(x) for x in a[c]] == [list(x) for x in b[c]] for c in COLUMNS)
    print(f"{bid}: {len(a) - 1} steps with release, {len(b) - 1} without; "
          f"{'bit-identical' if same else 'DIFFERENT'} (D, perm, perm1, clearing order)")
    if not same:
        print(pd.concat({"release": a, "none": b}, axis=1).to_string())
    return 0 if same else 1


if __name__ == "__main__":
    if sys.argv[1] == "variant":
        variant(sys.argv[2], sys.argv[3], float(sys.argv[4]), Path(sys.argv[5]),
                sys.argv[6] == "True")
    else:
        sys.exit(main(*sys.argv[1:4]))
