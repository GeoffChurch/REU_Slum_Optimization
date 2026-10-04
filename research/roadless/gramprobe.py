"""Where the greedy's gram (Catchment) spends its time, per step: the sweep's prepare (Kahn's
waves, a launch and a host read per wave), the number of waves, and each chunk's run (a launch
per wave), against the gram's total. The greedy itself is unchanged (S0.01cat, J_2).

    python -u research/roadless/gramprobe.py <id> <along> <d_max>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import cupy as cp  # noqa: E402
import lifted  # noqa: E402

STATS: dict = {}


def _synced(f):
    cp.cuda.Device().synchronize()
    t = time.time()
    out = f()
    cp.cuda.Device().synchronize()
    return out, time.time() - t


def _prepare(orig):
    def prepare(self, A, u, rowsum):
        run, dt = _synced(lambda: orig(self, A, u, rowsum))
        waves = run.__closure__[run.__code__.co_freevars.index("waves")].cell_contents
        STATS.update(prepare=dt, waves=len(waves) - 1, n=len(u), runs=[])

        def run_(rows, cols, C):
            out, dt = _synced(lambda: run(rows, cols, C))
            STATS["runs"].append((C, dt))
            return out
        return run_
    return prepare


def main(bid: str, along: str, d_max: float) -> None:
    scans = lifted.scans_of("gpu")
    (b,) = common.build_blocks([bid])
    p = lifted.Params(ell_m=3.0, K=8, along=lifted.along_of(along, scans),
                      solver=lifted.solver_of("gpu"))
    c = clear.Clearing(b, 0.5, p, population=common.POPULATIONS["area"])
    clear.GpuSweep.prepare = _prepare(clear.GpuSweep.prepare)
    picker = clear.picker_of("S0.01cat", clear.sweep_of("gpu"))
    gram = picker.source.gram
    step = [0]

    def gram_(c_, t, cand, power):
        step[0] += 1
        H, dt = _synced(lambda: gram(c_, t, cand, power))
        runs = STATS["runs"]
        print(f"step {step[0]}: gram {dt:.2f}s, {len(cand)} cand | sweep prepare "
              f"{STATS['prepare']:.2f}s, {STATS['waves']} waves over {STATS['n']} unknowns | "
              f"{len(runs)} chunks, run " + " + ".join(f"{r:.2f}s ({C})" for C, r in runs)
              + f" | rest {dt - STATS['prepare'] - sum(r for _, r in runs):.2f}s", flush=True)
        return H

    object.__setattr__(picker.source, "gram", gram_)
    J0 = c.sc.J(c.sc.u0, 2.0)
    for _pk, _D in clear.grow(c, picker, 2.0, d_max, J0):
        pass


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]))
