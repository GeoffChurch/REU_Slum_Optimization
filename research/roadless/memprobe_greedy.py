"""Where the greedy's GPU memory goes, per step and phase, on one block (the diagnosis of the
translucent greedy running out of memory on 30796 at 48 GB). Before and after each phase
(tension, gram, scoring solve) it prints the cupy pool's used and held bytes and the bytes of
device arrays reachable from each long-lived holder: the grid's cache (solver structures,
patterns), each conductance's layers cache, the Clearing. On a failed allocation it prints the
same, then re-raises. Rows are not written: this is a probe, not a run.

    python -u research/roadless/memprobe_greedy.py <id> <along>[@<search>] <d_max> [default|async|release|gc]
"""
from __future__ import annotations

import dataclasses
import gc
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clear  # noqa: E402
import common  # noqa: E402
import cupy as cp  # noqa: E402
import lifted  # noqa: E402

GB = 2**30
PEAK = [0]                      # the pool's most bytes in use since the last report
POOL = [cp.get_default_memory_pool()]   # the pool allocations come from (argv: default | async)


def _tracking_malloc(size: int):
    """The pool's malloc, recording its high-water mark of bytes in use."""
    pool = POOL[0]
    mem = pool.malloc(size)
    PEAK[0] = max(PEAK[0], pool.used_bytes())
    return mem


def dev_bytes(obj, seen: set[int] | None = None) -> int:
    """Bytes of cupy arrays reachable from obj through containers and object fields."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, cp.ndarray):
        return obj.nbytes
    if hasattr(obj, "data") and hasattr(obj, "indices") and hasattr(obj, "indptr"):
        return sum(dev_bytes(getattr(obj, a), seen) for a in ("data", "indices", "indptr"))
    if isinstance(obj, dict):
        return sum(dev_bytes(v, seen) for v in obj.values())
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(dev_bytes(v, seen) for v in obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return sum(dev_bytes(getattr(obj, f.name), seen) for f in dataclasses.fields(obj))
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return sum(dev_bytes(v, seen) for v in vars(obj).values())
    return 0


def report(tag: str, c: clear.Clearing) -> None:
    pool = POOL[0]
    free, total = cp.cuda.Device().mem_info
    parts = {f"grid.cache[{k!r}]"[:60]: dev_bytes(v) for k, v in c.sc.grid.cache.items()}
    parts["p.along._cache"] = dev_bytes(getattr(c.p.along, "_cache", {}))
    if c.ps.along is not c.p.along:
        parts["ps.along._cache"] = dev_bytes(getattr(c.ps.along, "_cache", {}))
    parts["clearing (bfree, inside_flat)"] = dev_bytes([c.bfree, c.inside_flat])
    parts["scorer (but its grid)"] = dev_bytes({k: v for k, v in vars(c.sc).items() if k != "grid"})
    big = sorted(((v, k) for k, v in parts.items() if v > 0.05 * GB), reverse=True)
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: pool used {pool.used_bytes() / GB:.2f} "
          f"(peak since last {PEAK[0] / GB:.2f}) held {pool.total_bytes() / GB:.2f}, "
          f"device free {free / GB:.2f} of {total / GB:.2f}"
          + "".join(f"\n    {v / GB:6.2f} GB  {k}" for v, k in big), flush=True)
    PEAK[0] = pool.used_bytes()


def main(bid: str, spec: str, d_max: float, allocator: str) -> None:
    if allocator == "async":
        POOL[0] = cp.cuda.MemoryAsyncPool()
    release = allocator == "release"    # free the pool's unused chunks before each phase
    collect = allocator == "gc"         # run the cycle collector before each phase
    cp.cuda.set_allocator(_tracking_malloc)
    specs = spec.split("@")
    scans = lifted.scans_of("gpu")
    along = lifted.along_of(specs[0], scans)
    search = lifted.along_of(specs[1], scans) if len(specs) == 2 else None
    (b,) = common.build_blocks([bid])
    p = lifted.Params(ell_m=3.0, K=8, along=along, solver=lifted.solver_of("gpu"))
    mesh = lifted.UniformMesh(0.5, offset=lifted.OFFSET)
    c = clear.Clearing(b, mesh, p, population=common.POPULATIONS["area"], search=search)
    g = c.sc.grid
    print(f"{bid} {spec}: grid {g.inside.shape}, inside {int(g.inside.sum())}, "
          f"layers array {8 * g.inside.size * 8 / GB:.2f} GB (K x ny x nx float64)", flush=True)
    report("after setup", c)

    picker = clear.picker_of("S0.01cat", clear.sweep_of("gpu"))
    tension, gram, exact = c.tension, picker.source.gram, clear._exact
    step = [0]

    def tension_(power, restore=False):
        step[0] += 1
        report(f"step {step[0]} before tension", c)
        if release:
            POOL[0].free_all_blocks()
            report(f"step {step[0]} released", c)
        if collect:
            print(f"    gc.collect: {gc.collect()} objects", flush=True)
            report(f"step {step[0]} collected", c)
        t = tension(power, restore)
        report(f"step {step[0]} after tension", c)
        return t

    def gram_(c_, t, cand, power):
        H = gram(c_, t, cand, power)
        report(f"step {step[0]} after gram ({len(cand)} candidates)", c)
        return H

    def exact_(c_, cleared, power):
        report(f"step {step[0]} before scoring", c)
        if release:
            POOL[0].free_all_blocks()
            report(f"step {step[0]} released", c)
        if collect:
            print(f"    gc.collect: {gc.collect()} objects", flush=True)
            report(f"step {step[0]} collected", c)
        try:
            out = exact(c_, cleared, power)
        except Exception:
            report(f"step {step[0]} FAILED in scoring", c)
            raise
        report(f"step {step[0]} after scoring", c)
        return out

    c.tension = tension_
    object.__setattr__(picker.source, "gram", gram_)      # the source is a frozen dataclass
    clear._exact = exact_
    J0 = c.sc.J(c.sc.u0, 2.0)
    for pk, D in clear.grow(c, picker, 2.0, d_max, J0, True):
        print(f"  step {step[0]} D {D:.3f} perm' {1 - (pk.J / J0) ** 0.5:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]),
         sys.argv[4] if len(sys.argv) > 4 else "default")
