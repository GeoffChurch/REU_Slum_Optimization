"""The road lineup on the large blocks, scored three ways and set against the stored clearings
(BACKLOG 4(b), "the price of roads"): the 4-block sentinel before any full large-block run.

Each method's network (reblock.derivations.propose, the derive cache), cut to its canonical
street-first prefixes (common.prefix_to) at BUDGETS and whole, scored
  (a) by the shipped permeability (reblock.permeability) on COUNT-weight prefixes;
  (b) roadless, the walking model changed only: count population, p 1, the same prefixes;
  (c) roadless as the optimizers use it: area population, J_2, on AREA-weight prefixes;
CARVE rule (common.Carve), every solve keeping both powers (P and J_2 come from one solve).
The stored clearings (SIMP default, the cheap preset) are rescored on the same scorer: their
stored Lens A is the check that the scorer reproduces the optimizers' own.

    S=research/roadless/road_rescore.py    # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S propose <scratch> <id> <method> <cache|fresh>
    uv run python $S propose-all <scratch> <procs> <ids,> <methods,> <timeout h> <time-hits|no>
    uv run python $S prefixes <scratch> <procs> <ids,>
    CUDA_PATH=/usr uv run python $S score <scratch> <mesh> <pops,> <ids,>
    uv run python $S report <mesh> <composite mesh>
    uv run python $S sizes <composite mesh> <procs>
    uv run python $S project <mesh> <composite mesh>

The sentinel (BACKLOG, "How we run experiments"): ZAF.9.3.1_1_22422, ZAF.9.3.1_1_30848,
ZAF.9.3.1_1_5810, ZAF.9.5.4_1_9712. Lens (b) reads P (p 1) from the count rows, (c) J_2 from
the area rows; every row keeps both.

<scratch>: networks, ordered networks, logs (bulky, not committed). propose `cache` goes through
the derive cache (and says whether it hit); `fresh` calls the method directly, bypassing the
cache, to time a cached proposal (propose-all queues one after every cache hit) and to check the
cached roads against a recomputation. Rows: road_rows/proposals/ (feasibility), road_rows/
prefixes_<budgets>/ (the prefixes, their displacement under both weightings, and (a)),
road_rows/<mesh>_ell<ell>_K<K>_<along>_<pop>_<budgets>/ ((b) for pop count, (c) for area, and the
stored clearings rescored on that mesh). Every process single-threaded; one GPU process.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402

ROADLESS = common.REPO / "research" / "roadless"
ROWS = ROADLESS / "road_rows"

# ------------------------------------------------------------------ the run's parameters
# Everything that can change a number is here or on the command line, and in the rows' paths.
BUDGETS = (0.02, 0.05, 0.10, 0.15, 0.20)     # displacement budgets of the canonical prefixes
ELL_M = 3.0                                  # turning length (m)
K = 8                                        # headings
ALONG = "uni"                                # the along conductance (lifted.along_of)
DEVICE = "gpu"                               # solver and scans (lifted.solver_of / scans_of)
SCORE_RTOL = 1e-5                            # clear.RTOL_SCORE: the stored clearings' exact score
CLEARING_POWER = 2.0                         # the stored clearings' objective (J_2)
# The stored clearings at D 0.05 (area population, J_2, uni): (rows, Lens A column, cleared
# column, D column, the mesh they were optimized and scored on).
STORED = {
    "simp_default": ("relax_rows/uni/fw0.q3.i10.t0.001.e0.0001.k1s1e-06.p256w8x2/D0.05",
                     "simp_perm", "simp_cleared", "simp_D", "0.5"),
    "cheap": ("polish_rows/uni/S0.01cat.P64w8/D0.05", "perm", "cleared", "D", "0.5"),
    "cheap_a5x8": ("polish_rows/uni/S0.01cat.P64w8.a5x8/D0.05", "perm", "cleared", "D",
                   "0.5a5x8"),
}
SHORT_SLOTS = 3                              # propose-all: slots kept for the cheapest jobs
# propose-all's ORDER only (no result depends on it): a guess at each method's cost per building
COST_GUESS = {"cycle_native_betweenness_contrast": 3.0, "cycle_native": 2.5,
              "greedy_arterial_access_displacement": 2.0, "resistance_lp": 2.0,
              "clearance_looped": 1.0, "clearance": 0.7, "euclidean_grid": 0.2}
SINGLE = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
              NUMBA_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")


def budgets_tag() -> str:
    return "D" + "-".join(f"{b:g}" for b in BUDGETS)


def score_dir(mesh_token: str, pop: str) -> Path:
    mesh = common.mesh_of(mesh_token)
    return ROWS / f"{mesh.name}_ell{ELL_M:g}_K{K}_{ALONG}_{pop}_{budgets_tag()}"


def prefix_dir() -> Path:
    return ROWS / f"prefixes_{budgets_tag()}"


def net_path(scratch: Path, bid: str, name: str, fresh: bool) -> Path:
    return scratch / "networks" / bid / f"{name}{'.fresh' if fresh else ''}.parquet"


def ordered_path(scratch: Path, bid: str, name: str) -> Path:
    return scratch / "networks" / bid / f"{name}.ordered.parquet"


def proposal_row(bid: str, name: str, fresh: bool) -> Path:
    return ROWS / "proposals" / f"{bid}__{name}{'__fresh' if fresh else ''}.parquet"


def _check_single_threaded() -> None:
    bad = {k: os.environ.get(k) for k, v in SINGLE.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f"run single-threaded: set {bad} to {SINGLE}")


# ------------------------------------------------------------------ memory and time

def _status_gb(field: str) -> float:
    """/proc/self/status `field` (VmRSS, VmHWM) in GB."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) / 2 ** 20
    raise RuntimeError(f"no {field} in /proc/self/status")


class Sampler:
    """Peak RSS while a block of work runs (sampled every `every` s), CPU and wall time, and a
    progress file rewritten every `write_every` s: what is known of a job that gets killed."""

    def __init__(self, progress: Path, every: float, write_every: float):
        self.progress, self.every, self.write_every = progress, every, write_every
        self.peak = 0.0
        self._stop = threading.Event()

    def _run(self) -> None:
        last = 0.0
        while not self._stop.wait(self.every):
            self.peak = max(self.peak, _status_gb("VmRSS"))
            if time.perf_counter() - last >= self.write_every:
                last = time.perf_counter()
                self._write()

    def _write(self) -> None:
        tmp = self.progress.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(wall_s=time.perf_counter() - self.t0,
                                       cpu_s=time.process_time() - self.c0,
                                       peak_rss_gb=self.peak)))
        os.replace(tmp, self.progress)

    def __enter__(self) -> Sampler:
        self.progress.parent.mkdir(parents=True, exist_ok=True)
        self.t0, self.c0 = time.perf_counter(), time.process_time()
        self.peak = _status_gb("VmRSS")
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._th.join()
        self.peak = max(self.peak, _status_gb("VmRSS"))
        self.wall, self.cpu = time.perf_counter() - self.t0, time.process_time() - self.c0
        self._write()


# ------------------------------------------------------------------ propose

def _arms() -> dict:
    """common.arms(), with the arterial's shortlist on one thread (its `threads` is outside the
    method's identity: the same roads, the same cache key)."""
    out = common.arms()
    m = out["greedy_arterial_access_displacement"]
    out["greedy_arterial_access_displacement"] = dataclasses.replace(
        m, engine=dataclasses.replace(m.engine, threads=1))
    return out


def propose(scratch: Path, bid: str, name: str, fresh: bool) -> None:
    """One (block, method) network, timed; the roads to <scratch>/networks, a feasibility row to
    road_rows/proposals."""
    _check_single_threaded()
    from reblock import derivations
    from reblock.derive_graph import DerivationKey, _code_version, _l2, env_version
    t_all = time.perf_counter()
    [b] = common.build_blocks([bid])
    method = _arms()[name]
    t_build = time.perf_counter() - t_all
    rss_before = _status_gb("VmRSS")
    fn = derivations._propose_impl
    key = DerivationKey(fn=f"{fn.__module__}.{fn.__qualname__}",
                        code=_code_version(fn, (method, b)), env=env_version(),
                        inputs=(method.identity, b.identity))
    cached = bool(_l2.check_call_in_cache(key, fn, (method, b)))
    print(f"{time.strftime('%H:%M:%S')} {bid} n={len(b.buildings)} {name} "
          f"{'fresh' if fresh else 'cache'}: cached {cached}, block built in {t_build:.1f}s, "
          f"RSS {rss_before:.2f} GB", flush=True)
    progress = scratch / "progress" / f"{bid}__{name}{'__fresh' if fresh else ''}.json"
    row = dict(block=bid, n=len(b.buildings), area_km2=b.boundary.area / 1e6, method=name,
               fresh=fresh, cache_hit=cached, rss_before_gb=rss_before)
    try:
        with Sampler(progress, every=0.5, write_every=30.0) as s:
            prop = fn(method, b) if fresh else derivations.propose(method, b)
        roads = prop.roads
        n_roads = 0 if roads is None else len(roads)
        road_m = 0.0 if roads is None else float(roads.geometry.length.sum())
        out = net_path(scratch, bid, name, fresh)
        out.parent.mkdir(parents=True, exist_ok=True)
        if roads is not None:
            roads.to_parquet(out)
        same = np.nan
        if fresh and net_path(scratch, bid, name, False).exists():
            import geopandas as gpd
            other = gpd.read_parquet(net_path(scratch, bid, name, False))
            same = float(len(other) == n_roads and bool(
                other.geometry.reset_index(drop=True).geom_equals_exact(
                    roads.geometry.reset_index(drop=True), tolerance=0.0).all()))
        row.update(status="ok", wall_s=s.wall, cpu_s=s.cpu, peak_rss_gb=s.peak,
                   hwm_gb=_status_gb("VmHWM"), n_roads=n_roads, road_m=road_m,
                   same_as_cached=same)
        print(f"{time.strftime('%H:%M:%S')} {bid} {name}: {n_roads} roads, {road_m:.0f} m, "
              f"{s.wall:.0f}s wall, {s.cpu:.0f}s CPU, peak RSS {s.peak:.2f} GB", flush=True)
    except Exception as e:                        # a failure is a finding: record it
        traceback.print_exc()
        row.update(status=f"failed {type(e).__name__}: {e}"[:300], hwm_gb=_status_gb("VmHWM"))
    row["total_wall_s"] = time.perf_counter() - t_all
    dest = proposal_row(bid, name, fresh)
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_parquet(dest)


def propose_all(scratch: Path, procs: int, ids: list[str], methods: list[str], timeout_s: float,
                time_hits: bool) -> None:
    """Every (block, method) network without a row, for `methods` of the lineup, `procs`
    processes at a time: the costliest first (COST_GUESS x buildings) on procs - SHORT_SLOTS
    slots, the cheapest first on the rest (so small blocks finish early and can be scored). With
    `time_hits` a cache hit queues a fresh run of the same pair (its cost). A job past
    `timeout_s` of wall time is killed (by PID) and recorded: a finding (the sentinel ran 2 h)."""
    _check_single_threaded()
    unknown = sorted(set(methods) - set(common.LINEUP))
    if unknown:
        raise SystemExit(f"not in common.LINEUP: {unknown}")
    sizes = pd.read_parquet(ROADLESS / "large82_area.parquet").set_index("block")["n"]
    snap = scratch / "road_rescore_snapshot.py"         # the children run this copy: edits to the
    shutil.copy(__file__, snap)                          # script mid-run cannot reach them
    env = {**os.environ, **SINGLE,
           "PYTHONPATH": os.pathsep.join([str(common.REPO), str(ROADLESS)])}
    (scratch / "logs").mkdir(parents=True, exist_ok=True)

    def cost(job: tuple[str, str, bool]) -> float:
        return float(sizes[job[0]]) * COST_GUESS[job[1]]

    queue = sorted(((bid, name, False) for bid in ids for name in methods
                    if not proposal_row(bid, name, False).exists()), key=cost, reverse=True)
    for bid in ids if time_hits else []:                # fresh runs of hits already recorded
        for name in methods:
            p = proposal_row(bid, name, False)
            if (p.exists() and bool(pd.read_parquet(p).cache_hit.iloc[0])
                    and not proposal_row(bid, name, True).exists()):
                queue.append((bid, name, True))
    queue.sort(key=cost, reverse=True)
    running: dict[int, tuple] = {}                      # pid -> (Popen, job, start, long?, log)
    print(f"{time.strftime('%H:%M:%S')} {len(queue)} jobs, {procs} at a time", flush=True)
    while queue or running:
        while queue and len(running) < procs:
            n_long = sum(1 for r in running.values() if r[3])
            long_ = n_long < procs - SHORT_SLOTS
            job = queue.pop(0) if long_ else queue.pop()
            bid, name, fresh = job
            log = scratch / "logs" / f"propose_{bid}__{name}{'__fresh' if fresh else ''}.log"
            fh = open(log, "w")
            p = subprocess.Popen([sys.executable, str(snap), "propose", str(scratch), bid, name,
                                  "fresh" if fresh else "cache"], stdout=fh,
                                 stderr=subprocess.STDOUT, env=env, cwd=str(common.REPO))
            running[p.pid] = (p, job, time.perf_counter(), long_, fh)
            print(f"{time.strftime('%H:%M:%S')} start pid {p.pid} {bid} {name}"
                  f"{' fresh' if fresh else ''} ({'long' if long_ else 'short'} slot)",
                  flush=True)
        time.sleep(5.0)
        for pid, (p, job, t0, _long, fh) in list(running.items()):
            bid, name, fresh = job
            rc = p.poll()
            if rc is None and time.perf_counter() - t0 > timeout_s:
                p.kill()                                # by PID, never a pattern
                p.wait()
                prog = scratch / "progress" / f"{bid}__{name}{'__fresh' if fresh else ''}.json"
                known = json.loads(prog.read_text()) if prog.exists() else {}
                dest = proposal_row(bid, name, fresh)
                dest.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([dict(block=bid, n=int(sizes[bid]), method=name, fresh=fresh,
                                   status=f"timeout after {timeout_s:.0f} s wall",
                                   wall_s=known.get("wall_s", np.nan),
                                   cpu_s=known.get("cpu_s", np.nan),
                                   peak_rss_gb=known.get("peak_rss_gb", np.nan))]).to_parquet(dest)
                rc = "killed"
            if rc is None:
                continue
            fh.close()
            del running[pid]
            print(f"{time.strftime('%H:%M:%S')} done pid {pid} {bid} {name}"
                  f"{' fresh' if fresh else ''}: rc {rc}, {time.perf_counter() - t0:.0f}s",
                  flush=True)
            row = proposal_row(bid, name, fresh)
            if (time_hits and rc == 0 and not fresh and row.exists()
                    and bool(pd.read_parquet(row).cache_hit.iloc[0])
                    and not proposal_row(bid, name, True).exists()):
                queue.append((bid, name, True))
                queue.sort(key=cost, reverse=True)
                print(f"  cache hit: queued a fresh run of {bid} {name}", flush=True)


# ------------------------------------------------------------------ prefixes (CPU)

POPS = ("count", "area")         # the two weightings: (a) and (b) on count, (c) on area


def prefix_len(block, ordered, target: float, w: np.ndarray) -> int:
    """common.prefix_to's length on roads already in canonical order (street_first_ordered once
    per network rather than once per budget): the shortest prefix whose `w`-weighted
    displacement reaches `target`, or all of them. Checked against common.prefix_to on every
    network (prefixes_one)."""
    if common.displacement(block, ordered, w) < target:
        return len(ordered)
    lo, hi = 0, len(ordered)
    while lo < hi:
        mid = (lo + hi) // 2
        if common.displacement(block, ordered.iloc[:mid], w) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


_BLOCKS: dict = {}
_SCRATCH: list[Path] = []


def prefixes_one(job: tuple[str, str]) -> str | None:
    """One network's canonical prefixes: per weighting and budget its length, and per distinct
    prefix its road length, displacement under both weightings and (a). The ordered network to
    <scratch>, the rows to prefix_dir(). The failed job's name, or None."""
    bid, name = job
    try:
        _prefixes_one(bid, name)
    except Exception as e:
        traceback.print_exc()
        print(f"{bid} {name} prefixes FAILED {type(e).__name__}: {e}"[:300], flush=True)
        return f"{bid} {name}"
    return None


def _prefixes_one(bid: str, name: str) -> None:
    import geopandas as gpd

    from reblock.budget import STREET_TOL, street_first_ordered
    from reblock.compare import load_permeability_config
    from reblock.permeability import EgressContext, permeability
    scratch, b = _SCRATCH[0], _BLOCKS[bid]
    t0 = time.perf_counter()
    roads = gpd.read_parquet(net_path(scratch, bid, name, False))
    ordered = street_first_ordered(b, roads, STREET_TOL)
    ordered.to_parquet(ordered_path(scratch, bid, name))
    t_order = time.perf_counter() - t0
    polys = np.asarray(b.buildings.outlines)
    W = {pop: common.POPULATIONS[pop].weights(polys) for pop in POPS}
    lens = [dict(pop=pop, budget=np.nan if bud is None else bud, full=bud is None,
                 m=len(ordered) if bud is None else prefix_len(b, ordered, bud, W[pop]))
            for pop in POPS for bud in (*BUDGETS, None)]
    # the same prefixes as common.prefix_to (which orders afresh for each budget)
    for pop in POPS:
        ref = common.prefix_to(b, roads, BUDGETS[1], W[pop])
        [m] = [r["m"] for r in lens if r["pop"] == pop and r["budget"] == BUDGETS[1]]
        if len(ref) != m or not ref.geometry.reset_index(drop=True).geom_equals_exact(
                ordered.geometry.iloc[:m].reset_index(drop=True), tolerance=0.0).all():
            raise AssertionError(f"{bid} {name} {pop}: prefix differs from common.prefix_to")
    t_pre = time.perf_counter() - t0
    ctx = EgressContext.of(b, load_permeability_config(common.REPO / "conf").params)
    per_m = {}
    for m in sorted({r["m"] for r in lens}):
        pre = ordered.iloc[:m]
        ta = time.perf_counter()
        P_old = permeability(ctx, pre)               # (a), on every distinct prefix
        per_m[m] = dict(n_roads=m, road_m=float(pre.geometry.length.sum()),
                        D_count=common.displacement(b, pre, W["count"]),
                        D_area=common.displacement(b, pre, W["area"]), P_old=P_old,
                        t_P_old=time.perf_counter() - ta)
    rows = [dict(block=bid, n=len(polys), method=name, **r, **per_m[r["m"]]) for r in lens]
    out = prefix_dir() / f"{bid}__{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["t_order"], df["t_prefixes"], df["t_total"] = t_order, t_pre, time.perf_counter() - t0
    df.to_parquet(out)
    print(f"{time.strftime('%H:%M:%S')} {bid} {name}: {len(ordered)} roads, order {t_order:.1f}s, "
          f"prefixes {t_pre:.1f}s, total {time.perf_counter() - t0:.1f}s", flush=True)


def prefixes(scratch: Path, procs: int, ids: list[str]) -> None:
    """prefixes_one for every network present without rows, `procs` processes."""
    _check_single_threaded()
    import multiprocessing
    jobs = [(bid, name) for bid in ids for name in common.LINEUP
            if net_path(scratch, bid, name, False).exists()
            and not (prefix_dir() / f"{bid}__{name}.parquet").exists()]
    if not jobs:
        print("nothing to do", flush=True)
        return
    _SCRATCH.append(scratch)
    _BLOCKS.update({b.block_id: b for b in common.build_blocks(sorted({j[0] for j in jobs}))})
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=4) as pool:
        failed = [f for f in pool.imap_unordered(prefixes_one, jobs) if f]
    if failed:
        raise SystemExit(f"{len(failed)} failed: {failed}")


# ------------------------------------------------------------------ score (GPU)

class GpuWatch:
    """Peak device memory of this process (nvidia-smi's per-process figure, every `every` s, in
    a thread) and the most cupy's pool held at the moments `note_pool` is called (after each
    solve, in the main thread), while a block is scored."""

    def __init__(self, every: float):
        self.every = every
        self.peak_mib = 0.0
        self.pool_peak_gb = 0.0
        self._stop = threading.Event()

    def note_pool(self) -> None:
        import cupy
        self.pool_peak_gb = max(self.pool_peak_gb,
                                cupy.get_default_memory_pool().total_bytes() / 2 ** 30)

    def _sample(self) -> None:
        pid = str(os.getpid())
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True).stdout
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2 and parts[0] == pid and parts[1].isdigit():
                self.peak_mib = max(self.peak_mib, float(parts[1]))

    def _run(self) -> None:
        while not self._stop.wait(self.every):
            self._sample()

    def __enter__(self) -> GpuWatch:
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._th.join()
        self._sample()


class Corridors:
    """Carve's free field for every prefix of one ordered network on one grid, road by road: a
    sub-sample is inside the union of the roads' buffers iff it is inside one of them, so each
    buffer tests only the sub-samples of the cells in its bounding box, and the prefixes cost
    about one corridor test between them where `grid.sub_of(road_corridor(prefix))` (Carve) tests
    every sub-sample of the block once per prefix (22 -- 30 s each on 22422). The same buffers
    as budget.road_corridor (width_m / 2); `check` compares the whole network's mask with Carve's
    own."""

    def __init__(self, grid, X: np.ndarray, Y: np.ndarray):
        self.grid, self.X, self.Y = grid, X, Y          # X, Y: grid._sub_xy(), computed once
        import lifted
        self.composite = isinstance(grid, lifted.CompositeGrid)
        if self.composite:
            half = grid.size / 2
            self.lo_x, self.hi_x = grid.xy[:, 0] - half, grid.xy[:, 0] + half
            self.lo_y, self.hi_y = grid.xy[:, 1] - half, grid.xy[:, 1] + half
        else:
            self.xs, self.ys = grid.xy[0, :, 0], grid.xy[:, 0, 1]

    def _cells(self, bounds) -> tuple:
        """An index into the grid's cells covering every sub-sample in `bounds`."""
        x0, y0, x1, y1 = bounds
        if self.composite:
            return (np.flatnonzero((self.hi_x >= x0) & (self.lo_x <= x1) & (self.hi_y >= y0)
                                   & (self.lo_y <= y1)),)
        h = self.grid.h
        c0, c1 = np.searchsorted(self.xs, x0 - h), np.searchsorted(self.xs, x1 + h, side="right")
        r0, r1 = np.searchsorted(self.ys, y0 - h), np.searchsorted(self.ys, y1 + h, side="right")
        return (slice(r0, r1), slice(c0, c1))

    def masks(self, ordered, stops: list[int]):
        """Yield (m, free field) for each m in `stops` (ascending) after the first m roads."""
        import shapely
        g = self.grid
        bufs = np.asarray(ordered.geometry.buffer(ordered["width_m"].to_numpy(dtype=float) / 2.0))
        mask = np.zeros_like(g.isub)
        want = sorted(set(stops))
        i = 0
        for m in want:
            while i < m:
                buf = bufs[i]
                idx = self._cells(buf.bounds)
                sub = g.isub[idx]
                shapely.prepare(buf)
                hit = np.zeros_like(sub)
                hit[sub] = shapely.contains_xy(buf, self.X[idx][sub], self.Y[idx][sub])
                mask[idx] |= hit
                i += 1
            yield m, mask, (g.isub & (~g.bsub | mask)).mean(axis=-1)


def score(scratch: Path, mesh_token: str, pops: list[str], ids: list[str]) -> None:
    """Every network with prefix rows, on `mesh_token`, for each of `pops` (count: (b) on the
    count prefixes; area: (c) on the area prefixes, and the stored clearings rescored); one block
    at a time, one scorer per population on one grid."""
    _check_single_threaded()
    import clear
    import geopandas as gpd
    import lifted
    import relax

    from reblock.budget import road_corridor
    mesh = common.mesh_of(mesh_token)
    if "area" not in pops:
        raise SystemExit("score builds the area scorer (the clearings' check) on every mesh")
    p = lifted.Params(ell_m=ELL_M, K=K, along=lifted.along_of(ALONG, lifted.scans_of(DEVICE)),
                      solver=lifted.solver_of(DEVICE))
    for pop in pops:
        score_dir(mesh_token, pop).mkdir(parents=True, exist_ok=True)
    for bid in ids:
        todo = [name for name in common.LINEUP
                if (prefix_dir() / f"{bid}__{name}.parquet").exists()
                and any(not (score_dir(mesh_token, pop) / f"{bid}__{name}.parquet").exists()
                        for pop in pops)]
        clear_out = score_dir(mesh_token, "area") / f"{bid}__clearings.parquet"
        if not todo and clear_out.exists():
            continue
        [b] = common.build_blocks([bid])
        print(f"{time.strftime('%H:%M:%S')} {bid} n={len(b.buildings)} on {mesh.name}: "
              f"{len(todo)} networks", flush=True)
        with GpuWatch(every=1.0) as gw:
            t0 = time.perf_counter()
            c = clear.Clearing(b, mesh, p, population=common.POPULATIONS["area"])
            t_area = time.perf_counter() - t0
            scorers = {"area": c.sc}
            if "count" in pops:
                scorers["count"] = common.Scorer.on_grid(b, c.sc.grid, p,
                                                         common.CountPopulation())
            t_setup = time.perf_counter() - t0
            print(f"  scorers in {t_setup:.0f}s (grid, area scorer and clearing {t_area:.0f}s), "
                  f"{int(c.sc.grid.inside.sum())} cells", flush=True)
            if not clear_out.exists():
                rel = relax.Relaxation(c, CLEARING_POWER)
                rows = []
                for cname, (rel_dir, perm_col, cleared_col, D_col, stored_mesh) in STORED.items():
                    r = pd.read_parquet(ROADLESS / rel_dir / f"{bid}_p{CLEARING_POWER:g}.parquet"
                                        ).iloc[0]
                    x = np.zeros(c.n)
                    x[np.asarray(r[cleared_col], dtype=np.int64)] = 1.0
                    ts = time.perf_counter()
                    J, P = rel.exact_JP(x)
                    gw.note_pool()
                    rows.append(dict(block=bid, mesh=mesh.name, clearing=cname,
                                     stored_mesh=common.mesh_of(stored_mesh).name,
                                     n_cleared=int(x.sum()), D=float(c.cost @ x),
                                     stored_D=float(r[D_col]), stored_perm=float(r[perm_col]),
                                     perm=rel.perm(J), perm1=1.0 - P / c.sc.P0,
                                     t=time.perf_counter() - ts))
                    print(f"  {cname}: rescored {rows[-1]['perm']:.7f}, stored "
                          f"{rows[-1]['stored_perm']:.7f} (on {rows[-1]['stored_mesh']})",
                          flush=True)
                pd.DataFrame(rows).to_parquet(clear_out)
            X, Y = c.sc.grid._sub_xy()
            cor = Corridors(c.sc.grid, X, Y)
            for name in todo:
                pre = pd.read_parquet(prefix_dir() / f"{bid}__{name}.parquet")
                ordered = gpd.read_parquet(ordered_path(scratch, bid, name))
                need = {pop: pre[pre["pop"] == pop] for pop in pops}
                stops = sorted({int(m) for pop in pops for m in need[pop]["m"]})
                got: dict[tuple[str, int], dict] = {}
                tn = time.perf_counter()
                t_masks = 0.0
                tm = time.perf_counter()
                for m, mask, free in cor.masks(ordered, stops):
                    t_masks += time.perf_counter() - tm
                    if m == len(ordered):                # the whole network: Carve's own mask
                        tc = time.perf_counter()
                        ref = c.sc.grid.sub_of(road_corridor(ordered))
                        if not np.array_equal(ref, mask):
                            raise AssertionError(f"{bid} {name}: road-by-road corridor differs "
                                                 f"from Carve's on {int((ref != mask).sum())} "
                                                 "sub-samples")
                        t_check = time.perf_counter() - tc
                    for pop in pops:
                        if m not in set(need[pop]["m"].astype(int)):
                            continue
                        sc = scorers[pop]
                        ts = time.perf_counter()
                        sol = lifted.solve(sc.grid, free, sc.f, p, rtol=SCORE_RTOL)
                        gw.note_pool()
                        u = sc.home_u_of(sol, free)
                        got[(pop, m)] = dict(
                            P1=1.0 - sol.P / sc.P0, P2=sc.perm_p(u, 2.0),
                            u95=float(np.nanpercentile(u, 95) / np.nanpercentile(sc.u0, 95)),
                            umax=float(np.nanmax(u) / np.nanmax(sc.u0)),
                            umed=float(np.nanmedian(u) / np.nanmedian(sc.u0)),
                            unknowns=sol.n_unknowns, t_solve=time.perf_counter() - ts)
                    tm = time.perf_counter()
                for pop in pops:
                    rows = [dict(r.to_dict(), mesh=mesh.name, **got[(pop, int(r["m"]))])
                            for _, r in need[pop].iterrows()]
                    df = pd.DataFrame(rows)
                    df["t_masks"], df["t_check"] = t_masks, t_check
                    df["t_network"] = time.perf_counter() - tn
                    df.to_parquet(score_dir(mesh_token, pop) / f"{bid}__{name}.parquet")
                lensA = {pop: got[(pop, int(need[pop][need[pop].budget == BUDGETS[1]].m.iloc[0]))]
                         for pop in pops}
                print(f"  {name}: {len(ordered)} roads, {len(stops)} prefixes, masks "
                      f"{t_masks:.0f}s, check {t_check:.0f}s, total "
                      f"{time.perf_counter() - tn:.0f}s; at D {BUDGETS[1]:g} "
                      + ", ".join(f"{pop} P1 {v['P1']:.4f} P2 {v['P2']:.4f}"
                                  for pop, v in lensA.items()), flush=True)
            del X, Y, cor
        for pop in pops:
            sc = scorers[pop]
            pd.DataFrame([dict(block=bid, n=len(b.buildings), mesh=mesh.name, pop=pop,
                               cells=int(c.sc.grid.inside.sum()), P0=sc.P0,
                               J2_0=sc.J(sc.u0, 2.0), stranded=sc.stranded,
                               t_setup=t_setup, t_block=time.perf_counter() - t0,
                               networks=",".join(todo),
                               gpu_peak_gb=gw.peak_mib / 1024, pool_peak_gb=gw.pool_peak_gb,
                               host_hwm_gb=_status_gb("VmHWM"))]).to_parquet(
                score_dir(mesh_token, pop) / f"{bid}__setup__{int(time.time())}.parquet")
        print(f"{time.strftime('%H:%M:%S')} {bid} done in {time.perf_counter() - t0:.0f}s, GPU "
              f"peak {gw.peak_mib / 1024:.1f} GB (pool {gw.pool_peak_gb:.1f} GB)", flush=True)
        del c, scorers
        p.solver.release()


# ------------------------------------------------------------------ report

SHORT = {"clearance": "clear", "clearance_looped": "clear_loop", "cycle_native": "cycle",
         "cycle_native_betweenness_contrast": "cycle_desire", "resistance_lp": "resist_lp",
         "euclidean_grid": "grid", "greedy_arterial_access_displacement": "arterial"}
REPORT_BUDGETS = (0.05, 0.10)     # the table's budgets (Lens A now and Lens A of the 220 study)
PRICE_BUDGET = 0.05               # the stored clearings' budget


def _read_all(paths: list[Path]) -> pd.DataFrame:
    return (pd.concat([pd.read_parquet(f) for f in paths], ignore_index=True) if paths
            else pd.DataFrame())


def proposals_table() -> pd.DataFrame:
    """Per (block, method): the network (the cached run's) and what computing it cost (the fresh
    run's where the cache hit)."""
    d = _read_all(sorted((ROWS / "proposals").glob("*.parquet")))
    # propose-all's row for a run it stopped carries no cache_hit; a stopped cache-mode run was a
    # miss (a hit returns in milliseconds)
    stopped = d.status.str.startswith("timeout")
    d.loc[stopped & ~d.fresh.astype(bool), "cache_hit"] = False
    d["area_km2"] = d.groupby("block").area_km2.transform("max")
    cache = d[~d.fresh.astype(bool)].drop(columns=["fresh"])
    fresh = d[d.fresh.astype(bool)][["block", "method", "status", "wall_s", "cpu_s",
                                     "peak_rss_gb", "same_as_cached"]]
    out = cache.merge(fresh, on=["block", "method"], how="left", suffixes=("", "_fresh"))
    hit = out.cache_hit.astype(bool)
    for col in ("status", "wall_s", "cpu_s", "peak_rss_gb"):
        out[f"cost_{col}"] = np.where(hit, out[f"{col}_fresh"], out[col])
    out["cost_status"] = out.cost_status.where(out.cost_status.notna(), "no fresh run yet")
    return out


def score_rows(mesh_token: str, pop: str) -> pd.DataFrame:
    d = score_dir(mesh_token, pop)
    return _read_all([f for name in common.LINEUP for f in sorted(d.glob(f"*__{name}.parquet"))])


def _tau(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import kendalltau
    return float(kendalltau(x, y).statistic) if len(x) >= 3 else np.nan


def lens_table(mesh_token: str, budget: float) -> pd.DataFrame:
    """Per (block, method) at `budget`: (a) and (b) on the count prefix, (c) on the area prefix,
    each prefix's road length and displacement, and whether it reaches the budget."""
    keys = ["block", "method"]
    cnt = score_rows(mesh_token, "count")
    cnt = cnt[np.isclose(cnt.budget, budget)][keys + ["n", "m", "road_m", "D_count", "D_area",
                                                      "P_old", "P1", "P2"]]
    cnt = cnt.rename(columns={"m": "m_count", "road_m": "road_m_count", "D_count": "Dc_count",
                              "D_area": "Da_count", "P_old": "a", "P1": "b", "P2": "b_P2"})
    area = score_rows(mesh_token, "area")
    area = area[np.isclose(area.budget, budget)][keys + ["m", "road_m", "D_count", "D_area",
                                                         "P_old", "P1", "P2"]]
    area = area.rename(columns={"m": "m_area", "road_m": "road_m_area", "D_count": "Dc_area",
                                "D_area": "Da_area", "P_old": "a_on_area", "P1": "c_P1",
                                "P2": "c"})
    t = cnt.merge(area, on=keys, how="outer")
    t["reach_count"] = t.Dc_count >= budget - 1e-9
    t["reach_area"] = t.Da_area >= budget - 1e-9
    for col, reach in (("a", "reach_count"), ("b", "reach_count"), ("c", "reach_area")):
        t[f"rank_{col}"] = (t[t[reach]].groupby("block")[col].rank(ascending=False, method="min")
                            .reindex(t.index))
    return t


def report(mesh_token: str, composite_token: str) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    mesh, comp = common.mesh_of(mesh_token), common.mesh_of(composite_token)
    # feasibility
    pt = proposals_table()
    print("FEASIBILITY (the cost: the fresh run where the derive cache hit; single-threaded, "
          "on a shared machine)")
    for bid, g in pt.sort_values(["n", "method"]).groupby("block", sort=False):
        print(f"  {bid} n={int(g.n.iloc[0])} {g.area_km2.iloc[0]:.3f} km^2")
        for _, r in g.iterrows():
            same = ("" if not bool(r.cache_hit) else
                    f"  [cache hit in {r.wall_s:.2f}s; fresh roads identical: "
                    f"{bool(r.same_as_cached_fresh == 1.0)}]")
            net = (f"{int(r.n_roads):5d} roads {r.road_m / 1000:7.2f} km" if r.status == "ok"
                   else "no network")
            print(f"    {SHORT[r.method]:13s} {r.cost_status[:28]:28s} wall {r.cost_wall_s:7.0f}s  "
                  f"CPU {r.cost_cpu_s:7.0f}s  peak RSS {r.cost_peak_rss_gb:5.2f} GB  {net}{same}")
    # the scorer against the stored clearings
    print(f"\nSCORER CHECK: the stored clearings rescored on {mesh.name} and {comp.name} "
          "(J_2, area population, rtol 1e-5) against their stored Lens A")
    cl = _read_all(sorted(score_dir(mesh_token, "area").glob("*__clearings.parquet"))
                   + sorted(score_dir(composite_token, "area").glob("*__clearings.parquet")))
    own = cl[cl.mesh == cl.stored_mesh]
    for _, r in own.iterrows():
        print(f"  {r.block:18s} {r.clearing:13s} on {r.mesh:9s} stored {r.stored_perm:.7f} "
              f"rescored {r.perm:.7f}  diff {r.perm - r.stored_perm:+.1e}  D {r.D:.5f} (stored "
              f"{r.stored_D:.5f})")
    # the table
    for budget in REPORT_BUDGETS:
        t = lens_table(mesh_token, budget)
        print(f"\nAT D {budget:g} on {mesh.name}: (a) shipped permeability and (b) roadless count "
              "p 1 on the count prefix, (c) roadless area J_2 on the area prefix; [rank of the "
              "methods reaching the budget]; * = the whole network falls short")
        for bid, g in t.sort_values(["n", "method"]).groupby("block", sort=False):
            print(f"  {bid} n={int(g.n.iloc[0])}")
            print(f"    {'method':13s} {'(a)':>12s} {'(b)':>12s} {'(c)':>12s}   road km count | "
                  "area   D count prefix | area prefix")
            for _, r in g.sort_values("c", ascending=False).iterrows():
                def cell(v, rk, ok):
                    return f"{v:.4f}{'*' if not ok else ''} [{'-' if np.isnan(rk) else int(rk)}]"
                print(f"    {SHORT[r.method]:13s} {cell(r.a, r.rank_a, r.reach_count):>12s} "
                      f"{cell(r.b, r.rank_b, r.reach_count):>12s} "
                      f"{cell(r.c, r.rank_c, r.reach_area):>12s}   {r.road_m_count / 1000:6.2f} | "
                      f"{r.road_m_area / 1000:6.2f}   {r.Dc_count:.4f} | {r.Da_area:.4f}")
            both = g[g.reach_count & g.reach_area]
            print(f"    Kendall tau over the {len(both)} methods reaching both: (a,b) "
                  f"{_tau(both.a.to_numpy(), both.b.to_numpy()):+.3f}  (a,c) "
                  f"{_tau(both.a.to_numpy(), both.c.to_numpy()):+.3f}  (b,c) "
                  f"{_tau(both.b.to_numpy(), both.c.to_numpy()):+.3f}")
    # the price of roads
    print(f"\nTHE PRICE OF ROADS at D {PRICE_BUDGET:g} under (c) on {mesh.name} (a lower bound: "
          "the "
          "optimizers are heuristics; the road prefix is the first reaching the budget, the "
          "clearings stay within it)")
    t = lens_table(mesh_token, PRICE_BUDGET)
    cl_m = cl[cl.mesh == mesh.name]
    for bid, g in t.sort_values(["n"]).groupby("block", sort=False):
        ok = g[g.reach_area]
        best = ok.loc[ok.c.idxmax()]
        line = (f"  {bid:18s} best road {SHORT[best.method]:12s} {best.c:.4f} "
                f"(D {best.Da_area:.4f},"
                f" {best.road_m_area / 1000:.2f} km)")
        for cname in ("simp_default", "cheap"):
            r = cl_m[(cl_m.block == bid) & (cl_m.clearing == cname)].iloc[0]
            line += f"; {cname} {r.perm:.4f} (D {r.D:.4f}) minus best {r.perm - best.c:+.4f}"
        print(line)
    # the composite mesh
    print(f"\nCOMPOSITE MESH: (c) on {comp.name} minus on {mesh.name}, same prefixes")
    ca = score_rows(composite_token, "area")
    if len(ca):
        ua = score_rows(mesh_token, "area")
        keys = ["block", "method", "pop", "budget", "full", "m"]
        mg = ca.merge(ua, on=keys, suffixes=("_comp", "_uni"))
        mg["dP2"], mg["dP1"] = mg.P2_comp - mg.P2_uni, mg.P1_comp - mg.P1_uni
        for bid, g in mg.groupby("block"):
            j = g.dP2.abs().idxmax()
            lensA = g[np.isclose(g.budget, PRICE_BUDGET)]
            print(f"  {bid}: {len(g)} prefixes, J_2 diff median {g.dP2.median():+.5f}, largest "
                  f"|diff| {g.dP2.abs().max():.5f} ({SHORT[g.method[j]]} at "
                  f"{'full' if g.full[j] else g.budget[j]}); at D {PRICE_BUDGET:g} "
                  f"{lensA.dP2.min():+.5f} .. {lensA.dP2.max():+.5f}; P diff largest "
                  f"{g.dP1.abs().max():.5f}")
            o = mg[mg.block == bid]
            rk = [_tau(o[np.isclose(o.budget, bud)].P2_comp.to_numpy(),
                       o[np.isclose(o.budget, bud)].P2_uni.to_numpy()) for bud in BUDGETS]
            print("    ranking of the methods, composite vs uniform, Kendall tau per budget: "
                  + ", ".join(f"{b:g} {x:+.2f}" for b, x in zip(BUDGETS, rk, strict=True)))
        cc = cl.pivot_table(index=["block", "clearing"], columns="mesh", values="perm")
        if comp.name in cc and mesh.name in cc:
            cc = cc.dropna(subset=[comp.name, mesh.name])
            for (bid, cname), r in cc.iterrows():
                print(f"  clearing {bid} {cname:13s}: {comp.name} {r[comp.name]:.5f} - "
                      f"{mesh.name} {r[mesh.name]:.5f} = {r[comp.name] - r[mesh.name]:+.5f}")


# ------------------------------------------------------------------ cost projection

def _fit_power(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """log y = a + b log x by least squares: (a, b)."""
    b, a = np.polyfit(np.log(x), np.log(y), 1)
    return float(a), float(b)


def full_run_blocks() -> pd.DataFrame:
    """The full run's blocks: the 82 large ones and whatever else has a stored SIMP default row,
    each with its building count, area and the mesh its SIMP default used."""
    big = pd.read_parquet(ROADLESS / "large82_area.parquet")
    simp = STORED["simp_default"][0]
    rows = []
    for d in sorted((ROADLESS / simp).parent.parent.glob(Path(simp).parent.name + "*")):
        tok = d.name[len(Path(simp).parent.name):]
        if tok and not re.fullmatch(r"\.a[0-9.]+x[0-9.]+", tok):    # the default plan on a
            continue                                                 # composite only
        for f in sorted((d / "D0.05").glob("*_p2.parquet")):
            rows.append(dict(block=f.name.rsplit("_p", 1)[0], simp_mesh=tok or ".h0.5",
                             n_simp=int(pd.read_parquet(f, columns=["n"]).n.iloc[0])))
    s = pd.DataFrame(rows)
    return s.merge(big, on="block", how="left")


SIZES_MIN_M = 40.0      # sizes: blocks with more unknowns than this at h 0.5 (millions; and any
                        # block outside large82) get their composite cell count, the candidates
                        # for not fitting a 48 GB card
_SIZE_MESH: list = []


def _cells(bid: str) -> tuple[str, float, float]:
    [b] = common.build_blocks([bid])
    return bid, b.boundary.area / 1e6, float(_SIZE_MESH[0].cells(b))


def sizes_path(composite_token: str) -> Path:
    return ROWS / f"sizes_{common.mesh_of(composite_token).name}.parquet"


def sizes(composite_token: str, procs: int) -> None:
    """The composite mesh's cell count (and the area) of every full-run block that may not fit
    a 48 GB card at h 0.5: the projection's input for the oversized blocks."""
    _check_single_threaded()
    import multiprocessing
    blocks = full_run_blocks()
    todo = sorted(blocks[~(blocks.M_unknowns <= SIZES_MIN_M)].block)
    _SIZE_MESH.append(common.mesh_of(composite_token))
    t0 = time.perf_counter()
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=1) as pool:
        got = list(pool.imap_unordered(_cells, todo))
    df = pd.DataFrame(got, columns=["block", "area_km2", "cells_comp"])
    df["mesh"] = _SIZE_MESH[0].name
    df.to_parquet(sizes_path(composite_token))
    print(f"{len(df)} blocks sized in {time.perf_counter() - t0:.0f}s", flush=True)


def per_block_scoring(mesh_token: str) -> pd.DataFrame:
    """Per block on `mesh_token`: what scoring all its networks cost on the GPU, over however
    many passes it took: one setup (the median of the passes'), every network's own time (count
    and area together) and the clearings', the peak memory of any pass."""
    d = score_dir(mesh_token, "area")
    st = _read_all(sorted(d.glob("*__setup__*.parquet")))
    per = st.groupby("block").agg(t_setup=("t_setup", "median"), passes=("t_setup", "size"),
                                  gpu_peak_gb=("gpu_peak_gb", "max"),
                                  pool_peak_gb=("pool_peak_gb", "max"),
                                  host_hwm_gb=("host_hwm_gb", "max"))
    sr = score_rows(mesh_token, "area")
    net = sr.groupby(["block", "method"]).t_network.first().groupby("block")
    per["t_networks"], per["networks"] = net.sum(), net.size()
    cl = _read_all(sorted(d.glob("*__clearings.parquet")))
    per["t_clearings"] = cl.groupby("block").t.sum()
    per["t_full"] = per.t_setup + per.t_networks + per.t_clearings
    # all of the lineup's networks at this block's mean cost per network (blocks still missing
    # some), one setup, the clearings
    per["t_lineup"] = (per.t_setup + len(common.LINEUP) * per.t_networks / per.networks
                       + per.t_clearings)
    big = pd.read_parquet(ROADLESS / "large82_area.parquet")
    return per.reset_index().merge(big[["block", "M_unknowns"]], on="block")


def project(mesh_token: str, composite_token: str) -> None:
    """The full run's cost from the sentinel's: proposals (CPU-hours per method, a power law in
    the building count fitted on the four blocks), scoring (GPU-hours, a power law in unknowns),
    and which blocks a 48 GB card cannot hold at `mesh_token` (they take `composite_token`)."""
    pt = proposals_table()
    blocks = full_run_blocks()
    print(f"{len(blocks)} blocks in the full run ({int(blocks.n.notna().sum())} of the large 82; "
          f"others: {sorted(blocks[blocks.n.isna()].block)})")
    print("\nPROPOSALS: CPU seconds per (block, method) measured on the sentinel; the full run "
          "EXTRAPOLATED as the median and the slowest sentinel block's time for every block (a "
          "stopped run counts at its CPU time when stopped: a floor). The sentinel spans n 1,721 "
          "-- 6,619 and 0.39 -- 1.02 km^2; the oversized blocks are 1.4 -- 52 km^2 with 1,000 -- "
          "4,500 buildings, outside it.")
    n_all = blocks.n.fillna(blocks.n_simp).to_numpy(dtype=float)
    lo_tot = hi_tot = 0.0
    proj_rows = []
    for name in common.LINEUP:
        g_all = pt[pt.method == name]
        g = g_all[g_all.cost_status == "ok"]
        cut = g_all[g_all.cost_status.str.startswith("timeout")]
        meas = ", ".join([f"{int(n)}: {c:.0f}s" for n, c in zip(g.n, g.cost_cpu_s, strict=True)]
                         + [f"{int(n)}: >{c:.0f}s ({st})" for n, c, st
                            in zip(cut.n, cut.cost_cpu_s, cut.cost_status, strict=True)])
        med = float(g_all.cost_cpu_s.median()) * len(n_all)
        hi = float(g_all.cost_cpu_s.max()) * len(n_all)
        lo_tot, hi_tot = lo_tot + med, hi_tot + hi
        if len(g) >= 3:
            a, bexp = _fit_power(g.n.to_numpy(float),
                                 np.maximum(g.cost_cpu_s.to_numpy(float), 0.05))
            fit = (f"; CPU ~ n^{bexp:.2f} on the {len(g)} completed blocks"
                   + (f" ({np.exp(a) * (n_all ** bexp).sum() / 3600:.1f} CPU-h by that law)"
                      if bexp > 0 else " (falls with n: no size law)"))
        else:
            bexp, fit = np.nan, f"; {len(g)} completed blocks: no size law"
        floor = " (floors)" if len(cut) else ""
        proj_rows.append(dict(method=name, exponent_n=bexp, cpu_h_median=med / 3600,
                              cpu_h_max=hi / 3600, stopped=len(cut)))
        print(f"  {SHORT[name]:13s} {meas}{fit}; full run {med / 3600:.1f} -- {hi / 3600:.1f} "
              f"CPU-h{floor}")
    print(f"  all methods: {lo_tot / 3600:.0f} -- {hi_tot / 3600:.0f} CPU-h (the arterial's a "
          "floor), single-threaded processes, measured under a load of 40 -- 110 on 48 cores")
    # scoring
    print(f"\nSCORING on the GPU (lenses (b) and (c), 7 networks, the clearings), per block: "
          f"measured on the sentinel at {common.mesh_of(mesh_token).name}")
    st = per_block_scoring(mesh_token)
    for _, r in st.iterrows():
        print(f"  {r.block}: {r.M_unknowns:.1f}M unknowns (cells x {K}), {r.t_full:.0f}s for "
              f"{r.networks} networks (setup {r.t_setup:.0f}s, networks {r.t_networks:.0f}s, "
              f"clearings {r.t_clearings:.0f}s; {r.passes} passes), GPU peak "
              f"{r.gpu_peak_gb:.1f} GB "
              f"(cupy pool {r.pool_peak_gb:.1f} GB), host {r.host_hwm_gb:.1f} GB")
    rate = st.gpu_peak_gb / st.M_unknowns
    gb_per_m = float(rate[st.M_unknowns.idxmax()])     # the largest sentinel block's rate
    gb_per_m_max = float(rate.max())
    sec = st.t_lineup / st.M_unknowns                   # GPU seconds per million unknowns
    s_mid, s_lo, s_hi = float(sec.median()), float(sec.min()), float(sec.max())
    print("  the whole lineup per block / its unknowns: "
          + ", ".join(f"{b.split('_')[-1]} {t:.0f}s / {m:.1f}M = {x:.1f} s/M"
                      for b, t, m, x in zip(st.block, st.t_lineup, st.M_unknowns, sec,
                                            strict=True))
          + f"; GPU memory per million unknowns {', '.join(f'{x:.3f}' for x in rate)} GB: sized "
          f"at {gb_per_m:.3f} (the largest block's), {gb_per_m_max:.3f} at worst")
    cap = 48.0
    blocks["M"] = blocks.M_unknowns
    blocks["gb_uniform"] = blocks.M * gb_per_m
    risky = blocks[(blocks.M * gb_per_m < cap) & ~(blocks.M * gb_per_m_max < cap)]
    print("  at the worst rate these would not fit either: "
          + ", ".join(f"{r.block} ({r.M:.1f}M, ~{r.M * gb_per_m_max:.0f} GB)"
                      for _, r in risky.iterrows()))
    over = blocks[~(blocks.gb_uniform < cap)].copy()
    comp_mesh = common.mesh_of(composite_token)
    got = pd.read_parquet(sizes_path(composite_token)).set_index("block")
    missing = sorted(set(over.block) - set(got.index))
    if missing:
        raise SystemExit(f"run `sizes` with a lower SIZES_MIN_M: {missing} are not sized")
    over["area_km2"] = got.area_km2.reindex(over.block).to_numpy()
    over["M_comp"] = got.cells_comp.reindex(over.block).to_numpy() * K / 1e6
    blocks = blocks.merge(over[["block", "M_comp"]], on="block", how="left")
    blocks["area_km2"] = blocks.area_km2.fillna(blocks.block.map(
        dict(zip(over.block, over.area_km2, strict=True))))
    h = common.mesh_of(mesh_token).h                  # a block outside large82 (a region)
    blocks["M"] = blocks.M.fillna(blocks.area_km2 * 1e6 / h ** 2 * K / 1e6)
    blocks["gb_uniform"] = blocks.M * gb_per_m
    fits = blocks.gb_uniform < cap
    blocks["M_run"] = np.where(fits, blocks.M, blocks.M_comp)
    blocks["t_pred"] = s_mid * blocks.M_run
    print(f"  {int(fits.sum())} blocks fit {cap:g} GB at {common.mesh_of(mesh_token).name} "
          f"(estimate <= {blocks[fits].gb_uniform.max():.1f} GB); {int((~fits).sum())} do not "
          f"and take {comp_mesh.name} ({blocks[~fits].M_comp.min():.1f} -- "
          f"{blocks[~fits].M_comp.max():.1f}M unknowns there, <= "
          f"{(blocks[~fits].M_comp * gb_per_m).max():.1f} GB):")
    for _, r in blocks[~fits].sort_values("M").iterrows():
        print(f"    {r.block:26s} {r.area_km2:6.2f} km^2  {r.M:7.1f}M at h 0.5 "
              f"(~{r.gb_uniform:.0f} GB), {r.M_comp:5.1f}M on the composite")
    msum = float(blocks.M_run.sum())
    print(f"  GPU time EXTRAPOLATED (unknowns x the median {s_mid:.1f} s/M; the composite's cells "
          f"assumed to cost as the uniform grid's): {blocks.t_pred.sum() / 3600:.1f} GPU-h in all "
          f"({blocks[fits].t_pred.sum() / 3600:.1f} at h 0.5 over "
          f"{float(blocks[fits].M_run.sum()):.0f}M "
          f"unknowns, {blocks[~fits].t_pred.sum() / 3600:.1f} on the composite over "
          f"{float(blocks[~fits].M_run.sum()):.0f}M), {msum * s_lo / 3600:.1f} -- "
          f"{msum * s_hi / 3600:.1f} h at the sentinel's fastest and slowest rates; the largest "
          f"block {blocks.t_pred.max() / 3600:.2f} h")
    out = ROWS / f"projection_{common.mesh_of(mesh_token).name}_{comp_mesh.name}.parquet"
    blocks.to_parquet(out)
    pd.DataFrame(proj_rows).to_parquet(out.with_name("projection_proposals.parquet"))
    print(f"  rows: {out}")


# ------------------------------------------------------------------ main

if __name__ == "__main__":
    a = sys.argv[1:]
    cmd = a[0]
    if cmd == "propose":
        if a[4] not in ("cache", "fresh"):
            raise SystemExit(f"propose takes cache or fresh, not {a[4]!r}")
        propose(Path(a[1]), a[2], a[3], a[4] == "fresh")
    elif cmd == "propose-all":
        if a[6] not in ("time-hits", "no"):
            raise SystemExit(f"propose-all takes time-hits or no, not {a[6]!r}")
        propose_all(Path(a[1]), int(a[2]), a[3].split(","), a[4].split(","),
                    float(a[5]) * 3600, a[6] == "time-hits")
    elif cmd == "prefixes":
        prefixes(Path(a[1]), int(a[2]), a[3].split(","))
    elif cmd == "score":
        score(Path(a[1]), a[2], a[3].split(","), a[4].split(","))
    elif cmd == "report":
        report(a[1], a[2])
    elif cmd == "sizes":
        sizes(a[1], int(a[2]))
    elif cmd == "project":
        project(a[1], a[2])
    else:
        raise SystemExit(f"unknown mode {cmd!r}")
