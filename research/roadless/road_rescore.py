"""The road lineup on the large blocks, on the gap complex, set against the presets' clearings
(BACKLOG 4(b), "the price of roads"): the 60 large blocks.

Each method's network (reblock.derivations.propose, the derive cache), cut to its canonical
prefixes (common.prefix_to: the roads it holds first, if it marks any, then street-first) at
BUDGETS and whole, under both weightings (count and area: POPS), each prefix's displacement and
the shipped permeability (reblock.permeability) recorded; then the area prefixes scored on the
complex (decision (2)).

    S=research/roadless/road_rescore.py    # from the repo root, PYTHONPATH=., single-threaded
    uv run python $S propose <rows> <scratch> <id> <method> <cache|fresh>
    uv run python $S propose-all <rows> <scratch> <procs> <ids,> <methods,> <timeout h> \
        <time-hits|no>
    uv run python $S prefixes <rows> <lineup,> <scratch> <procs> <ids,>
    uv run python $S cx <rows> <scratch> <metric> <cpu|gpu> <id> <items,> [--replace]
    uv run python $S cxreport <rows> <lineup,> <metric> <cpu|gpu>
    uv run python $S lineup-one <scratch> <id> <method> [--replace]   # propose, cut, score
    uv run python $S lineup-score <scratch> <id> <methods,> [--replace]   # score only

ON THE COMPLEX (phase 2b; NOTES, "The complex as the runtime: the ports and the cross-checks"):
`cx` scores the area prefixes on the gap complex's metric (<metric>, refused unless it is one:
gap_clear.complex_of), each item of a block in turn: a network of <rows>' prefixes (cx_network: the
network's fixed mesh with its prefixes' corridors cut into the footprints, gap_clear.Carving, built
in a temporary directory; every prefix a state on it, L_P, L_{P_1/2}, L_J2 and B (i)), or `plain`
(cx_plain: the block's plain mesh, its homes' P_{1/2} weights, the stored clearings of cx_stored,
then SIMP's default and the cheap preset under P at D 0.05, their polish scored as the device's).
Rows under <rows>/<metric>/<device>/area_<budgets>/; `cxreport` reads them (B's order on L_P,
L_{P_1/2} and L_J2; the price of roads under the complex) and writes
<rows>/<metric>/summary_<device>_*.csv. (Item 4's scoring on the raster, `score`, `report`, `sizes`
and `project`, went with the raster: road_rows/summary_h0.5_*.csv its record.)

<rows> is the rows root and <lineup,> the networks it scores, both required: item 4 ran with
research/roadless/road_rows and common.LINEUP (clearance,clearance_looped,cycle_native,
cycle_native_betweenness_contrast,resistance_lp,euclidean_grid,
greedy_arterial_access_displacement). propose and propose-all take the method by its name in
compare_config's all_methods (common.arms()); propose() itself takes the method, resolved by its
caller.

<scratch>: networks, ordered networks, logs (bulky, not committed). propose `cache` goes through
the derive cache (and says whether it hit); `fresh` calls the method directly, bypassing the
cache (a refiner's base too), to time a cached proposal (propose-all queues one after every cache
hit) and to check the cached roads against a recomputation. Rows: <rows>/proposals/
(feasibility), <rows>/prefixes_<budgets>/ (the prefixes, their displacement under both
weightings, and the shipped permeability). Every process single-threaded.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import common  # noqa: E402
import gap_clear as gcl  # noqa: E402
import gap_complex as gc  # noqa: E402
import gap_factor  # noqa: E402
import polish_greedy  # noqa: E402
import relax  # noqa: E402
import search  # noqa: E402

from reblock.budget import road_corridor  # noqa: E402

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

    from reblock.contracts import Block, Method

ROADLESS = common.REPO / "research" / "roadless"

# ------------------------------------------------------------------ the run's parameters
# Everything that can change a number is here or on the command line, and in the rows' paths.
BUDGETS = (0.02, 0.05, 0.10, 0.15, 0.20)     # displacement budgets of the canonical prefixes
SHORT_SLOTS = 3                              # propose-all: slots kept for the cheapest jobs
# propose-all's ORDER only (no result depends on it): a guess at each method's cost per building
COST_GUESS = {"cycle_native_betweenness_contrast": 3.0, "cycle_native": 2.5,
              "greedy_arterial_access_displacement": 2.0, "resistance_lp": 2.0,
              "clearance_looped": 1.0, "clearance": 0.7, "euclidean_grid": 0.2}
SINGLE = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
              NUMBA_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")


def budgets_tag() -> str:
    return "D" + "-".join(f"{b:g}" for b in BUDGETS)


def prefix_dir(rows: Path) -> Path:
    return rows / f"prefixes_{budgets_tag()}"


def net_path(scratch: Path, bid: str, name: str, fresh: bool) -> Path:
    return scratch / "networks" / bid / f"{name}{'.fresh' if fresh else ''}.parquet"


def ordered_path(scratch: Path, bid: str, name: str) -> Path:
    return scratch / "networks" / bid / f"{name}.ordered.parquet"


def proposal_row(rows: Path, bid: str, name: str, fresh: bool) -> Path:
    return rows / "proposals" / f"{bid}__{name}{'__fresh' if fresh else ''}.parquet"


def _check_single_threaded() -> None:
    bad = {k: os.environ.get(k) for k, v in SINGLE.items() if os.environ.get(k) != v}
    if bad:
        raise SystemExit(f"run single-threaded: set {bad} to {SINGLE}")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """`df` to `path` whole or not at all: written under a name no pull matches (cluster.py's
    pulls take `*.parquet`, while a task runs too, and never replace a local file), then renamed.
    A GeoDataFrame writes its own way (its to_parquet)."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp)
    os.replace(tmp, path)


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


# The research lineup (the consolidated pass; BACKLOG 5's TODO, owner 2026-10-12 "leave a TODO"):
# every method that holds roads first holds free_roads.lineup_free() (FreeEnclosed under RUNTIME)
# in place of HoldNothing (clearance, clearance_looped's tree, cycle_native, cycle_desire,
# resistance_lp; the grid and the arterial hold no stage). The arterial runs as its preset says:
# its budget is the preset's ScoredArea(share=0.20) (this lineup's own area D), its max_roads a
# safety cap (the lineup once lifted the preset's 60 to 300 so it reached the Lens A budget: at 60
# it reached area D 0.011 -- 0.044 on the sentinel's 9712, 22422 and 5810; NOTES, "Re-scoring the
# road methods: the large-block sentinel"). Its rows' root carries the stage's and the arterial's
# identities (lineup_rows), so no row of another lineup, metric or arterial reads as its own.
ARTERIAL = "greedy_arterial_access_displacement"


def lineup_arms() -> dict[str, Method]:
    """The research lineup's methods (above), by name."""
    import free_roads
    from reblock.methods.clearance import ClearanceReblocker
    from reblock.methods.cycle_native import CycleNativeReblocker
    from reblock.methods.loop_closure import LoopClosureRefiner
    from reblock.methods.resistance_lp import ResistanceLPReblocker
    free = free_roads.lineup_free()
    out: dict[str, Method] = {}
    for name, m in _arms().items():
        if isinstance(m, LoopClosureRefiner):     # clearance_looped: the stage in its tree
            assert isinstance(m.base, ClearanceReblocker), name
            m = dataclasses.replace(m, base=dataclasses.replace(m.base, free_first=free))
        elif isinstance(m, CycleNativeReblocker | ResistanceLPReblocker | ClearanceReblocker):
            m = dataclasses.replace(m, free_first=free)
        out[name] = m
    return out


def lineup_rows() -> Path:
    """The research lineup's rows root: road_rows/lineup-<digest of its free-first stage's
    identity (the metric's name in it) and the arterial's whole identity>. The root changes
    whenever the arterial's identity does (its sampler, budget or cap): its networks change."""
    import hashlib

    import free_roads
    ident = repr((free_roads.lineup_free().identity, lineup_arms()[ARTERIAL].identity))
    return ROADLESS / "road_rows" / f"lineup-{hashlib.sha256(ident.encode()).hexdigest()[:8]}"


def _lineup_root(rows: Path) -> Path:
    """`rows` if it is the research lineup's root (lineup_rows), else a refusal: proposals and
    prefixes are made by the lineup's arms only, under its identity."""
    if rows.resolve() != lineup_rows().resolve():
        raise SystemExit(f"{rows}: the lineup's rows are {lineup_rows()} (its stage and arterial)")
    return rows


def _fresh(method, block):
    """`method` on `block` with no derive cache anywhere: a refiner (LoopClosureRefiner), which
    fetches its base's proposal through the cache when it is given none, is handed its base's
    fresh proposal as its prior."""
    from reblock.methods.loop_closure import LoopClosureRefiner
    if isinstance(method, LoopClosureRefiner):
        return method.propose(block, prior=method.base.propose(block))
    return method.propose(block)


def propose(rows: Path, scratch: Path, bid: str, name: str, method, fresh: bool) -> None:
    """One (block, network) proposal by `method` (resolved by the caller), timed; the roads to
    <scratch>/networks as `name`, a feasibility row to <rows>/proposals."""
    _check_single_threaded()
    from reblock import derivations
    from reblock.derive_graph import DerivationKey, _code_version, _l2, env_version
    t_all = time.perf_counter()
    [b] = common.build_blocks([bid])
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
            prop = _fresh(method, b) if fresh else derivations.propose(method, b)
        roads = prop.roads
        n_roads = 0 if roads is None else len(roads)
        road_m = 0.0 if roads is None else float(roads.geometry.length.sum())
        out = net_path(scratch, bid, name, fresh)
        out.parent.mkdir(parents=True, exist_ok=True)
        if roads is not None:
            write_parquet(roads, out)
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
    dest = proposal_row(rows, bid, name, fresh)
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_parquet(pd.DataFrame([row]), dest)


def propose_all(rows: Path, scratch: Path, procs: int, ids: list[str], methods: list[str],
                timeout_s: float, time_hits: bool) -> None:
    """Every (block, method) network without a row, for `methods` of compare_config's
    all_methods (common.arms()), `procs` processes at a time: the costliest first (COST_GUESS x
    buildings) on procs - SHORT_SLOTS slots, the cheapest first on the rest (so small blocks
    finish early and can be scored). With `time_hits` a cache hit queues a fresh run of the same
    pair (its cost). A job past `timeout_s` of wall time is killed (by PID) and recorded: a
    finding (the sentinel ran 2 h)."""
    _check_single_threaded()
    unknown = sorted(set(methods) - set(lineup_arms()))
    if unknown:
        raise SystemExit(f"not methods of the lineup (lineup_arms): {unknown}")
    sizes = pd.read_parquet(ROADLESS / "large82_area.parquet").set_index("block")["n"]
    snap = scratch / "road_rescore_snapshot.py"         # the children run this copy: edits to the
    shutil.copy(__file__, snap)                          # script mid-run cannot reach them
    env = {**os.environ, **SINGLE,
           "PYTHONPATH": os.pathsep.join([str(common.REPO), str(ROADLESS)])}
    (scratch / "logs").mkdir(parents=True, exist_ok=True)

    def cost(job: tuple[str, str, bool]) -> float:
        return float(sizes[job[0]]) * COST_GUESS[job[1]]

    queue = sorted(((bid, name, False) for bid in ids for name in methods
                    if not proposal_row(rows, bid, name, False).exists()), key=cost, reverse=True)
    for bid in ids if time_hits else []:                # fresh runs of hits already recorded
        for name in methods:
            p = proposal_row(rows, bid, name, False)
            if (p.exists() and bool(pd.read_parquet(p).cache_hit.iloc[0])
                    and not proposal_row(rows, bid, name, True).exists()):
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
            p = subprocess.Popen([sys.executable, str(snap), "propose", str(rows), str(scratch),
                                  bid, name, "fresh" if fresh else "cache"], stdout=fh,
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
                dest = proposal_row(rows, bid, name, fresh)
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
            row = proposal_row(rows, bid, name, fresh)
            if (time_hits and rc == 0 and not fresh and row.exists()
                    and bool(pd.read_parquet(row).cache_hit.iloc[0])
                    and not proposal_row(rows, bid, name, True).exists()):
                queue.append((bid, name, True))
                queue.sort(key=cost, reverse=True)
                print(f"  cache hit: queued a fresh run of {bid} {name}", flush=True)


# ------------------------------------------------------------------ prefixes (CPU)

POPS = ("count", "area")         # the two weightings the prefixes are cut by


def prefix_len(block, ordered, target: float, w: np.ndarray) -> int:
    """common.prefix_to's length on roads already in canonical order (common.canonical_order once
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


_BLOCKS: dict = {}       # the jobs' blocks, built once before the pool forks


class PrefixJob(NamedTuple):
    rows: Path
    scratch: Path
    bid: str
    name: str


def prefixes_one(job: PrefixJob) -> str | None:
    """One network's canonical prefixes (common.canonical_order: its held roads, if the network
    marks any, then street-first): per weighting and budget its length, and per distinct prefix
    its road length, displacement under both weightings and (a). The ordered network to
    <scratch>, the rows to prefix_dir(<rows>). The failed job's name, or None."""
    try:
        _prefixes_one(job)
    except Exception as e:
        traceback.print_exc()
        print(f"{job.bid} {job.name} prefixes FAILED {type(e).__name__}: {e}"[:300], flush=True)
        return f"{job.bid} {job.name}"
    return None


def _prefixes_one(job: PrefixJob) -> None:
    import geopandas as gpd

    from reblock.compare import load_permeability_config
    from reblock.permeability import EgressContext, permeability
    scratch, bid, name, b = job.scratch, job.bid, job.name, _BLOCKS[job.bid]
    t0 = time.perf_counter()
    roads = gpd.read_parquet(net_path(scratch, bid, name, False))
    ordered = common.canonical_order(b, roads)
    write_parquet(ordered, ordered_path(scratch, bid, name))
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
    prefix_rows = [dict(block=bid, n=len(polys), method=name, **r, **per_m[r["m"]]) for r in lens]
    out = prefix_dir(job.rows) / f"{bid}__{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(prefix_rows)
    df["t_order"], df["t_prefixes"], df["t_total"] = t_order, t_pre, time.perf_counter() - t0
    write_parquet(df, out)
    print(f"{time.strftime('%H:%M:%S')} {bid} {name}: {len(ordered)} roads, order {t_order:.1f}s, "
          f"prefixes {t_pre:.1f}s, total {time.perf_counter() - t0:.1f}s", flush=True)


def prefixes(rows: Path, lineup: tuple[str, ...], scratch: Path, procs: int,
             ids: list[str]) -> None:
    """prefixes_one for every network of `lineup` present without rows, `procs` processes."""
    _check_single_threaded()
    import multiprocessing
    jobs = [PrefixJob(rows, scratch, bid, name) for bid in ids for name in lineup
            if net_path(scratch, bid, name, False).exists()
            and not (prefix_dir(rows) / f"{bid}__{name}.parquet").exists()]
    if not jobs:
        print("nothing to do", flush=True)
        return
    _BLOCKS.update({b.block_id: b for b in common.build_blocks(sorted({j.bid for j in jobs}))})
    with multiprocessing.get_context("fork").Pool(procs, maxtasksperchild=4) as pool:
        failed = [f for f in pool.imap_unordered(prefixes_one, jobs) if f]
    if failed:
        raise SystemExit(f"{len(failed)} failed: {failed}")


# ------------------------------------------------------------------ score on the complex
# NOTES, "The complex as the runtime: the ports and the cross-checks". Each (block, network) is
# its own fixed mesh with the network's prefix corridors cut into the footprints
# (gap_clear.Carving), every area prefix a state on it; the block's plain mesh scores the stored
# clearings and runs the two presets, under P.

CX_POWER = 1.0                  # the presets' objective on the complex: P (decision (1)); the
                                # presets relax.SIMP_DEFAULT and polish_greedy.CHEAP (the
                                # translucent search since phase 3: its rows under the metric's
                                # @e0.05, its clearing `cheap_cx<E>@e0.05`), each polish scored as
                                # the device's (`cx` main): E on a core, the eps world on a GPU
                                # (phase 2a: the same clearings)
CX_POP = "area"                 # the prefixes cut by area population (decision (2))


class Stored(NamedTuple):
    """The stored clearings scored on the complex beside the roads (clearings.Store's run ids):
    SIMP's default and the cheap preset (its plain form, the metric's own tension) at area D 0.05
    as a complex metric optimized them (P: phase 1's acceptance, the 13: records). A row's
    `clearing` is its field's name. (The raster's own clearings, item 4's and phase 1's, were
    scored here too until its retirement: the 60's rows hold them, the cross-checks' record.)"""
    simp_cx: str
    cheap_cx: str


def cx_stored(metric: str) -> Stored:
    """The stored clearings' runs under the complex `metric`."""
    plan, c = relax.SIMP_DEFAULT, polish_greedy.CHEAP
    cheap = f"{c.picker}.P{c.tries}w{c.width}"
    return Stored(simp_cx=f"relax_rows/{metric}/{plan}/D0.05.p1",
                  cheap_cx=f"polish_rows/{metric}/{cheap}/D0.05.p1")


class Presets(NamedTuple):
    """The two presets' rows directories under one complex metric and polish scoring
    (relax.rows_of, polish_greedy.rows_of; a run's id in the store is its directory with
    `.p<power>`, store_run): SIMP's default and the cheap preset."""
    simp: Path
    cheap: Path


def cx_presets(metric: gcl.ComplexMetric, scoring: search.PolishScoring) -> Presets:
    """The presets' rows under the complex `metric`, their polish scored by `scoring`: SIMP's
    default under it, the cheap preset under its search of it (polish_greedy.CHEAP.on)."""
    plan = relax.plan_of(relax.SIMP_DEFAULT)._replace(polish_scoring=scoring)
    c = polish_greedy.CHEAP
    return Presets(simp=relax.rows_of(plan, metric.name, PRICE_BUDGET),
                   cheap=polish_greedy.rows_of(c.picker, c.tries, c.width, c.pairs, scoring,
                                               c.on(metric).name, PRICE_BUDGET))


def store_run(rows_dir: Path) -> str:
    """A rows directory's run id in the store (clearings_export: the path with `.p<power>`)."""
    return f"{rows_dir.relative_to(ROADLESS).as_posix()}.p{CX_POWER:g}"


def cx_dir(rows: Path, metric: gcl.ComplexMetric) -> Path:
    """The complex's rows: under the metric's name (its rows never a raster run's: the name is
    gap_clear.check_disjoint's), then the device (the same scores to the floating-point level:
    each run keeps its own)."""
    return rows / metric.name / metric.device / f"{CX_POP}_{budgets_tag()}"


def corridor_of(ordered: gpd.GeoDataFrame, m: int) -> BaseGeometry:
    """The corridor of an ordered network's first `m` roads (reblock.budget.road_corridor: each
    buffered by half its width, unioned), the prefix Carve frees."""
    return road_corridor(cast("gpd.GeoDataFrame", ordered.iloc[:m]))


def _lens_row(m: gcl.Model, L: gcl.Lenses) -> dict[str, float]:
    """A state's lenses on the model's baseline: L_P, L_{P_1/2}, L_J2 (Lens A's form at p 2),
    B (i)."""
    return dict(L_P=1.0 - L.P / m.base.P, L_Pq=1.0 - L.Pq / m.base.Pq,
                L_J2=1.0 - (L.J2 / m.base.J2) ** 0.5, enclosed=L.enclosed,
                n_enclosed=L.n_enclosed, unknowns=L.unknowns)


def _block_row(m: gcl.Model) -> dict[str, object]:
    return dict(n=m.n, P0=m.base.P, Pq0=m.base.Pq, J2_0=m.base.J2, cells0=m.cells,
                enclosed0=float(m.cost[m.walled0].sum()), n_enclosed0=int(m.walled0.sum()),
                nodem=int(m.nodem.sum()), host=socket.gethostname())


def cx_network(rows: Path, scratch: Path, metric: gcl.ComplexMetric, b: Block,
               name: str, *, replace: bool) -> None:
    """One network on the complex: its area prefixes' corridors (prefix_dir's rows: every budget
    and the whole network) cut into the footprints, one mesh (built in a temporary directory:
    it serves this network only), each prefix a state; the rows, the prefix rows with the lenses
    added, to cx_dir, written once (clearings.write_once, unless `replace`)."""
    bid = b.block_id
    out = cx_dir(rows, metric) / f"{bid}__{name}.parquet"
    clearings.check_new(out, replace=replace)
    t0 = time.perf_counter()
    every = pd.read_parquet(prefix_dir(rows) / f"{bid}__{name}.parquet")
    pre = every[every["pop"] == CX_POP].reset_index(drop=True)
    ordered = gpd.read_parquet(ordered_path(scratch, bid, name))
    stops = sorted({int(m) for m in pre.m})
    geo = gc.geometry(b)
    carving = gcl.Carving.of(geo, [[geo.local(corridor_of(ordered, m)) for m in stops]])
    t_carve = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as tmp:
        fx = metric.fixed(b, cache=Path(tmp), carving=carving)
    t_mesh = time.perf_counter() - t0 - t_carve
    m = metric.model(b, fx, demand=common.PopulationDemand())
    t_model = time.perf_counter() - t0 - t_carve - t_mesh
    got = {}
    for k, mk in enumerate(stops, start=1):
        ts = time.perf_counter()
        got[mk] = dict(_lens_row(m, m.road_lenses(gcl.Prefix(net=0, stop=k))),
                       t_state=time.perf_counter() - ts)
    df = pd.DataFrame([dict(r.to_dict(), **got[int(r.m)]) for _, r in pre.iterrows()]).assign(
        metric=metric.name, device=metric.device, atoms=carving.n,
        carved=int((carving.stop > 0).any(axis=1).sum()), nt=fx.fm.nt,
        **{k: v for k, v in _block_row(m).items() if k != "n"}, t_carve=t_carve,
        t_mesh=t_mesh, t_model=t_model, t_total=time.perf_counter() - t0,
        peak_rss_gb=_status_gb("VmHWM"))
    clearings.write_once(df, out, replace=replace)
    at = df[np.isclose(df.budget, PRICE_BUDGET)]
    print(f"{time.strftime('%H:%M:%S')} {bid} {name}: {len(stops)} prefixes, {carving.n} atoms, "
          f"mesh {fx.fm.nt} triangles in {t_mesh:.0f}s, model {t_model:.0f}s, states "
          f"{sum(v['t_state'] for v in got.values()):.0f}s; at D {PRICE_BUDGET:g} L_P "
          + ", ".join(f"{v:.4f}" for v in at.L_P), flush=True)


def cx_plain(rows: Path, metric: gcl.ComplexMetric, b: Block, *,
             scoring: search.PolishScoring, replace: bool) -> None:
    """The block's plain mesh (the node's cache): its baseline and each home's P_{1/2} weight
    (`<id>__homes`), the stored clearings (cx_stored) scored, then SIMP's default and the cheap
    preset run under P at D PRICE_BUDGET (their rows where relax.py and polish_greedy.py keep them,
    their polish scored by `scoring`; a preset whose rows stand is read, not run again) and their
    clearings scored (`<id>__plain`). Every row written once (clearings.write_once, unless
    `replace`, which runs the presets again too)."""
    bid = b.block_id
    d = cx_dir(rows, metric)
    out = d / f"{bid}__plain.parquet"
    clearings.check_new(out, replace=replace)
    pop = common.PopulationDemand()
    t0 = time.perf_counter()
    m = metric.clearing(b, demand=pop)
    clearings.write_once(pd.DataFrame(dict(block=bid, home=np.arange(m.n), pop=m.pop,
                                           u0=m.u_pop0, a_q=m.inj_q / m.pop, nodem=m.nodem,
                                           walled0=m.walled0)),
                         d / f"{bid}__homes.parquet", replace=replace)
    store = clearings.Store.committed()
    out_rows = [dict(clearing="baseline", run="", n_cleared=0, D=0.0, stored_perm=np.nan,
                     **_lens_row(m, m.base), t=0.0)]

    def score(key: str, run: str, ids: np.ndarray, stored: float) -> None:
        x = np.zeros(m.n, dtype=bool)
        x[ids] = True
        ts = time.perf_counter()
        out_rows.append(dict(clearing=key, run=run, n_cleared=int(x.sum()),
                             D=float(m.cost[x].sum()), stored_perm=stored,
                             **_lens_row(m, m.lenses(x)), t=time.perf_counter() - ts))
        print(f"  {bid} {key}: L_P {out_rows[-1]['L_P']:.4f} L_Pq {out_rows[-1]['L_Pq']:.4f} L_J2 "
              f"{out_rows[-1]['L_J2']:.4f} enclosed {out_rows[-1]['enclosed']:.5f}", flush=True)

    runs = store.runs()
    for key, run in cx_stored(metric.name)._asdict().items():
        if run not in runs:
            raise SystemExit(f"{run}: not in the store (clearings_export.py export first)")
        t = store.table(run)
        if not (t["block"] == bid).any():
            continue
        [stored] = t.loc[t["block"] == bid, "perm"].to_numpy(dtype=float)
        score(key, run, store.cleared(run, b, step=1), stored)
    t_stored = time.perf_counter() - t0
    del m
    # the presets under P on the complex (B's optimizer: free first, then the search)
    plan = relax.plan_of(relax.SIMP_DEFAULT)._replace(polish_scoring=scoring)
    c = polish_greedy.CHEAP
    dirs = cx_presets(metric, scoring)
    simp_rows = dirs.simp / f"{bid}_p{CX_POWER:g}.parquet"
    ts = time.perf_counter()
    if replace or not simp_rows.exists():
        clearings.write_once(pd.DataFrame([relax.one(bid, CX_POWER, metric, plan, PRICE_BUDGET,
                                                     demand=pop)]), simp_rows, replace=replace)
    t_simp = time.perf_counter() - ts
    cheap_rows = dirs.cheap / f"{bid}_p{CX_POWER:g}.parquet"
    ts = time.perf_counter()
    if replace or not cheap_rows.exists():
        polish_greedy.main(bid, CX_POWER, c.on(metric), c.picker, c.tries, c.width,
                           PRICE_BUDGET, c.pairs, scoring=scoring, replace=replace)
    t_cheap = time.perf_counter() - ts
    m = metric.clearing(b, demand=pop)
    cheap_search = c.on(metric).name.removeprefix(metric.name)       # the cheap preset's @e<eps>
    for key, f, ids_col, perm_col in ((f"simp_cx{scoring.suffix}", simp_rows, "simp_cleared",
                                       "simp_perm"),
                                      (f"cheap_cx{scoring.suffix}{cheap_search}", cheap_rows,
                                       "cleared", "perm")):
        r = pd.read_parquet(f).iloc[0]
        score(key, store_run(f.parent), np.asarray(r[ids_col], dtype=np.int64),
              float(r[perm_col]))
    clearings.write_once(pd.DataFrame(out_rows).assign(
        block=bid, metric=metric.name, device=metric.device, **_block_row(m),
        t_stored=t_stored, t_simp=t_simp, t_cheap=t_cheap, t_total=time.perf_counter() - t0,
        peak_rss_gb=_status_gb("VmHWM")), out, replace=replace)
    print(f"{time.strftime('%H:%M:%S')} {bid} plain: {len(out_rows)} states, SIMP {t_simp:.0f}s, "
          f"cheap {t_cheap:.0f}s, total {time.perf_counter() - t0:.0f}s", flush=True)


def cx(rows: Path, scratch: Path, metric: gcl.ComplexMetric, bid: str, items: list[str], *,
       scoring: search.PolishScoring, replace: bool) -> None:
    """The block's items on the complex, one after another: `plain` (cx_plain) or a network of
    prefix_dir (cx_network)."""
    _check_single_threaded()
    [b] = common.build_blocks([bid])
    for item in items:
        if item == "plain":
            cx_plain(rows, metric, b, scoring=scoring, replace=replace)
        else:
            cx_network(rows, scratch, metric, b, item, replace=replace)


# ------------------------------------------------------------------ the lineup, one (block, method)

def lineup_one(scratch: Path, bid: str, name: str, *, replace: bool) -> None:
    """One (block, method) of the research lineup end to end in this process (a cluster `cpu`
    task): its network proposed by lineup_arms' method (propose, through the derive cache), its
    canonical prefixes (prefixes_one), and its area prefixes scored on the complex under RUNTIME
    on a core (cx_network), every row under lineup_rows()."""
    import metrics
    _check_single_threaded()
    rows = lineup_rows()
    propose(rows, scratch, bid, name, lineup_arms()[name], False)
    if not net_path(scratch, bid, name, False).exists():
        raise SystemExit(f"{bid} {name}: no network (its proposal row says why)")
    [b] = common.build_blocks([bid])
    _BLOCKS[bid] = b
    if prefixes_one(PrefixJob(rows, scratch, bid, name)) is not None:
        raise SystemExit(f"{bid} {name}: its prefixes failed")
    metric = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.factorization_of("cpu"))
    cx_network(rows, scratch, metric, b, name, replace=replace)


def lineup_score(scratch: Path, bid: str, names: list[str], *, replace: bool) -> None:
    """The lineup's networks of `bid` already proposed and cut (consolidated `reuse-networks`),
    scored on the complex under RUNTIME on a core (cx_network)."""
    import metrics
    _check_single_threaded()
    [b] = common.build_blocks([bid])
    metric = gcl.complex_of(metrics.RUNTIME, factorization=gap_factor.factorization_of("cpu"))
    for name in names:
        cx_network(lineup_rows(), scratch, metric, b, name, replace=replace)


# ------------------------------------------------------------------ report

SHORT = {"clearance": "clear", "clearance_looped": "clear_loop", "cycle_native": "cycle",
         "cycle_native_betweenness_contrast": "cycle_desire", "resistance_lp": "resist_lp",
         "euclidean_grid": "grid", "greedy_arterial_access_displacement": "arterial"}
REPORT_BUDGETS = (0.05, 0.10)     # the table's budgets (Lens A now and Lens A of the 220 study)
PRICE_BUDGET = 0.05               # the stored clearings' budget


def _read_all(paths: list[Path]) -> pd.DataFrame:
    return (pd.concat([pd.read_parquet(f) for f in paths], ignore_index=True) if paths
            else pd.DataFrame())


def _b_rank(g: pd.DataFrame, enc: str, val: str) -> np.ndarray:
    """Ranks in B's order (owner 2026-10-11, "B SGTM"): the least population left enclosed first,
    then the larger Lens A over the rest. 1 + the rows strictly ahead, so exact ties share one."""
    e, v = g[enc].to_numpy(), g[val].to_numpy()
    ahead = (e[None, :] < e[:, None]) | ((e[None, :] == e[:, None]) & (v[None, :] > v[:, None]))
    return 1.0 + ahead.sum(axis=1)


# ------------------------------------------------------------------ report on the complex

CX_LENSES = ("L_P", "L_Pq", "L_J2")      # P the headline, P_{1/2} the equity view, J_2 the old (c)


def cx_rows(rows: Path, metric: gcl.ComplexMetric, lineup: tuple[str, ...]) -> pd.DataFrame:
    """The networks' prefix rows on the complex (cx_network), every block's."""
    d = cx_dir(rows, metric)
    return _read_all([f for name in lineup for f in sorted(d.glob(f"*__{name}.parquet"))])


def cx_plain_rows(rows: Path, metric: gcl.ComplexMetric) -> pd.DataFrame:
    """The plain meshes' rows (cx_plain: the baseline, the stored clearings, the presets)."""
    return _read_all(sorted(cx_dir(rows, metric).glob("*__plain.parquet")))


def cx_lens_table(t: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Per (block, method) at `budget` (the area prefix reaching it): each lens with its rank among
    the methods reaching the budget in B's order (the population left walled in first)."""
    at = t[np.isclose(t.budget, budget)].copy()
    at["reach"] = at.D_area >= budget - 1e-9
    for lens in CX_LENSES:
        rank = pd.Series(np.nan, index=at.index)
        for _, g in at[at.reach].groupby("block"):
            rank[g.index] = _b_rank(g, "enclosed", lens)
        at[f"rank_{lens}"] = rank
    return at


def cxreport(rows: Path, lineup: tuple[str, ...], metric: gcl.ComplexMetric) -> None:
    """The road lineup on the complex (module docstring): per budget of REPORT_BUDGETS the lens
    table and the leaders under each lens in B's order; the price of roads at PRICE_BUDGET (each
    preset's clearing on the complex against the best road prefix in B's order, by lens); the
    homes the metric walls in and what the prefixes and clearings free. Summaries
    <rows>/<metric>/summary_<device>_*.csv (committed)."""
    pd.set_option("display.width", 250)
    t = cx_rows(rows, metric, lineup)
    pl = cx_plain_rows(rows, metric)
    out = rows / metric.name

    def path(what: str) -> Path:
        return out / f"summary_{metric.device}_{CX_POP}_{budgets_tag()}_{what}.csv"
    print(f"{t.block.nunique()} blocks with networks on {metric.name} ({metric.device}), "
          f"{pl.block.nunique()} with the plain mesh's rows")
    for budget in REPORT_BUDGETS:
        at = cx_lens_table(t, budget)
        at.to_csv(path(f"lens_D{budget:g}"), index=False)
        print(f"\nAT AREA D {budget:g}: methods reaching it per block "
              f"{at.groupby('block').reach.sum().median():.0f} (median); the medians of each lens")
        print(at[at.reach].groupby("method")[list(CX_LENSES)].median().round(4)
              .rename(index=SHORT).to_string())
        for lens in CX_LENSES:
            lead = at[at[f"rank_{lens}"] == 1].drop_duplicates("block").method.map(SHORT)
            print(f"  leader under {lens} in B's order: "
                  + ", ".join(f"{k} {v}" for k, v in lead.value_counts().items()))
    # the price of roads
    at = cx_lens_table(t, PRICE_BUDGET)
    prow = []
    clear = pl[pl.clearing.str.fullmatch(r"(simp|cheap)_cxE(@e[0-9.e-]+)?")]
    for bid, g in at[at.reach].groupby("block"):
        c = clear[clear.block == bid].set_index("clearing")
        for lens in CX_LENSES:
            best = g.loc[g[f"rank_{lens}"].idxmin()]
            r = dict(block=bid, lens=lens, best=best.method, best_enc=float(best.enclosed),
                     best_val=float(best[lens]), best_D=float(best.D_area),
                     best_km=float(best.road_m) / 1000)
            for key, cr in c.iterrows():
                fam = key.split("_")[0]
                di, dii = float(cr.enclosed) - r["best_enc"], float(cr[lens]) - r["best_val"]
                r |= {f"{fam}_val": float(cr[lens]), f"{fam}_enc": float(cr.enclosed),
                      f"{fam}_D": float(cr.D), f"{fam}_di": di, f"{fam}_dii": dii,
                      f"{fam}_ahead": bool(di < 0 or (di == 0 and dii > 0))}
            prow.append(r)
    pr = pd.DataFrame(prow)
    pr.to_csv(path("price"), index=False)
    print(f"\nTHE PRICE OF ROADS at area D {PRICE_BUDGET:g} on the complex, in B's order (a lower "
          "bound: the optimizers are heuristics)")
    for lens in CX_LENSES:
        q = pr[(pr.lens == lens) & pr.simp_val.notna()]
        for fam in ("simp", "cheap"):
            tie = q[q[f"{fam}_di"] == 0]
            share = (tie.best_val / tie[f"{fam}_val"])
            print(f"  {lens} {fam}: ahead of the best road on {int(q[f'{fam}_ahead'].sum())} of "
                  f"{len(q)}; behind on (i) on {int((q[f'{fam}_di'] > 0).sum())}; where (i) ties "
                  f"({len(tie)}), minus the best road median {tie[f'{fam}_dii'].median():+.4f} "
                  f"(p10 {tie[f'{fam}_dii'].quantile(0.1):+.4f}, p90 "
                  f"{tie[f'{fam}_dii'].quantile(0.9):+.4f}; {tie[f'{fam}_dii'].min():+.4f} .. "
                  f"{tie[f'{fam}_dii'].max():+.4f}), the best road's share {share.median():.2f} "
                  f"(p10 {share.quantile(0.1):.2f}, p90 {share.quantile(0.9):.2f})")
    # the walled-in homes
    base = pl[pl.clearing == "baseline"].set_index("block")
    held = base[base.enclosed0 > 0]
    print(f"\nWALLED IN AT BASELINE on the complex: {len(held)} of {len(base)} blocks "
          f"({int(held.n_enclosed0.sum())} homes, a share {held.enclosed0.min():.5f} -- "
          f"{held.enclosed0.max():.5f}, median {held.enclosed0.median():.5f})")
    pre = at[at.block.isin(held.index) & at.reach]
    for mth, g in pre.groupby("method"):
        some = int((g.enclosed < g.block.map(held.enclosed0)).sum())
        print(f"  {SHORT[mth]:13s} {len(g):3d} blocks at D {PRICE_BUDGET:g}: frees some on "
              f"{some}, all on {int((g.enclosed == 0).sum())}")
    for key, g in pl[pl.block.isin(held.index) & (pl.clearing != "baseline")].groupby("clearing"):
        print(f"  {key:10s} {len(g):3d} blocks: frees some on "
              f"{int((g.enclosed < g.block.map(held.enclosed0)).sum())}, all on "
              f"{int((g.enclosed == 0).sum())}")
    pl.to_csv(path("clearings"), index=False)


# ------------------------------------------------------------------ main

def lineup_of(arg: str) -> tuple[str, ...]:
    """A lineup named on the command line: the networks, comma-separated, each once."""
    names = tuple(arg.split(","))
    if not all(names) or len(set(names)) != len(names):
        raise SystemExit(f"a lineup names each network once, none empty: {arg!r}")
    return names


if __name__ == "__main__":
    a, again = clearings.replace_flag(sys.argv[1:])
    cmd = a[0]
    if cmd == "propose":
        if a[5] not in ("cache", "fresh"):
            raise SystemExit(f"propose takes cache or fresh, not {a[5]!r}")
        propose(_lineup_root(Path(a[1])), Path(a[2]), a[3], a[4], lineup_arms()[a[4]],
                a[5] == "fresh")
    elif cmd == "propose-all":
        if a[7] not in ("time-hits", "no"):
            raise SystemExit(f"propose-all takes time-hits or no, not {a[7]!r}")
        propose_all(_lineup_root(Path(a[1])), Path(a[2]), int(a[3]), common.ids_of(a[4]),
                    a[5].split(","), float(a[6]) * 3600, a[7] == "time-hits")
    elif cmd == "prefixes":
        prefixes(_lineup_root(Path(a[1])), lineup_of(a[2]), Path(a[3]), int(a[4]),
                 common.ids_of(a[5]))
    elif cmd == "cx":
        if len(a) != 7:
            raise SystemExit("cx <rows> <scratch> <metric> <cpu|gpu> <id> <items,>")
        # the device resolves the factorization and the presets' polish scoring (phase 2a: E on
        # a core, the eps world on a GPU; the same clearings)
        polish = {"cpu": "E", "gpu": ""}[a[4]]
        cx(Path(a[1]), Path(a[2]),
           gcl.complex_of(a[3], factorization=gap_factor.factorization_of(a[4])), a[5],
           a[6].split(","), scoring=search.polish_scoring_of(polish), replace=again)
    elif cmd == "lineup-one":
        if len(a) != 4:
            raise SystemExit("lineup-one <scratch> <id> <method> [--replace]")
        lineup_one(Path(a[1]), common.ids_of(a[2])[0], a[3], replace=again)
    elif cmd == "lineup-score":
        if len(a) != 4:
            raise SystemExit("lineup-score <scratch> <id> <methods,> [--replace]")
        lineup_score(Path(a[1]), common.ids_of(a[2])[0], a[3].split(","), replace=again)
    elif cmd == "cxreport":
        if len(a) != 5:
            raise SystemExit("cxreport <rows> <lineup,> <metric> <cpu|gpu>")
        cxreport(Path(a[1]), lineup_of(a[2]),
                 gcl.complex_of(a[3], factorization=gap_factor.factorization_of(a[4])))
    else:
        raise SystemExit(f"unknown mode {cmd!r}")
