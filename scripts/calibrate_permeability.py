"""Permeability calibration probe: for each internal-capable method, at its NATURAL config (no
over-provisioning), on each example region, build the permeability-vs-displacement frontier and
print the tables + a proposal a human uses to pick the two universal thresholds -- the
matched-displacement % `D` (Lens A) and the matched-permeability `P*` (Lens B) -- baked into
`conf/permeability.yaml` at the Task 3 checkpoint. This script writes NO repo constants; it only
prints.

Design lessons carried from an earlier (now-retired) calibration probe -- two bugs this one is
built to avoid:

1. IN-PROCESS CROSS-REGION STATE BLEED: running all 7 regions in one process previously returned
   the WRONG city's blocks for a spec (a Cape Town region spec resolved to Nairobi data). Every
   region here loads and reblocks in a FRESH SUBPROCESS: `--region <name>` loads+computes exactly
   ONE region and prints its result (a human-readable table, then one JSON line); the no-arg
   orchestrator `subprocess.run`s this same module once per region (`python -m
   scripts.calibrate_permeability --region <name>`) and aggregates the parsed JSON. Never loop over
   regions in-process.
2. OVER-PROVISIONING SUPPRESSED THE METRIC: the old probe pushed every method's budget knobs
   (max_roads, budget_frac, spacing) well past the example generators' own settings, to force a
   search plateau -- which buries the metric's natural dynamic range. This probe runs each method
   at EXACTLY the config its own example generator uses; no `over_provision`-style overrides.

Region loading mirrors two generators, replayed verbatim (down to the literal hydra override
strings), NOT unified into one shared config -- replaying each generator's own choices IS the
"natural config" this probe is required to use:
  - `scripts/gen_multiblock_example.py`, for the 6 multiblock regions ({depth, depth_density,
    density_compactness} metric x {capetown, nairobi} city): dense_compact screen + dense_cluster
    region_builder up to 3000 buildings, arterial's `candidate_policy=fixed`/`max_anchors=64`
    tractability knobs, and clearance_looped/euclidean_grid retuned for regional scale.
  - `scripts/gen_method_comparison.py`, for the pinned single-block flagship (`ZAF.9.3.1_1_40972`):
    identity screen/region_builder (an explicit `block_ids` group), arterial capped to
    `max_roads=8`, clearance_looped/euclidean_grid left at `compare_config.yaml` defaults.
`osm_footpaths` (the real as-built footpath network) is added wherever a committed OSM snapshot
exists for that region -- a REFERENCE only: it is shown in every table but excluded from the (D,
P*) proposal math below, which is about the three tunable synthetic methods.

Run one region (fast; also the smoke-test / debugging entry point):
    pixi run python -m scripts.calibrate_permeability --region depth/capetown
    pixi run python -m scripts.calibrate_permeability --region method_comparison

Run the full probe (SLOW -- 3-4 methods x 7 regions, each a subprocess reblock + curve build; the
controller should budget real wall-clock time, not assume a quick turnaround):
    pixi run python -m scripts.calibrate_permeability

PARALLELISM (two independent levels, both embarrassingly parallel -- no shared mutable state
crosses a region or a method, so results are IDENTICAL to a fully serial run, just faster):
  - Level 1 (across regions): the no-arg orchestrator runs the up-to-7 `--region` subprocesses
    CONCURRENTLY on a `ThreadPoolExecutor` (threads are fine -- each just blocks on
    `subprocess.run`, releasing the GIL) capped at `--max-region-concurrency`
    (default `DEFAULT_MAX_REGION_CONCURRENCY`). Each region STAYS a fresh isolated subprocess --
    this only changes how many run at once, never collapses them into one process (that would
    reintroduce the cross-region state bleed bug (1) above).
  - Level 2 (across methods, inside one region): `run_region` builds each method's frontier (a
    reblock + up to 21 independent sparse solves -- CPU-bound) on a `ProcessPoolExecutor` (fork
    context, mirrors `reblock.screen.dense_compact`'s worker-pool pattern) capped at
    `METHOD_WORKERS`. The methods share the region's `Block` read-only (each worker gets its own
    pickled copy; nothing is mutated), so this is safe.
  - Progress: STARTED/DONE + elapsed for every region and every method, plus a live
    "solve i/total" counter per method's `permeability_curve` sweep, all on STDERR (so they never
    interleave with a `--region` subprocess's STDOUT `_MARKER` JSON line).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from reblock.budget import Curve, building_radii, displacement_curve
from reblock.contracts import Block, Method, Screen, Source
from reblock.derivations import propose
from reblock.permeability import PermeabilityParams, permeability_curve
from reblock.pipeline import build_regions
from reblock.region import RegionBuilder, region_block

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "conf"

# The three tunable synthetic methods this probe calibrates against (the internal-capable
# reblockers); osm_footpaths is a fourth, reference-only method added per-region below.
SYNTH_METHODS = ("greedy_arterial_repulsion", "clearance_looped", "euclidean_grid")
METRICS = ("depth", "depth_density", "density_compactness")
CITIES = ("capetown", "nairobi")
# The deepest block in a topology-tractable size window, pinned in gen_method_comparison.py.
PINNED_BLOCK_ID = "ZAF.9.3.1_1_40972"
PINNED_SNAPSHOT = ROOT / "examples" / "method-comparison" / "desire_lines_40972.geojson"

DISPLACEMENT_PCTS = (0.10, 0.20, 0.30, 0.40)     # Lens A candidates: "permeability at D%"
PERMEABILITY_LEVELS = (0.3, 0.4, 0.5)            # Lens B candidates: "displacement to reach P*"

# Level 2 cap: at most this many methods' frontiers build concurrently within one region (there
# are at most 4 -- 3 synthetic + osm_footpaths -- so this never actually queues).
METHOD_WORKERS = 4
# Level 1 default cap: at most this many region subprocesses run concurrently. Each region briefly
# forks its OWN 16-worker screen fine-pass pool (reblock.screen.dense_compact) before settling
# into its <=4 method workers, so 5 regions concurrent can brief-spike to ~80 processes during
# that screening phase -- acceptable on 48 cores per the coordinator's guidance; tune via
# `--max-region-concurrency` if it thrashes.
DEFAULT_MAX_REGION_CONCURRENCY = 5

# Prefixes the one stdout line a `--region` subprocess uses to hand its JSON result back to the
# orchestrator; every other line that process prints is the human-readable table (ignored by the
# orchestrator, which reprints from the parsed JSON instead). All progress/STARTED/DONE logging
# goes to STDERR instead (see `_log`), so it never lands on this stdout channel.
_MARKER = "###RESULT_JSON### "


def _log(msg: str) -> None:
    """Progress logging -- always STDERR (never stdout, which carries only the human table + the
    one `_MARKER` JSON line a `--region` subprocess's caller parses) and always flushed (the
    controller watches this live during the slow full run)."""
    print(msg, file=sys.stderr, flush=True)


@dataclass(frozen=True)
class MethodFrontier:
    """One method's permeability-vs-displacement frontier summary on one region, read off the
    index-aligned curves at a few fixed displacement %s and permeability levels."""
    n_roads: int
    terminal_permeability: float
    terminal_displacement: float
    perm_at_displacement: dict[str, float]          # "10%".."40%" -> permeability (-inf unreached)
    displacement_at_permeability: dict[str, float]  # "0.3".."0.5" -> displacement (inf unreached)


@dataclass(frozen=True)
class RegionResult:
    region: str
    block_id: str
    n_parcels: int
    methods: dict[str, MethodFrontier | None]        # None = the method proposed no roads


def region_names() -> list[str]:
    """Every region this probe covers, in the `--region` display-name form: `<metric>/<city>` for
    the 6 multiblock regions, plus `method_comparison` for the pinned flagship block."""
    return [f"{metric}/{city}" for metric in METRICS for city in CITIES] + ["method_comparison"]


def _load_permeability_params() -> PermeabilityParams:
    raw = cast(DictConfig, OmegaConf.load(CONFIG_DIR / "permeability.yaml"))
    return PermeabilityParams(g_walk=float(raw.g_walk), g_road=float(raw.g_road),
                              g_street=float(raw.g_street), corridor_m=float(raw.corridor_m))


def _load_pinned_block() -> tuple[Block, dict[str, Method]]:
    """The pinned method-comparison flagship, loaded EXACTLY as `scripts/gen_method_comparison.py`
    loads it: the same hydra overrides (data, the explicit `block_ids` seed group, arterial's
    `max_roads=8`, the committed OSM snapshot), the default (identity) screen/region_builder, then
    `region_block` on the singleton region (a no-op union on one member)."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="compare_config", overrides=[
            "data=capetown_full", f"block_ids=[[{PINNED_BLOCK_ID}]]", "max_blocks=1",
            "all_methods.greedy_arterial_repulsion.max_roads=8",
            f"desire_source.snapshot={PINNED_SNAPSHOT}"])
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))
    region = build_regions(source, screen, region_builder, [[PINNED_BLOCK_ID]], 1)[0]
    block = region_block(region)
    methods = {n: cast(Method, instantiate(cfg.all_methods[n])) for n in SYNTH_METHODS}
    # The pinned flagship's OSM snapshot is committed (this is why it's pinned) -- unlike the
    # multiblock regions below, no existence guard is needed; mirrors gen_method_comparison.py.
    methods["osm_footpaths"] = cast(Method, instantiate(cfg.all_methods.osm_footpaths))
    return block, methods


def _load_multiblock_region(metric: str, city: str) -> tuple[Block, dict[str, Method]]:
    """One multiblock region, loaded EXACTLY as `scripts/gen_multiblock_example.py` loads it: the
    same hydra overrides (metric, city data, the dense_compact screen, the dense_cluster
    region_builder, the SAME method tractability/retuning overrides that generator applies -- these
    are not over-provisioning, they are what that generator calls "natural" at regional scale), the
    same screen-selection seed -> `desire_lines_<seed>.geojson` snapshot lookup for osm_footpaths
    (added only if that generator has already produced/committed the snapshot for this region)."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="compare_config", overrides=[
            f"metric={metric}", f"data={city}_full", "screen=dense_compact",
            "region_builder=dense_cluster", "region_builder.max_buildings=3000", "max_blocks=1",
            "all_methods.greedy_arterial_repulsion.candidate_policy=fixed",
            "+all_methods.greedy_arterial_repulsion.max_anchors=64",
            "all_methods.clearance_looped.base.depth_target=3",
            "all_methods.clearance_looped.base.max_roads=3000",
            "all_methods.clearance_looped.budget_frac=0.30",
            "all_methods.clearance_looped.search_radius_m=60",
            "all_methods.euclidean_grid.spacing=250"])
    source = cast(Source, instantiate(cfg.data))
    screen = cast(Screen, instantiate(cfg.screen))
    region_builder = cast(RegionBuilder, instantiate(cfg.region_builder))

    selection = screen.select(source) or []
    if not selection:
        raise RuntimeError(f"metric={metric!r} city={city!r} flagged 0 blocks -- check its gate")
    seed = selection[0]

    source.block_ids = None                                          # type: ignore[attr-defined]
    region = build_regions(source, screen, region_builder, None, 1)[0]

    block = region_block(region)
    methods = {n: cast(Method, instantiate(cfg.all_methods[n])) for n in SYNTH_METHODS}

    out_dir = (ROOT / f"examples/multiblock_{metric}" if city == "capetown"
              else ROOT / f"examples/{city}/multiblock_{metric}")
    snapshot = out_dir / f"desire_lines_{seed}.geojson"
    if snapshot.exists():
        with open_dict(cfg):
            cfg.desire_source.snapshot = str(snapshot)
        methods["osm_footpaths"] = cast(Method, instantiate(cfg.all_methods.osm_footpaths))
    return block, methods


def _load_region(name: str) -> tuple[Block, dict[str, Method]]:
    if name == "method_comparison":
        return _load_pinned_block()
    metric, sep, city = name.partition("/")
    if not sep or metric not in METRICS or city not in CITIES:
        raise ValueError(f"unknown region {name!r}; expected one of {region_names()}")
    return _load_multiblock_region(metric, city)


def permeability_at_displacement(perm_curve: Curve, disp_curve: Curve, d: float) -> float:
    """The permeability at the first drainage-ordered `_sweep` sample whose displacement fraction is
    >= `d`. `perm_curve` and `disp_curve` must come from the SAME `roads`/`n_points`/`tol` (both
    built over the shared `_sweep`, so their `.cost` lists -- hence sample indices -- line up
    exactly; see `permeability_curve`/`displacement_curve`). `float("-inf")` if no sample reaches
    `d` (the method's full network falls short of that displacement budget) -- mirrors the retired
    `max_internal_within`'s unreachable sentinel."""
    for i, disp in enumerate(disp_curve.benefit):
        if disp >= d:
            return perm_curve.benefit[i]
    return float("-inf")


def displacement_at_permeability(perm_curve: Curve, disp_curve: Curve, p: float) -> float:
    """The displacement fraction at the first `_sweep` sample whose permeability is >= `p` (same
    index-alignment precondition as `permeability_at_displacement`). `float("inf")` if `p` is never
    reached. This is `permeability_at_displacement` with the two curves and the comparison
    direction swapped -- no independent logic, so it rides on that function's unit test rather than
    getting its own (only the brief's named frontier-extraction helper is required to be tested)."""
    for i, perm in enumerate(perm_curve.benefit):
        if perm >= p:
            return disp_curve.benefit[i]
    return float("inf")


def _method_frontier(block: Block, method: Method, params: PermeabilityParams, *,
                     label: str) -> MethodFrontier | None:
    """One method's frontier summary on `block`: reblock once (natural config, already baked into
    `method`), build the index-aligned permeability/displacement curves, and read off the table
    cells. `None` if the method proposes no roads (skip it). `label` (e.g.
    `"depth/capetown · clearance_looped"`) prefixes the live
    "solve i/total" progress line `permeability_curve`'s sparse-solve sweep emits -- the only
    per-sample-granular part of the pipeline (`displacement_curve` is pure geometry, no solve)."""
    prop = propose(method, block)
    roads = prop.roads
    if roads is None or roads.empty:
        return None
    radii = building_radii(block.building_points, params.corridor_m)

    def _report(i: int, total: int) -> None:
        _log(f"    {label}: solve {i}/{total}")

    perm_curve = permeability_curve(block, roads, params, n_points=20, progress=_report)
    disp_curve = displacement_curve(block, roads, radii, corridor_m=params.corridor_m, n_points=20)
    return MethodFrontier(
        n_roads=int(len(roads)),
        terminal_permeability=perm_curve.benefit[-1],
        terminal_displacement=disp_curve.benefit[-1],
        perm_at_displacement={
            f"{int(round(d * 100))}%": permeability_at_displacement(perm_curve, disp_curve, d)
            for d in DISPLACEMENT_PCTS
        },
        displacement_at_permeability={
            str(p): displacement_at_permeability(perm_curve, disp_curve, p)
            for p in PERMEABILITY_LEVELS
        },
    )


def _frontier_worker(label: str, block: Block, method: Method,
                     params: PermeabilityParams) -> MethodFrontier | None:
    """Module-level (NOT a closure) so a fork `ProcessPoolExecutor` can dispatch it -- mirrors
    `reblock.screen.dense_compact._chunk_depths`'s pattern. Runs entirely inside the forked child;
    STARTED/DONE/solve-progress `_log` calls print to that child's inherited stderr fd, so they
    land in the region subprocess's own stderr stream with no extra plumbing back to the parent."""
    t0 = time.monotonic()
    _log(f"  {label}: STARTED")
    result = _method_frontier(block, method, params, label=label)
    elapsed = time.monotonic() - t0
    roads_note = "no roads proposed" if result is None else f"{result.n_roads} roads"
    _log(f"  {label}: DONE in {elapsed:.1f}s ({roads_note})")
    return result


def run_region(name: str) -> RegionResult:
    """Load + reblock + score ONE region, entirely within this process -- the unit of Level-1
    subprocess isolation (see the module docstring's bug (1)). The methods' frontiers build on a
    Level-2 fork pool (`METHOD_WORKERS` workers, capped to the actual method count) when forking is
    available; a plain serial loop otherwise (e.g. a single method, or no `fork` start method) --
    same code path `_method_frontier` runs either way, so results are identical."""
    block, methods = _load_region(name)
    params = _load_permeability_params()
    workers = min(METHOD_WORKERS, len(methods), os.cpu_count() or 1)
    if workers > 1 and "fork" in multiprocessing.get_all_start_methods():
        ctx = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = {
                ex.submit(_frontier_worker, f"{name} · {mname}", block, method, params): mname
                for mname, method in methods.items()
            }
            unordered = {futures[fut]: fut.result() for fut in as_completed(futures)}
    else:
        unordered = {mname: _frontier_worker(f"{name} · {mname}", block, method, params)
                    for mname, method in methods.items()}
    # Reassemble in the original method order (as_completed finishes out of order) so the printed
    # table reads identically to a serial run.
    method_results = {mname: unordered[mname] for mname in methods}
    return RegionResult(region=name, block_id=block.block_id, n_parcels=int(len(block.parcels)),
                        methods=method_results)


def _to_jsonable(result: RegionResult) -> dict[str, object]:
    return {
        "region": result.region, "block_id": result.block_id, "n_parcels": result.n_parcels,
        "methods": {
            name: (None if mf is None else {
                "n_roads": mf.n_roads,
                "terminal_permeability": mf.terminal_permeability,
                "terminal_displacement": mf.terminal_displacement,
                "perm_at_displacement": mf.perm_at_displacement,
                "displacement_at_permeability": mf.displacement_at_permeability,
            })
            for name, mf in result.methods.items()
        },
    }


def _from_jsonable(data: dict[str, Any]) -> RegionResult:
    methods: dict[str, MethodFrontier | None] = {}
    for name, raw in data["methods"].items():
        methods[name] = None if raw is None else MethodFrontier(
            n_roads=raw["n_roads"], terminal_permeability=raw["terminal_permeability"],
            terminal_displacement=raw["terminal_displacement"],
            perm_at_displacement=raw["perm_at_displacement"],
            displacement_at_permeability=raw["displacement_at_permeability"])
    return RegionResult(region=data["region"], block_id=data["block_id"],
                        n_parcels=data["n_parcels"], methods=methods)


def _fmt(v: float, unreached: float) -> str:
    return "unreached" if v == unreached else f"{v:.4f}"


def _print_table(result: RegionResult) -> None:
    print(f"=== region {result.region} ({result.block_id}, {result.n_parcels} parcels) ===")
    for mname, mf in result.methods.items():
        if mf is None:
            print(f"  {mname}: no roads proposed -- skip")
            continue
        pa = ", ".join(f"{k}->{_fmt(v, float('-inf'))}" for k, v in mf.perm_at_displacement.items())
        da = ", ".join(f"P*={k}->{_fmt(v, float('inf'))}"
                       for k, v in mf.displacement_at_permeability.items())
        print(f"  {mname}: {mf.n_roads} roads, terminal permeability="
              f"{mf.terminal_permeability:.4f} @ displacement={mf.terminal_displacement:.4f}")
        print(f"    permeability at displacement %: {pa}")
        print(f"    displacement frac to reach P*:  {da}")


def _propose(all_results: list[RegionResult]) -> None:
    """Print the (D, P*) proposal inputs -- NOT a config write. `D` is chosen from the humane
    (<=30%) candidates by the widest mean cross-method permeability spread (the dynamic-range
    sweet spot); `P*` is chosen as the level with the most (method, region) coverage among the
    three synthetic methods (ties broken toward the higher/harder bar). Both are printed as a
    starting point for the human sign-off at the Task 3 checkpoint, not baked into anything here."""
    print("\n" + "=" * 78)
    print("PROPOSAL INPUTS -- human reviews these; this probe writes NO repo constants")
    print("=" * 78)

    print("\n--- Lens A candidate D: mean cross-method permeability spread by displacement % ---")
    best_d: str | None = None
    best_d_spread = -1.0
    for d in DISPLACEMENT_PCTS:
        key = f"{int(round(d * 100))}%"
        spreads = []
        for r in all_results:
            vals = [mf.perm_at_displacement[key] for name, mf in r.methods.items()
                    if name in SYNTH_METHODS and mf is not None
                    and mf.perm_at_displacement[key] != float("-inf")]
            if len(vals) >= 2:
                spreads.append(max(vals) - min(vals))
        mean_spread = sum(spreads) / len(spreads) if spreads else float("nan")
        print(f"  D={key}: mean spread={mean_spread:.4f} over {len(spreads)}/{len(all_results)} "
              "regions (>=2 synthetic methods reaching it)")
        if d <= 0.30 and spreads and mean_spread > best_d_spread:
            best_d, best_d_spread = key, mean_spread

    print("\n--- Lens B candidate P*: (method, region) coverage + displacement-cost spread ---")
    best_p: str | None = None
    best_p_coverage = -1
    for p in PERMEABILITY_LEVELS:
        key = str(p)
        reached = [mf.displacement_at_permeability[key] for r in all_results
                  for name, mf in r.methods.items()
                  if name in SYNTH_METHODS and mf is not None
                  and mf.displacement_at_permeability[key] != float("inf")]
        spread = (max(reached) - min(reached)) if len(reached) >= 2 else float("nan")
        total_pairs = len(all_results) * len(SYNTH_METHODS)
        print(f"  P*={p}: {len(reached)}/{total_pairs} (method, region) pairs reach it, "
              f"displacement-cost spread={spread:.4f}")
        if len(reached) >= best_p_coverage:
            best_p, best_p_coverage = key, len(reached)

    print("\n--- proposal ---")
    if best_d is not None:
        print(f"  proposed D  = {best_d} (max cross-method spread among candidates <= 30%)")
    else:
        print("  proposed D: no candidate had >=2 synthetic methods reaching it -- "
              "inspect the tables above by hand")
    if best_p is not None:
        print(f"  proposed P* = {best_p} (highest level with the most (method, region) coverage)")
    else:
        print("  proposed P*: no candidate was reached by any synthetic method")
    print("\nHuman: review the frontiers above, sign off on (D, P*), then bake them into "
          "conf/permeability.yaml by hand.")


def _run_region_subprocess(name: str, *, timeout_s: float | None) -> RegionResult:
    """Run region `name` in a FRESH subprocess and parse its `_MARKER`-prefixed JSON result line --
    the isolation the module docstring's bug (1) requires. On failure, relay the child's stdout/
    stderr before raising, so the failure is diagnosable without re-running by hand. The child's
    own STARTED/DONE/solve-progress lines land directly on ITS stderr (inherited, unbuffered by
    `capture_output` only insofar as we don't relay them live here -- see `_run_all_regions`,
    which wraps this call with the Level-1 region-level STARTED/DONE logging instead)."""
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.calibrate_permeability", "--region", name],
        cwd=str(ROOT), capture_output=True, text=True, timeout=timeout_s,
    )
    marker_line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(_MARKER)), None)
    if proc.returncode != 0 or marker_line is None:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"region {name!r} subprocess failed (exit {proc.returncode}) or produced no result "
            "marker -- stdout/stderr relayed above")
    return _from_jsonable(cast("dict[str, Any]", json.loads(marker_line[len(_MARKER):])))


def _run_all_regions(names: list[str], *, timeout_s: float | None,
                     max_region_concurrency: int) -> list[RegionResult]:
    """Level 1: run every region's `_run_region_subprocess` CONCURRENTLY on a `ThreadPoolExecutor`
    capped at `max_region_concurrency` (threads block on `subprocess.run`, so this only bounds how
    many child processes are in flight, not CPU -- each region stays its own fresh, isolated OS
    subprocess). Logs region STARTED/DONE + elapsed + a running `k/total` completion count to
    stderr; returns results in `names` order regardless of completion order, so downstream printing
    reads identically to a serial run."""
    total = len(names)
    done = 0
    lock = threading.Lock()

    def _run_one(name: str) -> tuple[str, RegionResult]:
        nonlocal done
        t0 = time.monotonic()
        _log(f">>> region {name}: STARTED")
        result = _run_region_subprocess(name, timeout_s=timeout_s)
        elapsed = time.monotonic() - t0
        with lock:
            done += 1
            k = done
        _log(f"<<< region {name}: DONE in {elapsed:.1f}s ({k}/{total})")
        return name, result

    with ThreadPoolExecutor(max_workers=max_region_concurrency) as ex:
        futures = [ex.submit(_run_one, name) for name in names]
        by_name = dict(fut.result() for fut in as_completed(futures))
    return [by_name[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", choices=region_names(), default=None,
                        help="run + print ONE region in THIS process (the subprocess-isolation "
                             "unit the no-arg orchestrator below shells out to)")
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-region subprocess timeout in seconds (orchestrator mode only)")
    parser.add_argument("--max-region-concurrency", type=int,
                        default=DEFAULT_MAX_REGION_CONCURRENCY,
                        help="Level-1 cap on concurrently-running region subprocesses "
                             f"(orchestrator mode only; default {DEFAULT_MAX_REGION_CONCURRENCY})")
    args = parser.parse_args()

    if args.region is not None:
        result = run_region(args.region)
        _print_table(result)
        print(f"{_MARKER}{json.dumps(_to_jsonable(result))}")
        return

    all_results = _run_all_regions(region_names(), timeout_s=args.timeout,
                                   max_region_concurrency=args.max_region_concurrency)
    for result in all_results:
        _print_table(result)
    _propose(all_results)


if __name__ == "__main__":
    main()
