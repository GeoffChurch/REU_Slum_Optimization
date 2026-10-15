"""Checks of the committed clearings (clearings.py) that need the real blocks and the scorer; the
rest are unit tests (tests/test_clearings.py).

    PYTHONPATH=. uv run python research/roadless/clearings_check.py blocks
    PYTHONPATH=. uv run python research/roadless/clearings_check.py banks
    PYTHONPATH=. uv run python research/roadless/clearings_check.py rescore <ids,> <run[@step]>,... \
        <own|metric>

From the repo root, on the CPU, single-threaded (OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1).

`blocks`: every committed block rebuilt through common.build_blocks is the block its clearings were
committed for (Store.check: the count and the footprints' fingerprint), and every committed row
passes the export's row checks against it (clearings_export.check_rows: positions in range and
distinct, D recomputed from them equal to the row's D, free first's buildings inside the clearing).
Run it after anything that could change a block's footprints or their order.

`banks`: the blocks the cluster runs were shipped (.cluster_submit/runs/*/bank.pkl, written by
common.write_bank from the source at the time of the run, 2026-10-03 on) have the footprints
committed today, block by block: the clearings written on the cluster index the buildings the
cluster saw, and those are the buildings of today. Reports the span of dates each block was
seen over, and every disagreement.

`rescore`: on each block, each run's committed clearing read through the reader (the fingerprint
checked), scored again on the CPU under the run's own metric (`own`) or under the metric named (a
complex metric's name, say: every clearing under it, keyed by the name, its lens beside the
run's own as a cross-metric reading, not a match), resolved from its name
(`metric_of_run`, metrics.metric_of: the gap complex's model for a complex name, keyed by it;
the raster's for a conductance, on the run's grid) through the seam (seam.ClearingModel: the
exact J_power and its baseline, B (i)), with the run's power and demand (`demand_of`: the
population's or priority-weighted P), and set beside the row's own D, Lens A (`perm`) and
enclosed share. `run@step` names a nested run's row: a number, `last` or `mid` (a single-budget
run is step 1, written `run` alone). A complex run must match: its name carries every constant
that can move a score. A raster run named by a bare along-conductance was made under the step
rule before the face aperture (2026-10-13) and is scored under that along's `+apwfs` form, the
rule now, so a DIFFERS there measures the rule's change, not a fault; a row made before
lifted.Placement (metric preB) also scored without the homes the nearest-building rule had given
a neighbour.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import clearings_export as exporter  # noqa: E402
from clearings import RunInfo, Store  # noqa: E402

MATCH = 1e-7        # |Lens A rescored - stored| below this is a match (pyamg's random start vectors
                    # alone move a score by ~1e-11; the 60 large blocks' 38 unshadowed rescored to
                    # 3.2e-9, NOTES "Re-scoring the road methods: the 60 large blocks under B")


def check_blocks(store: Store) -> None:
    runs = store.runs()
    prints = store.prints()
    print(f"{len(runs)} runs, {len(prints)} blocks; rebuilding the blocks", flush=True)
    data = exporter.build_blocks(prints, batch=40)
    for block, bd in data.items():
        if bd.print != prints[block]:
            raise clearings.StaleClearing(
                f"{block}: committed for n {prints[block].n}, "
                f"{prints[block].fingerprint[:12]}; now n {bd.print.n}, {bd.print.fingerprint[:12]}")
    rows = 0
    for run, info in runs.items():
        t = store.table(run)
        recs: list[dict[str, Any]] = [{str(k): v for k, v in r.items()}
                                      for r in t.to_dict("records")]
        exporter.check_rows(run, info.mode, info.population, recs, data)
        rows += len(recs)
    print(f"every one of the {len(prints)} blocks is the block its clearings were committed for; "
          f"{rows} rows in {len(runs)} runs pass the row checks")


def check_banks(store: Store) -> None:
    import pickle
    from collections import defaultdict
    prints = store.prints()
    seen: dict[str, list[float]] = defaultdict(list)
    bad: list[str] = []
    unreadable: list[str] = []
    banks = sorted(exporter.CLUSTER_RUNS.glob("*/bank.pkl"))
    for bank in banks:
        try:
            with open(bank, "rb") as f:
                held = pickle.load(f)
        except Exception as e:                                  # an old pickle: report, never skip
            unreadable.append(f"{bank.parent.name}: {type(e).__name__}: {str(e)[:80]}")
            continue
        when = bank.stat().st_mtime
        for block_id, block in held.items():
            if block_id not in prints:
                continue
            got = clearings.BlockPrint.of(block)
            if got != prints[block_id]:
                bad.append(f"{bank.parent.name}: {block_id} n {got.n} vs {prints[block_id].n}, "
                           f"{got.fingerprint[:12]} vs {prints[block_id].fingerprint[:12]}")
            seen[block_id].append(when)
        del held
    import datetime as dt
    first = min(min(w) for w in seen.values())
    last = max(max(w) for w in seen.values())
    print(f"{len(banks) - len(unreadable)} of {len(banks)} banks read; {len(seen)} of the "
          f"{len(prints)} committed blocks seen {sum(len(w) for w in seen.values())} times, "
          f"banks written {dt.datetime.fromtimestamp(first):%Y-%m-%d %H:%M} .. "
          f"{dt.datetime.fromtimestamp(last):%Y-%m-%d %H:%M}")
    print(f"{len(bad)} disagreements" + "".join(f"\n  {b}" for b in bad[:20]))
    print(f"{len(unreadable)} unreadable" + "".join(f"\n  {u}" for u in unreadable[:20]))
    unseen = sorted(set(prints) - set(seen))
    print(f"{len(unseen)} committed blocks in no bank (small blocks run here): {unseen[:6]}")
    span = sorted(seen.items(), key=lambda kv: min(kv[1]))
    print("earliest bank per block, a sample: " + ", ".join(
        f"{b.split('_')[-1]} {dt.datetime.fromtimestamp(min(w)):%m-%d}" for b, w in span[:3]
        + span[len(span) // 2:len(span) // 2 + 2] + span[-2:]))


def step_of(info: RunInfo, store: Store, run: str, block: str, token: str) -> int:
    t = store.table(run)
    steps = sorted(int(s) for s in t.loc[t["block"] == block, "step"])
    if not steps:
        raise KeyError(f"{run} holds no clearing of {block}")
    if token == "last":
        return steps[-1]
    if token == "mid":
        return steps[len(steps) // 2]
    return int(token)


def demand_of(name: str) -> Any:
    """A run's demand by its name (RunInfo.demand): `w` the population's own
    (common.PopulationDemand), `pq<q>` priority-weighted P (common.PriorityDemand(q))."""
    import common
    if name == "w":
        return common.PopulationDemand()
    if name.startswith("pq"):
        return common.PriorityDemand(float(name[2:]))
    raise ValueError(f"unknown demand {name!r}: w or pq<q>")


def metric_of_run(info: RunInfo) -> Any:
    """The run's metric on the CPU (seam.Metric, metrics.metric_of): the gap complex's by its
    name (no grid: the run's mesh is ''), or the raster's conductance on the run's grid; a bare
    along-conductance (the step rule before the face aperture) as its `+apwfs` form."""
    import common
    import lifted
    import metrics
    if info.along.startswith(metrics.COMPLEX):
        return metrics.metric_of(info.along, device="cpu",
                                 mesh=lifted.UniformMesh(0.5, offset=lifted.OFFSET))
    name = (info.along if info.along.endswith(f"+{lifted.APERTURE}")
            else f"{info.along}+{lifted.APERTURE}")
    return metrics.metric_of(name, device="cpu", mesh=common.mesh_of(info.mesh.removeprefix("h")))


def rescore(store: Store, block_ids: Sequence[str], specs: Sequence[str], under: str) -> None:
    """`under`: `own` (each run under its own metric) or a metric's name (every run under it)."""
    import common
    import lifted
    import metrics

    runs = store.runs()
    for block in common.build_blocks(list(block_ids)):
        models: dict[tuple[str, str, str, str], Any] = {}
        for spec in specs:
            run, _, token = spec.partition("@")
            info = runs[run]
            t = store.table(run)
            if not (t["block"] == block.block_id).any():
                print(f"{block.block_id} {spec}: not in the run")
                continue
            if info.power is None or not info.along:
                raise ValueError(f"{run} does not record its metric settings: cannot rescore it")
            step = step_of(info, store, run, block.block_id, token or "1")
            ids = store.cleared(run, block, step=step)                # the fingerprint is checked
            row = t.loc[(t["block"] == block.block_id) & (t["step"] == step)].iloc[0]
            metric = (metric_of_run(info) if under == "own" else metrics.metric_of(
                under, device="cpu", mesh=lifted.UniformMesh(0.5, offset=lifted.OFFSET)))
            key = (metric.name, info.mesh, info.population, info.demand)
            if info.population != "area":
                raise ValueError(f"{run}: population {info.population}; the models' cost is the "
                                 "area population's")
            if key not in models:
                models[key] = metric.clearing(block, demand=demand_of(info.demand))
            c = models[key]
            removed = np.zeros(c.n, dtype=bool)
            removed[ids] = True
            J, _P = c.exact(removed, info.power)
            J0, _P0 = c.baseline(info.power)
            perm = 1.0 - (J / J0) ** (1.0 / info.power)
            enclosed = c.enclosed_left(removed)
            stored_enc = None if np.isnan(row.get("enclosed", np.nan)) else float(row["enclosed"])
            diff = perm - float(row["perm"])
            verdict = ("MATCH" if abs(diff) <= MATCH else "DIFFERS") if under == "own" else (
                "another metric")
            print(f"{block.block_id} n={c.n} {spec}  [{info.metric}] under {metric.name}  D "
                  f"{float(c.cost[removed].sum()):.6f} (stored {float(row['D']):.6f})  Lens A "
                  f"{perm:.9f} (stored {float(row['perm']):.9f}, diff {diff:+.2e}, {verdict})  "
                  f"enclosed {enclosed:.6f}"
                  + ("" if stored_enc is None else f" (stored {stored_enc:.6f})"), flush=True)
        del models


def main(argv: Sequence[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("blocks")
    sub.add_parser("banks")
    rs = sub.add_parser("rescore")
    rs.add_argument("ids")
    rs.add_argument("runs")
    rs.add_argument("under", help="own (each run under its own metric) or a metric's name")
    args = ap.parse_args(argv)
    store = Store.committed()
    if args.cmd == "blocks":
        check_blocks(store)
    elif args.cmd == "banks":
        check_banks(store)
    else:
        rescore(store, args.ids.split(","), args.runs.split(","), args.under)


if __name__ == "__main__":
    main(sys.argv[1:])
