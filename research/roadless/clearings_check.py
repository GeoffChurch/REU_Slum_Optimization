"""Checks of the committed clearings (clearings.py) that need the real blocks and the scorer; the
rest are unit tests (tests/test_clearings.py).

    PYTHONPATH=. uv run python research/roadless/clearings_check.py blocks
    PYTHONPATH=. uv run python research/roadless/clearings_check.py banks
    PYTHONPATH=. uv run python research/roadless/clearings_check.py rescore <ids,> <run[@step]>,... \
        <own|metric>
    PYTHONPATH=. uv run python research/roadless/clearings_check.py overwrite <ids,> [fault]

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
checked), scored again on the CPU under the run's own metric (`own`) or under the complex metric
named (every clearing under it, keyed by the name, its lens beside the run's own as a
cross-metric reading, not a match), resolved from its name (`metric_of_run`, metrics.metric_of:
the gap complex's model, gap_clear.Model: the exact J_power and its baseline, B (i)), with the
run's power and demand (`demand_of`: the population's or priority-weighted P), and set beside the
row's own D, Lens A (`perm`) and enclosed share. `run@step` names a nested run's row: a number,
`last` or `mid` (a single-budget run is step 1, written `run` alone). A complex run must match:
its name carries every constant that can move a score. A run the retired raster made is a record:
its clearings read under a complex metric named, never under its own (`own` raises).

`overwrite`: the optimizers' no-overwrite guard and the pool's keeping of what stands
(clearings.Standing; NOTES, "The consolidated pass"; the lease pool's spec, 3.3). On each block,
in a temporary rows root, SIMP (relax.main `one`, a short plan), the greedy and its polish
(polish_greedy.main, a coarse picker) and the arterial's proposal (arterial_study.propose, its
`smoke` arm: its proposal row) each run under the Standing its flags name (clearings.standing_of),
its work (relax.one, polish_greedy.run, the proposal's road_rescore.Sampler) counted and its output
read:
- as a person's run (`Refuse`), once: a row written, its row_marker.ROW_WRITTEN line printed;
  again: refused (clearings.Exists) before any work, the first row's bytes left;
- with `--replace` (`Replace`): a new row in its place, its ROW_WRITTEN line printed;
- with `--keep-standing` (`Keep`): the row standing, not run again and its bytes kept, no
  ROW_WRITTEN; the row removed, a row written; the row removed and a row written between Keep's
  check and its write (a double claim's other run), that row's bytes left, `[row] kept <path>`
  printed, nothing raised and no ROW_WRITTEN (a kept row is no progress: the plan's Ruling 6).
And standing_of: no flag `Refuse`, `--replace` `Replace`, `--keep-standing` `Keep`, both a
SystemExit. Faults (`overwrite <ids,> <fault>`): `no-check` (Refuse.todo a no-op and the exclusive
link an os.replace: the check removed), `keep-raises` (Keep.write raises the Exists),
`kept-counted` (Keep.write prints ROW_WRITTEN for a kept row), `replace-quiet` (Replace.write
prints no ROW_WRITTEN), `both-taken` (standing_of takes both flags, the later winning).
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clearings  # noqa: E402
import clearings_export as exporter  # noqa: E402
import row_marker  # noqa: E402
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
    """The run's metric on the CPU (metrics.metric_of): the gap complex's by its name. A run under
    another metric (the retired raster's, or another code's complex) is a record: it raises."""
    import metrics
    if not info.along.startswith(metrics.COMPLEX):
        raise ValueError(f"{info.run}: made under the retired raster's {info.along!r}, a record: "
                         "not rescorable under its own metric (name a complex metric instead)")
    return metrics.metric_of(info.along, device="cpu")


def rescore(store: Store, block_ids: Sequence[str], specs: Sequence[str], under: str) -> None:
    """`under`: `own` (each run under its own metric) or a metric's name (every run under it)."""
    import common
    import metrics

    runs = store.runs()
    for block in common.build_blocks(list(block_ids)):
        models: dict[tuple[str, str, str], Any] = {}
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
            metric = (metric_of_run(info) if under == "own"
                      else metrics.metric_of(under, device="cpu"))
            key = (metric.name, info.population, info.demand)
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


# `overwrite`'s runs: cheap settings, the guard is of the writer, not the optimizer
GUARD_PLAN = "fw0.q1.5.i2"                     # SIMP: one stage, two updates
GUARD_GREEDY = ("S0.05cat", 4, 2)              # the greedy's picker, the polish's tries and width


class Optimizer(NamedTuple):
    """One of `overwrite`'s optimizers on one block."""
    go: Callable[[clearings.Standing], None]     # one run of the identity under a Standing
    row: Path                                    # its row: the identity's path
    module: ModuleType                           # where `go` looks its work up by name
    work: str                                    # the work's name there (relax.one, ...)


class Ran(NamedTuple):
    """One run of an Optimizer: what it raised, how often its work ran, what it printed."""
    raised: BaseException | None
    calls: int
    lines: list[str]


def _ran(opt: Optimizer, flags: list[str], *, after: Callable[[], None]) -> Ran:
    """`opt` run once under the Standing `flags` name (clearings.standing_of), each call of its
    work followed by `after`, its output captured (and echoed)."""
    import contextlib
    import io
    from unittest import mock
    rest, standing = clearings.standing_of(flags)
    if rest:
        raise ValueError(f"not a Standing's flags: {rest}")
    real = getattr(opt.module, opt.work)

    def work(*args: Any, **kwargs: Any) -> Any:
        got = real(*args, **kwargs)
        after()
        return got
    out = io.StringIO()
    raised: BaseException | None = None
    with mock.patch.object(opt.module, opt.work, side_effect=work) as counted, \
            contextlib.redirect_stdout(out):
        try:
            opt.go(standing)
        except (Exception, SystemExit) as e:        # noqa: BLE001 -- the case reads it
            raised = e
    sys.stdout.write(out.getvalue())
    return Ran(raised=raised, calls=counted.call_count, lines=out.getvalue().splitlines())


def _standing(what: str, opt: Optimizer) -> list[str]:
    """`overwrite`'s cases (module docstring) for one optimizer on one block: the failures."""
    import os

    row = opt.row
    wrote, kept = f"{row_marker.ROW_WRITTEN}{row}", f"[row] kept {row}"

    def state() -> tuple[bytes, int]:
        return row.read_bytes(), os.stat(row).st_ino

    def nothing() -> None:
        return None
    bad = []
    r = _ran(opt, [], after=nothing)                                    # Refuse: the first run
    if r.raised is not None or not row.exists() or wrote not in r.lines:
        return [f"{what}: its first run raised {r.raised!r}, or wrote no row, or printed no "
                f"{wrote!r}"]
    first = state()
    r = _ran(opt, [], after=nothing)                                    # Refuse: again
    if not isinstance(r.raised, clearings.Exists) or r.calls:
        bad.append(f"{what}: a second run of its identity was not refused before any work (its "
                   f"work ran {r.calls} times; raised {r.raised!r})")
    else:
        print(f"{what}: the second run refused before any work ({r.raised})", flush=True)
    if state() != first:
        bad.append(f"{what}: the second run replaced the first row")
    r = _ran(opt, [row_marker.REPLACE], after=nothing)                   # Replace
    if r.raised is not None or state()[1] == first[1]:
        bad.append(f"{what}: {row_marker.REPLACE} wrote no new row (raised {r.raised!r})")
    elif wrote not in r.lines:
        bad.append(f"{what}: {row_marker.REPLACE}'s row printed no {wrote!r}")
    else:
        print(f"{what}: the replacement named, a new row in its place", flush=True)
    replaced = state()
    keep = [row_marker.KEEP_STANDING]
    r = _ran(opt, keep, after=nothing)                                  # Keep: the row stands
    if r.raised is not None or r.calls:
        bad.append(f"{what}: {row_marker.KEEP_STANDING} ran a standing identity again (its work "
                   f"ran {r.calls} times; raised {r.raised!r})")
    if state() != replaced or wrote in r.lines:
        bad.append(f"{what}: {row_marker.KEEP_STANDING} did not keep the standing row's bytes, or "
                   f"printed {wrote!r}")
    row.unlink()
    r = _ran(opt, keep, after=nothing)                                  # Keep: no row stands
    if r.raised is not None or r.calls != 1 or not row.exists() or wrote not in r.lines:
        bad.append(f"{what}: {row_marker.KEEP_STANDING} wrote no row for an identity without one "
                   f"(its work ran {r.calls} times; raised {r.raised!r})")
        return bad
    print(f"{what}: {row_marker.KEEP_STANDING} kept the standing row unrun and wrote the missing "
          "one", flush=True)
    other = row.read_bytes()
    row.unlink()
    raced: list[tuple[bytes, int]] = []

    def double_claim() -> None:
        """A double claim's other run writes the row between Keep's check and its write."""
        row.write_bytes(other)
        raced.append(state())
    r = _ran(opt, keep, after=double_claim)                             # Keep: the race
    if r.raised is not None:
        bad.append(f"{what}: {row_marker.KEEP_STANDING} raised {r.raised!r} on a row written "
                   "between its check and its write (a double claim's)")
    if raced != [state()]:
        bad.append(f"{what}: {row_marker.KEEP_STANDING} did not keep a row written between its "
                   "check and its write")
    if wrote in r.lines or kept not in r.lines:
        bad.append(f"{what}: {row_marker.KEEP_STANDING} printed {wrote!r} for a kept row, or no "
                   f"{kept!r}")
    if not bad:
        print(f"{what}: a row written between {row_marker.KEEP_STANDING}'s check and its write "
              "kept, nothing raised, no row counted", flush=True)
    return bad


def _standing_of() -> list[str]:
    """standing_of's resolution (module docstring): the failures."""
    want: dict[tuple[str, ...], tuple[list[str], clearings.Standing]] = {
        ("one", "x"): (["one", "x"], clearings.Refuse()),
        ("one", row_marker.REPLACE, "x"): (["one", "x"], clearings.Replace()),
        (row_marker.KEEP_STANDING, "one", "x"): (["one", "x"], clearings.Keep())}
    bad = [f"standing_of({list(argv)}) is {got}, not {w}" for argv, w in want.items()
           if (got := clearings.standing_of(list(argv))) != w]
    both = [row_marker.REPLACE, row_marker.KEEP_STANDING]
    try:
        bad.append(f"standing_of({both}) took both: {clearings.standing_of(both)}")
    except SystemExit as e:
        print(f"standing_of({both}) refused: {e}", flush=True)
    return bad


def overwrite(block_ids: Sequence[str]) -> list[str]:
    """`overwrite` (module docstring): the failures, one line each."""
    import os
    import tempfile
    from unittest import mock

    import arterial_study
    import metrics
    import polish_greedy
    import relax
    import road_rescore as rr
    import search

    metric = metrics.metric_of(metrics.RUNTIME, device="cpu")
    plan = relax.plan_of(GUARD_PLAN)
    picker, tries, width = GUARD_GREEDY
    arm = "smoke"                       # the arterial's runner check: 5 roads, seconds
    name = arterial_study.network_name(arm, arterial_study.method_of(arterial_study.ARMS[arm]))
    bad = _standing_of()
    # the arterial's proposal checks the single-threaded environment of a cluster task, which
    # names PYTHONDONTWRITEBYTECODE too (road_rescore.SINGLE)
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.object(relax, "OUT", Path(tmp) / "relax_rows"), \
            mock.patch.object(polish_greedy, "HERE", Path(tmp)), \
            mock.patch.dict(os.environ, rr.SINGLE):
        arterial_rows = Path(tmp) / "arterial"
        for bid in block_ids:
            def simp(standing: clearings.Standing, b: str = bid) -> None:
                relax.main("one", [b], 1.0, metric, plan, 1, relax.D_LENS, standing=standing)

            def greedy(standing: clearings.Standing, b: str = bid) -> None:
                polish_greedy.main(b, 1.0, metric, picker, tries, width, relax.D_LENS, False,
                                   scoring=search.InEps(), standing=standing)

            def propose(standing: clearings.Standing, b: str = bid) -> None:
                arterial_study.propose(arterial_rows, b, arm, standing=standing)
            runs = {"simp": Optimizer(go=simp, row=relax.rows_of(plan, metric.name, relax.D_LENS)
                                      / f"{bid}_p1.parquet", module=relax, work="one"),
                    "greedy+polish": Optimizer(go=greedy, row=polish_greedy.rows_of(
                        picker, tries, width, False, search.InEps(), metric.name, relax.D_LENS)
                        / f"{bid}_p1.parquet", module=polish_greedy, work="run"),
                    # its work: the proposal, timed under the Sampler it starts
                    "arterial propose": Optimizer(go=propose, row=rr.proposal_row(
                        arterial_rows, bid, name, False), module=rr, work="Sampler")}
            for what, opt in runs.items():
                bad += _standing(f"{bid} {what}", opt)
    return bad


def _no_check() -> AbstractContextManager[Any]:
    """The fault: the no-overwrite check removed (Refuse.todo a no-op; the exclusive link an
    os.replace, as an unguarded writer's to_parquet would)."""
    import contextlib
    import os
    from unittest import mock
    stack = contextlib.ExitStack()
    stack.enter_context(mock.patch.object(clearings.Refuse, "todo", lambda self, path: True))
    stack.enter_context(mock.patch.object(clearings, "_exclusive", os.replace))
    return stack


def _keep_raises() -> AbstractContextManager[Any]:
    """The fault: Keep.write raises the Exists of a row written meanwhile (Refuse's write)."""
    from unittest import mock
    return mock.patch.object(clearings.Keep, "write",
                             lambda self, df, path: clearings.write_once(df, path))


def _kept_counted() -> AbstractContextManager[Any]:
    """The fault: Keep.write prints row_marker.ROW_WRITTEN for the row it kept."""
    from unittest import mock

    def write(self: clearings.Keep, df: Any, path: Path) -> None:
        try:
            clearings.write_once(df, path)
        except clearings.Exists:
            print(f"{row_marker.ROW_WRITTEN}{path}", flush=True)
    return mock.patch.object(clearings.Keep, "write", write)


def _replace_quiet() -> AbstractContextManager[Any]:
    """The fault: Replace.write puts its row in place and prints no row_marker.ROW_WRITTEN."""
    import os
    from unittest import mock
    return mock.patch.object(clearings.Replace, "write",
                             lambda self, df, path: clearings.place(df, path, os.replace))


def _both_taken() -> AbstractContextManager[Any]:
    """The fault: standing_of takes --replace and --keep-standing together, the later winning."""
    from unittest import mock
    real = clearings.standing_of

    def standing_of(argv: list[str]) -> tuple[list[str], clearings.Standing]:
        if row_marker.REPLACE in argv and row_marker.KEEP_STANDING in argv:
            first = min(row_marker.REPLACE, row_marker.KEEP_STANDING, key=argv.index)
            return real([a for a in argv if a != first])
        return real(argv)
    return mock.patch.object(clearings, "standing_of", standing_of)


OVERWRITE_FAULTS: dict[str, Callable[[], AbstractContextManager[Any]]] = {
    "no-check": _no_check, "keep-raises": _keep_raises, "kept-counted": _kept_counted,
    "replace-quiet": _replace_quiet, "both-taken": _both_taken}


def main(argv: Sequence[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("blocks")
    sub.add_parser("banks")
    ow = sub.add_parser("overwrite")
    ow.add_argument("ids")
    ow.add_argument("fault", nargs="?", choices=sorted(OVERWRITE_FAULTS))
    rs = sub.add_parser("rescore")
    rs.add_argument("ids")
    rs.add_argument("runs")
    rs.add_argument("under", help="own (each run under its own metric) or a metric's name")
    args = ap.parse_args(argv)
    if args.cmd == "overwrite":
        import common
        ids = common.ids_of(args.ids)
        if args.fault is None:
            bad = overwrite(ids)
        else:
            with OVERWRITE_FAULTS[args.fault]():
                bad = overwrite(ids)
        print(f"overwrite{'' if args.fault is None else f' under the fault {args.fault}'}: "
              + ("FAILS: " + "; ".join(bad) if bad else "passes"), flush=True)
        if bad and args.fault is None:
            raise SystemExit(1)
        if not bad and args.fault is not None:
            raise SystemExit(f"the fault {args.fault} went unseen: the guard guards nothing")
        return
    store = Store.committed()
    if args.cmd == "blocks":
        check_blocks(store)
    elif args.cmd == "banks":
        check_banks(store)
    else:
        import common
        rescore(store, common.ids_of(args.ids), args.runs.split(","), args.under)


if __name__ == "__main__":
    main(sys.argv[1:])
